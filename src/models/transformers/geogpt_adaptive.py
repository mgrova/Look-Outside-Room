# full auto-regressive GeoGPT
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange

from src.main import instantiate_from_config
from src.modules.losses.pose_losses import CameraPoseLoss
from src.data.custom.custom_abs import rotmat2qvec

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class GeoTransformer(nn.Module):
    def __init__(self,
                 transformer_config,
                 first_stage_config,
                 cond_stage_config,
                 merge_channels=None,
                 use_depth=True,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_key="image",
                 cond_stage_key="depth",
                 use_scheduler=False,
                 scheduler_config=None,
                 emb_stage_config=None,
                 emb_stage_key="camera",
                 emb_stage_trainable=True,
                 top_k=None
                 ):

        super().__init__()
            
        self.init_first_stage_from_ckpt(first_stage_config)
        self.init_cond_stage_from_ckpt(cond_stage_config)
        self.transformer = instantiate_from_config(config=transformer_config)

        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key

        self.use_scheduler = use_scheduler
        
        if use_scheduler:
            assert scheduler_config is not None
            self.scheduler_config = scheduler_config
            
        self.emb_stage_key = emb_stage_key
        self.emb_stage_trainable = emb_stage_trainable and emb_stage_config is not None
        self.init_emb_stage_from_ckpt(emb_stage_config)
        self.top_k = top_k if top_k is not None else 100

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.pose_loss = CameraPoseLoss().to("cuda")
        
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing keys and {len(unexpected)} unexpected keys.")

    def init_first_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train

    def init_cond_stage_from_ckpt(self, config):
        if config == "__is_first_stage__":
            print("Using first stage also as cond stage.")
            self.cond_stage_model = self.first_stage_model
        else:
            model = instantiate_from_config(config)
            self.cond_stage_model = model.eval()
            self.cond_stage_model.train = disabled_train

    def init_emb_stage_from_ckpt(self, config):
        if config is None:
            self.emb_stage_model = None
        else:
            model = instantiate_from_config(config)
            self.emb_stage_model = model
            if not self.emb_stage_trainable:
                self.emb_stage_model.eval()
                self.emb_stage_model.train = disabled_train

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, _, info = self.first_stage_model.encode(x)
        indices = info[2].view(quant_z.shape[0], -1)
        return quant_z, indices

    @torch.no_grad()
    def encode_to_c(self, c):
        quant_c, _, info = self.cond_stage_model.encode(c)
        indices = info[2].view(quant_c.shape[0], -1)
        return quant_c, indices

    def encode_to_e(self, batch):
        return self.emb_stage_model.process(batch)

    def get_normalized_c(self, batch):
        with torch.no_grad():
            quant_c, c_indices = self.encode_to_c(batch["src_img"])
            quant_d = None
       
        embeddings = self.encode_to_e(batch)
        dc_indices = c_indices

        # check that unmasking is correct
        total_cond_length = embeddings.shape[1] + dc_indices.shape[1]
        assert total_cond_length == self.transformer.config.n_unmasked, (
            embeddings.shape[1], dc_indices.shape[1], self.transformer.config.n_unmasked)

        return quant_d, quant_c, dc_indices, embeddings
    
    def encode_to_p(self, batch):
        inputs = []
        
        for k in ["R_rel", "t_rel", "K", "K_inv"]:
            entry = batch[k].reshape(batch[k].shape[0], -1)
            inputs.append(entry)
            
        p = torch.cat(inputs, dim=1) # B, 30
        
        return p
    
    def decode_from_p(self, p):
        # Initialize a dictionary to store decoded entries
        batch = {"R_rel": None,
                "t_rel": None,
                "K": None,
                "K_inv": None}

        p = p.reshape(-1, p.size(-1))

        # Calculate the number of elements per entry
        num_elements_rot   = 9
        num_elements_trans = 3

        # Extract and reshape the entries
        start_idx = 0

        # Decode R_rel entry
        r_rel_entry = p[start_idx : start_idx + num_elements_rot, :]
        reshaped_r_rel_entry = r_rel_entry.reshape(3, 3, -1)
        batch["R_rel"] = reshaped_r_rel_entry
        start_idx += num_elements_rot

        # Decode t_rel entry
        t_rel_entry = p[start_idx : start_idx + num_elements_trans, :]
        reshaped_t_rel_entry = t_rel_entry.reshape(3, -1)
        batch["t_rel"] = reshaped_t_rel_entry
        start_idx += num_elements_trans

        # Decode K entry
        k_entry = p[start_idx : start_idx + num_elements_rot, :]
        reshaped_k_entry = k_entry.reshape(3, 3, -1)
        batch["K"] = reshaped_k_entry
        start_idx += num_elements_rot

        # Decode K_inv entry
        k_inv_entry = p[start_idx : start_idx + num_elements_rot, :]
        reshaped_k_inv_entry = k_inv_entry.reshape(3, 3, -1)
        batch["K_inv"] = reshaped_k_inv_entry

        return batch
        
    def forward(self, batch):
        # get time
        B, time_len = batch["rgbs"].shape[0], batch["rgbs"].shape[2]
        
        # create dict
        example = dict()
        example["K"] = batch["K"]
        example["K_inv"] = batch["K_inv"]
        
        conditions = [] # list of [camera, frame] 
        gts_imgs   = [] # gt imgs  | except the first img
        gts_poses  = [] # gt poses | except the first pose
        p = []

        for t in range(0, time_len-1): 
            _, c_indices = self.encode_to_c(batch["rgbs"][:, :, t, ...])
            c_emb = self.transformer.tok_emb(c_indices)
            conditions.append(c_emb)
            
            if t == 0:
                example["R_rel"] = batch["R_01"]
                example["t_rel"] = batch["t_01"]
                embeddings_warp = self.encode_to_e(example)
                p.append(self.encode_to_p(example))
                conditions.append(embeddings_warp)
                
            if t == 1:
                example["R_rel"] = batch["R_02"]
                example["t_rel"] = batch["t_02"]
                embeddings_warp = self.encode_to_e(example)
                p.append(self.encode_to_p(example))
                conditions.append(embeddings_warp)

            if t > 0:
                gts_imgs.append(c_indices) # for loss
        
        # TODO. Make sense use this GT? We are prediction image 1 and 2, so we want to 
        # learn relative transformation between each t image
        gts_poses.append([batch["R_01"].cpu().numpy(), batch["t_01"].cpu().numpy()])
        gts_poses.append([batch["R_12"].cpu().numpy(), batch["t_12"].cpu().numpy()])

        _, c_indices = self.encode_to_c(batch["rgbs"][:, :, time_len-1, ...]) # final frame
        c_emb = self.transformer.tok_emb(c_indices)
        conditions.append(c_emb)
        gts_imgs.append(c_indices)
        
        conditions = torch.cat(conditions, 1) # B, L, 1024
        prototype = conditions[:, 0:286, :]
        z_emb = conditions[:, 286::, :]
        
        # p3 
        example["R_rel"] = batch["R_12"]
        example["t_rel"] = batch["t_12"]
        p.append(self.encode_to_p(example))
        
        images_pred, poses_pred, _ = self.transformer.iter_forward(prototype, z_emb, p = p)
        
        forecasts_imgs = []
        for t in range(0, time_len-2):
            forecasts_imgs.append(images_pred[:, 286*t:286*t+256, :])
        forecasts_imgs.append(images_pred[:, -256::, :]) # final frame

        image_loss = self.compute_image_loss(forecasts_imgs, gts_imgs)

        # TODO. Check if this make sense. It is the pose T_01 and T_02. I think so.
        # Make it as for loop depending of the time_len (as forecast_images)
        forecasts_poses = []
        forecasts_poses.append(poses_pred[:, 256 : 286, :])
        forecasts_poses.append(poses_pred[:, 286+256 : 286+256+30, :])
        
        poses_loss = self.compute_pose_loss(forecasts_poses, gts_poses)

        return forecasts_imgs, gts_imgs, image_loss, forecasts_poses, gts_poses, poses_loss

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    @torch.no_grad()
    def sample_latent_visual(self, x, c, p, steps, temperature=1.0, sample=False, top_k=None,
               callback=lambda k: None, embeddings=None, **kwargs):
        # in the current variant we always use embeddings for camera
        # assert embeddings is not None
        # check n_unmasked and conditioning length
        # total_cond_length = embeddings.shape[1] + c.shape[1]
        # assert total_cond_length == self.transformer.config.n_unmasked, (
        #     embeddings.shape[1], c.shape[1], self.transformer.config.n_unmasked)

        assert not self.transformer.training
        
        bias = None
        
        for k in range(steps):
            callback(k)
            x_cond = x
            
            if bias is None:
                logits, bias = self.transformer.test(c, x_cond, p, embeddings=embeddings, return_bias=True)
            else:
                logits, _ = self.transformer.test(c, x_cond, p, embeddings=embeddings)
                
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            probs = F.softmax(logits, dim=-1)
            
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
                
            x = torch.cat((x, ix), dim=1)   

        return x, bias
    
    @torch.no_grad()
    def sample_latent(self, x, c, p, steps, temperature=1.0, sample=False, top_k=None,
               callback=lambda k: None, embeddings=None, **kwargs):
        # in the current variant we always use embeddings for camera
        # assert embeddings is not None
        # check n_unmasked and conditioning length
        # total_cond_length = embeddings.shape[1] + c.shape[1]
        # assert total_cond_length == self.transformer.config.n_unmasked, (
        #     embeddings.shape[1], c.shape[1], self.transformer.config.n_unmasked)

        assert not self.transformer.training
        
        for k in range(steps):
            callback(k)
            x_cond = x            
            logits, _ = self.transformer.test(c, x_cond, p, embeddings=embeddings)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            probs = F.softmax(logits, dim=-1)
            
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
                
            x = torch.cat((x, ix), dim=1)   

        return x
    
    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, sample=False, top_k=None,
               callback=lambda k: None, embeddings=None, **kwargs):
        # in the current variant we always use embeddings for camera
        assert embeddings is not None
        # check n_unmasked and conditioning length
        total_cond_length = embeddings.shape[1] + c.shape[1]
        assert total_cond_length == self.transformer.config.n_unmasked, (
            embeddings.shape[1], c.shape[1], self.transformer.config.n_unmasked)

        x = torch.cat((c,x),dim=1)
        block_size = self.transformer.get_block_size()
        assert not self.transformer.training
        for k in range(steps):
            callback(k)
            assert x.size(1) <= block_size  # make sure model can see conditioning
            # do not crop as this messes with n_unmasked
            #x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
            x_cond = x
            logits, _ = self.transformer(x_cond, embeddings=embeddings)
            # pluck the logits at the final step and scale by temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # append to the sequence and continue
            x = torch.cat((x, ix), dim=1)
        # cut off conditioning
        x = x[:, c.shape[1]:]
        return x


    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])
        quant_z = self.first_stage_model.quantize.get_codebook_entry(
            index.reshape(-1), shape=bhwc)
        x = self.first_stage_model.decode(quant_z)
        return x
    
    def compute_image_loss(self, logits, targets):
        logits_array  = torch.cat(logits, 0)
        logits_final  = logits_array.reshape(-1, logits_array.size(-1))
        
        targets_array = torch.cat(targets, 0)
        targets_finals = targets_array.reshape(-1)

        loss = F.cross_entropy(logits_final, targets_finals)
        return loss
    
    def compute_pose_loss(self, predictions, gts):
        # Convert predictions to Nx7 tensor
        est_poses = []
        for pred in predictions:
            pred_batch = self.decode_from_p(pred.detach().cpu())
            q   = rotmat2qvec(pred_batch["R_rel"].numpy())
            pos = pred_batch["t_rel"][0:3].numpy().flat
            est_pose = [pos[0], pos[1], pos[2], 
                        q[0], q[1], q[2], q[3]]
            est_poses.append(est_pose)

        est_poses = torch.tensor(est_poses).to("cuda")

        # Convert gts to Nx7 tensor
        gt_poses = []
        for rot, trans in gts:
            q = rotmat2qvec(rot)
            pos = trans.flat # Due to multiples batches
            gt_pose = [pos[0], pos[1], pos[2], 
                       q[0], q[1], q[2], q[3]]
            gt_poses.append(gt_pose)
        
        gt_poses = torch.tensor(gt_poses).to("cuda")

        # x,y,z,qw,qx,qy,qz
        loss = self.pose_loss(est_poses, gt_poses)
        return loss


    def configure_optimizers(self):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.Parameter)
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('frame_emb')
        no_decay.add('camera_emb')
        no_decay.add('time_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
#         assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
#                                                     % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(param_dict.keys() - union_params))], "weight_decay": 0.0},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        extra_parameters = list()
        if self.emb_stage_trainable:
            extra_parameters += list(self.emb_stage_model.parameters())
        
        optim_groups.append({"params": extra_parameters, "weight_decay": 0.0})
        print(f"Optimizing {len(extra_parameters)} extra parameters.")
        
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        
        if self.use_scheduler:
            print("Setting up LambdaLR scheduler...")
            scheduler = instantiate_from_config(self.scheduler_config)
            scheduler = LambdaLR(optimizer, lr_lambda=scheduler.schedule)

            return optimizer, scheduler
        
        return optimizer