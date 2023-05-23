import random
import os
import argparse
import json
import math

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys, datetime, glob, importlib

sys.path.insert(0, ".")

import torch
import torchvision
from torchvision import transforms
from src.data.realestate.realestate_cview import ToTensorVideo, NormalizeVideo

from omegaconf import OmegaConf

# args
parser = argparse.ArgumentParser(description="training codes")
parser.add_argument("--base", type=str, default="mp3d_16x16_sine_cview_adaptive",
                    help="experiments name")
parser.add_argument("--exp", type=str, default="try_1",
                    help="experiments name")
parser.add_argument("--input_image_path", type=str, default="../configs/custom/sample_data/0.png")
parser.add_argument("--input_poses_path", type=str, default="../configs/custom/transforms.json")
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument("--seed", type=int, default=2333, help="")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# fix the seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# config
config_path = "../configs/mp3d/%s.yaml" % args.base
cpt_path = "./pretrained_models/matterport/last.ckpt"


# load custom data
def load_poses(pose_file_path):
    with open(pose_file_path, "r") as f:
        poses = json.load(f)
        if 'frames' in poses:
            poses = poses['frames']
            poses = [torch.from_numpy(np.array(p['transform_matrix'])).cuda().float() for p in poses]
        else:
            poses = [torch.from_numpy(np.array(p)).cuda().float() for i, p in poses.items()]
        return poses


def get_pinhole_intrinsics_from_fov(H, W, fov_in_degrees=55.0):
    px, py = (W - 1) / 2., (H - 1) / 2.
    fx = fy = W / (2. * np.tan(fov_in_degrees / 360. * np.pi))
    k_ref = np.array([[fx, 0.0, px, 0.0], [0.0, fy, py, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                     dtype=np.float32)
    k_ref = torch.tensor(k_ref)  # K is [4,4]

    return k_ref


def load_intrinsics(transforms_file_path, H, W, K):
    with open(transforms_file_path, "r") as f:
        transforms_dict = json.load(f)

        if 'camera_angle_x' in transforms_dict:
            # fov is independent of image resolution, so use #get_pinhole_intrinsics_from_fov() to convert: fov --> focal length
            fov = transforms_dict['camera_angle_x'] * 180.0 / math.pi
            K = get_pinhole_intrinsics_from_fov(H, W, fov)
        elif 'x_fov' in transforms_dict:
            # fov is independent of image resolution, so use #get_pinhole_intrinsics_from_fov() to convert: fov --> focal length
            fov = transforms_dict['x_fov']
            K = get_pinhole_intrinsics_from_fov(H, W, fov)
        else:
            K[0, 0] = transforms_dict['fl_x']
            K[1, 1] = transforms_dict['fl_y']
            K[0, 2] = transforms_dict['cx']
            K[1, 2] = transforms_dict['cy']

            # convert intrinsics to actual used H, W in the images
            if H != transforms_dict['h'] or W != transforms_dict['w']:
                scaling_factor_w = W / transforms_dict['w']
                scaling_factor_h = H / transforms_dict['h']
                K[0, 0] = K[0, 0] * scaling_factor_w
                K[0, 2] = K[0, 2] * scaling_factor_w
                K[1, 1] = K[1, 1] * scaling_factor_h
                K[1, 2] = K[1, 2] * scaling_factor_h

        return K


poses = load_poses(args.input_poses_path)
start_image = torch.from_numpy(np.array(Image.open(args.input_image_path).resize((256, 256)))).cuda()[None]
K = load_intrinsics(args.input_poses_path, 256, 256, torch.eye(3)).cuda()[None, :3, :3]

transform = transforms.Compose([
    ToTensorVideo(),
    NormalizeVideo((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
])

start_image = transform(start_image).permute(1, 0, 2, 3)

# create out dir
target_save_path = "./experiments/custom/%s/evaluate_frame_%s_len_%d/" % (args.exp, os.path.basename(args.input_image_path), len(poses))
os.makedirs(target_save_path, exist_ok=True)


# load model
def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


# define relative pose
def compute_camera_pose(R_dst, t_dst, R_src, t_src):
    # first compute R_src_inv
    R_src_inv = R_src.transpose(-1, -2)

    R_rel = R_dst @ R_src_inv
    t_rel = t_dst - R_rel @ t_src

    return R_rel, t_rel


config = OmegaConf.load(config_path)
model = instantiate_from_config(config.model)
model.cuda()
model.load_state_dict(torch.load(cpt_path))
model.eval()


def as_png(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = (x + 1.0) * 127.5
    x = x.clip(0, 255).astype(np.uint8)
    return Image.fromarray(x)

def evaluate_per_batch(temp_model, start_image, poses, show=False):
    video_clips = []
    video_clips.append(start_image)

    # first generate one frame
    with torch.no_grad():
        conditions = []
        R_src = poses[0][:3, :3]
        R_src_inv = R_src.transpose(-1, -2)
        t_src = poses[0][:3, 3]

        # create dict
        example = dict()
        example["K"] = K
        example["K_inv"] = K.inverse()

        example["src_img"] = video_clips[-1]
        _, c_indices = temp_model.encode_to_c(example["src_img"])
        c_emb = temp_model.transformer.tok_emb(c_indices)
        conditions.append(c_emb)

        R_dst = poses[1][:3, :3]
        t_dst = poses[1][:3, 3]

        R_rel = R_dst @ R_src_inv
        t_rel = t_dst - R_rel @ t_src

        example["R_rel"] = R_rel.unsqueeze(0)
        example["t_rel"] = t_rel.unsqueeze(0)

        embeddings_warp = temp_model.encode_to_e(example)
        conditions.append(embeddings_warp)
        # p1
        p1 = temp_model.encode_to_p(example)

        prototype = torch.cat(conditions, 1)
        z_start_indices = c_indices[:, :0]
        index_sample = temp_model.sample_latent(z_start_indices, prototype, [p1, None, None],
                                                steps=c_indices.shape[1],
                                                temperature=1.0,
                                                sample=False,
                                                top_k=100,
                                                callback=lambda k: None,
                                                embeddings=None)

        sample_dec = temp_model.decode_to_img(index_sample, [1, 256, 16, 16])
        video_clips.append(sample_dec)  # update video_clips list
        current_im = as_png(sample_dec.permute(0, 2, 3, 1)[0])

        if show:
            plt.imshow(current_im)
            plt.show()

    # then generate second
    with torch.no_grad():
        for i in range(0, total_time_len - 2, time_len):
            conditions = []

            R_src = batch["R_s"][0, i, ...]
            R_src_inv = R_src.transpose(-1, -2)
            t_src = batch["t_s"][0, i, ...]

            # create dict
            example = dict()
            example["K"] = batch["K"]
            example["K_inv"] = batch["K_inv"]

            for t in range(time_len):
                example["src_img"] = video_clips[-2]
                _, c_indices = temp_model.encode_to_c(example["src_img"])
                c_emb = temp_model.transformer.tok_emb(c_indices)
                conditions.append(c_emb)

                R_dst = batch["R_s"][0, i + t + 1, ...]
                t_dst = batch["t_s"][0, i + t + 1, ...]

                R_rel = R_dst @ R_src_inv
                t_rel = t_dst - R_rel @ t_src

                example["R_rel"] = R_rel.unsqueeze(0)
                example["t_rel"] = t_rel.unsqueeze(0)
                embeddings_warp = temp_model.encode_to_e(example)
                conditions.append(embeddings_warp)
                # p1
                p1 = temp_model.encode_to_p(example)

                example["src_img"] = video_clips[-1]
                _, c_indices = temp_model.encode_to_c(example["src_img"])
                c_emb = temp_model.transformer.tok_emb(c_indices)
                conditions.append(c_emb)

                R_dst = batch["R_s"][0, i + t + 2, ...]
                t_dst = batch["t_s"][0, i + t + 2, ...]

                R_rel = R_dst @ R_src_inv
                t_rel = t_dst - R_rel @ t_src

                example["R_rel"] = R_rel.unsqueeze(0)
                example["t_rel"] = t_rel.unsqueeze(0)
                embeddings_warp = temp_model.encode_to_e(example)
                conditions.append(embeddings_warp)
                # p2
                p2 = temp_model.encode_to_p(example)
                # p3
                R_rel, t_rel = compute_camera_pose(batch["R_s"][0, i + t + 2, ...], batch["t_s"][0, i + t + 2, ...],
                                                   batch["R_s"][0, i + t + 1, ...], batch["t_s"][0, i + t + 1, ...])
                example["R_rel"] = R_rel.unsqueeze(0)
                example["t_rel"] = t_rel.unsqueeze(0)
                p3 = temp_model.encode_to_p(example)

                prototype = torch.cat(conditions, 1)

                z_start_indices = c_indices[:, :0]
                index_sample = temp_model.sample_latent(z_start_indices, prototype, [p1, p2, p3],
                                                        steps=c_indices.shape[1],
                                                        temperature=1.0,
                                                        sample=False,
                                                        top_k=100,
                                                        callback=lambda k: None,
                                                        embeddings=None)

                sample_dec = temp_model.decode_to_img(index_sample, [1, 256, 16, 16])
                current_im = as_png(sample_dec.permute(0, 2, 3, 1)[0])
                video_clips.append(sample_dec)  # update video_clips list

                if show:
                    plt.imshow(current_im)
                    plt.show()

    return video_clips


# generate
generate_video = evaluate_per_batch(model, start_image, poses, show=True)

# save to file
for i in range(1, len(generate_video)):
    forecast_img = np.array(as_png(generate_video[i][0].permute(1, 2, 0)))
    cv2.imwrite(os.path.join(target_save_path, "predict_%02d.png" % i), forecast_img[:, :, [2, 1, 0]])
