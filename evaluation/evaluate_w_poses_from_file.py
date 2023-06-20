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
from src.data.custom.custom_cview import ToTensorVideo, NormalizeVideo, qvec2rotmat

from omegaconf import OmegaConf

# format -> x y z qw qx qy qz
def read_target_poses_from_file(filename):
    if not os.path.isfile(filename):
        raise Exception("File {} does not exist".format(filename))
    
    poses = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('#'):
                continue
            else:
                pose = line.split(' ')
                pose = [float(el) for el in pose]
                poses.append(pose)

    # Convert format (x, y, z, qw, qx, qy, qz) to 4x4 matrix
    matrices = []
    for pose in poses:
        tvec = np.array(tuple(map(float, pose[0:3])))
        qvec = np.array(tuple(map(float, pose[3:7])))

        T_mat = np.zeros([3, 4])
        T_mat[:3, :3] = qvec2rotmat(qvec)
        T_mat[:3, 3]  = tvec

        matrices.append(torch.from_numpy(T_mat).cuda().float())

    return matrices
    
def setup_intrinsics():
    K = np.array([[128.0, 0.0, 127.0],
                  [0.0, 128.0, 127.0],
                  [0.0, 0.0, 1.0]], dtype=np.float32)

    return K

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

def as_png(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = (x + 1.0) * 127.5
    x = x.clip(0, 255).astype(np.uint8)
    return Image.fromarray(x)

def evaluate_per_batch(temp_model, start_image, K, poses, show=False, total_time_len=20):
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
    N = min(total_time_len, len(poses))
    with torch.no_grad():
        for i in tqdm(range(0, N - 2, 1)):
            conditions = []

            R_src = poses[i][:3, :3]
            R_src_inv = R_src.transpose(-1, -2)
            t_src = poses[i][:3, 3]

            # create dict
            example = dict()
            example["K"] = K
            example["K_inv"] = K.inverse()

            for t in range(1):
                example["src_img"] = video_clips[-2]
                _, c_indices = temp_model.encode_to_c(example["src_img"])
                c_emb = temp_model.transformer.tok_emb(c_indices)
                conditions.append(c_emb)

                R_dst = poses[i + t + 1][:3, :3]
                t_dst = poses[i + t + 1][:3, 3]

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

                R_dst = poses[i + t + 2][:3, :3]
                t_dst = poses[i + t + 2][:3, 3]

                R_rel = R_dst @ R_src_inv
                t_rel = t_dst - R_rel @ t_src

                example["R_rel"] = R_rel.unsqueeze(0)
                example["t_rel"] = t_rel.unsqueeze(0)
                embeddings_warp = temp_model.encode_to_e(example)
                conditions.append(embeddings_warp)
                # p2
                p2 = temp_model.encode_to_p(example)
                # p3
                R_rel, t_rel = compute_camera_pose(poses[i + t + 2][:3, :3], poses[i + t + 2][:3, 3],
                                                   poses[i + t + 1][:3, :3], poses[i + t + 1][:3, 3])
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

def main():
    # args
    parser = argparse.ArgumentParser(description="training codes")
    parser.add_argument("--base", type=str, default="custom_16x16_sine_cview_adaptive",
                        help="experiments name")
    parser.add_argument("--exp", type=str, default="try_1",
                        help="experiments name")
    parser.add_argument("--input_image_path", type=str, default="./configs/custom/sample_data/69302567.png")
    parser.add_argument("--input_poses_path", type=str, default="./configs/custom/transforms_simple_forward.json")
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument("--video_limit", type=int, default=20, help="# of video to test")
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
    config_path = "./configs/custom/%s.yaml" % args.base
    cpt_path = "./pretrained_models/custom/last.ckpt"

    poses = read_target_poses_from_file(args.input_poses_path)

    img_rgb = Image.open(args.input_image_path).resize((256, 256)).convert('RGB')
    start_image = torch.from_numpy(np.array(img_rgb)).cuda()[None]
    K = torch.from_numpy(setup_intrinsics()).cuda()

    transform = transforms.Compose([
        ToTensorVideo(),
        NormalizeVideo((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
    ])

    start_image = transform(start_image).permute(1, 0, 2, 3)

    # create out dir
    target_save_path = "./experiments/custom/%s/evaluate_transforms_%s_frame_%s_poses_%d_len_%d/" % (args.exp, os.path.basename(args.input_poses_path), os.path.basename(args.input_image_path), len(poses), args.video_limit)
    os.makedirs(target_save_path, exist_ok=True)

    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    model.cuda()
    model.load_state_dict(torch.load(cpt_path))
    model.eval()

    # generate
    generate_video = evaluate_per_batch(model, start_image, K, poses, show=False, total_time_len=args.video_limit)

    # save to file

    for i in range(len(generate_video)):
        img_pil = as_png(generate_video[i][0].permute(1, 2, 0))
        forecast_img = np.array(img_pil)
        cv2.imwrite(os.path.join(target_save_path, "predict_%02d.png" % i), forecast_img[:, :, [2, 1, 0]])

        estim_img = img_pil.resize((256, 256))
        out_file_name = os.path.join(target_save_path, "output_rendering/render_{}".format(str(i).zfill(3)))
        if not out_file_name.endswith(".png"):
            out_file_name += ".png"
        estim_img.save(out_file_name)

if __name__ == "__main__":
    main()