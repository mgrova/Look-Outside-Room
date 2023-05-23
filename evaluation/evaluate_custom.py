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
parser.add_argument("--input_image_path", type=str, default="../configs/custom/sample_data/6.png")
parser.add_argument("--input_poses_path", type=str, default="../configs/custom/transforms_simple_forward.json")
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
config_path = "../configs/mp3d/%s.yaml" % args.base
cpt_path = "./pretrained_models/matterport/last.ckpt"


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> torch.Tensor:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")

def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)


def convert_pose_from_nerf_convention(cam_to_world: torch.Tensor) -> torch.Tensor:
    """
    Converts a cam_to_world matrix in the original nerf convention (e.g. as in nerf_synthetic dataset) to a world_to_cam matrix in pytorch3d convention.

    We have camera orientation as in OpenGL/Blender which is:
    +X is right, +Y is up, and +Z is pointing back and away from the camera. -Z is the look-at direction.
    World Space also has a specific convetion which is:
    Our world space is oriented such that the up vector is +Z. The XY plane is parallel to the ground plane.
    See here for details: https://docs.nerf.studio/en/latest/quickstart/data_conventions.html

    We need pytorch3d convention which is:
    +X is left, +Y is up, and +Z is pointing front. +Z is the look-at direction.
    See here for details: https://pytorch3d.org/docs/cameras

    :param cam_to_world: [4,4] torch.Tensor or [3,4] torch.Tensor
    :return: world_to_cam: [4,4] torch.Tensor
    """
    cam_to_world = cam_to_world.clone()

    if cam_to_world.shape[0] == 3:
        cam_to_world = torch.cat([
            cam_to_world,
            torch.tensor([[0, 0, 0, 1]]).to(cam_to_world)
        ], dim=0)

    # zxy --> xyz
    zxy_to_xyz = torch.tensor([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ]).to(cam_to_world)
    cam_to_world = zxy_to_xyz @ cam_to_world

    # have cam_to_world, need world_to_cam
    world_to_cam = cam_to_world.inverse()

    # change translation, XZ translation must be flipped
    world_to_cam[0, 3] = -world_to_cam[0, 3]
    world_to_cam[2, 3] = -world_to_cam[2, 3]

    # change rotation, XZ rotation must be flipped
    angles = matrix_to_euler_angles(world_to_cam[:3, :3], "XYZ")
    angles[0] = -angles[0]
    angles[2] = -angles[2]
    world_to_cam[:3, :3] = euler_angles_to_matrix(angles, "XYZ")

    return world_to_cam


# load custom data
def load_poses(pose_file_path, convert_from_nerf=True):
    with open(pose_file_path, "r") as f:
        poses = json.load(f)
        if 'frames' in poses:
            poses = poses['frames']
            poses = [torch.from_numpy(np.array(p['transform_matrix'])).cuda().float() for p in poses]
        else:
            poses = [torch.from_numpy(np.array(p)).cuda().float() for i, p in poses.items()]

        if convert_from_nerf:
            poses = [convert_pose_from_nerf_convention(p) for p in poses]

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


poses = load_poses(args.input_poses_path, convert_from_nerf=True)
start_image = torch.from_numpy(np.array(Image.open(args.input_image_path).resize((256, 256)))).cuda()[None]
K = load_intrinsics(args.input_poses_path, 256, 256, torch.eye(3)).cuda()[None, :3, :3]
K_inv = K.inverse()

transform = transforms.Compose([
    ToTensorVideo(),
    NormalizeVideo((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
])

start_image = transform(start_image).permute(1, 0, 2, 3)

# create out dir
target_save_path = "./experiments/custom/%s/evaluate_transforms_%s_frame_%s_poses_%d_len_%d/" % (args.exp, os.path.basename(args.input_poses_path), os.path.basename(args.input_image_path), len(poses), args.video_limit)
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

def evaluate_per_batch(temp_model, start_image, poses, show=False, total_time_len=20):
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
        example["K_inv"] = K_inv

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
            example["K_inv"] = K_inv

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


# generate
generate_video = evaluate_per_batch(model, start_image, poses, show=False, total_time_len=args.video_limit)

# save to file
with open(args.input_poses_path, "r") as f:
    transforms = json.load(f)
    for i in range(len(generate_video)):
        img_pil = as_png(generate_video[i][0].permute(1, 2, 0))
        forecast_img = np.array(img_pil)
        cv2.imwrite(os.path.join(target_save_path, "predict_%02d.png" % i), forecast_img[:, :, [2, 1, 0]])

        nerf_img = img_pil.resize((transforms['w'], transforms['h']))
        out_file_name = os.path.join(target_save_path, os.path.basename(transforms['frames'][i]['file_path']))
        if not out_file_name.endswith(".png"):
            out_file_name += ".png"
        nerf_img.save(out_file_name)
