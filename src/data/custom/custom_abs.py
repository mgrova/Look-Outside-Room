# naive dataloader for real estate dataset
# this dataloader is used to build a absolute dataset

import numpy as np
import os
from tqdm import tqdm
from PIL import Image

import PIL
import cv2
import random
import glob
import pickle
import quaternion

import torchvision.transforms as tfs
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from src.data.realestate.realestate_cview import ToTensorVideo, NormalizeVideo

def resize(clip, target_size, interpolation_mode = "bilinear"):
    assert len(target_size) == 2, "target size should be tuple (height, width)"
    return torch.nn.functional.interpolate(
        clip, size=target_size, mode=interpolation_mode, align_corners=False
    )

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

# format -> id x y z qw qx qy qz
def read_poses_from_file(filename):
    if not os.path.isfile(filename):
        raise Exception("File {} does not exist".format(filename))
    
    with open(filename, 'r') as file:
        lines = file.readlines()
        poses = []
        for line in lines:
            if line.startswith('#'):
                continue
            else:
                pose = line.split(' ')
                pose = [float(el) for el in pose]
                poses.append(pose)
        return poses

# set global parameter
# init for 256 * 256
K = np.array([[128.0, 0.0, 127.0],
              [0.0, 128.0, 127.0],
              [0.0, 0.0, 1.0]], dtype=np.float32)

K_inv = np.linalg.inv(K)

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, root_path = "/custom_data/train", image_size = [256,256], 
                 length = 3, low = 3, high = 20, is_validation = False):
        super(VideoDataset, self).__init__()
        
        self.image_size = image_size
        self.length = length
        self.low = low
        self.high = high
        
        self.clip_length = self.length + (self.length - 1) * (self.high - 1)

        clip_paths = []

        # To be able to read scenes inside test and train folders
        scene_paths = glob.glob(os.path.join(root_path, "*/*"))

        print("----------------Loading the CUSTOM dataset----------------")
        for scene_path in tqdm(scene_paths):

            if (not os.path.isdir(scene_path)):
                continue

            seq = scene_path.split("/")[-1]

            im_root = os.path.join(scene_path, 'rgb')

            frames = [fname for fname in os.listdir(im_root) if fname.endswith(".png")]
            n = len(frames)
            
            # filter the data which is too short
            if n > self.clip_length:
                clip_paths.append(scene_path)
                
        print("----------------Finish loading CUSTOM dataset----------------")

        self.clip_paths = clip_paths
        
        self.size = len(self.clip_paths) # num of scene
        
        self.is_validation = is_validation
        self.transform = transforms.Compose(
        [
            ToTensorVideo(),
            NormalizeVideo((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])

    def __getitem__(self, index):
        # root = os.path.join(self.sequence_dir, seq)
        im_root = os.path.join(self.clip_paths[index], 'rgb')
        
        frames = sorted([os.path.join(im_root, fname) for fname in os.listdir(im_root) if fname.endswith(".png")])
        n = len(frames)
        
        # set global parameter
        # init for 256 * 256
        K = np.array([[128.0, 0.0, 127.0],
                      [0.0, 128.0, 127.0],
                      [0.0, 0.0, 1.0]], dtype=np.float32)

        K_inv = np.linalg.inv(K)

        # then sample a inter-index
        inter_index = random.randint(0, n - self.clip_length)
        
        rgbs = []
        R_s = []
        t_s = []
        
        img_idx = inter_index
        previous_key = None
        
        for idx in range(self.length):
            if idx != 0:
                gap = random.randint(self.low, self.high)
            else:
                gap = 0
            
            img_idx += gap

            poses_file = os.path.join(self.clip_paths[index], 'poses.txt')
            poses = read_poses_from_file(poses_file)
            curr_pose = poses[img_idx]

            qvec = np.array(tuple(map(float, curr_pose[4:8])))
            tvec = np.array(tuple(map(float, curr_pose[1:4])))

            R_dst = qvec2rotmat(qvec)
            t_dst = tvec
            
            R_s.append(R_dst)
            t_s.append(t_dst)

            # get the name
            image_path = frames[img_idx]
            im = Image.open(image_path).resize((self.image_size[1],self.image_size[0]), resample=Image.LANCZOS)
            im = np.array(im)
            rgbs.append(im)

        # post-process using size

        # TODO. Read these parameters from camera
        w = 256
        h = 256
        # post-process using size
        if self.image_size is not None and (self.image_size[0] != h or self.image_size[1] != w):
            K[0,:] = K[0,:]*self.image_size[1]/w
            K[1,:] = K[1,:]*self.image_size[0]/h

        rgbs = torch.from_numpy(np.stack(rgbs))
        rgbs = self.transform(rgbs)
        
        example = {
            "rgbs": rgbs,
            "src_points": np.zeros((1,3), dtype=np.float32),
            "K": K.astype(np.float32),
            "K_inv": K_inv.astype(np.float32),
            "R_s": np.stack(R_s).astype(np.float32),
            "t_s": np.stack(t_s).astype(np.float32)
        }

        return example

    def __len__(self):
        return self.size

    def name(self):
        return 'VideoDataset'
