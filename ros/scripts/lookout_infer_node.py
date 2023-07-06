#!/usr/bin/env python3

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray
from message_filters import Subscriber, ApproximateTimeSynchronizer

import cv2 as cv
import sys, importlib
import os
import time
import torch
import numpy as np
import random

uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
sys.path.append(uppath(__file__, 3))

from omegaconf import OmegaConf
from torchvision import transforms

from src.data.custom.custom_cview import ToTensorVideo, NormalizeVideo, qvec2rotmat

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key 'target' to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

class LookOutTransformer():
    def __init__(self, node_name):
        rospy.init_node(node_name, anonymous=False)

        config_path     = rospy.get_param('~config_path', default="custom_16x16_sine_cview_adaptive")
        exp_name        = rospy.get_param('~exp_name', default="exp_emb16384")
        gpu             = rospy.get_param('~gpu', default="0")
        seed            = rospy.get_param('~seed', default=2333)
        checkpoint      = rospy.get_param('~checkpoint', default="/home/aiiacvmllab/Projects/nvs_repos/Look-Outside-Room/experiments/custom/exp_emb16384/model/last.ckpt")
        self.__pred_len = rospy.get_param('~prediction_len', default=4)
        
        start_image_topic       = rospy.get_param('~start_image_topic', default="start_image")
        desired_posearray_topic = rospy.get_param('~desired_posearray_topic', default="desired_pose_array")
        inference_image_topic   = rospy.get_param('~inference_image_topic', default="inference_img")

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        # TODO: Read from camera_info
        K = np.array([[128.0, 0.0, 127.0],
                      [0.0, 128.0, 127.0],
                      [0.0, 0.0, 1.0]], dtype=np.float32)
        self.__K = torch.from_numpy(K).cuda()[None, :3, :3]

        # fix the seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Setup image transformations
        self.__transform = transforms.Compose([
            ToTensorVideo(),
            NormalizeVideo((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])

        # Load and setup model
        if checkpoint == "":
            checkpoint = "./experiments/custom/{}/model/last.ckpt".format(exp_name)
        self.model = None
        self.__setup_model(config_path, checkpoint)
        rospy.loginfo("Succesfully loaded model")

        # Setup publishers and subscribers
        self.__bridge = CvBridge()
        self.__inference_pub = rospy.Publisher(inference_image_topic, Image, queue_size=1, latch=True)

        self.__image_sub             = Subscriber(start_image_topic, Image)
        self.__desired_posearray_sub = Subscriber(desired_posearray_topic, PoseArray)

        # Synchronize the image and posearray messages based on their timestamps
        self.__ts = ApproximateTimeSynchronizer([self.__image_sub, self.__desired_posearray_sub], queue_size=1, slop=0.1)
        self.__ts.registerCallback(self.__inference_cb)

        rospy.loginfo("Succesfully configured subs/pubs")
        
    def __inference_cb(self, image_msg, posearray_msg):
        
        rospy.loginfo("Received new info. Starting inference ...")

        start_image = self.__convert_image_msg_to_tensor(image_msg)
        poses       = self.__convert_posearray_msg_to_tensor(posearray_msg)

        self.infer_from_poses_and_image(poses, start_image)

    def __convert_image_msg_to_tensor(self, image_msg):
        # Setup initial image
        cv_image = self.__bridge.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
        resized_image = cv.resize(cv_image, (256, 256))

        if (not image_msg.encoding == "rgb8"):
            raise Exception("Invalid image encoding: {}".format(image_msg.encoding))
        
        tensor_image = torch.from_numpy(resized_image).cuda()[None]
        tensor_image = self.__transform(tensor_image).permute(1, 0, 2, 3)
        return tensor_image
    
    def __convert_posearray_msg_to_tensor(self, posearray_msg):
        # Convert pose to 4x4 matrix
        poses_mat_tensor = []
        for pose in posearray_msg.poses:
            
            tvec = np.array([pose.position.x, pose.position.y, pose.position.z])
            qvec = np.array([pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z])

            T_mat = np.zeros([3, 4])
            T_mat[:3, :3] = qvec2rotmat(qvec)
            T_mat[:3, 3]  = tvec

            poses_mat_tensor.append(torch.from_numpy(T_mat).cuda().float())

        return poses_mat_tensor

    def __setup_model(self, config_path, checkpoint):
        config = OmegaConf.load(config_path)
        self.model = instantiate_from_config(config.model)
        self.model.cuda()
        self.model.load_state_dict(torch.load(checkpoint))
        self.model.eval()

    def __compute_camera_pose(self, R_dst, t_dst, R_src, t_src):
        # first compute R_src_inv
        R_src_inv = R_src.transpose(-1, -2)

        R_rel = R_dst @ R_src_inv
        t_rel = t_dst - R_rel @ t_src

        return R_rel, t_rel
    
    def __infer_to_ros_image(self, x):
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        x = (x + 1.0) * 127.5
        x = x.clip(0, 255).astype(np.uint8)

        ros_image = self.__bridge.cv2_to_imgmsg(x, encoding="passthrough")
        ros_image.header.stamp    = rospy.Time.now()
        ros_image.header.frame_id = "camera_frame_id"
        ros_image.encoding        = "rgb8"
        return ros_image

    def infer_from_poses_and_image(self, poses, start_image):
        for pose in poses:
            if (not pose.is_cuda):
                print("Poses must be CUDA tensor")
                return

        if (not start_image.is_cuda):
            print("Poses must be cuda tensor")
            return

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
            example["K"] = self.__K
            example["K_inv"] = self.__K.inverse()

            example["src_img"] = video_clips[-1]
            _, c_indices = self.model.encode_to_c(example["src_img"])
            c_emb = self.model.transformer.tok_emb(c_indices)
            conditions.append(c_emb)

            R_dst = poses[1][:3, :3]
            t_dst = poses[1][:3, 3]

            R_rel = R_dst @ R_src_inv
            t_rel = t_dst - R_rel @ t_src

            example["R_rel"] = R_rel.unsqueeze(0)
            example["t_rel"] = t_rel.unsqueeze(0)

            embeddings_warp = self.model.encode_to_e(example)
            conditions.append(embeddings_warp)
            # p1
            p1 = self.model.encode_to_p(example)

            prototype = torch.cat(conditions, 1)
            z_start_indices = c_indices[:, :0]

            start = time.time()
            index_sample = self.model.sample_latent(z_start_indices, prototype, [p1, None, None],
                                                    steps=c_indices.shape[1],
                                                    temperature=1.0,
                                                    sample=False,
                                                    top_k=100,
                                                    callback=lambda k: None,
                                                    embeddings=None)
            sample_dec = self.model.decode_to_img(index_sample, [1, 256, 16, 16])
            video_clips.append(sample_dec)  # update video_clips list
            end = time.time()
            rospy.loginfo("Inference time: {} sec".format(end - start))

            current_im = self.__infer_to_ros_image(sample_dec.permute(0, 2, 3, 1)[0])
            self.__inference_pub.publish(current_im)
        
        # then generate second
        frame_limit = min(self.__pred_len, len(poses)) # Select the minimal value between the lenght of the poses or the defined lenght
        with torch.no_grad():
            for i in (range(0, frame_limit - 2, 1)):
                conditions = []

                R_src = poses[i][:3, :3]
                R_src_inv = R_src.transpose(-1, -2)
                t_src = poses[i][:3, 3]

                # create dict
                example = dict()
                example["K"] = self.__K
                example["K_inv"] = self.__K.inverse()

                for t in range(1):
                    example["src_img"] = video_clips[-2]
                    _, c_indices = self.model.encode_to_c(example["src_img"])
                    c_emb = self.model.transformer.tok_emb(c_indices)
                    conditions.append(c_emb)

                    R_dst = poses[i + t + 1][:3, :3]
                    t_dst = poses[i + t + 1][:3, 3]

                    R_rel = R_dst @ R_src_inv
                    t_rel = t_dst - R_rel @ t_src

                    example["R_rel"] = R_rel.unsqueeze(0)
                    example["t_rel"] = t_rel.unsqueeze(0)
                    embeddings_warp = self.model.encode_to_e(example)
                    conditions.append(embeddings_warp)
                    # p1
                    p1 = self.model.encode_to_p(example)

                    example["src_img"] = video_clips[-1]
                    _, c_indices = self.model.encode_to_c(example["src_img"])
                    c_emb = self.model.transformer.tok_emb(c_indices)
                    conditions.append(c_emb)

                    R_dst = poses[i + t + 2][:3, :3]
                    t_dst = poses[i + t + 2][:3, 3]

                    R_rel = R_dst @ R_src_inv
                    t_rel = t_dst - R_rel @ t_src

                    example["R_rel"] = R_rel.unsqueeze(0)
                    example["t_rel"] = t_rel.unsqueeze(0)
                    embeddings_warp = self.model.encode_to_e(example)
                    conditions.append(embeddings_warp)
                    # p2
                    p2 = self.model.encode_to_p(example)
                    # p3
                    R_rel, t_rel = self.__compute_camera_pose(poses[i + t + 2][:3, :3], poses[i + t + 2][:3, 3],
                                                              poses[i + t + 1][:3, :3], poses[i + t + 1][:3, 3])
                    example["R_rel"] = R_rel.unsqueeze(0)
                    example["t_rel"] = t_rel.unsqueeze(0)
                    p3 = self.model.encode_to_p(example)

                    prototype = torch.cat(conditions, 1)

                    z_start_indices = c_indices[:, :0]
                    start = time.time()
                    index_sample = self.model.sample_latent(z_start_indices, prototype, [p1, p2, p3],
                                                            steps=c_indices.shape[1],
                                                            temperature=1.0,
                                                            sample=False,
                                                            top_k=100,
                                                            callback=lambda k: None,
                                                            embeddings=None)

                    sample_dec = self.model.decode_to_img(index_sample, [1, 256, 16, 16])
                    end = time.time()
                    rospy.loginfo("Inference time: {} sec".format(end - start))

                    current_im = self.__infer_to_ros_image(sample_dec.permute(0, 2, 3, 1)[0])
                    self.__inference_pub.publish(current_im)
                    video_clips.append(sample_dec)  # update video_clips list
        

if __name__ == '__main__':
    try:
        _ = LookOutTransformer('lookout_transformer_inference_node')
        rospy.spin()
    except rospy.ROSInterruptException:
        pass