#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Pose, PoseArray
from sensor_msgs.msg   import Image
from cv_bridge import CvBridge

import os
import cv2 as cv
import numpy as np

def read_target_poses_from_file(file_path, len):
    if not os.path.isfile(file_path):
        raise Exception("File {} does not exist".format(file_path))
    
    poses = []
    with open(file_path, 'r') as file:
        index = 0
        lines = file.readlines()
        for line in lines:
            if line.startswith('#'):
                print("Ensure that is the format is propperly deparsed. Current format: {}".format(line))
                continue
            else:
                pose = line.split(' ')
                pose = [float(el) for el in pose]
                poses.append(pose)
            index += 1

            if (index == len):
                break
    
    return poses

def create_pose_msg(tvec, qvec):
    curr_pose = Pose()

    curr_pose.position.x = tvec[0]
    curr_pose.position.y = tvec[1]
    curr_pose.position.z = tvec[2]

    curr_pose.orientation.w = qvec[0]
    curr_pose.orientation.x = qvec[1]
    curr_pose.orientation.y = qvec[2]
    curr_pose.orientation.z = qvec[3]

    return curr_pose

def convert_poses_to_posearray_msg(poses):
    posearray_msg = PoseArray()
    for pose in poses:
        if len(pose) == 8:
            # format (id, x, y, z, qw, qx, qy, qz)
            tvec = np.array(tuple(map(float, pose[1:4])))
            qvec = np.array(tuple(map(float, pose[4:8])))
        elif len(pose) == 7:
            # format (x, y, z, qw, qx, qy, qz)
            tvec = np.array(tuple(map(float, pose[0:3])))
            qvec = np.array(tuple(map(float, pose[3:7])))
        else:
            raise Exception("Invalid poses lenght: {}".format(len(poses)))

        posearray_msg.poses.append(create_pose_msg(tvec, qvec))

    return posearray_msg

def read_image_from_file(file_path):
    cv_image = cv.imread(file_path)
    cv_image_rgb = cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)
    bridge = CvBridge()
    image_msg = bridge.cv2_to_imgmsg(cv_image_rgb)
    return image_msg

def image_poses_to_infer_publisher():
    rospy.init_node('image_poses_to_infer_publisher', anonymous=False)
    
    desired_posearray_topic = rospy.get_param('~desired_posearray_topic', default="desired_pose_array")
    start_image_topic       = rospy.get_param('~start_image_topic', default="start_image")

    desired_posearray_pub = rospy.Publisher(desired_posearray_topic, PoseArray, queue_size=1)
    start_image_pub       = rospy.Publisher(start_image_topic, Image, queue_size=1)
    
    poses = read_target_poses_from_file("/home/aiiacvmllab/Documents/datasets/LookOut_UE4/test/dataset_2023-06-19_12:21:19/poses.txt", 10)
    posearray_msg = convert_poses_to_posearray_msg(poses)
    posearray_msg.header.stamp    = rospy.Time.now()
    posearray_msg.header.frame_id = "camera_frame_id"

    image_msg = read_image_from_file("/home/aiiacvmllab/Documents/datasets/LookOut_UE4/test/dataset_2023-06-19_12:21:19/rgb/rgb_0000.png")
    image_msg.header.stamp    = rospy.Time.now()
    image_msg.header.frame_id = "camera_frame_id"
    
    # Only publish data once
    desired_posearray_pub.publish(posearray_msg)
    start_image_pub.publish(image_msg)

if __name__ == '__main__':
    try:
        image_poses_to_infer_publisher()
    except rospy.ROSInterruptException:
        pass