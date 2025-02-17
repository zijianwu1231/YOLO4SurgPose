import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import yaml

## image directory
"""For example, here we use video 100002 as training set, 
   and video 100003 as validation set.
"""

# argparser = argparse.ArgumentParser()
# argparser.add_argument("--traj_id", type=str)
# argparser.add_argument("--camera_id", type=str)
# argparser.add_argument("--image_dir", type=str)

# Image directory
traj_id = "100002"
camera_id = "left"
data_root_dir = "/bigdata/SurgPose"
image_dir = os.path.join(data_root_dir, traj_id, 'regular', camera_id+'_frames')
image_list = sorted(os.listdir(image_dir), key=lambda x: int(x.split('.')[0][5:]))  
# print(image_list)

# Keypointss directory
keypoints_path = "/home/zijianwu/Codes/segment-anything-2/keypoints/keypoints_100002_left.yaml"
bbox_path = "/home/zijianwu/Codes/YOLO4SurgPose/bbox/100002_left.yml"

kpt_dict = yaml.load(open(keypoints_path, 'r'), Loader=yaml.FullLoader)
bbox_dict = yaml.load(open(bbox_path, 'r'), Loader=yaml.FullLoader)

re_index = True
initial_index = 0

class_index = 0

original_image_width = 1400
original_image_height = 986

if re_index:
    for idx, img in enumerate(image_list):
        curr_index = int(img.split('.')[0][5:])
        new_index = curr_index + initial_index
        new_name = str(new_index).zfill(12)+'.png'

        # read bbox labels for the image
        bbox_label = bbox_dict[img]
        kpt_label = kpt_dict[curr_index]
        assert len(kpt_label) == 10, "bbox_label should have 10 elements"

        # generate label file
        for tool_id in range(2):
            
            bbox = bbox_label[tool_id]

            label_temp = []

            # add class index
            label_temp.append(class_index) 

            # add bbox
            label_temp += bbox[tool_id] 

            # add keypoints
            if tool_id == 0:
                for keypoint in range(1, 6):
                    kpt = kpt_label[keypoint]
                    kpt_norm = [kpt[0]/original_image_width, kpt[1]/original_image_height]
                    if kpt:
                        label_temp += kpt_norm
                    
            elif tool_id == 1:
                for keypoint in range(8, 13):
                    kpt = kpt_label[keypoint]
                    kpt_norm = [kpt[0]/original_image_width, kpt[1]/original_image_height]
                    if kpt:
                        label_temp += kpt_norm

            with open(f'./datasets/labels/train/{str(new_index).zfill(12)}.txt', 'a') as the_file:
                the_file.write(" ".join(map(str, label_temp)))
                the_file.write("\n")

        breakpoint()