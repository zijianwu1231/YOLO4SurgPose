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

# bbox directory
"""The bbox is generated from the segmentation masks
"""
mask_dir = "/home/zijianwu/Codes/segment-anything-2/results/100002_bbox/mask_gt"
mask_list = sorted(os.listdir(mask_dir), key=lambda x: int(x.split('.')[0][6:]))

bbox_dir = "./bbox"
os.makedirs(bbox_dir, exist_ok=True)

bbox_dict = {}
for idx, mask in enumerate(mask_list):
    mask_path = os.path.join(mask_dir, mask)
    mask = np.load(mask_path)

    mask1 = mask.copy()
    mask2 = mask.copy()

    mask1[mask1 != 1] = 0
    mask2[mask2 != 2] = 0

    # generate bbox
    contours, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        x1, y1, w1, h1 = cv2.boundingRect(contours[0])
        cv2.rectangle(mask1, (x1, y1), (x1 + w1, y1 + h1), (255, 255, 255), 2)
    # plt.imshow(mask1)
    # plt.show()
    # breakpoint()

    contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        x2, y2, w2, h2 = cv2.boundingRect(contours[0])
        cv2.rectangle(mask2, (x2, y2), (x2 + w2, y2 + h2), (255, 255, 255), 2)
    # plt.imshow(mask2)
    # plt.show()

    # print(x1, y1, w1, h1)
    # print(x2, y2, w2, h2)

    # the center of the bounding box, and normalize it to [0, 1]
    x1 = (x1 + w1/2) / mask.shape[1]
    y1 = (y1 + h1/2) / mask.shape[0]
    w1 = w1 / mask.shape[1]
    h1 = h1 / mask.shape[0]

    x2 = (x2 + w2/2) / mask.shape[1]
    y2 = (y2 + h2/2) / mask.shape[0]
    w2 = w2 / mask.shape[1]
    h2 = h2 / mask.shape[0]

    bbox_dict[image_list[idx]] = [{0 : [x1, y1, w1, h1]}, {1 : [x2, y2, w2, h2]}]

with open('./bbox/100002_left.yml', 'w') as outfile:
    yaml.dump(bbox_dict, outfile, default_flow_style=False)
