import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt

val_json = "/bigdata/SurgPose_dev/annotations/annotation_val_new.json"

with open(val_json, 'r') as f:
    val_dict = json.load(f)

annotations = val_dict['annotations']
images = val_dict['images']

img_root_dir = "/bigdata/SurgPose_dev/val"

det_list = []

for idx, ann in enumerate(annotations):
    tempt_dict = {}
    tempt_dict['bbox'] = ann['bbox']
    tempt_dict["category_id"] = 1
    tempt_dict['image_id'] = ann['image_id']
    tempt_dict['score'] = 0.9999
    det_list.append(tempt_dict)
    # print(ann)
    # breakpoint()
    # bbox = ann['bbox']
    # image_file = str(ann['image_id']).zfill(12) + '.png'
    # img = cv2.imread(os.path.join(img_root_dir, image_file))
    # x, y, w, h = bbox
    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    # plt.imshow(img)
    # plt.show()
    # breakpoint()

# save the det_list as a json file 
with open('SurgPose_val_detections_new.json', 'w') as f:
    json.dump(det_list, f)

breakpoint()
with open('SurgPose_val_detections.json', 'r') as f:
    det_list_check = json.load(f)

breakpoint()