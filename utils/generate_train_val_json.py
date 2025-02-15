import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

json_trajectories = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19]
root_dir = "/bigdata/SurgPose_dev/annotations"

presaved_json = "/bigdata/SurgPose_dev/annotations/annotation_train_new_000007.json"
with open(presaved_json, 'r') as f:
    json_dict_presaved = json.load(f)

json_annotations = json_dict_presaved['annotations']
json_images = json_dict_presaved['images']

for json_traj in json_trajectories:
    json_traj_file = os.path.join(root_dir, f"annotation_{str(json_traj).zfill(6)}_left.json")

    with open(json_traj_file, 'r') as f:
        json_dict = json.load(f)

    json_images = json_images + json_dict['images']
    json_annotations = json_annotations + json_dict['annotations']

print(len(json_images))
print(len(json_annotations))

breakpoint()

json_dict['images'] = json_images
json_dict['annotations'] = json_annotations
json_dict['categories'] = [{"id": 1, "name": "surgical_tool"}]

new_json = "/bigdata/SurgPose_dev/annotations/annotation_0_16_19.json"

# save the det_list as a json file
with open(new_json, 'w') as f:
    json.dump(json_dict, f)