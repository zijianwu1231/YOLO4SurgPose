import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

val_json = "/bigdata/SurgPose_ViTPose/annotations/annotation_val.json"

with open(val_json, 'r') as f:
    val_dict = json.load(f)

train_json = "/bigdata/SurgPose_ViTPose/annotations/annotation_train.json"

with open(train_json, 'r') as f:
    train_dict = json.load(f)

val_images = val_dict['images']
train_images = train_dict['images']

val_annotations = val_dict['annotations']
train_annotations = train_dict['annotations']

new_train_image = train_images + val_images
new_train_annotations = train_annotations + val_annotations

breakpoint()

train_dict['images'] = new_train_image
train_dict['annotations'] = new_train_annotations

breakpoint()
with open('annotation_train_new.json', 'w') as f:
    json.dump(train_dict, f)