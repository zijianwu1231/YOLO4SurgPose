from ultralytics import YOLO
import os
import json
from metrics import *

# Load a model
model = YOLO("/home/zijianwu/Codes/yolopose/runs/pose/train8/weights/best.pt")  # load a custom model

# ground truth file
annFile = '/bigdata/SurgPose_dev/annotations/annotation_val_new.json'
with open(annFile) as f:
    annotations = json.load(f)

ann_instances = annotations['annotations']

# Predict with the model
image_dir = "/home/zijianwu/Codes/yolopose/datasets/images/val"
# images = sorted(os.listdir(image_dir))

for ann in ann_instances:
    image = str(ann['image_id']).zfill(12) + '.png'
    image_path = os.path.join(image_dir, image)
    results = model(image_path)  # predict on an image
    probs = results[0].probs  # Probs object for classification outputs
    pred_keypoints = results[0].keypoints.xy.cpu().detach().numpy()  # Keypoints object for pose outputs

    # add visibility for predicted keypoints
    pred_keypoints = pred_keypoints.tolist()
    pred_kpts = []
    for obj in pred_keypoints:
        obj_kpt = []
        for kpt in obj:
            if kpt[0] == 0 and kpt[1] == 0:
                obj_kpt.append(kpt[0])
                obj_kpt.append(kpt[1])
                obj_kpt.append(0)
            else:
                obj_kpt.append(kpt[0])
                obj_kpt.append(kpt[1])
                obj_kpt.append(2)
        pred_kpts.append(obj_kpt)

    gt_keypoints = ann['keypoints']

    breakpoint()

    # calculate OKS
    sigma_tool = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
    max_oks = 0
    for i in range(2):
        oks = OKS(pred_kpts[i], gt_keypoints, sigma=sigma_tool, area=10000) #ann['area'])
        print(oks)
        if oks > max_oks:
            max_oks = oks
    print(f"max OKS: {max_oks}")

    breakpoint()

    # calculate distance
    min_distance = 100000
    for i in range(2):
        distance = Distance(pred_kpts[i], gt_keypoints)
        print(distance)
        if distance < min_distance:
            min_distance = distance
    print(f"min distance: {min_distance}")

    # result.show()  # display to screen
    # result.save(filename="result.jpg")  # save to disk

    breakpoint()