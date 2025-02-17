# YOLO4SurgPose
A simple implementation using YOLO for surgical tool pose estimation.

## Environment

Python 3.11.9 torch 2.4.0 torchvision 0.19.0

Run `conda env create -f environment.yml`

## Data Format

### SurgPose to YOLO
[Reference](https://docs.ultralytics.com/datasets/pose/#ultralytics-yolo-format) of the YOLO Ultralytics Pose format.

1. Bounding Box: (x, y, width, height), in which x, y are the coordinates of the center of the object, normalized to be between 0 and 1.
 
We provide a script `./utils/surgpose2yolo.py` to convert the label format from SurgPose to YOLO.

### SurgPose to COCO

## Train 

## Evaluation

