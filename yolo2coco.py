from ultralytics import YOLO
import json

import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

annType = ['segm','bbox','keypoints']
annType = annType[2]      #specify type here
prefix = 'annotation' if annType=='keypoints' else 'instances'
print('Running demo for *%s* results.'%(annType))

#initialize COCO ground truth api
dataDir='/bigdata/SurgPose_dev'
dataType='val_new'
annFile = '%s/annotations/%s_%s.json'%(dataDir,prefix,dataType)

with open(annFile) as f:
    gt = json.load(f)

for i in range(len(gt['annotations'])):
    gt['annotations'][i]['keypoints'] = gt['annotations'][i]['keypoints'] + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

with open('/home/zijianwu/Codes/yolopose/runs/pose/val3/annotations_changed.json', 'w') as f:
    json.dump(gt, f)

annFile = '/home/zijianwu/Codes/yolopose/runs/pose/val3/annotations_changed.json'

cocoGt=COCO(annFile)


with open('/home/zijianwu/Codes/yolopose/runs/pose/val3/predictions.json') as f:
    predicted_pose = json.load(f) 

for i in range(len(predicted_pose)):
    predicted_pose[i].pop('bbox', None)
    predicted_pose[i]['category_id'] = 1
    # change visibility to 1
    predicted_pose[i]['keypoints'][2::3] = [1] * 17
    breakpoint()
    predicted_pose[i]['keypoints'] = predicted_pose[i]['keypoints'] + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

with open('/home/zijianwu/Codes/yolopose/runs/pose/val3/predictions_changed.json', 'w') as f:
    json.dump(predicted_pose, f)

breakpoint()

#initialize COCO detections api
resFile='/home/zijianwu/Codes/yolopose/runs/pose/val3/predictions_changed.json'
# resFile = resFile%(dataDir, prefix, dataType, annType)
cocoDt=cocoGt.loadRes(resFile)

imgIds=sorted(cocoGt.getImgIds())
# imgIds=imgIds[0:100]
# imgId = imgIds[np.random.randint(100)]

breakpoint()

# running evaluation
cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.params.imgIds  = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()



# model = YOLO('/home/zijianwu/Codes/yolopose/runs/pose/train8/weights/best.pt')

# metrics = model.val(save_json=True)  # no arguments needed, dataset and settings remembered

