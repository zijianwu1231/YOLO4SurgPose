from ultralytics import YOLO

model = YOLO('/home/zijianwu/Codes/yolopose/runs/pose/train8/weights/best.pt')

metrics = model.val(save_json=True)  # no arguments needed, dataset and settings remembered
breakpoint()
print(metrics.box.map)  # map50-95

# metrics.box.map50  # map50
# metrics.box.map75  # map75
# metrics.box.maps  # a list contains map50-95 of each category