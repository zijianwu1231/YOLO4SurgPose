from ultralytics import YOLO
from ultralytics.data.augment import Albumentations
from ultralytics.utils import LOGGER, colorstr

# def __init__(self, p=1.0):
#     """Initialize the transform object for YOLO bbox formatted params."""
#     self.p = p
#     self.transform = None
#     prefix = colorstr("albumentations: ")
#     try:
#         import albumentations as A         

#         # Insert required transformation here
#         T = [ A.RandomRain(p=0.1, slant_lower=-10, slant_upper=10, 
#                            drop_length=20, drop_width=1, drop_color=(200, 200, 200), 
#                            blur_value=5, brightness_coefficient=0.9, rain_type=None),
#                 A.Rotate(limit = 10, p=0.5),
#                 A.Blur(p=0.1),
#                 A.MedianBlur(p=0.1),
#                 A.ToGray(p=0.01),
#                 A.CLAHE(p=0.01),
#                 A.ImageCompression(quality_lower=75, p=0.0),
#             ]
#         self.transform = A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

#         LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
#     except ImportError:  # package not installed, skip
#         pass
#     except Exception as e:
#         LOGGER.info(f"{prefix}{e}")


model = YOLO('yolov8x-pose.pt')
# model = YOLO('yolov8x-pose-p6.pt')
# model = YOLO('yolov8l-pose.pt')

results = model.train(data='./surgpose-pose.yaml', epochs=50, imgsz=640, optimizer='AdamW', weight_decay=0.1, lr0=1e-6, cos_lr=True, dropout=0.5,
                      augment=True,
                    # box=7.5, cls=0.5, dfl=1.5, pose=24, kobj=1.0,
                    degrees=90,
                    hsv_h=0.5,
                    hsv_s=0.9,
                    hsv_v=0.8,
                    translate=0.5,
                    scale=0.7,
                    shear=90,
                    perspective=0.001,
                    bgr=0.5,
                    mosaic=1.0,
                    mixup=0.5,
                    copy_paste=0.5,
                    auto_augment='randaugment',
                    erasing=0.9,
                    )
