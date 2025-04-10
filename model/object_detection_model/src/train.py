from ultralytics import YOLO
import time

model = YOLO("yolov5n.pt")

timestamp = time.strftime("%Y%m%d-%H%M%S")

model.train(
    data="../dataset/data.yaml",
    epochs=30,
    imgsz=640,
    project="../models/",
    name=f"tank_detection_{timestamp}",
    batch=2,
    workers=4,
    patience=10,
    cos_lr=True,
    optimizer="AdamW",
    label_smoothing=0.1,
    weight_decay=0.0005,
    hsv_h=0.01,
    hsv_s=0.4,
    hsv_v=0.4,
    fliplr=0.5,
    mosaic=0.5,
    scale=0.3,
    shear=0.1,
    amp=True,
)
