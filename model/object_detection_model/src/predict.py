from PIL import Image
from ultralytics import YOLO
import os

model = YOLO(
    "model/object_detection_model/models/attempt_2/tank_detection_20250410-144504/weights/best.pt"
)


def predict_single_image(image: Image.Image):
    results = model(image)

    boxes = results[0].boxes.xyxy.tolist() if results[0].boxes is not None else []
    scores = results[0].boxes.conf.tolist() if results[0].boxes is not None else []
    labels = (
        [results[0].names[int(cls)] for cls in results[0].boxes.cls]
        if results[0].boxes is not None
        else []
    )

    predictions = [
        {"label": label, "confidence": score, "bbox": box}
        for label, score, box in zip(labels, scores, boxes)
    ]

    result_image = results[0].plot()

    return predictions, result_image


def predict_folder(folder_path: str):
    results_dict = {}

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(folder_path, filename)
            results = model(image_path)
            results[0].save()

            boxes = results[0].boxes
            names = results[0].names
            predictions = []

            if boxes is not None:
                for box, cls_id, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
                    label = names[int(cls_id)]
                    predictions.append(
                        f"Predicted: {label} with confidence {conf:.2f}, Bounding Box: {
                            box
                        }"
                    )

            results_dict[filename] = predictions

    return results_dict
