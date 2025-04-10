import os
from PIL import Image
from ultralytics import YOLO

model = YOLO("models/attempt_2/tank_detection_20250410-144504/weights/best.pt")


def predict_single_image(image_path: str):
    img = Image.open(image_path)
    results = model(image_path)

    results[0].show()
    results[0].save(filename="result.jpg")

    boxes = results[0].boxes
    names = results[0].names

    print(f"\nPredictions for {image_path}:")
    if boxes is not None:
        for box, cls_id, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
            label = names[int(cls_id)]
            print(
                f"  → Predicted: {label} with confidence {conf:.2f}, Bounding Box: {
                    box
                }"
            )


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
