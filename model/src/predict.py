import torch
from PIL import Image
from model import SimpleCNN
from dataset import transform
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_image(folder_path):
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("../models/tank_model_20250408-165406.pth"))
    model.eval()

    results = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith("png"):
            image_path = os.path.join(folder_path, filename)

            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image)
            image_tensor = image_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(image_tensor)
                _, predicted = torch.max(output, 1)

            results[filename] = "Tank" if predicted.item() == 0 else "Not Tank"

    return results


print(predict_image("../dataset/test"))
