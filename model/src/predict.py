import torch
from PIL import Image
from model import CustomResNet
from dataset import transform
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_image(folder_path):
    # model = SimpleCNN().to(device)
    model = CustomResNet().to(device)
    model.load_state_dict(torch.load(
        "../models/tank_model_20250408-195041.pth"))
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

            results[filename] = "Not Tank" if predicted.item() == 0 else "Tank"

    return results


print(predict_image("../dataset/test"))
