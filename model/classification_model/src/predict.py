import torch
from PIL import Image
from torchvision.transforms import transforms
from model.classification_model.src.model import CustomResNet
from model.classification_model.src.dataset import transform
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = CustomResNet().to(device)
model.load_state_dict(
    torch.load("model/classification_model/models/tank_model_20250408-195041.pth")
)
model.eval()


def predict_folder(folder_path):
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


def predict_single_image(image: Image.Image) -> str:
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    return "Not Tank" if predicted.item() == 0 else "Tank"
