import torch
import time
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloader
from model import SimpleCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(num_epochs=5):
    model = SimpleCNN().to(device)
    train_loader = get_dataloader()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_filename = f"../models/tank_model_{timestamp}.pth"
    torch.save(model.state_dict(), model_filename)
    print("Model saved!")


train_model()
