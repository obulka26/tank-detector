# import torch
# import torch.nn as nn
#
#
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(32 * 32 * 32, 128)
#         self.fc2 = nn.Linear(128, 2)
#         self.dropout = nn.Dropout(0.5)
#
#     def forward(self, x):
#         x = self.pool(torch.relu(self.conv1(x)))
#         x = self.dropout(x)
#         x = self.pool(torch.relu(self.conv2(x)))
#         x = self.dropout(x)
#         x = x.view(x.shape[0], -1)
#         x = torch.relu(self.fc1(x))
#         x = self.dropout(x)
#         return self.fc2(x)

import torch
import torch.nn as nn
import torchvision.models as models


class CustomResNet(nn.Module):
    def __init__(self):
        super(CustomResNet, self).__init__()
        self.model = models.resnet18(
            weights="IMAGENET1K_V1"
        )  # Завантажуємо передтреновану модель
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),  # Додаємо Dropout, як у твоїй моделі
            nn.Linear(num_ftrs, 2),  # 2 виходи: "Tank" і "Not Tank"
        )

    def forward(self, x):
        return self.model(x)
