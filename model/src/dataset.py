import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch as torch
from torch.utils.data import DataLoader, random_split
from collections import Counter

transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


def get_dataloader(batch_size=32, train_ratio=0.8):
    dataset = datasets.ImageFolder(
        root="../dataset/train/", transform=transform)
    print(dataset.class_to_idx)

    class_counts = Counter(label for _, label in dataset.samples)

    print("Кількість зображень у кожному класі:", class_counts)
    torch.manual_seed(42)
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    train_labels = [label for _, label in train_dataset]
    val_labels = [label for _, label in val_dataset]

    print("Train classes:", set(train_labels))
    print("Val classes:", set(val_labels))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    for images, labels in train_loader:
        print("First batch labels:", labels.tolist())
        break

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader
