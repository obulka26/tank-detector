import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


def get_dataloader(batch_size=32):
    train_data = datasets.ImageFolder(root="../dataset", transform=transform)
    return DataLoader(train_data, batch_size=batch_size, shuffle=True)
