from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_mnist_loaders(data_dir: Path, batch_size: int, test_batch_size: int):
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader
