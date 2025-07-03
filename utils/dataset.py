import torch
from torchvision import datasets, transforms

def create_data_loader():
    train_data = datasets.CIFAR100(root="./cifar100_data", train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.CIFAR100(root="./cifar100_data", train=False, download=True, transform=transforms.ToTensor())

    train_size = int(len(train_data) * 0.85)
    val_size = len(train_data) - train_size

    train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)
    return train_loader, val_loader, test_loader


