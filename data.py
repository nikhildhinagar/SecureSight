import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

def get_datasets():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize for ResNet and ViT
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for CIFAR-10
    ])
    train_data = CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_data = CIFAR10(root="./data", train=False, download=True, transform=transform)
    return train_data, test_data