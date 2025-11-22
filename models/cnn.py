import torch.nn as nn
from torchvision.models import resnet18

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        
        # Load pretrained ResNet18
        self.resnet = resnet18(pretrained=True)
        
        # Modify the final fully connected layer to match CIFAR-10 classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)