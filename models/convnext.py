import torch.nn as nn
from torchvision.models import convnext_tiny

class ConvNextTiny(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNextTiny, self).__init__()
    
        self.backbone = convnext_tiny(weights="DEFAULT" or None)

        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        # return self.resnet(x)
        return self.backbone(x)