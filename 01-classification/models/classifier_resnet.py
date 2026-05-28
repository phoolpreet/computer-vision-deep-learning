# Model definition
from torch import nn
from torchvision import models


class ClassifierResNet(nn.Module):
    def __init__(self, num_classes=37, dropout=0.25):
        super(ClassifierResNet, self).__init__()
        # Load a pre-trained ResNet model
        self.backbone = models.resnet50(weights="DEFAULT")


        # Freeze the early layers
        for param in list(self.backbone.parameters())[:-10]:
            param.requires_grad = False

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 128),  # New hidden layer with 256 units
            nn.ReLU(),  # Activation function for the hidden layer
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),  # Output layer
        )

    def forward(self, x):
        return self.backbone(x)
