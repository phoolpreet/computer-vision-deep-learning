# Model definition
from torch import nn
from torchvision import models


class PetClassifierResNet(nn.Module):
    def __init__(self, num_classes=37):
        super(PetClassifierResNet, self).__init__()
        # Load a pre-trained ResNet18 model
        self.backbone = models.resnet18(weights="IMAGENET1K_V1")

        # for name, param in self.backbone.named_parameters():
        # print(name, param.shape)
        # print(len(list(self.backbone.named_parameters())))

        # Freeze the early layers
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(num_features, 256),  # New hidden layer with 256 units
            nn.ReLU(),  # Activation function for the hidden layer
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),  # Output layer
        )

    def forward(self, x):
        return self.backbone(x)
