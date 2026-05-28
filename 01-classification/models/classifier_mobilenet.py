from torch import nn
from torchvision import models


class ClassifierMobileNet(nn.Module):
    def __init__(self, num_classes, dropout=0.25):
        super(ClassifierMobileNet, self).__init__()

        # Load pretrained MobileNetV2
        self.backbone = models.mobilenet_v2(weights="DEFAULT")

        # Freeze early layers
        # for param in list(self.backbone.parameters())[:-10]:
        #     # print(param.names)
        #     param.requires_grad = False

        # Get input features of classifier
        num_features = self.backbone.classifier[1].in_features

        # Replace classifier head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)