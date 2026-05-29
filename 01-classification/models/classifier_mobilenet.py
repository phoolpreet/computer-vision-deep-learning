from torch import nn
from torchvision import models


class ClassifierMobileNet(nn.Module):
    def __init__(self, num_classes, dropout=0.25):
        super(ClassifierMobileNet, self).__init__()

        # Load pretrained MobileNetV2
        # self.backbone = models.mobilenet_v2(weights="DEFAULT")
        self.backbone = models.mobilenet_v3_large(weights="DEFAULT")

        # Freeze early layers
        for param in list(self.backbone.parameters())[:-10]:
            # print(param.names)
            param.requires_grad = False

        # Get input features of classifier
        num_features = self.backbone.classifier[0].in_features

        # Replace classifier head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)