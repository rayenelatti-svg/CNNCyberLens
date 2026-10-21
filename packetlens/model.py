from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class TrafficCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(64, 128), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(128, num_classes),
        )

    def forward(self, inputs):
        return self.classifier(self.features(inputs))


class EnhancedTrafficCNN(nn.Module):
    """Normalized three-block CNN for byte-map classification."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs):
        return self.classifier(self.features(inputs))


class MobileNetTraffic(nn.Module):
    """ImageNet-pretrained MobileNetV3 adapted for one-channel byte maps."""

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights)
        first = self.backbone.features[0][0]
        grayscale = nn.Conv2d(
            1, first.out_channels, first.kernel_size, first.stride,
            first.padding, bias=False,
        )
        with torch.no_grad():
            grayscale.weight.copy_(first.weight.mean(dim=1, keepdim=True))
        self.backbone.features[0][0] = grayscale
        self.backbone.classifier[3] = nn.Linear(
            self.backbone.classifier[3].in_features, num_classes
        )

    def forward(self, inputs):
        inputs = nn.functional.interpolate(
            inputs, size=(224, 224), mode="bilinear", align_corners=False
        )
        inputs = (inputs - 0.485) / 0.229
        return self.backbone(inputs)


def create_model(name: str, num_classes: int, pretrained: bool = True):
    if name == "mobilenet_v3_small":
        return MobileNetTraffic(num_classes, pretrained=pretrained)
    if name == "native":
        return TrafficCNN(num_classes)
    if name == "native_enhanced":
        return EnhancedTrafficCNN(num_classes)
    raise ValueError(f"Unsupported model: {name}")
