"""Image-level ResNet-50 and DenseNet-121 classification branches."""

import torch
from torch import nn
from torchvision.models import densenet121, resnet50


class ResNetBranch(nn.Module):
    """ResNet-50 with an explicit feature and logit output."""

    def __init__(self, num_classes: int = 2, pretrained: bool = False) -> None:
        super().__init__()
        weights = "IMAGENET1K_V2" if pretrained else None
        network = resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(network.children())[:-1])
        self.classifier = nn.Linear(network.fc.in_features, num_classes)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(images).flatten(1)
        return self.classifier(features), features


class DenseNetBranch(nn.Module):
    """DenseNet-121 with an explicit feature and logit output."""

    def __init__(self, num_classes: int = 2, pretrained: bool = False) -> None:
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        network = densenet121(weights=weights)
        self.backbone = network.features
        self.classifier = nn.Linear(network.classifier.in_features, num_classes)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.nn.functional.relu(self.backbone(images), inplace=False)
        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        return self.classifier(features), features
