#!/usr/bin/env python3
"""
ResNet model adapted for 64-channel Google Satellite Embedding input.

Instead of using 4 spectral bands (B, G, R, NIR), this model uses
64-dimensional embedding vectors from Google's Satellite Embedding dataset.

Author: GitHub Copilot
Date: November 20, 2025
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNetEmbedding(nn.Module):
    """
    ResNet adapted for 64-channel embedding input.

    This model replaces the first convolutional layer to accept 64-channel
    embedding vectors instead of 3-channel RGB input.
    """

    def __init__(self, architecture='resnet18', num_classes=2, pretrained=True):
        """
        Initialize ResNet for embedding input.

        Args:
            architecture: ResNet variant ('resnet18', 'resnet34', 'resnet50')
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights (adapted for 64 channels)
        """
        super(ResNetEmbedding, self).__init__()

        # Load pretrained ResNet
        if architecture == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
        elif architecture == 'resnet34':
            self.model = models.resnet34(pretrained=pretrained)
        elif architecture == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        # Get original first conv layer parameters
        original_conv = self.model.conv1

        # Create new first conv layer with 64 input channels
        self.model.conv1 = nn.Conv2d(
            in_channels=64,  # 64 embedding dimensions
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )

        # Always use He initialization for the first layer
        # This is more appropriate for 64-channel embeddings than adapted ImageNet weights
        nn.init.kaiming_normal_(self.model.conv1.weight,
                                mode='fan_out', nonlinearity='relu')

        # Replace final fully connected layer
        if architecture in ['resnet18', 'resnet34']:
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)
        else:  # resnet50, resnet101, resnet152
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        """Forward pass."""
        return self.model(x)


def get_model(architecture='resnet18', num_classes=2, pretrained=True):
    """
    Factory function to create ResNet embedding model.

    Args:
        architecture: ResNet variant
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights

    Returns:
        ResNetEmbedding model
    """
    return ResNetEmbedding(architecture, num_classes, pretrained)


if __name__ == '__main__':
    # Test model creation
    model = get_model('resnet18', num_classes=2, pretrained=True)

    # Test forward pass with 64-channel input
    x = torch.randn(4, 64, 64, 64)  # (batch, channels, height, width)
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
