#!/usr/bin/env python3
"""
ResNet Model for 4-Channel Satellite Imagery
=============================================

Modified ResNet to accept 4-channel input (RGB + NIR).

Author: GitHub Copilot
Date: November 14, 2025
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet4Channel(nn.Module):
    """ResNet modified to accept 4-channel input (B, G, R, NIR)."""

    def __init__(self, architecture='resnet18', pretrained=True, num_classes=2):
        """
        Args:
            architecture: 'resnet18', 'resnet34', 'resnet50'
            pretrained: Use ImageNet pretrained weights
            num_classes: Number of output classes
        """
        super(ResNet4Channel, self).__init__()

        # Load pretrained ResNet
        if architecture == 'resnet18':
            self.resnet = models.resnet18(pretrained=pretrained)
        elif architecture == 'resnet34':
            self.resnet = models.resnet34(pretrained=pretrained)
        elif architecture == 'resnet50':
            self.resnet = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        # Modify first conv layer to accept 4 channels
        original_conv1 = self.resnet.conv1

        # Create new conv layer with 4 input channels
        self.resnet.conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False
        )

        # Initialize new conv layer
        if pretrained:
            # Copy RGB weights from pretrained model
            with torch.no_grad():
                # Average the RGB weights and use for all 4 channels
                rgb_weights = original_conv1.weight.data
                # Initialize all 4 channels with averaged RGB weights
                avg_weights = rgb_weights.mean(dim=1, keepdim=True)
                self.resnet.conv1.weight.data = avg_weights.repeat(1, 4, 1, 1)

                # Alternative: Copy RGB to first 3 channels, initialize NIR channel
                # self.resnet.conv1.weight.data[:, :3, :, :] = rgb_weights
                # self.resnet.conv1.weight.data[:, 3:4, :, :] = rgb_weights.mean(dim=1, keepdim=True)

        # Modify final fully connected layer for binary classification
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

        self.architecture = architecture

    def forward(self, x):
        """Forward pass."""
        return self.resnet(x)

    def get_num_params(self):
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(architecture='resnet18', pretrained=True, num_classes=2):
    """
    Create a ResNet model for 4-channel imagery.

    Args:
        architecture: ResNet architecture ('resnet18', 'resnet34', 'resnet50')
        pretrained: Use ImageNet pretrained weights
        num_classes: Number of output classes (default: 2 for binary)

    Returns:
        model: ResNet4Channel model
    """
    model = ResNet4Channel(
        architecture=architecture,
        pretrained=pretrained,
        num_classes=num_classes
    )

    print(f"Created {architecture} model:")
    print(f"  Pretrained: {pretrained}")
    print(f"  Input channels: 4 (B, G, R, NIR)")
    print(f"  Output classes: {num_classes}")
    print(f"  Parameters: {model.get_num_params():,}")

    return model


if __name__ == '__main__':
    # Test model creation
    print("Testing model creation...\n")

    for arch in ['resnet18', 'resnet34']:
        print(f"\n{'='*60}")
        model = create_model(arch, pretrained=True)

        # Test forward pass
        x = torch.randn(2, 4, 64, 64)  # Batch of 2 images
        y = model(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {y.shape}")
        print(f"{'='*60}")

    print("\n✓ Model creation successful!")
