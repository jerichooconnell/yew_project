#!/usr/bin/env python3
"""
Simple CNN for Google Satellite Embeddings
===========================================

A lightweight CNN designed specifically for 64-channel embeddings.
Since embeddings are already high-level features, we don't need 
a deep architecture like ResNet.
"""

import torch
import torch.nn as nn


class SimpleEmbeddingCNN(nn.Module):
    """
    Simple CNN for 64-channel satellite embeddings.
    
    Architecture:
    - 3 convolutional blocks with batch norm and max pooling
    - Global average pooling
    - Fully connected classifier
    
    This is much lighter than ResNet and more appropriate for 
    already-processed embedding features.
    """
    
    def __init__(self, num_classes=2, dropout=0.5):
        super(SimpleEmbeddingCNN, self).__init__()
        
        # Input: 64 x 64 x 64
        self.features = nn.Sequential(
            # Block 1: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64x64 -> 32x32
            
            # Block 2: 128 -> 256 channels
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
            
            # Block 3: 256 -> 512 channels
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8
        )
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_model(num_classes=2, dropout=0.5):
    """Create and return the model."""
    return SimpleEmbeddingCNN(num_classes=num_classes, dropout=dropout)


if __name__ == '__main__':
    # Test model
    model = SimpleEmbeddingCNN(num_classes=2)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: SimpleEmbeddingCNN")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    x = torch.randn(2, 64, 64, 64)
    y = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output values: {y}")
