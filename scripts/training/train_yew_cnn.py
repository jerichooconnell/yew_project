#!/usr/bin/env python3
"""
Train CNN for Yew Detection using 64x64 Image Patches
=====================================================

Trains a convolutional neural network on Sentinel-2 image patches
to classify yew presence in forest sites.

Author: GitHub Copilot
Date: November 7, 2025
"""

from yew_image_dataset import create_data_loaders
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('scripts/preprocessing')


class YewCNN(nn.Module):
    """
    Convolutional Neural Network for yew detection from 4-band satellite imagery.

    Input: (batch, 4, 64, 64) - [Blue, Green, Red, NIR]
    Output: (batch, 2) - [no_yew, yew] logits
    """

    def __init__(self, dropout_rate=0.5):
        super(YewCNN, self).__init__()

        # Convolutional layers
        self.features = nn.Sequential(
            # Block 1: 64x64 -> 32x32
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            # Block 2: 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            # Block 3: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            # Block 4: 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Store for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            probs = torch.softmax(outputs, dim=1)
            # Probability of yew class
            all_probs.extend(probs[:, 1].cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_training_history(history, save_path):
    """Plot training and validation metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history['train_loss'], label='Train', marker='o')
    axes[0].plot(history['val_loss'], label='Validation', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history['train_acc'], label='Train', marker='o')
    axes[1].plot(history['val_acc'], label='Validation', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✓ Training history saved: {save_path}")
    plt.close()


def plot_confusion_matrix(cm, save_path):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['No Yew', 'Yew'],
        yticklabels=['No Yew', 'Yew']
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix - Test Set')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✓ Confusion matrix saved: {save_path}")
    plt.close()


def main():
    print("="*80)
    print("CNN TRAINING FOR YEW DETECTION")
    print("="*80)

    # Configuration
    config = {
        'image_dir': 'data/ee_imagery/image_patches_64x64',
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'yew_weight': 10.0,
        'num_workers': 4,
        'dropout_rate': 0.5
    }

    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create data loaders
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)

    metadata_csv = f"{config['image_dir']}/image_metadata.csv"

    train_loader, val_loader, test_loader = create_data_loaders(
        image_dir=config['image_dir'],
        metadata_csv=metadata_csv,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        yew_weight=config['yew_weight']
    )

    # Create model
    print("\n" + "="*80)
    print("BUILDING MODEL")
    print("="*80)

    model = YewCNN(dropout_rate=config['dropout_rate']).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    # Use weighted loss to handle class imbalance
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_loss = float('inf')
    best_model_path = Path('models/checkpoints/yew_cnn_best.pth')
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc, _, _, _ = validate(
            model, val_loader, criterion, device
        )

        # Update learning rate
        scheduler.step(val_loss)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'config': config
            }, best_model_path)
            print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")

    # Load best model for testing
    print("\n" + "="*80)
    print("TESTING ON BEST MODEL")
    print("="*80)

    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']+1}")

    # Test
    test_loss, test_acc, test_preds, test_labels, test_probs = validate(
        model, test_loader, criterion, device
    )

    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.2f}%")

    # Detailed metrics
    print("\n" + "-"*80)
    print("Classification Report:")
    print("-"*80)
    print(classification_report(
        test_labels, test_preds,
        target_names=['No Yew', 'Yew'],
        digits=4
    ))

    # ROC AUC
    if len(np.unique(test_labels)) > 1:
        auc = roc_auc_score(test_labels, test_probs)
        print(f"ROC AUC Score: {auc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    print("\nConfusion Matrix:")
    print(cm)

    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    output_dir = Path('results/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot training history
    plot_training_history(history, output_dir / 'yew_cnn_training_history.png')

    # Plot confusion matrix
    plot_confusion_matrix(cm, output_dir / 'yew_cnn_confusion_matrix.png')

    # Save final model
    final_model_path = Path('models/artifacts/yew_cnn_final.pth')
    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), final_model_path)
    print(f"✓ Final model saved: {final_model_path}")

    # Save training summary
    summary = {
        'config': config,
        'best_epoch': checkpoint['epoch'] + 1,
        'best_val_loss': float(checkpoint['val_loss']),
        'best_val_acc': float(checkpoint['val_acc']),
        'test_loss': float(test_loss),
        'test_acc': float(test_acc),
        'test_auc': float(auc) if len(np.unique(test_labels)) > 1 else None,
        'training_date': datetime.now().isoformat()
    }

    summary_path = Path('results/reports/yew_cnn_training_summary.json')
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Training summary saved: {summary_path}")

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
