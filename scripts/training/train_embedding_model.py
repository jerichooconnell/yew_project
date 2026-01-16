#!/usr/bin/env python3
"""
Train ResNet on Google Satellite Embedding data for yew detection.

This script trains a ResNet model using 64-channel embedding inputs
instead of traditional spectral bands.

Usage:
    python scripts/training/train_embedding_model.py \
        --train-csv data/processed/train_split_filtered.csv \
        --val-csv data/processed/val_split_filtered.csv \
        --image-dir data/ee_imagery/embedding_patches_64x64 \
        --architecture resnet18 \
        --epochs 50 \
        --batch-size 16

Author: GitHub Copilot
Date: November 20, 2025
"""

from dataset_embedding import get_dataloaders
from simple_embedding_model import SimpleEmbeddingCNN
from model_embedding import ResNetEmbedding
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({
            'loss': f'{running_loss / (pbar.n + 1):.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

            pbar.set_postfix({
                'loss': f'{running_loss / (pbar.n + 1):.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc, all_labels, all_preds, all_probs


def plot_training_curves(train_losses, train_accs, val_losses, val_accs, save_path):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss',
                  fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy',
                  fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(labels, preds, save_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Yew', 'Yew'],
                yticklabels=['No Yew', 'Yew'])
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Train ResNet on Google Satellite Embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--train-csv', type=str, required=True,
                        help='Training metadata CSV')
    parser.add_argument('--val-csv', type=str, required=True,
                        help='Validation metadata CSV')
    parser.add_argument('--image-dir', type=str, required=True,
                        help='Directory containing embedding patches')
    parser.add_argument('--architecture', type=str, default='simple',
                        choices=['simple', 'resnet18', 'resnet34', 'resnet50'],
                        help='Model architecture (simple=lightweight CNN, resnet=full ResNet)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"{args.architecture}_embedding_{timestamp}"

    print(f"\n{'='*80}")
    print(
        f"TRAINING {args.architecture.upper()} ON GOOGLE SATELLITE EMBEDDINGS")
    print(f"{'='*80}")
    print(f"Experiment: {experiment_name}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"{'='*80}\n")

    # Load data
    print("Loading data...")
    train_loader, val_loader, train_dataset, val_dataset = get_dataloaders(
        args.train_csv,
        args.val_csv,
        args.image_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    print(f"  Train size: {len(train_dataset)}")
    print(f"  Val size: {len(val_dataset)}")

    # Create model
    if args.architecture == 'simple':
        print(f"\nCreating SimpleEmbeddingCNN model...")
        model = SimpleEmbeddingCNN(num_classes=2, dropout=0.5)
    else:
        print(f"\nCreating {args.architecture} model...")
        model = ResNetEmbedding(
            architecture=args.architecture,
            num_classes=2,
            pretrained=True
        )
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training loop
    best_val_acc = 0.0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    print(f"\nStarting training...\n")

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print("-" * 80)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validate
        val_loss, val_acc, val_labels, val_preds, val_probs = validate(
            model, val_loader, criterion, device
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Learning rate scheduling
        scheduler.step(val_loss)

        print(f"\nEpoch Summary:")
        print(
            f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1

            # Save checkpoint
            checkpoint_dir = Path('models/checkpoints')
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            checkpoint = {
                'epoch': epoch + 1,
                'architecture': args.architecture,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'train_csv': args.train_csv,
                'val_csv': args.val_csv,
                'image_dir': args.image_dir,
                'input_type': 'embedding_64d',
                'experiment_name': experiment_name
            }

            torch.save(checkpoint, checkpoint_dir /
                       f"{experiment_name}_best.pth")
            print(
                f"  ✓ Saved best model (epoch {best_epoch}, val_acc: {best_val_acc*100:.2f}%)")

        # Save periodic checkpoints
        if (epoch + 1) % 5 == 0:
            checkpoint_dir = Path('models/checkpoints')
            torch.save(checkpoint, checkpoint_dir /
                       f"{experiment_name}_epoch{epoch + 1}.pth")

        print()

    # Final evaluation
    print(f"\n{'='*80}")
    print("FINAL EVALUATION")
    print(f"{'='*80}")
    print(
        f"Best Validation Accuracy: {best_val_acc*100:.2f}% (Epoch {best_epoch})")

    # Load best model for final metrics
    best_checkpoint = torch.load(
        checkpoint_dir / f"{experiment_name}_best.pth")
    model.load_state_dict(best_checkpoint['model_state_dict'])

    _, _, val_labels, val_preds, val_probs = validate(
        model, val_loader, criterion, device)

    # Calculate metrics
    auc = roc_auc_score(val_labels, val_probs)
    cm = confusion_matrix(val_labels, val_preds)
    report = classification_report(val_labels, val_preds,
                                   target_names=['No Yew', 'Yew'],
                                   output_dict=True)

    print(f"\nAUC-ROC: {auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"\nClassification Report:")
    print(classification_report(val_labels,
          val_preds, target_names=['No Yew', 'Yew']))

    # Save results
    results_dir = Path('results/training')
    results_dir.mkdir(parents=True, exist_ok=True)

    # Plot training curves
    plot_training_curves(
        train_losses, train_accs, val_losses, val_accs,
        results_dir / f"{experiment_name}_training_curves.png"
    )

    # Plot confusion matrix
    plot_confusion_matrix(
        val_labels, val_preds,
        results_dir / f"{experiment_name}_evaluation.png"
    )

    # Save metrics
    metrics = {
        'val_loss': float(best_checkpoint['val_loss']),
        'val_acc': float(best_val_acc),
        'auc': float(auc),
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }

    with open(results_dir / f"{experiment_name}_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Best model: models/checkpoints/{experiment_name}_best.pth")
    print(f"Results: {results_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
