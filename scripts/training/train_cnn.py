#!/usr/bin/env python3
"""
CNN Training Pipeline for Yew Detection
========================================

Train a ResNet model to detect Pacific Yew from satellite imagery.

Author: GitHub Copilot
Date: November 14, 2025

Usage:
    python scripts/training/train_cnn.py --architecture resnet18 --epochs 50
"""

import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import seaborn as sns
from datetime import datetime
from tqdm import tqdm

from model import create_model
from dataset import get_dataloaders


class Trainer:
    """Training manager for yew detection CNN."""

    def __init__(self, model, train_loader, val_loader, config):
        """
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration dict
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Setup device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"\nUsing device: {self.device}")

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # Tracking
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }

        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.epochs_no_improve = 0

        # Setup output directories
        self.checkpoint_dir = Path('models/checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.results_dir = Path('results/training')
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Experiment name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_name = f"{config['architecture']}_{timestamp}"

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels, _ in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track metrics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    def validate(self):
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        all_labels = []
        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            for images, labels, _ in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Track metrics
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Store for metrics
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

                # Get probabilities
                probs = torch.softmax(outputs, dim=1)
                all_probabilities.extend(probs[:, 1].cpu().numpy())

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        return epoch_loss, epoch_acc, all_labels, all_predictions, all_probabilities

    def train(self, num_epochs):
        """Train the model."""
        print(f"\n{'='*80}")
        print(f"TRAINING {self.config['architecture'].upper()}")
        print(f"{'='*80}")
        print(f"Experiment: {self.experiment_name}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Learning rate: {self.config['learning_rate']}")
        print(f"{'='*80}\n")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print('-' * 60)

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc, val_labels, val_preds, val_probs = self.validate()

            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)

            # Print summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(
                f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.epochs_no_improve = 0

                checkpoint_path = self.checkpoint_dir / \
                    f'{self.experiment_name}_best.pth'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'config': self.config,
                    'history': self.history
                }, checkpoint_path)

                print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
            else:
                self.epochs_no_improve += 1

            # Early stopping
            if self.epochs_no_improve >= self.config['early_stopping_patience']:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

            # Save latest checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_path = self.checkpoint_dir / \
                    f'{self.experiment_name}_epoch{epoch+1}.pth'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'config': self.config,
                    'history': self.history
                }, checkpoint_path)

        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"Best Val Acc: {self.best_val_acc*100:.2f}%")
        print(f"{'='*80}\n")

        return self.history

    def plot_training_history(self):
        """Plot training curves."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        epochs = range(1, len(self.history['train_loss']) + 1)

        # Loss
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train')
        axes[0].plot(epochs, self.history['val_loss'],
                     'r-', label='Validation')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy
        axes[1].plot(
            epochs, [a*100 for a in self.history['train_acc']], 'b-', label='Train')
        axes[1].plot(epochs, [a*100 for a in self.history['val_acc']],
                     'r-', label='Validation')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Learning rate
        axes[2].plot(epochs, self.history['learning_rate'], 'g-')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        save_path = self.results_dir / \
            f'{self.experiment_name}_training_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves to: {save_path}")

        plt.close()

    def evaluate_final(self):
        """Final evaluation with detailed metrics."""
        print(f"\n{'='*80}")
        print("FINAL EVALUATION")
        print(f"{'='*80}\n")

        # Load best model
        checkpoint_path = self.checkpoint_dir / \
            f'{self.experiment_name}_best.pth'
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Evaluate
        val_loss, val_acc, val_labels, val_preds, val_probs = self.validate()

        # Classification report
        print("Classification Report:")
        print(classification_report(val_labels, val_preds,
                                    target_names=['No Yew', 'Yew'],
                                    digits=4))

        # Confusion matrix
        cm = confusion_matrix(val_labels, val_preds)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=['No Yew', 'Yew'],
                    yticklabels=['No Yew', 'Yew'])
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')
        axes[0].set_title('Confusion Matrix')

        # ROC curve
        fpr, tpr, thresholds = roc_curve(val_labels, val_probs)
        auc = roc_auc_score(val_labels, val_probs)

        axes[1].plot(fpr, tpr, 'b-', linewidth=2,
                     label=f'ROC (AUC = {auc:.3f})')
        axes[1].plot([0, 1], [0, 1], 'r--', label='Random')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        save_path = self.results_dir / f'{self.experiment_name}_evaluation.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved evaluation plots to: {save_path}")

        plt.close()

        # Save metrics
        metrics = {
            'val_loss': float(val_loss),
            'val_acc': float(val_acc),
            'auc': float(auc),
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(val_labels, val_preds,
                                                           target_names=[
                                                               'No Yew', 'Yew'],
                                                           output_dict=True)
        }

        metrics_path = self.results_dir / \
            f'{self.experiment_name}_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"Saved metrics to: {metrics_path}")

        return metrics


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train CNN for yew detection')
    parser.add_argument('--architecture', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help='ResNet architecture')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                        help='Weight decay')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split fraction')
    parser.add_argument('--early-stopping', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--use-filtered', action='store_true',
                        help='Use filtered dataset (CA removed, only "good" samples)')

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Configuration
    config = {
        'architecture': args.architecture,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'val_split': args.val_split,
        'early_stopping_patience': args.early_stopping,
        'num_workers': args.num_workers,
        'random_seed': args.seed,
        'use_filtered': args.use_filtered
    }

    # Data paths
    if args.use_filtered:
        print("\n*** Using FILTERED dataset (CA removed, only 'good' samples) ***\n")
        yew_metadata = 'data/processed/inat_yew_filtered_good.csv'
        no_yew_metadata = 'data/ee_imagery/image_patches_64x64/no_yew_image_metadata.csv'
        image_base_dir = 'data/ee_imagery/image_patches_64x64'
    else:
        print("\n*** Using ORIGINAL dataset (all samples) ***\n")
        yew_metadata = 'data/ee_imagery/image_patches_64x64/inat_yew_image_metadata.csv'
        no_yew_metadata = 'data/ee_imagery/image_patches_64x64/no_yew_image_metadata.csv'
        image_base_dir = 'data/ee_imagery/image_patches_64x64'

    # Create dataloaders
    print("Loading datasets...")
    train_loader, val_loader, dataset_info = get_dataloaders(
        yew_metadata, no_yew_metadata, image_base_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        random_seed=args.seed
    )

    print(f"\nDataset Summary:")
    for key, value in dataset_info.items():
        print(f"  {key}: {value}")

    # Create model
    print("\nCreating model...")
    model = create_model(
        architecture=args.architecture,
        pretrained=True,
        num_classes=2
    )

    # Train
    trainer = Trainer(model, train_loader, val_loader, config)
    history = trainer.train(args.epochs)

    # Plot results
    trainer.plot_training_history()

    # Final evaluation
    metrics = trainer.evaluate_final()

    print(f"\n{'='*80}")
    print("TRAINING PIPELINE COMPLETE")
    print(f"{'='*80}")
    print(f"Best model: models/checkpoints/{trainer.experiment_name}_best.pth")
    print(f"Results: results/training/{trainer.experiment_name}_*")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
