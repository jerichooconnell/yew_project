#!/usr/bin/env python3
"""
Quick Training Monitor
======================

Monitor training progress in real-time.

Usage:
    python scripts/training/monitor_training.py
"""

import json
import time
from pathlib import Path


def monitor_training():
    """Monitor the latest training run."""
    results_dir = Path('results/training')
    checkpoint_dir = Path('models/checkpoints')

    print("\n" + "="*80)
    print("TRAINING MONITOR")
    print("="*80 + "\n")

    # Find latest experiment
    checkpoints = sorted(checkpoint_dir.glob(
        'resnet*_best.pth'), key=lambda x: x.stat().st_mtime)

    if not checkpoints:
        print("No training runs found yet.")
        return

    latest_checkpoint = checkpoints[-1]
    experiment_name = latest_checkpoint.stem.replace('_best', '')

    print(f"Latest experiment: {experiment_name}\n")

    # Load checkpoint
    import torch
    checkpoint = torch.load(latest_checkpoint, map_location='cpu')

    history = checkpoint.get('history', {})

    if not history:
        print("No training history found yet.")
        return

    epochs_completed = len(history['train_loss'])

    print(f"Epochs completed: {epochs_completed}")
    print(f"\nBest metrics so far:")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    print(f"  Val Acc: {checkpoint['val_acc']*100:.2f}%")
    print(f"\nLatest epoch:")
    print(f"  Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"  Train Acc: {history['train_acc'][-1]*100:.2f}%")
    print(f"  Val Loss: {history['val_loss'][-1]:.4f}")
    print(f"  Val Acc: {history['val_acc'][-1]*100:.2f}%")
    print(f"  Learning Rate: {history['learning_rate'][-1]:.6f}")

    # Check if training curves exist
    curve_file = results_dir / f'{experiment_name}_training_curves.png'
    if curve_file.exists():
        print(f"\n✓ Training curves: {curve_file}")

    # Check if final evaluation exists
    eval_file = results_dir / f'{experiment_name}_evaluation.png'
    metrics_file = results_dir / f'{experiment_name}_metrics.json'

    if metrics_file.exists():
        print(f"✓ Final evaluation complete!")
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        print(f"\nFinal Test Metrics:")
        print(f"  Accuracy: {metrics['val_acc']*100:.2f}%")
        print(f"  AUC: {metrics['auc']:.4f}")

        report = metrics['classification_report']
        print(f"\n  Yew Detection:")
        print(f"    Precision: {report['Yew']['precision']:.4f}")
        print(f"    Recall: {report['Yew']['recall']:.4f}")
        print(f"    F1-Score: {report['Yew']['f1-score']:.4f}")

    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    monitor_training()
