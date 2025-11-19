#!/usr/bin/env python3
"""
Visualize model misclassifications - false positives and false negatives.

This tool identifies the images the model struggles with most and displays them
for analysis. Helps understand model weaknesses and guide data collection.

Usage:
    python scripts/visualization/visualize_misclassifications.py \
        --model models/checkpoints/resnet18_20251114_125300_best.pth \
        --data-split val \
        --top-k 20

Author: GitHub Copilot
Date: 2024-11-14
"""

from training.model import ResNet4Channel
from training.dataset import get_dataloaders
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.gridspec import GridSpec

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def load_model(model_path, device):
    """Load trained model from checkpoint."""
    print(f"Loading model from: {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Get architecture from checkpoint or default to resnet18
    architecture = checkpoint.get('architecture', 'resnet18')

    # Initialize model
    model = ResNet4Channel(architecture=architecture, num_classes=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"  Architecture: {architecture}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Val Accuracy: {checkpoint.get('val_accuracy', 'unknown'):.4f}" if checkpoint.get(
        'val_accuracy') else "")

    return model


def get_predictions(model, dataloader, device):
    """Get model predictions and probabilities for all samples."""
    all_preds = []
    all_probs = []
    all_labels = []
    all_metadata = []

    with torch.no_grad():
        for images, labels, metadata in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_metadata.extend(metadata)

    return (np.array(all_preds),
            np.array(all_probs),
            np.array(all_labels),
            all_metadata)


def find_misclassifications(predictions, probabilities, labels, metadata, top_k=20):
    """Find false positives and false negatives with highest confidence."""
    # Find misclassified samples
    misclassified = predictions != labels

    # Separate false positives and false negatives
    false_positives = misclassified & (
        predictions == 1)  # Predicted yew, actually not
    # Predicted not yew, actually yew
    false_negatives = misclassified & (predictions == 0)

    # Get confidence scores (probability of predicted class)
    confidence = np.array([probs[pred]
                          for probs, pred in zip(probabilities, predictions)])

    # Find top-k false positives (most confident wrong predictions)
    fp_indices = np.where(false_positives)[0]
    fp_confidences = confidence[fp_indices]
    top_fp_idx = fp_indices[np.argsort(fp_confidences)[-top_k:]][::-1]

    # Find top-k false negatives
    fn_indices = np.where(false_negatives)[0]
    fn_confidences = confidence[fn_indices]
    top_fn_idx = fn_indices[np.argsort(fn_confidences)[-top_k:]][::-1]

    # Compile results
    fp_results = []
    for idx in top_fp_idx:
        fp_results.append({
            'index': int(idx),
            'true_label': int(labels[idx]),
            'pred_label': int(predictions[idx]),
            'confidence': float(confidence[idx]),
            'prob_yew': float(probabilities[idx, 1]),
            'prob_not_yew': float(probabilities[idx, 0]),
            'metadata': metadata[idx]
        })

    fn_results = []
    for idx in top_fn_idx:
        fn_results.append({
            'index': int(idx),
            'true_label': int(labels[idx]),
            'pred_label': int(predictions[idx]),
            'confidence': float(confidence[idx]),
            'prob_yew': float(probabilities[idx, 1]),
            'prob_not_yew': float(probabilities[idx, 0]),
            'metadata': metadata[idx]
        })

    return fp_results, fn_results


def visualize_misclassifications(fp_results, fn_results, output_dir, image_base_dir):
    """Create visualization of false positives and false negatives."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Visualize false positives
    if fp_results:
        print(f"\nVisualizing {len(fp_results)} false positives...")
        fig = create_misclassification_grid(
            fp_results,
            title="False Positives (Model predicted YEW, actually NOT YEW)",
            color='red',
            image_base_dir=image_base_dir
        )
        fp_path = output_dir / 'false_positives.png'
        fig.savefig(fp_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {fp_path}")

    # Visualize false negatives
    if fn_results:
        print(f"Visualizing {len(fn_results)} false negatives...")
        fig = create_misclassification_grid(
            fn_results,
            title="False Negatives (Model predicted NOT YEW, actually YEW)",
            color='orange',
            image_base_dir=image_base_dir
        )
        fn_path = output_dir / 'false_negatives.png'
        fig.savefig(fn_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {fn_path}")

    # Create summary report
    create_summary_report(fp_results, fn_results, output_dir)


def create_misclassification_grid(results, title, color, image_base_dir):
    """Create grid visualization of misclassified images."""
    n_images = len(results)
    n_cols = min(5, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows + 1))
    gs = GridSpec(n_rows + 1, n_cols, figure=fig, hspace=0.4, wspace=0.3)

    # Title
    fig.suptitle(title, fontsize=16, fontweight='bold', color=color, y=0.98)

    # Plot each image
    for i, result in enumerate(results):
        row = i // n_cols
        col = i % n_cols
        ax = fig.add_subplot(gs[row, col])

        # Load image
        metadata = result['metadata']
        # Construct full path from base directory and relative path
        image_path = Path(image_base_dir) / metadata['image_path']

        if image_path.exists():
            image = np.load(image_path)

            # Convert to RGB (use bands 2,1,0 for true color: Red, Green, Blue)
            if image.shape[0] == 4:
                rgb = image[[2, 1, 0], :, :]  # R, G, B
            else:
                rgb = image[:3, :, :]

            # Normalize for display
            rgb = np.transpose(rgb, (1, 2, 0))
            rgb_norm = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
            rgb_norm = np.clip(rgb_norm, 0, 1)

            ax.imshow(rgb_norm)
        else:
            ax.text(0.5, 0.5, 'Image not found', ha='center', va='center')

        # Title with confidence
        confidence_pct = result['confidence'] * 100
        ax.set_title(
            f"Confidence: {confidence_pct:.1f}%\n"
            f"P(yew)={result['prob_yew']:.3f}",
            fontsize=10,
            color=color
        )

        # Add location info if available
        if metadata.get('latitude') is not None and metadata.get('longitude') is not None:
            ax.text(
                0.02, 0.98,
                f"Lat: {metadata['latitude']:.4f}\nLon: {metadata['longitude']:.4f}",
                transform=ax.transAxes,
                fontsize=8,
                va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
            )

        ax.axis('off')

    return fig


def create_summary_report(fp_results, fn_results, output_dir):
    """Create text summary of misclassifications."""
    report_path = output_dir / 'misclassification_summary.txt'

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MISCLASSIFICATION ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        # False Positives
        f.write(f"FALSE POSITIVES: {len(fp_results)}\n")
        f.write("(Model predicted YEW, but actually NOT YEW)\n")
        f.write("-" * 80 + "\n")
        for i, result in enumerate(fp_results, 1):
            metadata = result['metadata']
            f.write(f"\n{i}. Confidence: {result['confidence']*100:.1f}%\n")
            f.write(f"   P(yew) = {result['prob_yew']:.3f}\n")
            f.write(f"   Image: {metadata.get('image_path', 'unknown')}\n")
            if metadata.get('latitude') is not None and metadata.get('longitude') is not None:
                f.write(
                    f"   Location: {metadata['latitude']:.4f}, {metadata['longitude']:.4f}\n")
            if metadata.get('source_dataset') is not None:
                f.write(f"   Source: {metadata['source_dataset']}\n")

        f.write("\n\n")

        # False Negatives
        f.write(f"FALSE NEGATIVES: {len(fn_results)}\n")
        f.write("(Model predicted NOT YEW, but actually YEW)\n")
        f.write("-" * 80 + "\n")
        for i, result in enumerate(fn_results, 1):
            metadata = result['metadata']
            f.write(f"\n{i}. Confidence: {result['confidence']*100:.1f}%\n")
            f.write(f"   P(not yew) = {result['prob_not_yew']:.3f}\n")
            f.write(f"   Image: {metadata.get('image_path', 'unknown')}\n")
            if metadata.get('latitude') is not None and metadata.get('longitude') is not None:
                f.write(
                    f"   Location: {metadata['latitude']:.4f}, {metadata['longitude']:.4f}\n")
            if metadata.get('observation_id') is not None:
                f.write(f"   iNaturalist ID: {metadata['observation_id']}\n")
                f.write(
                    f"   URL: https://www.inaturalist.org/observations/{metadata['observation_id']}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("\nKEY INSIGHTS:\n")
        f.write("-" * 80 + "\n")
        f.write("• Review false positives for common visual features\n")
        f.write("• Check if false negatives have poor image quality\n")
        f.write("• Look for geographic patterns in misclassifications\n")
        f.write("• Consider excluding problematic samples from training\n")
        f.write("\n")

    print(f"  Saved summary: {report_path}")

    # Save as JSON for programmatic access
    json_path = output_dir / 'misclassifications.json'
    with open(json_path, 'w') as f:
        json.dump({
            'false_positives': fp_results,
            'false_negatives': fn_results
        }, f, indent=2)
    print(f"  Saved JSON: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize model misclassifications',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze validation set
    python scripts/visualization/visualize_misclassifications.py \\
        --model models/checkpoints/resnet18_20251114_125300_best.pth \\
        --data-split val \\
        --top-k 20
    
    # Analyze training set
    python scripts/visualization/visualize_misclassifications.py \\
        --model models/checkpoints/resnet18_20251114_125300_best.pth \\
        --data-split train \\
        --top-k 30
        """
    )

    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--data-split', type=str, default='val',
                        choices=['train', 'val'],
                        help='Data split to analyze (default: val)')
    parser.add_argument('--use-filtered', action='store_true',
                        help='Use filtered dataset splits')
    parser.add_argument('--top-k', type=int, default=20,
                        help='Number of top misclassifications to show (default: 20)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for inference (default: 16)')
    parser.add_argument('--output-dir', type=str,
                        default='results/misclassifications',
                        help='Output directory for visualizations')
    parser.add_argument('--image-base-dir', type=str,
                        default='data/ee_imagery/image_patches_64x64',
                        help='Base directory for image patches')

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = load_model(args.model, device)

    # Load data
    print(f"\nLoading {args.data_split} dataset...")

    # Use filtered splits if requested
    if args.use_filtered:
        split_file = f'data/processed/{args.data_split}_split_filtered.csv'
    else:
        split_file = f'data/processed/{args.data_split}_split.csv'

    if not Path(split_file).exists():
        print(f"✗ Split file not found: {split_file}")
        print("Please run training first to generate splits, or use correct --use-filtered flag")
        return

    # Load split dataframe
    split_df = pd.read_csv(split_file)
    print(f"  Dataset size: {len(split_df)} samples")

    # Create dataset and dataloader
    from training.dataset import SatelliteImageDataset
    image_base_dir = Path(args.image_base_dir)
    dataset = SatelliteImageDataset(
        metadata_files=split_file,
        image_base_dir=image_base_dir,
        augment=False  # Never augment for evaluation
    )

    # Custom collate function to handle metadata dictionaries
    def custom_collate(batch):
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        metadata = [item[2] for item in batch]  # Keep as list of dicts
        return images, labels, metadata

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Keep order for analysis
        num_workers=4,
        collate_fn=custom_collate
    )

    # Get predictions
    print("\nRunning inference...")
    predictions, probabilities, labels, metadata = get_predictions(
        model, dataloader, device
    )

    # Calculate accuracy
    accuracy = (predictions == labels).mean()
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Correct: {(predictions == labels).sum()}/{len(labels)}")
    print(f"  Misclassified: {(predictions != labels).sum()}")

    # Find misclassifications
    print("\nFinding top misclassifications...")
    fp_results, fn_results = find_misclassifications(
        predictions, probabilities, labels, metadata, top_k=args.top_k
    )

    print(f"  False Positives: {len(fp_results)}")
    print(f"  False Negatives: {len(fn_results)}")

    # Visualize
    print("\nCreating visualizations...")
    visualize_misclassifications(
        fp_results, fn_results, args.output_dir, image_base_dir)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {args.output_dir}")
    print("\nNext steps:")
    print("  1. Review false_positives.png - What do non-yew sites look like?")
    print("  2. Review false_negatives.png - Why did model miss these yew sites?")
    print("  3. Check misclassification_summary.txt for detailed info")
    print("  4. Consider excluding problematic samples from training")
    print("=" * 80)


if __name__ == '__main__':
    main()
