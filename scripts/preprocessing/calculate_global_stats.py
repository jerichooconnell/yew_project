#!/usr/bin/env python3
"""
Calculate Global Normalization Statistics
==========================================

Calculate mean and std for all 4 channels across the entire training dataset
to preserve relative reflectance values during normalization.

Author: GitHub Copilot
Date: November 20, 2025
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def calculate_global_statistics(metadata_csv, image_base_dir, output_path):
    """
    Calculate global mean and std for each channel across all images.

    Args:
        metadata_csv: Path to metadata CSV file
        image_base_dir: Base directory containing images
        output_path: Path to save statistics JSON
    """
    print("Loading metadata...")
    df = pd.read_csv(metadata_csv)
    image_base_dir = Path(image_base_dir)

    print(f"Found {len(df)} images")

    # Collect all pixel values for each channel
    print("Loading all images...")
    all_values = [[] for _ in range(4)]  # 4 channels

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = image_base_dir / row['image_path']

        if img_path.exists():
            try:
                img = np.load(img_path)  # Shape: (4, 64, 64)

                for channel in range(4):
                    channel_data = img[channel, :, :].flatten()
                    all_values[channel].extend(channel_data.tolist())

            except Exception as e:
                print(f"Error loading {img_path}: {e}")

    # Calculate statistics
    print("\nCalculating statistics...")
    stats = {
        'channels': ['B2', 'B3', 'B4', 'B8'],
        'mean': [],
        'std': [],
        'percentile_2': [],
        'percentile_98': [],
        'min': [],
        'max': [],
        'num_images': len(df),
        'num_pixels_per_channel': len(all_values[0])
    }

    for channel in range(4):
        values = np.array(all_values[channel])

        mean = float(np.mean(values))
        std = float(np.std(values))
        p2 = float(np.percentile(values, 2))
        p98 = float(np.percentile(values, 98))
        min_val = float(np.min(values))
        max_val = float(np.max(values))

        stats['mean'].append(mean)
        stats['std'].append(std)
        stats['percentile_2'].append(p2)
        stats['percentile_98'].append(p98)
        stats['min'].append(min_val)
        stats['max'].append(max_val)

        print(f"\nChannel {channel} ({stats['channels'][channel]}):")
        print(f"  Mean: {mean:.2f}")
        print(f"  Std:  {std:.2f}")
        print(f"  Min:  {min_val:.2f}")
        print(f"  Max:  {max_val:.2f}")
        print(f"  2nd percentile:  {p2:.2f}")
        print(f"  98th percentile: {p98:.2f}")

    # Save statistics
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n✓ Statistics saved to: {output_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Calculate global normalization statistics')
    parser.add_argument('--train-csv', required=True,
                        help='Path to training metadata CSV')
    parser.add_argument('--image-dir', default='data/ee_imagery/image_patches_64x64',
                        help='Base directory for images')
    parser.add_argument('--output', default='data/processed/global_normalization_stats.json',
                        help='Output path for statistics JSON')

    args = parser.parse_args()

    calculate_global_statistics(args.train_csv, args.image_dir, args.output)


if __name__ == '__main__':
    main()
