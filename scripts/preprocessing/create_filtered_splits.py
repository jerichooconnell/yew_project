#!/usr/bin/env python3
"""
Create Filtered Training Splits
================================

Create train/val splits using only filtered, high-quality samples:
- Yew: Only iNaturalist samples from BC/WA (no California) marked as "good"
- Non-yew: All forestry inventory samples from CWH/ICH zones

Author: GitHub Copilot
Date: November 14, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path


def create_filtered_splits():
    """Create filtered train/val splits."""
    print("\n" + "="*80)
    print("CREATING FILTERED TRAINING SPLITS")
    print("="*80)

    # Load filtered datasets
    yew_file = Path('data/processed/inat_yew_filtered_good.csv')
    no_yew_file = Path(
        'data/ee_imagery/image_patches_64x64/no_yew_image_metadata.csv')

    if not yew_file.exists():
        print(f"\n✗ Error: {yew_file} not found!")
        print("Run: python scripts/preprocessing/filter_and_extract_dataset.py first")
        return

    if not no_yew_file.exists():
        print(f"\n✗ Error: {no_yew_file} not found!")
        return

    yew_df = pd.read_csv(yew_file)
    no_yew_df = pd.read_csv(no_yew_file)

    print(f"\nDataset sizes:")
    print(f"  Yew (BC/WA, filtered 'good'): {len(yew_df)}")
    print(f"  Non-yew (CWH/ICH zones): {len(no_yew_df)}")
    print(f"  Total: {len(yew_df) + len(no_yew_df)}")
    print(
        f"  Class balance: {len(yew_df)/(len(yew_df)+len(no_yew_df))*100:.1f}% yew")

    # Ensure has_yew column is set
    yew_df['has_yew'] = True
    no_yew_df['has_yew'] = False

    # Shuffle and split with stratification
    val_split = 0.2
    seed = 42
    np.random.seed(seed)

    # Split each class
    yew_indices = np.random.permutation(len(yew_df))
    no_yew_indices = np.random.permutation(len(no_yew_df))

    yew_val_size = int(len(yew_df) * val_split)
    no_yew_val_size = int(len(no_yew_df) * val_split)

    yew_train_idx = yew_indices[yew_val_size:]
    yew_val_idx = yew_indices[:yew_val_size]
    no_yew_train_idx = no_yew_indices[no_yew_val_size:]
    no_yew_val_idx = no_yew_indices[:no_yew_val_size]

    # Create splits
    train_df = pd.concat([
        yew_df.iloc[yew_train_idx],
        no_yew_df.iloc[no_yew_train_idx]
    ], ignore_index=True)

    val_df = pd.concat([
        yew_df.iloc[yew_val_idx],
        no_yew_df.iloc[no_yew_val_idx]
    ], ignore_index=True)

    # Shuffle the combined datasets
    train_df = train_df.sample(
        frac=1, random_state=seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Save
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / 'train_split_filtered.csv'
    val_path = output_dir / 'val_split_filtered.csv'

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"\n✓ Training split: {len(train_df)} samples")
    print(f"    Yew: {len(yew_train_idx)} | Non-yew: {len(no_yew_train_idx)}")
    print(f"    Balance: {len(yew_train_idx)/len(train_df)*100:.1f}% yew")

    print(f"\n✓ Validation split: {len(val_df)} samples")
    print(f"    Yew: {len(yew_val_idx)} | Non-yew: {len(no_yew_val_idx)}")
    print(f"    Balance: {len(yew_val_idx)/len(val_df)*100:.1f}% yew")

    print(f"\n✓ Saved filtered splits:")
    print(f"    Training: {train_path}")
    print(f"    Validation: {val_path}")

    print(f"\n" + "="*80)
    print("DATASET READY FOR TRAINING")
    print("="*80)
    print("\nTo train with filtered data, run:")
    print("  python scripts/training/train_cnn.py --use-filtered --epochs 50")
    print("="*80)


if __name__ == '__main__':
    create_filtered_splits()
