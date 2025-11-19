#!/usr/bin/env python3
"""
Dataset Status Check
====================

Quick status check for filtered dataset preparation.
"""

from pathlib import Path
import pandas as pd


def check_status():
    print("\n" + "="*80)
    print("DATASET STATUS CHECK")
    print("="*80)

    # Check filtered yew data
    yew_file = Path('data/processed/inat_yew_filtered_good.csv')
    if yew_file.exists():
        yew_df = pd.read_csv(yew_file)
        print(f"\n✓ Filtered Yew Dataset: {len(yew_df)} samples")
        print(f"  File: {yew_file}")
        print(
            f"  Lat range: {yew_df['lat'].min():.2f}°N - {yew_df['lat'].max():.2f}°N")
        print(
            f"  GPS accuracy (median): {yew_df['positional_accuracy'].median():.1f}m")
    else:
        print(f"\n✗ Filtered yew data not found: {yew_file}")

    # Check non-yew data
    no_yew_file = Path(
        'data/ee_imagery/image_patches_64x64/no_yew_image_metadata.csv')
    no_yew_dir = Path('data/ee_imagery/image_patches_64x64/no_yew')

    if no_yew_file.exists():
        no_yew_df = pd.read_csv(no_yew_file)
        n_images = len(list(no_yew_dir.glob('*.npy'))
                       ) if no_yew_dir.exists() else 0

        print(f"\n✓ Non-Yew Dataset:")
        print(f"  Metadata entries: {len(no_yew_df)}")
        print(f"  Image files (.npy): {n_images}")

        if n_images < len(no_yew_df):
            print(f"  ⚠ Warning: {len(no_yew_df) - n_images} images missing")
        elif n_images > len(no_yew_df):
            print(
                f"  ⚠ Warning: {n_images - len(no_yew_df)} images not in metadata")
        else:
            print(f"  ✓ Metadata and images match!")
    else:
        print(f"\n✗ Non-yew data not found: {no_yew_file}")

    # Check training splits
    train_file = Path('data/processed/train_split_filtered.csv')
    val_file = Path('data/processed/val_split_filtered.csv')

    if train_file.exists() and val_file.exists():
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)

        train_yew = train_df['has_yew'].sum()
        train_no_yew = (~train_df['has_yew']).sum()
        val_yew = val_df['has_yew'].sum()
        val_no_yew = (~val_df['has_yew']).sum()

        print(f"\n✓ Training Splits Created:")
        print(
            f"  Training: {len(train_df)} samples ({train_yew} yew, {train_no_yew} non-yew)")
        print(
            f"  Validation: {len(val_df)} samples ({val_yew} yew, {val_no_yew} non-yew)")
        print(
            f"  Class balance: {train_yew/(train_yew+train_no_yew)*100:.1f}% yew in training")
    else:
        print(f"\n✗ Training splits not created yet")
        print(f"  Run: python scripts/preprocessing/create_filtered_splits.py")

    # Summary
    print(f"\n" + "="*80)
    if yew_file.exists() and no_yew_file.exists():
        total = len(yew_df) + len(no_yew_df)
        print(
            f"TOTAL DATASET: {total} samples ({len(yew_df)} yew + {len(no_yew_df)} non-yew)")

        if train_file.exists():
            print(f"STATUS: ✓ Ready for training with --use-filtered flag")
            print(f"\nNext step:")
            print(f"  python scripts/training/train_cnn.py --use-filtered --epochs 50")
        else:
            print(f"STATUS: ⏳ Need to create training splits")
            print(f"\nNext step:")
            print(f"  python scripts/preprocessing/create_filtered_splits.py")
    else:
        print(f"STATUS: ⏳ Waiting for data extraction/filtering to complete")

    print("="*80 + "\n")


if __name__ == '__main__':
    check_status()
