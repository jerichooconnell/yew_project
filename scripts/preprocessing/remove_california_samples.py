#!/usr/bin/env python3
"""
Remove California Samples from Yew Dataset
===========================================

Filter out California yew samples (lat < 42°N) from the existing dataset.

Author: GitHub Copilot
Date: November 14, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import shutil


def remove_california_samples():
    """Remove California samples from yew dataset."""
    print("\n" + "="*80)
    print("REMOVING CALIFORNIA SAMPLES FROM YEW DATASET")
    print("="*80)

    # Load metadata
    metadata_file = Path(
        'data/ee_imagery/image_patches_64x64/inat_yew_image_metadata.csv')

    if not metadata_file.exists():
        print(f"\n✗ Metadata file not found: {metadata_file}")
        return

    df = pd.read_csv(metadata_file)
    print(f"\nOriginal dataset: {len(df)} samples")

    # Separate CA and BC/WA samples
    ca_samples = df[df['lat'] < 42.0].copy()
    bc_wa_samples = df[df['lat'] >= 42.0].copy()

    print(f"California samples (lat < 42°N): {len(ca_samples)}")
    print(f"BC/WA samples (lat >= 42°N): {len(bc_wa_samples)}")

    if len(ca_samples) == 0:
        print("\n✓ No California samples found - dataset already filtered!")
        return

    # Show CA sample details
    print(f"\nCalifornia samples to remove:")
    print(
        f"  Latitude range: {ca_samples['lat'].min():.2f}°N - {ca_samples['lat'].max():.2f}°N")
    print(f"  Observation IDs: {ca_samples['observation_id'].tolist()}")

    # Backup original metadata
    backup_file = metadata_file.parent / 'inat_yew_image_metadata_backup.csv'
    if not backup_file.exists():
        shutil.copy(metadata_file, backup_file)
        print(f"\n✓ Backed up original metadata to: {backup_file}")

    # Archive CA image files
    archive_dir = Path(
        'data/ee_imagery/image_patches_64x64/inat_yew_california_removed')
    archive_dir.mkdir(parents=True, exist_ok=True)

    moved_count = 0
    for idx, row in ca_samples.iterrows():
        img_path = Path('data/ee_imagery/image_patches_64x64') / \
            row['image_path']
        if img_path.exists():
            archive_path = archive_dir / img_path.name
            shutil.move(str(img_path), str(archive_path))
            moved_count += 1

    print(f"\n✓ Moved {moved_count} California image files to: {archive_dir}")

    # Save filtered metadata
    bc_wa_samples.to_csv(metadata_file, index=False)
    print(f"✓ Saved filtered metadata: {len(bc_wa_samples)} samples")

    # Summary
    print(f"\n" + "="*80)
    print("FILTERING COMPLETE")
    print("="*80)
    print(f"Removed: {len(ca_samples)} California samples")
    print(f"Remaining: {len(bc_wa_samples)} BC/WA samples")
    print(
        f"Geographic range: {bc_wa_samples['lat'].min():.2f}°N - {bc_wa_samples['lat'].max():.2f}°N")
    print("="*80)

    return bc_wa_samples


if __name__ == '__main__':
    remove_california_samples()
