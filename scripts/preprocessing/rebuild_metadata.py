#!/usr/bin/env python3
"""
Rebuild Metadata from Existing Images
======================================

Regenerate the metadata CSV from existing .npy files and original iNaturalist data.

Author: GitHub Copilot
Date: November 14, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from datetime import datetime


def rebuild_metadata():
    """Rebuild metadata from existing image files."""

    print("="*80)
    print("REBUILDING METADATA")
    print("="*80)

    # Load iNaturalist observations
    inat_df = pd.read_csv('data/inat_observations/observations-558049.csv')
    print(f"\nLoaded {len(inat_df)} iNaturalist observations")

    # Find all extracted images
    image_dir = Path('data/ee_imagery/image_patches_64x64/inat_yew')
    npy_files = sorted(image_dir.glob('*.npy'))
    print(f"Found {len(npy_files)} extracted images")

    if len(npy_files) == 0:
        print("✗ No images found!")
        return

    # Extract observation IDs from filenames
    metadata_records = []

    for npy_file in npy_files:
        # Extract obs_id from filename (e.g., inat_12345.npy)
        match = re.search(r'inat_(\d+)\.npy', npy_file.name)
        if not match:
            print(f"Warning: Could not parse {npy_file.name}")
            continue

        obs_id = int(match.group(1))

        # Find corresponding iNat observation
        inat_row = inat_df[inat_df['id'] == obs_id]

        if len(inat_row) == 0:
            print(f"Warning: No iNat data for observation {obs_id}")
            continue

        obs = inat_row.iloc[0]

        # Load image to get shape
        try:
            img = np.load(npy_file)
            image_shape = str(img.shape)
        except:
            print(f"Warning: Could not load {npy_file.name}")
            continue

        # Get observation year
        obs_date = pd.to_datetime(obs['observed_on'])
        obs_year = obs_date.year if pd.notna(obs_date) else None

        # Create metadata record
        record = {
            'observation_id': obs_id,
            'has_yew': True,
            'source': 'iNaturalist',
            'observation_year': obs_year,
            'positional_accuracy': obs.get('positional_accuracy'),
            'lon': obs['longitude'],
            'lat': obs['latitude'],
            'image_path': f'inat_yew/{npy_file.name}',
            'image_shape': image_shape,
            'num_source_images': None,  # Unknown for rebuilt metadata
            'extraction_date': datetime.now().isoformat()
        }

        metadata_records.append(record)

    # Create DataFrame
    metadata_df = pd.DataFrame(metadata_records)

    # Sort by observation_id
    metadata_df = metadata_df.sort_values(
        'observation_id').reset_index(drop=True)

    # Save
    output_file = Path(
        'data/ee_imagery/image_patches_64x64/inat_yew_image_metadata.csv')
    metadata_df.to_csv(output_file, index=False)

    print(f"\n✓ Rebuilt metadata for {len(metadata_df)} images")
    print(f"✓ Saved to: {output_file}")

    # Summary
    print(f"\nMetadata summary:")
    print(
        f"  Observation years: {metadata_df['observation_year'].min():.0f} - {metadata_df['observation_year'].max():.0f}")
    print(
        f"  Median GPS accuracy: {metadata_df['positional_accuracy'].median():.1f}m")
    print(
        f"  Mean GPS accuracy: {metadata_df['positional_accuracy'].mean():.1f}m")


if __name__ == '__main__':
    rebuild_metadata()
