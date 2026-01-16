#!/usr/bin/env python3
"""
Extract Google Satellite Embedding patches for training/prediction.

The Google Satellite Embedding dataset provides 64-dimensional learned representations
of satellite imagery that encode temporal trajectories of surface conditions.

Usage:
    python scripts/preprocessing/extract_embedding_patches.py \
        --metadata data/processed/train_split_filtered.csv \
        --output data/ee_imagery/embedding_patches_64x64 \
        --year 2024 \
        --patch-size 64

Author: GitHub Copilot
Date: November 20, 2025
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


def initialize_earth_engine():
    """Initialize Earth Engine."""
    try:
        ee.Initialize(project='carbon-storm-206002')
        print("✓ Earth Engine initialized")
    except Exception as e:
        print(f"✗ Earth Engine initialization failed: {e}")
        print("  Run: earthengine authenticate")
        sys.exit(1)


def extract_embedding_patch(lat, lon, year, patch_size=64, scale=10):
    """
    Extract a Google Satellite Embedding patch centered on coordinates.

    Args:
        lat, lon: Center coordinates
        year: Calendar year for embeddings (e.g., 2024)
        patch_size: Size of patch in pixels
        scale: Pixel resolution in meters (10m for embeddings)

    Returns:
        numpy array of shape (64, patch_size, patch_size) or None
    """
    try:
        # Create point and region
        point = ee.Geometry.Point([lon, lat])

        # Calculate region bounds (centered on point)
        buffer_meters = (patch_size * scale) / 2
        region = point.buffer(buffer_meters).bounds()

        # Load Google Satellite Embedding collection
        dataset = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')

        # Filter by year and location
        start_date = f'{year}-01-01'
        end_date = f'{year + 1}-01-01'

        image = (dataset
                 .filterDate(start_date, end_date)
                 .filterBounds(point)
                 .first())

        if image is None:
            return None

        # Select all 64 embedding bands (A00 through A63)
        bands = [f'A{i:02d}' for i in range(64)]
        embedding = image.select(bands)

        # Get as numpy array
        url = embedding.getDownloadURL({
            'region': region,
            'dimensions': [patch_size, patch_size],
            'format': 'NPY'
        })

        # Download and load
        response = requests.get(url, timeout=60)
        if response.status_code == 200:
            # Save temporarily and load
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name

            data = np.load(tmp_path)
            Path(tmp_path).unlink()  # Clean up

            # Handle structured array from Earth Engine
            if data.dtype.names is not None:
                # Structured array with named fields (A00, A01, ..., A63)
                data_list = [data[band] for band in bands]
                data = np.stack(data_list, axis=0)  # (64, H, W)
            elif data.ndim == 3:
                # Regular array - transpose to (channels, height, width)
                data = np.transpose(data, (2, 0, 1))

            # Ensure correct shape
            if data.shape != (64, patch_size, patch_size):
                print(
                    f"Warning: Unexpected shape {data.shape} for {lat}, {lon}")
                return None

            return data

        return None

    except Exception as e:
        print(f"Error extracting embedding at {lat}, {lon}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Extract Google Satellite Embedding patches',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--metadata', type=str, required=True,
                        help='CSV file with latitude, longitude, and label columns')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for embedding patches')
    parser.add_argument('--year', type=int, default=2024,
                        help='Year for embeddings (default: 2024)')
    parser.add_argument('--patch-size', type=int, default=64,
                        help='Patch size in pixels (default: 64)')
    parser.add_argument('--scale', type=int, default=10,
                        help='Pixel resolution in meters (default: 10)')
    parser.add_argument('--resume', action='store_true',
                        help='Skip already extracted patches')

    args = parser.parse_args()

    # Initialize Earth Engine
    initialize_earth_engine()

    # Load metadata
    print(f"\nLoading metadata from: {args.metadata}")
    df = pd.read_csv(args.metadata)
    print(f"  Total records: {len(df)}")

    # Filter to only records with valid coordinates
    # Handle both 'lat'/'lon' and 'latitude'/'longitude' column names
    lat_col = 'lat' if 'lat' in df.columns else 'latitude'
    lon_col = 'lon' if 'lon' in df.columns else 'longitude'

    df = df[~df[lat_col].isna() & ~df[lon_col].isna()].copy()
    print(f"  Records with valid coordinates: {len(df)}")

    if len(df) == 0:
        print("  ERROR: No records with valid coordinates found!")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract patches
    print(
        f"\nExtracting {args.patch_size}x{args.patch_size} embedding patches for year {args.year}...")
    print(f"Output: {output_dir}")

    successful = 0
    failed = 0
    skipped = 0

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Handle both 'lat'/'lon' and 'latitude'/'longitude' column names
        lat = row.get('lat', row.get('latitude'))
        lon = row.get('lon', row.get('longitude'))

        # Skip if coordinates are NaN
        if pd.isna(lat) or pd.isna(lon):
            failed += 1
            continue

        # Generate filename
        filename = f'embedding_{lat:.6f}_{lon:.6f}.npy'
        output_path = output_dir / filename

        # Skip if already exists and resume flag is set
        if args.resume and output_path.exists():
            skipped += 1
            continue

        # Extract patch
        patch = extract_embedding_patch(
            lat, lon, args.year, args.patch_size, args.scale)

        if patch is not None:
            np.save(output_path, patch)
            successful += 1
        else:
            failed += 1

    # Summary
    print(f"\n{'='*80}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}")

    # Create metadata file
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'source_metadata': args.metadata,
        'year': args.year,
        'patch_size': args.patch_size,
        'scale': args.scale,
        'total_locations': len(df),
        'successful_extractions': successful,
        'failed_extractions': failed,
        'output_directory': str(output_dir)
    }

    import json
    metadata_path = output_dir / 'extraction_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to: {metadata_path}")


if __name__ == '__main__':
    main()
