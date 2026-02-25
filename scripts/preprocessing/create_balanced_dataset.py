#!/usr/bin/env python3
"""
Create a balanced dataset with 5:1 non-yew to yew ratio.

This script:
1. Loads existing yew samples from train/val splits
2. Randomly samples non-yew locations from BC sample data
3. Downloads embeddings for new non-yew samples
4. Creates new train/val splits with 5:1 ratio

Usage:
    python scripts/preprocessing/create_balanced_dataset.py --ratio 5
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import ee
import numpy as np
import pandas as pd
from pyproj import Transformer
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Create balanced yew dataset')
    parser.add_argument('--ratio', type=int, default=5,
                        help='Ratio of non-yew to yew samples (default: 5)')
    parser.add_argument('--train-csv', type=str,
                        default='data/processed/train_split_filtered.csv',
                        help='Path to existing training CSV')
    parser.add_argument('--val-csv', type=str,
                        default='data/processed/val_split_filtered.csv',
                        help='Path to existing validation CSV')
    parser.add_argument('--bc-sample-csv', type=str,
                        default='data/processed/bc_sample_data_deduplicated.csv',
                        help='Path to BC sample data CSV')
    parser.add_argument('--output-dir', type=str,
                        default='data/processed',
                        help='Output directory for new CSV files')
    parser.add_argument('--embedding-dir', type=str,
                        default='data/ee_imagery/embedding_patches_64x64',
                        help='Directory for embedding patches')
    parser.add_argument('--year', type=int, default=2024,
                        help='Year for embeddings')
    parser.add_argument('--gee-project', type=str,
                        default='carbon-storm-206002',
                        help='GEE project ID')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip downloading new embeddings (use existing)')
    return parser.parse_args()


def init_earth_engine(project_id):
    """Initialize Earth Engine."""
    print("Initializing Earth Engine...")
    try:
        ee.Initialize(project=project_id)
        print("✓ Earth Engine initialized")
    except Exception as e:
        print(f"Initializing with authentication...")
        ee.Authenticate()
        ee.Initialize(project=project_id)
        print("✓ Earth Engine initialized after authentication")


def get_embedding_image(year):
    """Get the Prithvi embedding image for a given year."""
    # Use summer months for best imagery
    start_date = f'{year}-06-01'
    end_date = f'{year}-09-30'
    
    embedding = ee.ImageCollection('projects/sat-io/open-datasets/PRITHVI/PRITHVI-EO-V2') \
        .filterDate(start_date, end_date) \
        .mean()
    
    return embedding


def download_embedding_patch(lat, lon, embedding_image, output_path, patch_size=64, scale=10):
    """Download a single embedding patch."""
    if output_path.exists():
        return True
    
    try:
        # Create a region around the point
        point = ee.Geometry.Point([lon, lat])
        half_size = (patch_size * scale) / 2
        region = point.buffer(half_size).bounds()
        
        # Download
        url = embedding_image.getDownloadURL({
            'region': region,
            'scale': scale,
            'format': 'NPY'
        })
        
        import requests
        response = requests.get(url, timeout=60)
        
        if response.status_code == 200:
            data = np.load(io.BytesIO(response.content))
            
            # Ensure correct shape
            if data.shape[0] == 64:  # bands first
                data = np.transpose(data, (1, 2, 0))
            
            # Save
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, data)
            return True
        else:
            return False
            
    except Exception as e:
        print(f"  Error downloading {lat}, {lon}: {e}")
        return False


def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    embedding_dir = Path(args.embedding_dir)
    
    # Load existing data
    print("\n" + "="*60)
    print("Loading existing data...")
    print("="*60)
    
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    
    # Get yew samples
    train_yew = train_df[train_df['has_yew'] == True].copy()
    val_yew = val_df[val_df['has_yew'] == True].copy()
    
    total_yew = len(train_yew) + len(val_yew)
    print(f"  Yew samples: {total_yew} (train: {len(train_yew)}, val: {len(val_yew)})")
    
    # Calculate needed non-yew samples
    needed_non_yew = args.ratio * total_yew
    print(f"  Need {needed_non_yew} non-yew samples ({args.ratio}:1 ratio)")
    
    # Load BC sample data
    print("\nLoading BC sample data...")
    bc_df = pd.read_csv(args.bc_sample_csv)
    print(f"  Total BC samples: {len(bc_df)}")
    
    # Convert BC Albers to lat/lon
    print("  Converting coordinates...")
    transformer = Transformer.from_crs('EPSG:3005', 'EPSG:4326', always_xy=True)
    lons, lats = transformer.transform(bc_df['BC_ALBERS_X'].values, bc_df['BC_ALBERS_Y'].values)
    bc_df['lat'] = lats
    bc_df['lon'] = lons
    
    # Filter to reasonable lat range (similar to yew samples)
    # Yew typically found in southern BC / Pacific Northwest
    yew_lat_range = (train_yew['lat'].min(), train_yew['lat'].max())
    yew_lon_range = (train_yew['lon'].min(), train_yew['lon'].max())
    print(f"  Yew lat range: {yew_lat_range[0]:.2f} to {yew_lat_range[1]:.2f}")
    print(f"  Yew lon range: {yew_lon_range[0]:.2f} to {yew_lon_range[1]:.2f}")
    
    # Filter BC samples to be within similar geographic range (with buffer)
    lat_buffer = 2.0
    lon_buffer = 2.0
    bc_filtered = bc_df[
        (bc_df['lat'] >= yew_lat_range[0] - lat_buffer) &
        (bc_df['lat'] <= yew_lat_range[1] + lat_buffer) &
        (bc_df['lon'] >= yew_lon_range[0] - lon_buffer) &
        (bc_df['lon'] <= yew_lon_range[1] + lon_buffer)
    ].copy()
    print(f"  BC samples in similar range: {len(bc_filtered)}")
    
    if len(bc_filtered) < needed_non_yew:
        print(f"  Warning: Not enough BC samples in range, using all available")
        bc_filtered = bc_df.copy()
    
    # Random sample
    print(f"\nRandomly sampling {needed_non_yew} non-yew locations...")
    np.random.seed(42)
    sampled_bc = bc_filtered.sample(n=min(needed_non_yew, len(bc_filtered)), random_state=42)
    
    # Create non-yew dataframe with required columns
    non_yew_df = pd.DataFrame({
        'observation_id': sampled_bc['SITE_IDENTIFIER'].values,
        'has_yew': False,
        'source': 'BC_VRI',
        'observation_year': sampled_bc['MEAS_YR'].values,
        'positional_accuracy': 10.0,
        'lon': sampled_bc['lon'].values,
        'lat': sampled_bc['lat'].values,
        'image_path': '',
        'image_shape': '',
        'num_source_images': '',
        'extraction_date': datetime.now().isoformat(),
        'site_identifier': sampled_bc['SITE_IDENTIFIER'].values,
    })
    
    print(f"  Created {len(non_yew_df)} non-yew samples")
    
    # Initialize Earth Engine and download embeddings
    if not args.skip_download:
        print("\n" + "="*60)
        print("Downloading embeddings for new non-yew samples...")
        print("="*60)
        
        init_earth_engine(args.gee_project)
        embedding_image = get_embedding_image(args.year)
        
        import io
        import requests
        
        successful = 0
        failed = 0
        
        for idx, row in tqdm(non_yew_df.iterrows(), total=len(non_yew_df), desc="Downloading"):
            lat, lon = row['lat'], row['lon']
            output_path = embedding_dir / f'embedding_{lat:.6f}_{lon:.6f}.npy'
            
            if output_path.exists():
                successful += 1
                non_yew_df.loc[idx, 'image_path'] = str(output_path.relative_to(embedding_dir.parent.parent))
                continue
            
            try:
                point = ee.Geometry.Point([lon, lat])
                half_size = (64 * 10) / 2
                region = point.buffer(half_size).bounds()
                
                url = embedding_image.getDownloadURL({
                    'region': region,
                    'scale': 10,
                    'format': 'NPY'
                })
                
                response = requests.get(url, timeout=60)
                
                if response.status_code == 200:
                    data = np.load(io.BytesIO(response.content))
                    
                    if data.shape[0] == 64:
                        data = np.transpose(data, (1, 2, 0))
                    
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(output_path, data)
                    
                    non_yew_df.loc[idx, 'image_path'] = str(output_path.relative_to(embedding_dir.parent.parent))
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                failed += 1
                if failed < 5:
                    print(f"  Error: {e}")
            
            # Rate limiting
            time.sleep(0.1)
        
        print(f"\n  Downloaded: {successful}, Failed: {failed}")
        
        # Filter to only successful downloads
        non_yew_df = non_yew_df[non_yew_df['image_path'] != ''].copy()
        print(f"  Final non-yew samples with embeddings: {len(non_yew_df)}")
    
    # Check which non-yew samples have existing embeddings
    print("\nChecking for existing embeddings...")
    valid_non_yew = []
    for idx, row in non_yew_df.iterrows():
        emb_path = embedding_dir / f'embedding_{row["lat"]:.6f}_{row["lon"]:.6f}.npy'
        if emb_path.exists():
            non_yew_df.loc[idx, 'image_path'] = f'ee_imagery/embedding_patches_64x64/embedding_{row["lat"]:.6f}_{row["lon"]:.6f}.npy'
            valid_non_yew.append(idx)
    
    non_yew_df = non_yew_df.loc[valid_non_yew].copy()
    print(f"  Non-yew samples with valid embeddings: {len(non_yew_df)}")
    
    # Split non-yew into train/val (80/20)
    print("\n" + "="*60)
    print("Creating new train/val splits...")
    print("="*60)
    
    n_train_non_yew = int(len(non_yew_df) * 0.8)
    non_yew_shuffled = non_yew_df.sample(frac=1, random_state=42)
    train_non_yew = non_yew_shuffled.iloc[:n_train_non_yew]
    val_non_yew = non_yew_shuffled.iloc[n_train_non_yew:]
    
    # Combine with yew samples
    new_train_df = pd.concat([train_yew, train_non_yew], ignore_index=True)
    new_val_df = pd.concat([val_yew, val_non_yew], ignore_index=True)
    
    # Shuffle
    new_train_df = new_train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    new_val_df = new_val_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nNew dataset stats:")
    print(f"  Train: {len(new_train_df)} total")
    print(f"    Yew: {(new_train_df['has_yew'] == True).sum()}")
    print(f"    Non-yew: {(new_train_df['has_yew'] == False).sum()}")
    print(f"    Ratio: {(new_train_df['has_yew'] == False).sum() / (new_train_df['has_yew'] == True).sum():.1f}:1")
    
    print(f"\n  Val: {len(new_val_df)} total")
    print(f"    Yew: {(new_val_df['has_yew'] == True).sum()}")
    print(f"    Non-yew: {(new_val_df['has_yew'] == False).sum()}")
    print(f"    Ratio: {(new_val_df['has_yew'] == False).sum() / (new_val_df['has_yew'] == True).sum():.1f}:1")
    
    # Save new CSVs
    train_output = output_dir / f'train_split_balanced_{args.ratio}to1.csv'
    val_output = output_dir / f'val_split_balanced_{args.ratio}to1.csv'
    
    new_train_df.to_csv(train_output, index=False)
    new_val_df.to_csv(val_output, index=False)
    
    print(f"\n✓ Saved: {train_output}")
    print(f"✓ Saved: {val_output}")


if __name__ == '__main__':
    main()
