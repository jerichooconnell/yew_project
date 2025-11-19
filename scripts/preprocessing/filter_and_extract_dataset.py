#!/usr/bin/env python3
"""
Filter Dataset and Extract Additional Samples
==============================================

1. Filter out California samples from iNaturalist data (lat < 42°N)
2. Extract 300 additional non-yew samples
3. Apply manual review filtering (only 'good' samples)

Author: GitHub Copilot
Date: November 14, 2025
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import ee
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def filter_california_samples(metadata_csv, review_json):
    """
    Filter out California samples and apply manual review filtering.

    Args:
        metadata_csv: Path to inat_yew_image_metadata.csv
        review_json: Path to image_review_results.json

    Returns:
        Filtered DataFrame with only BC/WA samples marked as 'good'
    """
    print("\n" + "="*80)
    print("FILTERING INATURALIST YEW SAMPLES")
    print("="*80)

    # Load metadata
    df = pd.read_csv(metadata_csv)
    print(f"\nOriginal dataset: {len(df)} samples")

    # Filter by latitude (BC/WA only, exclude California)
    # California is approximately < 42°N
    df_filtered = df[df['lat'] >= 42.0].copy()
    print(
        f"After removing California (lat >= 42°N): {len(df_filtered)} samples")

    # Load manual reviews
    with open(review_json, 'r') as f:
        reviews = json.load(f)

    # Filter to only 'good' samples
    good_obs_ids = [int(obs_id)
                    for obs_id, status in reviews.items() if status == 'good']
    df_good = df_filtered[df_filtered['observation_id'].isin(
        good_obs_ids)].copy()

    print(
        f"After applying manual review (only 'good'): {len(df_good)} samples")

    # Save filtered dataset
    output_path = Path('data/processed/inat_yew_filtered_good.csv')
    df_good.to_csv(output_path, index=False)
    print(f"\n✓ Saved filtered yew dataset to: {output_path}")

    # Summary statistics
    print(f"\nFiltered Dataset Summary:")
    print(f"  Total samples: {len(df_good)}")
    print(
        f"  Latitude range: {df_good['lat'].min():.2f}°N - {df_good['lat'].max():.2f}°N")
    print(
        f"  Longitude range: {df_good['lon'].min():.2f}°E - {df_good['lon'].max():.2f}°E")
    print(
        f"  Median GPS accuracy: {df_good['positional_accuracy'].median():.1f}m")
    print(
        f"  Observation years: {df_good['observation_year'].min():.0f} - {df_good['observation_year'].max():.0f}")

    # Show California samples that were removed
    df_california = df[df['lat'] < 42.0]
    print(f"\nRemoved {len(df_california)} California samples:")
    print(
        f"  Latitude range: {df_california['lat'].min():.2f}°N - {df_california['lat'].max():.2f}°N")

    return df_good


def extract_additional_no_yew_samples(num_samples=300, workers=6):
    """
    Extract additional non-yew samples from forestry inventory.

    Args:
        num_samples: Number of additional samples to extract
        workers: Number of parallel workers
    """
    print("\n" + "="*80)
    print(f"EXTRACTING {num_samples} ADDITIONAL NON-YEW SAMPLES")
    print("="*80)

    # Initialize Earth Engine
    try:
        ee.Initialize()
    except:
        print("Authenticating with Earth Engine...")
        ee.Authenticate()
        ee.Initialize()

    # Load BC sample data
    bc_data = pd.read_csv('data/processed/bc_sample_data_deduplicated.csv')

    # Filter to sites without yew in CWH/ICH zones
    no_yew_sites = bc_data[
        (bc_data['SPECIES_CD_1'] != 'Y') &
        (bc_data['SPECIES_CD_2'] != 'Y') &
        (bc_data['SPECIES_CD_3'] != 'Y') &
        (bc_data['SPECIES_CD_4'] != 'Y') &
        (bc_data['SPECIES_CD_5'] != 'Y') &
        (bc_data['SPECIES_CD_6'] != 'Y') &
        (bc_data['BGC_ZONE'].isin(['CWH', 'ICH']))
    ].copy()

    print(f"\nAvailable non-yew sites: {len(no_yew_sites)}")

    # Check existing extractions
    existing_metadata = Path(
        'data/ee_imagery/image_patches_64x64/no_yew_image_metadata.csv')
    if existing_metadata.exists():
        existing_df = pd.read_csv(existing_metadata)
        existing_sample_ids = set(existing_df['SAMPLE_ID'].values)
        print(f"Existing non-yew samples: {len(existing_df)}")

        # Filter out already extracted samples
        no_yew_sites = no_yew_sites[~no_yew_sites['SAMPLE_ID'].isin(
            existing_sample_ids)]
        print(f"New sites available for extraction: {len(no_yew_sites)}")
    else:
        existing_df = None
        existing_sample_ids = set()

    # Randomly sample
    if len(no_yew_sites) < num_samples:
        print(
            f"Warning: Only {len(no_yew_sites)} sites available, extracting all")
        num_samples = len(no_yew_sites)

    sample_sites = no_yew_sites.sample(n=num_samples, random_state=42)

    # Setup Sentinel-2 collection
    def create_sentinel2_composite():
        """Create a cloud-free Sentinel-2 composite."""
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterDate('2020-01-01', '2024-12-31') \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

        def mask_clouds(image):
            qa = image.select('QA60')
            cloud_mask = qa.bitwiseAnd(1 << 10).eq(0).And(
                qa.bitwiseAnd(1 << 11).eq(0))
            return image.updateMask(cloud_mask)

        return s2.map(mask_clouds).median()

    composite = create_sentinel2_composite()

    # Extract function
    def extract_image(row):
        """Extract a single image."""
        sample_id = row['SAMPLE_ID']
        lat = row['LATITUDE']
        lon = row['LONGITUDE']

        try:
            # Define region
            point = ee.Geometry.Point([lon, lat])
            region = point.buffer(320).bounds()

            # Select bands and reproject
            image = composite.select(['B2', 'B3', 'B4', 'B8']) \
                .reproject('EPSG:4326', scale=10)

            # Sample
            sample = image.sampleRectangle(region, defaultValue=0)

            # Get arrays
            arrays = {
                'B2': np.array(sample.get('B2').getInfo()),
                'B3': np.array(sample.get('B3').getInfo()),
                'B4': np.array(sample.get('B4').getInfo()),
                'B8': np.array(sample.get('B8').getInfo())
            }

            # Stack into 4-channel image
            img = np.stack([arrays['B2'], arrays['B3'],
                           arrays['B4'], arrays['B8']], axis=0)

            # Resize to 64x64 if needed
            if img.shape[1] != 64 or img.shape[2] != 64:
                from scipy.ndimage import zoom
                factors = (1, 64/img.shape[1], 64/img.shape[2])
                img = zoom(img, factors, order=1)

            # Save
            output_dir = Path('data/ee_imagery/image_patches_64x64/no_yew')
            output_dir.mkdir(parents=True, exist_ok=True)

            output_path = output_dir / f'no_yew_{sample_id}.npy'
            np.save(output_path, img)

            # Metadata
            metadata = {
                'SAMPLE_ID': sample_id,
                'has_yew': False,
                'source': 'BC_Forestry',
                'BGC_ZONE': row['BGC_ZONE'],
                'BGC_SUBZON': row['BGC_SUBZON'],
                'lon': lon,
                'lat': lat,
                'image_path': f'no_yew/no_yew_{sample_id}.npy',
                'image_shape': str(img.shape),
                'num_source_images': None,
                'extraction_date': datetime.now().isoformat()
            }

            return metadata

        except Exception as e:
            print(f"Error extracting sample {sample_id}: {e}")
            return None

    # Extract in parallel
    print(f"\nExtracting {num_samples} samples with {workers} workers...")

    metadata_list = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(extract_image, row): idx
                   for idx, row in sample_sites.iterrows()}

        with tqdm(total=len(futures), desc="Extracting") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    metadata_list.append(result)
                pbar.update(1)
                time.sleep(0.1)  # Rate limiting

    # Create metadata DataFrame
    new_metadata_df = pd.DataFrame(metadata_list)

    # Combine with existing metadata
    if existing_df is not None:
        combined_df = pd.concat(
            [existing_df, new_metadata_df], ignore_index=True)
    else:
        combined_df = new_metadata_df

    # Save
    output_path = Path(
        'data/ee_imagery/image_patches_64x64/no_yew_image_metadata.csv')
    combined_df.to_csv(output_path, index=False)

    print(f"\n✓ Successfully extracted {len(metadata_list)} new samples")
    print(f"✓ Total non-yew samples: {len(combined_df)}")
    print(f"✓ Saved metadata to: {output_path}")

    return combined_df


def create_filtered_training_splits():
    """Create filtered train/val splits using only good samples."""
    print("\n" + "="*80)
    print("CREATING FILTERED TRAINING SPLITS")
    print("="*80)

    # Load filtered datasets
    yew_df = pd.read_csv('data/processed/inat_yew_filtered_good.csv')
    no_yew_df = pd.read_csv(
        'data/ee_imagery/image_patches_64x64/no_yew_image_metadata.csv')

    print(f"\nDataset sizes:")
    print(f"  Yew (filtered): {len(yew_df)}")
    print(f"  Non-yew: {len(no_yew_df)}")

    # Add labels
    yew_df['has_yew'] = True
    no_yew_df['has_yew'] = False

    # Shuffle and split
    val_split = 0.2
    seed = 42
    np.random.seed(seed)

    yew_indices = np.random.permutation(len(yew_df))
    no_yew_indices = np.random.permutation(len(no_yew_df))

    yew_val_size = int(len(yew_df) * val_split)
    no_yew_val_size = int(len(no_yew_df) * val_split)

    # Split
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

    # Save
    train_df.to_csv('data/processed/train_split_filtered.csv', index=False)
    val_df.to_csv('data/processed/val_split_filtered.csv', index=False)

    print(f"\n✓ Training split: {len(train_df)} samples")
    print(f"    Yew: {len(yew_train_idx)} | Non-yew: {len(no_yew_train_idx)}")
    print(f"✓ Validation split: {len(val_df)} samples")
    print(f"    Yew: {len(yew_val_idx)} | Non-yew: {len(no_yew_val_idx)}")

    print(f"\n✓ Saved to:")
    print(f"    data/processed/train_split_filtered.csv")
    print(f"    data/processed/val_split_filtered.csv")


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("DATASET FILTERING AND EXPANSION PIPELINE")
    print("="*80)

    # Step 1: Filter California and apply manual reviews
    print("\n### Step 1: Filter iNaturalist samples ###")
    filter_california_samples(
        'data/ee_imagery/image_patches_64x64/inat_yew_image_metadata.csv',
        'data/ee_imagery/image_review_results.json'
    )

    # Step 2: Extract additional non-yew samples
    print("\n### Step 2: Extract additional non-yew samples ###")
    extract_additional_no_yew_samples(num_samples=300, workers=6)

    # Step 3: Create training splits
    print("\n### Step 3: Create filtered training splits ###")
    create_filtered_training_splits()

    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print("\nFiltered dataset ready for training!")
    print("Use: python scripts/training/train_cnn.py --use-filtered")


if __name__ == '__main__':
    main()
