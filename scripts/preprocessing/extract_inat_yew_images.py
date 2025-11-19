#!/usr/bin/env python3
"""
Extract Yew Images from iNaturalist Observations
=================================================

Uses iNaturalist citizen science observations of Pacific Yew to extract
satellite imagery. These represent actual confirmed yew sightings.

Author: GitHub Copilot
Date: November 14, 2025
"""

import pandas as pd
import numpy as np
import ee
from pathlib import Path
import time
from tqdm import tqdm
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


def initialize_earth_engine():
    """Initialize Earth Engine with authentication."""
    try:
        ee.Initialize(project='carbon-storm-206002')
        return True
    except Exception as e:
        print(f"✗ Earth Engine initialization failed: {e}")
        return False


def load_inat_observations(max_accuracy=100):
    """
    Load iNaturalist yew observations.

    Args:
        max_accuracy: Maximum positional accuracy in meters (filters out imprecise observations)

    Returns:
        DataFrame with filtered observations
    """
    print("\nLoading iNaturalist yew observations...")

    df = pd.read_csv('data/inat_observations/observations-558049.csv')

    # Filter for good coordinates and accuracy
    df_filtered = df[
        df['latitude'].notna() &
        df['longitude'].notna() &
        (df['positional_accuracy'] <= max_accuracy)
    ].copy()

    print(f"  Total observations: {len(df)}")
    print(
        f"  With coordinates: {df[['latitude', 'longitude']].notna().all(axis=1).sum()}")
    print(f"  After accuracy filter (≤{max_accuracy}m): {len(df_filtered)}")

    # Add observation year
    df_filtered['obs_year'] = pd.to_datetime(
        df_filtered['observed_on']).dt.year

    print(
        f"\nYear range: {df_filtered['obs_year'].min()} - {df_filtered['obs_year'].max()}")
    print(
        f"Median positional accuracy: {df_filtered['positional_accuracy'].median():.1f}m")

    return df_filtered


def extract_single_site(site_data, output_dir):
    """Extract image for a single iNaturalist observation."""
    obs_id, lon, lat, obs_year, accuracy = site_data

    try:
        point = ee.Geometry.Point([lon, lat])
        patch_size = 64
        scale = 10
        half_size_meters = (patch_size * scale) / 2
        meters_per_degree = 111320
        half_size_deg = half_size_meters / meters_per_degree

        region = ee.Geometry.Rectangle(
            [lon - half_size_deg, lat - half_size_deg,
             lon + half_size_deg, lat + half_size_deg]
        )

        # Use 2020-2024 imagery for consistency (Sentinel-2 availability)
        # Most observations are recent anyway
        s2 = ee.ImageCollection('COPERNICUS/S2_SR') \
            .filterDate('2020-01-01', '2024-12-31') \
            .filterBounds(point) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
            .select(['B2', 'B3', 'B4', 'B8'])

        count = s2.size().getInfo()
        if count == 0:
            return None, f"No images for obs_{obs_id}"

        composite = s2.median()
        # Reproject with explicit scale to get pixel-level data
        composite_reprojected = composite.reproject('EPSG:4326', scale=scale)
        sample = composite_reprojected.sampleRectangle(
            region=region, defaultValue=0)

        # Extract bands
        blue = np.array(sample.get('B2').getInfo())
        green = np.array(sample.get('B3').getInfo())
        red = np.array(sample.get('B4').getInfo())
        nir = np.array(sample.get('B8').getInfo())

        # Validate
        if blue.size == 0 or blue.ndim != 2:
            return None, f"Invalid array for obs_{obs_id}"

        if (blue.std() == 0 and green.std() == 0 and
                red.std() == 0 and nir.std() == 0):
            return None, f"No spatial variation for obs_{obs_id}"

        # Resize if needed
        if blue.shape != (patch_size, patch_size):
            from scipy.ndimage import zoom
            zoom_factor_y = patch_size / blue.shape[0]
            zoom_factor_x = patch_size / blue.shape[1]

            blue = zoom(blue, (zoom_factor_y, zoom_factor_x), order=1)[
                :patch_size, :patch_size]
            green = zoom(green, (zoom_factor_y, zoom_factor_x), order=1)[
                :patch_size, :patch_size]
            red = zoom(red, (zoom_factor_y, zoom_factor_x), order=1)[
                :patch_size, :patch_size]
            nir = zoom(nir, (zoom_factor_y, zoom_factor_x), order=1)[
                :patch_size, :patch_size]

        # Save image
        image_array = np.stack([blue, green, red, nir],
                               axis=0).astype(np.float32)
        image_path = output_dir / 'inat_yew' / f"inat_{obs_id}.npy"
        np.save(image_path, image_array)

        # Return metadata
        metadata = {
            'observation_id': obs_id,
            'has_yew': True,
            'source': 'iNaturalist',
            'observation_year': obs_year,
            'positional_accuracy': accuracy,
            'lon': lon,
            'lat': lat,
            'image_path': f'inat_yew/inat_{obs_id}.npy',
            'image_shape': str(image_array.shape),
            'num_source_images': count,
            'extraction_date': datetime.now().isoformat()
        }

        return metadata, None

    except Exception as e:
        return None, f"Error for obs_{obs_id}: {str(e)[:50]}"


def extract_parallel(obs_df, output_dir, max_workers=4, limit=None):
    """Extract images in parallel."""
    output_dir = Path(output_dir)
    inat_dir = output_dir / 'inat_yew'
    inat_dir.mkdir(parents=True, exist_ok=True)

    # Limit if specified
    if limit:
        obs_df = obs_df.head(limit)

    # Prepare site data
    site_data_list = [
        (row['id'], row['longitude'], row['latitude'],
         row['obs_year'], row['positional_accuracy'])
        for _, row in obs_df.iterrows()
    ]

    metadata_records = []
    errors = []

    print(
        f"\nExtracting {len(site_data_list)} iNaturalist observations with {max_workers} workers...")

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_site = {
            executor.submit(extract_single_site, site_data, output_dir): site_data[0]
            for site_data in site_data_list
        }

        # Process as completed
        for future in tqdm(as_completed(future_to_site), total=len(site_data_list)):
            metadata, error = future.result()
            if metadata:
                metadata_records.append(metadata)
            if error:
                errors.append(error)

    # Save metadata
    if metadata_records:
        meta_df = pd.DataFrame(metadata_records)
        metadata_file = output_dir / 'inat_yew_image_metadata.csv'
        meta_df.to_csv(metadata_file, index=False)

    # Summary
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"Success: {len(metadata_records)}/{len(site_data_list)}")
    print(f"Failed: {len(errors)}/{len(site_data_list)}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average: {total_time/len(site_data_list):.1f} seconds per site")

    if errors:
        print(f"\nFirst 10 errors:")
        for err in errors[:10]:
            print(f"  {err}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract satellite images from iNaturalist yew observations')
    parser.add_argument('--max-accuracy', type=int, default=100,
                        help='Maximum positional accuracy in meters (default: 100)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of observations to extract (for testing)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')

    args = parser.parse_args()

    print("="*80)
    print("INATURALIST YEW IMAGE EXTRACTION")
    print("="*80)

    if not initialize_earth_engine():
        return

    obs_df = load_inat_observations(max_accuracy=args.max_accuracy)
    if len(obs_df) == 0:
        print("\n✗ No observations found!")
        return

    output_dir = Path('data/ee_imagery/image_patches_64x64')
    extract_parallel(obs_df, output_dir,
                     max_workers=args.workers, limit=args.limit)

    print("\n✓ Done!")
    print(f"\nNext steps:")
    print(f"  1. Convert to PNG: python scripts/preprocessing/convert_npy_to_png.py")
    print(f"  2. Visualize: python scripts/visualization/display_ee_patches.py")


if __name__ == '__main__':
    main()
