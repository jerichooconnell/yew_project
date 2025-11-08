#!/usr/bin/env python3
"""
Extract 64x64 Pixel Image Patches from Earth Engine
===================================================

Downloads Sentinel-2 image patches for CNN training, filtered to:
- Unique SITE_IDENTIFIERs only (deduplicated)
- CWH and ICH biogeoclimatic zones only

Saves images as numpy arrays (.npy) for efficient loading during training.

Author: GitHub Copilot
Date: November 7, 2025
"""

import pandas as pd
import numpy as np
import ee
from pathlib import Path
import time
from tqdm import tqdm
import json
import re
from datetime import datetime
from pyproj import Transformer


def initialize_earth_engine():
    """Initialize Earth Engine with authentication."""
    try:
        ee.Initialize(project='carbon-storm-206002')
        print("✓ Earth Engine initialized")
        return True
    except Exception as e:
        print(f"✗ Earth Engine initialization failed: {e}")
        print("  Please run: earthengine authenticate")
        return False


def parse_yew_presence(composition_string):
    """Parse species composition string to extract Pacific Yew presence."""
    if not composition_string or pd.isna(composition_string):
        return False
    pattern = r'TW(\d{2,3})'
    match = re.search(pattern, str(composition_string))
    return match is not None


def load_cwh_ich_sites():
    """Load unique sites from CWH and ICH biogeoclimatic zones."""
    print("\nLoading site data...")

    # Load deduplicated inventory data
    inv_df = pd.read_csv(
        'data/processed/bc_sample_data_deduplicated.csv', low_memory=False)
    print(f"  Total unique sites: {len(inv_df)}")

    # Filter to CWH and ICH zones
    cwh_ich = inv_df[inv_df['BEC_ZONE'].isin(['CWH', 'ICH'])].copy()
    print(f"  Sites in CWH/ICH: {len(cwh_ich)}")

    # Parse yew presence
    cwh_ich['has_yew'] = cwh_ich['SPB_CPCT_LS'].apply(parse_yew_presence)
    print(f"    With yew: {cwh_ich['has_yew'].sum()}")
    print(f"    Without yew: {(~cwh_ich['has_yew']).sum()}")

    # Filter to sites with coordinates
    cwh_ich_coords = cwh_ich[cwh_ich['BC_ALBERS_X'].notna(
    ) & cwh_ich['BC_ALBERS_Y'].notna()].copy()
    print(f"  Sites with coordinates: {len(cwh_ich_coords)}")

    # Convert BC Albers (EPSG:3005) to WGS84 (EPSG:4326) for Earth Engine
    print("  Converting coordinates to lat/lon...")
    transformer = Transformer.from_crs(
        "EPSG:3005", "EPSG:4326", always_xy=True)
    coords = transformer.transform(cwh_ich_coords['BC_ALBERS_X'].values,
                                   cwh_ich_coords['BC_ALBERS_Y'].values)
    cwh_ich_coords['lon'] = coords[0]
    cwh_ich_coords['lat'] = coords[1]

    # Get measurement year (use most recent if multiple)
    cwh_ich_coords['measurement_year'] = pd.to_numeric(
        cwh_ich_coords['MEAS_YR'], errors='coerce'
    ).fillna(2020).astype(int)

    print(f"\n✓ Ready to extract {len(cwh_ich_coords)} sites")
    print(
        f"  Year range: {cwh_ich_coords['measurement_year'].min()} - {cwh_ich_coords['measurement_year'].max()}")

    return cwh_ich_coords


def extract_sentinel2_patch(lon, lat, year, patch_size=64, scale=10):
    """
    Extract a 64x64 pixel Sentinel-2 image patch.

    Args:
        lon: Longitude (WGS84)
        lat: Latitude (WGS84)
        year: Year for imagery
        patch_size: Size of patch in pixels (default 64)
        scale: Resolution in meters (default 10m for Sentinel-2)

    Returns:
        Dictionary with band arrays (blue, green, red, nir) or None if failed
    """
    try:
        # Define the point
        point = ee.Geometry.Point([lon, lat])

        # Define the region more precisely using a square
        # 64 pixels * 10m = 640m side length
        half_size_meters = (patch_size * scale) / 2  # 320m
        # Convert meters to degrees (approximate at this latitude)
        meters_per_degree = 111320  # at equator, varies with latitude
        half_size_deg = half_size_meters / meters_per_degree

        region = ee.Geometry.Rectangle(
            [lon - half_size_deg, lat - half_size_deg,
             lon + half_size_deg, lat + half_size_deg]
        )

        # Date range for the year
        start_date = f'{year}-01-01'
        end_date = f'{year}-12-31'

        # Load Sentinel-2 Surface Reflectance
        s2 = ee.ImageCollection('COPERNICUS/S2_SR') \
            .filterDate(start_date, end_date) \
            .filterBounds(point) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
            .select(['B2', 'B3', 'B4', 'B8'])  # Blue, Green, Red, NIR

        # Check if any images available
        count = s2.size().getInfo()
        if count == 0:
            return None

        # Take median composite
        composite = s2.median()

        # Sample the rectangle to get pixel arrays with proper scale
        sample = composite.sampleRectangle(
            region=region,
            defaultValue=0
        )

        # Extract band arrays
        blue = np.array(sample.get('B2').getInfo())
        green = np.array(sample.get('B3').getInfo())
        red = np.array(sample.get('B4').getInfo())
        nir = np.array(sample.get('B8').getInfo())

        # Check if we got valid arrays with spatial variation
        if blue.size == 0 or blue.ndim != 2:
            return None

        # Check for spatial variation (not all same value)
        if (blue.std() == 0 and green.std() == 0 and
                red.std() == 0 and nir.std() == 0):
            # All bands are constant - extraction failed to get pixel-level data
            return None

        # Verify dimensions and resize if needed
        if blue.shape != (patch_size, patch_size):
            # Resize to exact dimensions
            from scipy.ndimage import zoom
            if blue.size > 0:
                zoom_factor_y = patch_size / blue.shape[0]
                zoom_factor_x = patch_size / blue.shape[1]

                blue = zoom(blue, (zoom_factor_y, zoom_factor_x), order=1)
                green = zoom(green, (zoom_factor_y, zoom_factor_x), order=1)
                red = zoom(red, (zoom_factor_y, zoom_factor_x), order=1)
                nir = zoom(nir, (zoom_factor_y, zoom_factor_x), order=1)

                # Crop or pad to exact size
                blue = blue[:patch_size, :patch_size]
                green = green[:patch_size, :patch_size]
                red = red[:patch_size, :patch_size]
                nir = nir[:patch_size, :patch_size]
            else:
                return None

        # Stack into single array: (4, 64, 64) for (channels, height, width)
        image_array = np.stack([blue, green, red, nir],
                               axis=0).astype(np.float32)

        return {
            'image': image_array,
            'shape': image_array.shape,
            'num_images': count
        }

    except Exception as e:
        return None


def extract_and_save_patches(sites_df, output_dir, batch_size=100, resume=True):
    """
    Extract image patches for all sites and save to disk.

    Args:
        sites_df: DataFrame with site information
        output_dir: Directory to save images
        batch_size: Number of sites to process before saving progress
        resume: If True, skip already processed sites
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for yew and non-yew
    yew_dir = output_dir / 'yew'
    no_yew_dir = output_dir / 'no_yew'
    yew_dir.mkdir(exist_ok=True)
    no_yew_dir.mkdir(exist_ok=True)

    # Progress tracking file
    progress_file = output_dir / 'extraction_progress.json'
    metadata_file = output_dir / 'image_metadata.csv'

    # Load existing progress
    processed_sites = set()
    metadata_records = []

    if resume and progress_file.exists():
        with open(progress_file, 'r') as f:
            progress = json.load(f)
            processed_sites = set(progress.get('processed_sites', []))
        print(f"\n✓ Resuming: {len(processed_sites)} sites already processed")

    if resume and metadata_file.exists():
        existing_meta = pd.read_csv(metadata_file)
        metadata_records = existing_meta.to_dict('records')

    # Extract images
    print(f"\n{'='*80}")
    print(f"EXTRACTING IMAGE PATCHES")
    print(f"{'='*80}")

    success_count = 0
    failure_count = 0
    skipped_count = len(processed_sites)

    start_time = time.time()

    for idx, row in tqdm(sites_df.iterrows(), total=len(sites_df), desc="Extracting"):
        site_id = str(row['SITE_IDENTIFIER'])

        # Skip if already processed
        if site_id in processed_sites:
            continue

        # Determine output path
        if row['has_yew']:
            image_path = yew_dir / f"{site_id}.npy"
        else:
            image_path = no_yew_dir / f"{site_id}.npy"

        # Extract image
        result = extract_sentinel2_patch(
            row['lon'],
            row['lat'],
            row['measurement_year']
        )

        if result is not None:
            # Save image as numpy array
            np.save(image_path, result['image'])

            # Record metadata
            metadata_records.append({
                'site_identifier': site_id,
                'has_yew': row['has_yew'],
                'bec_zone': row['BEC_ZONE'],
                'measurement_year': row['measurement_year'],
                'lon': row['lon'],
                'lat': row['lat'],
                'image_path': str(image_path.relative_to(output_dir)),
                'image_shape': str(result['shape']),
                'num_source_images': result['num_images'],
                'extraction_date': datetime.now().isoformat()
            })

            success_count += 1
            processed_sites.add(site_id)

        else:
            failure_count += 1

        # Save progress periodically
        if (success_count + failure_count) % batch_size == 0:
            # Save progress
            with open(progress_file, 'w') as f:
                json.dump({
                    'processed_sites': list(processed_sites),
                    'success_count': success_count,
                    'failure_count': failure_count,
                    'last_update': datetime.now().isoformat()
                }, f)

            # Save metadata
            meta_df = pd.DataFrame(metadata_records)
            meta_df.to_csv(metadata_file, index=False)

            elapsed = time.time() - start_time
            rate = (success_count + failure_count) / elapsed
            remaining = (len(sites_df) - len(processed_sites)) / rate / 60

            print(
                f"\n  Progress: {success_count} success, {failure_count} failed")
            print(
                f"  Rate: {rate:.1f} sites/sec, Est. remaining: {remaining:.1f} min")

        # Rate limiting to avoid Earth Engine quota issues
        time.sleep(0.1)

    # Final save
    with open(progress_file, 'w') as f:
        json.dump({
            'processed_sites': list(processed_sites),
            'success_count': success_count,
            'failure_count': failure_count,
            'skipped_count': skipped_count,
            'last_update': datetime.now().isoformat(),
            'completed': True
        }, f)

    meta_df = pd.DataFrame(metadata_records)
    meta_df.to_csv(metadata_file, index=False)

    # Summary
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"Success: {success_count}")
    print(f"Failed: {failure_count}")
    print(f"Skipped (already done): {skipped_count}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(
        f"Average rate: {(success_count + failure_count)/total_time:.2f} sites/sec")
    print(f"\nImages saved to: {output_dir}")
    print(f"Metadata saved to: {metadata_file}")

    # Summary by class
    yew_images = len(list(yew_dir.glob('*.npy')))
    no_yew_images = len(list(no_yew_dir.glob('*.npy')))
    print(f"\nClass distribution:")
    print(f"  Yew images: {yew_images}")
    print(f"  No-yew images: {no_yew_images}")
    print(f"  Class imbalance ratio: 1:{no_yew_images/max(yew_images, 1):.1f}")


def main():
    """Main execution."""
    print("="*80)
    print("EARTH ENGINE IMAGE PATCH EXTRACTION FOR CNN TRAINING")
    print("="*80)
    print("\nConfiguration:")
    print("  - Zones: CWH, ICH only")
    print("  - Sites: Unique (deduplicated) only")
    print("  - Patch size: 64x64 pixels")
    print("  - Resolution: 10m/pixel")
    print("  - Bands: Blue, Green, Red, NIR")
    print("  - Format: NumPy arrays (.npy)")

    # Initialize Earth Engine
    print("\nInitializing Earth Engine...")
    if not initialize_earth_engine():
        return

    # Load sites
    try:
        sites_df = load_cwh_ich_sites()
    except ImportError:
        print("\n✗ pyproj library required for coordinate transformation")
        print("  Install with: pip install pyproj")
        return
    except Exception as e:
        print(f"\n✗ Error loading sites: {e}")
        return

    # Extract and save patches
    output_dir = Path('data/ee_imagery/image_patches_64x64')
    extract_and_save_patches(sites_df, output_dir, batch_size=100, resume=True)

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)


if __name__ == '__main__':
    main()
