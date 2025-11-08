#!/usr/bin/env python3
"""
Fast Extraction of Yew Site Images Only
========================================

Extracts 64x64 pixel Sentinel-2 images ONLY for sites with Pacific Yew
in CWH and ICH zones. Optimized for speed.

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
        return False


def parse_yew_presence(composition_string):
    """Parse species composition string to extract Pacific Yew presence."""
    if not composition_string or pd.isna(composition_string):
        return False
    pattern = r'TW(\d{2,3})'
    match = re.search(pattern, str(composition_string))
    return match is not None


def load_yew_sites_only():
    """Load ONLY yew sites from CWH and ICH zones."""
    print("\nLoading yew site data...")

    # Load deduplicated inventory data
    inv_df = pd.read_csv(
        'data/processed/bc_sample_data_deduplicated.csv', low_memory=False)

    # Parse yew presence
    inv_df['has_yew'] = inv_df['SPB_CPCT_LS'].apply(parse_yew_presence)

    # Filter to CWH/ICH zones AND yew presence
    yew_sites = inv_df[
        (inv_df['BEC_ZONE'].isin(['CWH', 'ICH'])) &
        (inv_df['has_yew'] == True)
    ].copy()

    print(f"  Found {len(yew_sites)} yew sites in CWH/ICH")

    # Filter to sites with coordinates
    yew_sites = yew_sites[
        yew_sites['BC_ALBERS_X'].notna() &
        yew_sites['BC_ALBERS_Y'].notna()
    ].copy()

    print(f"  Sites with coordinates: {len(yew_sites)}")

    # Convert BC Albers (EPSG:3005) to WGS84 (EPSG:4326)
    print("  Converting coordinates...")
    transformer = Transformer.from_crs(
        "EPSG:3005", "EPSG:4326", always_xy=True)
    coords = transformer.transform(yew_sites['BC_ALBERS_X'].values,
                                   yew_sites['BC_ALBERS_Y'].values)
    yew_sites['lon'] = coords[0]
    yew_sites['lat'] = coords[1]

    # Get measurement year
    yew_sites['measurement_year'] = pd.to_numeric(
        yew_sites['MEAS_YR'], errors='coerce'
    ).fillna(2020).astype(int)

    print(f"\n✓ Ready to extract {len(yew_sites)} yew sites")
    print(
        f"  By zone: CWH={len(yew_sites[yew_sites['BEC_ZONE']=='CWH'])}, ICH={len(yew_sites[yew_sites['BEC_ZONE']=='ICH'])}")

    return yew_sites


def extract_sentinel2_patch_fast(lon, lat, year, patch_size=64, scale=10):
    """
    Fast extraction using getThumbURL method.
    More reliable than sampleRectangle for getting actual pixel arrays.
    """
    try:
        # Define the point and region
        point = ee.Geometry.Point([lon, lat])
        half_size_meters = (patch_size * scale) / 2
        meters_per_degree = 111320
        half_size_deg = half_size_meters / meters_per_degree

        region = ee.Geometry.Rectangle(
            [lon - half_size_deg, lat - half_size_deg,
             lon + half_size_deg, lat + half_size_deg]
        )

        # Date range
        start_date = f'{year}-01-01'
        end_date = f'{year}-12-31'

        # Load Sentinel-2 Surface Reflectance
        s2 = ee.ImageCollection('COPERNICUS/S2_SR') \
            .filterDate(start_date, end_date) \
            .filterBounds(point) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
            .select(['B2', 'B3', 'B4', 'B8'])

        count = s2.size().getInfo()
        if count == 0:
            return None

        # Take median composite
        composite = s2.median()

        # Use getThumbURL to get actual pixel data as PNG
        # This is more reliable than sampleRectangle
        url = composite.getThumbURL({
            'region': region,
            'dimensions': [patch_size, patch_size],
            'format': 'png',
            'min': 0,
            'max': 3000  # Typical Sentinel-2 range
        })

        # Download the thumbnail
        import requests
        from PIL import Image
        from io import BytesIO

        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return None

        # Load as image
        img = Image.open(BytesIO(response.content))
        img_array = np.array(img)

        # The PNG is RGB (3 channels), but we need 4 bands
        # So use sampleRectangle for the actual band values
        sample = composite.sampleRectangle(region=region, defaultValue=0)

        blue = np.array(sample.get('B2').getInfo())
        green = np.array(sample.get('B3').getInfo())
        red = np.array(sample.get('B4').getInfo())
        nir = np.array(sample.get('B8').getInfo())

        # Validate spatial variation
        if blue.size == 0 or blue.ndim != 2:
            return None

        if (blue.std() == 0 and green.std() == 0 and
                red.std() == 0 and nir.std() == 0):
            return None

        # Resize to exact dimensions if needed
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

        # Stack into array
        image_array = np.stack([blue, green, red, nir],
                               axis=0).astype(np.float32)

        return {
            'image': image_array,
            'shape': image_array.shape,
            'num_images': count
        }

    except Exception as e:
        return None


def extract_yew_sites(sites_df, output_dir):
    """Extract images for yew sites only."""
    output_dir = Path(output_dir)
    yew_dir = output_dir / 'yew'
    yew_dir.mkdir(parents=True, exist_ok=True)

    metadata_records = []
    success_count = 0
    failure_count = 0

    print(f"\n{'='*80}")
    print(f"EXTRACTING YEW SITE IMAGES")
    print(f"{'='*80}")

    start_time = time.time()

    for idx, row in tqdm(sites_df.iterrows(), total=len(sites_df), desc="Extracting yew sites"):
        site_id = str(row['SITE_IDENTIFIER'])
        image_path = yew_dir / f"{site_id}.npy"

        # Extract image
        result = extract_sentinel2_patch_fast(
            row['lon'],
            row['lat'],
            row['measurement_year']
        )

        if result is not None:
            # Save image
            np.save(image_path, result['image'])

            # Record metadata
            metadata_records.append({
                'site_identifier': site_id,
                'has_yew': True,
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
        else:
            failure_count += 1

        # Rate limiting
        time.sleep(0.1)

    # Save metadata
    if metadata_records:
        meta_df = pd.DataFrame(metadata_records)
        metadata_file = output_dir / 'yew_image_metadata.csv'
        meta_df.to_csv(metadata_file, index=False)
        print(f"\n✓ Metadata saved: {metadata_file}")

    # Summary
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"Success: {success_count}/{len(sites_df)}")
    print(f"Failed: {failure_count}/{len(sites_df)}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average: {total_time/len(sites_df):.1f} seconds per site")
    print(f"\nImages saved to: {yew_dir}")


def main():
    """Main execution."""
    print("="*80)
    print("FAST YEW SITE IMAGE EXTRACTION")
    print("="*80)
    print("\nConfiguration:")
    print("  - Sites: YEW ONLY (CWH/ICH zones)")
    print("  - Patch size: 64x64 pixels")
    print("  - Resolution: 10m/pixel")
    print("  - Bands: Blue, Green, Red, NIR")

    # Initialize Earth Engine
    if not initialize_earth_engine():
        return

    # Load yew sites only
    try:
        yew_sites = load_yew_sites_only()
    except Exception as e:
        print(f"\n✗ Error loading sites: {e}")
        return

    if len(yew_sites) == 0:
        print("\n✗ No yew sites found!")
        return

    # Extract images
    output_dir = Path('data/ee_imagery/image_patches_64x64')
    extract_yew_sites(yew_sites, output_dir)

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)


if __name__ == '__main__':
    main()
