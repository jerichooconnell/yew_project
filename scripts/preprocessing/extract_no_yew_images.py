#!/usr/bin/env python3
"""
Extract Non-Yew Site Images
============================

Extracts satellite imagery from 100 sites that do NOT contain Pacific Yew.
Uses the same methodology as yew site extraction for comparison.

Author: GitHub Copilot
Date: November 8, 2025
"""

from preprocessing.city_filter import filter_dataframe
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


def initialize_earth_engine():
    """Initialize Earth Engine with authentication."""
    try:
        ee.Initialize(project='carbon-storm-206002')
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


def load_no_yew_sites(n_sites=100, skip_existing=True):
    """Load sites WITHOUT yew from CWH and ICH zones."""
    print("\nLoading non-yew site data...")

    inv_df = pd.read_csv(
        'data/processed/bc_sample_data_deduplicated.csv', low_memory=False)
    inv_df['has_yew'] = inv_df['SPB_CPCT_LS'].apply(parse_yew_presence)

    # Filter: CWH/ICH zones, NO yew, has coordinates
    no_yew_sites = inv_df[
        (inv_df['BEC_ZONE'].isin(['CWH', 'ICH'])) &
        (inv_df['has_yew'] == False) &
        inv_df['BC_ALBERS_X'].notna() &
        inv_df['BC_ALBERS_Y'].notna()
    ].copy()

    # Skip already extracted samples
    if skip_existing:
        metadata_file = Path(
            'data/ee_imagery/image_patches_64x64/no_yew_image_metadata.csv')
        if metadata_file.exists():
            existing_df = pd.read_csv(metadata_file)
            existing_ids = set(existing_df['site_identifier'].values)
            no_yew_sites = no_yew_sites[~no_yew_sites['SITE_IDENTIFIER'].isin(
                existing_ids)]
            print(f"  Skipping {len(existing_ids)} already extracted samples")

    # Randomly sample n_sites
    if len(no_yew_sites) > n_sites:
        # Different seed for new samples
        no_yew_sites = no_yew_sites.sample(n=n_sites, random_state=43)

    print(f"  Found {len(no_yew_sites)} non-yew sites to extract")

    # Convert coordinates
    transformer = Transformer.from_crs(
        "EPSG:3005", "EPSG:4326", always_xy=True)
    coords = transformer.transform(no_yew_sites['BC_ALBERS_X'].values,
                                   no_yew_sites['BC_ALBERS_Y'].values)
    no_yew_sites['lon'] = coords[0]
    no_yew_sites['lat'] = coords[1]
    no_yew_sites['measurement_year'] = pd.to_numeric(
        no_yew_sites['MEAS_YR'], errors='coerce'
    ).fillna(2020).astype(int)

    return no_yew_sites


def extract_single_site(site_data, output_dir):
    """Extract image for a single site."""
    site_id, lon, lat, year, bec_zone = site_data

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

        # Load imagery - use 2020-2024 for all sites (Sentinel-2 only from 2015+)
        s2 = ee.ImageCollection('COPERNICUS/S2_SR') \
            .filterDate('2020-01-01', '2024-12-31') \
            .filterBounds(point) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
            .select(['B2', 'B3', 'B4', 'B8'])

        count = s2.size().getInfo()
        if count == 0:
            return None, f"No images for {site_id}"

        composite = s2.median()
        # KEY FIX: Reproject with explicit scale before sampling
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
            return None, f"Invalid array for {site_id}"

        if (blue.std() == 0 and green.std() == 0 and
                red.std() == 0 and nir.std() == 0):
            return None, f"No spatial variation for {site_id}"

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
        image_path = output_dir / 'no_yew' / f"{site_id}.npy"
        np.save(image_path, image_array)

        # Return metadata
        metadata = {
            'site_identifier': site_id,
            'has_yew': False,
            'bec_zone': bec_zone,
            'measurement_year': year,
            'lon': lon,
            'lat': lat,
            'image_path': f'no_yew/{site_id}.npy',
            'image_shape': str(image_array.shape),
            'num_source_images': count,
            'extraction_date': datetime.now().isoformat()
        }

        return metadata, None

    except Exception as e:
        return None, f"Error for {site_id}: {str(e)[:50]}"


def extract_parallel(sites_df, output_dir, max_workers=4):
    """Extract images in parallel."""
    output_dir = Path(output_dir)
    no_yew_dir = output_dir / 'no_yew'
    no_yew_dir.mkdir(parents=True, exist_ok=True)

    # Prepare site data
    site_data_list = [
        (row['SITE_IDENTIFIER'], row['lon'], row['lat'],
         row['measurement_year'], row['BEC_ZONE'])
        for _, row in sites_df.iterrows()
    ]

    metadata_records = []
    errors = []

    print(
        f"\nExtracting {len(site_data_list)} non-yew sites with {max_workers} workers...")

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
        metadata_file = output_dir / 'no_yew_image_metadata.csv'
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
    parser = argparse.ArgumentParser(description='Extract non-yew site images')
    parser.add_argument('--limit', type=int, default=100,
                        help='Number of sites to extract')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--filter-cities', action='store_true',
                        help='Exclude observations near major NW cities')
    parser.add_argument('--min-city-distance', type=float, default=None,
                        help='Minimum distance from cities in km (uses city-specific defaults if not set)')
    args = parser.parse_args()

    print("="*80)
    print("NON-YEW SITE IMAGE EXTRACTION")
    print("="*80)

    if not initialize_earth_engine():
        return

    no_yew_sites = load_no_yew_sites(n_sites=args.limit, skip_existing=True)

    # Apply city filter if requested
    if args.filter_cities:
        print("\nApplying city filter...")
        no_yew_sites = filter_dataframe(
            no_yew_sites,
            lat_col='latitude',
            lon_col='longitude',
            min_distance_km=args.min_city_distance,
            add_city_info=True
        )
        print(f"  Remaining after city filter: {len(no_yew_sites)}")

    if len(no_yew_sites) == 0:
        print("\n✗ No non-yew sites found!")
        return

    output_dir = Path('data/ee_imagery/image_patches_64x64')
    extract_parallel(no_yew_sites, output_dir, max_workers=args.workers)

    print("\n✓ Done!")


if __name__ == '__main__':
    main()
