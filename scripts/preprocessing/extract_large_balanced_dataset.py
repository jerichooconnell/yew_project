#!/usr/bin/env python3
"""
Extract Large Balanced Dataset (1000 Yew + 1000 Non-Yew)
=========================================================

Extracts a large, balanced dataset of satellite imagery for yew detection,
automatically filtering out urban areas for both positive and negative samples.

Features:
    - 1000 yew observations from iNaturalist (BC/WA only, no California)
    - 1000 non-yew sites from BC forestry inventory (CWH/ICH zones)
    - Automatic city filtering (excludes major NW cities)
    - Parallel extraction with progress tracking
    - Comprehensive metadata and quality checks

Author: GitHub Copilot
Date: November 19, 2025
"""

import pandas as pd
import numpy as np
import ee
from pathlib import Path
import time
import sys
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import sys
from pyproj import Transformer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import city filter - handle different import scenarios
try:
    from preprocessing.city_filter import filter_dataframe
except ModuleNotFoundError:
    # Direct import from same directory
    import importlib.util
    city_filter_path = Path(__file__).parent / 'city_filter.py'
    spec = importlib.util.spec_from_file_location(
        "city_filter", city_filter_path)
    city_filter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(city_filter)
    filter_dataframe = city_filter.filter_dataframe


def initialize_earth_engine():
    """Initialize Earth Engine."""
    try:
        ee.Initialize(project='carbon-storm-206002')
        print("✓ Earth Engine initialized")
        return True
    except Exception as e:
        print(f"✗ Earth Engine initialization failed: {e}")
        return False


def create_sentinel2_composite():
    """Create cloud-free Sentinel-2 composite (2020-2024)."""
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate('2020-01-01', '2024-12-31') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

    def mask_clouds(image):
        qa = image.select('QA60')
        cloud_mask = qa.bitwiseAnd(1 << 10).eq(0).And(
            qa.bitwiseAnd(1 << 11).eq(0))
        return image.updateMask(cloud_mask)

    return s2.map(mask_clouds).median()


# ============================================================================
# YEW OBSERVATIONS
# ============================================================================

def load_yew_observations(max_accuracy=50, filter_cities=True):
    """
    Load iNaturalist yew observations (BC/WA only, filtered for cities).

    Args:
        max_accuracy: Maximum GPS positional accuracy in meters
        filter_cities: Whether to exclude observations near cities

    Returns:
        DataFrame of filtered observations
    """
    print("\n" + "="*80)
    print("LOADING YEW OBSERVATIONS")
    print("="*80)

    inat_df = pd.read_csv('data/inat_observations/observations-558049.csv')
    print(f"  Total iNaturalist observations: {len(inat_df)}")

    # Filter by positional accuracy
    accurate = inat_df[inat_df['positional_accuracy'] <= max_accuracy].copy()
    print(f"  With accuracy <= {max_accuracy}m: {len(accurate)}")

    # Filter by latitude (exclude California: lat >= 42°N)
    bc_wa = accurate[accurate['latitude'] >= 42.0].copy()
    print(f"  BC/WA only (lat >= 42°N): {len(bc_wa)}")

    # Apply city filter
    if filter_cities:
        print("\n  Applying city filter to yew observations...")
        bc_wa = filter_dataframe(
            bc_wa,
            lat_col='latitude',
            lon_col='longitude',
            min_distance_km=None,  # Use city-specific radii
            add_city_info=True
        )
        print(f"  Remaining after city filter: {len(bc_wa)}")

    # Filter out already extracted
    existing_meta = Path(
        'data/ee_imagery/image_patches_64x64/inat_yew_image_metadata.csv')
    if existing_meta.exists():
        existing_df = pd.read_csv(existing_meta)
        existing_ids = set(existing_df['observation_id'].values)
        bc_wa = bc_wa[~bc_wa['id'].isin(existing_ids)]
        print(
            f"  After removing {len(existing_ids)} already extracted: {len(bc_wa)}")

    # Sort by accuracy (best first) and nearest city distance (farthest first)
    if 'distance_to_city' in bc_wa.columns:
        bc_wa = bc_wa.sort_values(['positional_accuracy', 'distance_to_city'],
                                  ascending=[True, False]).reset_index(drop=True)
    else:
        bc_wa = bc_wa.sort_values('positional_accuracy').reset_index(drop=True)

    print(f"\n✓ Available yew observations: {len(bc_wa)}")
    return bc_wa


def extract_yew_observation(obs, composite, output_dir):
    """Extract image for a single yew observation."""
    obs_id = obs['id']
    lat = obs['latitude']
    lon = obs['longitude']

    try:
        # Define region (64x64 pixels at 10m resolution = 640m x 640m)
        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(320).bounds()

        # Select bands and reproject (CRITICAL for pixel-level data)
        image = composite.select(['B2', 'B3', 'B4', 'B8']) \
            .reproject('EPSG:4326', scale=10)

        # Sample as array
        sample = image.sampleRectangle(region=region, defaultValue=0)

        # Get arrays
        arrays = {
            'B2': np.array(sample.get('B2').getInfo()),
            'B3': np.array(sample.get('B3').getInfo()),
            'B4': np.array(sample.get('B4').getInfo()),
            'B8': np.array(sample.get('B8').getInfo())
        }

        # Stack into (4, H, W) array
        stacked = np.stack([arrays['B2'], arrays['B3'],
                           arrays['B4'], arrays['B8']], axis=0)

        # Resize/crop to exactly 64x64
        if stacked.shape[1] != 64 or stacked.shape[2] != 64:
            from scipy.ndimage import zoom
            factors = (1, 64/stacked.shape[1], 64/stacked.shape[2])
            stacked = zoom(stacked, factors, order=1)

        # Save image
        output_file = output_dir / f'inat_{obs_id}.npy'
        np.save(output_file, stacked.astype(np.float32))

        # Return metadata
        metadata = {
            'observation_id': obs_id,
            'has_yew': True,
            'source': 'iNaturalist',
            'observation_year': obs.get('observed_on_string', '')[:4] if pd.notna(obs.get('observed_on_string')) else None,
            'positional_accuracy': obs['positional_accuracy'],
            'lon': lon,
            'lat': lat,
            'image_path': f'inat_yew/inat_{obs_id}.npy',
            'image_shape': stacked.shape,
            'num_source_images': 'composite',
            'extraction_date': datetime.now().strftime('%Y-%m-%d')
        }

        # Add city info if available
        if 'nearest_city' in obs and pd.notna(obs['nearest_city']):
            metadata['nearest_city'] = obs['nearest_city']
            metadata['distance_to_city'] = obs['distance_to_city']

        return metadata

    except Exception as e:
        print(f"\n✗ Failed to extract yew observation {obs_id}: {e}")
        return None


# ============================================================================
# NON-YEW SITES
# ============================================================================

def parse_yew_presence(composition_string):
    """Parse species composition string to check for Pacific Yew."""
    if not composition_string or pd.isna(composition_string):
        return False
    pattern = r'TW(\d{2,3})'
    match = re.search(pattern, str(composition_string))
    return match is not None


def load_no_yew_sites(n_sites=1000, filter_cities=True):
    """
    Load BC forestry sites WITHOUT yew from CWH and ICH zones.

    Args:
        n_sites: Number of sites to load
        filter_cities: Whether to exclude sites near cities

    Returns:
        DataFrame of no-yew sites
    """
    print("\n" + "="*80)
    print("LOADING NON-YEW SITES")
    print("="*80)

    inv_df = pd.read_csv(
        'data/processed/bc_sample_data_deduplicated.csv', low_memory=False)
    print(f"  Total BC inventory sites: {len(inv_df)}")

    # Parse yew presence
    inv_df['has_yew'] = inv_df['SPB_CPCT_LS'].apply(parse_yew_presence)

    # Filter for sites WITHOUT yew
    no_yew = inv_df[inv_df['has_yew'] == False].copy()
    print(f"  Sites without yew: {len(no_yew)}")

    # Filter for CWH and ICH zones (yew habitat)
    no_yew = no_yew[no_yew['BEC_ZONE'].isin(['CWH', 'ICH'])].copy()
    print(f"  In CWH/ICH zones: {len(no_yew)}")

    # Filter for sites with coordinates
    no_yew = no_yew[no_yew['BC_ALBERS_X'].notna(
    ) & no_yew['BC_ALBERS_Y'].notna()].copy()
    print(f"  With valid coordinates: {len(no_yew)}")

    # Convert BC Albers to Lat/Lon
    print("  Converting coordinates to lat/lon...")
    transformer = Transformer.from_crs(
        'EPSG:3005', 'EPSG:4326', always_xy=True)
    coords = transformer.transform(
        no_yew['BC_ALBERS_X'].values, no_yew['BC_ALBERS_Y'].values)
    no_yew['longitude'] = coords[0]
    no_yew['latitude'] = coords[1]

    # Filter for BC/WA region (same as yew: lat >= 42°N)
    no_yew = no_yew[no_yew['latitude'] >= 42.0].copy()
    print(f"  In BC/WA region (lat >= 42°N): {len(no_yew)}")

    # Apply city filter
    if filter_cities:
        print("\n  Applying city filter to non-yew sites...")
        no_yew = filter_dataframe(
            no_yew,
            lat_col='latitude',
            lon_col='longitude',
            min_distance_km=None,  # Use city-specific radii
            add_city_info=True
        )
        print(f"  Remaining after city filter: {len(no_yew)}")

    # Filter out already extracted
    existing_meta = Path(
        'data/ee_imagery/image_patches_64x64/no_yew_image_metadata.csv')
    if existing_meta.exists():
        existing_df = pd.read_csv(existing_meta)
        existing_ids = set(existing_df['site_identifier'].values)
        no_yew = no_yew[~no_yew['SITE_IDENTIFIER'].isin(existing_ids)].copy()
        print(
            f"  After removing {len(existing_ids)} already extracted: {len(no_yew)}")

    # Sort by distance to city (farthest first if available)
    if 'distance_to_city' in no_yew.columns:
        no_yew = no_yew.sort_values(
            'distance_to_city', ascending=False).reset_index(drop=True)
    else:
        no_yew = no_yew.sample(frac=1, random_state=42).reset_index(drop=True)

    # Limit to requested number
    # Get 2x in case of extraction failures
    no_yew = no_yew.head(n_sites * 2).copy()

    print(f"\n✓ Available non-yew sites: {len(no_yew)}")
    return no_yew


def extract_no_yew_site(site, composite, output_dir):
    """Extract image for a single non-yew site."""
    site_id = site['SITE_IDENTIFIER']
    lat = site['latitude']
    lon = site['longitude']

    try:
        # Define region (64x64 pixels at 10m resolution = 640m x 640m)
        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(320).bounds()

        # Select bands and reproject
        image = composite.select(['B2', 'B3', 'B4', 'B8']) \
            .reproject('EPSG:4326', scale=10)

        # Sample as array
        sample = image.sampleRectangle(region=region, defaultValue=0)

        # Get arrays
        arrays = {
            'B2': np.array(sample.get('B2').getInfo()),
            'B3': np.array(sample.get('B3').getInfo()),
            'B4': np.array(sample.get('B4').getInfo()),
            'B8': np.array(sample.get('B8').getInfo())
        }

        # Stack into (4, H, W) array
        stacked = np.stack([arrays['B2'], arrays['B3'],
                           arrays['B4'], arrays['B8']], axis=0)

        # Resize/crop to exactly 64x64
        if stacked.shape[1] != 64 or stacked.shape[2] != 64:
            from scipy.ndimage import zoom
            factors = (1, 64/stacked.shape[1], 64/stacked.shape[2])
            stacked = zoom(stacked, factors, order=1)

        # Save image
        output_file = output_dir / f'{site_id}.npy'
        np.save(output_file, stacked.astype(np.float32))

        # Return metadata
        metadata = {
            'site_identifier': site_id,
            'has_yew': False,
            'source': 'BC_Forestry',
            'bec_zone': site.get('BEC_ZONE'),
            'bec_subzone': site.get('BEC_SUBZONE'),
            'lon': lon,
            'lat': lat,
            'image_path': f'no_yew/{site_id}.npy',
            'image_shape': stacked.shape,
            'num_source_images': 'composite',
            'extraction_date': datetime.now().strftime('%Y-%m-%d')
        }

        # Add city info if available
        if 'nearest_city' in site and pd.notna(site['nearest_city']):
            metadata['nearest_city'] = site['nearest_city']
            metadata['distance_to_city'] = site['distance_to_city']

        return metadata

    except Exception as e:
        print(f"\n✗ Failed to extract site {site_id}: {e}")
        return None


# ============================================================================
# PARALLEL EXTRACTION
# ============================================================================

def extract_parallel(observations, composite, output_dir, limit, workers=6, dataset_type='yew'):
    """
    Extract multiple observations in parallel.

    Args:
        observations: DataFrame of observations to extract
        composite: Sentinel-2 composite image
        output_dir: Output directory for images
        limit: Maximum number to extract
        workers: Number of parallel workers
        dataset_type: 'yew' or 'no_yew'
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Limit observations
    obs_to_extract = observations.head(limit)

    print(
        f"\nExtracting {len(obs_to_extract)} {dataset_type} images with {workers} workers...")
    print("This may take a while (2-3 seconds per image)...")

    metadata_list = []
    failed_count = 0

    # Choose extraction function
    extract_func = extract_yew_observation if dataset_type == 'yew' else extract_no_yew_site

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all jobs
        future_to_obs = {
            executor.submit(extract_func, row, composite, output_dir): idx
            for idx, row in obs_to_extract.iterrows()
        }

        # Process as completed with progress bar
        with tqdm(total=len(future_to_obs), desc=f"Extracting {dataset_type}") as pbar:
            for future in as_completed(future_to_obs):
                try:
                    metadata = future.result()
                    if metadata:
                        metadata_list.append(metadata)
                    else:
                        failed_count += 1
                except Exception as e:
                    failed_count += 1
                    print(f"\n✗ Extraction error: {e}")
                finally:
                    pbar.update(1)

    print(f"\n✓ Successfully extracted: {len(metadata_list)} images")
    if failed_count > 0:
        print(f"✗ Failed extractions: {failed_count}")

    return metadata_list


# ============================================================================
# METADATA MANAGEMENT
# ============================================================================

def save_metadata(metadata_list, output_file, merge_existing=True):
    """Save or update metadata file."""
    new_df = pd.DataFrame(metadata_list)

    output_file = Path(output_file)

    if merge_existing and output_file.exists():
        print(f"\n  Merging with existing metadata...")
        existing_df = pd.read_csv(output_file)

        # Determine ID column based on dataset
        if 'observation_id' in new_df.columns:
            id_col = 'observation_id'
        else:
            id_col = 'site_identifier'

        # Combine and remove duplicates
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=[id_col], keep='last')
    else:
        combined_df = new_df

    # Sort
    if 'observation_id' in combined_df.columns:
        combined_df = combined_df.sort_values(
            'observation_id').reset_index(drop=True)
    else:
        combined_df = combined_df.sort_values(
            'site_identifier').reset_index(drop=True)

    # Save
    combined_df.to_csv(output_file, index=False)

    print(f"✓ Updated metadata: {output_file}")
    print(f"  New samples: {len(new_df)}")
    print(f"  Total samples: {len(combined_df)}")

    return combined_df


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Extract large balanced dataset (1000 yew + 1000 non-yew)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract 1000 of each with city filtering
    python scripts/preprocessing/extract_large_balanced_dataset.py \\
        --yew-limit 1000 \\
        --no-yew-limit 1000 \\
        --filter-cities \\
        --workers 6
    
    # Extract without city filtering
    python scripts/preprocessing/extract_large_balanced_dataset.py \\
        --yew-limit 1000 \\
        --no-yew-limit 1000 \\
        --workers 8
    
    # Extract with stricter accuracy requirement
    python scripts/preprocessing/extract_large_balanced_dataset.py \\
        --yew-limit 1000 \\
        --no-yew-limit 1000 \\
        --max-accuracy 30 \\
        --filter-cities
        """
    )

    parser.add_argument('--yew-limit', type=int, default=1000,
                        help='Number of yew samples to extract (default: 1000)')
    parser.add_argument('--no-yew-limit', type=int, default=1000,
                        help='Number of non-yew samples to extract (default: 1000)')
    parser.add_argument('--max-accuracy', type=float, default=50,
                        help='Maximum GPS accuracy in meters for yew (default: 50)')
    parser.add_argument('--filter-cities', action='store_true',
                        help='Exclude observations near major NW cities (RECOMMENDED)')
    parser.add_argument('--workers', type=int, default=6,
                        help='Number of parallel workers (default: 6)')
    parser.add_argument('--skip-yew', action='store_true',
                        help='Skip yew extraction (only extract non-yew)')
    parser.add_argument('--skip-no-yew', action='store_true',
                        help='Skip non-yew extraction (only extract yew)')

    args = parser.parse_args()

    print("="*80)
    print("EXTRACT LARGE BALANCED DATASET")
    print("="*80)
    print(
        f"Target: {args.yew_limit} yew + {args.no_yew_limit} non-yew samples")
    print(f"City filtering: {'ENABLED' if args.filter_cities else 'DISABLED'}")
    print(f"Workers: {args.workers}")
    print("="*80)

    # Initialize Earth Engine
    if not initialize_earth_engine():
        return

    # Create composite
    print("\nCreating Sentinel-2 composite (2020-2024)...")
    composite = create_sentinel2_composite()
    print("✓ Composite ready")

    # Setup output directories
    base_dir = Path('data/ee_imagery/image_patches_64x64')
    yew_dir = base_dir / 'inat_yew'
    no_yew_dir = base_dir / 'no_yew'
    yew_dir.mkdir(parents=True, exist_ok=True)
    no_yew_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # EXTRACT YEW OBSERVATIONS
    # ========================================================================

    if not args.skip_yew:
        yew_observations = load_yew_observations(
            max_accuracy=args.max_accuracy,
            filter_cities=args.filter_cities
        )

        if len(yew_observations) < args.yew_limit:
            print(
                f"\n⚠ Warning: Only {len(yew_observations)} yew observations available")
            print(f"  Requested: {args.yew_limit}")
            print(f"  Will extract all available samples")
            actual_yew_limit = len(yew_observations)
        else:
            actual_yew_limit = args.yew_limit

        if len(yew_observations) > 0:
            yew_metadata = extract_parallel(
                yew_observations,
                composite,
                yew_dir,
                limit=actual_yew_limit,
                workers=args.workers,
                dataset_type='yew'
            )

            if yew_metadata:
                yew_df = save_metadata(
                    yew_metadata,
                    base_dir / 'inat_yew_image_metadata.csv',
                    merge_existing=True
                )
            else:
                print("\n✗ No yew images were successfully extracted")
        else:
            print("\n✗ No yew observations available to extract")

    # ========================================================================
    # EXTRACT NON-YEW SITES
    # ========================================================================

    if not args.skip_no_yew:
        no_yew_sites = load_no_yew_sites(
            n_sites=args.no_yew_limit,
            filter_cities=args.filter_cities
        )

        if len(no_yew_sites) < args.no_yew_limit:
            print(
                f"\n⚠ Warning: Only {len(no_yew_sites)} non-yew sites available")
            print(f"  Requested: {args.no_yew_limit}")
            print(f"  Will extract all available samples")
            actual_no_yew_limit = len(no_yew_sites)
        else:
            actual_no_yew_limit = args.no_yew_limit

        if len(no_yew_sites) > 0:
            no_yew_metadata = extract_parallel(
                no_yew_sites,
                composite,
                no_yew_dir,
                limit=actual_no_yew_limit,
                workers=args.workers,
                dataset_type='no_yew'
            )

            if no_yew_metadata:
                no_yew_df = save_metadata(
                    no_yew_metadata,
                    base_dir / 'no_yew_image_metadata.csv',
                    merge_existing=True
                )
            else:
                print("\n✗ No non-yew images were successfully extracted")
        else:
            print("\n✗ No non-yew sites available to extract")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    print("\n" + "="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)

    # Count current totals
    yew_meta_file = base_dir / 'inat_yew_image_metadata.csv'
    no_yew_meta_file = base_dir / 'no_yew_image_metadata.csv'

    if yew_meta_file.exists():
        yew_total = len(pd.read_csv(yew_meta_file))
        print(f"Total yew samples: {yew_total}")

    if no_yew_meta_file.exists():
        no_yew_total = len(pd.read_csv(no_yew_meta_file))
        print(f"Total non-yew samples: {no_yew_total}")

    if yew_meta_file.exists() and no_yew_meta_file.exists():
        print(f"Dataset balance: {yew_total} yew : {no_yew_total} non-yew")
        print(f"Class ratio: {yew_total/(yew_total+no_yew_total):.1%} yew")

    print("\nAll samples filtered to exclude California (lat >= 42°N)")
    if args.filter_cities:
        print("All samples filtered to exclude major NW cities")

    print("\nNext steps:")
    print("  1. Review samples: python scripts/visualization/review_all_images.py --dataset all")
    print("  2. Create splits: python scripts/preprocessing/create_filtered_splits.py")
    print("  3. Train model: python scripts/training/train_cnn.py --use-filtered")
    print("="*80)


if __name__ == '__main__':
    main()
