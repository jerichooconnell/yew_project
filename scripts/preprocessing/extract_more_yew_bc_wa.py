#!/usr/bin/env python3
"""
Extract Additional Yew Images (BC/WA Only)
===========================================

Extract more iNaturalist yew observations from BC and Washington,
excluding California samples (lat < 42°N).

Author: GitHub Copilot
Date: November 14, 2025
"""

from preprocessing.city_filter import filter_dataframe
import pandas as pd
import numpy as np
import ee
from pathlib import Path
import time
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


def initialize_earth_engine():
    """Initialize Earth Engine."""
    try:
        ee.Initialize(project='carbon-storm-206002')
        print("✓ Earth Engine initialized")
        return True
    except:
        print("✗ Earth Engine initialization failed")
        return False


def load_bc_wa_observations(max_accuracy=50, min_lat=42.0):
    """
    Load iNaturalist yew observations from BC and Washington.

    Args:
        max_accuracy: Maximum GPS positional accuracy in meters
        min_lat: Minimum latitude (42.0 excludes California)

    Returns:
        DataFrame of filtered observations
    """
    print("\nLoading iNaturalist observations...")

    inat_df = pd.read_csv('data/inat_observations/observations-558049.csv')
    print(f"  Total observations: {len(inat_df)}")

    # Filter by positional accuracy
    accurate = inat_df[inat_df['positional_accuracy'] <= max_accuracy].copy()
    print(f"  With accuracy <= {max_accuracy}m: {len(accurate)}")

    # Filter by latitude (exclude California)
    bc_wa = accurate[accurate['latitude'] >= min_lat].copy()
    print(f"  BC/WA only (lat >= {min_lat}°N): {len(bc_wa)}")

    # Filter out already extracted
    existing_meta = Path(
        'data/ee_imagery/image_patches_64x64/inat_yew_image_metadata.csv')
    if existing_meta.exists():
        existing_df = pd.read_csv(existing_meta)
        existing_ids = set(existing_df['observation_id'].values)
        bc_wa = bc_wa[~bc_wa['id'].isin(existing_ids)]
        print(
            f"  After removing {len(existing_ids)} already extracted: {len(bc_wa)}")

    # Sort by accuracy (best first)
    bc_wa = bc_wa.sort_values('positional_accuracy').reset_index(drop=True)

    return bc_wa


def create_sentinel2_composite():
    """Create cloud-free Sentinel-2 composite."""
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate('2020-01-01', '2024-12-31') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

    def mask_clouds(image):
        qa = image.select('QA60')
        cloud_mask = qa.bitwiseAnd(1 << 10).eq(0).And(
            qa.bitwiseAnd(1 << 11).eq(0))
        return image.updateMask(cloud_mask)

    return s2.map(mask_clouds).median()


def extract_single_observation(obs, composite):
    """Extract image for a single observation."""
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

        # Sample the region
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

        # Validate
        if img.size == 0 or img.ndim != 3:
            return None

        # Check for spatial variation
        if all(img[i].std() == 0 for i in range(4)):
            return None

        # Resize to 64x64 if needed
        if img.shape[1] != 64 or img.shape[2] != 64:
            from scipy.ndimage import zoom
            factors = (1, 64/img.shape[1], 64/img.shape[2])
            img = zoom(img, factors, order=1)

        # Save
        output_dir = Path('data/ee_imagery/image_patches_64x64/inat_yew')
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f'inat_{obs_id}.npy'
        np.save(output_path, img.astype(np.float32))

        # Metadata
        metadata = {
            'observation_id': obs_id,
            'has_yew': True,
            'source': 'iNaturalist',
            'observation_year': pd.to_datetime(obs['observed_on']).year if pd.notna(obs['observed_on']) else None,
            'positional_accuracy': obs['positional_accuracy'],
            'lon': lon,
            'lat': lat,
            'image_path': f'inat_yew/inat_{obs_id}.npy',
            'image_shape': str(img.shape),
            'num_source_images': None,
            'extraction_date': datetime.now().isoformat()
        }

        return metadata

    except Exception as e:
        print(f"Error extracting {obs_id}: {e}")
        return None


def extract_parallel(observations, limit=None, workers=6):
    """Extract images in parallel."""
    if limit:
        observations = observations.head(limit)

    print(
        f"\nExtracting {len(observations)} observations with {workers} workers...")
    print("This may take a while (2-3 seconds per image)...")

    composite = create_sentinel2_composite()

    metadata_list = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(extract_single_observation, row, composite): idx
                   for idx, row in observations.iterrows()}

        with tqdm(total=len(futures), desc="Extracting") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    metadata_list.append(result)
                pbar.update(1)
                time.sleep(0.1)  # Rate limiting

    return metadata_list


def update_metadata(new_metadata_list):
    """Update or create the metadata file."""
    output_file = Path(
        'data/ee_imagery/image_patches_64x64/inat_yew_image_metadata.csv')

    new_df = pd.DataFrame(new_metadata_list)

    # Combine with existing if it exists
    if output_file.exists():
        existing_df = pd.read_csv(output_file)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        # Remove duplicates
        combined_df = combined_df.drop_duplicates(
            subset=['observation_id'], keep='last')
    else:
        combined_df = new_df

    # Sort by observation_id
    combined_df = combined_df.sort_values(
        'observation_id').reset_index(drop=True)

    # Save
    combined_df.to_csv(output_file, index=False)

    print(f"\n✓ Updated metadata with {len(new_df)} new samples")
    print(f"✓ Total yew samples: {len(combined_df)}")
    print(f"✓ Saved to: {output_file}")

    return combined_df


def main():
    """Main extraction pipeline."""
    import argparse
    parser = argparse.ArgumentParser(
        description='Extract more yew images from BC/WA')
    parser.add_argument('--limit', type=int, default=100,
                        help='Number of additional samples to extract')
    parser.add_argument('--max-accuracy', type=float, default=50,
                        help='Maximum GPS accuracy in meters')
    parser.add_argument('--workers', type=int, default=6,
                        help='Number of parallel workers')
    parser.add_argument('--filter-cities', action='store_true',
                        help='Exclude observations near major NW cities')
    parser.add_argument('--min-city-distance', type=float, default=None,
                        help='Minimum distance from cities in km (uses city-specific defaults if not set)')
    args = parser.parse_args()

    print("="*80)
    print("EXTRACT ADDITIONAL YEW IMAGES (BC/WA ONLY)")
    print("="*80)

    if not initialize_earth_engine():
        return

    # Load observations (BC/WA only, excludes California)
    observations = load_bc_wa_observations(
        max_accuracy=args.max_accuracy,
        min_lat=42.0  # Excludes California
    )

    # Apply city filter if requested
    if args.filter_cities:
        print("\nApplying city filter...")
        observations = filter_dataframe(
            observations,
            lat_col='latitude',
            lon_col='longitude',
            min_distance_km=args.min_city_distance,
            add_city_info=True
        )
        print(f"  Remaining after city filter: {len(observations)}")

    if len(observations) == 0:
        print("\n✗ No new observations to extract!")
        return

    # Extract images
    metadata_list = extract_parallel(
        observations,
        limit=args.limit,
        workers=args.workers
    )

    if metadata_list:
        # Update metadata file
        final_df = update_metadata(metadata_list)

        print("\n" + "="*80)
        print("EXTRACTION COMPLETE")
        print("="*80)
        print(f"Successfully extracted: {len(metadata_list)} samples")
        print(f"Total BC/WA yew samples: {len(final_df)}")
        print(f"All samples lat >= 42°N (no California)")
        print("="*80)
    else:
        print("\n✗ No images were successfully extracted")


if __name__ == '__main__':
    main()
