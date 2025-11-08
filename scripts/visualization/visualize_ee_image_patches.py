#!/usr/bin/env python3
"""
Visualize Earth Engine Image Patches
====================================

Extract and visualize 64x64 pixel Sentinel-2 image patches for yew and non-yew sites
directly from Earth Engine.

Author: GitHub Copilot
Date: November 7, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import ee
from tqdm import tqdm
import time


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


def get_sentinel2_image(lon, lat, year, patch_size=64):
    """
    Extract a Sentinel-2 image patch at given location.

    Args:
        lon: Longitude
        lat: Latitude
        year: Year for imagery
        patch_size: Size of patch in pixels (default 64)

    Returns:
        Dictionary with band arrays or None if failed
    """
    try:
        # Define the point
        point = ee.Geometry.Point([lon, lat])

        # Define the region (buffer around point)
        # Sentinel-2 is 10m resolution, so 64 pixels = 640m
        # Buffer by 320m to get 64x64 patch
        scale = 10  # meters per pixel
        buffer_distance = (patch_size * scale) / 2
        region = point.buffer(buffer_distance).bounds()

        # Get imagery for the specified year
        start_date = f'{year}-01-01'
        end_date = f'{year}-12-31'

        # Load Sentinel-2 surface reflectance data
        s2 = ee.ImageCollection('COPERNICUS/S2_SR') \
            .filterDate(start_date, end_date) \
            .filterBounds(point) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
            .map(lambda img: img.select(['B2', 'B3', 'B4', 'B8'])) \
            .median()

        # Sample the image to get pixel arrays
        # Use getThumbURL to get the actual image patch
        vis_params = {
            'dimensions': [patch_size, patch_size],
            'region': region,
            'format': 'png'
        }

        # Get individual bands as arrays
        sample = s2.sampleRectangle(region=region, defaultValue=0)

        # Extract band arrays
        blue = sample.get('B2').getInfo()
        green = sample.get('B3').getInfo()
        red = sample.get('B4').getInfo()
        nir = sample.get('B8').getInfo()

        return {
            'blue': np.array(blue),
            'green': np.array(green),
            'red': np.array(red),
            'nir': np.array(nir)
        }

    except Exception as e:
        print(f"  Error extracting image: {str(e)[:50]}")
        return None


def normalize_band(band_array, percentile_clip=2):
    """Normalize band values for visualization (0-255)."""
    if band_array is None or band_array.size == 0:
        return np.zeros((64, 64), dtype=np.uint8)

    # Clip extreme values
    p_low = np.percentile(band_array, percentile_clip)
    p_high = np.percentile(band_array, 100 - percentile_clip)

    # Normalize to 0-255
    band_clipped = np.clip(band_array, p_low, p_high)
    band_norm = ((band_clipped - p_low) /
                 (p_high - p_low) * 255).astype(np.uint8)

    return band_norm


def create_rgb_composite(bands_dict):
    """Create RGB composite from bands dictionary."""
    red = normalize_band(bands_dict['red'])
    green = normalize_band(bands_dict['green'])
    blue = normalize_band(bands_dict['blue'])

    rgb = np.dstack([red, green, blue])
    return rgb


def create_false_color_composite(bands_dict):
    """Create false color (NIR-Red-Green) composite."""
    nir = normalize_band(bands_dict['nir'])
    red = normalize_band(bands_dict['red'])
    green = normalize_band(bands_dict['green'])

    false_color = np.dstack([nir, red, green])
    return false_color


def load_sample_sites():
    """Load sample of yew and non-yew sites."""
    print("Loading site data...")

    # Load inventory data
    inv_path = Path('data/processed/bc_sample_data_deduplicated.csv')
    df = pd.read_csv(inv_path, low_memory=False)

    # Parse yew presence
    import re

    def parse_yew(composition_string):
        if not composition_string or pd.isna(composition_string):
            return False
        pattern = r'TW(\d{2,3})'
        match = re.search(pattern, str(composition_string))
        return match is not None

    df['has_yew'] = df['SPB_CPCT_LS'].apply(parse_yew)

    # Get sites with coordinates
    df_with_coords = df[df['POINT_X'].notna() & df['POINT_Y'].notna()].copy()

    # Convert BC Albers to Lat/Lon (approximate)
    # For better results, should use proper projection, but this is quick approximation
    # BC Albers EPSG:3005 to WGS84 EPSG:4326
    from pyproj import Transformer
    transformer = Transformer.from_crs(
        "EPSG:3005", "EPSG:4326", always_xy=True)

    coords = transformer.transform(df_with_coords['POINT_X'].values,
                                   df_with_coords['POINT_Y'].values)
    df_with_coords['lon'] = coords[0]
    df_with_coords['lat'] = coords[1]

    # Sample sites
    yew_sites = df_with_coords[df_with_coords['has_yew'] == True].sample(n=min(
        10, len(df_with_coords[df_with_coords['has_yew'] == True])), random_state=42)
    no_yew_sites = df_with_coords[df_with_coords['has_yew'] == False].sample(n=min(
        10, len(df_with_coords[df_with_coords['has_yew'] == False])), random_state=42)

    print(
        f"  Selected {len(yew_sites)} yew sites and {len(no_yew_sites)} non-yew sites")

    return yew_sites, no_yew_sites


def extract_and_visualize(yew_sites, no_yew_sites, output_dir, n_samples=5):
    """Extract images and create visualizations."""

    print(f"\nExtracting {n_samples} yew site images...")
    yew_images = []
    yew_metadata = []

    for idx, (_, row) in enumerate(yew_sites.head(n_samples).iterrows()):
        print(f"  Extracting yew site {idx+1}/{n_samples}...")
        year = int(row.get('MEASUREMENT_YEAR', 2020))
        bands = get_sentinel2_image(row['lon'], row['lat'], year)

        if bands is not None:
            rgb = create_rgb_composite(bands)
            false_color = create_false_color_composite(bands)

            yew_images.append({
                'rgb': rgb,
                'false_color': false_color,
                'bands': bands
            })
            yew_metadata.append({
                'site_id': row['SITE_IDENTIFIER'],
                'year': year,
                'elevation': row.get('ELEVATION', None)
            })

        time.sleep(0.5)  # Rate limiting

    print(f"\nExtracting {n_samples} non-yew site images...")
    no_yew_images = []
    no_yew_metadata = []

    for idx, (_, row) in enumerate(no_yew_sites.head(n_samples).iterrows()):
        print(f"  Extracting non-yew site {idx+1}/{n_samples}...")
        year = int(row.get('MEASUREMENT_YEAR', 2020))
        bands = get_sentinel2_image(row['lon'], row['lat'], year)

        if bands is not None:
            rgb = create_rgb_composite(bands)
            false_color = create_false_color_composite(bands)

            no_yew_images.append({
                'rgb': rgb,
                'false_color': false_color,
                'bands': bands
            })
            no_yew_metadata.append({
                'site_id': row['SITE_IDENTIFIER'],
                'year': year,
                'elevation': row.get('ELEVATION', None)
            })

        time.sleep(0.5)

    print(
        f"\n✓ Extracted {len(yew_images)} yew and {len(no_yew_images)} non-yew images")

    # Create visualizations
    create_comparison_grid(yew_images, no_yew_images,
                           yew_metadata, no_yew_metadata, output_dir)
    create_band_analysis(yew_images, no_yew_images, output_dir)


def create_comparison_grid(yew_images, no_yew_images, yew_meta, no_yew_meta, output_dir):
    """Create comparison grid of RGB and false color images."""
    n_samples = min(len(yew_images), len(no_yew_images), 5)

    if n_samples == 0:
        print("Not enough images for visualization")
        return

    # RGB comparison
    fig, axes = plt.subplots(2, n_samples, figsize=(3*n_samples, 7))

    for i in range(n_samples):
        # Yew sites
        axes[0, i].imshow(yew_images[i]['rgb'])
        axes[0, i].axis('off')
        axes[0, i].set_title(f"YEW SITE\n{yew_meta[i]['site_id']}",
                             fontsize=10, color='green', fontweight='bold')

        # Non-yew sites
        axes[1, i].imshow(no_yew_images[i]['rgb'])
        axes[1, i].axis('off')
        axes[1, i].set_title(f"NON-YEW SITE\n{no_yew_meta[i]['site_id']}",
                             fontsize=10)

    plt.suptitle('Sentinel-2 True Color: Yew vs Non-Yew Sites (64x64 pixels)',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    save_path = output_dir / 'ee_patches_true_color.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

    # False color comparison
    fig, axes = plt.subplots(2, n_samples, figsize=(3*n_samples, 7))

    for i in range(n_samples):
        # Yew sites
        axes[0, i].imshow(yew_images[i]['false_color'])
        axes[0, i].axis('off')
        axes[0, i].set_title(f"YEW SITE\n{yew_meta[i]['site_id']}",
                             fontsize=10, color='green', fontweight='bold')

        # Non-yew sites
        axes[1, i].imshow(no_yew_images[i]['false_color'])
        axes[1, i].axis('off')
        axes[1, i].set_title(f"NON-YEW SITE\n{no_yew_meta[i]['site_id']}",
                             fontsize=10)

    plt.suptitle('Sentinel-2 False Color (NIR-R-G): Yew vs Non-Yew Sites (64x64 pixels)',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    save_path = output_dir / 'ee_patches_false_color.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def create_band_analysis(yew_images, no_yew_images, output_dir):
    """Create spectral band analysis plots."""
    if len(yew_images) == 0 or len(no_yew_images) == 0:
        return

    # Compute average spectral signatures
    bands = ['blue', 'green', 'red', 'nir']
    yew_means = {band: [] for band in bands}
    no_yew_means = {band: [] for band in bands}

    for img in yew_images:
        for band in bands:
            yew_means[band].append(np.mean(img['bands'][band]))

    for img in no_yew_images:
        for band in bands:
            no_yew_means[band].append(np.mean(img['bands'][band]))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(bands))
    yew_avg = [np.mean(yew_means[band]) for band in bands]
    no_yew_avg = [np.mean(no_yew_means[band]) for band in bands]
    yew_std = [np.std(yew_means[band]) for band in bands]
    no_yew_std = [np.std(no_yew_means[band]) for band in bands]

    width = 0.35
    ax.bar(x - width/2, yew_avg, width, yerr=yew_std, label='Yew Sites',
           color='green', alpha=0.7, capsize=5)
    ax.bar(x + width/2, no_yew_avg, width, yerr=no_yew_std, label='Non-Yew Sites',
           color='gray', alpha=0.7, capsize=5)

    ax.set_xlabel('Spectral Band', fontsize=12)
    ax.set_ylabel('Mean Reflectance Value', fontsize=12)
    ax.set_title('Spectral Signatures: Yew vs Non-Yew Forest Sites',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_path = output_dir / 'ee_patches_spectral_analysis.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def main():
    """Main execution."""
    print("=" * 80)
    print("EARTH ENGINE IMAGE PATCH EXTRACTION AND VISUALIZATION")
    print("=" * 80)

    # Initialize Earth Engine
    print("\nInitializing Earth Engine...")
    if not initialize_earth_engine():
        print("\n✗ Cannot proceed without Earth Engine")
        return

    # Create output directory
    output_dir = Path('results/figures/ee_thumbnails')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load sites
    try:
        yew_sites, no_yew_sites = load_sample_sites()
    except ImportError:
        print("\n✗ pyproj library required for coordinate transformation")
        print("  Install with: pip install pyproj")
        return
    except Exception as e:
        print(f"\n✗ Error loading sites: {e}")
        return

    # Extract and visualize
    extract_and_visualize(yew_sites, no_yew_sites, output_dir, n_samples=5)

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print(f"Images saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
