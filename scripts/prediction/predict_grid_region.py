#!/usr/bin/env python3
"""
Extract images from a grid and predict yew probability for mapping.

This script:
1. Creates a regular grid of points across a region
2. Extracts satellite imagery for each point
3. Runs the trained model to predict yew probability
4. Creates a spatial map of predictions

Usage:
    python scripts/prediction/predict_grid_region.py \
        --model models/checkpoints/resnet18_20251119_121738_best.pth \
        --lat-min 48.44 --lat-max 48.47 \
        --lon-min -124.11 --lon-max -124.00 \
        --grid-spacing 0.001 \
        --output results/predictions/southern_vancouver_island

Author: GitHub Copilot
Date: November 19, 2025
"""

from training.model import ResNet4Channel
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import ee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def initialize_earth_engine():
    """Initialize Earth Engine."""
    try:
        ee.Initialize(project='carbon-storm-206002')
        print("✓ Earth Engine initialized")
    except Exception as e:
        print(f"✗ Earth Engine initialization failed: {e}")
        print("  Run: earthengine authenticate")
        sys.exit(1)


def create_grid(lat_min, lat_max, lon_min, lon_max, spacing):
    """Create a regular grid of points."""
    lats = np.arange(lat_min, lat_max, spacing)
    lons = np.arange(lon_min, lon_max, spacing)

    grid_points = []
    for lat in lats:
        for lon in lons:
            grid_points.append({
                'latitude': lat,
                'longitude': lon,
                'point_id': f"{lat:.6f}_{lon:.6f}"
            })

    return pd.DataFrame(grid_points)


def extract_sentinel2_patch(lat, lon, patch_size=64, scale=10, cache_dir=None):
    """
    Extract a single Sentinel-2 patch centered on coordinates.

    Args:
        lat, lon: Center coordinates
        patch_size: Size of patch in pixels
        scale: Pixel resolution in meters (10m for Sentinel-2)
        cache_dir: Directory to cache extracted tiles (optional)

    Returns:
        numpy array of shape (4, patch_size, patch_size) or None if failed
    """
    try:
        # Check cache first
        if cache_dir is not None:
            cache_path = Path(cache_dir) / f"tile_{lat:.6f}_{lon:.6f}.npy"
            if cache_path.exists():
                try:
                    return np.load(cache_path)
                except:
                    pass  # If cached file is corrupted, re-extract

        # Create point
        point = ee.Geometry.Point([lon, lat])

        # Define region around point
        half_size = (patch_size * scale) / 2
        region = point.buffer(half_size).bounds()

        # Get most recent cloud-free Sentinel-2 image
        s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterBounds(point)
              .filterDate('2023-01-01', '2024-12-31')
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
              .sort('CLOUDY_PIXEL_PERCENTAGE')
              .first())

        if s2 is None:
            return None

        # Select bands: B2 (Blue), B3 (Green), B4 (Red), B8 (NIR)
        bands = s2.select(['B2', 'B3', 'B4', 'B8'])

        # Get as numpy array
        url = bands.getDownloadURL({
            'region': region,
            'dimensions': [patch_size, patch_size],
            'format': 'NPY'
        })

        # Download and load
        import requests
        response = requests.get(url)
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
                # Structured array with named fields (B2, B3, B4, B8)
                bands = ['B2', 'B3', 'B4', 'B8']
                data_list = [data[band] for band in bands]
                data = np.stack(data_list, axis=0)  # (4, 64, 64)
            elif data.ndim == 3:
                # Regular array - transpose to (channels, height, width)
                data = np.transpose(data, (2, 0, 1))

            # Ensure correct shape
            if data.shape != (4, 64, 64):
                return None

            # Save to cache
            if cache_dir is not None and data is not None:
                cache_path = Path(cache_dir) / f"tile_{lat:.6f}_{lon:.6f}.npy"
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, data)

            return data

        return None

    except Exception as e:
        return None


def preprocess_image(image):
    """Preprocess image for model input."""
    if image is None:
        return None

    # Ensure correct shape (4, 64, 64)
    if image.shape != (4, 64, 64):
        return None

    # Convert to (H, W, C) for preprocessing
    img = np.transpose(image, (1, 2, 0))  # (64, 64, 4)

    # Normalize to [0, 1] using percentile clipping
    img_normalized = np.zeros_like(img, dtype=np.float32)
    for i in range(4):
        band = img[:, :, i]
        p_low = np.percentile(band, 2)
        p_high = np.percentile(band, 98)
        band_clipped = np.clip(band, p_low, p_high)
        band_min = band_clipped.min()
        band_max = band_clipped.max()
        if band_max > band_min:
            img_normalized[:, :, i] = (
                band_clipped - band_min) / (band_max - band_min)
        else:
            img_normalized[:, :, i] = 0.0

    # Convert to tensor and add batch dimension
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)

    return img_tensor


def load_model(model_path, device):
    """Load trained model."""
    print(f"Loading model from: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    architecture = checkpoint.get('architecture', 'resnet18')

    model = ResNet4Channel(architecture=architecture, num_classes=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"  Architecture: {architecture}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")

    return model


def predict_batch(model, images, device):
    """Run prediction on a batch of images."""
    if len(images) == 0:
        return []

    # Stack images
    batch = torch.cat(images, dim=0).to(device)

    with torch.no_grad():
        outputs = model(batch)
        probs = F.softmax(outputs, dim=1)
        yew_probs = probs[:, 1].cpu().numpy()

    return yew_probs


def create_satellite_overlay(df, output_dir, cache_dir):
    """Create satellite imagery with probability heatmap overlay."""
    print("\nCreating satellite imagery overlay...")

    cache_dir = Path(cache_dir)

    # Get bounds
    lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
    lon_min, lon_max = df['longitude'].min(), df['longitude'].max()

    # Create a mosaic from cached tiles
    # Sort by latitude (descending) and longitude (ascending) for proper arrangement
    df_sorted = df.sort_values(
        ['latitude', 'longitude'], ascending=[False, True])

    # Determine grid dimensions
    unique_lats = sorted(df['latitude'].unique(), reverse=True)
    unique_lons = sorted(df['longitude'].unique())
    n_rows = len(unique_lats)
    n_cols = len(unique_lons)

    # Initialize mosaic array
    tile_size = 64
    mosaic = np.zeros((n_rows * tile_size, n_cols *
                      tile_size, 3), dtype=np.uint8)
    prob_grid = np.zeros((n_rows, n_cols))

    # Load tiles and create mosaic
    for i, lat in enumerate(unique_lats):
        for j, lon in enumerate(unique_lons):
            # Find corresponding tile
            tile_path = cache_dir / f"tile_{lat:.6f}_{lon:.6f}.npy"

            if tile_path.exists():
                try:
                    # Load tile
                    tile = np.load(tile_path)  # Shape: (4, 64, 64)

                    # Convert to RGB (bands 2,1,0 = R,G,B)
                    rgb = tile[[2, 1, 0], :, :]  # R, G, B
                    rgb = np.transpose(rgb, (1, 2, 0))  # (64, 64, 3)

                    # Normalize for display (percentile clipping)
                    rgb_norm = np.zeros_like(rgb, dtype=np.float32)
                    for band in range(3):
                        b = rgb[:, :, band]
                        p_low, p_high = np.percentile(b, [2, 98])
                        b_clipped = np.clip(b, p_low, p_high)
                        if b_clipped.max() > b_clipped.min():
                            rgb_norm[:, :, band] = (
                                b_clipped - b_clipped.min()) / (b_clipped.max() - b_clipped.min())

                    # Convert to uint8
                    rgb_uint8 = (rgb_norm * 255).astype(np.uint8)

                    # Place in mosaic
                    row_start = i * tile_size
                    row_end = (i + 1) * tile_size
                    col_start = j * tile_size
                    col_end = (j + 1) * tile_size
                    mosaic[row_start:row_end, col_start:col_end] = rgb_uint8

                    # Get probability for this location
                    row_data = df[(df['latitude'] == lat) &
                                  (df['longitude'] == lon)]
                    if len(row_data) > 0:
                        prob_grid[i, j] = row_data.iloc[0]['yew_probability']

                except Exception as e:
                    pass  # Skip problematic tiles

    # Create figure with satellite base and probability overlay
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Left: Satellite imagery
    ax1.imshow(mosaic, extent=[lon_min, lon_max,
               lat_min, lat_max], aspect='auto')
    ax1.set_xlabel('Longitude', fontsize=12)
    ax1.set_ylabel('Latitude', fontsize=12)
    ax1.set_title('Sentinel-2 Satellite Imagery (True Color)',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Right: Overlay with transparency
    ax2.imshow(mosaic, extent=[lon_min, lon_max,
               lat_min, lat_max], aspect='auto')

    # Create probability heatmap overlay
    colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0',
              '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
    cmap = LinearSegmentedColormap.from_list('yew_prob', colors, N=100)

    # Overlay probability heatmap with transparency
    heatmap = ax2.imshow(prob_grid, extent=[lon_min, lon_max, lat_min, lat_max],
                         aspect='auto', cmap=cmap, alpha=0.6, vmin=0, vmax=1,
                         interpolation='bilinear')

    ax2.set_xlabel('Longitude', fontsize=12)
    ax2.set_ylabel('Latitude', fontsize=12)
    ax2.set_title('Yew Probability Overlay on Satellite Imagery',
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(heatmap, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Yew Probability', fontsize=12)

    # Add statistics
    stats_text = (
        f"Region: {lat_min:.4f}°N - {lat_max:.4f}°N\n"
        f"        {lon_min:.4f}°W - {lon_max:.4f}°W\n"
        f"Tiles: {n_rows} × {n_cols} = {len(df)}\n"
        f"Mean prob: {df['yew_probability'].mean():.3f}\n"
        f"High (>0.7): {(df['yew_probability'] > 0.7).sum()}"
    )
    ax2.text(0.02, 0.98, stats_text,
             transform=ax2.transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=10,
             family='monospace')

    overlay_path = output_dir / 'yew_satellite_overlay.png'
    plt.tight_layout()
    plt.savefig(overlay_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved overlay: {overlay_path}")

    # Also create a single high-res overlay image
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(mosaic, extent=[lon_min, lon_max,
              lat_min, lat_max], aspect='auto')
    heatmap = ax.imshow(prob_grid, extent=[lon_min, lon_max, lat_min, lat_max],
                        aspect='auto', cmap=cmap, alpha=0.5, vmin=0, vmax=1,
                        interpolation='bilinear')
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Latitude', fontsize=14)
    ax.set_title('Yew Probability Heatmap Over Satellite Imagery\nSouthern Vancouver Island',
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, color='white', linewidth=0.5)

    cbar = plt.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Probability of Yew Presence', fontsize=12)

    single_path = output_dir / 'yew_overlay_high_res.png'
    plt.tight_layout()
    plt.savefig(single_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved high-res overlay: {single_path}")


def create_probability_map(df, output_dir, cache_dir=None):
    """Create visualization of yew probability across the region."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to successful predictions
    df_pred = df[df['yew_probability'].notna()].copy()

    if len(df_pred) == 0:
        print("No successful predictions to map")
        return

    print(f"\nCreating probability map from {len(df_pred)} predictions...")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Custom colormap (blue = low probability, red = high probability)
    colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0',
              '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('yew_prob', colors, N=n_bins)

    # Create scatter plot
    scatter = ax.scatter(
        df_pred['longitude'],
        df_pred['latitude'],
        c=df_pred['yew_probability'],
        cmap=cmap,
        s=100,
        alpha=0.8,
        edgecolors='black',
        linewidths=0.5,
        vmin=0,
        vmax=1
    )

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Yew Probability')
    cbar.set_label('Probability of Yew Presence', fontsize=12)

    # Labels and title
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Predicted Yew Probability - Southern Vancouver Island',
                 fontsize=14, fontweight='bold')

    # Grid
    ax.grid(True, alpha=0.3)

    # Statistics text
    stats_text = (
        f"Total points: {len(df_pred)}\n"
        f"Mean probability: {df_pred['yew_probability'].mean():.3f}\n"
        f"High probability (>0.7): {(df_pred['yew_probability'] > 0.7).sum()}\n"
        f"Medium probability (0.3-0.7): {((df_pred['yew_probability'] >= 0.3) & (df_pred['yew_probability'] <= 0.7)).sum()}\n"
        f"Low probability (<0.3): {(df_pred['yew_probability'] < 0.3).sum()}"
    )
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=10,
            family='monospace')

    # Save
    map_path = output_dir / 'yew_probability_map.png'
    plt.tight_layout()
    plt.savefig(map_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved map: {map_path}")

    # Also create a high-probability hotspot map
    fig, ax = plt.subplots(figsize=(12, 10))

    # Separate into categories
    high = df_pred[df_pred['yew_probability'] > 0.7]
    medium = df_pred[(df_pred['yew_probability'] >= 0.3) &
                     (df_pred['yew_probability'] <= 0.7)]
    low = df_pred[df_pred['yew_probability'] < 0.3]

    # Plot by category
    if len(low) > 0:
        ax.scatter(low['longitude'], low['latitude'],
                   c='lightblue', s=50, alpha=0.5, label='Low (<0.3)')
    if len(medium) > 0:
        ax.scatter(medium['longitude'], medium['latitude'],
                   c='orange', s=100, alpha=0.7, label='Medium (0.3-0.7)')
    if len(high) > 0:
        ax.scatter(high['longitude'], high['latitude'],
                   c='red', s=150, alpha=0.9, label='High (>0.7)',
                   edgecolors='darkred', linewidths=2)

    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Yew Probability Hotspots - Southern Vancouver Island',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)

    hotspot_path = output_dir / 'yew_hotspots_map.png'
    plt.tight_layout()
    plt.savefig(hotspot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved hotspots: {hotspot_path}")

    # Create overlay with satellite imagery
    if cache_dir is not None:
        create_satellite_overlay(df_pred, output_dir, cache_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Extract grid and predict yew probability for mapping',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--lat-min', type=float, required=True,
                        help='Minimum latitude')
    parser.add_argument('--lat-max', type=float, required=True,
                        help='Maximum latitude')
    parser.add_argument('--lon-min', type=float, required=True,
                        help='Minimum longitude')
    parser.add_argument('--lon-max', type=float, required=True,
                        help='Maximum longitude')
    parser.add_argument('--grid-spacing', type=float, default=0.001,
                        help='Grid spacing in degrees (default: 0.001 = ~100m)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for prediction (default: 32)')
    parser.add_argument('--output', type=str,
                        default='results/predictions/grid_prediction',
                        help='Output directory')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable tile caching (re-extract all tiles)')

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize Earth Engine
    initialize_earth_engine()

    # Create grid
    print(f"\nCreating grid:")
    print(f"  Latitude: {args.lat_min} to {args.lat_max}")
    print(f"  Longitude: {args.lon_min} to {args.lon_max}")
    print(f"  Spacing: {args.grid_spacing}°")

    grid_df = create_grid(args.lat_min, args.lat_max,
                          args.lon_min, args.lon_max,
                          args.grid_spacing)

    print(f"  Total points: {len(grid_df)}")

    # Load model
    model = load_model(args.model, device)

    # Setup cache directory
    cache_dir = Path(args.output) / 'tile_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nTile cache: {cache_dir}")

    # Extract and predict
    print(f"\nExtracting imagery and predicting...")

    results = []
    batch_images = []
    batch_indices = []

    for idx, row in tqdm(grid_df.iterrows(), total=len(grid_df)):
        # Extract image
        image = extract_sentinel2_patch(
            row['latitude'], row['longitude'], cache_dir=cache_dir)

        if image is not None:
            # Preprocess
            img_tensor = preprocess_image(image)

            if img_tensor is not None:
                batch_images.append(img_tensor)
                batch_indices.append(idx)

                # Process batch
                if len(batch_images) >= args.batch_size:
                    probs = predict_batch(model, batch_images, device)

                    for batch_idx, prob in zip(batch_indices, probs):
                        results.append({
                            'grid_index': batch_idx,
                            'yew_probability': float(prob)
                        })

                    batch_images = []
                    batch_indices = []

    # Process remaining batch
    if len(batch_images) > 0:
        probs = predict_batch(model, batch_images, device)
        for batch_idx, prob in zip(batch_indices, probs):
            results.append({
                'grid_index': batch_idx,
                'yew_probability': float(prob)
            })

    # Merge results with grid
    if len(results) > 0:
        results_df = pd.DataFrame(results)
        grid_df = grid_df.merge(
            results_df, left_index=True, right_on='grid_index', how='left')
    else:
        # No successful predictions - add empty column
        grid_df['yew_probability'] = np.nan
        grid_df['grid_index'] = grid_df.index

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = output_dir / f'predictions_{timestamp}.csv'
    grid_df.to_csv(results_path, index=False)

    print(f"\nSaved predictions: {results_path}")
    print(f"  Total points: {len(grid_df)}")
    print(
        f"  Successful predictions: {grid_df['yew_probability'].notna().sum()}")
    print(f"  Failed extractions: {grid_df['yew_probability'].isna().sum()}")

    # Statistics
    if grid_df['yew_probability'].notna().sum() > 0:
        pred_df = grid_df[grid_df['yew_probability'].notna()]
        print(f"\nPrediction Statistics:")
        print(f"  Mean probability: {pred_df['yew_probability'].mean():.3f}")
        print(f"  Std deviation: {pred_df['yew_probability'].std():.3f}")
        print(f"  Min: {pred_df['yew_probability'].min():.3f}")
        print(f"  Max: {pred_df['yew_probability'].max():.3f}")
        print(
            f"  High probability (>0.7): {(pred_df['yew_probability'] > 0.7).sum()}")
        print(
            f"  Medium probability (0.3-0.7): {((pred_df['yew_probability'] >= 0.3) & (pred_df['yew_probability'] <= 0.7)).sum()}")
        print(
            f"  Low probability (<0.3): {(pred_df['yew_probability'] < 0.3).sum()}")

    # Create maps
    create_probability_map(grid_df, output_dir, cache_dir)

    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'model_path': args.model,
        'region': {
            'lat_min': args.lat_min,
            'lat_max': args.lat_max,
            'lon_min': args.lon_min,
            'lon_max': args.lon_max
        },
        'grid_spacing': args.grid_spacing,
        'total_points': len(grid_df),
        'successful_predictions': int(grid_df['yew_probability'].notna().sum()),
        'device': str(device)
    }

    metadata_path = output_dir / f'metadata_{timestamp}.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*80}")
    print("PREDICTION COMPLETE")
    print(f"{'='*80}")
    print(f"Results: {output_dir}")
    print(f"  - predictions_{timestamp}.csv")
    print(f"  - yew_probability_map.png")
    print(f"  - yew_hotspots_map.png")
    print(f"  - metadata_{timestamp}.json")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
