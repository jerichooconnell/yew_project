#!/usr/bin/env python3
"""
Predict Yew Probability on Grid
================================

Extracts satellite imagery on a grid over southern Vancouver Island and 
predicts yew probability at each location using the trained model.

Author: GitHub Copilot
Date: November 19, 2025
"""

from training.model import ResNet4Channel
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from datetime import datetime
import json
from tqdm import tqdm
import torch.nn.functional as F
import torch
import pandas as pd
import numpy as np
import ee
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


def initialize_earth_engine():
    """Initialize Earth Engine with authentication."""
    try:
        ee.Initialize(project='carbon-storm-206002')
        print("✓ Earth Engine initialized")
        return True
    except Exception as e:
        print(f"✗ Earth Engine initialization failed: {e}")
        return False


def create_grid(lat_min, lat_max, lon_min, lon_max, spacing_km=2.0):
    """
    Create a regular grid of points.

    Args:
        lat_min, lat_max: Latitude bounds
        lon_min, lon_max: Longitude bounds
        spacing_km: Spacing between grid points in kilometers

    Returns:
        DataFrame with lat, lon columns
    """
    # Approximate degrees per km (at ~48°N)
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians((lat_min + lat_max) / 2))

    spacing_lat = spacing_km / km_per_deg_lat
    spacing_lon = spacing_km / km_per_deg_lon

    lats = np.arange(lat_min, lat_max, spacing_lat)
    lons = np.arange(lon_min, lon_max, spacing_lon)

    lon_grid, lat_grid = np.meshgrid(lons, lats)

    grid_df = pd.DataFrame({
        'latitude': lat_grid.flatten(),
        'longitude': lon_grid.flatten(),
        'grid_id': range(len(lat_grid.flatten()))
    })

    print(f"\nCreated grid:")
    print(f"  Lat range: {lat_min:.4f} to {lat_max:.4f}")
    print(f"  Lon range: {lon_min:.4f} to {lon_max:.4f}")
    print(f"  Spacing: {spacing_km} km")
    print(f"  Total points: {len(grid_df)}")

    return grid_df


def extract_image_at_point(point_data):
    """
    Extract satellite image at a single point.

    Args:
        point_data: Tuple of (grid_id, lat, lon)

    Returns:
        Tuple of (grid_id, image_array or None, error_message or None)
    """
    grid_id, lat, lon = point_data

    try:
        # Create point geometry
        point = ee.Geometry.Point([lon, lat])

        # Define 640m buffer (64 pixels * 10m)
        roi = point.buffer(320)

        # Get Sentinel-2 SR Harmonized collection
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                      .filterBounds(roi)
                      .filterDate('2020-01-01', '2023-12-31')
                      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))

        # Get median composite
        image = collection.median()

        # Select bands: B2 (Blue), B3 (Green), B4 (Red), B8 (NIR)
        image = image.select(['B2', 'B3', 'B4', 'B8'])

        # Sample as array
        arrays = image.sampleRectangle(region=roi, defaultValue=0)

        # Get numpy arrays
        b2 = np.array(arrays.get('B2').getInfo())
        b3 = np.array(arrays.get('B3').getInfo())
        b4 = np.array(arrays.get('B4').getInfo())
        b8 = np.array(arrays.get('B8').getInfo())

        # Stack bands (B, G, R, NIR)
        img_array = np.stack([b2, b3, b4, b8], axis=0)

        # Resize to 64x64 if needed
        if img_array.shape[1] != 64 or img_array.shape[2] != 64:
            from scipy.ndimage import zoom
            zoom_factors = [1, 64/img_array.shape[1], 64/img_array.shape[2]]
            img_array = zoom(img_array, zoom_factors, order=1)

        return (grid_id, img_array, None)

    except Exception as e:
        return (grid_id, None, str(e))


def extract_images_parallel(grid_df, max_workers=4):
    """
    Extract images for all grid points in parallel.

    Args:
        grid_df: DataFrame with latitude, longitude, grid_id
        max_workers: Number of parallel workers

    Returns:
        Dictionary mapping grid_id to image array
    """
    print(f"\nExtracting images with {max_workers} workers...")

    point_data = [(row['grid_id'], row['latitude'], row['longitude'])
                  for _, row in grid_df.iterrows()]

    images = {}
    errors = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_image_at_point, pd): pd[0]
                   for pd in point_data}

        for future in tqdm(as_completed(futures), total=len(futures)):
            grid_id, img_array, error = future.result()

            if img_array is not None:
                images[grid_id] = img_array
            else:
                errors.append((grid_id, error))

    print(f"✓ Extracted {len(images)} images")
    if errors:
        print(f"✗ Failed to extract {len(errors)} images")

    return images, errors


def load_model(model_path, device):
    """Load trained model from checkpoint."""
    print(f"\nLoading model from: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    architecture = checkpoint.get('architecture', 'resnet18')

    model = ResNet4Channel(architecture=architecture, num_classes=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"  Architecture: {architecture}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")

    return model


def preprocess_image(img_array):
    """
    Preprocess image for model input.

    Args:
        img_array: (4, H, W) numpy array

    Returns:
        (1, 4, H, W) torch tensor
    """
    # Convert to (H, W, C)
    img = np.transpose(img_array, (1, 2, 0))

    # Normalize each channel using percentile clipping
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

    # Convert to tensor (C, H, W) and add batch dimension
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)

    return img_tensor


def predict_probabilities(model, images, device):
    """
    Predict yew probability for all images.

    Args:
        model: Trained model
        images: Dictionary mapping grid_id to image array
        device: torch device

    Returns:
        Dictionary mapping grid_id to yew probability
    """
    print(f"\nRunning predictions on {len(images)} images...")

    probabilities = {}

    with torch.no_grad():
        for grid_id, img_array in tqdm(images.items()):
            # Preprocess
            img_tensor = preprocess_image(img_array).to(device)

            # Predict
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)
            yew_prob = probs[0, 1].cpu().item()

            probabilities[grid_id] = yew_prob

    print(f"✓ Predictions complete")

    return probabilities


def create_probability_map(grid_df, probabilities, output_path, title="Yew Probability Map"):
    """
    Create a map visualization of yew probabilities.

    Args:
        grid_df: DataFrame with grid points
        probabilities: Dictionary mapping grid_id to probability
        output_path: Path to save figure
        title: Map title
    """
    print("\nCreating probability map...")

    # Add probabilities to dataframe
    grid_df['yew_probability'] = grid_df['grid_id'].map(probabilities)

    # Remove points without predictions
    grid_df = grid_df.dropna(subset=['yew_probability'])

    # Create figure
    fig = plt.figure(figsize=(14, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Set extent
    lon_min, lon_max = grid_df['longitude'].min(), grid_df['longitude'].max()
    lat_min, lat_max = grid_df['latitude'].min(), grid_df['latitude'].max()
    margin = 0.1
    ax.set_extent([lon_min - margin, lon_max + margin,
                   lat_min - margin, lat_max + margin],
                  crs=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)

    # Create custom colormap (white -> yellow -> red)
    colors = ['#ffffff', '#ffffcc', '#ffeda0', '#fed976',
              '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#b10026']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('yew_prob', colors, N=n_bins)

    # Plot probability as scatter
    scatter = ax.scatter(
        grid_df['longitude'],
        grid_df['latitude'],
        c=grid_df['yew_probability'],
        cmap=cmap,
        vmin=0,
        vmax=1,
        s=50,
        alpha=0.8,
        edgecolors='none',
        transform=ccrs.PlateCarree()
    )

    # Add colorbar
    cbar = plt.colorbar(
        scatter, ax=ax, orientation='vertical', pad=0.05, shrink=0.8)
    cbar.set_label('Yew Probability', fontsize=12, fontweight='bold')

    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5,
                      color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    # Title and labels
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Add statistics box
    stats_text = (
        f"Total Points: {len(grid_df)}\n"
        f"Mean Probability: {grid_df['yew_probability'].mean():.3f}\n"
        f"Max Probability: {grid_df['yew_probability'].max():.3f}\n"
        f"High Prob (>0.7): {(grid_df['yew_probability'] > 0.7).sum()}"
    )
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved map to: {output_path}")


def save_results(grid_df, probabilities, images, output_dir):
    """Save prediction results and metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions as CSV
    grid_df['yew_probability'] = grid_df['grid_id'].map(probabilities)
    csv_path = output_dir / 'yew_predictions.csv'
    grid_df.to_csv(csv_path, index=False)
    print(f"✓ Saved predictions to: {csv_path}")

    # Save images as numpy arrays
    images_dir = output_dir / 'images'
    images_dir.mkdir(exist_ok=True)
    for grid_id, img_array in tqdm(images.items(), desc="Saving images"):
        img_path = images_dir / f'grid_{grid_id:05d}.npy'
        np.save(img_path, img_array)
    print(f"✓ Saved {len(images)} images to: {images_dir}")

    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'n_points': len(grid_df),
        'n_predictions': len(probabilities),
        'mean_probability': float(grid_df['yew_probability'].mean()),
        'max_probability': float(grid_df['yew_probability'].max()),
        'min_probability': float(grid_df['yew_probability'].min()),
        'high_prob_count': int((grid_df['yew_probability'] > 0.7).sum())
    }
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Predict yew probability on a grid',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    # Southern Vancouver Island
    python scripts/prediction/predict_yew_probability_grid.py \\
        --model models/checkpoints/resnet18_20251119_121738_best.pth \\
        --lat-min 48.3 --lat-max 48.7 \\
        --lon-min -123.8 --lon-max -123.2 \\
        --spacing 2.0 \\
        --output-dir results/predictions/south_van_island
        """
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
    parser.add_argument('--spacing', type=float, default=2.0,
                        help='Grid spacing in kilometers (default: 2.0)')
    parser.add_argument('--output-dir', type=str,
                        default='results/predictions/grid',
                        help='Output directory')
    parser.add_argument('--max-workers', type=int, default=4,
                        help='Number of parallel workers for extraction (default: 4)')

    args = parser.parse_args()

    # Initialize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if not initialize_earth_engine():
        return

    # Create grid
    grid_df = create_grid(
        args.lat_min, args.lat_max,
        args.lon_min, args.lon_max,
        args.spacing
    )

    # Extract images
    images, errors = extract_images_parallel(grid_df, args.max_workers)

    if not images:
        print("✗ No images extracted successfully")
        return

    # Load model
    model = load_model(args.model, device)

    # Predict
    probabilities = predict_probabilities(model, images, device)

    # Create map
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    map_path = output_dir / 'yew_probability_map.png'
    create_probability_map(grid_df, probabilities, map_path,
                           title=f"Yew Probability Map - {args.lat_min:.2f}°N to {args.lat_max:.2f}°N")

    # Save results
    save_results(grid_df, probabilities, images, output_dir)

    print("\n" + "="*80)
    print("PREDICTION COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    print(f"  - yew_probability_map.png: Visualization")
    print(f"  - yew_predictions.csv: Predictions for all points")
    print(f"  - images/: Extracted satellite images")
    print(f"  - metadata.json: Summary statistics")
    print("="*80)


if __name__ == '__main__':
    main()
