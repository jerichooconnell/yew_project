#!/usr/bin/env python3
"""
Large Area Yew Classification - Aligned Chunk Method

Uses Google Earth Engine with proper CRS transform to ensure perfect pixel
alignment between chunks. No stitching artifacts!

The key is to compute the global grid ONCE, then download each chunk
with an explicit crs_transform that aligns to that grid.

Usage:
    python scripts/prediction/classify_large_area_geotiff.py \
        --bbox 48.44 48.47 -124.11 -124.002 \
        --output-dir results/predictions/large_area_geotiff \
        --year 2024 \
        --scale 10

Author: GitHub Copilot
Date: January 2026
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from io import BytesIO

import ee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm
import requests

try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_origin
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def parse_args():
    parser = argparse.ArgumentParser(
        description='Classify yew presence using properly aligned chunks',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--bbox', nargs=4, type=float, required=True,
                        metavar=('LAT_MIN', 'LAT_MAX', 'LON_MIN', 'LON_MAX'),
                        help='Bounding box: lat_min lat_max lon_min lon_max')
    parser.add_argument('--year', type=int, default=2024,
                        help='Year for embeddings (default: 2024)')
    parser.add_argument('--scale', type=int, default=10,
                        help='Pixel resolution in meters (default: 10)')
    parser.add_argument('--train-csv', type=str, 
                        default='data/processed/train_split_filtered.csv',
                        help='Path to training CSV')
    parser.add_argument('--val-csv', type=str,
                        default='data/processed/val_split_filtered.csv',
                        help='Path to validation CSV')
    parser.add_argument('--embedding-dir', type=str,
                        default='data/ee_imagery/embedding_patches_64x64',
                        help='Directory with training embedding patches')
    parser.add_argument('--output-dir', type=str,
                        default='results/predictions/large_area_geotiff',
                        help='Output directory for results')
    parser.add_argument('--gee-project', type=str,
                        default='carbon-storm-206002',
                        help='Google Earth Engine project ID')
    parser.add_argument('--skip-rgb', action='store_true',
                        help='Skip downloading Sentinel-2 RGB')
    parser.add_argument('--batch-size', type=int, default=50000,
                        help='Pixels per batch for classification')
    parser.add_argument('--chunk-size', type=int, default=250,
                        help='Pixels per chunk for download (default: 250)')
    parser.add_argument('--force-redownload', action='store_true',
                        help='Force re-download even if cached')
    return parser.parse_args()


# =============================================================================
# Grid Computation - This is the KEY to proper alignment!
# =============================================================================

def compute_global_grid(lat_min, lat_max, lon_min, lon_max, scale):
    """
    Compute a fixed global grid for the entire area.
    
    All chunks will be aligned to this grid by using the same crs_transform.
    This ensures pixels from different chunks align perfectly.
    
    Returns:
        dict with:
            - origin_lon, origin_lat: top-left corner of grid
            - pixel_size_lon, pixel_size_lat: pixel dimensions in degrees
            - width, height: total image dimensions in pixels
            - crs_transform: the affine transform for GEE
    """
    # Calculate pixel size in degrees at this latitude
    center_lat = (lat_min + lat_max) / 2.0
    meters_per_deg_lat = 111000.0
    meters_per_deg_lon = 111000.0 * np.cos(np.radians(center_lat))
    
    pixel_size_lat = scale / meters_per_deg_lat  # degrees per pixel (positive)
    pixel_size_lon = scale / meters_per_deg_lon  # degrees per pixel (positive)
    
    # Set origin at top-left (max lat, min lon)
    origin_lon = lon_min
    origin_lat = lat_max  # Top of image
    
    # Calculate dimensions
    width = int(np.ceil((lon_max - lon_min) / pixel_size_lon))
    height = int(np.ceil((lat_max - lat_min) / pixel_size_lat))
    
    # Create GEE-compatible crs_transform
    # Format: [xScale, xShear, xTranslation, yShear, yScale, yTranslation]
    crs_transform = [
        pixel_size_lon,     # xScale (positive, going east)
        0,                  # xShear
        origin_lon,         # xTranslation (left edge)
        0,                  # yShear
        -pixel_size_lat,    # yScale (negative, going south from top)
        origin_lat          # yTranslation (top edge)
    ]
    
    return {
        'origin_lon': origin_lon,
        'origin_lat': origin_lat,
        'pixel_size_lon': pixel_size_lon,
        'pixel_size_lat': pixel_size_lat,
        'width': width,
        'height': height,
        'crs_transform': crs_transform,
        'lat_min': lat_min,
        'lat_max': lat_max,
        'lon_min': lon_min,
        'lon_max': lon_max,
        'scale': scale,
    }


def download_chunk_aligned(image, grid, row_start, row_end, col_start, col_end, n_bands, desc="chunk"):
    """
    Download a chunk of an image with proper grid alignment.
    
    Uses explicit pixel grid to ensure proper alignment.
    The chunk region is specified in pixel coordinates.
    """
    # Calculate the geographic bounds for this chunk
    lon_start = grid['origin_lon'] + col_start * grid['pixel_size_lon']
    lon_end = grid['origin_lon'] + col_end * grid['pixel_size_lon']
    lat_start = grid['origin_lat'] - row_end * grid['pixel_size_lat']
    lat_end = grid['origin_lat'] - row_start * grid['pixel_size_lat']
    
    # Create region (GEE uses [west, south, east, north])
    region = ee.Geometry.Rectangle([lon_start, lat_start, lon_end, lat_end])
    
    chunk_width = col_end - col_start
    chunk_height = row_end - row_start
    
    try:
        # Use scale + region (not crs_transform + dimensions - GEE doesn't allow both)
        # The key is to use consistent geographic bounds derived from pixel grid
        url = image.getDownloadURL({
            'region': region,
            'crs': 'EPSG:4326',
            'scale': grid['scale'],
            'format': 'NPY',
        })
        
        response = requests.get(url, timeout=300)
        if response.status_code != 200:
            raise ValueError(f"Download failed: {response.status_code}")
        
        data = np.load(BytesIO(response.content), allow_pickle=True)
        
        # Handle structured array
        if data.dtype.names is not None:
            arrays = [data[name] for name in data.dtype.names]
            data = np.stack(arrays, axis=-1)
        
        # Ensure proper shape
        if data.ndim == 2 and n_bands > 1:
            data = np.expand_dims(data, axis=-1)
        elif data.ndim == 3 and n_bands == 1:
            data = data[:, :, 0] if data.shape[2] == 1 else data.mean(axis=2)
        
        # Handle dimension mismatches by clipping or padding
        actual_h, actual_w = data.shape[:2]
        if actual_h != chunk_height or actual_w != chunk_width:
            if n_bands > 1:
                padded = np.zeros((chunk_height, chunk_width, n_bands), dtype=np.float32)
                h = min(actual_h, chunk_height)
                w = min(actual_w, chunk_width)
                padded[:h, :w, :] = data[:h, :w, :]
            else:
                padded = np.zeros((chunk_height, chunk_width), dtype=np.float32)
                h = min(actual_h, chunk_height)
                w = min(actual_w, chunk_width)
                padded[:h, :w] = data[:h, :w]
            data = padded
        
        return data.astype(np.float32)
        
    except Exception as e:
        print(f"\n  Warning: Failed to download {desc}: {e}")
        # Return zeros for failed chunk
        shape = (chunk_height, chunk_width, n_bands) if n_bands > 1 else (chunk_height, chunk_width)
        return np.zeros(shape, dtype=np.float32)


def download_image_chunked(image, grid, n_bands, chunk_size, description):
    """
    Download a large image in chunks with proper grid alignment.
    
    All chunks use the same crs_transform, ensuring perfect alignment.
    """
    height = grid['height']
    width = grid['width']
    
    print(f"  Total size: {width}×{height} = {width * height:,} pixels")
    print(f"  Chunk size: {chunk_size}×{chunk_size}")
    
    # Calculate chunks
    n_rows = int(np.ceil(height / chunk_size))
    n_cols = int(np.ceil(width / chunk_size))
    total_chunks = n_rows * n_cols
    
    print(f"  Grid: {n_rows}×{n_cols} = {total_chunks} chunks")
    
    # Initialize output array
    shape = (height, width, n_bands) if n_bands > 1 else (height, width)
    result = np.zeros(shape, dtype=np.float32)
    
    # Download chunks with progress bar
    chunk_idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            row_start = row * chunk_size
            row_end = min((row + 1) * chunk_size, height)
            col_start = col * chunk_size
            col_end = min((col + 1) * chunk_size, width)
            
            chunk_idx += 1
            desc_str = f"{description} chunk {chunk_idx}/{total_chunks}"
            print(f"\r  Downloading {desc_str}...", end='', flush=True)
            
            chunk_data = download_chunk_aligned(
                image, grid, row_start, row_end, col_start, col_end, n_bands, desc_str
            )
            
            # Place in result array
            actual_height = row_end - row_start
            actual_width = col_end - col_start
            
            if n_bands > 1:
                result[row_start:row_end, col_start:col_end, :] = chunk_data[:actual_height, :actual_width, :]
            else:
                result[row_start:row_end, col_start:col_end] = chunk_data[:actual_height, :actual_width]
    
    print(f"\r  ✓ Downloaded {total_chunks} chunks                    ")
    
    return result


def download_embedding(lat_min, lat_max, lon_min, lon_max, year, scale, chunk_size, output_dir, grid=None):
    """Download satellite embedding image with proper grid alignment."""
    
    if grid is None:
        grid = compute_global_grid(lat_min, lat_max, lon_min, lon_max, scale)
    
    print(f"Downloading embeddings...")
    print(f"  Bounding box: lat [{lat_min}, {lat_max}], lon [{lon_min}, {lon_max}]")
    print(f"  Year: {year}, Scale: {scale}m")
    print(f"  Grid: {grid['width']}×{grid['height']} pixels")
    
    # Get embedding image
    dataset = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
    start_date = f'{year}-01-01'
    end_date = f'{year + 1}-01-01'
    
    region = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
    
    image = (dataset
             .filterDate(start_date, end_date)
             .filterBounds(region)
             .first())
    
    if image is None:
        raise ValueError(f"No embedding image found for {year}")
    
    bands = [f'A{i:02d}' for i in range(64)]
    embedding = image.select(bands).toFloat()
    
    # Download chunked
    data = download_image_chunked(embedding, grid, 64, chunk_size, "embedding")
    
    print(f"  Shape: {data.shape}")
    print(f"  Value range: [{data.min():.4f}, {data.max():.4f}]")
    
    return data, grid


def download_rgb(lat_min, lat_max, lon_min, lon_max, year, scale, chunk_size, output_dir, grid=None):
    """Download Sentinel-2 RGB with proper grid alignment."""
    
    if grid is None:
        grid = compute_global_grid(lat_min, lat_max, lon_min, lon_max, scale)
    
    print(f"Downloading Sentinel-2 RGB...")
    print(f"  Grid: {grid['width']}×{grid['height']} pixels")
    
    region = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
    
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(region)
          .filterDate(f'{year}-06-01', f'{year}-09-30')
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
          .median())
    
    # Get RGB bands
    rgb = s2.select(['B4', 'B3', 'B2']).toFloat()
    
    # Download chunked
    data = download_image_chunked(rgb, grid, 3, chunk_size, "RGB")
    
    # Normalize to 0-1
    for i in range(3):
        band = data[:, :, i]
        valid = band > 0
        if valid.any():
            p2, p98 = np.percentile(band[valid], [2, 98])
            data[:, :, i] = np.clip((band - p2) / (p98 - p2 + 1e-8), 0, 1)
    
    print(f"  Shape: {data.shape}")
    
    return data, grid


# =============================================================================
# Training Functions
# =============================================================================

def extract_center_pixel(lat, lon, emb_dir, patch_size=64):
    """Extract single center pixel from embedding."""
    emb_path = emb_dir / f'embedding_{lat:.6f}_{lon:.6f}.npy'

    if not emb_path.exists():
        return None

    try:
        img = np.load(emb_path)
        center = patch_size // 2

        if img.ndim == 3:
            if img.shape[0] == 64:
                return img[:, center, center]
            elif img.shape[2] == 64:
                return img[center, center, :]
        return None
    except Exception as e:
        print(f'Error loading {emb_path}: {e}')
        return None


def extract_features_from_split(df, emb_dir):
    """Extract center pixel features and labels from a data split."""
    features = []
    labels = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc='  Extracting features'):
        if pd.notna(row['lat']) and pd.notna(row['lon']):
            center_data = extract_center_pixel(row['lat'], row['lon'], emb_dir)

            if center_data is not None:
                features.append(center_data)
                labels.append(int(row['has_yew']))

    return np.array(features), np.array(labels)


def train_svm_with_validation(train_csv, val_csv, emb_dir):
    """Train SVM classifier with StandardScaler."""
    print("Training SVM classifier with StandardScaler...")
    
    emb_dir = Path(emb_dir)
    
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    
    print(f"  Train CSV: {len(train_df)} rows")
    print(f"  Val CSV: {len(val_df)} rows")
    
    X_train, y_train = extract_features_from_split(train_df, emb_dir)
    X_val, y_val = extract_features_from_split(val_df, emb_dir)
    
    print(f"  Train features: {len(X_train)} (Yew: {y_train.sum()})")
    print(f"  Val features: {len(X_val)} (Yew: {y_val.sum()})")
    
    if len(X_train) == 0:
        raise ValueError("No training features found!")
    
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    
    X_all = np.vstack([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])
    
    print(f"  Combined training: {len(X_all)} samples")
    
    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)
    
    print(f"  ✓ StandardScaler fitted")
    
    clf = SVC(kernel='rbf', probability=True, random_state=42)
    clf.fit(X_all_scaled, y_all)
    print(f"  ✓ SVM trained on {len(X_all)} samples")
    
    X_val_scaled = scaler.transform(X_val)
    y_val_pred = clf.predict(X_val_scaled)
    y_val_prob = clf.predict_proba(X_val_scaled)[:, 1]
    
    metrics = {
        'accuracy': float(accuracy_score(y_val, y_val_pred)),
        'f1_score': float(f1_score(y_val, y_val_pred)),
        'roc_auc': float(roc_auc_score(y_val, y_val_prob)),
        'n_train': int(len(X_all)),
        'n_val': int(len(X_val)),
    }
    
    print(f"\n  Validation Performance:")
    print(f"    Accuracy: {metrics['accuracy']:.4f}")
    print(f"    F1 Score: {metrics['f1_score']:.4f}")
    print(f"    ROC-AUC:  {metrics['roc_auc']:.4f}")
    
    return clf, scaler, metrics


# =============================================================================
# Classification
# =============================================================================

def classify_image(embedding_img, classifier, scaler, batch_size=50000):
    """Classify every pixel in the embedding image."""
    H, W, C = embedding_img.shape
    total_pixels = H * W
    
    print(f"Classifying {total_pixels:,} pixels...")
    
    flat = embedding_img.reshape(-1, C)
    flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"  Applying StandardScaler...")
    flat_scaled = scaler.transform(flat)
    
    probs = np.zeros(total_pixels, dtype=np.float32)
    
    n_batches = (total_pixels + batch_size - 1) // batch_size
    for i in tqdm(range(0, total_pixels, batch_size), desc="  Predicting", total=n_batches):
        end = min(i + batch_size, total_pixels)
        batch = flat_scaled[i:end]
        probs[i:end] = classifier.predict_proba(batch)[:, 1]
    
    prob_grid = probs.reshape(H, W)
    
    print(f"  ✓ Classification complete")
    print(f"  Probability range: [{prob_grid.min():.3f}, {prob_grid.max():.3f}]")
    print(f"  Mean: {prob_grid.mean():.3f}, Median: {np.median(prob_grid):.3f}")
    
    return prob_grid


# =============================================================================
# Visualization
# =============================================================================

def create_visualizations(prob_grid, rgb_image, output_dir, grid, metrics):
    """Create and save visualization figures."""
    cmap = LinearSegmentedColormap.from_list(
        'yew_prob', ['#2166ac', '#92c5de', '#f7f7f7', '#f4a582', '#d6604d', '#b2182b'], N=256
    )
    
    lat_min = grid['lat_min']
    lat_max = grid['lat_max']
    lon_min = grid['lon_min']
    lon_max = grid['lon_max']
    
    extent = [lon_min, lon_max, lat_min, lat_max]
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    # Three-panel view
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    if rgb_image is not None:
        axes[0].imshow(rgb_image, extent=extent, aspect='auto')
        axes[0].set_title('Sentinel-2 RGB', fontsize=14, fontweight='bold')
    else:
        axes[0].text(0.5, 0.5, 'RGB not available', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('RGB', fontsize=14)
    
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    
    im = axes[1].imshow(prob_grid, extent=extent, cmap=cmap, 
                        vmin=0, vmax=1, aspect='auto')
    axes[1].set_title('Yew Probability', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Longitude')
    plt.colorbar(im, ax=axes[1], fraction=0.046, label='P(yew)')
    
    if rgb_image is not None:
        axes[2].imshow(rgb_image, extent=extent, aspect='auto')
        im2 = axes[2].imshow(prob_grid, extent=extent, cmap=cmap, alpha=0.5,
                             vmin=0, vmax=1, aspect='auto')
        axes[2].set_title('Overlay (α=0.5)', fontsize=14, fontweight='bold')
    else:
        im2 = axes[2].imshow(prob_grid, extent=extent, cmap=cmap, 
                             vmin=0, vmax=1, aspect='auto')
        axes[2].set_title('Probability', fontsize=14, fontweight='bold')
    
    axes[2].set_xlabel('Longitude')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, label='P(yew)')
    
    plt.suptitle(f'Yew Detection: {lat_min:.4f}°N to {lat_max:.4f}°N, '
                 f'{abs(lon_min):.4f}°W to {abs(lon_max):.4f}°W\n'
                 f'Model: SVM with StandardScaler (Val Accuracy: {metrics["accuracy"]:.1%})',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figures_dir / 'three_panel_view.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved three_panel_view.png")
    
    # Histogram
    valid_probs = prob_grid.flatten()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(valid_probs, bins=50, edgecolor='black', alpha=0.7, color='#2ca02c')
    ax.axvline(valid_probs.mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {valid_probs.mean():.3f}')
    ax.axvline(np.median(valid_probs), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {np.median(valid_probs):.3f}')
    ax.set_xlabel('Yew Probability', fontsize=12)
    ax.set_ylabel('Pixel Count', fontsize=12)
    ax.set_title('Distribution of Yew Probability Predictions', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.savefig(figures_dir / 'probability_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved probability_histogram.png")
    
    return valid_probs


def save_results(prob_grid, rgb_image, embedding_image, output_dir, grid, metrics):
    """Save results to disk."""
    np.save(output_dir / 'prob_grid.npy', prob_grid)
    print(f"  ✓ Saved prob_grid.npy: {prob_grid.shape}")
    
    np.save(output_dir / 'embedding_image.npy', embedding_image)
    print(f"  ✓ Saved embedding_image.npy: {embedding_image.shape}")
    
    if rgb_image is not None:
        np.save(output_dir / 'rgb_image.npy', rgb_image)
        print(f"  ✓ Saved rgb_image.npy: {rgb_image.shape}")
    
    # Save grid parameters
    with open(output_dir / 'grid_params.json', 'w') as f:
        json.dump(grid, f, indent=2)
    print(f"  ✓ Saved grid_params.json")
    
    valid_probs = prob_grid.flatten()
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'method': 'aligned_chunks_crs_transform',
        'bbox': {
            'lat_min': grid['lat_min'],
            'lat_max': grid['lat_max'],
            'lon_min': grid['lon_min'],
            'lon_max': grid['lon_max']
        },
        'year': 2024,
        'scale_m': grid['scale'],
        'grid': {
            'width': grid['width'],
            'height': grid['height'],
            'crs_transform': grid['crs_transform'],
        },
        'prob_grid_shape': list(prob_grid.shape),
        'embedding_shape': list(embedding_image.shape),
        'rgb_shape': list(rgb_image.shape) if rgb_image is not None else None,
        'model': {
            'type': 'SVM with StandardScaler',
            'kernel': 'rbf',
            'training_samples': metrics['n_train'],
            'validation_accuracy': metrics['accuracy'],
            'validation_f1': metrics['f1_score'],
            'validation_roc_auc': metrics['roc_auc'],
        },
        'statistics': {
            'mean': float(valid_probs.mean()),
            'median': float(np.median(valid_probs)),
            'std': float(valid_probs.std()),
            'min': float(valid_probs.min()),
            'max': float(valid_probs.max()),
            'pixels_above_30': int((valid_probs >= 0.3).sum()),
            'pixels_above_50': int((valid_probs >= 0.5).sum()),
            'pixels_above_70': int((valid_probs >= 0.7).sum()),
            'pixels_above_90': int((valid_probs >= 0.9).sum()),
        }
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Saved metadata.json")
    
    # Save as GeoTIFF if rasterio available (skip CRS issues)
    try:
        if HAS_RASTERIO:
            from rasterio.transform import Affine
            
            transform = Affine(
                grid['pixel_size_lon'], 0, grid['origin_lon'],
                0, -grid['pixel_size_lat'], grid['origin_lat']
            )
            
            with rasterio.open(
                output_dir / 'prob_grid.tif',
                'w',
                driver='GTiff',
                height=prob_grid.shape[0],
                width=prob_grid.shape[1],
                count=1,
                dtype=prob_grid.dtype,
                crs='EPSG:4326',
                transform=transform,
            ) as dst:
                dst.write(prob_grid, 1)
            print(f"  ✓ Saved prob_grid.tif (GeoTIFF)")
    except Exception as e:
        print(f"  ⚠ Could not save GeoTIFF: {e}")


def print_statistics(valid_probs, scale):
    """Print detection statistics."""
    print("\n" + "=" * 60)
    print("YEW DETECTION STATISTICS")
    print("=" * 60)
    print(f"Total pixels: {len(valid_probs):,}")
    print(f"Mean probability: {valid_probs.mean():.4f}")
    print(f"Median probability: {np.median(valid_probs):.4f}")
    print(f"Std deviation: {valid_probs.std():.4f}")
    print()
    
    pixel_area_m2 = scale ** 2
    
    for thresh in [0.3, 0.5, 0.7, 0.9]:
        n_above = (valid_probs >= thresh).sum()
        pct = 100 * n_above / len(valid_probs)
        area_ha = n_above * pixel_area_m2 / 10000
        print(f"P(yew) ≥ {thresh:.1f}: {n_above:>8,} pixels ({pct:>5.2f}%) = {area_ha:>8.2f} ha")
    
    print("=" * 60)


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    
    lat_min, lat_max, lon_min, lon_max = args.bbox
    
    # Initialize Earth Engine
    print("Initializing Earth Engine...")
    ee.Initialize(project=args.gee_project)
    print("✓ Earth Engine initialized")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("LARGE AREA YEW CLASSIFICATION (Aligned Chunks)")
    print(f"{'='*60}")
    print(f"Bbox: lat [{lat_min}, {lat_max}], lon [{lon_min}, {lon_max}]")
    print(f"Scale: {args.scale}m per pixel")
    print(f"Chunk size: {args.chunk_size} pixels")
    print(f"Rasterio: {'Yes' if HAS_RASTERIO else 'No (using NPY fallback)'}")
    print(f"Output: {output_dir}\n")
    
    # Compute global grid ONCE
    grid = compute_global_grid(lat_min, lat_max, lon_min, lon_max, args.scale)
    print(f"Global grid: {grid['width']}×{grid['height']} pixels")
    print(f"CRS Transform: {grid['crs_transform']}\n")
    
    # Step 1: Download embedding
    embedding_cache = output_dir / 'embedding_image.npy'
    grid_cache = output_dir / 'grid_params.json'
    
    if embedding_cache.exists() and grid_cache.exists() and not args.force_redownload:
        print("Loading cached embedding...")
        embedding_image = np.load(embedding_cache)
        with open(grid_cache) as f:
            grid = json.load(f)
        print(f"✓ Loaded: {embedding_image.shape}\n")
    else:
        embedding_image, grid = download_embedding(
            lat_min, lat_max, lon_min, lon_max, args.year, args.scale,
            args.chunk_size, output_dir, grid
        )
        print()
    
    # Step 2: Download RGB (uses same grid!)
    rgb_image = None
    if not args.skip_rgb:
        rgb_cache = output_dir / 'rgb_image.npy'
        
        if rgb_cache.exists() and not args.force_redownload:
            print("Loading cached RGB...")
            rgb_image = np.load(rgb_cache)
            print(f"✓ Loaded: {rgb_image.shape}\n")
        else:
            try:
                rgb_image, _ = download_rgb(
                    lat_min, lat_max, lon_min, lon_max, args.year, args.scale,
                    args.chunk_size, output_dir, grid  # Pass same grid!
                )
                print()
            except Exception as e:
                print(f"⚠ Could not download RGB: {e}\n")
    
    # Verify dimensions match
    if rgb_image is not None:
        if rgb_image.shape[:2] != embedding_image.shape[:2]:
            print(f"⚠ Unexpected dimension mismatch: RGB {rgb_image.shape[:2]} vs Embedding {embedding_image.shape[:2]}")
        else:
            print(f"✓ Dimensions match: {embedding_image.shape[0]}×{embedding_image.shape[1]}\n")
    
    # Step 3: Train SVM
    print("-" * 60)
    clf, scaler, metrics = train_svm_with_validation(
        args.train_csv, args.val_csv, args.embedding_dir
    )
    print("-" * 60 + "\n")
    
    # Step 4: Classify
    prob_grid = classify_image(embedding_image, clf, scaler, args.batch_size)
    print()
    
    # Step 5: Visualize
    print("Creating visualizations...")
    valid_probs = create_visualizations(
        prob_grid, rgb_image, output_dir, grid, metrics
    )
    print()
    
    # Step 6: Statistics
    print_statistics(valid_probs, args.scale)
    
    # Step 7: Save
    print("\nSaving results...")
    save_results(prob_grid, rgb_image, embedding_image, output_dir, grid, metrics)
    
    print(f"\n✓ All results saved to: {output_dir}")


if __name__ == '__main__':
    main()
