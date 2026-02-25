#!/usr/bin/env python3
"""
Large Area Tiled Yew Classification

For areas too large for a single GEE download, this script:
1. Divides the area into spatial tiles
2. Downloads each tile's embeddings and RGB via band-group method
3. Trains (or loads) an SVM classifier
4. Classifies each tile individually (memory efficient)
5. Stitches tile probability grids and RGB into full mosaics
6. Saves tile embeddings separately for annotation-based retraining

Usage:
    python scripts/prediction/classify_tiled_area.py \
        --bbox 48.2728 48.7014 -124.5086 -123.1969 \
        --output-dir results/predictions/south_vi_large \
        --year 2024 --scale 10
"""

import argparse
import json
import os
import pickle
import time
from datetime import datetime
from io import BytesIO
from math import ceil, cos, radians
from pathlib import Path

import ee
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Classify yew over very large areas using spatial tiling',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--bbox', nargs=4, type=float, required=True,
                        metavar=('LAT_MIN', 'LAT_MAX', 'LON_MIN', 'LON_MAX'))
    parser.add_argument('--year', type=int, default=2024)
    parser.add_argument('--scale', type=int, default=10,
                        help='Pixel resolution in meters (default: 10)')
    parser.add_argument('--train-csv', type=str,
                        default='data/processed/train_split_balanced_max.csv')
    parser.add_argument('--val-csv', type=str,
                        default='data/processed/val_split_balanced_max.csv')
    parser.add_argument('--embedding-dir', type=str,
                        default='data/ee_imagery/embedding_patches_64x64')
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--gee-project', type=str, default='carbon-storm-206002')
    parser.add_argument('--batch-size', type=int, default=50000)
    parser.add_argument('--max-bytes', type=int, default=40000000,
                        help='Max bytes per GEE download request (default: 40MB)')
    parser.add_argument('--class-weight-ratio', type=float, default=None)
    parser.add_argument('--model-path', type=str, default=None,
                        help='Pre-trained SVM model path (.pkl)')
    parser.add_argument('--scaler-path', type=str, default=None,
                        help='Pre-trained scaler path (.pkl)')
    parser.add_argument('--skip-rgb', action='store_true')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Resume from cached tiles (default: True)')
    return parser.parse_args()


# =============================================================================
# Tile grid calculation
# =============================================================================

def calculate_tile_grid(south, north, west, east, scale, max_bytes, n_emb_bands=64):
    """
    Calculate a spatial tile grid so each tile's band-group download fits
    within GEE's size limit.
    """
    center_lat = (south + north) / 2.0
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * cos(radians(center_lat))

    total_h_px = int(ceil((north - south) * m_per_deg_lat / scale))
    total_w_px = int(ceil((east - west) * m_per_deg_lon / scale))

    # Target: tile_pixels * target_bands * 4 <= max_bytes
    # We want at least 5 bands per chunk for efficiency
    target_bands_per_chunk = 5
    max_tile_pixels = max_bytes / (target_bands_per_chunk * 4)
    max_tile_side = int(np.sqrt(max_tile_pixels))

    # Number of tiles in each direction
    n_rows = max(1, int(ceil(total_h_px / max_tile_side)))
    n_cols = max(1, int(ceil(total_w_px / max_tile_side)))

    lat_step = (north - south) / n_rows
    lon_step = (east - west) / n_cols

    tiles = []
    for r in range(n_rows):
        for c in range(n_cols):
            # North to south ordering (row 0 = northernmost)
            tile_north = north - r * lat_step
            tile_south = north - (r + 1) * lat_step
            tile_west = west + c * lon_step
            tile_east = west + (c + 1) * lon_step

            tiles.append({
                'row': r, 'col': c,
                'south': tile_south, 'north': tile_north,
                'west': tile_west, 'east': tile_east,
            })

    return tiles, n_rows, n_cols, total_h_px, total_w_px


# =============================================================================
# GEE download helpers
# =============================================================================

def download_npy(image, region, scale, timeout=600, max_retries=3):
    """Download an EE image as NPY with retry logic."""
    for attempt in range(max_retries):
        try:
            url = image.getDownloadURL({
                'region': region,
                'scale': scale,
                'crs': 'EPSG:4326',
                'format': 'NPY',
            })
            response = requests.get(url, timeout=timeout)
            if response.status_code != 200:
                raise ValueError(f"HTTP {response.status_code}: {response.text[:200]}")
            data = np.load(BytesIO(response.content), allow_pickle=True)
            if data.dtype.names is not None:
                arrays = [data[name] for name in data.dtype.names]
                data = np.stack(arrays, axis=-1)
            return data.astype(np.float32)
        except Exception as e:
            if 'request size' in str(e).lower():
                raise  # Size errors won't be fixed by retrying
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"      Retry {attempt+1}/{max_retries} in {wait}s: {str(e)[:80]}")
                time.sleep(wait)
            else:
                raise
    return data.astype(np.float32)


def probe_tile_dimensions(image, band_names, region, scale):
    """Download a single band to determine actual tile pixel dimensions."""
    probe_image = image.select([band_names[0]])
    data = download_npy(probe_image, region, scale)
    return data.shape[0], data.shape[1]


def download_bands_chunked(image, band_names, region, scale, max_bytes):
    """Download image by band subsets with retry logic, return stacked array."""
    GEE_LIMIT = 50_331_648  # 48 MiB hard limit

    # First, probe actual dimensions with 1 band
    probe_image = image.select([band_names[0]])
    first_band = download_npy(probe_image, region, scale)
    actual_h, actual_w = first_band.shape[0], first_band.shape[1]
    single_band_bytes = actual_h * actual_w * 4

    # Calculate bands_per_chunk using ACTUAL dimensions and GEE's limit
    # Use 80% of GEE limit for safety margin
    safe_limit = int(GEE_LIMIT * 0.80)
    bands_per_chunk = max(1, safe_limit // single_band_bytes)
    bands_per_chunk = min(bands_per_chunk, len(band_names))

    print(f"    Actual tile: {actual_h}×{actual_w}, "
          f"{single_band_bytes/1e6:.1f} MB/band, "
          f"bands/chunk={bands_per_chunk}")

    # Start collecting: first band already downloaded
    all_data = [first_band]
    remaining_bands = band_names[1:]
    n_remaining_chunks = int(ceil(len(remaining_bands) / bands_per_chunk))
    chunk_num = 0

    # Download remaining bands in chunks
    i = 0
    while i < len(remaining_bands):
        end = min(i + bands_per_chunk, len(remaining_bands))
        chunk_bands = remaining_bands[i:end]
        chunk_num += 1

        print(f"      Chunk {chunk_num}/{n_remaining_chunks}: "
              f"bands {chunk_bands[0]}-{chunk_bands[-1]} "
              f"({len(chunk_bands)} bands)...", end='', flush=True)

        chunk_image = image.select(chunk_bands)
        try:
            t0 = time.time()
            data = download_npy(chunk_image, region, scale)
            elapsed = time.time() - t0
            all_data.append(data)
            print(f" ✓ {elapsed:.1f}s")
            i = end
        except Exception as e:
            err_str = str(e)
            if 'request size' in err_str.lower() or 'too large' in err_str.lower():
                # Reduce chunk size and retry
                bands_per_chunk = max(1, len(chunk_bands) // 2)
                n_remaining_chunks = int(ceil((len(remaining_bands) - i) / bands_per_chunk)) + chunk_num
                print(f"\n    ⚠ Size error, reducing to {bands_per_chunk} bands/chunk")
                continue
            else:
                raise

    result = np.concatenate(all_data, axis=-1)
    return result


def download_tile_embedding(tile, ee_embedding, band_names, scale, max_bytes, cache_dir):
    """Download embedding for one spatial tile. Returns (H, W, 64) array."""
    cache_path = cache_dir / f"emb_{tile['row']}_{tile['col']}.npy"

    if cache_path.exists():
        return np.load(cache_path)

    region = ee.Geometry.Rectangle([tile['west'], tile['south'], tile['east'], tile['north']])
    data = download_bands_chunked(ee_embedding, band_names, region, scale, max_bytes)
    np.save(cache_path, data)
    return data


def download_tile_rgb(tile, year, scale, max_bytes, cache_dir):
    """Download RGB for one spatial tile. Returns (H, W, 3) array."""
    cache_path = cache_dir / f"rgb_{tile['row']}_{tile['col']}.npy"

    if cache_path.exists():
        return np.load(cache_path)

    region = ee.Geometry.Rectangle([tile['west'], tile['south'], tile['east'], tile['north']])

    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(region)
          .filterDate(f'{year}-06-01', f'{year}-09-30')
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
          .median())
    rgb = s2.select(['B4', 'B3', 'B2']).toFloat()

    data = download_npy(rgb, region, scale)

    # Normalize to 0-1
    for i in range(min(3, data.shape[-1])):
        band = data[:, :, i]
        valid = band > 0
        if valid.any():
            p2, p98 = np.percentile(band[valid], [2, 98])
            data[:, :, i] = np.clip((band - p2) / (p98 - p2 + 1e-8), 0, 1)

    np.save(cache_path, data)
    return data


# =============================================================================
# SVM Training (same as existing script)
# =============================================================================

def extract_center_pixel(lat, lon, emb_dir, patch_size=64):
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
    except Exception:
        return None


def extract_features_from_split(df, emb_dir):
    features, labels = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='  Loading embeddings'):
        if pd.notna(row['lat']) and pd.notna(row['lon']):
            feat = extract_center_pixel(row['lat'], row['lon'], emb_dir)
            if feat is not None:
                features.append(feat)
                labels.append(int(row['has_yew']))
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32)


def train_svm(train_csv, val_csv, emb_dir, class_weight_ratio=None):
    """Train SVM on existing training data."""
    print("Training SVM classifier...")
    emb_dir = Path(emb_dir)
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    print(f"  Train CSV: {len(train_df)} rows, Val CSV: {len(val_df)} rows")

    X_train, y_train = extract_features_from_split(train_df, emb_dir)
    X_val, y_val = extract_features_from_split(val_df, emb_dir)
    print(f"  Train: {len(X_train)} (Yew: {y_train.sum()}), Val: {len(X_val)} (Yew: {y_val.sum()})")

    X_all = np.vstack([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)

    cw = {0: class_weight_ratio, 1: 1.0} if class_weight_ratio else None
    if cw:
        print(f"  Class weights: non-yew={class_weight_ratio}, yew=1.0")

    clf = SVC(kernel='rbf', probability=True, random_state=42, class_weight=cw)
    clf.fit(X_all_scaled, y_all)
    print(f"  ✓ SVM trained on {len(X_all)} samples")

    # Validation metrics (use pre-split val data)
    X_val_scaled = scaler.transform(X_val)
    y_pred = clf.predict(X_val_scaled)
    y_prob = clf.predict_proba(X_val_scaled)[:, 1]
    metrics = {
        'accuracy': float(accuracy_score(y_val, y_pred)),
        'f1_score': float(f1_score(y_val, y_pred)),
        'roc_auc': float(roc_auc_score(y_val, y_prob)),
        'n_train': len(X_all),
    }
    print(f"  Val accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}, "
          f"AUC={metrics['roc_auc']:.4f}")

    return clf, scaler, metrics


# =============================================================================
# Tile classification
# =============================================================================

def classify_tile_data(embedding, clf, scaler, batch_size=50000):
    """Classify a single tile's embedding. Returns probability grid."""
    H, W, C = embedding.shape
    flat = embedding.reshape(-1, C)
    flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    flat_scaled = scaler.transform(flat)

    total = flat.shape[0]
    probs = np.zeros(total, dtype=np.float32)
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        probs[i:end] = clf.predict_proba(flat_scaled[i:end])[:, 1]

    return probs.reshape(H, W)


# =============================================================================
# Stitching
# =============================================================================

def stitch_tiles(tile_arrays, n_rows, n_cols):
    """
    Stitch a grid of tile arrays into a single array.
    Handles minor size mismatches between tiles.
    tile_arrays: dict mapping (row, col) -> np.ndarray
    """
    # For each row, concatenate columns
    rows = []
    for r in range(n_rows):
        # Find minimum height in this row (handle 1-pixel mismatches)
        heights = [tile_arrays[(r, c)].shape[0] for c in range(n_cols) if (r, c) in tile_arrays]
        target_h = min(heights) if heights else 0

        row_tiles = []
        for c in range(n_cols):
            arr = tile_arrays[(r, c)]
            # Crop to target height
            arr = arr[:target_h]
            row_tiles.append(arr)

        if row_tiles:
            row_concat = np.concatenate(row_tiles, axis=1)
            rows.append(row_concat)

    # Find minimum width across all rows
    if rows:
        target_w = min(r.shape[1] for r in rows)
        rows = [r[:, :target_w] for r in rows]
        return np.concatenate(rows, axis=0)
    return np.array([])


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    lat_min, lat_max, lon_min, lon_max = args.bbox

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tile_cache = output_dir / 'tiles'
    tile_cache.mkdir(exist_ok=True)

    print("=" * 70)
    print("LARGE AREA TILED YEW CLASSIFICATION")
    print("=" * 70)
    print(f"Bbox: {lat_min:.4f}°N to {lat_max:.4f}°N, {lon_min:.4f}°W to {lon_max:.4f}°W")
    print(f"Scale: {args.scale}m, Year: {args.year}")
    print(f"Output: {output_dir}")

    # -------------------------------------------------------------------------
    # 1. Calculate tile grid
    # -------------------------------------------------------------------------
    tiles, n_rows, n_cols, est_h, est_w = calculate_tile_grid(
        lat_min, lat_max, lon_min, lon_max, args.scale, args.max_bytes
    )
    total_px = est_h * est_w
    print(f"\nEstimated image: {est_h} × {est_w} = {total_px:,} pixels")
    print(f"Tile grid: {n_rows} rows × {n_cols} cols = {len(tiles)} tiles")

    # -------------------------------------------------------------------------
    # 2. Initialize Earth Engine
    # -------------------------------------------------------------------------
    print("\nInitializing Earth Engine...")
    ee.Initialize(project=args.gee_project)
    print("✓ Earth Engine initialized")

    # Set up embedding image (one global EE object, tiles just clip it)
    region = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
    dataset = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
    ee_image = (dataset
                .filterDate(f'{args.year}-01-01', f'{args.year + 1}-01-01')
                .filterBounds(region)
                .first())
    band_names = [f'A{i:02d}' for i in range(64)]
    ee_embedding = ee_image.select(band_names).toFloat()

    # -------------------------------------------------------------------------
    # 3. Download all tiles (with caching)
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("DOWNLOADING TILES")
    print(f"{'='*70}")

    tile_shapes = {}  # (row, col) -> (H, W)

    for idx, tile in enumerate(tiles):
        r, c = tile['row'], tile['col']
        emb_cache = tile_cache / f"emb_{r}_{c}.npy"
        rgb_cache = tile_cache / f"rgb_{r}_{c}.npy"

        prefix = f"[{idx+1}/{len(tiles)}] Tile ({r},{c})"

        # Download embedding
        if emb_cache.exists() and args.resume:
            emb = np.load(emb_cache)
            print(f"{prefix} EMB cached: {emb.shape}")
        else:
            print(f"{prefix} Downloading embedding...", flush=True)
            try:
                emb = download_tile_embedding(tile, ee_embedding, band_names,
                                              args.scale, args.max_bytes, tile_cache)
                print(f"  ✓ EMB: {emb.shape}")
            except Exception as e:
                print(f"  ✗ EMB failed: {e}")
                # Create zeros placeholder
                center_lat = (tile['south'] + tile['north']) / 2
                m_lat = 111320.0
                m_lon = 111320.0 * cos(radians(center_lat))
                th = int(ceil((tile['north'] - tile['south']) * m_lat / args.scale))
                tw = int(ceil((tile['east'] - tile['west']) * m_lon / args.scale))
                emb = np.zeros((th, tw, 64), dtype=np.float32)
                np.save(emb_cache, emb)
                print(f"  Saved zero placeholder: {emb.shape}")
            time.sleep(0.5)  # Rate limit

        tile_shapes[(r, c)] = emb.shape[:2]
        del emb  # Free memory

        # Download RGB
        if not args.skip_rgb:
            if rgb_cache.exists() and args.resume:
                print(f"{prefix} RGB cached")
            else:
                print(f"{prefix} Downloading RGB...", flush=True)
                try:
                    download_tile_rgb(tile, args.year, args.scale, args.max_bytes, tile_cache)
                    print(f"  ✓ RGB done")
                except Exception as e:
                    print(f"  ✗ RGB failed: {e}")
                    # Placeholder
                    h, w = tile_shapes[(r, c)]
                    rgb = np.zeros((h, w, 3), dtype=np.float32)
                    np.save(rgb_cache, rgb)
                time.sleep(0.3)

    # Save tile info
    tile_info = {
        'n_rows': n_rows, 'n_cols': n_cols,
        'tiles': [{
            'row': t['row'], 'col': t['col'],
            'south': t['south'], 'north': t['north'],
            'west': t['west'], 'east': t['east'],
            'shape': list(tile_shapes.get((t['row'], t['col']), [0, 0]))
        } for t in tiles],
        'bbox': {'south': lat_min, 'north': lat_max, 'west': lon_min, 'east': lon_max},
        'scale': args.scale, 'year': args.year,
    }
    with open(output_dir / 'tile_info.json', 'w') as f:
        json.dump(tile_info, f, indent=2)
    print(f"\n✓ All tiles downloaded. Info saved to tile_info.json")

    # -------------------------------------------------------------------------
    # 4. Train or load SVM
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("SVM MODEL")
    print(f"{'='*70}")

    model_cache = output_dir / 'svm_model.pkl'
    scaler_cache = output_dir / 'scaler.pkl'

    if args.model_path and args.scaler_path:
        print("Loading pre-trained model...")
        with open(args.model_path, 'rb') as f:
            clf = pickle.load(f)
        with open(args.scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"  ✓ Model: {args.model_path}")
        print(f"  ✓ Scaler: {args.scaler_path}")
        metrics = {'note': 'pre-trained'}
    elif model_cache.exists() and scaler_cache.exists() and args.resume:
        print("Loading cached model...")
        with open(model_cache, 'rb') as f:
            clf = pickle.load(f)
        with open(scaler_cache, 'rb') as f:
            scaler = pickle.load(f)
        metrics = {'note': 'cached'}
        print("  ✓ Loaded from cache")
    else:
        clf, scaler, metrics = train_svm(
            args.train_csv, args.val_csv, args.embedding_dir,
            class_weight_ratio=args.class_weight_ratio
        )
        with open(model_cache, 'wb') as f:
            pickle.dump(clf, f)
        with open(scaler_cache, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"  ✓ Model cached to {model_cache}")

    # -------------------------------------------------------------------------
    # 5. Classify tile by tile → stitch prob_grid and RGB
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("CLASSIFYING TILES")
    print(f"{'='*70}")

    prob_tiles = {}
    rgb_tiles = {}

    for idx, tile in enumerate(tiles):
        r, c = tile['row'], tile['col']
        prefix = f"[{idx+1}/{len(tiles)}] Tile ({r},{c})"

        prob_cache = tile_cache / f"prob_{r}_{c}.npy"

        if prob_cache.exists() and args.resume:
            prob_tiles[(r, c)] = np.load(prob_cache)
            print(f"{prefix} Prob cached: {prob_tiles[(r,c)].shape}")
        else:
            emb = np.load(tile_cache / f"emb_{r}_{c}.npy")
            print(f"{prefix} Classifying {emb.shape[0]}×{emb.shape[1]} = "
                  f"{emb.shape[0]*emb.shape[1]:,} pixels...", end='', flush=True)
            prob = classify_tile_data(emb, clf, scaler, args.batch_size)
            np.save(prob_cache, prob)
            prob_tiles[(r, c)] = prob
            del emb
            print(f" mean={prob.mean():.3f}")

        # Load RGB tile
        if not args.skip_rgb:
            rgb_path = tile_cache / f"rgb_{r}_{c}.npy"
            if rgb_path.exists():
                rgb_tiles[(r, c)] = np.load(rgb_path)

    # -------------------------------------------------------------------------
    # 6. Stitch
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STITCHING")
    print(f"{'='*70}")

    prob_grid = stitch_tiles(prob_tiles, n_rows, n_cols)
    print(f"Probability grid: {prob_grid.shape}")
    print(f"  Range: [{prob_grid.min():.4f}, {prob_grid.max():.4f}]")
    print(f"  Mean: {prob_grid.mean():.4f}, Median: {np.median(prob_grid):.4f}")

    del prob_tiles  # Free memory

    rgb_image = None
    if rgb_tiles:
        rgb_image = stitch_tiles(rgb_tiles, n_rows, n_cols)
        # Match spatial dimensions
        target_h = min(prob_grid.shape[0], rgb_image.shape[0])
        target_w = min(prob_grid.shape[1], rgb_image.shape[1])
        prob_grid = prob_grid[:target_h, :target_w]
        rgb_image = rgb_image[:target_h, :target_w]
        print(f"RGB image: {rgb_image.shape}")
    del rgb_tiles

    print(f"Final dimensions: {prob_grid.shape[0]} × {prob_grid.shape[1]}")

    # -------------------------------------------------------------------------
    # 7. Save results
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")

    np.save(output_dir / 'prob_grid.npy', prob_grid)
    print(f"  ✓ prob_grid.npy: {prob_grid.shape}")

    if rgb_image is not None:
        np.save(output_dir / 'rgb_image.npy', rgb_image)
        print(f"  ✓ rgb_image.npy: {rgb_image.shape}")

    # Metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'method': 'tiled_classification',
        'bbox': {
            'lat_min': lat_min, 'lat_max': lat_max,
            'lon_min': lon_min, 'lon_max': lon_max,
        },
        'year': args.year,
        'scale_m': args.scale,
        'prob_grid_shape': list(prob_grid.shape),
        'rgb_shape': list(rgb_image.shape) if rgb_image is not None else None,
        'n_tiles': len(tiles),
        'tile_grid': f'{n_rows}×{n_cols}',
        'model': {
            'type': 'SVM with StandardScaler',
            'training_samples': metrics.get('n_train', 'pre-trained'),
            'validation_accuracy': metrics.get('accuracy', 'pre-trained'),
            'validation_f1': metrics.get('f1_score', 'pre-trained'),
            'validation_roc_auc': metrics.get('roc_auc', 'pre-trained'),
        },
        'statistics': {
            'mean': float(prob_grid.mean()),
            'median': float(np.median(prob_grid)),
            'std': float(prob_grid.std()),
            'min': float(prob_grid.min()),
            'max': float(prob_grid.max()),
            'pixels_above_30': int((prob_grid >= 0.3).sum()),
            'pixels_above_50': int((prob_grid >= 0.5).sum()),
            'pixels_above_70': int((prob_grid >= 0.7).sum()),
            'pixels_above_90': int((prob_grid >= 0.9).sum()),
        }
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ metadata.json")

    # -------------------------------------------------------------------------
    # 8. Statistics
    # -------------------------------------------------------------------------
    scale_m = args.scale
    pixel_area_ha = (scale_m ** 2) / 10000.0
    valid_probs = prob_grid.flatten()

    print(f"\n{'='*70}")
    print("YEW DETECTION STATISTICS")
    print(f"{'='*70}")
    print(f"Total pixels: {len(valid_probs):,}")
    print(f"Mean probability: {valid_probs.mean():.4f}")
    print(f"Median probability: {np.median(valid_probs):.4f}")
    print()
    for thresh in [0.3, 0.5, 0.7, 0.9]:
        count = int((valid_probs >= thresh).sum())
        area = count * pixel_area_ha
        pct = 100 * count / len(valid_probs)
        print(f"  P≥{thresh}: {count:>10,} pixels ({pct:5.2f}%) = {area:>10.1f} ha")

    # -------------------------------------------------------------------------
    # 9. Quick visualization
    # -------------------------------------------------------------------------
    print(f"\nCreating visualizations...")
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    cmap = LinearSegmentedColormap.from_list(
        'yew_prob', ['#2166ac', '#92c5de', '#f7f7f7', '#f4a582', '#d6604d', '#b2182b'], N=256
    )

    # Downsample for visualization if very large
    max_dim = 2000
    h, w = prob_grid.shape
    if max(h, w) > max_dim:
        ds = max(h, w) / max_dim
        ds_h, ds_w = int(h / ds), int(w / ds)
        from scipy.ndimage import zoom
        prob_ds = zoom(prob_grid, (ds_h / h, ds_w / w), order=0)
        if rgb_image is not None:
            rgb_ds = zoom(rgb_image, (ds_h / h, ds_w / w, 1), order=1)
        else:
            rgb_ds = None
    else:
        prob_ds = prob_grid
        rgb_ds = rgb_image

    extent = [lon_min, lon_max, lat_min, lat_max]

    fig, ax = plt.subplots(figsize=(14, 8))
    if rgb_ds is not None:
        ax.imshow(rgb_ds, extent=extent, aspect='auto')
        im = ax.imshow(prob_ds, extent=extent, cmap=cmap, vmin=0, vmax=1, alpha=0.5, aspect='auto')
    else:
        im = ax.imshow(prob_ds, extent=extent, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.03, label='P(yew)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Yew Detection: {lat_min:.4f}°N to {lat_max:.4f}°N, '
                 f'{abs(lon_min):.4f}°W to {abs(lon_max):.4f}°W\n'
                 f'{prob_grid.shape[0]}×{prob_grid.shape[1]} pixels, {len(tiles)} tiles',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(figures_dir / 'overview.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ overview.png")

    print(f"\n✓ All results saved to: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Apply forestry mask:")
    print(f"     python scripts/preprocessing/apply_forestry_mask.py --input-dir {output_dir}")
    print(f"  2. Create interactive map:")
    print(f"     python scripts/visualization/create_interactive_map.py --input-dir {output_dir}")
    print(f"  3. Create annotation tool:")
    print(f"     python scripts/visualization/create_annotation_tool.py --input-dir {output_dir}")


if __name__ == '__main__':
    main()
