#!/usr/bin/env python3
"""
Classify every pixel in embedding tiles inside a bbox and save dense probability grid.

Produces:
 - probability grid numpy: `<output_dir>/prob_grid.npy`
 - overlaid PNG: `<output_dir>/prob_grid_overlay.png`

This script trains an SVM on combined train+val center pixels (64-d embeddings)
and then predicts for every pixel in each embedding tile placed into a
geographic composite matching the notebook logic.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.neighbors import NearestNeighbors
import logging
import traceback
import sys
from datetime import datetime


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--verbose', action='store_true',
                   help='Enable verbose (DEBUG) logging')
    p.add_argument('--embedding-dir',
                   default='data/ee_imagery/embedding_patches_64x64')
    p.add_argument(
        '--train-path', default='data/processed/train_split_filtered.csv')
    p.add_argument(
        '--val-path', default='data/processed/val_split_filtered.csv')
    p.add_argument('--bbox', nargs=4, type=float,
                   help='lat_min lat_max lon_min lon_max', default=None)
    p.add_argument(
        '--tile-cache', default='results/predictions/southern_vancouver_island/tile_cache')
    p.add_argument(
        '--output-dir', default='results/predictions/vancouver_island_dense')
    p.add_argument('--pixel-resolution', type=float,
                   default=10.0, help='meters per pixel')
    p.add_argument('--sample-size', type=int,
                   default=64, help='tile pixel size')
    p.add_argument('--bin-size', type=int, default=1,
                   help='bin pixels by taking mean over NxN blocks (1=no binning)')
    return p.parse_args()


def load_embedding_list(emb_dir):
    emb_dir = Path(emb_dir)
    files = list(emb_dir.glob('embedding_*.npy'))
    rows = []
    for f in files:
        try:
            _, lat_str, lon_str = f.stem.split('_')
            lat = float(lat_str)
            lon = float(lon_str)
            rows.append((f, lat, lon))
        except Exception:
            logging.getLogger(__name__).warning(
                f'Could not parse embedding filename: {f.name}')
            continue
    return rows


def extract_center_pixel_from_file(path, patch_size=64):
    try:
        arr = np.load(path)  # expected shape (H, W, C) or (C, H, W)
        # Support both (H,W,C) and (C,H,W) shapes defensively
        if arr.ndim == 3 and arr.shape[2] >= 1:
            # assume (H, W, C)
            H, W, C = arr.shape
            c = patch_size // 2
            if c >= H or c >= W:
                logging.getLogger(__name__).warning(
                    f'Patch too small for {path.name}: shape={arr.shape}')
                return None
            feat = arr[c, c, :]
            return feat
        elif arr.ndim == 3 and arr.shape[0] >= 1:
            # maybe (C, H, W)
            C, H, W = arr.shape
            c = patch_size // 2
            if c >= H or c >= W:
                logging.getLogger(__name__).warning(
                    f'Patch too small for {path.name}: shape={arr.shape}')
                return None
            feat = arr[:, c, c]
            return feat
        else:
            logging.getLogger(__name__).error(
                f'Unexpected array shape for {path.name}: {arr.shape}')
            return None
    except Exception:
        logging.getLogger(__name__).exception(
            f'Failed to load embedding: {path}')
        return None


def train_final_svm(train_csv, val_csv, emb_dir, patch_size=64):
    """Train SVM on center pixel embeddings without scaling."""
    logger = logging.getLogger(__name__)
    train_path = Path(train_csv)
    val_path = Path(val_csv)
    if not train_path.exists():
        logger.error(f'Training CSV not found: {train_csv}')
        raise FileNotFoundError(train_csv)
    if not val_path.exists():
        logger.error(f'Validation CSV not found: {val_csv}')
        raise FileNotFoundError(val_csv)

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df = pd.concat([df_train, df_val], ignore_index=True)

    X = []
    y = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Extracting train/val features'):
        lat = row.get('lat') or row.get('latitude')
        lon = row.get('lon') or row.get('longitude')
        if pd.isna(lat) or pd.isna(lon):
            continue
        feat = extract_center_pixel_from_file(
            Path(emb_dir) / f'embedding_{lat:.6f}_{lon:.6f}.npy', patch_size)
        if feat is None:
            continue
        X.append(feat)
        y.append(int(row['has_yew']))

    X = np.array(X)
    y = np.array(y)

    if X.size == 0:
        logger.error(
            'No training features extracted from embeddings. Aborting model training.')
        raise ValueError(
            'No training features found (check embedding paths and CSV coordinate fields).')

    # Clean invalid values but NO scaling - preserve relative tile scale
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    clf = SVC(kernel='rbf', probability=True,
              random_state=42, class_weight='balanced')
    clf.fit(X, y)

    logger.info(
        f'Trained SVM classifier on {len(X)} samples (no scaling); positive fraction={(y.sum()/len(y)):.3f}')

    return clf


def main():
    args = parse_args()
    # Configure logging
    log_level = logging.DEBUG if getattr(
        args, 'verbose', False) else logging.INFO
    logging.basicConfig(
        level=log_level, format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger(__name__)

    emb_dir = Path(args.embedding_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.bbox:
        if len(args.bbox) != 4:
            logger.error(
                '`--bbox` requires 4 floats: lat_min lat_max lon_min lon_max')
            raise ValueError('Invalid bbox argument')
        lat_min, lat_max, lon_min, lon_max = args.bbox
    else:
        lat_min, lat_max, lon_min, lon_max = 48.0, 50.9, -125.9, -123.0

    logger.info(
        f'Using bbox: lat {lat_min}..{lat_max}, lon {lon_min}..{lon_max}')

    logger.info(
        'Training final SVM on combined train+val (center pixel, no scaling)...')
    try:
        clf = train_final_svm(
            args.train_path, args.val_path, emb_dir, args.sample_size)
    except Exception as e:
        logger.exception('Model training failed')
        raise
    logger.info('Model trained.')

    bin_size = args.bin_size
    if bin_size > 1:
        logger.info(
            f'Using {bin_size}x{bin_size} pixel binning (mean pooling) for faster inference')

    all_emb = load_embedding_list(emb_dir)
    # filter to bbox
    tiles = [(p, lat, lon) for (p, lat, lon) in all_emb if (
        lat_min <= lat <= lat_max and lon_min <= lon <= lon_max)]
    if not tiles:
        logger.warning('No embedding tiles found in bbox')
        return

    lats = sorted({lat for (_, lat, _) in tiles}, reverse=True)
    lons = sorted({lon for (_, _, lon) in tiles})

    # spacing
    lat_spacing = lats[0] - lats[1] if len(lats) > 1 else 0.002
    lon_spacing = lons[1] - lons[0] if len(lons) > 1 else 0.002

    meters_per_degree_lat = 111000.0
    meters_per_degree_lon = 111000.0 * \
        np.cos(np.radians((lat_min + lat_max) / 2.0))

    lat_range_m = (lat_max - lat_min) * meters_per_degree_lat
    lon_range_m = (lon_max - lon_min) * meters_per_degree_lon

    sample_size = args.sample_size
    pixel_resolution = args.pixel_resolution

    composite_height = int(lat_range_m / pixel_resolution) + sample_size
    composite_width = int(lon_range_m / pixel_resolution) + sample_size

    logger.info(
        f'Composite dimensions: {composite_width} × {composite_height} pixels')

    # Use accumulation grids to average overlapping tiles
    prob_sum = np.zeros((composite_height, composite_width), dtype=np.float64)
    prob_count = np.zeros((composite_height, composite_width), dtype=np.int32)

    # Optionally build an RGB composite if tile cache exists
    composite = None
    composite_count = None
    tile_cache = Path(args.tile_cache)
    if tile_cache.exists():
        composite = np.zeros(
            (composite_height, composite_width, 3), dtype=np.float64)
        composite_count = np.zeros(
            (composite_height, composite_width), dtype=np.int32)

    tiles_loaded = 0

    # Process tiles in batches per tile to vectorize predictions
    for fpath, lat, lon in tqdm(tiles, desc='Processing tiles'):
        try:
            if not Path(fpath).exists():
                logger.warning(f'Embedding file missing: {fpath}')
                continue
            emb = np.load(fpath)  # expected (H, W, C)
            H, W, C = emb.shape

            # Apply binning if requested (mean pooling over NxN blocks)
            if bin_size > 1:
                # Pad if needed to make divisible by bin_size
                pad_h = (bin_size - H % bin_size) % bin_size
                pad_w = (bin_size - W % bin_size) % bin_size
                if pad_h > 0 or pad_w > 0:
                    emb = np.pad(
                        emb, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
                    H, W = emb.shape[0], emb.shape[1]
                # Reshape and take mean over bins
                new_h, new_w = H // bin_size, W // bin_size
                emb_binned = emb.reshape(new_h, bin_size, new_w, bin_size, C)
                emb_binned = emb_binned.mean(axis=(1, 3))  # (new_h, new_w, C)
                # Predict on binned features
                flat = emb_binned.reshape(-1, C)
                flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
                probs = clf.predict_proba(flat)[:, 1]
                probs_binned = probs.reshape(new_h, new_w)
                # Upsample back to original size using nearest neighbor
                probs2d = np.repeat(
                    np.repeat(probs_binned, bin_size, axis=0), bin_size, axis=1)
                # trim to original size
                probs2d = probs2d[:sample_size, :sample_size]
            else:
                # No binning - predict every pixel
                flat = emb.reshape(-1, C)
                flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
                probs = clf.predict_proba(flat)[:, 1]
                probs2d = probs.reshape(H, W)

            # compute pixel position
            lat_offset_m = (lat_max - lat) * meters_per_degree_lat
            lon_offset_m = (lon - lon_min) * meters_per_degree_lon
            y_center = int(lat_offset_m / pixel_resolution)
            x_center = int(lon_offset_m / pixel_resolution)
            y_start = y_center - sample_size // 2
            x_start = x_center - sample_size // 2

            # bounds check
            y0 = max(0, y_start)
            x0 = max(0, x_start)
            y1 = min(composite_height, y_start + sample_size)
            x1 = min(composite_width, x_start + sample_size)

            sy = y0 - y_start
            sx = x0 - x_start
            ey = sy + (y1 - y0)
            ex = sx + (x1 - x0)

            # Accumulate probabilities (will average later)
            prob_sum[y0:y1, x0:x1] += probs2d[sy:ey, sx:ex]
            prob_count[y0:y1, x0:x1] += 1

            # if composite available, try to load corresponding RGB tile
            if composite is not None:
                tile_path = tile_cache / f'tile_{lat:.6f}_{lon:.6f}.npy'
                if tile_path.exists():
                    try:
                        # (4, H, W) - typically Sentinel-2 bands
                        tile = np.load(tile_path)
                        if tile.shape[1] == sample_size and tile.shape[2] == sample_size:
                            # Extract RGB bands (B4=Red, B3=Green, B2=Blue for Sentinel-2)
                            rgb = tile[[2, 1, 0], :, :]  # Reorder to RGB
                            rgb = np.transpose(rgb, (1, 2, 0)).astype(
                                np.float64)  # (H, W, 3)
                            # Accumulate raw values (will average later)
                            composite[y0:y1, x0:x1] += rgb[sy:ey, sx:ex]
                            composite_count[y0:y1, x0:x1] += 1
                    except Exception:
                        logger.exception(
                            f'Failed to load/process tile {tile_path.name}')

            tiles_loaded += 1
        except Exception:
            logger.exception(f'Error processing {fpath}')

    logger.info(f'Placed {tiles_loaded}/{len(tiles)} tiles')

    # Average overlapping tiles to create smooth probability grid
    prob_grid = np.full((composite_height, composite_width),
                        np.nan, dtype=np.float32)
    valid_mask = prob_count > 0
    prob_grid[valid_mask] = (prob_sum[valid_mask] /
                             prob_count[valid_mask]).astype(np.float32)
    logger.info(
        f'Averaged overlapping tiles (max overlap: {prob_count.max()} tiles per pixel)')

    # Average overlapping RGB tiles
    if composite is not None and composite_count is not None:
        rgb_valid = composite_count > 0
        for c in range(3):
            composite[:, :, c][rgb_valid] /= composite_count[rgb_valid]
        composite = composite.astype(np.float32)

    out_npy = out_dir / 'prob_grid.npy'
    # Save raw grid before any optional filling
    out_raw = out_dir / 'prob_grid_raw.npy'
    np.save(out_raw, prob_grid)
    logger.info(f'Saved raw probability grid: {out_raw}')

    # Compute coverage
    total_pixels = prob_grid.size
    valid_pixels = int(valid_mask.sum())
    coverage = valid_pixels / total_pixels
    logger.info(f'Coverage: {valid_pixels}/{total_pixels} ({coverage:.3%})')

    # If coverage is very sparse, fill missing values with nearest-neighbor to aid visualization
    fill_threshold = 0.10
    if coverage < fill_threshold:
        try:
            logger.warning(
                f'Coverage below {fill_threshold:.0%}; performing nearest-neighbor fill for visualization')
            # Prepare coordinates
            yy, xx = np.where(valid_mask)
            coords_valid = np.column_stack((yy, xx))
            vals = prob_grid[yy, xx]

            yy_m, xx_m = np.where(~valid_mask)
            coords_missing = np.column_stack((yy_m, xx_m))

            if len(coords_valid) == 0:
                logger.error(
                    'No valid pixels available to perform nearest-neighbor fill')
            else:
                nbrs = NearestNeighbors(
                    n_neighbors=1, algorithm='auto').fit(coords_valid)
                dists, idxs = nbrs.kneighbors(coords_missing)
                filled_vals = vals[idxs[:, 0]]
                prob_grid[yy_m, xx_m] = filled_vals
                logger.info(
                    f'Filled {len(coords_missing)} missing pixels using nearest neighbor')
                # Save filled grid separately
                out_filled = out_dir / 'prob_grid_filled.npy'
                np.save(out_filled, prob_grid)
                logger.info(f'Saved filled probability grid: {out_filled}')
        except Exception:
            logger.exception(
                'Nearest-neighbor fill failed; leaving raw grid as-is')
    else:
        logger.info('Coverage adequate; skipping fill')

    # Apply global normalization to RGB composite to eliminate tile boundary lines
    if composite is not None:
        logger.info('Applying global normalization to RGB composite...')
        composite_norm = np.zeros_like(composite, dtype=np.float32)
        for b in range(3):
            band = composite[:, :, b]
            # Use global percentiles across entire composite (exclude zero/unfilled areas)
            valid_pix = band[band != 0]
            if len(valid_pix) > 0:
                p_low, p_high = np.percentile(valid_pix, [2, 98])
                band_clipped = np.clip(band, p_low, p_high)
                if p_high > p_low:
                    composite_norm[:, :, b] = (
                        band_clipped - p_low) / (p_high - p_low)
                else:
                    composite_norm[:, :, b] = 0.5
        # Convert to uint8 for display
        composite = (np.clip(composite_norm, 0, 1) * 255).astype(np.uint8)
        logger.info(
            f'RGB composite shape: {composite.shape}, dtype: {composite.dtype}')

    # =========================================================================
    # Compute Yew Population Statistics
    # =========================================================================
    logger.info('Computing yew population statistics...')

    valid_probs = prob_grid[np.isfinite(prob_grid)]
    stats = {}

    # Basic statistics
    stats['total_pixels'] = int(valid_probs.size)
    stats['mean_prob'] = float(np.mean(valid_probs))
    stats['median_prob'] = float(np.median(valid_probs))
    stats['std_prob'] = float(np.std(valid_probs))
    stats['min_prob'] = float(np.min(valid_probs))
    stats['max_prob'] = float(np.max(valid_probs))

    # Thresholded statistics
    for thresh in [0.3, 0.5, 0.7, 0.9]:
        n_above = int((valid_probs >= thresh).sum())
        pct_above = 100.0 * n_above / \
            len(valid_probs) if len(valid_probs) > 0 else 0
        stats[f'pixels_above_{int(thresh*100)}'] = n_above
        stats[f'pct_above_{int(thresh*100)}'] = pct_above

    # Area estimates (assuming 10m pixels, adjusted for bin_size)
    effective_resolution = pixel_resolution * bin_size
    pixel_area_m2 = effective_resolution ** 2
    pixel_area_ha = pixel_area_m2 / 10000.0

    stats['pixel_resolution_m'] = effective_resolution
    stats['total_area_ha'] = stats['total_pixels'] * pixel_area_ha
    stats['high_prob_area_ha'] = stats['pixels_above_70'] * pixel_area_ha
    stats['very_high_prob_area_ha'] = stats['pixels_above_90'] * pixel_area_ha

    # Bounding box info
    stats['lat_min'] = lat_min
    stats['lat_max'] = lat_max
    stats['lon_min'] = lon_min
    stats['lon_max'] = lon_max
    stats['n_tiles'] = tiles_loaded
    stats['bin_size'] = bin_size

    # Log statistics
    logger.info(f"  Total analyzed pixels: {stats['total_pixels']:,}")
    logger.info(f"  Mean probability: {stats['mean_prob']:.4f}")
    logger.info(f"  Median probability: {stats['median_prob']:.4f}")
    logger.info(
        f"  Pixels with P(yew) >= 0.5: {stats['pixels_above_50']:,} ({stats['pct_above_50']:.2f}%)")
    logger.info(
        f"  Pixels with P(yew) >= 0.7: {stats['pixels_above_70']:,} ({stats['pct_above_70']:.2f}%)")
    logger.info(
        f"  High probability area (>=70%): {stats['high_prob_area_ha']:.2f} ha")

    # =========================================================================
    # Create Visualizations and PDF Report
    # =========================================================================
    cmap = LinearSegmentedColormap.from_list(
        'yew_prob', ['#2166ac', '#92c5de', '#fddbc7', '#d6604d'], N=256)
    extent = [lon_min, lon_max, lat_min, lat_max]

    # Create output directory for figures
    fig_dir = out_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Calculate aspect ratio for the region
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min
    # Adjust for latitude (longitude degrees are smaller at higher latitudes)
    aspect_ratio = lon_range / lat_range * \
        np.cos(np.radians((lat_min + lat_max) / 2))

    # Figure 1: Three-panel view (vertical stack)
    panel_width = 10
    panel_height = panel_width / aspect_ratio
    fig1, axes = plt.subplots(3, 1, figsize=(
        panel_width, panel_height * 3 + 2))

    # Panel 1: Satellite image alone (RGB)
    if composite is not None:
        axes[0].imshow(composite, extent=extent, aspect='equal')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    axes[0].set_title('Satellite Imagery (RGB)')

    # Panel 2: Probability grid alone
    im2 = axes[1].imshow(prob_grid, extent=extent, cmap=cmap, vmin=0, vmax=1,
                         interpolation='bilinear', aspect='equal')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    axes[1].set_title('Yew Probability Map')
    plt.colorbar(im2, ax=axes[1], fraction=0.04, pad=0.02, label='P(yew)')

    # Panel 3: Overlay with lower alpha
    if composite is not None:
        axes[2].imshow(composite, extent=extent, aspect='equal')
    im3 = axes[2].imshow(prob_grid, extent=extent, cmap=cmap, alpha=0.8, vmin=0, vmax=1,
                         interpolation='bilinear', aspect='equal')
    axes[2].set_xlabel('Longitude')
    axes[2].set_ylabel('Latitude')
    axes[2].set_title('Overlay (α=0.4)')
    plt.colorbar(im3, ax=axes[2], fraction=0.04, pad=0.02, label='P(yew)')

    plt.suptitle(f'Yew Detection Analysis: {lat_min:.3f}°N to {lat_max:.3f}°N, {abs(lon_min):.3f}°W to {abs(lon_max):.3f}°W',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    three_panel_path = fig_dir / 'three_panel_view.png'
    fig1.savefig(three_panel_path, dpi=300, bbox_inches='tight')
    logger.info(f'Saved three-panel figure: {three_panel_path}')

    # Figure 2: Probability histogram
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.hist(valid_probs, bins=50, edgecolor='black',
             alpha=0.7, color='#2ca02c')
    ax2.axvline(stats['mean_prob'], color='red', linestyle='--',
                linewidth=2, label=f"Mean: {stats['mean_prob']:.3f}")
    ax2.axvline(stats['median_prob'], color='orange', linestyle='--',
                linewidth=2, label=f"Median: {stats['median_prob']:.3f}")
    ax2.axvline(0.5, color='purple', linestyle=':',
                linewidth=2, label='Threshold: 0.5')
    ax2.set_xlabel('Yew Probability', fontsize=12)
    ax2.set_ylabel('Pixel Count', fontsize=12)
    ax2.set_title('Distribution of Yew Probability Predictions', fontsize=14)
    ax2.legend()
    ax2.grid(alpha=0.3)

    hist_path = fig_dir / 'probability_histogram.png'
    fig2.savefig(hist_path, dpi=300, bbox_inches='tight')
    logger.info(f'Saved histogram: {hist_path}')

    # Figure 3: Cumulative distribution
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sorted_probs = np.sort(valid_probs)
    cdf = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs)
    ax3.plot(sorted_probs, cdf, linewidth=2, color='#1f77b4')
    ax3.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(0.5, color='purple', linestyle=':', linewidth=2, label='P=0.5')
    ax3.set_xlabel('Yew Probability Threshold', fontsize=12)
    ax3.set_ylabel('Cumulative Proportion of Pixels', fontsize=12)
    ax3.set_title('Cumulative Distribution of Yew Probabilities', fontsize=14)
    ax3.legend()
    ax3.grid(alpha=0.3)

    cdf_path = fig_dir / 'cumulative_distribution.png'
    fig3.savefig(cdf_path, dpi=300, bbox_inches='tight')
    logger.info(f'Saved CDF: {cdf_path}')

    # Figure 4: Threshold bar chart
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    thresholds = [30, 50, 70, 90]
    areas = [stats[f'pixels_above_{t}'] * pixel_area_ha for t in thresholds]
    colors = ['#fddbc7', '#f4a582', '#d6604d', '#b2182b']
    bars = ax4.bar([f'≥{t}%' for t in thresholds], areas,
                   color=colors, edgecolor='black')
    ax4.set_xlabel('Probability Threshold', fontsize=12)
    ax4.set_ylabel('Area (hectares)', fontsize=12)
    ax4.set_title('Estimated Area by Yew Probability Threshold', fontsize=14)
    ax4.grid(axis='y', alpha=0.3)
    for bar, area in zip(bars, areas):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(areas)*0.01,
                 f'{area:.1f} ha', ha='center', va='bottom', fontsize=10)

    bar_path = fig_dir / 'area_by_threshold.png'
    fig4.savefig(bar_path, dpi=300, bbox_inches='tight')
    logger.info(f'Saved bar chart: {bar_path}')

    # Save individual panels with preserved aspect ratio
    fig_img, ax_img = plt.subplots(figsize=(panel_width, panel_height))
    if composite is not None:
        ax_img.imshow(composite, extent=extent, aspect='equal')
    ax_img.set_xlabel('Longitude')
    ax_img.set_ylabel('Latitude')
    ax_img.set_title('Satellite Imagery (RGB)')
    img_only_path = fig_dir / 'satellite_image.png'
    fig_img.savefig(img_only_path, dpi=300, bbox_inches='tight')
    plt.close(fig_img)

    fig_prob, ax_prob = plt.subplots(figsize=(panel_width, panel_height))
    im_prob = ax_prob.imshow(prob_grid, extent=extent, cmap=cmap, vmin=0, vmax=1,
                             interpolation='bilinear', aspect='equal')
    ax_prob.set_xlabel('Longitude')
    ax_prob.set_ylabel('Latitude')
    ax_prob.set_title('Yew Probability Map')
    plt.colorbar(im_prob, ax=ax_prob, fraction=0.04, pad=0.02, label='P(yew)')
    prob_only_path = fig_dir / 'probability_map.png'
    fig_prob.savefig(prob_only_path, dpi=300, bbox_inches='tight')
    plt.close(fig_prob)

    fig_overlay, ax_overlay = plt.subplots(figsize=(panel_width, panel_height))
    if composite is not None:
        ax_overlay.imshow(composite, extent=extent, aspect='equal')
    im_overlay = ax_overlay.imshow(prob_grid, extent=extent, cmap=cmap, alpha=0.4, vmin=0, vmax=1,
                                   interpolation='bilinear', aspect='equal')
    ax_overlay.set_xlabel('Longitude')
    ax_overlay.set_ylabel('Latitude')
    ax_overlay.set_title('Yew Probability Overlay (α=0.4)')
    plt.colorbar(im_overlay, ax=ax_overlay,
                 fraction=0.04, pad=0.02, label='P(yew)')
    overlay_path = fig_dir / 'prob_grid_overlay.png'
    fig_overlay.savefig(overlay_path, dpi=300, bbox_inches='tight')
    plt.close(fig_overlay)

    # =========================================================================
    # Generate PDF Report
    # =========================================================================
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pdf_path = out_dir / f'yew_detection_report_{timestamp}.pdf'

    logger.info(f'Generating PDF report: {pdf_path}')

    with PdfPages(pdf_path) as pdf:
        # Title page
        fig_title = plt.figure(figsize=(11, 8.5))
        fig_title.text(0.5, 0.7, 'Yew Detection Analysis Report', fontsize=24, fontweight='bold',
                       ha='center', va='center')
        fig_title.text(
            0.5, 0.55, f'Region: {lat_min:.4f}°N to {lat_max:.4f}°N', fontsize=14, ha='center')
        fig_title.text(
            0.5, 0.50, f'         {abs(lon_min):.4f}°W to {abs(lon_max):.4f}°W', fontsize=14, ha='center')
        fig_title.text(
            0.5, 0.40, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', fontsize=12, ha='center')
        fig_title.text(
            0.5, 0.30, f'Tiles analyzed: {tiles_loaded} | Bin size: {bin_size}x{bin_size}', fontsize=12, ha='center')
        pdf.savefig(fig_title, bbox_inches='tight')
        plt.close(fig_title)

        # Statistics page
        fig_stats = plt.figure(figsize=(11, 8.5))
        stats_text = f"""
YEW DETECTION STATISTICS SUMMARY
{'='*50}

COVERAGE INFORMATION
  Total pixels analyzed:     {stats['total_pixels']:,}
  Effective pixel resolution: {stats['pixel_resolution_m']:.1f} m
  Total area analyzed:        {stats['total_area_ha']:.2f} hectares
  Number of tiles:            {stats['n_tiles']}

PROBABILITY DISTRIBUTION
  Mean probability:           {stats['mean_prob']:.4f}
  Median probability:         {stats['median_prob']:.4f}
  Standard deviation:         {stats['std_prob']:.4f}
  Minimum:                    {stats['min_prob']:.4f}
  Maximum:                    {stats['max_prob']:.4f}

THRESHOLDED RESULTS
  Pixels with P(yew) ≥ 30%:   {stats['pixels_above_30']:,} ({stats['pct_above_30']:.2f}%)
  Pixels with P(yew) ≥ 50%:   {stats['pixels_above_50']:,} ({stats['pct_above_50']:.2f}%)
  Pixels with P(yew) ≥ 70%:   {stats['pixels_above_70']:,} ({stats['pct_above_70']:.2f}%)
  Pixels with P(yew) ≥ 90%:   {stats['pixels_above_90']:,} ({stats['pct_above_90']:.2f}%)

ESTIMATED YEWS AREA
  High probability (≥70%):    {stats['high_prob_area_ha']:.2f} hectares
  Very high probability (≥90%): {stats['very_high_prob_area_ha']:.2f} hectares

BOUNDING BOX
  Latitude:  {lat_min:.6f} to {lat_max:.6f}
  Longitude: {lon_min:.6f} to {lon_max:.6f}
"""
        fig_stats.text(0.1, 0.95, stats_text, fontsize=11, fontfamily='monospace',
                       va='top', ha='left', transform=fig_stats.transFigure)
        pdf.savefig(fig_stats, bbox_inches='tight')
        plt.close(fig_stats)

        # Three-panel view
        pdf.savefig(fig1, bbox_inches='tight')

        # Histogram
        pdf.savefig(fig2, bbox_inches='tight')

        # CDF
        pdf.savefig(fig3, bbox_inches='tight')

        # Bar chart
        pdf.savefig(fig4, bbox_inches='tight')

    plt.close('all')
    logger.info(f'Saved PDF report: {pdf_path}')

    # Save statistics as JSON
    import json
    stats_path = out_dir / 'statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f'Saved statistics: {stats_path}')

    # Legacy overlay path (for backward compatibility)
    out_png = out_dir / 'prob_grid_overlay.png'
    logger.info(f'Saved overlay: {out_png}')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        logging.getLogger(__name__).exception(
            'Fatal error in classify_every_pixel_in_bbox')
        sys.exit(2)
