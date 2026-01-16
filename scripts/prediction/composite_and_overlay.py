#!/usr/bin/env python3
"""
Compose satellite tiles from a tile cache and overlay prediction points.

Assumes tiles are NumPy files named `tile_{lat:.6f}_{lon:.6f}.npy` with shape (bands, H, W).
This script:
 - Scans the tile cache for tiles within a bbox
 - Builds a composite at tile resolution (tile_size per tile)
 - Normalizes bands using global percentiles
 - Overlays prediction points and saves output images

Usage:
    python scripts/prediction/composite_and_overlay.py --tile-cache results/predictions/southern_vancouver_island/tile_cache \
        --pred-csv results/predictions/center_pixel_method/center_pixel_predictions_*.csv \
        --bbox 48.0 50.9 -125.9 -123.0 --output results/predictions/vancouver_island

"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def parse_tile_filename(name):
    # expects: tile_{lat}_{lon}.npy
    stem = Path(name).stem
    parts = stem.split('_')
    if len(parts) >= 3 and parts[0] == 'tile':
        try:
            lat = float(parts[1])
            lon = float(parts[2])
            return lat, lon
        except:
            return None
    return None


def build_composite(tile_cache, bbox, max_pixels=20000):
    """Return composite RGB uint8 image and extent (lon_min, lon_max, lat_min, lat_max)
    and mapping info for overlaying points.
    """
    files = list(Path(tile_cache).glob('tile_*.npy'))
    tiles = []
    for f in files:
        v = parse_tile_filename(f.name)
        if v is None:
            continue
        lat, lon = v
        if lat < bbox[0] or lat > bbox[1] or lon < bbox[2] or lon > bbox[3]:
            continue
        tiles.append((lat, lon, f))

    if len(tiles) == 0:
        raise SystemExit('No tiles found in cache within bbox')

    # Use geographic placement similar to visualize_yew_predictions notebook
    # Determine meters per degree and pixel resolution (assume 10m/pixel)
    meters_per_degree_lat = 111000
    meters_per_degree_lon = 111000 * \
        np.cos(np.radians((bbox[0] + bbox[1]) / 2))
    pixel_resolution = 10  # meters per pixel

    # Load a sample tile to get tile size
    sample_tile = np.load(tiles[0][2])
    bands, H, W = sample_tile.shape
    tile_h = H
    tile_w = W

    # Calculate geographic extent in meters
    lat_min, lat_max, lon_min, lon_max = bbox
    lat_range_m = (lat_max - lat_min) * meters_per_degree_lat
    lon_range_m = (lon_max - lon_min) * meters_per_degree_lon

    composite_height = int(lat_range_m / pixel_resolution) + tile_h
    composite_width = int(lon_range_m / pixel_resolution) + tile_w

    # Downscale if composite too large
    scale = 1
    if composite_height > max_pixels or composite_width > max_pixels:
        scale = max(1, int(max(composite_height / max_pixels,
                    composite_width / max_pixels)) + 1)
        composite_height = composite_height // scale
        composite_width = composite_width // scale

    print(
        f'Composing composite {composite_width}x{composite_height} (scale={scale}) using geographic placement')

    composite = np.zeros(
        (composite_height, composite_width, 3), dtype=np.uint8)
    prob_grid = np.full((composite_height, composite_width),
                        np.nan, dtype=np.float32)

    # Gather tiles for normalization
    all_tiles_rgb = []
    placements = []  # (lat, lon, prob, rgb_uint8, y_start, x_start)

    for lat, lon, path in tiles:
        arr = np.load(path)
        if arr.shape[0] >= 3:
            rgb = np.transpose(arr[[2, 1, 0], :, :],
                               (1, 2, 0)).astype(np.float32)
        else:
            rgb = np.transpose(
                np.repeat(arr[0:1, ...], 3, axis=0), (1, 2, 0)).astype(np.float32)

        if scale > 1:
            rgb = rgb[::scale, ::scale, :]

        all_tiles_rgb.append(rgb)

        # compute placement in composite
        lat_offset_m = (lat_max - lat) * meters_per_degree_lat
        lon_offset_m = (lon - lon_min) * meters_per_degree_lon
        y_center = int(lat_offset_m / pixel_resolution) // scale
        x_center = int(lon_offset_m / pixel_resolution) // scale
        y_start = y_center - (rgb.shape[0] // 2)
        x_start = x_center - (rgb.shape[1] // 2)

        placements.append((lat, lon, path, rgb, y_start, x_start))

    # Compute percentiles for normalization
    percentiles = []
    for b in range(3):
        data = np.concatenate([t[:, :, b].flatten() for t in all_tiles_rgb])
        p_low, p_high = np.percentile(data, [2, 98])
        percentiles.append((p_low, p_high))

    # Place tiles into composite using computed placements
    for lat, lon, path, rgb, y_start, x_start in placements:
        rgb_norm = np.zeros_like(rgb, dtype=np.float32)
        for b in range(3):
            p_low, p_high = percentiles[b]
            band = rgb[:, :, b]
            band = np.clip(band, p_low, p_high)
            if p_high > p_low:
                band = (band - p_low) / (p_high - p_low)
            else:
                band = np.clip(band, 0, 1)
            rgb_norm[:, :, b] = band

        rgb_uint8 = (rgb_norm * 255).astype(np.uint8)

        y0 = max(0, y_start)
        x0 = max(0, x_start)
        y1 = min(composite_height, y0 + rgb_uint8.shape[0])
        x1 = min(composite_width, x0 + rgb_uint8.shape[1])

        dy = y1 - y0
        dx = x1 - x0

        if dy > 0 and dx > 0:
            composite[y0:y1, x0:x1] = rgb_uint8[0:dy, 0:dx]

    # extent for imshow: lon_min, lon_max, lat_min, lat_max
    extent = (lon_min, lon_max, lat_min, lat_max)
    return composite, extent


def overlay_predictions_on_composite(composite, extent, pred_df, output_dir):
    cmap = LinearSegmentedColormap.from_list('yew_prob', [
                                             '#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#fddbc7', '#f4a582', '#d6604d', '#b2182b'], N=256)
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(composite, extent=[extent[0], extent[1],
              extent[2], extent[3]], aspect='auto')
    sc = ax.scatter(pred_df['longitude'], pred_df['latitude'], c=pred_df['yew_probability'],
                    cmap=cmap, s=40, vmin=0, vmax=1, edgecolors='black', linewidths=0.2)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('P(Yew)')
    out = Path(output_dir) / 'vancouver_island_composite_overlay.png'
    plt.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tile-cache', type=str, required=True)
    parser.add_argument('--pred-csv', type=str, default=None)
    parser.add_argument('--pred-dir', type=str,
                        default='results/predictions/center_pixel_method')
    parser.add_argument('--bbox', nargs=4, type=float,
                        required=True, help='lat_min lat_max lon_min lon_max')
    parser.add_argument('--output', type=str,
                        default='results/predictions/vancouver_island')
    args = parser.parse_args()

    tile_cache = Path(args.tile_cache)
    if not tile_cache.exists():
        raise SystemExit('Tile cache not found')

    pred_file = Path(args.pred_csv) if args.pred_csv else None
    if pred_file is None:
        # find latest
        files = sorted(Path(args.pred_dir).glob(
            'center_pixel_predictions_*.csv'))
        if not files:
            raise SystemExit('No prediction CSV found')
        pred_file = files[-1]

    pred_df = pd.read_csv(pred_file)
    lat_min, lat_max, lon_min, lon_max = args.bbox
    bbox = (lat_min, lat_max, lon_min, lon_max)

    # filter preds inside bbox
    preds_sub = pred_df[(pred_df['latitude'] >= lat_min) & (pred_df['latitude'] <= lat_max) & (
        pred_df['longitude'] >= lon_min) & (pred_df['longitude'] <= lon_max)].copy()
    if preds_sub.empty:
        raise SystemExit('No predictions within bbox')

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    composite, extent = build_composite(tile_cache, bbox)
    comp_path = out_dir / 'vancouver_island_composite.png'
    # Use matplotlib to save PNG to avoid external dependency on imageio
    from matplotlib.image import imsave
    imsave(comp_path, composite)
    print('Saved composite to', comp_path)

    overlay_out = overlay_predictions_on_composite(
        composite, extent, preds_sub, out_dir)
    print('Saved overlay to', overlay_out)


if __name__ == '__main__':
    main()
