#!/usr/bin/env python3
"""
Create a KMZ ground-overlay file from the CWH 300k-point yew probability data.

Steps:
  1. Rasterize sample_predictions.csv onto a regular lon/lat grid
  2. Apply a Natural Earth land mask (removes ocean)
  3. Apply colormap with transparency for low-probability / no-data cells
  4. Export as a KMZ (zipped KML + PNG overlay) for Google Earth

Usage:
    python scripts/visualization/create_cwh_kmz.py \
        --input results/analysis/cwh_yew_population_300k/sample_predictions.csv \
        --output results/analysis/cwh_yew_population_300k/cwh_yew_300k.kmz \
        --resolution 0.03 \
        --threshold 0.0
"""

import argparse
import io
import json
import math
import sys
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import box


# ---------------------------------------------------------------------------
# Colourmap: transparent at 0 → yellow at 0.5 → dark green at 1.0
# ---------------------------------------------------------------------------
YEWCMAP = LinearSegmentedColormap.from_list(
    'yew',
    [
        (0.00, (0.10, 0.60, 0.10, 0.00)),   # P=0   → fully transparent
        (0.15, (0.10, 0.60, 0.10, 0.30)),   # P=0.15 → dim green, semi-transparent
        (0.30, (0.50, 0.80, 0.10, 0.70)),   # P=0.30 → yellow-green, mostly opaque
        (0.50, (1.00, 0.90, 0.00, 0.85)),   # P=0.50 → yellow
        (0.70, (0.80, 0.20, 0.10, 0.90)),   # P=0.70 → orange-red
        (1.00, (0.60, 0.00, 0.60, 0.95)),   # P=1.0  → purple
    ],
    N=256,
)


def download_land_polygons(cache_path='/tmp/ne_10m_land.geojson'):
    """Download Natural Earth 10m land polygons (once, then cache)."""
    cache = Path(cache_path)
    if cache.exists():
        print(f'  Using cached land polygons: {cache}')
    else:
        url = (
            'https://raw.githubusercontent.com/nvkelso/natural-earth-vector'
            '/master/geojson/ne_10m_land.geojson'
        )
        print(f'  Downloading 10m land polygons from naturalearth …')
        urllib.request.urlretrieve(url, cache_path)
        print(f'  Saved to {cache}')
    gdf = gpd.read_file(cache)
    return gdf


def rasterize_points(df, resolution=0.03, padding=0.1):
    """
    Bin sample points onto a regular lon/lat grid and compute the mean
    yew probability per cell.

    Returns (grid_prob, grid_count, lon_min, lat_min, lon_max, lat_max)
    """
    lon_min = df['lon'].min() - padding
    lon_max = df['lon'].max() + padding
    lat_min = df['lat'].min() - padding
    lat_max = df['lat'].max() + padding

    n_lon = int(math.ceil((lon_max - lon_min) / resolution))
    n_lat = int(math.ceil((lat_max - lat_min) / resolution))
    print(f'  Grid: {n_lon} × {n_lat} cells at {resolution}° resolution')

    # Map each point to a grid cell
    col_idx = ((df['lon'] - lon_min) / resolution).astype(int).clip(0, n_lon - 1)
    row_idx = ((df['lat'] - lat_min) / resolution).astype(int).clip(0, n_lat - 1)

    prob_sum = np.zeros((n_lat, n_lon), dtype=np.float64)
    count    = np.zeros((n_lat, n_lon), dtype=np.int32)

    np.add.at(prob_sum, (row_idx.values, col_idx.values), df['prob'].values)
    np.add.at(count,    (row_idx.values, col_idx.values), 1)

    with np.errstate(invalid='ignore'):
        grid_prob = np.where(count > 0, prob_sum / count, np.nan)

    return grid_prob, count, lon_min, lat_min, lon_max, lat_max


def build_land_mask(land_gdf, lon_min, lat_min, lon_max, lat_max, n_lon, n_lat, resolution):
    """
    Build a boolean raster (True = land) from land polygons resampled to the
    prediction grid.

    Works by testing the centre of each grid cell against the union of all
    land polygons.
    """
    print('  Building land mask …')

    # Clip land polygons to the bbox for speed
    bbox = box(lon_min, lat_min, lon_max, lat_max)
    land_clipped = land_gdf[land_gdf.intersects(bbox)].copy()
    if land_clipped.empty:
        print('  WARNING: no land polygons cover this bbox — skipping mask')
        return None

    # Build array of cell-centre coordinates
    lons = lon_min + (np.arange(n_lon) + 0.5) * resolution
    lats = lat_min + (np.arange(n_lat) + 0.5) * resolution
    lon_grid, lat_grid = np.meshgrid(lons, lats)   # shape (n_lat, n_lon)

    # Test all points against each land polygon and union the results
    lon_flat = lon_grid.ravel()
    lat_flat = lat_grid.ravel()
    land_mask = np.zeros(lon_flat.shape, dtype=bool)

    for geom in land_clipped.geometry:
        try:
            land_mask |= shapely.contains_xy(geom, lon_flat, lat_flat)
        except Exception:
            pass

    return land_mask.reshape(n_lat, n_lon)


def grid_to_rgba(grid_prob, land_mask, cmap, vmin=0.0, vmax=1.0, min_count_mask=None):
    """Convert probability grid to RGBA image array, zeroing out ocean cells."""
    norm_prob = np.clip((grid_prob - vmin) / (vmax - vmin), 0, 1)

    # Map probabilities through colormap → RGBA float [0,1]
    rgba = cmap(norm_prob)  # shape (n_lat, n_lon, 4)

    # NaN cells → fully transparent
    nan_mask = np.isnan(grid_prob)
    rgba[nan_mask, 3] = 0

    # Ocean cells → fully transparent
    if land_mask is not None:
        rgba[~land_mask, 3] = 0

    # Flip vertically: our grid has row 0 = south → PNG expects row 0 = north
    rgba = rgba[::-1, :, :]

    return (rgba * 255).astype(np.uint8)


def build_kml(name, lon_min, lat_min, lon_max, lat_max, overlay_filename='overlay.png'):
    """Return a KML string for a GroundOverlay."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Folder>
    <name>{name}</name>
    <GroundOverlay>
      <name>{name}</name>
      <description>Yew probability map — CWH BEC Zone, BC (tiled GEE sampling, n=209k)</description>
      <Icon>
        <href>{overlay_filename}</href>
        <viewBoundScale>0.75</viewBoundScale>
      </Icon>
      <LatLonBox>
        <north>{lat_max}</north>
        <south>{lat_min}</south>
        <east>{lon_max}</east>
        <west>{lon_min}</west>
      </LatLonBox>
      <drawOrder>0</drawOrder>
    </GroundOverlay>
  </Folder>
</kml>"""


def build_legend():
    """Return a small PNG legend as bytes."""
    fig, ax = plt.subplots(figsize=(1.8, 3.2))
    gradient = np.linspace(0, 1, 256).reshape(256, 1)
    ax.imshow(gradient[::-1], aspect='auto', cmap=YEWCMAP, vmin=0, vmax=1,
              extent=[0, 1, 0, 1])
    ax.set_yticks([0, 0.3, 0.5, 0.7, 1.0])
    ax.set_yticklabels(['0%', '30%', '50%', '70%', '100%'], fontsize=8)
    ax.set_xticks([])
    ax.set_title('P(yew)', fontsize=9)
    ax.yaxis.set_label_position('right')
    ax.yaxis.tick_right()
    plt.tight_layout(pad=0.4)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor='white', transparent=False)
    plt.close()
    buf.seek(0)
    return buf.read()


def create_cwh_kmz(input_csv, output_kmz, resolution=0.03, threshold=0.0,
                   name='CWH Yew Probability', land_cache='/tmp/ne_110m_land.geojson'):
    output_kmz = Path(output_kmz)
    output_kmz.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load predictions
    # ------------------------------------------------------------------
    print(f'\n  Loading {input_csv} …')
    df = pd.read_csv(input_csv)
    print(f'  {len(df):,} sample points, lat {df.lat.min():.2f}–{df.lat.max():.2f}, '
          f'lon {df.lon.min():.2f}–{df.lon.max():.2f}')

    # Optionally threshold: set cells below threshold to NaN in the viz
    # (we still rasterize all points for accurate cell means)

    # ------------------------------------------------------------------
    # 2. Rasterize
    # ------------------------------------------------------------------
    # 2. Load land polygons and filter out ocean points
    # ------------------------------------------------------------------
    print('\n  Loading land polygons for water masking …')
    land_gdf = None
    try:
        land_gdf = download_land_polygons(land_cache)
    except Exception as e:
        print(f'  WARNING: land download failed ({e})')

    if land_gdf is not None:
        # Clip to bbox for speed
        from shapely.geometry import box as shapely_box
        bbox = shapely_box(df['lon'].min() - 0.5, df['lat'].min() - 0.5,
                           df['lon'].max() + 0.5, df['lat'].max() + 0.5)
        land_clipped = land_gdf[land_gdf.intersects(bbox)].copy()
        if not land_clipped.empty:
            land_union = land_clipped.union_all()
            print(f'  Testing {len(df):,} points against land polygons …')
            on_land = shapely.contains_xy(land_union, df['lon'].values, df['lat'].values)
            n_water = (~on_land).sum()
            df = df[on_land].copy()
            print(f'  Removed {n_water:,} water points, {len(df):,} land points remain')
        else:
            print(f'  WARNING: no land polygons in bbox — skipping water filter')

    # ------------------------------------------------------------------
    # 3. Rasterize
    # ------------------------------------------------------------------
    print('\n  Rasterizing sample points …')
    grid_prob, count, lon_min, lat_min, lon_max, lat_max = rasterize_points(
        df, resolution=resolution)
    n_lat, n_lon = grid_prob.shape
    print(f'  Grid shape: {n_lat} × {n_lon}, '
          f'non-empty cells: {(count > 0).sum():,} / {n_lat * n_lon:,}')

    # ------------------------------------------------------------------
    # 4. Apply threshold: cells below threshold → transparent
    # ------------------------------------------------------------------
    if threshold > 0:
        grid_prob_vis = grid_prob.copy()
        grid_prob_vis[grid_prob_vis < threshold] = np.nan
        print(f'  Threshold P≥{threshold}: '
              f'{(grid_prob >= threshold).sum():,} cells shown')
    else:
        grid_prob_vis = grid_prob

    # ------------------------------------------------------------------
    # 5. Build RGBA image (no separate land mask needed — already filtered)
    # ------------------------------------------------------------------
    print('\n  Building RGBA overlay image …')
    rgba = grid_to_rgba(grid_prob_vis, None, YEWCMAP)
    print(f'  Overlay size: {rgba.shape[1]} × {rgba.shape[0]} px')

    # ------------------------------------------------------------------
    # 6. Encode overlay PNG
    # ------------------------------------------------------------------
    from PIL import Image as PILImage
    img = PILImage.fromarray(rgba, mode='RGBA')
    overlay_buf = io.BytesIO()
    img.save(overlay_buf, format='PNG', compress_level=6)
    overlay_bytes = overlay_buf.getvalue()
    print(f'  PNG size: {len(overlay_bytes) / 1024:.0f} KB')

    # ------------------------------------------------------------------
    # 7. Package KMZ
    # ------------------------------------------------------------------
    kml_str = build_kml(name, lon_min, lat_min, lon_max, lat_max)
    legend_bytes = build_legend()

    print(f'\n  Writing KMZ: {output_kmz}')
    with zipfile.ZipFile(output_kmz, 'w', zipfile.ZIP_DEFLATED) as kmz:
        kmz.writestr('doc.kml', kml_str)
        kmz.writestr('overlay.png', overlay_bytes)
        kmz.writestr('legend.png', legend_bytes)

    print(f'  ✓ Done — {output_kmz.stat().st_size / 1024:.0f} KB')
    print(f'\n  Open in Google Earth: double-click {output_kmz.name}')
    return output_kmz


def parse_args():
    p = argparse.ArgumentParser(description='Create KMZ from CWH sample predictions')
    p.add_argument('--input', default='results/analysis/cwh_yew_population_300k/sample_predictions.csv')
    p.add_argument('--output', default='results/analysis/cwh_yew_population_300k/cwh_yew_300k.kmz')
    p.add_argument('--resolution', type=float, default=0.03,
                   help='Grid cell size in degrees (default: 0.03 ≈ 3 km)')
    p.add_argument('--threshold', type=float, default=0.0,
                   help='Hide cells below this probability (default: 0 = show all)')
    p.add_argument('--name', default='CWH Yew Probability (300k pts)')
    p.add_argument('--land-cache', default='/tmp/ne_10m_land.geojson',
                   help='Path to cache Natural Earth land GeoJSON')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    create_cwh_kmz(
        input_csv=args.input,
        output_kmz=args.output,
        resolution=args.resolution,
        threshold=args.threshold,
        name=args.name,
        land_cache=args.land_cache,
    )
