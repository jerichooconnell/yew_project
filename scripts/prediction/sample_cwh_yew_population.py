#!/usr/bin/env python3
"""
Stratified Random Sampling: Yew Population in the CWH BEC Zone

Instead of downloading every pixel tile across coastal BC, this script:
  1. Downloads CWH BEC zone polygons from BC government's public WFS service
  2. Creates a stratified random sample of N points across the zone
  3. Uses GEE's .sample() to extract the 64-band Prithvi embedding at each point
     (single-pixel extraction — much faster than full tiles)
  4. Applies the trained MLP model to classify each point
  5. Reports population estimates with 95% confidence intervals

The CWH (Coastal Western Hemlock) zone covers ~4 million ha of coastal BC.
A sample of N=3000 points gives a margin of error of ±1.8% at 95% confidence.

Usage:
    python scripts/prediction/sample_cwh_yew_population.py \
        --n-samples 3000 \
        --model-dir results/predictions/south_vi_large \
        --output-dir results/analysis/cwh_yew_population \
        --year 2024
"""

import argparse
import json
import math
import pickle
import time
from datetime import datetime
from pathlib import Path

import ee
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from shapely.geometry import Point, MultiPolygon
from tqdm import tqdm


# =============================================================================
# MLP model (must match classify_tiled_gpu.py)
# =============================================================================

class YewMLP(nn.Module):
    def __init__(self, input_dim=64, hidden_dims=(128, 64, 32)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Estimate yew population in BC CWH zone via stratified sampling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--n-samples', type=int, default=3000,
                        help='Number of random sample points (default: 3000, gives ±1.8%% margin)')
    parser.add_argument('--model-dir', type=str,
                        default='results/predictions/south_vi_large',
                        help='Directory containing mlp_model.pth and mlp_scaler.pkl')
    parser.add_argument('--output-dir', type=str,
                        default='results/analysis/cwh_yew_population')
    parser.add_argument('--year', type=int, default=2024)
    parser.add_argument('--gee-project', type=str, default='carbon-storm-206002')
    parser.add_argument('--gee-batch-size', type=int, default=500,
                        help='Points per GEE extraction batch (default: 500)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bec-zone', type=str, default='CWH',
                        help='BEC zone code to analyze (default: CWH)')
    parser.add_argument('--boundary-gpkg', type=str, default=None,
                        help='Path to a pre-built boundary GeoPackage (e.g. from '
                             'generate_cwh_negatives.py). Overrides WFS download.')
    parser.add_argument('--min-canopy-cover', type=int, default=0,
                        help='Minimum canopy cover %% for sampling mask (0 = no filter)')
    parser.add_argument('--tiled-sampling', action='store_true',
                        help='Use GEE tiled random sampling (much faster for large N). '
                             'Divides the zone bbox into a grid and calls GEE sample() '
                             'per tile — ideal for N > 10,000.')
    parser.add_argument('--grid-cols', type=int, default=10,
                        help='Grid columns for --tiled-sampling (default: 10)')
    parser.add_argument('--grid-rows', type=int, default=15,
                        help='Grid rows for --tiled-sampling (default: 15)')
    return parser.parse_args()


# =============================================================================
# Step 1: Download CWH zone from BC government WFS
# =============================================================================

def download_bec_zone(zone_code, output_dir, max_features=5000):
    """
    Download BEC zone polygons from BC government's public OGC WFS service.
    Uses the generalized 1:2,000,000 scale layer (manageable size).

    Caches the result as a GeoPackage to avoid re-downloading.
    """
    cache_path = output_dir / f'bec_{zone_code.lower()}_polygons.gpkg'

    if cache_path.exists():
        print(f"  Loading cached BEC {zone_code} polygons from {cache_path.name}...")
        gdf = gpd.read_file(cache_path)
        print(f"  Loaded {len(gdf)} polygons, {gdf.geometry.area.sum() / 1e10:.1f} million ha")
        return gdf

    print(f"  Downloading BEC {zone_code} zone from BC government WFS...")
    print("  (Using BEC generalized 1:2M scale layer)")

    # BC government public WFS endpoint for BEC zones (generalized)
    # Layer: WHSE_ECOLOGY_CLIMATIC.BEC_BIOGEOCLIMATIC_POLY
    # Alternatively use the 1:2M generalized version for faster download
    base_url = "https://openmaps.gov.bc.ca/geo/pub/wfs"
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeName": "pub:WHSE_ECOLOGY_CLIMATIC.BEC_BIOGEOCLIMATIC_POLY",
        "outputFormat": "application/json",
        "CQL_FILTER": f"ZONE='{zone_code}'",
        "count": max_features,
        "srsName": "EPSG:4326",
    }

    print(f"  URL: {base_url}")
    print(f"  Filter: ZONE='{zone_code}' (up to {max_features} features)")
    print("  This may take 1-5 minutes depending on zone size...")

    try:
        response = requests.get(base_url, params=params, timeout=300, stream=True)
        response.raise_for_status()

        data = response.json()
        n_features = len(data.get('features', []))
        print(f"  Downloaded {n_features} polygon features")

        if n_features == 0:
            raise ValueError(f"No features returned for BEC zone '{zone_code}'")

        gdf = gpd.GeoDataFrame.from_features(data['features'], crs='EPSG:4326')

        # Keep relevant columns
        keep_cols = [c for c in ['ZONE', 'SUBZONE', 'VARIANT', 'MAP_LABEL',
                                  'BGC_LABEL', 'FEATURE_AREA_SQM', 'geometry']
                     if c in gdf.columns]
        gdf = gdf[keep_cols].copy()

        # Add area in hectares
        gdf_proj = gdf.to_crs('EPSG:3005')
        gdf['area_ha'] = gdf_proj.geometry.area / 10000

        total_area_ha = gdf['area_ha'].sum()
        print(f"  Total {zone_code} zone area: {total_area_ha:,.0f} ha ({total_area_ha/1e6:.2f} million ha)")

        gdf.to_file(cache_path, driver='GPKG')
        print(f"  Cached to {cache_path.name}")
        return gdf

    except requests.exceptions.Timeout:
        print("\n  WFS request timed out. Trying simplified approach with bounding box subdivisions...")
        raise

    except Exception as e:
        print(f"\n  WFS download failed: {e}")
        print("  Falling back to manual CWH zone approximate boundary...")
        raise


def create_cwh_approximate_polygon():
    """
    Fallback: approximate CWH zone using known geographic extents.
    The CWH zone covers coastal BC from ~lat 48.3 to 55.5, primarily
    west of the Coast Mountains (roughly the first 50-150km inland).
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    import shapely

    # Approximate CWH sub-regions as rectangles
    # These cover the major CWH areas: Vancouver Island, Sunshine Coast,
    # Prince Rupert area, and the inter-mountain coastal corridors
    regions = [
        # Southern Vancouver Island + Gulf Islands
        Polygon([(-125.0, 48.3), (-123.0, 48.3), (-123.0, 49.0), (-125.0, 49.0)]),
        # Northern Vancouver Island
        Polygon([(-128.5, 49.0), (-124.0, 49.0), (-124.0, 50.8), (-128.5, 50.8)]),
        # Sunshine Coast / Howe Sound / Squamish
        Polygon([(-124.0, 49.3), (-122.0, 49.3), (-122.0, 50.5), (-124.0, 50.5)]),
        # Central Coast (Bella Coola, Rivers Inlet area)
        Polygon([(-128.5, 50.8), (-124.5, 50.8), (-124.5, 52.5), (-128.5, 52.5)]),
        # North Coast (Prince Rupert, Haida Gwaii excluded)
        Polygon([(-131.0, 52.5), (-127.0, 52.5), (-127.0, 55.5), (-131.0, 55.5)]),
        # Haida Gwaii (Queen Charlotte Islands)
        Polygon([(-133.0, 52.0), (-130.5, 52.0), (-130.5, 54.5), (-133.0, 54.5)]),
    ]
    return unary_union(regions)


# =============================================================================
# Step 2: Stratified random sampling
# =============================================================================

def stratified_sample_points(gdf, n_samples, seed=42):
    """
    Generate stratified random sample points within the BEC zone polygons.

    Stratification: allocate samples proportionally to polygon area,
    ensuring geographic coverage across the zone.
    Also adds a geographic grid stratification to avoid clustering.
    """
    rng = np.random.default_rng(seed)

    # Sort by area for stability
    gdf = gdf.copy().reset_index(drop=True)

    # Add area in ha if not present
    if 'area_ha' not in gdf.columns:
        gdf_proj = gdf.to_crs('EPSG:3005')
        gdf['area_ha'] = gdf_proj.geometry.area / 10000

    total_area = gdf['area_ha'].sum()
    print(f"  Total zone area: {total_area:,.0f} ha")
    print(f"  Number of polygons: {len(gdf)}")

    # Get bounding box of the entire zone
    bbox = gdf.total_bounds  # [minx, miny, maxx, maxy]
    lon_min, lat_min, lon_max, lat_max = bbox

    # Create a grid of cells for geographic stratification
    # Aim for ~50-100 cells to spread samples across the zone
    grid_cols = 10
    grid_rows = 10
    n_cells = grid_cols * grid_rows
    samples_per_cell = max(1, n_samples // n_cells)

    lon_step = (lon_max - lon_min) / grid_cols
    lat_step = (lat_max - lat_min) / grid_rows

    print(f"  Stratifying over {grid_cols}×{grid_rows} geographic grid ({n_cells} cells)")
    print(f"  Target {samples_per_cell} samples per occupied cell")

    # Dissolve all polygons into one geometry for point-in-polygon test
    print("  Dissolving polygons for sampling...")
    zone_geometry = gdf.geometry.unary_union

    all_points = []
    points_per_cell = {}

    # Sample per grid cell
    total_attempts = 0
    max_attempts = n_samples * 100

    print(f"  Generating {n_samples} sample points...")

    # Simple rejection sampling within bounding box
    # For each cell, try to get samples_per_cell points
    for row in range(grid_rows):
        for col in range(grid_cols):
            cell_lon_min = lon_min + col * lon_step
            cell_lon_max = lon_min + (col + 1) * lon_step
            cell_lat_min = lat_min + row * lat_step
            cell_lat_max = lat_min + (row + 1) * lat_step

            cell_pts = []
            attempts = 0
            max_cell_attempts = samples_per_cell * 200

            while len(cell_pts) < samples_per_cell and attempts < max_cell_attempts:
                lat = rng.uniform(cell_lat_min, cell_lat_max)
                lon = rng.uniform(cell_lon_min, cell_lon_max)
                pt = Point(lon, lat)
                if zone_geometry.contains(pt):
                    cell_pts.append((lat, lon))
                attempts += 1

            all_points.extend(cell_pts)
            if cell_pts:
                points_per_cell[(row, col)] = len(cell_pts)

    # If we didn't get enough from the grid, top up with pure random sampling
    print(f"  Got {len(all_points)} from grid. Topping up to {n_samples}...")
    attempts = 0
    while len(all_points) < n_samples and attempts < max_attempts:
        lat = rng.uniform(lat_min, lat_max)
        lon = rng.uniform(lon_min, lon_max)
        pt = Point(lon, lat)
        if zone_geometry.contains(pt):
            all_points.append((lat, lon))
        attempts += 1

    # Shuffle and truncate to exactly n_samples
    all_points = all_points[:n_samples]
    rng.shuffle(all_points)

    lats = [p[0] for p in all_points]
    lons = [p[1] for p in all_points]

    pts_gdf = gpd.GeoDataFrame(
        {'lat': lats, 'lon': lons},
        geometry=[Point(lon, lat) for lat, lon in all_points],
        crs='EPSG:4326'
    )

    print(f"  ✓ Generated {len(pts_gdf)} sample points")
    print(f"  Lat range: {min(lats):.3f}° to {max(lats):.3f}°N")
    print(f"  Lon range: {min(lons):.3f}° to {max(lons):.3f}°E")

    return pts_gdf


# =============================================================================
# Step 3: GEE embedding extraction at sample points
# =============================================================================

def extract_embeddings_gee(pts_gdf, year, gee_project, batch_size=200):
    """
    Extract 64-band Prithvi satellite embedding at each sample point using GEE.

    Uses image.sample() per geographic batch with filterBounds for correct image
    selection. Returns a DataFrame with 64 embedding columns plus lat/lon.

    Key: must use filterBounds + mosaic() so GEE returns the right regional image.
    """
    import ee

    print(f"\n  Initializing GEE project: {gee_project}")
    ee.Initialize(project=gee_project)

    band_names = [f'A{i:02d}' for i in range(64)]
    n_points = len(pts_gdf)
    print(f"  Extracting embeddings for {n_points} points in batches of {batch_size}...")
    print(f"  Embedding: GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL, year={year}")
    print(f"  Band format: A00–A63")

    all_features = []
    all_lats = []
    all_lons = []
    failed = 0

    lats = pts_gdf['lat'].values
    lons = pts_gdf['lon'].values

    batches = list(range(0, n_points, batch_size))
    for batch_idx, start in enumerate(tqdm(batches, desc='  GEE batches')):
        end = min(start + batch_size, n_points)
        batch_lats = lats[start:end]
        batch_lons = lons[start:end]

        # Build bounding box for this batch (with 0.05° padding)
        pad = 0.05
        bbox = ee.Geometry.Rectangle([
            float(batch_lons.min()) - pad,
            float(batch_lats.min()) - pad,
            float(batch_lons.max()) + pad,
            float(batch_lats.max()) + pad,
        ])

        try:
            # Filter image to this region — critical for correct image selection
            emb_image = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
                         .filterDate(f'{year}-01-01', f'{year+1}-01-01')
                         .filterBounds(bbox)
                         .mosaic()
                         .select(band_names)
                         .toFloat())

            # Build FeatureCollection of sample points for this batch
            ee_points = ee.FeatureCollection([
                ee.Feature(
                    ee.Geometry.Point([float(batch_lons[i]), float(batch_lats[i])]),
                    {'lat': float(batch_lats[i]), 'lon': float(batch_lons[i])}
                )
                for i in range(len(batch_lats))
            ])

            # sample() over the bounding box guarantees we get the points back
            # with pixel values. Use the points' bounding box as the sample region.
            sampled = emb_image.sampleRegions(
                collection=ee_points,
                scale=10,
                geometries=False,
                tileScale=4,
            )

            # Retrieve in pages of 100 to avoid memory limits
            page_size = 100
            offset = 0
            while True:
                result = sampled.toList(page_size, offset).getInfo()
                if not result:
                    break
                for feat_info in result:
                    props = feat_info.get('properties', {})
                    lat = props.get('lat')
                    lon = props.get('lon')
                    band_vals = [props.get(f'A{b:02d}') for b in range(64)]
                    if all(v is not None for v in band_vals) and lat is not None:
                        all_features.append(band_vals)
                        all_lats.append(lat)
                        all_lons.append(lon)
                    else:
                        failed += 1
                if len(result) < page_size:
                    break
                offset += page_size

        except Exception as e:
            print(f"\n  Batch {batch_idx+1} failed: {e}")
            failed += len(batch_lats)

        # Brief pause to respect GEE rate limits
        time.sleep(0.5)

    print(f"\n  ✓ Extracted {len(all_features)} valid embeddings")
    print(f"  Failed/outside coverage: {failed}")

    if len(all_features) == 0:
        raise RuntimeError("No embeddings extracted. Check GEE auth and that the "
                           "sample points fall within GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL coverage.")

    X = np.array(all_features, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    result_df = pd.DataFrame(X, columns=[f'emb_{i}' for i in range(64)])
    result_df['lat'] = all_lats
    result_df['lon'] = all_lons

    return result_df


def get_embedding_band_names(year, gee_project):
    """
    Query GEE for the actual band names of the embedding image,
    since the naming may differ from what we expect.
    """
    ee.Initialize(project=gee_project)
    emb = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL").filter(
        ee.Filter.calendarRange(year, year, 'year')
    ).first()
    info = emb.getInfo()
    bands = [b['id'] for b in info.get('bands', [])]
    return bands


def extract_embeddings_gee_tiled(zone_bounds, n_samples, year, gee_project,
                                  grid_cols=10, grid_rows=15, seed=42):
    """
    Extract embeddings using GEE's built-in random sampling per geographic tile.

    Instead of sending pre-computed point coordinates, this divides the zone
    bounding box into a cols×rows grid and asks GEE to randomly sample
    n_per_tile pixels per tile. This uses far fewer API calls than the
    per-point approach (grid_cols*grid_rows calls instead of N/batch_size).

    Args:
        zone_bounds: (lon_min, lat_min, lon_max, lat_max)
        n_samples:   total target sample count
        year:        embedding year
        gee_project: GEE project id
        grid_cols:   number of longitude grid cells
        grid_rows:   number of latitude grid cells
        seed:        random seed for GEE sampling

    Returns:  DataFrame with emb_0..emb_63, lat, lon columns
    """
    import ee
    print(f"\n  Initializing GEE project: {gee_project}")
    ee.Initialize(project=gee_project)

    band_names = [f'A{i:02d}' for i in range(64)]
    lon_min, lat_min, lon_max, lat_max = zone_bounds
    n_cells = grid_cols * grid_rows
    n_per_tile = max(1, math.ceil(n_samples / n_cells))

    print(f"  Grid: {grid_cols}×{grid_rows} = {n_cells} tiles")
    print(f"  Target {n_per_tile} samples per tile = {n_per_tile*n_cells:,} total")
    print(f"  ({n_cells} GEE API calls total)")

    lon_step = (lon_max - lon_min) / grid_cols
    lat_step = (lat_max - lat_min) / grid_rows

    all_features = []
    all_lats = []
    all_lons = []
    tile_count = 0
    failed_tiles = 0

    for row in tqdm(range(grid_rows), desc='  Grid rows'):
        for col in range(grid_cols):
            tile_lon_min = lon_min + col * lon_step
            tile_lon_max = lon_min + (col + 1) * lon_step
            tile_lat_min = lat_min + row * lat_step
            tile_lat_max = lat_min + (row + 1) * lat_step

            bbox = ee.Geometry.Rectangle(
                [tile_lon_min, tile_lat_min, tile_lon_max, tile_lat_max]
            )

            try:
                emb_image = (
                    ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
                    .filterDate(f'{year}-01-01', f'{year+1}-01-01')
                    .filterBounds(bbox)
                    .mosaic()
                    .select(band_names)
                    .toFloat()
                )

                # Add lat/lon bands so we know where each pixel is
                emb_with_coords = emb_image.addBands(
                    ee.Image.pixelLonLat().rename(['lon_px', 'lat_px'])
                )

                sampled = emb_with_coords.sample(
                    region=bbox,
                    scale=300,       # 300m grid (~3x Sentinel-2) for fast sampling
                    numPixels=n_per_tile,
                    seed=seed + row * grid_cols + col,
                    geometries=False,
                    tileScale=4,
                )

                # Retrieve in pages
                page_size = 500
                offset = 0
                tile_pts = 0
                while True:
                    result = sampled.toList(page_size, offset).getInfo()
                    if not result:
                        break
                    for feat_info in result:
                        props = feat_info.get('properties', {})
                        band_vals = [props.get(f'A{b:02d}') for b in range(64)]
                        lat = props.get('lat_px')
                        lon = props.get('lon_px')
                        if (all(v is not None for v in band_vals)
                                and lat is not None and lon is not None):
                            all_features.append(band_vals)
                            all_lats.append(lat)
                            all_lons.append(lon)
                            tile_pts += 1
                    if len(result) < page_size:
                        break
                    offset += page_size

                tile_count += 1

            except Exception as e:
                failed_tiles += 1
                tqdm.write(f"  Tile ({row},{col}) failed: {e}")

            time.sleep(0.2)

    print(f"\n  ✓ Extracted {len(all_features):,} valid points from {tile_count} tiles")
    if failed_tiles:
        print(f"  {failed_tiles} tiles failed (ocean / no coverage)")

    if len(all_features) == 0:
        raise RuntimeError("No embeddings extracted. Check GEE auth and zone bounds.")

    X = np.array(all_features, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    result_df = pd.DataFrame(X, columns=[f'emb_{i}' for i in range(64)])
    result_df['lat'] = all_lats
    result_df['lon'] = all_lons
    return result_df


# =============================================================================
# Step 4: MLP classification
# =============================================================================

def load_mlp_model(model_dir, device):
    """Load the trained MLP and scaler from a south_vi_large run."""
    model_dir = Path(model_dir)
    model_path = model_dir / 'mlp_model.pth'
    scaler_path = model_dir / 'mlp_scaler.pkl'

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    model = YewMLP(input_dim=64)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    print(f"  Loaded model from {model_path}")
    print(f"  Loaded scaler from {scaler_path}")
    return model, scaler


@torch.no_grad()
def classify_samples(X, model, scaler, device, batch_size=4096):
    """Apply MLP to extracted embedding features. Returns probability array."""
    X_scaled = scaler.transform(X).astype(np.float32)
    probs = []
    for i in range(0, len(X_scaled), batch_size):
        batch = torch.from_numpy(X_scaled[i:i+batch_size]).to(device)
        logits = model(batch)
        p = torch.sigmoid(logits).cpu().numpy()
        probs.extend(p.tolist())
    return np.array(probs, dtype=np.float32)


# =============================================================================
# Step 5: Statistics & output
# =============================================================================

def wilson_ci(k, n, z=1.96):
    """Wilson score interval for proportion k/n at confidence level z."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p_hat = k / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    spread = (z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    return p_hat, max(0.0, centre - spread), min(1.0, centre + spread)


def compute_statistics(probs, lats, lons, zone_area_ha, thresholds=(0.3, 0.5, 0.7)):
    """
    Compute population estimates and confidence intervals.

    Under simple random sampling, the sample proportion is an unbiased
    estimator of the population proportion. We use Wilson CIs.
    """
    n = len(probs)
    stats = {
        'n_samples': n,
        'zone_area_ha': zone_area_ha,
        'mean_probability': float(np.mean(probs)),
        'median_probability': float(np.median(probs)),
        'std_probability': float(np.std(probs)),
        'thresholds': {}
    }

    print(f"\n{'='*60}")
    print(f"YEW POPULATION STATISTICS — CWH Zone")
    print(f"{'='*60}")
    print(f"  Sample size (n):   {n:,}")
    print(f"  Zone area:         {zone_area_ha:,.0f} ha ({zone_area_ha/1e6:.2f} million ha)")
    print(f"  Mean probability:  {stats['mean_probability']:.4f}")
    print(f"  Median probability:{stats['median_probability']:.4f}")

    print(f"\n  Threshold Analysis (95% confidence intervals):")
    print(f"  {'Threshold':>10} {'Proportion':>12} {'95% CI':>20} {'Area (ha)':>14} {'CI Range (ha)':>16}")
    print(f"  {'-'*10} {'-'*12} {'-'*20} {'-'*14} {'-'*16}")

    for thresh in thresholds:
        k = int((probs >= thresh).sum())
        p_hat, ci_lo, ci_hi = wilson_ci(k, n)
        area_est = p_hat * zone_area_ha
        area_lo = ci_lo * zone_area_ha
        area_hi = ci_hi * zone_area_ha
        stats['thresholds'][thresh] = {
            'k': k, 'proportion': p_hat,
            'ci_low': ci_lo, 'ci_high': ci_hi,
            'area_ha': area_est,
            'area_ha_ci_low': area_lo,
            'area_ha_ci_high': area_hi,
        }
        print(f"  P≥{thresh:.1f}:     {p_hat:>10.3%}   [{ci_lo:.3%} – {ci_hi:.3%}]   "
              f"{area_est:>12,.0f}   [{area_lo:,.0f} – {area_hi:,.0f}]")

    return stats


def make_figures(results_df, stats, gdf, output_dir):
    """Generate summary figures: map + histogram + subzone breakdown."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Geographic map of sample probabilities
    ax = axes[0]
    if gdf is not None:
        try:
            gdf.boundary.plot(ax=ax, color='#aaaaaa', linewidth=0.3, alpha=0.5)
        except Exception:
            pass
    n_pts = len(results_df)
    if n_pts > 50_000:
        # Dense polygon-like map: tiny pixels, low alpha
        marker_s = 0.8
        marker_alpha = 0.4
    elif n_pts > 10_000:
        marker_s = 2
        marker_alpha = 0.5
    else:
        marker_s = 6
        marker_alpha = 0.7
    sc = ax.scatter(results_df['lon'], results_df['lat'],
                    c=results_df['prob'], cmap='RdYlGn',
                    vmin=0, vmax=1, s=marker_s, alpha=marker_alpha, zorder=5,
                    rasterized=True)
    plt.colorbar(sc, ax=ax, label='P(yew)')
    ax.set_title('Sample Points — Yew Probability\n(CWH BEC Zone, BC)', fontsize=11)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # 2. Probability histogram
    ax = axes[1]
    ax.hist(results_df['prob'], bins=50, color='steelblue', edgecolor='white', alpha=0.85)
    for thresh, color in [(0.3, 'orange'), (0.5, 'red')]:
        ax.axvline(thresh, color=color, linestyle='--', linewidth=1.5,
                   label=f'P≥{thresh}: {(results_df["prob"]>=thresh).mean():.1%}')
    ax.set_xlabel('Yew Probability')
    ax.set_ylabel('Number of Sample Points')
    ax.set_title('Distribution of Yew Probabilities\n(CWH Zone Samples)', fontsize=11)
    ax.legend(fontsize=9)

    # 3. Latitudinal breakdown
    ax = axes[2]
    results_df['lat_bin'] = pd.cut(results_df['lat'],
                                    bins=np.arange(48, 57, 0.5),
                                    labels=[f'{l:.1f}°N' for l in np.arange(48, 56.5, 0.5)])
    lat_stats = results_df.groupby('lat_bin', observed=True).agg(
        mean_prob=('prob', 'mean'),
        n=('prob', 'count'),
        prop_05=('prob', lambda x: (x >= 0.5).mean())
    ).reset_index()

    bars = ax.barh(range(len(lat_stats)), lat_stats['prop_05'],
                   color=[plt.cm.RdYlGn(p) for p in lat_stats['prop_05']], alpha=0.85)
    ax.set_yticks(range(len(lat_stats)))
    ax.set_yticklabels(lat_stats['lat_bin'].astype(str), fontsize=8)
    ax.set_xlabel('Proportion of pixels with P(yew)≥0.5')
    ax.set_title('Yew Presence by Latitude Band\n(P≥0.5, CWH Zone)', fontsize=11)
    for i, (bar, row) in enumerate(zip(bars, lat_stats.itertuples())):
        if row.n >= 5:
            ax.text(bar.get_width() + 0.002, i, f'n={row.n}', va='center', fontsize=7)

    plt.tight_layout()
    dpi = 200 if len(results_df) > 50_000 else 150
    fig_path = output_dir / 'cwh_yew_population_summary.png'
    plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved figure: {fig_path.name}")
    return fig_path


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("YEW POPULATION SAMPLING — CWH BEC ZONE")
    print("=" * 60)
    print(f"  BEC zone:    {args.bec_zone}")
    print(f"  N samples:   {args.n_samples:,}")
    print(f"  Year:        {args.year}")
    print(f"  GEE project: {args.gee_project}")
    print(f"  Output:      {output_dir}")

    # -------------------------------------------------------------------------
    # 1. Load BEC zone polygon
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"STEP 1: Loading {args.bec_zone} BEC Zone Boundary")
    print(f"{'='*60}")

    gdf = None
    zone_area_ha = None

    if args.boundary_gpkg and Path(args.boundary_gpkg).exists():
        print(f"  Loading boundary from: {args.boundary_gpkg}")
        gdf = gpd.read_file(args.boundary_gpkg)
        if 'area_ha' not in gdf.columns:
            gdf_proj = gdf.to_crs('EPSG:3005')
            gdf['area_ha'] = gdf_proj.geometry.area / 10000
        zone_area_ha = float(gdf['area_ha'].sum())
        print(f"  {args.bec_zone} zone area (forestry boundary): {zone_area_ha:,.0f} ha")
    else:
        try:
            gdf = download_bec_zone(args.bec_zone, output_dir)
            zone_area_ha = float(gdf['area_ha'].sum())
        except Exception as e:
            print(f"\n  WFS download failed ({e}), using approximate polygon...")
            approx_geom = create_cwh_approximate_polygon()
            gdf = gpd.GeoDataFrame(
                [{'ZONE': args.bec_zone, 'geometry': approx_geom}],
                crs='EPSG:4326'
            )
            gdf_proj = gdf.to_crs('EPSG:3005')
            gdf['area_ha'] = gdf_proj.geometry.area / 10000
            zone_area_ha = float(gdf['area_ha'].sum())
            print(f"  Approximate {args.bec_zone} zone area: {zone_area_ha:,.0f} ha")

    # -------------------------------------------------------------------------
    # 2. Generate sample points (only needed for per-point GEE extraction)
    # -------------------------------------------------------------------------
    pts_gdf = None
    if not args.tiled_sampling:
        print(f"\n{'='*60}")
        print("STEP 2: Generating Stratified Random Sample Points")
        print(f"{'='*60}")

        pts_cache = output_dir / f'sample_points_{args.n_samples}_seed{args.seed}.csv'
        if pts_cache.exists():
            print(f"  Loading cached sample points from {pts_cache.name}...")
            pts_df = pd.read_csv(pts_cache)
            pts_gdf = gpd.GeoDataFrame(
                pts_df,
                geometry=[Point(row.lon, row.lat) for row in pts_df.itertuples()],
                crs='EPSG:4326'
            )
        else:
            pts_gdf = stratified_sample_points(gdf, args.n_samples, seed=args.seed)
            pts_gdf[['lat', 'lon']].to_csv(pts_cache, index=False)
            print(f"  Cached sample points to {pts_cache.name}")
    else:
        print(f"\nSTEP 2: Skipped (tiled GEE sampling — no pre-generated points needed)")

    # -------------------------------------------------------------------------
    # 3. Check for existing extraction results
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("STEP 3: Extracting GEE Embeddings")
    print(f"{'='*60}")

    method_tag = 'tiled' if args.tiled_sampling else 'points'
    emb_cache = output_dir / f'embeddings_{args.n_samples}_seed{args.seed}_year{args.year}_{method_tag}.csv'
    if emb_cache.exists():
        print(f"  Loading cached embeddings from {emb_cache.name}...")
        emb_df = pd.read_csv(emb_cache)
        print(f"  Loaded {len(emb_df):,} embeddings")
    elif args.tiled_sampling:
        # Fast path: GEE samples random pixels per grid tile
        bounds = gdf.total_bounds  # [lon_min, lat_min, lon_max, lat_max]
        emb_df = extract_embeddings_gee_tiled(
            zone_bounds=bounds,
            n_samples=args.n_samples,
            year=args.year,
            gee_project=args.gee_project,
            grid_cols=args.grid_cols,
            grid_rows=args.grid_rows,
            seed=args.seed,
        )
        emb_df.to_csv(emb_cache, index=False)
        print(f"  Cached embeddings to {emb_cache.name}")
    else:
        # Per-point extraction (slower, requires pre-generated points)
        emb_df = extract_embeddings_gee(pts_gdf, args.year, args.gee_project,
                                         batch_size=args.gee_batch_size)
        emb_df.to_csv(emb_cache, index=False)
        print(f"  Cached embeddings to {emb_cache.name}")

    # -------------------------------------------------------------------------
    # 4. Load model and classify
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("STEP 4: Classifying Sample Points with MLP Model")
    print(f"{'='*60}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    model, scaler = load_mlp_model(args.model_dir, device)

    # Extract embedding columns
    emb_cols = [c for c in emb_df.columns if c.startswith('emb_')]
    if len(emb_cols) != 64:
        raise ValueError(f"Expected 64 embedding columns, got {len(emb_cols)}: {emb_cols[:5]}")

    X = emb_df[emb_cols].values.astype(np.float32)
    probs = classify_samples(X, model, scaler, device)

    # Build results DataFrame
    results_df = pd.DataFrame({
        'lat': emb_df['lat'].values,
        'lon': emb_df['lon'].values,
        'prob': probs,
        'pred_yew': (probs >= 0.5).astype(int),
    })

    results_path = output_dir / 'sample_predictions.csv'
    results_df.to_csv(results_path, index=False)
    print(f"  ✓ Classification complete: {len(results_df)} points")
    print(f"  ✓ Saved predictions to {results_path.name}")

    # -------------------------------------------------------------------------
    # 5. Statistics
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("STEP 5: Computing Population Statistics")
    print(f"{'='*60}")

    stats = compute_statistics(
        probs, results_df['lat'].values, results_df['lon'].values,
        zone_area_ha, thresholds=(0.3, 0.5, 0.7)
    )

    # Margin of error
    n = stats['n_samples']
    p50 = stats['thresholds'][0.5]['proportion']
    moe = 1.96 * math.sqrt(p50 * (1 - p50) / n)
    print(f"\n  Margin of error (P≥0.5, 95% CI): ±{moe:.2%}")
    print(f"  Sampling scale: 1 sample point ≈ {zone_area_ha/n:,.0f} ha")

    # Subzone breakdown if available
    if 'SUBZONE' in gdf.columns:
        print(f"\n  Sampling by BEC subzone (top subzones):")
        try:
            results_gdf = gpd.GeoDataFrame(
                results_df,
                geometry=[Point(lon, lat) for lon, lat in zip(results_df['lon'], results_df['lat'])],
                crs='EPSG:4326'
            )
            joined = gpd.sjoin(results_gdf, gdf[['SUBZONE', 'geometry']], how='left', predicate='within')
            subzone_stats = joined.groupby('SUBZONE').agg(
                n=('prob', 'count'),
                mean_prob=('prob', 'mean'),
                prop_05=('prob', lambda x: (x >= 0.5).mean())
            ).sort_values('n', ascending=False)
            print(f"\n  {'Subzone':>10} {'n':>6} {'Mean P':>10} {'P≥0.5':>10}")
            for sz, row in subzone_stats.head(10).iterrows():
                print(f"  {str(sz):>10} {row['n']:>6} {row['mean_prob']:>10.3f} {row['prop_05']:>10.1%}")
            stats['subzone_breakdown'] = subzone_stats.to_dict()
        except Exception as e:
            print(f"  (Subzone breakdown failed: {e})")

    # -------------------------------------------------------------------------
    # 6. Save JSON summary
    # -------------------------------------------------------------------------
    stats['timestamp'] = datetime.now().isoformat()
    stats['bec_zone'] = args.bec_zone
    stats['year'] = args.year
    stats['seed'] = args.seed
    stats['model_dir'] = str(args.model_dir)

    stats_path = output_dir / 'population_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n  ✓ Statistics saved to {stats_path.name}")

    # -------------------------------------------------------------------------
    # 7. Figures
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("STEP 6: Generating Figures")
    print(f"{'='*60}")
    make_figures(results_df, stats, gdf, output_dir)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    p50_info = stats['thresholds'][0.5]
    print(f"  Zone: {args.bec_zone} BEC Zone, BC")
    print(f"  Total zone area: {zone_area_ha:,.0f} ha")
    print(f"  Samples classified: {n:,}")
    print(f"  Estimated area with P(yew)≥0.5:")
    print(f"    {p50_info['area_ha']:,.0f} ha  ({p50_info['proportion']:.1%})")
    print(f"    95% CI: [{p50_info['area_ha_ci_low']:,.0f} – {p50_info['area_ha_ci_high']:,.0f}] ha")
    print(f"\n  Output files in: {output_dir}")
    print(f"    sample_predictions.csv")
    print(f"    population_statistics.json")
    print(f"    cwh_yew_population_summary.png")


if __name__ == '__main__':
    main()
