#!/usr/bin/env python3
"""
Re-sample 100k points from the actual CWH boundary and classify with the MLP.

Unlike sample_cwh_yew_population.py (which falls back to an approximate
rectangular bounding box when the BC WFS download fails), this script loads
the pre-built CWH boundary from data/processed/cwh_negatives/cwh_boundary_forestry.gpkg
and passes the actual polygon geometry to GEE as the sampling region.

Usage:
    conda run -n yew_pytorch python scripts/prediction/resample_cwh_100k.py \
        --n-samples 100000 \
        --model-dir results/predictions/south_vi_large \
        --output-dir results/analysis/cwh_yew_population_100k
"""

import argparse
import json
import math
import pickle
import time
from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm


# ---------------------------------------------------------------------------
# MLP (must match classify_tiled_gpu.py)
# ---------------------------------------------------------------------------
class YewMLP(nn.Module):
    def __init__(self, input_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),         nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='Sample 100k CWH points and classify')
    p.add_argument('--n-samples',   type=int,  default=100_000)
    p.add_argument('--year',        type=int,  default=2024)
    p.add_argument('--model-dir',   type=str,  default='results/predictions/south_vi_large')
    p.add_argument('--output-dir',  type=str,  default='results/analysis/cwh_yew_population_100k')
    p.add_argument('--boundary',    type=str,  default='data/processed/cwh_negatives/cwh_boundary_forestry.gpkg',
                   help='CWH boundary GeoPackage (MultiPolygon, EPSG:4326)')
    p.add_argument('--gee-project', type=str,  default='carbon-storm-206002')
    p.add_argument('--scale',       type=int,  default=300,
                   help='GEE sampling scale in metres (default 300 ≈ 9 ha/pixel)')
    p.add_argument('--seed',        type=int,  default=42)
    p.add_argument('--skip-extract', action='store_true',
                   help='Skip GEE extraction, re-use cached embeddings_cwh_100k.csv')
    return p.parse_args()


# ---------------------------------------------------------------------------
# GEE extraction: sample within each polygon part
# ---------------------------------------------------------------------------
def gdf_to_ee_geometry(gdf):
    """Convert a geopandas GeoDataFrame (EPSG:4326) to a GEE Geometry."""
    geom = gdf.geometry.union_all()          # single (Multi)Polygon
    geojson = json.loads(gpd.GeoSeries([geom]).to_json())
    feat = geojson['features'][0]['geometry']
    return ee.Geometry(feat)


def extract_embeddings_from_boundary(boundary_gdf, n_samples, year, gee_project,
                                     scale=300, seed=42):
    """
    Sample n_samples random pixels from the Google satellite-embedding archive
    within the CWH MultiPolygon boundary.  Each polygon part receives a quota
    proportional to its area so the full CWH zone is represented.

    Returns a DataFrame with columns emb_0..emb_63, lat, lon.
    """
    print(f'\n  Initialising GEE project: {gee_project}')
    ee.Initialize(project=gee_project)

    band_names = [f'A{i:02d}' for i in range(64)]
    emb_image = (
        ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
        .filterDate(f'{year}-01-01', f'{year+1}-01-01')
        .mosaic()
        .select(band_names)
        .toFloat()
        .addBands(ee.Image.pixelLonLat().rename(['lon_px', 'lat_px']))
    )

    # Compute per-part area quotas
    parts_gdf = boundary_gdf.explode(index_parts=False).reset_index(drop=True)
    parts_proj = parts_gdf.to_crs('EPSG:3005')
    areas_ha   = parts_proj.geometry.area / 1e4
    total_ha   = areas_ha.sum()
    quotas     = (areas_ha / total_ha * n_samples).round().astype(int).tolist()
    # Adjust rounding error on first part
    quotas[0] += n_samples - sum(quotas)

    print(f'  CWH boundary: {len(parts_gdf)} polygon parts, {total_ha:,.0f} ha total')
    print(f'  Sampling {n_samples:,} points at {scale} m  ({len(parts_gdf)} GEE calls)')

    all_features, all_lats, all_lons = [], [], []

    for idx, (_, row) in enumerate(tqdm(parts_gdf.iterrows(), total=len(parts_gdf),
                                        desc='  Polygon parts')):
        quota = quotas[idx]
        if quota < 1:
            continue
        try:
            region = ee.Geometry(
                json.loads(gpd.GeoSeries([row.geometry]).to_json())['features'][0]['geometry']
            )
            sampled = emb_image.sample(
                region=region,
                scale=scale,
                numPixels=quota,
                seed=seed + idx,
                geometries=False,
                tileScale=4,
            )
            page_size = 500
            offset = 0
            while True:
                result = sampled.toList(page_size, offset).getInfo()
                if not result:
                    break
                for feat in result:
                    props = feat.get('properties', {})
                    vals  = [props.get(f'A{b:02d}') for b in range(64)]
                    lat, lon = props.get('lat_px'), props.get('lon_px')
                    if all(v is not None for v in vals) and lat and lon:
                        all_features.append(vals)
                        all_lats.append(lat)
                        all_lons.append(lon)
                if len(result) < page_size:
                    break
                offset += page_size
        except Exception as e:
            tqdm.write(f'  Part {idx} failed: {e}')
        time.sleep(0.15)

    print(f'\n  ✓ Extracted {len(all_features):,} valid points')
    X = np.array(all_features, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    df = pd.DataFrame(X, columns=[f'emb_{i}' for i in range(64)])
    df['lat'] = all_lats
    df['lon'] = all_lons
    return df


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def classify(df, model_dir, device):
    model_dir = Path(model_dir)
    scaler = pickle.load(open(model_dir / 'mlp_scaler.pkl', 'rb'))
    model  = YewMLP(64)
    model.load_state_dict(torch.load(model_dir / 'mlp_model.pth', map_location=device))
    model.eval().to(device)
    emb_cols = [c for c in df.columns if c.startswith('emb_')]
    X = scaler.transform(df[emb_cols].values.astype(np.float32))
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(X, device=device))).cpu().numpy().flatten()
    return probs


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def print_stats(probs, cwh_ha, n):
    print(f'\n{"="*60}')
    print('YEW POPULATION STATISTICS — CWH Zone (100k re-sample)')
    print(f'{"="*60}')
    print(f'  Sample size (n):   {n:,}')
    print(f'  Zone area:         {cwh_ha:,.0f} ha ({cwh_ha/1e6:.2f} million ha)')
    print(f'  Mean probability:  {probs.mean():.4f}')
    print(f'  Median probability:{np.median(probs):.4f}')
    print()
    print(f'  {"Threshold":>10}  {"Proportion":>12}  {"95% CI":>22}  {"Area (ha)":>14}  {"CI Range (ha)":>18}')
    print(f'  {"-"*10}  {"-"*12}  {"-"*22}  {"-"*14}  {"-"*18}')
    for thr in [0.3, 0.5, 0.7]:
        p   = (probs >= thr).mean()
        se  = math.sqrt(p * (1 - p) / n)
        lo, hi = p - 1.96*se, p + 1.96*se
        print(f'  P≥{thr:.1f}:  {p*100:10.3f}%  '
              f'[{lo*100:.3f}% – {hi*100:.3f}%]  '
              f'{p*cwh_ha:>14,.0f}  '
              f'[{lo*cwh_ha:,.0f} – {hi*cwh_ha:,.0f}]')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load CWH boundary
    print(f'\nLoading CWH boundary from {args.boundary}...')
    boundary_gdf = gpd.read_file(args.boundary)
    cwh_proj     = boundary_gdf.to_crs('EPSG:3005')
    cwh_ha       = float(cwh_proj.geometry.area.sum() / 1e4)
    print(f'  CWH area: {cwh_ha:,.0f} ha')

    # Extract or load embeddings
    emb_cache = output_dir / f'embeddings_cwh_{args.n_samples}_seed{args.seed}.csv'
    if args.skip_extract and emb_cache.exists():
        print(f'\nLoading cached embeddings from {emb_cache.name}...')
        emb_df = pd.read_csv(emb_cache)
        print(f'  Loaded {len(emb_df):,} embeddings')
    else:
        emb_df = extract_embeddings_from_boundary(
            boundary_gdf, args.n_samples, args.year,
            args.gee_project, args.scale, args.seed
        )
        emb_df.to_csv(emb_cache, index=False)
        print(f'  Cached embeddings to {emb_cache.name}')

    # Classify
    print('\nClassifying with MLP model...')
    probs = classify(emb_df, args.model_dir, device)
    emb_df['prob'] = probs
    pred_path = output_dir / 'sample_predictions_cwh.csv'
    emb_df[['lat', 'lon', 'prob']].to_csv(pred_path, index=False)
    print(f'  ✓ Saved predictions to {pred_path.name}')

    # Stats
    n = len(emb_df)
    print_stats(probs, cwh_ha, n)

    # Save stats JSON
    stats = {
        'n': n,
        'cwh_area_ha': cwh_ha,
        'mean_prob': float(probs.mean()),
        'median_prob': float(np.median(probs)),
        'p30_area_ha': float((probs >= 0.3).mean() * cwh_ha),
        'p50_area_ha': float((probs >= 0.5).mean() * cwh_ha),
        'p70_area_ha': float((probs >= 0.7).mean() * cwh_ha),
    }
    json.dump(stats, open(output_dir / 'population_statistics.json', 'w'), indent=2)
    print(f'\n  Output files in: {output_dir}')
    print(f'    sample_predictions_cwh.csv')
    print(f'    population_statistics.json')
    print(f'\nTo generate a KMZ map run:')
    print(f'  conda run -n yew_pytorch python scripts/visualization/create_cwh_kmz.py \\')
    print(f'    --input {pred_path} \\')
    print(f'    --output {output_dir}/cwh_yew_100k.kmz')


if __name__ == '__main__':
    main()
