#!/usr/bin/env python3
"""
Sample and classify the coastal BC study region defined by:
- NE corner: 54.7786°N, 127.8119°W
- SE corner: 49.0325°N, 119.5254°W
- Western boundary: BC coastline (all islands incl. Haida Gwaii)
- Southern boundary: US-Canada border (extends below 49th parallel to capture Vancouver Island)
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
import torch
import torch.nn as nn
from datetime import datetime
from scipy.stats import norm
import json
import pickle
import time
import ee
from tqdm import tqdm


class YewMLP(nn.Module):
    """MLP classifier matching the trained model architecture."""
    def __init__(self, input_dim=64, hidden_dims=[128, 64, 32]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_model(model_path, device='cpu'):
    """Load trained YewMLP model."""
    model = YewMLP(input_dim=64, hidden_dims=[128, 64, 32])
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def sample_points_in_region(gdf, n_points, seed=42):
    """
    Sample random points uniformly within a multi-polygon region.
    
    For a MultiPolygon, allocate points proportionally by area.
    """
    np.random.seed(seed)
    
    # Project to equal-area CRS for accurate area calculation
    gdf_proj = gdf.to_crs(epsg=3005)  # BC Albers
    
    if len(gdf) > 1 or gdf.geometry.iloc[0].geom_type == 'MultiPolygon':
        # Explode MultiPolygon into individual parts
        gdf_parts = gdf_proj.explode(index_parts=True).reset_index(drop=True)
        gdf_parts['area_m2'] = gdf_parts.geometry.area
        total_area = gdf_parts['area_m2'].sum()
        gdf_parts['n_samples'] = (gdf_parts['area_m2'] / total_area * n_points).round().astype(int)
        
        # Adjust to ensure exact total
        diff = n_points - gdf_parts['n_samples'].sum()
        if diff != 0:
            # Add/subtract from largest polygon
            idx_max = gdf_parts['area_m2'].idxmax()
            gdf_parts.loc[idx_max, 'n_samples'] += diff
        
        logging.info(f"Sampling {len(gdf_parts)} polygon parts:")
        for i, row in gdf_parts.iterrows():
            logging.info(f"  Part {i+1}: {row['n_samples']:,} points, area {row['area_m2']/1e6:.1f} km²")
    else:
        gdf_parts = gdf_proj.copy()
        gdf_parts['n_samples'] = n_points
    
    # Sample points within each part
    samples = []
    for _, row in gdf_parts.iterrows():
        n = int(row['n_samples'])
        if n == 0:
            continue
        
        geom = row.geometry
        bounds = geom.bounds  # minx, miny, maxx, maxy
        
        # Rejection sampling
        pts = []
        while len(pts) < n:
            # Generate random points in bounding box
            n_needed = int((n - len(pts)) * 1.5)  # oversample
            x = np.random.uniform(bounds[0], bounds[2], n_needed)
            y = np.random.uniform(bounds[1], bounds[3], n_needed)
            
            # Keep points inside polygon
            for xi, yi in zip(x, y):
                from shapely.geometry import Point
                if geom.contains(Point(xi, yi)):
                    pts.append((xi, yi))
                    if len(pts) >= n:
                        break
        
        samples.extend(pts[:n])
    
    # Convert to GeoDataFrame and reproject back to WGS84
    from shapely.geometry import Point
    gdf_samples = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in samples],
        crs='EPSG:3005'
    )
    gdf_samples = gdf_samples.to_crs(epsg=4326)
    
    df = pd.DataFrame({
        'lon': [p.x for p in gdf_samples.geometry],
        'lat': [p.y for p in gdf_samples.geometry]
    })
    
    return df


def extract_embeddings_from_boundary(boundary_gdf, n_samples, year, gee_project,
                                     scale=300, seed=42):
    """
    Sample n_samples random pixels from Google satellite-embedding archive
    within the study region MultiPolygon boundary. Each polygon part receives
    a quota proportional to its area.

    Returns a DataFrame with columns emb_0..emb_63, lat, lon.
    """
    logging.info(f'Initializing GEE project: {gee_project}')
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

    logging.info(f'Study region: {len(parts_gdf)} polygon parts, {total_ha:,.0f} ha total')
    logging.info(f'Sampling {n_samples:,} points at {scale} m  ({len(parts_gdf)} GEE calls)')

    all_features, all_lats, all_lons = [], [], []

    for idx, (_, row) in enumerate(tqdm(parts_gdf.iterrows(), total=len(parts_gdf),
                                        desc='GEE extraction')):
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
            tqdm.write(f'Part {idx} failed: {e}')
        time.sleep(0.15)

    logging.info(f'Extracted {len(all_features):,} valid points')
    X = np.array(all_features, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    df = pd.DataFrame(X, columns=[f'emb_{i}' for i in range(64)])
    df['lat'] = all_lats
    df['lon'] = all_lons
    return df


def classify_samples(embeddings_df, model_path, device='cpu'):
    """Run classification on extracted embeddings."""
    model_dir = Path(model_path).parent
    
    # Load scaler
    scaler_path = model_dir / 'mlp_scaler.pkl'
    if scaler_path.exists():
        scaler = pickle.load(open(scaler_path, 'rb'))
    else:
        logging.warning(f"Scaler not found at {scaler_path}, using raw embeddings")
        scaler = None
    
    # Load model
    model = load_model(model_path, device=device)
    
    embed_cols = [f'emb_{i}' for i in range(64)]
    X = embeddings_df[embed_cols].values.astype(np.float32)
    
    # Apply scaler if available
    if scaler is not None:
        X = scaler.transform(X)
    
    # Predict in batches
    batch_size = 1024
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i:i+batch_size]).to(device)
            logits = model(batch).squeeze(-1)
            batch_probs = torch.sigmoid(logits).cpu().numpy()
            probs.extend(batch_probs)
    
    embeddings_df = embeddings_df.copy()
    embeddings_df['prob'] = probs
    
    return embeddings_df


def compute_population_stats(df, region_area_ha, thresholds=[0.3, 0.5, 0.7]):
    """
    Compute population estimates with confidence intervals.
    
    Args:
        df: DataFrame with 'prob' column
        region_area_ha: Total area of study region in hectares
        thresholds: Probability thresholds to report
    """
    n_samples = len(df)
    
    stats = {
        'n_samples': n_samples,
        'region_area_ha': region_area_ha,
        'thresholds': {}
    }
    
    for thresh in thresholds:
        n_above = (df['prob'] >= thresh).sum()
        p_hat = n_above / n_samples
        
        # Wilson score 95% CI
        z = 1.96
        denom = 1 + z**2 / n_samples
        center = (p_hat + z**2 / (2 * n_samples)) / denom
        margin = z * np.sqrt((p_hat * (1 - p_hat) / n_samples + z**2 / (4 * n_samples**2))) / denom
        ci_low = max(0, center - margin)
        ci_high = min(1, center + margin)
        
        area_est = p_hat * region_area_ha
        area_ci_low = ci_low * region_area_ha
        area_ci_high = ci_high * region_area_ha
        
        stats['thresholds'][f'p_{thresh}'] = {
            'count': int(n_above),
            'proportion': float(p_hat),
            'prop_ci_low': float(ci_low),
            'prop_ci_high': float(ci_high),
            'area_ha': float(area_est),
            'area_ci_low': float(area_ci_low),
            'area_ci_high': float(area_ci_high)
        }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-samples', type=int, default=100000,
                        help='Number of points to sample (default: 100000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--year', type=int, default=2023,
                        help='Year for GEE satellite embeddings (default: 2023)')
    parser.add_argument('--gee-project', type=str, default='ee-jericho-yew',
                        help='Google Earth Engine project ID')
    parser.add_argument('--scale', type=int, default=300,
                        help='GEE sampling scale in meters (default: 300)')
    parser.add_argument('--model-path', type=str,
                        default='results/predictions/south_vi_large/mlp_model.pth',
                        help='Path to trained YewMLP model')
    parser.add_argument('--region-boundary', type=str,
                        default='data/processed/coastal_study_region.geojson',
                        help='Path to study region boundary GeoJSON')
    parser.add_argument('--output-dir', type=str,
                        default='results/analysis/coastal_region_100k',
                        help='Output directory')
    parser.add_argument('--skip-extract', action='store_true',
                        help='Skip GEE extraction, use cached embeddings CSV')
    parser.add_argument('--cache-file', type=str,
                        default=None,
                        help='Embeddings cache file (auto-generated if not specified)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Default cache file
    if args.cache_file is None:
        args.cache_file = f'{args.output_dir}/embeddings_coastal_{args.n_samples}_seed{args.seed}.csv'
    
    # ── 1. Load study region boundary ────────────────────────────────────────
    logging.info(f"Loading study region: {args.region_boundary}")
    region_gdf = gpd.read_file(args.region_boundary)
    
    # Compute area
    region_proj = region_gdf.to_crs(epsg=3005)
    area_m2 = region_proj.geometry.area.sum()
    area_ha = area_m2 / 1e4
    area_km2 = area_m2 / 1e6
    logging.info(f"Study region area: {area_km2:,.0f} km² ({area_ha:,.0f} ha)")
    
    bounds = region_gdf.total_bounds
    logging.info(f"Bounds: lon [{bounds[0]:.3f}, {bounds[2]:.3f}], "
                 f"lat [{bounds[1]:.3f}, {bounds[3]:.3f}]")
    
    # ── 2. Extract embeddings or load cached ────────────────────────────────
    if args.skip_extract and os.path.exists(args.cache_file):
        logging.info(f"Loading cached embeddings: {args.cache_file}")
        embeddings_df = pd.read_csv(args.cache_file)
        logging.info(f"Loaded {len(embeddings_df):,} cached points")
    else:
        logging.info("Extracting Sentinel-2 embeddings from Google Earth Engine...")
        embeddings_df = extract_embeddings_from_boundary(
            region_gdf,
            args.n_samples,
            args.year,
            args.gee_project,
            args.scale,
            args.seed
        )
        
        # Cache
        embeddings_df.to_csv(args.cache_file, index=False)
        logging.info(f"Cached embeddings: {args.cache_file}")
    
    # ── 3. Load model and classify ───────────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Loading model: {args.model_path} (device: {device})")
    
    logging.info("Classifying samples...")
    predictions_df = classify_samples(embeddings_df, args.model_path, device=device)
    
    logging.info(f"Probability range: {predictions_df['prob'].min():.4f} - "
                 f"{predictions_df['prob'].max():.4f}")
    logging.info(f"Mean probability: {predictions_df['prob'].mean():.4f}")
    logging.info(f"Median probability: {predictions_df['prob'].median():.4f}")
    
    # ── 4. Save predictions ──────────────────────────────────────────────────
    pred_csv = f'{args.output_dir}/sample_predictions_coastal.csv'
    predictions_df[['lat', 'lon', 'prob']].to_csv(pred_csv, index=False)
    logging.info(f"Saved predictions: {pred_csv}")
    
    # ── 5. Compute population statistics ─────────────────────────────────────
    logging.info("Computing population statistics...")
    stats = compute_population_stats(predictions_df, area_ha, thresholds=[0.3, 0.5, 0.7])
    
    stats_json = f'{args.output_dir}/population_statistics.json'
    with open(stats_json, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logging.info("\n" + "="*70)
    logging.info("POPULATION ESTIMATES")
    logging.info("="*70)
    for thresh_key, thresh_stats in stats['thresholds'].items():
        thresh = float(thresh_key.split('_')[1])
        logging.info(f"\nProbability ≥ {thresh:.1f}:")
        logging.info(f"  Area: {thresh_stats['area_ha']:,.0f} ha "
                     f"(95% CI: {thresh_stats['area_ci_low']:,.0f} - "
                     f"{thresh_stats['area_ci_high']:,.0f} ha)")
        logging.info(f"  Proportion: {thresh_stats['proportion']:.4f} "
                     f"(95% CI: {thresh_stats['prop_ci_low']:.4f} - "
                     f"{thresh_stats['prop_ci_high']:.4f})")
        logging.info(f"  Sample count: {thresh_stats['count']:,} / {stats['n_samples']:,}")
    
    logging.info("\n" + "="*70)
    logging.info(f"✓ Complete — results saved to: {args.output_dir}")
    logging.info("="*70)


if __name__ == '__main__':
    main()
