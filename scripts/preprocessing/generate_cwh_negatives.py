#!/usr/bin/env python3
"""
Generate non-yew training samples from high-probability CWH predictions,
and build a real CWH boundary from forestry site locations.

This script:
  1. Loads FAIB forestry header to get CWH site coordinates (real BEC data)
  2. Creates a proper CWH zone polygon from the forestry sites
  3. Selects high-probability false-positive sample points from the 300k run
     that are far from known yew observations — these become non-yew negatives
  4. Extracts 64-band GEE embeddings for the selected negative sites
  5. Saves everything for use in retraining

Usage:
    python scripts/preprocessing/generate_cwh_negatives.py \
        --n-negatives 500 \
        --prob-threshold 0.7 \
        --output-dir data/processed/cwh_negatives
"""

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.ops import unary_union
from tqdm import tqdm


def build_cwh_boundary_from_forestry(faib_header_csv, buffer_km=5.0):
    """
    Build a CWH zone polygon from FAIB forestry inventory site coordinates.
    
    Uses a concave hull (alpha shape) of all CWH-classified sites to create
    a realistic boundary — much more accurate than the rectangular approximation.
    
    Args:
        faib_header_csv: Path to faib_header.csv
        buffer_km: Buffer around sites in km (default: 5km)
        
    Returns:
        GeoDataFrame with CWH boundary polygon(s)
    """
    print("  Loading FAIB header data...")
    fh = pd.read_csv(faib_header_csv)
    cwh = fh[fh['BEC_ZONE'] == 'CWH'].copy()
    cwh = cwh.dropna(subset=['Latitude', 'Longitude'])
    print(f"  CWH sites with coordinates: {len(cwh)}")
    print(f"  Lat range: {cwh.Latitude.min():.4f} to {cwh.Latitude.max():.4f}")
    print(f"  Lon range: {cwh.Longitude.min():.4f} to {cwh.Longitude.max():.4f}")
    
    # Create GeoDataFrame of site points
    points = [Point(row.Longitude, row.Latitude) for _, row in cwh.iterrows()]
    gdf_pts = gpd.GeoDataFrame(cwh, geometry=points, crs='EPSG:4326')
    
    # Build concave hull by gridded convex hulls
    # (alpha shapes aren't in base shapely, so we approximate by
    #  computing convex hulls of geographic grid cells and merging)
    print(f"  Building CWH zone boundary (gridded convex hull with {buffer_km}km buffer)...")
    
    # Grid into 0.5° cells
    cell_size = 0.5
    lon_min, lat_min = cwh.Longitude.min(), cwh.Latitude.min()
    lon_max, lat_max = cwh.Longitude.max(), cwh.Latitude.max()
    
    cell_hulls = []
    for lat_start in np.arange(lat_min - cell_size, lat_max + cell_size, cell_size):
        for lon_start in np.arange(lon_min - cell_size, lon_max + cell_size, cell_size):
            mask = (
                (cwh.Latitude >= lat_start) & (cwh.Latitude < lat_start + cell_size) &
                (cwh.Longitude >= lon_start) & (cwh.Longitude < lon_start + cell_size)
            )
            cell_pts = cwh[mask]
            if len(cell_pts) >= 3:
                mp = MultiPoint([(r.Longitude, r.Latitude) for _, r in cell_pts.iterrows()])
                cell_hulls.append(mp.convex_hull)
            elif len(cell_pts) >= 1:
                # Buffer individual points
                for _, r in cell_pts.iterrows():
                    cell_hulls.append(Point(r.Longitude, r.Latitude).buffer(0.05))
    
    if cell_hulls:
        # Merge and buffer
        merged = unary_union(cell_hulls)
        # Buffer by ~5km (≈0.045° at 50°N)
        buffer_deg = buffer_km / 111.0
        cwh_polygon = merged.buffer(buffer_deg)
        
        gdf = gpd.GeoDataFrame({'zone': ['CWH']}, geometry=[cwh_polygon], crs='EPSG:4326')
        area_ha = gdf.to_crs('EPSG:3005').geometry.area.sum() / 10000
        print(f"  CWH boundary area: {area_ha:,.0f} ha")
        return gdf, gdf_pts
    else:
        raise RuntimeError("No CWH sites found to build boundary")


def select_high_prob_negatives(predictions_csv, known_yew_csv, n_negatives=500,
                                prob_threshold=0.7, min_distance_km=20.0):
    """
    Select high-probability prediction points that are likely false positives,
    as non-yew training examples.
    
    Strategy: pick points with P >= threshold that are far from known yew
    observations. These are the model's worst false positives.
    
    Args:
        predictions_csv: Path to sample_predictions.csv from the 300k run
        known_yew_csv: Path to iNaturalist observations or combined annotations
        n_negatives: Number of negatives to select
        prob_threshold: Minimum probability to consider
        min_distance_km: Minimum distance from known yew (km)
        
    Returns:
        DataFrame with lat, lon, prob columns
    """
    print(f"  Loading predictions from {predictions_csv}...")
    pred = pd.read_csv(predictions_csv)
    high = pred[pred['prob'] >= prob_threshold].copy()
    print(f"  Predictions with P >= {prob_threshold}: {len(high):,}")
    
    # Load known yew locations
    print(f"  Loading known yew locations...")
    # Try iNaturalist observations first
    inat_path = Path('data/inat_observations/observations-558049.csv')
    if inat_path.exists():
        inat = pd.read_csv(inat_path, usecols=['latitude', 'longitude'])
        inat = inat.dropna()
        print(f"  iNaturalist yew observations: {len(inat)}")
    else:
        inat = pd.DataFrame(columns=['latitude', 'longitude'])
    
    # Also load annotation yews
    ann_path = Path(known_yew_csv)
    if ann_path.exists():
        ann = pd.read_csv(ann_path)
        ann_yew = ann[ann['has_yew'] == 1][['lat', 'lon']].rename(
            columns={'lat': 'latitude', 'lon': 'longitude'})
        print(f"  Annotated yew locations: {len(ann_yew)}")
    else:
        ann_yew = pd.DataFrame(columns=['latitude', 'longitude'])
    
    yew_locs = pd.concat([inat, ann_yew], ignore_index=True)
    print(f"  Total known yew locations: {len(yew_locs)}")
    
    if len(yew_locs) > 0:
        # Compute minimum distance from each high-prob point to nearest yew
        # Using simple Euclidean in degrees (good enough for filtering)
        # 1° lat ≈ 111 km, 1° lon ≈ 111 * cos(lat) km
        yew_coords = yew_locs[['latitude', 'longitude']].values
        
        min_dists = []
        mean_lat = high['lat'].mean()
        km_per_deg_lon = 111.0 * np.cos(np.radians(mean_lat))
        
        print(f"  Computing distances to known yew locations...")
        for _, row in tqdm(high.iterrows(), total=len(high), desc='  Distance calc'):
            dlat = (yew_coords[:, 0] - row['lat']) * 111.0
            dlon = (yew_coords[:, 1] - row['lon']) * km_per_deg_lon
            dist = np.sqrt(dlat**2 + dlon**2).min()
            min_dists.append(dist)
        
        high = high.copy()
        high['min_yew_dist_km'] = min_dists
        
        # Filter by minimum distance
        far_from_yew = high[high['min_yew_dist_km'] >= min_distance_km]
        print(f"  Points >= {min_distance_km}km from known yew: {len(far_from_yew):,}")
    else:
        far_from_yew = high
        far_from_yew['min_yew_dist_km'] = 999.0
    
    # Sample geographically spread negatives
    if len(far_from_yew) <= n_negatives:
        selected = far_from_yew
    else:
        # Stratified by latitude to get geographic spread
        far_from_yew = far_from_yew.copy()
        far_from_yew['lat_bin'] = pd.qcut(far_from_yew['lat'], min(20, len(far_from_yew) // 5),
                                           duplicates='drop')
        per_bin = max(1, n_negatives // far_from_yew['lat_bin'].nunique())
        selected = far_from_yew.groupby('lat_bin', observed=True).apply(
            lambda g: g.nlargest(per_bin, 'prob')
        ).reset_index(drop=True)
        # Top up if needed
        if len(selected) < n_negatives:
            remaining = far_from_yew[~far_from_yew.index.isin(selected.index)]
            extra = remaining.nlargest(n_negatives - len(selected), 'prob')
            selected = pd.concat([selected, extra], ignore_index=True)
        selected = selected.head(n_negatives)
    
    print(f"  Selected {len(selected)} high-prob negatives")
    print(f"  Prob range: {selected['prob'].min():.4f} — {selected['prob'].max():.4f}")
    print(f"  Distance range: {selected['min_yew_dist_km'].min():.1f} — {selected['min_yew_dist_km'].max():.1f} km")
    
    return selected[['lat', 'lon', 'prob', 'min_yew_dist_km']]


def extract_embeddings_for_points(points_df, year=2024, gee_project='carbon-storm-206002',
                                   batch_size=200):
    """
    Extract 64-band GEE embeddings at specific lat/lon points using sampleRegions.
    """
    print(f"\n  Initializing GEE project: {gee_project}")
    ee.Initialize(project=gee_project)
    
    band_names = [f'A{i:02d}' for i in range(64)]
    
    all_features = []
    all_lats = []
    all_lons = []
    n_batches = math.ceil(len(points_df) / batch_size)
    
    print(f"  Extracting embeddings for {len(points_df)} points in {n_batches} batches...")
    
    for batch_idx in tqdm(range(n_batches), desc='  GEE batches'):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(points_df))
        batch = points_df.iloc[start:end]
        
        # Create point features
        features = []
        for _, row in batch.iterrows():
            pt = ee.Geometry.Point([row['lon'], row['lat']])
            features.append(ee.Feature(pt, {'lat': row['lat'], 'lon': row['lon']}))
        
        fc = ee.FeatureCollection(features)
        bbox = fc.geometry().bounds()
        
        # Get embedding image
        emb_image = (
            ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
            .filterDate(f'{year}-01-01', f'{year+1}-01-01')
            .filterBounds(bbox)
            .mosaic()
            .select(band_names)
            .toFloat()
        )
        
        try:
            sampled = emb_image.sampleRegions(
                collection=fc,
                scale=10,
                geometries=False,
                tileScale=4
            )
            
            # Retrieve results
            page_size = 500
            offset = 0
            while True:
                result = sampled.toList(page_size, offset).getInfo()
                if not result:
                    break
                for feat_info in result:
                    props = feat_info.get('properties', {})
                    vals = [props.get(f'A{b:02d}') for b in range(64)]
                    lat = props.get('lat')
                    lon = props.get('lon')
                    if all(v is not None for v in vals) and lat is not None:
                        all_features.append(vals)
                        all_lats.append(lat)
                        all_lons.append(lon)
                if len(result) < page_size:
                    break
                offset += page_size
                
        except Exception as e:
            tqdm.write(f"  Batch {batch_idx} failed: {e}")
        
        time.sleep(0.3)
    
    print(f"  ✓ Extracted {len(all_features)} valid embeddings")
    
    X = np.array(all_features, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    df = pd.DataFrame(X, columns=[f'emb_{i}' for i in range(64)])
    df['lat'] = all_lats
    df['lon'] = all_lons
    return df


def parse_args():
    p = argparse.ArgumentParser(description='Generate non-yew negatives from CWH high-prob predictions')
    p.add_argument('--predictions', default='results/analysis/cwh_yew_population_300k/sample_predictions.csv')
    p.add_argument('--faib-header', default='data/raw/faib_header.csv')
    p.add_argument('--annotations', default='data/raw/yew_annotations_combined.csv')
    p.add_argument('--n-negatives', type=int, default=500,
                   help='Number of high-prob negatives to extract')
    p.add_argument('--prob-threshold', type=float, default=0.7,
                   help='Minimum probability for negative selection')
    p.add_argument('--min-distance-km', type=float, default=20.0,
                   help='Minimum distance from known yew (km)')
    p.add_argument('--output-dir', default='data/processed/cwh_negatives')
    p.add_argument('--year', type=int, default=2024)
    p.add_argument('--gee-project', default='carbon-storm-206002')
    p.add_argument('--buffer-km', type=float, default=5.0,
                   help='Buffer around CWH sites for boundary polygon (km)')
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("GENERATE CWH NON-YEW NEGATIVES + REAL CWH BOUNDARY")
    print("=" * 60)
    
    # ------------------------------------------------------------------
    # 1. Build real CWH boundary from forestry sites
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("STEP 1: Build CWH Boundary from Forestry Data")
    print(f"{'='*60}")
    
    cwh_boundary_path = output_dir / 'cwh_boundary_forestry.gpkg'
    cwh_sites_path = output_dir / 'cwh_forestry_sites.csv'
    
    if cwh_boundary_path.exists():
        print(f"  Loading cached boundary: {cwh_boundary_path}")
        cwh_gdf = gpd.read_file(cwh_boundary_path)
        cwh_sites = pd.read_csv(cwh_sites_path) if cwh_sites_path.exists() else None
    else:
        cwh_gdf, cwh_pts = build_cwh_boundary_from_forestry(
            args.faib_header, buffer_km=args.buffer_km)
        cwh_gdf.to_file(cwh_boundary_path, driver='GPKG')
        cwh_pts[['SITE_IDENTIFIER', 'Latitude', 'Longitude', 'BEC_ZONE', 'BEC_SBZ', 'BEC_VAR']].to_csv(
            cwh_sites_path, index=False)
        print(f"  ✓ Saved CWH boundary to {cwh_boundary_path}")
        print(f"  ✓ Saved CWH sites to {cwh_sites_path}")
    
    # ------------------------------------------------------------------
    # 2. Select high-prob false positive negatives
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("STEP 2: Select High-Probability False Positive Negatives")
    print(f"{'='*60}")
    
    negatives_csv = output_dir / 'high_prob_negatives.csv'
    
    if negatives_csv.exists():
        print(f"  Loading cached negatives: {negatives_csv}")
        neg_df = pd.read_csv(negatives_csv)
        print(f"  {len(neg_df)} negatives loaded")
    else:
        neg_df = select_high_prob_negatives(
            args.predictions,
            args.annotations,
            n_negatives=args.n_negatives,
            prob_threshold=args.prob_threshold,
            min_distance_km=args.min_distance_km,
        )
        neg_df.to_csv(negatives_csv, index=False)
        print(f"  ✓ Saved negatives to {negatives_csv}")
    
    # ------------------------------------------------------------------
    # 3. Extract GEE embeddings for these negatives
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("STEP 3: Extract GEE Embeddings for Negatives")
    print(f"{'='*60}")
    
    emb_cache = output_dir / 'negative_embeddings.csv'
    
    if emb_cache.exists():
        print(f"  Loading cached embeddings: {emb_cache}")
        emb_df = pd.read_csv(emb_cache)
        print(f"  {len(emb_df)} embeddings loaded")
    else:
        emb_df = extract_embeddings_for_points(
            neg_df, year=args.year, gee_project=args.gee_project)
        emb_df.to_csv(emb_cache, index=False)
        print(f"  ✓ Saved embeddings to {emb_cache}")
    
    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  CWH boundary: {cwh_boundary_path}")
    print(f"  CWH forestry sites: {cwh_sites_path}")
    print(f"  High-prob negatives: {negatives_csv} ({len(neg_df)} points)")
    print(f"  Negative embeddings: {emb_cache} ({len(emb_df)} vectors)")
    print(f"\n  To retrain with these negatives, run:")
    print(f"    python scripts/prediction/classify_tiled_gpu.py \\")
    print(f"        --input-dir results/predictions/south_vi_large \\")
    print(f"        --annotations data/raw/yew_annotations_combined.csv \\")
    print(f"        --gee-negatives {emb_cache}")


if __name__ == '__main__':
    main()
