#!/usr/bin/env python3
"""
Generate non-yew training negatives from HIGH-PROBABILITY predictions
that fall in LOGGED AREAS (Hansen Global Forest Change loss = 1).

Rationale: Yew does not regenerate after logging. High-probability predictions
in logged areas are confident false positives — good training negatives.

Uses:
  - Results from the CWH 300k tiled sampling (sample_predictions.csv)
  - Embeddings already extracted (embeddings_*_tiled.csv)
  - Hansen Global Forest Change v1.12 for loss detection (via GEE)

Usage:
    python scripts/preprocessing/generate_logged_negatives.py \
        --predictions results/analysis/cwh_yew_forestry_300k/sample_predictions.csv \
        --embeddings results/analysis/cwh_yew_forestry_300k/embeddings_300000_seed42_year2024_tiled.csv \
        --prob-threshold 0.5 \
        --max-negatives 500 \
        --output data/processed/cwh_negatives/logged_negatives.csv
"""

import argparse
import math
import time
from pathlib import Path

import ee
import numpy as np
import pandas as pd
from tqdm import tqdm


def query_hansen_loss(points_df, gee_project='carbon-storm-206002', batch_size=500):
    """
    Query Hansen Global Forest Change for each point, returning loss flag
    and loss year.
    
    Args:
        points_df: DataFrame with 'lat' and 'lon' columns
        gee_project: GEE project ID
        batch_size: Points per GEE API call
        
    Returns:
        DataFrame with original lat/lon plus 'loss', 'lossyear', 'treecover2000'
    """
    ee.Initialize(project=gee_project)
    hansen = ee.Image('UMD/hansen/global_forest_change_2024_v1_12')
    loss_img = hansen.select(['loss', 'lossyear', 'treecover2000'])
    
    n_batches = math.ceil(len(points_df) / batch_size)
    all_results = []
    
    for batch_idx in tqdm(range(n_batches), desc='  Hansen queries'):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(points_df))
        batch = points_df.iloc[start:end]
        
        features = []
        for i, row in enumerate(batch.itertuples()):
            pt = ee.Geometry.Point([row.lon, row.lat])
            features.append(ee.Feature(pt, {
                'orig_idx': int(start + i),
                'lat': float(row.lat),
                'lon': float(row.lon),
            }))
        
        fc = ee.FeatureCollection(features)
        
        try:
            sampled = loss_img.sampleRegions(
                collection=fc, scale=30, geometries=False, tileScale=4
            )
            
            # Retrieve in pages
            page_size = 500
            offset = 0
            while True:
                result = sampled.toList(page_size, offset).getInfo()
                if not result:
                    break
                for feat in result:
                    props = feat.get('properties', {})
                    all_results.append({
                        'lat': props.get('lat'),
                        'lon': props.get('lon'),
                        'loss': props.get('loss', 0),
                        'lossyear': props.get('lossyear'),
                        'treecover2000': props.get('treecover2000', 0),
                    })
                if len(result) < page_size:
                    break
                offset += page_size
                
        except Exception as e:
            tqdm.write(f'  Batch {batch_idx} failed: {e}')
        
        time.sleep(0.3)
    
    result_df = pd.DataFrame(all_results)
    return result_df


def main():
    p = argparse.ArgumentParser(
        description='Generate non-yew negatives from logged high-probability areas')
    p.add_argument('--predictions',
                   default='results/analysis/cwh_yew_forestry_300k/sample_predictions.csv')
    p.add_argument('--embeddings',
                   default='results/analysis/cwh_yew_forestry_300k/embeddings_300000_seed42_year2024_tiled.csv')
    p.add_argument('--prob-threshold', type=float, default=0.5,
                   help='Minimum yew probability to consider (default: 0.5)')
    p.add_argument('--max-negatives', type=int, default=500,
                   help='Maximum number of logged negatives to select')
    p.add_argument('--gee-project', default='carbon-storm-206002')
    p.add_argument('--output',
                   default='data/processed/cwh_negatives/logged_negatives.csv')
    args = p.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print('=' * 60)
    print('GENERATE NON-YEW NEGATIVES FROM LOGGED AREAS')
    print('=' * 60)
    
    # ------------------------------------------------------------------
    # 1. Load predictions and filter to high-probability
    # ------------------------------------------------------------------
    print(f'\nStep 1: Loading predictions')
    pred = pd.read_csv(args.predictions)
    high = pred[pred['prob'] >= args.prob_threshold].copy().reset_index(drop=True)
    print(f'  Total predictions: {len(pred):,}')
    print(f'  P >= {args.prob_threshold}: {len(high):,}')
    
    # ------------------------------------------------------------------
    # 2. Query Hansen loss for all high-probability points
    # ------------------------------------------------------------------
    print(f'\nStep 2: Querying Hansen Global Forest Change for loss detection')
    
    hansen_cache = output_path.parent / 'hansen_loss_cache.csv'
    if hansen_cache.exists():
        print(f'  Loading cached Hansen results: {hansen_cache}')
        hansen_df = pd.read_csv(hansen_cache)
    else:
        hansen_df = query_hansen_loss(high, gee_project=args.gee_project)
        hansen_df.to_csv(hansen_cache, index=False)
        print(f'  Cached to {hansen_cache}')
    
    print(f'  Hansen results: {len(hansen_df):,} points')
    n_loss = (hansen_df['loss'] == 1).sum()
    print(f'  Forest loss detected: {n_loss} ({100*n_loss/max(1,len(hansen_df)):.1f}%)')
    
    if n_loss > 0:
        loss_years = hansen_df.loc[hansen_df['loss'] == 1, 'lossyear'].dropna()
        if len(loss_years) > 0:
            print(f'  Loss years: {int(loss_years.min())+2000} to {int(loss_years.max())+2000}')
    
    # ------------------------------------------------------------------
    # 3. Select logged high-prob points as negatives
    # ------------------------------------------------------------------
    print(f'\nStep 3: Selecting logged negatives')
    logged = hansen_df[hansen_df['loss'] == 1].copy()
    
    if len(logged) == 0:
        print('  ERROR: No logged points found! Cannot generate negatives.')
        return
    
    # Merge back with predictions to get probability
    logged = logged.merge(
        high[['lat', 'lon', 'prob']],
        on=['lat', 'lon'], how='left'
    )
    
    # Take highest-probability logged points (worst false positives)
    logged = logged.sort_values('prob', ascending=False)
    selected = logged.head(args.max_negatives)
    print(f'  Selected {len(selected)} logged negatives')
    print(f'  Prob range: {selected["prob"].min():.4f} — {selected["prob"].max():.4f}')
    
    # ------------------------------------------------------------------
    # 4. Match to cached embeddings
    # ------------------------------------------------------------------
    print(f'\nStep 4: Matching to cached embeddings')
    emb_df = pd.read_csv(args.embeddings)
    print(f'  Embeddings loaded: {len(emb_df):,} rows')
    
    # Round lat/lon for matching (floating point precision)
    selected['lat_r'] = selected['lat'].round(8)
    selected['lon_r'] = selected['lon'].round(8)
    emb_df['lat_r'] = emb_df['lat'].round(8)
    emb_df['lon_r'] = emb_df['lon'].round(8)
    
    merged = selected.merge(emb_df, on=['lat_r', 'lon_r'], how='inner',
                            suffixes=('', '_emb'))
    
    emb_cols = [c for c in emb_df.columns if c.startswith('emb_')]
    
    if len(merged) < len(selected):
        print(f'  ⚠ Only {len(merged)}/{len(selected)} matched to embeddings')
        # Try nearest-neighbor matching for unmatched
        if len(merged) < len(selected) * 0.5:
            print('  Trying fuzzy lat/lon matching...')
            from scipy.spatial import cKDTree
            tree = cKDTree(emb_df[['lat', 'lon']].values)
            unmatched = selected[~selected.index.isin(merged.index)]
            dists, idxs = tree.query(unmatched[['lat', 'lon']].values)
            close = dists < 0.001  # ~100m
            if close.sum() > 0:
                extra_emb = emb_df.iloc[idxs[close]].copy()
                extra_emb['lat'] = unmatched.loc[unmatched.index[close], 'lat'].values
                extra_emb['lon'] = unmatched.loc[unmatched.index[close], 'lon'].values
                extra_emb['prob'] = unmatched.loc[unmatched.index[close], 'prob'].values
                extra_emb['loss'] = 1
                extra_emb['lossyear'] = unmatched.loc[unmatched.index[close], 'lossyear'].values
                merged = pd.concat([merged, extra_emb], ignore_index=True)
                print(f'  Fuzzy matched {close.sum()} more → total {len(merged)}')
    
    # Build output: emb_0..emb_63 + lat + lon
    out_cols = emb_cols + ['lat', 'lon']
    # Use the embedding lat/lon if available
    if 'lat_emb' in merged.columns:
        merged['lat'] = merged['lat_emb'].fillna(merged['lat'])
        merged['lon'] = merged['lon_emb'].fillna(merged['lon'])
    
    output_df = merged[out_cols].copy()
    output_df.to_csv(output_path, index=False)
    
    print(f'\n  ✓ Saved {len(output_df)} logged-area negative embeddings to {output_path}')
    print(f'\n  To retrain:')
    print(f'    python scripts/prediction/classify_tiled_gpu.py \\')
    print(f'        --input-dir results/predictions/south_vi_large \\')
    print(f'        --annotations data/raw/yew_annotations_combined.csv \\')
    print(f'        --gee-negatives {output_path}')


if __name__ == '__main__':
    main()
