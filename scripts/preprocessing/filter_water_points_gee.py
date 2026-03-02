#!/usr/bin/env python3
"""
Filter training data to remove points in water using Google Earth Engine.
Uses JRC Global Surface Water or MODIS Land Cover for fast batch queries.
"""
import numpy as np
import pandas as pd
import ee
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize Earth Engine
GEE_PROJECT = 'carbon-storm-206002'
ee.Initialize(project=GEE_PROJECT)
print("Earth Engine initialized")

def check_water_batch_gee(coords, batch_size=1000):
    """
    Check if coordinates are on water using GEE JRC Global Surface Water.
    Returns array of booleans (True = on land, False = in water).
    
    Uses JRC/GSW1_4/GlobalSurfaceWater - "occurrence" band shows % of time water present.
    Pixels with occurrence > 50% are considered water.
    Also uses MODIS Land Cover as backup.
    """
    results = np.ones(len(coords), dtype=bool)  # default to land
    
    # JRC Global Surface Water - occurrence band (0-100, % time with water)
    gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence')
    
    # Process in batches
    for batch_start in range(0, len(coords), batch_size):
        batch_end = min(batch_start + batch_size, len(coords))
        batch_coords = coords[batch_start:batch_end]
        
        # Create feature collection of points
        features = []
        for i, (lat, lon) in enumerate(batch_coords):
            features.append(ee.Feature(ee.Geometry.Point([lon, lat]), {'idx': batch_start + i}))
        fc = ee.FeatureCollection(features)
        
        # Sample the water occurrence at each point
        sampled = gsw.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.first(),
            scale=30  # 30m resolution
        )
        
        # Get results
        try:
            result_list = sampled.getInfo()['features']
            for feat in result_list:
                idx = feat['properties']['idx']
                occurrence = feat['properties'].get('first', 0)
                if occurrence is None:
                    occurrence = 0
                # If water occurrence > 50%, consider it water
                if occurrence > 50:
                    results[idx] = False
        except Exception as e:
            print(f"  Warning: batch {batch_start}-{batch_end} failed: {e}")
        
        if (batch_end) % 2000 == 0 or batch_end == len(coords):
            water_count = (~results[:batch_end]).sum()
            print(f"  Processed {batch_end}/{len(coords)}, water so far: {water_count}")
    
    return results


def check_water_single_fast(lat, lon):
    """Check single point for water using GEE (for testing)."""
    gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence')
    point = ee.Geometry.Point([lon, lat])
    result = gsw.sample(point, scale=30).first().getInfo()
    if result is None:
        return True  # No data = assume land
    occurrence = result.get('properties', {}).get('occurrence', 0) or 0
    return occurrence <= 50  # True = land


print("=" * 60)
print("STEP 1: Load positive samples")
print("=" * 60)

pos_meta = pd.read_csv('data/processed/all_positive_metadata.csv')
pos_emb = np.load('data/processed/all_positive_embeddings.npy')
print(f"Positives: {len(pos_meta)} samples")

print("\n" + "=" * 60)
print("STEP 2: Check positives for water (GEE)")
print("=" * 60)

pos_coords = list(zip(pos_meta['lat'], pos_meta['lon']))
t0 = time.time()
pos_on_land = check_water_batch_gee(pos_coords, batch_size=500)
print(f"Time: {time.time() - t0:.1f}s")
print(f"On land: {pos_on_land.sum()} / {len(pos_on_land)} ({100*pos_on_land.mean():.1f}%)")
print(f"In water: {(~pos_on_land).sum()}")

print("\n" + "=" * 60)
print("STEP 3: Load and check negative samples")
print("=" * 60)

neg_df = pd.read_csv('data/processed/combined_negative_embeddings.csv')
print(f"Negatives: {len(neg_df)} samples")

emb_cols = [f'emb_{i}' for i in range(64)]
neg_emb = neg_df[emb_cols].values.astype(np.float32)

neg_coords = list(zip(neg_df['lat'], neg_df['lon']))
t0 = time.time()
neg_on_land = check_water_batch_gee(neg_coords, batch_size=500)
print(f"Time: {time.time() - t0:.1f}s")
print(f"On land: {neg_on_land.sum()} / {len(neg_on_land)} ({100*neg_on_land.mean():.1f}%)")
print(f"In water: {(~neg_on_land).sum()}")

print("\n" + "=" * 60)
print("STEP 4: Filter and save")
print("=" * 60)

pos_emb_filtered = pos_emb[pos_on_land]
pos_meta_filtered = pos_meta[pos_on_land].reset_index(drop=True)
neg_emb_filtered = neg_emb[neg_on_land]

print(f"Filtered positives: {len(pos_emb_filtered)}")
print(f"Filtered negatives: {len(neg_emb_filtered)}")

X = np.vstack([pos_emb_filtered, neg_emb_filtered]).astype(np.float32)
y = np.concatenate([np.ones(len(pos_emb_filtered)), np.zeros(len(neg_emb_filtered))]).astype(np.float32)

print(f"Combined: X={X.shape}, y={y.shape}")

np.save('data/processed/expanded_X_land_only.npy', X)
np.save('data/processed/expanded_y_land_only.npy', y)
pos_meta_filtered.to_csv('data/processed/all_positive_metadata_land_only.csv', index=False)

# Save water points for review
water_pos = pos_meta[~pos_on_land]
water_pos.to_csv('data/processed/water_positives_removed.csv', index=False)
print(f"Saved removed positives: {len(water_pos)}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Original: {len(pos_meta)} positives, {len(neg_df)} negatives = {len(pos_meta)+len(neg_df)}")
print(f"After:    {len(pos_emb_filtered)} positives, {len(neg_emb_filtered)} negatives = {len(X)}")
print(f"Removed:  {len(pos_meta)-len(pos_emb_filtered)} positives, {len(neg_df)-len(neg_emb_filtered)} negatives")
