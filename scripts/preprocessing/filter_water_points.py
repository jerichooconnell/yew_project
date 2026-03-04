#!/usr/bin/env python3
"""
Filter training data to remove points in water, then retrain MLP.
Uses Natural Earth 10m land shapefile to identify land vs water points.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
import time

print("=" * 60)
print("STEP 1: Load land shapefile")
print("=" * 60)

land_shp = Path('data/lookup_tables/ne_10m_land/ne_10m_land.shp')
print(f"Loading {land_shp}...")
land = gpd.read_file(land_shp)
print(f"  CRS: {land.crs}")
print(f"  Polygons: {len(land)}")

# Create a single unified land geometry for faster spatial queries
land_union = land.unary_union
print("  Created unified land geometry")

def is_on_land(lat, lon):
    """Check if a point is on land."""
    point = Point(lon, lat)  # shapely uses (x, y) = (lon, lat)
    return land_union.contains(point)

print("\n" + "=" * 60)
print("STEP 2: Load positive samples")
print("=" * 60)

# Load positive metadata and embeddings
pos_meta = pd.read_csv('data/processed/all_positive_metadata.csv')
pos_emb = np.load('data/processed/all_positive_embeddings.npy')
print(f"Positives: {len(pos_meta)} samples, embeddings shape: {pos_emb.shape}")

# Check which positives are on land
print("Checking which positives are on land...")
t0 = time.time()
pos_on_land = []
for i, (lat, lon) in enumerate(zip(pos_meta['lat'], pos_meta['lon'])):
    pos_on_land.append(is_on_land(lat, lon))
    if (i + 1) % 1000 == 0:
        print(f"  {i+1}/{len(pos_meta)} checked...")
pos_on_land = np.array(pos_on_land)
print(f"  Time: {time.time() - t0:.1f}s")
print(f"  On land: {pos_on_land.sum()} / {len(pos_on_land)} ({100*pos_on_land.mean():.1f}%)")
print(f"  In water: {(~pos_on_land).sum()}")

print("\n" + "=" * 60)
print("STEP 3: Load negative samples")
print("=" * 60)

# Load negatives
neg_df = pd.read_csv('data/processed/combined_negative_embeddings.csv')
print(f"Negatives: {len(neg_df)} samples")

# Extract embeddings and coordinates
emb_cols = [f'emb_{i}' for i in range(64)]
neg_emb = neg_df[emb_cols].values.astype(np.float32)
neg_coords = neg_df[['lat', 'lon']].values

# Check which negatives are on land
print("Checking which negatives are on land...")
t0 = time.time()
neg_on_land = []
for i, (lat, lon) in enumerate(neg_coords):
    neg_on_land.append(is_on_land(lat, lon))
    if (i + 1) % 1000 == 0:
        print(f"  {i+1}/{len(neg_coords)} checked...")
neg_on_land = np.array(neg_on_land)
print(f"  Time: {time.time() - t0:.1f}s")
print(f"  On land: {neg_on_land.sum()} / {len(neg_on_land)} ({100*neg_on_land.mean():.1f}%)")
print(f"  In water: {(~neg_on_land).sum()}")

print("\n" + "=" * 60)
print("STEP 4: Filter and save training data")
print("=" * 60)

# Filter
pos_emb_filtered = pos_emb[pos_on_land]
pos_meta_filtered = pos_meta[pos_on_land].reset_index(drop=True)
neg_emb_filtered = neg_emb[neg_on_land]

print(f"Filtered positives: {len(pos_emb_filtered)}")
print(f"Filtered negatives: {len(neg_emb_filtered)}")

# Combine
X = np.vstack([pos_emb_filtered, neg_emb_filtered]).astype(np.float32)
y = np.concatenate([np.ones(len(pos_emb_filtered)), np.zeros(len(neg_emb_filtered))]).astype(np.float32)

print(f"Combined: X={X.shape}, y={y.shape}")
print(f"  Positives: {y.sum():.0f}, Negatives: {(1-y).sum():.0f}")

# Save
np.save('data/processed/expanded_X_land_only.npy', X)
np.save('data/processed/expanded_y_land_only.npy', y)
pos_meta_filtered.to_csv('data/processed/all_positive_metadata_land_only.csv', index=False)
print("Saved: expanded_X_land_only.npy, expanded_y_land_only.npy")

# Also save the water points for reference
water_pos = pos_meta[~pos_on_land]
water_pos.to_csv('data/processed/water_positives_removed.csv', index=False)
print(f"Saved removed water positives: {len(water_pos)} samples")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Original: {len(pos_meta)} positives, {len(neg_df)} negatives")
print(f"After filter: {len(pos_emb_filtered)} positives, {len(neg_emb_filtered)} negatives")
print(f"Removed from water: {len(pos_meta) - len(pos_emb_filtered)} positives, {len(neg_df) - len(neg_emb_filtered)} negatives")
