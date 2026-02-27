#!/usr/bin/env python3
"""
Extract non-yew training negatives from logged areas across the full CWH BEC zone.

Uses:
  - 300k-point CWH GEE sample (embeddings + model predictions)
  - Local BC VRI GDB for:
      * Suitability classification (logged / non-treed = low suitability)
      * Water polygon extraction for 100 m exclusion buffer

Points must satisfy ALL conditions:
  1. Model probability >= prob_threshold  (spectral false positive)
  2. VRI suitability  <= suit_threshold   (logged or non-vegetated area)
  3. Distance to nearest water polygon   >= water_buffer_m  (default 100 m)

Output: CSV with emb_0..emb_63, lat, lon, raw_prob, suitability — compatible
        with the --gee-negatives flag in classify_tiled_gpu.py.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
from shapely.ops import unary_union

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parents[2]))
from scripts.preprocessing.apply_forestry_mask import load_forestry_data, classify_polygons


GDB_PATH = '/home/jericho/Downloads/firefox-downloads/VEG_COMP_LYR_R1_POLY_2024.gdb'

# Tile size in degrees for loading VRI chunks
TILE_DEG = 0.5   # ~50 km at BC latitudes — large enough for 100 m buffer math
BUFFER_M = 100   # metres — exclusion zone around water/rivers


def load_cwh_samples(pred_csv, emb_csv):
    """Join sample_predictions.csv with embedding CSV by row index."""
    print("Loading CWH sample predictions + embeddings...")
    preds = pd.read_csv(pred_csv)
    embs  = pd.read_csv(emb_csv)
    if len(preds) != len(embs):
        raise ValueError(f"Row mismatch: {len(preds)} predictions vs {len(embs)} embeddings")

    emb_cols = [c for c in embs.columns if c.startswith('emb_')]
    df = pd.concat([preds[['lat','lon','prob']], embs[emb_cols]], axis=1)
    print(f"  {len(df):,} points, lat {df.lat.min():.2f}–{df.lat.max():.2f}, "
          f"lon {df.lon.min():.2f}–{df.lon.max():.2f}")
    return df


def build_tile_grid(lats, lons, tile_deg=TILE_DEG):
    """Return list of (lat_min, lat_max, lon_min, lon_max) tiles covering all points."""
    import math
    lat_min_g = math.floor(lats.min() / tile_deg) * tile_deg
    lat_max_g = math.ceil (lats.max() / tile_deg) * tile_deg
    lon_min_g = math.floor(lons.min() / tile_deg) * tile_deg
    lon_max_g = math.ceil (lons.max() / tile_deg) * tile_deg

    tiles = []
    lat = lat_min_g
    while lat < lat_max_g:
        lon = lon_min_g
        while lon < lon_max_g:
            tiles.append((lat, lat + tile_deg, lon, lon + tile_deg))
            lon += tile_deg
        lat += tile_deg
    return tiles


def check_tile(candidates_df, gdb_path, lat_min, lat_max, lon_min, lon_max,
               suit_threshold, water_buffer_m):
    """
    For one geographic tile:
      1. Load VRI polygons
      2. Classify suitability
      3. Extract water polygons and buffer them by water_buffer_m
      4. Return rows of candidates_df that pass both filters.
    """
    # Small expand so border points get proper polygon coverage
    expand = 0.01
    try:
        gdf = load_forestry_data(
            gdb_path,
            lat_min - expand, lat_max + expand,
            lon_min - expand, lon_max + expand
        )
    except Exception as e:
        print(f"    VRI load error for tile ({lat_min:.2f},{lon_min:.2f}): {e}")
        return pd.DataFrame()

    if gdf is None or len(gdf) == 0:
        return pd.DataFrame()

    # Classify suitability
    try: 
        gdf = classify_polygons(gdf)
    except Exception as e:
        print(f"    classify_polygons error: {e}")
        return pd.DataFrame()

    # -------------------------------------------------------------------------
    # Spatial join: assign each candidate point its VRI suitability
    # -------------------------------------------------------------------------
    pts_gdf = gpd.GeoDataFrame(
        candidates_df.copy(),
        geometry=[Point(lon, lat) for lat, lon in
                  zip(candidates_df.lat, candidates_df.lon)],
        crs='EPSG:4326'
    )

    # Keep only polygons with the columns we need
    poly_cols = ['geometry', 'yew_suitability', 'BCLCS_LEVEL_2']
    gdf_sub = gdf[poly_cols].copy()

    joined = gpd.sjoin(pts_gdf, gdf_sub, how='left', predicate='within')

    # If a point falls in multiple polygons take the minimum suitability
    # (most restrictive / most logged)
    if 'yew_suitability_right' in joined.columns:
        suit_col = 'yew_suitability_right'
    elif 'yew_suitability' in joined.columns:
        suit_col = 'yew_suitability'
    else:
        return pd.DataFrame()

    joined = (joined
              .groupby(joined.index, sort=False)[suit_col]
              .min()
              .rename('suitability')
              .pipe(lambda s: candidates_df.join(s)))

    # Drop unmatched (no VRI polygon)
    joined = joined.dropna(subset=['suitability'])

    # Filter by suitability threshold
    logged_mask = joined['suitability'] <= suit_threshold
    logged = joined[logged_mask].copy()
    if len(logged) == 0:
        return pd.DataFrame()

    # -------------------------------------------------------------------------
    # Water proximity filter — work in BC Albers (EPSG:3005, metres)
    # -------------------------------------------------------------------------
    water_polys = gdf[gdf['BCLCS_LEVEL_2'] == 'W']

    # Also treat non-vegetated / water (suitability==0, mask_reason contains 'water')
    if 'mask_reason' in gdf.columns:
        water_polys = pd.concat([
            water_polys,
            gdf[gdf['mask_reason'] == 'water']
        ]).drop_duplicates()

    if len(water_polys) == 0:
        # No water in this tile — all logged candidates pass
        logged['water_dist_m'] = np.inf
        return logged

    # Project to metres (EPSG:3005 = BC Albers, unit = metres)
    water_proj = water_polys.to_crs('EPSG:3005').copy()

    # Fix invalid/null geometries before buffering
    water_proj = water_proj[water_proj.geometry.notna()]
    water_proj.geometry = water_proj.geometry.buffer(0)   # repairs minor issues
    water_proj = water_proj[water_proj.geometry.is_valid & ~water_proj.geometry.is_empty]

    if len(water_proj) == 0:
        logged['water_dist_m'] = np.inf
        return logged

    # Build 100 m exclusion zone around all water polygons
    water_buffer_geom = unary_union(water_proj.geometry.buffer(water_buffer_m))

    # Project candidate points to EPSG:3005
    pts_geom = [Point(lon, lat) for lat, lon in zip(logged.lon, logged.lat)]
    pts_proj = gpd.GeoSeries(pts_geom, crs='EPSG:4326').to_crs('EPSG:3005')

    # Exclude points inside the exclusion buffer
    in_buffer = pts_proj.within(water_buffer_geom)

    # Compute distance to original (unbuffered) water for reporting
    water_orig = unary_union(water_proj.geometry)
    dists = pts_proj.distance(water_orig).values
    dists = np.where(np.isnan(dists) | np.isinf(dists), water_buffer_m * 10, dists)

    logged = logged.copy()
    logged['water_dist_m'] = dists
    return logged[~in_buffer.values]


def main():
    parser = argparse.ArgumentParser(
        description='Extract CWH-wide logged-area non-yew negatives with water buffer',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--pred-csv', default=
        'results/analysis/cwh_yew_forestry_300k/sample_predictions.csv')
    parser.add_argument('--emb-csv', default=
        'results/analysis/cwh_yew_forestry_300k/embeddings_300000_seed42_year2024_tiled.csv')
    parser.add_argument('--gdb', default=GDB_PATH,
        help='Path to BC VRI geodatabase')
    parser.add_argument('--prob-threshold', type=float, default=0.5,
        help='Minimum model probability to consider (default: 0.5)')
    parser.add_argument('--suit-threshold', type=float, default=0.0,
        help='Maximum VRI suitability to count as logged (default: 0.0 = recently logged/non-veg)')
    parser.add_argument('--water-buffer', type=float, default=100.0,
        help='Minimum distance from water polygons in metres (default: 100)')
    parser.add_argument('--max-samples', type=int, default=1000,
        help='Maximum output negatives (default: 1000)')
    parser.add_argument('--output', default=
        'data/processed/cwh_logged_negatives/cwh_logged_negative_embeddings.csv')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load & filter candidates
    # ------------------------------------------------------------------
    df = load_cwh_samples(args.pred_csv, args.emb_csv)

    candidates = df[df['prob'] >= args.prob_threshold].copy()
    print(f"\n  Candidates (P>={args.prob_threshold}): {len(candidates):,}")

    # ------------------------------------------------------------------
    # 2. Build tile grid over candidates
    # ------------------------------------------------------------------
    tiles = build_tile_grid(candidates.lat, candidates.lon, tile_deg=TILE_DEG)
    print(f"  Geographic tile grid: {len(tiles)} tiles "
          f"({TILE_DEG}° × {TILE_DEG}°)")

    # ------------------------------------------------------------------
    # 3. Process tiles
    # ------------------------------------------------------------------
    results = []
    emb_cols = [c for c in df.columns if c.startswith('emb_')]

    for i, (lat_min, lat_max, lon_min, lon_max) in enumerate(tiles):
        in_tile = candidates[
            (candidates.lat >= lat_min) & (candidates.lat < lat_max) &
            (candidates.lon >= lon_min) & (candidates.lon < lon_max)
        ]
        if len(in_tile) == 0:
            continue

        print(f"  Tile {i+1}/{len(tiles)} "
              f"({lat_min:.2f}–{lat_max:.2f}N, {lon_min:.2f}–{lon_max:.2f}W): "
              f"{len(in_tile)} candidates...", end='', flush=True)

        passed = check_tile(
            in_tile, args.gdb,
            lat_min, lat_max, lon_min, lon_max,
            args.suit_threshold, args.water_buffer
        )
        print(f" → {len(passed)} pass")
        if len(passed) > 0:
            results.append(passed)

    if not results:
        print("\nERROR: No qualifying points found. Try relaxing thresholds.")
        return

    all_results = pd.concat(results, ignore_index=True)
    print(f"\nTotal qualifying: {len(all_results):,} "
          f"(logged, P>={args.prob_threshold}, >{args.water_buffer:.0f}m from water)")

    # ------------------------------------------------------------------
    # 4. Sample down to max_samples (prefer highest prob)
    # ------------------------------------------------------------------
    if len(all_results) > args.max_samples:
        rng = np.random.RandomState(args.seed)
        n_top    = int(args.max_samples * 0.6)
        n_random = args.max_samples - n_top
        top_idx  = all_results['prob'].nlargest(n_top).index
        rest_idx = all_results.index.difference(top_idx)
        rand_idx = rng.choice(rest_idx, size=min(n_random, len(rest_idx)), replace=False)
        selected = all_results.loc[list(top_idx) + list(rand_idx)].copy()
    else:
        selected = all_results.copy()

    selected = selected.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    out_cols = emb_cols + ['lat', 'lon', 'prob', 'suitability', 'water_dist_m']
    # rename 'prob' -> 'raw_prob' to match existing loader
    selected = selected.rename(columns={'prob': 'raw_prob'})
    out_cols = emb_cols + ['lat', 'lon', 'raw_prob', 'suitability', 'water_dist_m']
    selected[out_cols].to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  CWH candidates (P>={args.prob_threshold}): {len(candidates):,}")
    print(f"  After VRI logged + {args.water_buffer:.0f}m water filter: {len(all_results):,}")
    print(f"  Saved:    {len(selected)}")
    print(f"  Output:   {out_path}")
    print(f"\n  Suitability breakdown:")
    for s, n in selected['suitability'].value_counts().sort_index().items():
        print(f"    {s:.2f}: {n}")
    print(f"\n  Water distance stats:")
    wd = selected['water_dist_m']
    print(f"    min={wd.min():.0f}m  median={wd.median():.0f}m  max={wd.max():.0f}m")
    print(f"\n  Lat range: {selected.lat.min():.3f}–{selected.lat.max():.3f}°N")
    print(f"  Lon range: {selected.lon.min():.3f}–{selected.lon.max():.3f}°W")

    print(f"\nTo use in training:")
    print(f"  python scripts/prediction/classify_tiled_gpu.py \\")
    print(f"    --gee-negatives {out_path} --gee-negatives-weight 2")


if __name__ == '__main__':
    main()
