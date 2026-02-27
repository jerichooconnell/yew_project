#!/usr/bin/env python3
"""
Extract non-yew training negatives from FAIB tree detail inventory.

Strategy:
  1. Load faib_tree_detail.csv and identify sites that contain TW (Pacific yew)
  2. Load faib_header.csv for site coordinates and BEC zone info
  3. Exclude all sites that have ANY TW tree records
  4. Randomly sample from remaining sites across all BEC zones
  5. Extract 64-band GEE satellite embeddings at each site location
  6. Save as CSV with emb_0..emb_63, lat, lon, bec_zone — compatible with
     the --gee-negatives flag in classify_tiled_gpu.py

The FAIB inventory covers 12 BEC zones across all of BC, providing diverse
forest-type negatives that teach the model what non-yew forests look like
across the province — not just in the CWH zone.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def identify_non_tw_sites(tree_detail_csv, header_csv):
    """
    Find FAIB sites that (a) have no TW trees and (b) have lat/lon coordinates.

    Returns:
        DataFrame with SITE_IDENTIFIER, Longitude, Latitude, BEC_ZONE, BEC_SBZ
    """
    print("Loading FAIB data...")
    tree = pd.read_csv(tree_detail_csv, dtype=str, usecols=['SITE_IDENTIFIER', 'SPECIES'])
    header = pd.read_csv(header_csv, dtype=str)

    # Sites that contain TW (yew) trees — must exclude
    tw_sites = set(tree[tree['SPECIES'] == 'TW']['SITE_IDENTIFIER'].unique())
    print(f"  Total sites in tree_detail: {tree['SITE_IDENTIFIER'].nunique()}")
    print(f"  Sites with TW (yew): {len(tw_sites)}")

    # Sites with valid coordinates
    header = header.copy()
    header['Latitude'] = pd.to_numeric(header['Latitude'], errors='coerce')
    header['Longitude'] = pd.to_numeric(header['Longitude'], errors='coerce')
    header_with_coords = header[header['Latitude'].notna() & header['Longitude'].notna()]
    print(f"  Sites with coordinates: {len(header_with_coords)}")

    # Filter out TW sites
    non_tw = header_with_coords[~header_with_coords['SITE_IDENTIFIER'].isin(tw_sites)].copy()
    print(f"  Non-TW sites with coordinates: {len(non_tw)}")

    # BEC zone distribution
    print(f"\n  BEC zone distribution of candidate negative sites:")
    for zone, count in non_tw['BEC_ZONE'].value_counts().items():
        print(f"    {zone}: {count}")

    return non_tw[['SITE_IDENTIFIER', 'Longitude', 'Latitude', 'BEC_ZONE', 'BEC_SBZ']]


def sample_negatives(candidates, n_samples=None, seed=42):
    """
    Randomly sample from candidate sites.

    If n_samples is None, use all candidates.
    """
    if n_samples is not None and n_samples < len(candidates):
        sampled = candidates.sample(n=n_samples, random_state=seed)
        print(f"\n  Sampled {n_samples} from {len(candidates)} candidates (seed={seed})")
    else:
        sampled = candidates.copy()
        print(f"\n  Using all {len(candidates)} candidate sites")

    print(f"  BEC zone distribution of sampled sites:")
    for zone, count in sampled['BEC_ZONE'].value_counts().items():
        print(f"    {zone}: {count}")

    return sampled


def extract_embeddings_gee(sites_df, year, gee_project, batch_size=200):
    """
    Extract 64-band satellite embeddings at each FAIB site location using GEE.

    Args:
        sites_df: DataFrame with Longitude, Latitude columns
        year: Imagery year
        gee_project: GEE project ID
        batch_size: Points per GEE request

    Returns:
        DataFrame with emb_0..emb_63, lat, lon, site_identifier, bec_zone
    """
    import ee

    print(f"\n  Initializing GEE project: {gee_project}")
    ee.Initialize(project=gee_project)

    band_names = [f'A{i:02d}' for i in range(64)]
    n_points = len(sites_df)
    print(f"  Extracting embeddings for {n_points} FAIB sites...")
    print(f"  Collection: GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL, year={year}")

    lats = sites_df['Latitude'].values.astype(float)
    lons = sites_df['Longitude'].values.astype(float)
    site_ids = sites_df['SITE_IDENTIFIER'].values
    bec_zones = sites_df['BEC_ZONE'].values

    all_features = []
    all_lats = []
    all_lons = []
    all_site_ids = []
    all_bec_zones = []
    failed = 0

    batches = list(range(0, n_points, batch_size))
    for batch_idx, start in enumerate(tqdm(batches, desc='  GEE batches')):
        end = min(start + batch_size, n_points)
        batch_lats = lats[start:end]
        batch_lons = lons[start:end]
        batch_sids = site_ids[start:end]
        batch_becs = bec_zones[start:end]

        # Build bounding box with padding
        pad = 0.05
        bbox = ee.Geometry.Rectangle([
            float(batch_lons.min()) - pad,
            float(batch_lats.min()) - pad,
            float(batch_lons.max()) + pad,
            float(batch_lats.max()) + pad,
        ])

        try:
            emb_image = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
                         .filterDate(f'{year}-01-01', f'{year+1}-01-01')
                         .filterBounds(bbox)
                         .mosaic()
                         .select(band_names)
                         .toFloat())

            ee_points = ee.FeatureCollection([
                ee.Feature(
                    ee.Geometry.Point([float(batch_lons[i]), float(batch_lats[i])]),
                    {'idx': int(i), 'lat': float(batch_lats[i]), 'lon': float(batch_lons[i])}
                )
                for i in range(len(batch_lats))
            ])

            sampled = emb_image.sampleRegions(
                collection=ee_points,
                scale=10,
                geometries=False,
                tileScale=4,
            )

            # Retrieve in pages
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
                    idx = props.get('idx')
                    band_vals = [props.get(f'A{b:02d}') for b in range(64)]
                    if all(v is not None for v in band_vals) and lat is not None:
                        all_features.append(band_vals)
                        all_lats.append(lat)
                        all_lons.append(lon)
                        all_site_ids.append(batch_sids[idx] if idx is not None else '')
                        all_bec_zones.append(batch_becs[idx] if idx is not None else '')
                    else:
                        failed += 1
                if len(result) < page_size:
                    break
                offset += page_size

        except Exception as e:
            print(f"\n  Batch {batch_idx+1} failed: {e}")
            failed += len(batch_lats)

        time.sleep(0.5)

    print(f"\n  ✓ Extracted {len(all_features)} valid embeddings")
    print(f"  Failed/outside coverage: {failed}")

    if len(all_features) == 0:
        raise RuntimeError("No embeddings extracted. Check GEE auth and coverage.")

    X = np.array(all_features, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Filter out zero-vector rows
    valid = np.any(X != 0, axis=1)
    print(f"  Valid (non-zero) embeddings: {valid.sum()} / {len(X)}")

    result_df = pd.DataFrame(X[valid], columns=[f'emb_{i}' for i in range(64)])
    result_df['lat'] = np.array(all_lats)[valid]
    result_df['lon'] = np.array(all_lons)[valid]
    result_df['site_identifier'] = np.array(all_site_ids)[valid]
    result_df['bec_zone'] = np.array(all_bec_zones)[valid]

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description='Extract FAIB tree inventory negatives with GEE embeddings'
    )
    parser.add_argument('--tree-detail', default='data/raw/faib_tree_detail.csv',
                        help='Path to faib_tree_detail.csv')
    parser.add_argument('--header', default='data/raw/faib_header.csv',
                        help='Path to faib_header.csv')
    parser.add_argument('--n-samples', type=int, default=None,
                        help='Number of negative sites to sample (default: all non-TW sites)')
    parser.add_argument('--year', type=int, default=2024,
                        help='Imagery year')
    parser.add_argument('--gee-project', default='carbon-storm-206002',
                        help='GEE project ID')
    parser.add_argument('--batch-size', type=int, default=200,
                        help='Points per GEE batch request')
    parser.add_argument('--output-dir', default='data/processed/faib_negatives',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Identify non-TW sites
    print("=" * 70)
    print("STEP 1: Identify non-TW FAIB sites")
    print("=" * 70)
    candidates = identify_non_tw_sites(args.tree_detail, args.header)

    # Step 2: Sample
    print("\n" + "=" * 70)
    print("STEP 2: Sample negative sites")
    print("=" * 70)
    sampled = sample_negatives(candidates, n_samples=args.n_samples, seed=args.seed)

    # Save site list before GEE extraction
    sites_csv = output_dir / 'faib_negative_sites.csv'
    sampled.to_csv(sites_csv, index=False)
    print(f"  Saved site list: {sites_csv}")

    # Step 3: Extract GEE embeddings
    print("\n" + "=" * 70)
    print("STEP 3: Extract GEE satellite embeddings")
    print("=" * 70)
    result_df = extract_embeddings_gee(
        sampled, args.year, args.gee_project, batch_size=args.batch_size
    )

    # Save embeddings
    emb_csv = output_dir / 'faib_negative_embeddings.csv'
    result_df.to_csv(emb_csv, index=False)
    print(f"\n  Saved embeddings: {emb_csv}")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total FAIB sites without TW: {len(candidates)}")
    print(f"  Sites sampled: {len(sampled)}")
    print(f"  Valid embeddings extracted: {len(result_df)}")
    print(f"  BEC zone coverage:")
    for zone, count in result_df['bec_zone'].value_counts().items():
        print(f"    {zone}: {count}")
    print(f"\n  Output: {emb_csv}")
    print(f"  Use with: --gee-negatives {emb_csv}")


if __name__ == '__main__':
    main()
