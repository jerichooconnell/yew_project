#!/usr/bin/env python3
"""
Extract non-yew training negatives from LOGGED areas identified by local BC VRI data.

Cross-references the raw probability grid (prob_grid.npy) with the forestry
suitability grid (suitability_grid.npy) to find pixels where:
  - The model predicts high yew probability (false positives)
  - The BC VRI data shows the area has been logged (low suitability)

These are reliable non-yew training examples: logged areas where yew doesn't
regenerate after clearcutting.

Extracts the 64-band satellite embeddings from cached tile .npy files and
saves as CSV compatible with the --gee-negatives flag in classify_tiled_gpu.py.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_tile_info(pred_dir):
    """Load tile layout info."""
    with open(pred_dir / 'tile_info.json') as f:
        return json.load(f)


def pixel_to_tile_coords(px_row, px_col, tile_info):
    """Convert global pixel row/col to (tile_row, tile_col, local_row, local_col)."""
    n_rows = tile_info['n_rows']
    n_cols = tile_info['n_cols']
    
    # All tiles have the same shape in this grid
    tile_shape = tile_info['tiles'][0]['shape']  # [rows, cols]
    tile_h, tile_w = tile_shape
    
    tile_row = px_row // tile_h
    tile_col = px_col // tile_w
    local_row = px_row % tile_h
    local_col = px_col % tile_w
    
    # Clamp to valid range
    tile_row = min(tile_row, n_rows - 1)
    tile_col = min(tile_col, n_cols - 1)
    
    return tile_row, tile_col, local_row, local_col


def pixel_to_latlon(px_row, px_col, bbox, grid_shape):
    """Convert pixel coordinates to lat/lon."""
    h, w = grid_shape
    lat = bbox['north'] - (px_row / h) * (bbox['north'] - bbox['south'])
    lon = bbox['west'] + (px_col / w) * (bbox['east'] - bbox['west'])
    return lat, lon


def main():
    parser = argparse.ArgumentParser(
        description='Extract logged-area negatives from local VRI forestry data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: prob >= 0.5, suitability <= 0.15, max 1000 samples
  python scripts/preprocessing/extract_logged_negatives.py

  # Stricter: only very high prob in very recently logged areas
  python scripts/preprocessing/extract_logged_negatives.py \\
    --prob-threshold 0.7 --suit-threshold 0.05 --max-samples 500
"""
    )
    parser.add_argument('--pred-dir', type=str,
                        default='results/predictions/south_vi_large',
                        help='Prediction directory with prob_grid.npy and tile cache')
    parser.add_argument('--forestry-dir', type=str,
                        default='results/predictions/south_vi_large_forestry',
                        help='Forestry directory with suitability_grid.npy')
    parser.add_argument('--prob-threshold', type=float, default=0.5,
                        help='Minimum raw probability to consider (default: 0.5)')
    parser.add_argument('--suit-threshold', type=float, default=0.15,
                        help='Maximum suitability to consider as "logged" (default: 0.15)')
    parser.add_argument('--max-samples', type=int, default=1000,
                        help='Maximum number of negative samples (default: 1000)')
    parser.add_argument('--output', type=str,
                        default='data/processed/logged_negatives/logged_negative_embeddings.csv',
                        help='Output CSV path')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling')
    args = parser.parse_args()
    
    pred_dir = Path(args.pred_dir)
    forestry_dir = Path(args.forestry_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("EXTRACTING LOGGED-AREA NEGATIVES FROM LOCAL VRI DATA")
    print("=" * 60)
    
    # Load grids
    print("\nLoading grids...")
    prob_grid = np.load(pred_dir / 'prob_grid.npy')
    suit_grid = np.load(forestry_dir / 'suitability_grid.npy')
    
    print(f"  prob_grid:  {prob_grid.shape}, range [{prob_grid.min():.4f}, {prob_grid.max():.4f}]")
    print(f"  suit_grid:  {suit_grid.shape}, range [{suit_grid.min():.4f}, {suit_grid.max():.4f}]")
    
    # Ensure same shape
    min_h = min(prob_grid.shape[0], suit_grid.shape[0])
    min_w = min(prob_grid.shape[1], suit_grid.shape[1])
    prob_grid = prob_grid[:min_h, :min_w]
    suit_grid = suit_grid[:min_h, :min_w]
    
    # Find logged high-probability pixels
    print(f"\nFinding pixels with prob >= {args.prob_threshold} AND suitability <= {args.suit_threshold}...")
    
    # Show distribution first
    for st in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
        count = np.sum((prob_grid >= args.prob_threshold) & (suit_grid <= st) & (suit_grid >= 0))
        print(f"  suitability <= {st:.2f}: {count:,} pixels")
    
    mask = (prob_grid >= args.prob_threshold) & (suit_grid <= args.suit_threshold) & (suit_grid >= 0)
    candidate_count = mask.sum()
    print(f"\n  Total candidates: {candidate_count:,} pixels")
    
    if candidate_count == 0:
        print("  ERROR: No candidate pixels found! Try relaxing thresholds.")
        return
    
    # Get pixel coordinates of candidates
    rows, cols = np.where(mask)
    probs = prob_grid[rows, cols]
    suits = suit_grid[rows, cols]
    
    # Sample if too many — prefer highest prob (worst false positives)
    rng = np.random.RandomState(args.seed)
    if len(rows) > args.max_samples:
        # Sort by probability descending (take the worst false positives first)
        sort_idx = np.argsort(-probs)
        # Take top 60% by probability, random 40% to add diversity
        n_top = int(args.max_samples * 0.6)
        n_random = args.max_samples - n_top
        
        top_idx = sort_idx[:n_top]
        remaining_idx = sort_idx[n_top:]
        random_idx = rng.choice(remaining_idx, size=min(n_random, len(remaining_idx)), replace=False)
        
        selected_idx = np.concatenate([top_idx, random_idx])
        rng.shuffle(selected_idx)  # Mix them up
    else:
        selected_idx = np.arange(len(rows))
    
    sel_rows = rows[selected_idx]
    sel_cols = cols[selected_idx]
    sel_probs = probs[selected_idx]
    sel_suits = suits[selected_idx]
    
    print(f"  Selected: {len(sel_rows)} samples")
    print(f"  Prob range: [{sel_probs.min():.3f}, {sel_probs.max():.3f}], mean={sel_probs.mean():.3f}")
    print(f"  Suit range: [{sel_suits.min():.3f}, {sel_suits.max():.3f}]")
    
    # Load tile info and extract embeddings
    print("\nLoading tile info and extracting embeddings...")
    tile_info = load_tile_info(pred_dir)
    tile_cache_dir = pred_dir / 'tiles'
    
    with open(pred_dir / 'metadata.json') as f:
        metadata = json.load(f)
    bbox = metadata['bbox']
    grid_shape = (min_h, min_w)
    
    # Group pixels by tile for efficient loading
    tile_groups = {}
    for i, (r, c) in enumerate(zip(sel_rows, sel_cols)):
        tr, tc, lr, lc = pixel_to_tile_coords(r, c, tile_info)
        key = (tr, tc)
        if key not in tile_groups:
            tile_groups[key] = []
        tile_groups[key].append((i, lr, lc))
    
    print(f"  Pixels spread across {len(tile_groups)} tiles")
    
    # Extract embeddings
    embeddings = np.zeros((len(sel_rows), 64), dtype=np.float32)
    valid_mask = np.ones(len(sel_rows), dtype=bool)
    
    for (tr, tc), pixels in sorted(tile_groups.items()):
        emb_path = tile_cache_dir / f'emb_{tr}_{tc}.npy'
        if not emb_path.exists():
            print(f"  WARNING: Missing {emb_path}, skipping {len(pixels)} pixels")
            for i, _, _ in pixels:
                valid_mask[i] = False
            continue
        
        emb_tile = np.load(emb_path)  # shape: (tile_h, tile_w, 64)
        for i, lr, lc in pixels:
            if lr < emb_tile.shape[0] and lc < emb_tile.shape[1]:
                embeddings[i] = emb_tile[lr, lc]
            else:
                valid_mask[i] = False
    
    # Filter out invalid
    embeddings = embeddings[valid_mask]
    sel_rows = sel_rows[valid_mask]
    sel_cols = sel_cols[valid_mask]
    sel_probs = sel_probs[valid_mask]
    sel_suits = sel_suits[valid_mask]
    
    # Also filter out all-zero embeddings (ocean/nodata)
    nonzero = np.any(embeddings != 0, axis=1)
    embeddings = embeddings[nonzero]
    sel_rows = sel_rows[nonzero]
    sel_cols = sel_cols[nonzero]
    sel_probs = sel_probs[nonzero]
    sel_suits = sel_suits[nonzero]
    
    print(f"  Valid embeddings: {len(embeddings)}")
    
    if len(embeddings) == 0:
        print("  ERROR: No valid embeddings extracted!")
        return
    
    # Convert pixel coords to lat/lon
    lats, lons = [], []
    for r, c in zip(sel_rows, sel_cols):
        lat, lon = pixel_to_latlon(r, c, bbox, grid_shape)
        lats.append(lat)
        lons.append(lon)
    
    # Build output DataFrame
    emb_cols = [f'emb_{i}' for i in range(64)]
    df = pd.DataFrame(embeddings, columns=emb_cols)
    df['lat'] = lats
    df['lon'] = lons
    df['raw_prob'] = sel_probs
    df['suitability'] = sel_suits
    df['px_row'] = sel_rows
    df['px_col'] = sel_cols
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"\n  Saved {len(df)} logged-area negatives to {output_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Total candidates (prob>={args.prob_threshold}, suit<={args.suit_threshold}): {candidate_count:,}")
    print(f"  Selected & valid: {len(df)}")
    print(f"  Mean raw probability: {df['raw_prob'].mean():.3f}")
    print(f"  Suitability breakdown:")
    for s in sorted(df['suitability'].unique()):
        n = (df['suitability'] == s).sum()
        print(f"    suitability={s:.2f}: {n} samples")
    
    print(f"\nTo use in training:")
    print(f"  python scripts/prediction/classify_tiled_gpu.py \\")
    print(f"    --gee-negatives {output_path} --gee-negatives-weight 2 \\")
    print(f"    --annotations data/processed/yew_annotations_combined.csv \\")
    print(f"    --annotation-weight 3")


if __name__ == '__main__':
    main()
