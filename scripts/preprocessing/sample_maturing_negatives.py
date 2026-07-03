#!/usr/bin/env python3
"""
Sample 5000 random negative points from VRI cat 4 (logged 40-80yr) and cat 5
(forest 80-150yr) pixels across all study tiles, using cached embeddings.

These represent the "false positive" risk in maturing second-growth — areas
where the model might misclassify as yew habitat but actually lack it due to
recent disturbance.

Run:
    /home/jericho/anaconda3/envs/yew_pytorch/bin/python scripts/preprocessing/sample_maturing_negatives.py
"""
import sys
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

TC = ROOT / "results/analysis/cwh_spot_comparisons/tile_cache"
TILES_JSON = ROOT / "docs/tiles/tiles.json"
OUT_EMB = ROOT / "data/processed/maturing_second_growth_negatives_embeddings.npy"
OUT_CSV = ROOT / "data/processed/maturing_second_growth_negatives.csv"
N_SAMPLES_PER_TILE = 100


def sample_tile_cat45(slug):
    """Sample up to N_SAMPLES_PER_TILE random pixels from cat 4+5 in this tile.

    Returns (pixel_indices, embeddings) tuple, or ([], []) if unavailable.
    """
    log_path = TC / f"{slug}_logging.npy"
    emb_path = TC / f"{slug}_emb.npy"

    if not log_path.exists() or not emb_path.exists():
        return [], np.array([])

    log = np.load(log_path)
    emb = np.load(emb_path)

    # Find cat 4+5 pixels
    cat45_idx = np.where((log == 4) | (log == 5))
    if len(cat45_idx[0]) == 0:
        return [], np.array([])

    # Sample up to N_SAMPLES_PER_TILE
    n = min(N_SAMPLES_PER_TILE, len(cat45_idx[0]))
    sample_idx = np.random.choice(len(cat45_idx[0]), size=n, replace=False)

    rows = cat45_idx[0][sample_idx]
    cols = cat45_idx[1][sample_idx]

    # Index embeddings at these pixel locations
    sampled_emb = emb[rows, cols, :]

    return list(zip(rows, cols)), sampled_emb


def main():
    np.random.seed(42)

    tiles = json.loads((TILES_JSON).read_text())

    print(f"Sampling ~{N_SAMPLES_PER_TILE} negatives per tile from cat 4&5 (maturing second-growth)")
    print(f"Using cached embeddings (no GEE download)\n")

    all_embeddings = []
    all_rows = []
    tile_count = 0

    for t in tiles:
        slug = t['slug']

        pixels, embs = sample_tile_cat45(slug)
        if len(pixels) == 0:
            continue

        all_embeddings.append(embs)

        for (r, c) in pixels:
            all_rows.append({
                'tile': slug,
                'row': r,
                'col': c,
                'source': 'maturing_second_growth_cat4_5',
                'label': 0,
            })

        tile_count += 1
        print(f"  [{tile_count:2d}] {slug:28s} {len(pixels)} samples")

    if all_embeddings:
        all_emb = np.vstack(all_embeddings).astype(np.float32)

        np.save(OUT_EMB, all_emb)
        print(f"\n✓ Saved {all_emb.shape} embeddings to {OUT_EMB.name}")

        import pandas as pd
        df = pd.DataFrame(all_rows)
        df.to_csv(OUT_CSV, index=False)
        print(f"✓ Saved {len(df)} metadata rows to {OUT_CSV.name}")
        print(f"\nTotal: {len(all_emb)} negatives sampled")
    else:
        print("ERROR: no embeddings extracted")


if __name__ == "__main__":
    main()
