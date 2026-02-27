#!/usr/bin/env python3
"""
Extract GEE embedding vectors from alpine / barren pixels in the CWH spot
comparison tile caches and add them as negatives to train/val splits.

Runs from the yew_project root:
    conda run -n yew_pytorch python scripts/preprocessing/extract_alpine_negatives.py

Output:
    data/processed/alpine_negatives/alpine_negative_embeddings.csv
    (columns: emb_0 … emb_63, has_yew=0)
"""

import os
import random
from math import radians, cos
from pathlib import Path

import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
CACHE_DIR   = Path('results/analysis/cwh_spot_comparisons/tile_cache')
OUT_DIR     = Path('data/processed/alpine_negatives')
OUT_CSV     = OUT_DIR / 'alpine_negative_embeddings.csv'

# Sample this many alpine negatives per area (more from high-confusion areas)
SAMPLE_PLAN = {
    'garibaldi_foothills':   600,
    'desolation_sound':      600,
    'bella_coola_valley':    400,
    'prince_rupert_hills':   300,
    'bute_inlet_slopes':     300,
    'squamish_highlands':    200,
    'kitimat_ranges':        200,
    'strathcona_highlands':  100,
    'campbell_river_uplands': 100,
}

# Prefer pixels that currently have high predicted probability (hardest negatives)
HIGH_PROB_THRESH = 0.20    # sample 80% from pixels above this threshold
HIGH_PROB_FRACTION = 0.80

random.seed(42)
np.random.seed(42)


def sample_alpine_embeddings(slug, n_sample):
    emb_path = CACHE_DIR / f'{slug}_emb.npy'
    log_path = CACHE_DIR / f'{slug}_logging.npy'
    grid_path = CACHE_DIR / f'{slug}_grid.npy'

    if not emb_path.exists():
        print(f"  [{slug}] ✗ embedding not found — skip")
        return None
    if not log_path.exists():
        print(f"  [{slug}] ✗ logging raster not found — skip")
        return None

    emb  = np.load(str(emb_path))   # H×W×64
    log  = np.load(str(log_path))   # H×W uint8
    grid = np.load(str(grid_path)) if grid_path.exists() else None

    # Cat 6 = alpine/barren
    alpine_mask = (log == 6)
    if alpine_mask.sum() == 0:
        print(f"  [{slug}] no cat-6 pixels — skip")
        return None

    # Split into hard (high predicted prob) and easy subsets
    n_hard = min(int(n_sample * HIGH_PROB_FRACTION), n_sample)
    n_easy = n_sample - n_hard

    rows_idx, cols_idx = np.where(alpine_mask)

    if grid is not None:
        probs = grid[rows_idx, cols_idx]
        hard_sel = probs >= HIGH_PROB_THRESH
        hard_r, hard_c = rows_idx[hard_sel], cols_idx[hard_sel]
        easy_r, easy_c = rows_idx[~hard_sel], cols_idx[~hard_sel]
    else:
        hard_r, hard_c = rows_idx, cols_idx
        easy_r, easy_c = np.array([]), np.array([])
        n_hard = n_sample
        n_easy = 0

    chosen_r, chosen_c = [], []

    if len(hard_r) > 0:
        n_h = min(n_hard, len(hard_r))
        idx = np.random.choice(len(hard_r), n_h, replace=False)
        chosen_r.extend(hard_r[idx].tolist())
        chosen_c.extend(hard_c[idx].tolist())

    if len(easy_r) > 0 and n_easy > 0:
        n_e = min(n_easy, len(easy_r))
        idx = np.random.choice(len(easy_r), n_e, replace=False)
        chosen_r.extend(easy_r[idx].tolist())
        chosen_c.extend(easy_c[idx].tolist())

    if not chosen_r:
        print(f"  [{slug}] no samples collected — skip")
        return None

    chosen_r = np.array(chosen_r)
    chosen_c = np.array(chosen_c)

    # Extract 64-d embedding vectors
    vecs = emb[chosen_r, chosen_c, :]   # N×64

    # Filter out all-zero rows (GEE nodata)
    valid = np.any(vecs != 0, axis=1)
    vecs = vecs[valid]

    print(f"  [{slug}] {vecs.shape[0]} alpine negatives "
          f"(hard≥{HIGH_PROB_THRESH}: {int(valid.sum() - max(0, valid.sum()-n_hard))} / "
          f"total alpine px: {alpine_mask.sum():,})")
    return vecs


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_vecs = []

    print("Extracting alpine negative embeddings from spot tile caches…")
    for slug, n in SAMPLE_PLAN.items():
        vecs = sample_alpine_embeddings(slug, n)
        if vecs is not None:
            all_vecs.append(vecs)

    if not all_vecs:
        print("ERROR: no embeddings collected — check cache paths")
        return

    combined = np.vstack(all_vecs)
    print(f"\nTotal alpine negatives: {len(combined):,}")

    cols = [f'emb_{i}' for i in range(64)]
    df = pd.DataFrame(combined, columns=cols)
    df['has_yew'] = 0
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved → {OUT_CSV}")


if __name__ == '__main__':
    main()
