#!/usr/bin/env python3
"""
Reclassify all cached tile embeddings using the new retrained XGBoost model
(xgb_raw_model_expanded_v2_cat45neg.json) to update the grid.npy files.

This generates new probability grids reflecting the more conservative
model trained with maturing-second-growth negatives (cat 4&5).

Run:
    /home/jericho/anaconda3/envs/yew_pytorch/bin/python scripts/analysis/reclassify_with_new_model.py
"""
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
TC = ROOT / "results/analysis/cwh_spot_comparisons/tile_cache"
NEW_MODEL = ROOT / "results/predictions/south_vi_large/xgb_raw_model_expanded_v2_cat45neg.json"
BACKUP_SUFFIX = "_old_model"


def main():
    if not NEW_MODEL.exists():
        print(f"ERROR: New model not found: {NEW_MODEL}")
        return

    print("=" * 70)
    print("RECLASSIFY: Tile grids with new retrained model")
    print("=" * 70)

    bst = xgb.Booster()
    bst.load_model(str(NEW_MODEL))
    print(f"\n✓ Loaded new model: {NEW_MODEL.name}\n")

    # Find all tiles with embeddings
    emb_files = sorted(TC.glob("*_emb.npy"))
    print(f"Found {len(emb_files)} tiles with embeddings\n")

    done = fail = 0
    for i, emb_path in enumerate(emb_files, 1):
        slug = emb_path.name[:-8]  # remove _emb.npy
        grid_path = TC / f"{slug}_grid.npy"

        try:
            emb = np.load(emb_path)
            H, W, _ = emb.shape

            # Backup old grid if it exists
            if grid_path.exists():
                backup_path = TC / f"{slug}_grid{BACKUP_SUFFIX}.npy"
                if not backup_path.exists():
                    np.save(backup_path, np.load(grid_path))

            # Classify with new model
            pred = bst.predict(xgb.DMatrix(emb.reshape(-1, 64))).reshape(H, W)
            pred = pred.astype(np.float32)

            # Save new grid
            np.save(grid_path, pred)

            print(f"  [{i:3d}/{len(emb_files)}] {slug:30s} {H}×{W}  "
                  f"mean={pred.mean():.4f}  max={pred.max():.4f}")
            done += 1

        except Exception as e:
            print(f"  [{i:3d}/{len(emb_files)}] {slug:30s} FAIL {repr(e)[:60]}")
            fail += 1

    print(f"\n" + "=" * 70)
    print(f"RECLASSIFICATION COMPLETE")
    print(f"  Done: {done} tiles")
    print(f"  Failed: {fail} tiles")
    print(f"  Backups saved with suffix: {BACKUP_SUFFIX}.npy")
    print(f"  Ready to re-run: yew_logging_impact_by_bec.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
