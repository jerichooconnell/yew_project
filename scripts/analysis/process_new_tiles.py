#!/usr/bin/env python3
"""
Process newly-downloaded tile embeddings into the cache files that
yew_logging_impact_by_bec.py consumes:
  {slug}_grid.npy     — raw XGBoost yew probability (classify the embedding)
  {slug}_logging.npy  — VRI logging category (local VEG_COMP GDB, no GEE)

Elevation ({slug}_elev.npy) is handled separately by cache_elevation_grids.py.

Only processes study tiles that have an embedding but no grid yet.

Run:
    /home/jericho/anaconda3/envs/yew_pytorch/bin/python scripts/analysis/process_new_tiles.py
"""
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path("/home/jericho/yew_project")
sys.path.insert(0, str(ROOT / "scripts/analysis"))
sys.path.insert(0, str(ROOT / "scripts/prediction"))
import yew_logging_impact_by_bec as Y     # noqa: E402  (slugify, centre_to_bbox, STUDY_AREAS)
import classify_cwh_spots as C            # noqa: E402  (extract_logging_grid)

TC    = ROOT / "results/analysis/cwh_spot_comparisons/tile_cache"
MODEL = ROOT / "results/predictions/south_vi_large/xgb_raw_model_expanded.json"


def main():
    sa = {Y.slugify(nm): (lat, lon) for lat, lon, nm, desc in Y.STUDY_AREAS}
    emb = {p.name[:-8] for p in TC.glob("*_emb.npy")}
    grid = {p.name[:-9] for p in TC.glob("*_grid.npy")}
    todo = sorted(s for s in (emb - grid) if s in sa)      # study tiles needing a grid
    print(f"{len(todo)} tiles to process (have embedding, in STUDY_AREAS, no grid)\n")

    bst = xgb.Booster(); bst.load_model(str(MODEL))
    done = fail = 0
    for i, slug in enumerate(todo, 1):
        lat, lon = sa[slug]
        try:
            e = np.load(TC / f"{slug}_emb.npy")
            H, W, _ = e.shape

            gpath = TC / f"{slug}_grid.npy"
            if not gpath.exists():
                pred = bst.predict(xgb.DMatrix(e.reshape(-1, 64))).reshape(H, W).astype(np.float32)
                np.save(gpath, pred)
            else:
                pred = np.load(gpath)

            lpath = TC / f"{slug}_logging.npy"
            if not lpath.exists():
                bbox = Y.centre_to_bbox(lat, lon)          # (south, north, west, east)
                log = C.extract_logging_grid(bbox, H, W)
                if log is None:
                    print(f"  [{i:2d}/{len(todo)}] {slug:28s} ⚠ logging=None (VEG_COMP miss)")
                    fail += 1
                    continue
                np.save(lpath, log)
            else:
                log = np.load(lpath)

            og = int((log == 7).sum())
            print(f"  [{i:2d}/{len(todo)}] {slug:28s} {H}×{W}  "
                  f"grid μ={pred.mean():.3f}  OG={og/(H*W)*100:4.1f}%")
            done += 1
        except Exception as ex:
            print(f"  [{i:2d}/{len(todo)}] {slug:28s} ✗ {repr(ex)[:80]}")
            fail += 1

    print(f"\nDone: {done} processed, {fail} failed.")
    print("Next: cache_elevation_grids.py for elevation, then re-run yew_logging_impact_by_bec.py")


if __name__ == "__main__":
    main()
