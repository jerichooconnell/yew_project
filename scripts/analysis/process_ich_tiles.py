#!/usr/bin/env python3
"""
Download + process the 14 ICH study tiles (ich_pt01–15) into the cache files
yew_logging_impact_by_bec.py consumes: {slug}_emb / _grid / _logging .npy.

Embeddings are pulled from GEE in PARALLEL (ThreadPool over tiles) — the
speed-up discussed for the re-pull. Classification + VRI logging are then done
locally (serial; VEG_COMP GDB reads are not thread-safe). Elevation is skipped:
ICH tiles are interior and lie well above the 30 m coastal suppression ramp, so
their elevation factor is 1 (yew_logging_impact handles missing _elev.npy).

Run:
    /home/jericho/anaconda3/envs/yew_pytorch/bin/python scripts/analysis/process_ich_tiles.py
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import xgboost as xgb
import ee

ROOT = Path("/home/jericho/yew_project")
sys.path.insert(0, str(ROOT / "scripts/prediction"))
import classify_cwh_spots as C          # noqa: E402  download_embeddings_chunked, extract_logging_grid

TC    = ROOT / "results/analysis/cwh_spot_comparisons/tile_cache"
MODEL = ROOT / "results/predictions/south_vi_large/xgb_raw_model_expanded.json"
YEAR  = 2024
WORKERS = 5


def download_one(t):
    slug = t["slug"]
    cache = TC / f"{slug}_emb.npy"
    if cache.exists():
        return slug, "cached"
    b = t["bbox"]
    region = ee.Geometry.Rectangle([b["west"], b["south"], b["east"], b["north"]])
    try:
        emb = C.download_embeddings_chunked(region, YEAR, 10, cache)
        return slug, f"{emb.shape}"
    except Exception as e:
        return slug, f"FAIL {repr(e)[:80]}"


def main():
    ee.Initialize(project="carbon-storm-206002")
    tiles = [t for t in json.load(open(ROOT / "docs/tiles/tiles.json"))
             if t["slug"].startswith("ich_pt")]
    print(f"{len(tiles)} ICH tiles — downloading embeddings ({WORKERS} parallel)\n")

    # ── parallel embedding download ───────────────────────────────────────────
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(download_one, t): t for t in tiles}
        for f in as_completed(futs):
            slug, status = f.result()
            print(f"  emb {slug}: {status}")

    # ── classify + logging (serial) ───────────────────────────────────────────
    print("\nClassifying + extracting logging...")
    bst = xgb.Booster(); bst.load_model(str(MODEL))
    ok = 0
    for t in tiles:
        slug = t["slug"]
        epath = TC / f"{slug}_emb.npy"
        if not epath.exists():
            print(f"  {slug}: no embedding — skip")
            continue
        e = np.load(epath); H, W, _ = e.shape

        gpath = TC / f"{slug}_grid.npy"
        if not gpath.exists():
            pred = bst.predict(xgb.DMatrix(e.reshape(-1, 64))).reshape(H, W).astype(np.float32)
            np.save(gpath, pred)
        else:
            pred = np.load(gpath)

        lpath = TC / f"{slug}_logging.npy"
        if not lpath.exists():
            b = t["bbox"]
            log = C.extract_logging_grid((b["south"], b["north"], b["west"], b["east"]), H, W)
            if log is None:
                print(f"  {slug}: logging=None (VEG_COMP miss) — skip")
                continue
            np.save(lpath, log)
        else:
            log = np.load(lpath)

        print(f"  {slug}: {H}×{W}  grid μ={pred.mean():.3f}  OG={(log==7).mean()*100:4.1f}%")
        ok += 1

    print(f"\nDone: {ok}/{len(tiles)} ICH tiles ready.")


if __name__ == "__main__":
    main()
