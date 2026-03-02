#!/usr/bin/env python3
"""
Fast parallel extraction of Google Satellite Embedding patches for iNat locations.
Uses ThreadPoolExecutor for concurrent EE requests.

Usage:
    conda run -n yew_pytorch python scripts/preprocessing/extract_inat_embeddings_fast.py
"""

import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import ee
import numpy as np
import requests

# ── Config ──────────────────────────────────────────────────────────────────
INPUT_CSV = "data/processed/inat_need_embeddings.csv"
OUTPUT_DIR = Path("data/ee_imagery/embedding_patches_64x64")
YEAR = 2022          # AlphaEarth annual embedding year
PATCH_SIZE = 64
SCALE = 10           # metres per pixel
MAX_WORKERS = 8      # parallel EE requests
TIMEOUT = 60         # seconds per download
RETRY = 2            # retries on failure

BANDS = [f"A{i:02d}" for i in range(64)]


# ── Extraction ──────────────────────────────────────────────────────────────

def extract_one(lat: float, lon: float) -> np.ndarray | None:
    """Download 64×64×64 embedding patch; returns array or None."""
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer((PATCH_SIZE * SCALE) / 2).bounds()
    image = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
             .filterDate(f"{YEAR}-01-01", f"{YEAR + 1}-01-01")
             .filterBounds(point)
             .first()
             .select(BANDS))

    url = image.getDownloadURL({
        "region": region,
        "dimensions": [PATCH_SIZE, PATCH_SIZE],
        "format": "NPY",
    })
    resp = requests.get(url, timeout=TIMEOUT)
    if resp.status_code != 200:
        return None

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        tmp.write(resp.content)
        tmp_path = tmp.name
    data = np.load(tmp_path)
    Path(tmp_path).unlink()

    if data.dtype.names is not None:
        data = np.stack([data[b] for b in BANDS], axis=0)
    elif data.ndim == 3:
        data = np.transpose(data, (2, 0, 1))

    if data.shape != (64, PATCH_SIZE, PATCH_SIZE):
        return None
    return data.astype(np.float32)


def process_row(row: dict) -> str:
    """Extract embedding for one row; returns status string."""
    lat = float(row["latitude"])
    lon = float(row["longitude"])
    fname = f"embedding_{lat:.6f}_{lon:.6f}.npy"
    out_path = OUTPUT_DIR / fname

    if out_path.exists():
        return "skip"

    for attempt in range(1, RETRY + 1):
        try:
            arr = extract_one(lat, lon)
            if arr is not None:
                np.save(out_path, arr)
                return "ok"
        except Exception as e:
            if attempt == RETRY:
                return f"fail:{e}"
            time.sleep(2 * attempt)
    return "fail"


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ee.Initialize(project="carbon-storm-206002")
    print("✓ Earth Engine initialized")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(INPUT_CSV) as f:
        rows = list(csv.DictReader(f))
    print(f"Input: {len(rows)} locations from {INPUT_CSV}")

    ok = skip = fail = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(process_row, r): r for r in rows}
        for i, fut in enumerate(as_completed(futures), 1):
            status = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1

            if i % 50 == 0 or i == len(rows):
                elapsed = time.time() - t0
                rate = (ok + skip) / elapsed if elapsed > 0 else 0
                remaining = (len(rows) - i) / rate / 60 if rate > 0 else 0
                print(f"  [{i}/{len(rows)}]  ok={ok}  skip={skip}  fail={fail}  "
                      f"{rate:.1f}/s  ~{remaining:.0f} min left")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"DONE in {elapsed / 60:.1f} min")
    print(f"  Extracted: {ok}")
    print(f"  Skipped:   {skip}")
    print(f"  Failed:    {fail}")
    print(f"{'=' * 60}")

    # Save log
    meta = {
        "timestamp": datetime.now().isoformat(),
        "input": INPUT_CSV,
        "year": YEAR,
        "extracted": ok,
        "skipped": skip,
        "failed": fail,
        "elapsed_sec": round(elapsed, 1),
    }
    log_path = OUTPUT_DIR / "inat_extraction_log.json"
    with open(log_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
