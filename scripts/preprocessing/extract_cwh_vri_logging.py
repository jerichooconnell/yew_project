#!/usr/bin/env python3
"""
Extract and pre-classify the BC VRI logging categories for the entire CWH+CDF
land area and save as a compact GeoPackage for fast spatial joins.

Input  : data/VEG_COMP_LYR_R1_POLY_2024.gdb  (5.7 GB, 6.87 M features)
         data/processed/cwh_cdf_land.shp       (695 polygons, EPSG:3005)
Output : data/processed/cwh_vri_logging.gpkg   (~200-400 MB estimated)

The output has two columns beyond geometry:
  log_cat   uint8  — 1-6 integer category (same scheme as *_logging.npy tiles)
  log_label str    — human-readable label

Category definitions (matching classify_cwh_spots.py):
  1  water / non-forest
  2  logged  <20 yr
  3  logged  20-40 yr
  4  logged  40-80 yr
  5  forest  >80 yr / unlogged
  6  alpine / barren

Usage:
    conda run -n yew_pytorch python scripts/preprocessing/extract_cwh_vri_logging.py
    conda run -n yew_pytorch python scripts/preprocessing/extract_cwh_vri_logging.py --no-clip

Timing (benchmarked on this machine):
  GDB bbox read   : ~4-6 min  (reads ~1.5 M polygons inside CWH bbox)
  Classification  : <1 min
  Clip to CWH land: ~3 min
  Write gpkg      : ~1 min
  Total           : ~9-11 min  (one-time cost)

  Subsequent spatial joins against the output: ~50 s for 100 000 points.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np

ROOT      = Path(__file__).resolve().parents[2]
VRI_GDB   = ROOT / "data" / "VEG_COMP_LYR_R1_POLY_2024.gdb"
CWH_SHP   = ROOT / "data" / "processed" / "cwh_cdf_land.shp"
OUT_GPKG  = ROOT / "data" / "processed" / "cwh_vri_logging.gpkg"

VRI_LAYER = "VEG_COMP_LYR_R1_POLY"
VRI_COLS  = [
    "BCLCS_LEVEL_1", "BCLCS_LEVEL_2",
    "PROJ_AGE_1", "PROJ_AGE_CLASS_CD_1",
    "HARVEST_DATE", "LINE_7B_DISTURBANCE_HISTORY",
    "OPENING_IND", "OPENING_SOURCE", "ALPINE_DESIGNATION",
]

LOG_LABELS = {
    1: "water/non-forest",
    2: "logged <20yr",
    3: "logged 20-40yr",
    4: "logged 40-80yr",
    5: "forest >80yr",
    6: "alpine/barren",
}

# ── Age-class midpoint table (BC VRI standard) ────────────────────────────
_AGE_CLASS_MIDPOINT = {
    "1": 10, "2": 30, "3": 50, "4": 70,
    "5": 90, "6": 110, "7": 130, "8": 195, "9": 300,
}

_CUR_YEAR_2D  = datetime.now().year % 100
_CUR_CENTURY  = (datetime.now().year // 100) * 100


def _decode_7b_year(code_2d: str) -> int:
    y = int(code_2d)
    return _CUR_CENTURY + y if y <= _CUR_YEAR_2D else (_CUR_CENTURY - 100) + y


def _parse_7b_min_age(field_val, current_year: int):
    """Return age of most recent stand-resetting event from LINE_7B, or None."""
    if not field_val:
        return None
    min_age = None
    for part in str(field_val).split(";"):
        part = part.strip()
        if len(part) >= 3 and part[0].isalpha() and part[1:3].isdigit():
            try:
                age = current_year - _decode_7b_year(part[1:3])
                if age >= 0:
                    min_age = age if min_age is None else min(min_age, age)
            except Exception:
                pass
    return min_age


def classify_vri_row(row, current_year: int) -> int:
    """Return 1-6 log_cat for a single VRI polygon row (Series or dict-like)."""
    bclcs1 = str(row.get("BCLCS_LEVEL_1") or "").strip()
    bclcs2 = str(row.get("BCLCS_LEVEL_2") or "").strip()

    if bclcs1 == "W" or (bclcs1 == "N" and bclcs2 == "W"):
        return 1
    if bclcs1 == "N" and bclcs2 == "L":
        return 6
    if str(row.get("ALPINE_DESIGNATION") or "").strip() == "A":
        return 6

    ages = []

    pa1 = row.get("PROJ_AGE_1")
    if pa1 is not None:
        try:
            ages.append(int(pa1))
        except (ValueError, TypeError):
            pass

    pac = str(row.get("PROJ_AGE_CLASS_CD_1") or "").strip()
    if pac in _AGE_CLASS_MIDPOINT:
        ages.append(_AGE_CLASS_MIDPOINT[pac])

    dist7b = _parse_7b_min_age(row.get("LINE_7B_DISTURBANCE_HISTORY"), current_year)
    if dist7b is not None:
        ages.append(dist7b)

    hdate = row.get("HARVEST_DATE")
    if hdate:
        try:
            hy = hdate.year if hasattr(hdate, "year") else int(str(hdate)[:4])
            ages.append(current_year - hy)
        except Exception:
            pass

    if str(row.get("OPENING_IND") or "").strip() == "Y" and not ages:
        ages.append(0)

    opening_src = row.get("OPENING_SOURCE")
    try:
        osrc = int(opening_src) if opening_src is not None else None
    except (ValueError, TypeError):
        osrc = None
    if osrc in {3, 4, 7, 11} and not ages:
        ages.append(0)

    if not ages:
        if bclcs2 == "N":
            return 6
        return 5

    age = min(ages)
    if age < 20:
        return 2
    if age < 40:
        return 3
    if age < 80:
        return 4
    return 5


def classify_vri_array(gdf: gpd.GeoDataFrame, current_year: int) -> np.ndarray:
    """Vectorised-style classification — uses itertuples for speed vs iterrows."""
    cats = np.zeros(len(gdf), dtype=np.uint8)
    for i, row in enumerate(gdf.itertuples(index=False)):
        cats[i] = classify_vri_row(row._asdict(), current_year)
    return cats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-clip", action="store_true",
                        help="Skip clipping to cwh_cdf_land.shp (faster but larger file)")
    parser.add_argument("--out", default=str(OUT_GPKG),
                        help=f"Output GeoPackage path (default: {OUT_GPKG})")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    current_year = datetime.now().year

    # ── 1. Load CWH boundary to get bbox ──────────────────────────────────
    print(f"Loading CWH+CDF boundary: {CWH_SHP.name}")
    t0 = time.time()
    cwh = gpd.read_file(CWH_SHP)   # EPSG:3005
    xmin, ymin, xmax, ymax = cwh.total_bounds
    print(f"  {len(cwh)} polygons  EPSG:3005 bbox: "
          f"x=[{xmin:.0f}, {xmax:.0f}]  y=[{ymin:.0f}, {ymax:.0f}]")
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # ── 2. Read VRI GDB clipped to CWH bbox ───────────────────────────────
    print(f"\nReading VRI GDB (bbox filter) …")
    print(f"  Source: {VRI_GDB}")
    print(f"  Keeping columns: {VRI_COLS}")
    t0 = time.time()
    vri = gpd.read_file(
        str(VRI_GDB),
        layer=VRI_LAYER,
        bbox=(xmin, ymin, xmax, ymax),   # GDB is in EPSG:3005 — no transform needed
        columns=VRI_COLS,                 # pyogrio supports column selection
    )
    t_read = time.time() - t0
    print(f"  Read {len(vri):,} polygons in {t_read:.1f}s  ({t_read/60:.1f} min)")
    print(f"  CRS: {vri.crs}")

    # ── 3. Classify each polygon ───────────────────────────────────────────
    print(f"\nClassifying {len(vri):,} polygons …")
    t0 = time.time()
    cats = classify_vri_array(vri, current_year)
    vri["log_cat"]   = cats
    vri["log_label"] = [LOG_LABELS[c] for c in cats]
    t_cls = time.time() - t0
    print(f"  Done in {t_cls:.1f}s")

    # Summary
    from collections import Counter
    cnts = Counter(int(c) for c in cats)
    total = len(cats)
    for cat in sorted(cnts):
        pct = 100 * cnts[cat] / total
        print(f"    cat {cat} ({LOG_LABELS[cat]}): {cnts[cat]:,}  ({pct:.1f}%)")

    # ── 4. Keep only geometry + log_cat + log_label ────────────────────────
    vri = vri[["log_cat", "log_label", "geometry"]].copy()

    # ── 5. Optionally clip to CWH+CDF land polygon ────────────────────────
    if not args.no_clip:
        print(f"\nClipping to CWH+CDF land boundary …")
        t0 = time.time()
        # Dissolve CWH to a single multipolygon for fast clip
        cwh_union = cwh.dissolve()
        vri = gpd.clip(vri, cwh_union, keep_geom_type=True)
        print(f"  Clipped to {len(vri):,} polygons in {time.time()-t0:.1f}s")
    else:
        print("\n--no-clip: skipping land boundary clip")

    # ── 6. Write output ────────────────────────────────────────────────────
    print(f"\nWriting output: {out_path}")
    t0 = time.time()
    vri.to_file(str(out_path), driver="GPKG", layer="cwh_vri_logging")
    print(f"  Written in {time.time()-t0:.1f}s")
    print(f"  File size: {out_path.stat().st_size / 1e6:.1f} MB")
    print(f"  Features:  {len(vri):,}")
    print(f"\n✓ Done. Use this file for fast spatial joins:")
    print(f"    vri = gpd.read_file('{out_path}')  # ~8s")
    print(f"    joined = gpd.sjoin(pts_3005, vri[['log_cat','geometry']], how='left', predicate='within')")


if __name__ == "__main__":
    main()
