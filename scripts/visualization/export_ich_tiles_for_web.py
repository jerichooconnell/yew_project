#!/usr/bin/env python3
"""
Regenerate the 14 ICH web tiles (ich_pt01-15) from their new-model grids and
APPEND them to docs/tiles/tiles.json (which export_tiles_for_web.py rewrites
with only the 85 coastal tiles).

ICH tiles are interior: no low-elevation ramp (no _elev.npy), not lower-mainland.
Pipeline = logging mask -> fire modifier, matching the coastal export otherwise.
Bbox / name / desc / lat / lon are read from the committed manifest via git
(export_tiles_for_web.py has already overwritten the on-disk tiles.json).

Run AFTER export_tiles_for_web.py:
    /home/jericho/anaconda3/envs/yew_pytorch/bin/python scripts/visualization/export_ich_tiles_for_web.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/visualization"))
import export_tiles_for_web as E  # noqa: E402

CACHE_DIR = ROOT / "results/analysis/cwh_spot_comparisons/tile_cache"
OUT_DIR = ROOT / "docs/tiles"
MANIFEST = OUT_DIR / "tiles.json"


def committed_ich_entries():
    """Read the ICH entries (bbox/name/desc/lat/lon) from the last-committed manifest."""
    raw = subprocess.check_output(
        ["git", "show", "HEAD:docs/tiles/tiles.json"], cwd=str(ROOT)
    )
    d = json.loads(raw)
    return [x for x in d if x["slug"].startswith("ich_pt")]


def main():
    fires_gdf = E.load_fire_gdf()

    manifest = json.loads(MANIFEST.read_text())
    manifest = [x for x in manifest if not x["slug"].startswith("ich_pt")]  # coastal only

    ich = committed_ich_entries()
    print(f"Regenerating {len(ich)} ICH tiles with new-model grids\n")

    done = 0
    for src in ich:
        slug = src["slug"]
        grid_path = CACHE_DIR / f"{slug}_grid.npy"
        log_path = CACHE_DIR / f"{slug}_logging.npy"
        if not grid_path.exists():
            print(f"  SKIP {slug}: grid missing")
            continue

        raw_grid = np.load(grid_path)
        H, W = raw_grid.shape
        b = src["bbox"]
        south, north, west, east = b["south"], b["north"], b["west"], b["east"]

        # Step 1: logging / water / alpine mask
        log_grid = None
        if log_path.exists():
            log_grid = np.load(log_path)
            grid = E.apply_logging_mask(raw_grid, log_grid)
        else:
            grid = raw_grid.copy().astype(np.float32)

        # Step 2: fire-date modifier (no elevation ramp for interior ICH)
        fire_modifier, fire_year_raster = E.make_fire_modifier(
            H, W, west, south, east, north, fires_gdf
        )
        grid = (grid * fire_modifier).astype(np.float32)
        grid = np.clip(grid, 0.0, 1.0)

        stats = {
            "mean": float(np.mean(grid)),
            "median": float(np.median(grid)),
            "max": float(np.max(grid)),
            "p30_ha": float(np.sum(grid >= 0.30) * 10 * 10 / 1e4),
            "p50_ha": float(np.sum(grid >= 0.50) * 10 * 10 / 1e4),
            "p70_ha": float(np.sum(grid >= 0.70) * 10 * 10 / 1e4),
            "h": H,
            "w": W,
        }

        png_path = OUT_DIR / f"{slug}.png"
        E.grid_to_png(grid, E.YEWCMAP, png_path)

        entry = {
            "slug": slug,
            "name": src["name"],
            "desc": src["desc"],
            "lat": src["lat"],
            "lon": src["lon"],
            "bbox": src["bbox"],
            "stats": stats,
            "png": f"{slug}.png",
        }

        if log_grid is not None:
            log_png_path = OUT_DIR / f"{slug}_logging.png"
            E.logging_to_png(log_grid, log_png_path)
            entry["logging_png"] = f"{slug}_logging.png"
            total_px = log_grid.size
            entry["logging_stats"] = {
                "water_pct": round(float(np.sum(log_grid == 1)) / total_px * 100, 1),
                "logged_lt20_pct": round(float(np.sum(log_grid == 2)) / total_px * 100, 1),
                "logged_20_40_pct": round(float(np.sum(log_grid == 3)) / total_px * 100, 1),
                "logged_40_80_pct": round(float(np.sum(log_grid == 4)) / total_px * 100, 1),
                "forest_80_150_pct": round(float(np.sum(log_grid == 5)) / total_px * 100, 1),
                "alpine_pct": round(float(np.sum(log_grid == 6)) / total_px * 100, 1),
                "oldgrowth_pct": round(float(np.sum(log_grid == 7)) / total_px * 100, 1),
            }

        manifest.append(entry)
        done += 1
        print(f"  ✓ {slug}: {H}×{W}  P≥0.5={stats['p50_ha']:.0f} ha  mean={stats['mean']:.4f}")

    MANIFEST.write_text(json.dumps(manifest, indent=2))
    coastal = len([x for x in manifest if not x["slug"].startswith("ich_pt")])
    ich_n = len(manifest) - coastal
    print(f"\nManifest rewritten: {len(manifest)} tiles ({coastal} coastal + {ich_n} ICH)")


if __name__ == "__main__":
    main()
