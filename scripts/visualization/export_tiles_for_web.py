#!/usr/bin/env python3
"""
Export CWH spot comparison tile grids as web-ready PNG overlays
for deployment on GitHub Pages.

Reads each tile's *_grid.npy from the tile cache, applies the yew
probability colormap, and writes transparent PNG overlays plus a
tiles.json manifest with bounding boxes and statistics.

Usage:
    conda run -n yew_pytorch python scripts/visualization/export_tiles_for_web.py

Output:
    docs/tiles/<slug>.png          — transparent overlay images
    docs/tiles/tiles.json          — manifest with name, bbox, stats
"""

import json
import os
import sys
from math import cos, radians
from pathlib import Path

import numpy as np
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap

# ── Config ────────────────────────────────────────────────────────────────────
CACHE_DIR = Path('results/analysis/cwh_spot_comparisons/tile_cache')
OUT_DIR   = Path('docs/tiles')

AREA_KM = 10
YEW_TRANSPARENT_BELOW = 0.02

# Study areas (must match classify_cwh_spots.py)
STUDY_AREAS = [
    (48.440, -124.160, "Carmanah-Walbran",      "South VI old-growth CWH"),
    (48.600, -123.800, "Sooke Hills",            "South VI montane CWH"),
    (49.315, -124.980, "Clayoquot Sound",        "West central VI CWH"),
    (50.020, -125.240, "Campbell River Uplands", "North-central VI CWH"),
    (50.700, -127.100, "Quatsino Sound",         "Northern VI CWH"),
    (49.700, -123.150, "Squamish Highlands",     "Coast Mountains south CWH"),
    (50.720, -124.000, "Desolation Sound",       "Sunshine Coast north CWH"),
    (52.330, -126.600, "Bella Coola Valley",     "Central coast CWH"),
    (54.150, -129.700, "Prince Rupert Hills",    "North coast CWH"),
    (53.500, -128.600, "Kitimat Ranges",         "Skeena CWH fringe"),
    (49.900, -125.550, "Strathcona Highlands",   "Central VI CWH/MH boundary"),
    (49.860, -122.680, "Garibaldi Foothills",    "Mainland coast CWH near Whistler"),
    (50.830, -124.920, "Bute Inlet Slopes",      "Deep fjord CWH, Coast Mountains"),
    (49.020, -124.200, "Nanaimo Lakes",          "South VI mid-elevation CWH"),
    (51.400, -127.700, "Rivers Inlet",           "Central BC outer coast CWH"),
    (48.550, -124.420, "Port Renfrew",           "SW VI Pacific Rim old-growth CWH"),
    (48.820, -124.050, "Cowichan Uplands",       "South VI lower-elevation CWH"),
    (49.620, -125.100, "Comox Uplands",          "Central VI mid-elevation CWH"),
    (49.780, -126.020, "Gold River Forest",      "West-central VI inner CWH"),
    (50.720, -127.500, "Port Hardy Forest",      "North VI CWH valley bottoms"),
    (49.400, -123.720, "Sunshine Coast South",   "Lower Sunshine Coast CWH"),
    (49.520, -123.420, "Howe Sound East",        "Howe Sound montane CWH"),
    (49.780, -124.550, "Powell River Forest",    "Upper Sunshine Coast CWH"),
    (50.100, -124.060, "Jervis Inlet Slopes",    "Jervis Inlet fjord CWH"),
    (51.020, -124.480, "Toba Inlet Slopes",      "Remote fjord CWH, northern Sunshine Coast"),
    (51.080, -125.680, "Knight Inlet",           "Deep fjord, mainland Coast Mountains CWH"),
    (50.760, -126.480, "Broughton Archipelago",  "Outer archipelago CWH"),
    (51.220, -126.020, "Kingcome Inlet",         "Remote fjord valley, inner CWH"),
    (51.640, -126.520, "Owikeno Lake",           "Rivers Inlet drainage, interior CWH"),
    (52.090, -126.840, "Burke Channel",          "Outer fjord near Bella Coola"),
    (52.380, -127.680, "Ocean Falls",            "Outer coast CWH, high precipitation"),
    (52.720, -126.560, "Dean Channel",           "Dean River CWH, coast-interior edge"),
    (52.900, -128.700, "Princess Royal Island",  "Outer island CWH, spirit bear range"),
    (52.510, -128.580, "Milbanke Sound",         "Outer mid-coast CWH"),
    (54.820, -130.120, "Portland Inlet",         "Far north coast CWH near Nisga'a"),
]


def _make_yewcmap():
    return LinearSegmentedColormap.from_list(
        'yew',
        [
            (0.00, (0.20, 0.70, 0.20, 0.70)),
            (0.17, (0.45, 0.85, 0.05, 0.80)),
            (0.33, (1.00, 0.90, 0.00, 0.88)),
            (0.50, (1.00, 0.60, 0.00, 0.90)),
            (0.67, (0.90, 0.40, 0.10, 0.93)),
            (0.83, (0.80, 0.15, 0.30, 0.95)),
            (1.00, (0.65, 0.00, 0.45, 0.96)),
        ],
        N=256,
    )


YEWCMAP = _make_yewcmap()

# Logging-category RGBA colours (matching classify_cwh_spots.py)
# cat 0 = no data (transparent), 1 = water/non-forest, 2 = logged <20 yr,
# 3 = logged 20–40 yr, 4 = logged 40–80 yr, 5 = forest >80 yr,
# 6 = alpine / barren
LOG_RGBA = {
    1: (30,  100, 220, 180),   # water / non-forest
    2: (220, 50,  50,  170),   # logged  <20 yr
    3: (230, 120, 30,  150),   # logged 20–40 yr
    4: (220, 200, 50,  110),   # logged 40–80 yr
    5: (100, 200, 100,  70),   # forest  >80 yr / unlogged
    6: (175, 155, 125, 160),   # alpine / barren
}


def centre_to_bbox(lat, lon, km=10):
    half_lat = (km * 1000 / 2) / 111320.0
    half_lon = (km * 1000 / 2) / (111320.0 * cos(radians(lat)))
    return lat - half_lat, lat + half_lat, lon - half_lon, lon + half_lon


def slugify(name):
    return name.lower().replace(' ', '_').replace('-', '_').replace('/', '_')


def grid_to_png(grid, cmap, out_path):
    """Convert probability grid to transparent PNG overlay."""
    normed = np.clip(grid, 0.0, 1.0)
    rgba = cmap(normed)
    # Make near-zero pixels fully transparent
    rgba[grid < YEW_TRANSPARENT_BELOW, 3] = 0.0
    img = Image.fromarray((rgba * 255).astype(np.uint8), mode='RGBA')
    img.save(str(out_path), format='PNG', optimize=True)
    return os.path.getsize(out_path)


def logging_to_png(log_grid, out_path):
    """Convert uint8 logging category raster to transparent RGBA PNG."""
    h, w = log_grid.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for cat, color in LOG_RGBA.items():
        mask = log_grid == cat
        rgba[mask] = color
    img = Image.fromarray(rgba, 'RGBA')
    img.save(str(out_path), format='PNG', optimize=True)
    return os.path.getsize(out_path)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    total_bytes = 0
    total_log_bytes = 0

    print("=" * 60)
    print("EXPORT TILE OVERLAYS FOR GITHUB PAGES")
    print("=" * 60)

    for lat, lon, name, desc in STUDY_AREAS:
        slug = slugify(name)
        grid_path = CACHE_DIR / f'{slug}_grid.npy'
        log_path  = CACHE_DIR / f'{slug}_logging.npy'

        if not grid_path.exists():
            print(f"  SKIP {name}: {grid_path} not found")
            continue

        grid = np.load(grid_path)
        south, north, west, east = centre_to_bbox(lat, lon, km=AREA_KM)

        # Statistics
        valid = grid[grid >= YEW_TRANSPARENT_BELOW]
        stats = {
            'mean': float(np.mean(grid)),
            'median': float(np.median(grid)),
            'max': float(np.max(grid)),
            'p30_ha': float(np.sum(grid >= 0.30) * 10 * 10 / 1e4),
            'p50_ha': float(np.sum(grid >= 0.50) * 10 * 10 / 1e4),
            'p70_ha': float(np.sum(grid >= 0.70) * 10 * 10 / 1e4),
            'h': int(grid.shape[0]),
            'w': int(grid.shape[1]),
        }

        # Export yew probability PNG
        png_path = OUT_DIR / f'{slug}.png'
        size = grid_to_png(grid, YEWCMAP, png_path)
        total_bytes += size

        entry = {
            'slug': slug,
            'name': name,
            'desc': desc,
            'lat': lat,
            'lon': lon,
            'bbox': {
                'south': round(south, 6),
                'north': round(north, 6),
                'west': round(west, 6),
                'east': round(east, 6),
            },
            'stats': stats,
            'png': f'{slug}.png',
        }

        # Export forestry / logging overlay PNG (if available)
        if log_path.exists():
            log_grid = np.load(log_path)
            log_png_path = OUT_DIR / f'{slug}_logging.png'
            log_size = logging_to_png(log_grid, log_png_path)
            total_log_bytes += log_size
            entry['logging_png'] = f'{slug}_logging.png'

            # Logging stats
            total_px = log_grid.size
            entry['logging_stats'] = {
                'water_pct':    round(float(np.sum(log_grid == 1)) / total_px * 100, 1),
                'logged_lt20_pct': round(float(np.sum(log_grid == 2)) / total_px * 100, 1),
                'logged_20_40_pct': round(float(np.sum(log_grid == 3)) / total_px * 100, 1),
                'logged_40_80_pct': round(float(np.sum(log_grid == 4)) / total_px * 100, 1),
                'forest_gt80_pct': round(float(np.sum(log_grid == 5)) / total_px * 100, 1),
                'alpine_pct':   round(float(np.sum(log_grid == 6)) / total_px * 100, 1),
            }
            print(f"  ✓ {name}: {grid.shape[0]}×{grid.shape[1]} → "
                  f"yew {size/1024:.0f} KB + logging {log_size/1024:.0f} KB "
                  f"(P≥0.5: {stats['p50_ha']:.0f} ha)")
        else:
            print(f"  ✓ {name}: {grid.shape[0]}×{grid.shape[1]} → {size/1024:.0f} KB "
                  f"(P≥0.5: {stats['p50_ha']:.0f} ha) [no logging data]")

        manifest.append(entry)

    # Write manifest
    manifest_path = OUT_DIR / 'tiles.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Tiles exported: {len(manifest)}")
    print(f"  Total yew PNG size: {total_bytes / 1024 / 1024:.1f} MB")
    print(f"  Total logging PNG size: {total_log_bytes / 1024 / 1024:.1f} MB")
    print(f"  Combined size: {(total_bytes + total_log_bytes) / 1024 / 1024:.1f} MB")
    print(f"  Manifest: {manifest_path}")
    print(f"  Output dir: {OUT_DIR}")


if __name__ == '__main__':
    main()
