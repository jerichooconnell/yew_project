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

import geopandas as gpd
import numpy as np
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from rasterio.features import rasterize as rio_rasterize
from rasterio.transform import from_bounds as rio_bounds
from shapely.geometry import box as sbox

# ── Config ────────────────────────────────────────────────────────────────────
CACHE_DIR = Path('results/analysis/cwh_spot_comparisons/tile_cache')
OUT_DIR   = Path('docs/tiles')

AREA_KM = 10
YEW_TRANSPARENT_BELOW = 0.02

# ── Fire modifier config ─────────────────────────────────────────────────────
# Modifier = (FIRE_YEAR_NEW - fire_year) / FIRE_SPAN, clamped [0, 1]
# 0 = fire occurred in 2024 (complete suppression), 1 = fire in 1900 (full recovery)
FIRE_YEAR_OLD  = 1900
FIRE_YEAR_NEW  = 2024
FIRE_SPAN      = FIRE_YEAR_NEW - FIRE_YEAR_OLD  # 124 years

# Lower mainland tiles: subtract 0.2 outside protected areas (heavy logging not in VRI)
LOWER_MAINLAND_TILES = {'stave_lake', 'chilliwack_uplands'}

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
    # 10 gap-filling tiles (March 2026)
    (50.250, -125.750, "Muchalat Valley",        "Central VI CWHmm1/xm2 valley"),
    (49.250, -122.250, "Stave Lake",             "Lower Fraser Valley CWHvm1/dm"),
    (53.81907, -132.43530, "Haida Gwaii South",  "Moresby Island CWHvh3 outer coast"),
    (49.250, -121.750, "Chilliwack Uplands",     "Fraser Valley east CWHdm/ms1"),
    (55.250, -130.750, "Stewart Lowlands",       "Far north CWHvh3 near Stewart BC"),
    (51.250, -127.250, "Smith Sound",            "Mid-coast CWHvm2 mainland fjord"),
    (49.250, -125.250, "Alberni Valley",         "South-central VI CWHmm1/vm2"),
    (49.750, -123.750, "Sechelt Peninsula",      "Central Sunshine Coast CWHvm2/dm"),
    (52.750, -128.250, "Klemtu Forest",          "Mid-coast CWHvh2/vm2 inner islands"),
    (51.750, -127.750, "Namu Lowlands",          "Central coast CWHvh2 old-growth"),
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
# 3 = logged 20–40 yr, 4 = logged 40–80 yr, 5 = forest 80-150 yr,
# 6 = alpine / barren, 7 = old-growth >150 yr
LOG_RGBA = {
    1: (30,  100, 220, 180),   # water / non-forest
    2: (220, 50,  50,  170),   # logged  <20 yr
    3: (230, 120, 30,  150),   # logged 20–40 yr
    4: (220, 200, 50,  110),   # logged 40–80 yr
    5: (180, 220,  70,  70),   # forest 80–150 yr  (lime/chartreuse)
    6: (175, 155, 125, 160),   # alpine / barren
    7: (20,  100,  40,  70),   # old-growth >150 yr (deep forest green)
}

# Suppression factors by VRI logging category (same as classify_cwh_spots.py)
LOG_SUPPRESS = {
    1: 0.00,   # water / non-forest → zero out completely
    2: 0.00,   # logged  <20 yr     → zero out
    3: 0.00,   # logged 20–40 yr    → zero out (young second-growth, yew absent)
    4: 0.20,   # logged 40–80 yr    → heavily suppressed (yew slow to recover)
    5: 0.35,   # forest 80–150 yr   → partial suppression (maturing second-growth)
    6: 0.00,   # alpine / barren    → zero out
    7: 1.00,   # old-growth >150 yr → unchanged
}


def apply_logging_mask(grid, log_grid):
    """Suppress yew probabilities based on VRI logging categories."""
    masked = grid.copy()
    for cat, factor in LOG_SUPPRESS.items():
        where = log_grid == cat
        masked[where] *= factor
    return masked


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


def load_fire_gdf():
    """Load fire contours GeoJSON for modifier calculations."""
    path = OUT_DIR / 'fire_contours.geojson'
    if not path.exists():
        print(f'  WARNING: {path} not found — fire modifier skipped')
        return None
    gdf = gpd.read_file(path)
    gdf = gdf.dropna(subset=['FIRE_YEAR'])
    gdf['FIRE_YEAR'] = gdf['FIRE_YEAR'].astype(int)
    print(f'  Loaded {len(gdf)} fire polygons')
    return gdf


def load_park_gdf():
    """Load protected areas GeoJSON for the park mask."""
    path = OUT_DIR / 'park_contours.geojson'
    if not path.exists():
        print(f'  WARNING: {path} not found — park mask skipped')
        return None
    gdf = gpd.read_file(path)
    print(f'  Loaded {len(gdf)} park polygons')
    return gdf


def make_fire_modifier(H, W, west, south, east, north, fires_gdf):
    """Rasterize fire polygons → per-pixel modifier [0,1].

    modifier = (2024 - fire_year) / 124  (0 = 2024 fire, 1 = 1900 or unburned)
    For overlapping fires the most recent one wins (sort ascending → overwrite).
    Returns (modifier_array float32, fire_year_raster int16).
    """
    if fires_gdf is None:
        return np.ones((H, W), dtype=np.float32), np.zeros((H, W), dtype=np.int16)

    tile_box = sbox(west, south, east, north)
    local = fires_gdf[fires_gdf.intersects(tile_box)].copy()
    if len(local) == 0:
        return np.ones((H, W), dtype=np.float32), np.zeros((H, W), dtype=np.int16)

    transform = rio_bounds(west, south, east, north, W, H)
    # Sort ascending so more recent fires overwrite older ones
    local = local.sort_values('FIRE_YEAR', ascending=True)
    shapes = [(geom, int(yr)) for geom, yr in zip(local.geometry, local['FIRE_YEAR'])
              if geom is not None and not geom.is_empty]

    fire_year_raster = rio_rasterize(
        shapes, out_shape=(H, W), transform=transform, fill=0, dtype='int16',
    )
    modifier = np.where(
        fire_year_raster > 0,
        np.clip((FIRE_YEAR_NEW - fire_year_raster) / FIRE_SPAN, 0.0, 1.0),
        1.0,
    ).astype(np.float32)
    return modifier, fire_year_raster


def make_park_mask(H, W, west, south, east, north, parks_gdf):
    """Rasterize park polygons → boolean mask (True = inside a protected area)."""
    if parks_gdf is None:
        return np.zeros((H, W), dtype=bool)

    tile_box = sbox(west, south, east, north)
    local = parks_gdf[parks_gdf.intersects(tile_box)].copy()
    if len(local) == 0:
        return np.zeros((H, W), dtype=bool)

    transform = rio_bounds(west, south, east, north, W, H)
    shapes = [(geom, 1) for geom in local.geometry
              if geom is not None and not geom.is_empty]
    return rio_rasterize(
        shapes, out_shape=(H, W), transform=transform, fill=0, dtype='uint8',
    ).astype(bool)


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

    print('=' * 60)
    print('EXPORT TILE OVERLAYS FOR GITHUB PAGES')
    print('=' * 60)

    # Load overlay GeoDataFrames for fire modifier and park mask
    print('\nLoading overlay data...')
    fires_gdf = load_fire_gdf()
    parks_gdf = load_park_gdf()
    print()

    # Decade stats accumulator {decade_start: {burned_ha, yew_suppressed_ha, modifier_vals}}
    decade_stats = {}

    for lat, lon, name, desc in STUDY_AREAS:
        slug = slugify(name)
        grid_path = CACHE_DIR / f'{slug}_grid.npy'
        log_path  = CACHE_DIR / f'{slug}_logging.npy'

        if not grid_path.exists():
            print(f'  SKIP {name}: {grid_path} not found')
            continue

        raw_grid = np.load(grid_path)
        south, north, west, east = centre_to_bbox(lat, lon, km=AREA_KM)
        H, W = raw_grid.shape

        # Step 1: Apply logging / water / alpine mask
        if log_path.exists():
            log_grid = np.load(log_path)
            grid = apply_logging_mask(raw_grid, log_grid)
            zeroed = int((raw_grid > 0.02).sum()) - int((grid > 0.02).sum())
        else:
            grid = raw_grid.copy().astype(np.float32)
            log_grid = None
            zeroed = 0

        # Step 2: Apply fire-date modifier — (2024 - fire_year) / 124
        fire_modifier, fire_year_raster = make_fire_modifier(H, W, west, south, east, north, fires_gdf)
        pre_fire_grid = grid.copy()
        grid = (grid * fire_modifier).astype(np.float32)

        # Accumulate per-decade stats
        for yr in np.unique(fire_year_raster[fire_year_raster > 0]):
            decade_start = int(yr // 10 * 10)
            yr_mask = fire_year_raster == yr
            mod_val = float(np.clip((FIRE_YEAR_NEW - yr) / FIRE_SPAN, 0.0, 1.0))
            if decade_start not in decade_stats:
                decade_stats[decade_start] = {'burned_ha': 0.0, 'yew_suppressed_ha': 0.0, 'modifier_vals': []}
            decade_stats[decade_start]['burned_ha'] += float(np.sum(yr_mask)) * 0.01
            decade_stats[decade_start]['yew_suppressed_ha'] += float(
                np.sum(pre_fire_grid[yr_mask] * (1.0 - mod_val))
            ) * 0.01
            decade_stats[decade_start]['modifier_vals'].append(mod_val)

        # Step 3: Lower-mainland tiles — subtract 0.2 outside protected areas
        lm_info = ''
        if slug in LOWER_MAINLAND_TILES:
            park_mask = make_park_mask(H, W, west, south, east, north, parks_gdf)
            outside_count = int(np.sum(~park_mask & (grid > 0.0)))
            grid[~park_mask] = np.clip(grid[~park_mask] - 0.20, 0.0, 1.0)
            lm_info = f' [LM –0.2 outside parks: {outside_count:,} px]'

        grid = np.clip(grid, 0.0, 1.0)

        # Statistics (on final modified grid)
        stats = {
            'mean': float(np.mean(grid)),
            'median': float(np.median(grid)),
            'max': float(np.max(grid)),
            'p30_ha': float(np.sum(grid >= 0.30) * 10 * 10 / 1e4),
            'p50_ha': float(np.sum(grid >= 0.50) * 10 * 10 / 1e4),
            'p70_ha': float(np.sum(grid >= 0.70) * 10 * 10 / 1e4),
            'h': H,
            'w': W,
        }

        # Export yew probability PNG (all modifiers applied)
        png_path = OUT_DIR / f'{slug}.png'
        size = grid_to_png(grid, YEWCMAP, png_path)
        total_bytes += size

        fire_pct = float(np.sum(fire_year_raster > 0)) / (H * W) * 100

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
        if log_grid is not None:
            log_png_path = OUT_DIR / f'{slug}_logging.png'
            log_size = logging_to_png(log_grid, log_png_path)
            total_log_bytes += log_size
            entry['logging_png'] = f'{slug}_logging.png'

            total_px = log_grid.size
            entry['logging_stats'] = {
                'water_pct':          round(float(np.sum(log_grid == 1)) / total_px * 100, 1),
                'logged_lt20_pct':    round(float(np.sum(log_grid == 2)) / total_px * 100, 1),
                'logged_20_40_pct':   round(float(np.sum(log_grid == 3)) / total_px * 100, 1),
                'logged_40_80_pct':   round(float(np.sum(log_grid == 4)) / total_px * 100, 1),
                'forest_80_150_pct':  round(float(np.sum(log_grid == 5)) / total_px * 100, 1),
                'alpine_pct':         round(float(np.sum(log_grid == 6)) / total_px * 100, 1),
                'oldgrowth_pct':      round(float(np.sum(log_grid == 7)) / total_px * 100, 1),
            }
            print(f'  ✓ {name}: {H}×{W} → '
                  f'yew {size/1024:.0f} KB + logging {log_size/1024:.0f} KB '
                  f'(P≥0.5: {stats["p50_ha"]:.0f} ha, {zeroed:,} masked, '
                  f'fire {fire_pct:.1f}%){lm_info}')
        else:
            print(f'  ✓ {name}: {H}×{W} → {size/1024:.0f} KB '
                  f'(P≥0.5: {stats["p50_ha"]:.0f} ha, fire {fire_pct:.1f}%) [no VRI]{lm_info}')

        manifest.append(entry)

    # ── Write tiles.json ─────────────────────────────────────────────────────
    manifest_path = OUT_DIR / 'tiles.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # ── Write fire_stats.json for the Fire Impact tab ────────────────────────
    decades_out = []
    for ds in sorted(decade_stats.keys()):
        row = decade_stats[ds]
        modvals = row['modifier_vals']
        decades_out.append({
            'decade_start':      ds,
            'decade_label':      f'{ds}s',
            'fire_years_in_study': len(modvals),
            'burned_ha':         round(row['burned_ha'], 1),
            'yew_suppressed_ha': round(row['yew_suppressed_ha'], 2),
            'mean_modifier':     round(float(np.mean(modvals)), 3),
        })
    fire_stats = {
        'decades': decades_out,
        'total_yew_suppressed_ha': round(sum(d['yew_suppressed_ha'] for d in decades_out), 2),
        'total_burned_ha':         round(sum(d['burned_ha'] for d in decades_out), 1),
        'modifier_formula': '(2024 - fire_year) / 124, clamped [0, 1]',
        'lower_mainland_tiles': list(LOWER_MAINLAND_TILES),
        'lower_mainland_note': 'Additional -0.2 suppression applied outside protected areas',
    }
    fire_stats_path = OUT_DIR / 'fire_stats.json'
    with open(fire_stats_path, 'w') as f:
        json.dump(fire_stats, f, indent=2)

    print(f"\n{'=' * 60}")
    print('SUMMARY')
    print('=' * 60)
    print(f'  Tiles exported: {len(manifest)}')
    print(f'  Total yew PNG size: {total_bytes / 1024 / 1024:.1f} MB')
    print(f'  Total logging PNG size: {total_log_bytes / 1024 / 1024:.1f} MB')
    print(f'  Combined size: {(total_bytes + total_log_bytes) / 1024 / 1024:.1f} MB')
    print(f'  Manifest: {manifest_path}')
    print(f'  Fire stats: {fire_stats_path}')
    print(f'  Output dir: {OUT_DIR}')
    print(f'  Total yew suppressed by fires: {fire_stats["total_yew_suppressed_ha"]:.0f} ha')


if __name__ == '__main__':
    main()
