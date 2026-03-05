#!/usr/bin/env python3
"""
Regenerate all *_logging.npy cache files by re-reading the VRI GDB with the
updated category scheme:
  cat 2 = logged <20 yr   (×0.00)
  cat 3 = logged 20-40 yr (×0.00)
  cat 4 = logged 40-80 yr (×0.50)
  cat 5 = forest 80-150yr (×0.35)  ← NEW: split from old cat 5 (>80yr)
  cat 7 = old-growth >150yr (×1.00) ← NEW

Delete or overwrite existing *_logging.npy files.

Usage:
    conda run -n yew_pytorch python scripts/preprocessing/regen_logging_grids.py
"""

import sys
from datetime import datetime
from math import cos, radians
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio.features
from pyproj import Transformer as ProjTransformer
from rasterio.transform import from_bounds as rio_bounds

# ── Config ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
VEG_COMP_GDB = ROOT / 'data' / 'VEG_COMP_LYR_R1_POLY_2024.gdb'
VEG_COMP_LAYER = 'VEG_COMP_LYR_R1_POLY'
CACHE_DIR  = ROOT / 'results' / 'analysis' / 'cwh_spot_comparisons' / 'tile_cache'
AREA_KM    = 10

STUDY_AREAS = [
    (48.440, -124.160, "Carmanah-Walbran"),
    (48.600, -123.800, "Sooke Hills"),
    (49.315, -124.980, "Clayoquot Sound"),
    (50.020, -125.240, "Campbell River Uplands"),
    (50.700, -127.100, "Quatsino Sound"),
    (49.700, -123.150, "Squamish Highlands"),
    (50.720, -124.000, "Desolation Sound"),
    (52.330, -126.600, "Bella Coola Valley"),
    (54.150, -129.700, "Prince Rupert Hills"),
    (53.500, -128.600, "Kitimat Ranges"),
    (49.900, -125.550, "Strathcona Highlands"),
    (49.860, -122.680, "Garibaldi Foothills"),
    (50.830, -124.920, "Bute Inlet Slopes"),
    (49.020, -124.200, "Nanaimo Lakes"),
    (51.400, -127.700, "Rivers Inlet"),
    (48.550, -124.420, "Port Renfrew"),
    (48.820, -124.050, "Cowichan Uplands"),
    (49.620, -125.100, "Comox Uplands"),
    (49.780, -126.020, "Gold River Forest"),
    (50.720, -127.500, "Port Hardy Forest"),
    (49.400, -123.720, "Sunshine Coast South"),
    (49.520, -123.420, "Howe Sound East"),
    (49.780, -124.550, "Powell River Forest"),
    (50.100, -124.060, "Jervis Inlet Slopes"),
    (51.020, -124.480, "Toba Inlet Slopes"),
    (51.080, -125.680, "Knight Inlet"),
    (50.760, -126.480, "Broughton Archipelago"),
    (51.220, -126.020, "Kingcome Inlet"),
    (51.640, -126.520, "Owikeno Lake"),
    (52.090, -126.840, "Burke Channel"),
    (52.380, -127.680, "Ocean Falls"),
    (52.720, -126.560, "Dean Channel"),
    (52.900, -128.700, "Princess Royal Island"),
    (52.510, -128.580, "Milbanke Sound"),
    (54.820, -130.120, "Portland Inlet"),
    (50.250, -125.750, "Muchalat Valley"),
    (49.250, -122.250, "Stave Lake"),
    (53.81907, -132.43530, "Haida Gwaii South"),
    (49.250, -121.750, "Chilliwack Uplands"),
    (55.250, -130.750, "Stewart Lowlands"),
    (51.250, -127.250, "Smith Sound"),
    (49.250, -125.250, "Alberni Valley"),
    (49.750, -123.750, "Sechelt Peninsula"),
    (52.750, -128.250, "Klemtu Forest"),
    (51.750, -127.750, "Namu Lowlands"),
    # 40 new tiles (March 2026)
    (50.400, -126.050, "Sayward Forest"),
    (50.600, -128.000, "Cape Scott Lowlands"),
    (50.540, -127.760, "Holberg Inlet"),
    (48.970, -125.250, "Barkley Sound Slopes"),
    (49.450, -124.700, "Courtenay Uplands"),
    (49.050, -125.450, "Ucluelet Peninsula"),
    (49.650, -126.700, "Tahsis Narrows"),
    (50.480, -126.000, "Kelsey Bay Forest"),
    (50.050, -122.850, "Whistler Callaghan"),
    (49.700, -122.300, "Coquitlam Watershed"),
    (50.420, -122.600, "Lillooet R Corridor"),
    (49.050, -122.050, "Harrison Lowlands"),
    (49.980, -124.500, "Theodosia Inlet"),
    (50.600, -123.800, "Lillooet Lake Slopes"),
    (50.380, -124.650, "Loughborough Inlet"),
    (51.450, -124.650, "Homathko Canyon"),
    (51.750, -125.750, "Klinaklini Valley"),
    (52.300, -126.650, "Dean River Lower"),
    (53.000, -127.500, "Gardner Canal Slopes"),
    (53.350, -129.000, "Porcher Island"),
    (51.650, -128.050, "Calvert Island"),
    (52.150, -128.100, "Bella Bella Forest"),
    (52.800, -128.500, "Laredo Sound East"),
    (53.200, -130.050, "Banks Island NE"),
    (51.850, -127.300, "Roscoe Inlet"),
    (52.250, -127.750, "Khutze Inlet"),
    (54.350, -130.350, "Tsimpsean Peninsula"),
    (54.350, -130.150, "Chatham Sound Slopes"),
    (53.350, -132.050, "Haida Gwaii Central"),
    (53.680, -132.450, "Haida Gwaii E Graham"),
    (54.000, -132.050, "Tow Hill Area"),
    (53.550, -132.000, "Skidegate Flats"),
    (51.950, -128.200, "Seaforth Channel"),
    (52.850, -127.800, "Mucha Inlet"),
    (54.650, -130.350, "Observatory Inlet"),
    (50.950, -127.350, "Blunden Harbour"),
    (51.350, -126.700, "Tribune Channel"),
    (52.150, -126.200, "Tweedsmuir South"),
    (54.100, -130.100, "Skeena Estuary"),
    (54.650, -130.450, "Work Channel"),
]

# PROJ_AGE_CLASS_CD_1 midpoint ages
_AGE_CLASS_MIDPOINT = {
    '1': 10, '2': 30, '3': 50, '4': 70,
    '5': 90, '6': 110, '7': 130, '8': 195, '9': 300,
}


def slugify(name):
    return name.lower().replace(' ', '_').replace('-', '_').replace('/', '_')


def centre_to_bbox(lat, lon, km=10):
    half_lat = (km * 1000 / 2) / 111320.0
    half_lon = (km * 1000 / 2) / (111320.0 * cos(radians(lat)))
    return lat - half_lat, lat + half_lat, lon - half_lon, lon + half_lon


def _parse_7b_min_age(val, current_year):
    """Parse LINE_7B_DISTURBANCE_HISTORY and return minimum age in years."""
    if not val:
        return None
    s = str(val).strip()
    if not s or s in ('None', 'nan'):
        return None
    best = None
    for token in s.split(';'):
        token = token.strip()
        if len(token) >= 3:
            yr2 = token[1:3]
            try:
                yr2_int = int(yr2)
                yr4 = (1900 + yr2_int) if yr2_int >= 20 else (2000 + yr2_int)
                age = current_year - yr4
                if best is None or age < best:
                    best = age
            except ValueError:
                pass
    return best


def _classify_vri_row(row, current_year):
    """Return 1-7 logging/land-cover category for a single VRI polygon row.

    Categories:
      1 = water / non-forest
      2 = logged <20 yr        (×0.00)
      3 = logged 20-40 yr      (×0.00)
      4 = logged 40-80 yr      (×0.50)
      5 = forest 80-150 yr     (×0.35)
      6 = alpine / barren      (×0.00)
      7 = old-growth >150 yr   (×1.00)
    """
    bclcs1 = str(row.get('BCLCS_LEVEL_1') or '').strip()
    bclcs2 = str(row.get('BCLCS_LEVEL_2') or '').strip()

    if bclcs1 == 'W' or (bclcs1 == 'N' and bclcs2 == 'W'):
        return 1
    if bclcs1 == 'N' and bclcs2 == 'L':
        return 6
    if str(row.get('ALPINE_DESIGNATION') or '').strip() == 'A':
        return 6

    ages = []

    pa1 = row.get('PROJ_AGE_1')
    if pa1 is not None:
        try:
            ages.append(int(pa1))
        except (ValueError, TypeError):
            pass

    pac = str(row.get('PROJ_AGE_CLASS_CD_1') or '').strip()
    if pac in _AGE_CLASS_MIDPOINT:
        ages.append(_AGE_CLASS_MIDPOINT[pac])

    dist7b = _parse_7b_min_age(row.get('LINE_7B_DISTURBANCE_HISTORY'), current_year)
    if dist7b is not None:
        ages.append(dist7b)

    hdate = row.get('HARVEST_DATE')
    if hdate:
        try:
            hy = hdate.year if hasattr(hdate, 'year') else int(str(hdate)[:4])
            ages.append(current_year - hy)
        except Exception:
            pass

    if str(row.get('OPENING_IND') or '').strip() == 'Y' and not ages:
        ages.append(0)

    opening_src = row.get('OPENING_SOURCE')
    try:
        opening_src_int = int(opening_src) if opening_src is not None else None
    except (ValueError, TypeError):
        opening_src_int = None
    if opening_src_int in {3, 4, 7, 11} and not ages:
        ages.append(0)

    if not ages:
        if bclcs2 == 'N':
            return 6
        return 7  # unknown / assume old-growth

    age = min(ages)
    if age < 20:
        return 2
    if age < 40:
        return 3
    if age < 80:
        return 4
    if age < 150:
        return 5   # maturing forest 80-150yr
    return 7       # old-growth >150yr


def extract_logging_grid(south, north, west, east, grid_h, grid_w):
    """Read VRI GDB bbox and rasterize to uint8 category grid."""
    current_year = datetime.now().year
    t4326_3005 = ProjTransformer.from_crs('EPSG:4326', 'EPSG:3005', always_xy=True)
    x_min, y_min = t4326_3005.transform(west, south)
    x_max, y_max = t4326_3005.transform(east, north)

    try:
        gdf = gpd.read_file(
            str(VEG_COMP_GDB),
            bbox=(x_min, y_min, x_max, y_max),
            layer=VEG_COMP_LAYER,
            columns=['BCLCS_LEVEL_1', 'BCLCS_LEVEL_2', 'PROJ_AGE_1',
                     'PROJ_AGE_CLASS_CD_1', 'HARVEST_DATE',
                     'LINE_7B_DISTURBANCE_HISTORY', 'OPENING_IND',
                     'OPENING_SOURCE', 'ALPINE_DESIGNATION', 'geometry'],
        )
    except Exception as e:
        print(f'    ⚠  VEG_COMP read failed: {e}')
        return np.zeros((grid_h, grid_w), dtype=np.uint8)

    if gdf.empty:
        print(f'    No VEG_COMP polygons in bbox — writing empty raster')
        return np.zeros((grid_h, grid_w), dtype=np.uint8)

    print(f'    {len(gdf):,} polygons — classifying...')
    gdf = gdf.to_crs('EPSG:4326')
    gdf['cat'] = [_classify_vri_row(row, current_year) for _, row in gdf.iterrows()]

    transform = rio_bounds(west, south, east, north, grid_w, grid_h)
    shapes = [
        (geom, int(cat))
        for geom, cat in zip(gdf.geometry, gdf['cat'])
        if geom is not None and not geom.is_empty
    ]
    raster = rasterio.features.rasterize(
        shapes,
        out_shape=(grid_h, grid_w),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )
    return raster


def main():
    if not VEG_COMP_GDB.exists():
        sys.exit(f'ERROR: VEG_COMP GDB not found: {VEG_COMP_GDB}')

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('REGENERATE LOGGING GRIDS (new category scheme)')
    print('  cat5 = 80-150yr (×0.35)  cat7 = >150yr (×1.00)')
    print('=' * 60)

    processed = 0
    for lat, lon, name in STUDY_AREAS:
        slug = slugify(name)
        grid_path = CACHE_DIR / f'{slug}_grid.npy'
        log_path  = CACHE_DIR / f'{slug}_logging.npy'

        if not grid_path.exists():
            print(f'  SKIP {name}: no grid cache')
            continue

        raw_grid = np.load(str(grid_path))
        H, W = raw_grid.shape
        south, north, west, east = centre_to_bbox(lat, lon)

        print(f'  [{processed+1:2d}] {name} ({H}×{W})...')
        raster = extract_logging_grid(south, north, west, east, H, W)
        np.save(str(log_path), raster)

        cat_counts = {c: int((raster == c).sum()) for c in range(8)}
        pct_old = (cat_counts[5] + cat_counts[7]) / raster.size * 100
        pct_cat5 = cat_counts[5] / raster.size * 100
        pct_cat7 = cat_counts[7] / raster.size * 100
        pct_log  = sum(cat_counts[c] for c in (2, 3, 4)) / raster.size * 100
        print(f'       → saved {log_path.name}  '
              f'forest≥80yr {pct_old:.1f}% (80-150yr {pct_cat5:.1f}%, >150yr {pct_cat7:.1f}%)'
              f'  logged {pct_log:.1f}%')
        processed += 1

    print(f'\nDone. Regenerated {processed} logging grids.')


if __name__ == '__main__':
    main()
