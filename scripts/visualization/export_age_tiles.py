#!/usr/bin/env python3
"""
Export VRI forest-age rasters as transparent PNG overlays for every study tile.

For each tile the script:
  1. Reads BC VRI 2024 polygons clipped to the tile bbox (EPSG:4326).
  2. Derives the best per-polygon age estimate from:
       PROJ_AGE_1  >  PROJ_AGE_CLASS_CD_1 midpoint  >  HARVEST_DATE
       >  LINE_7B_DISTURBANCE_HISTORY  >  OPENING_SOURCE flag
  3. Rasterises age (uint16, years) onto the same grid as the yew-probability
     tiles and caches to  results/.../tile_cache/{slug}_age.npy.
  4. Applies a forest-age colormap (red = young → dark-green = old).
     Water / alpine / no-data pixels are transparent.
  5. Writes RGBA PNG to  docs/tiles/{slug}_age.png.
  6. Updates docs/tiles/tiles.json to add an "age_png" key per tile.

Age encoding in the .npy cache
  0     = no data  /  water  /  alpine  →  transparent
  1-499 = age in years (capped at 499)
  500   = "old forest, unknown age" (PROJ_AGE unknown but VRI shows treed)
          rendered as > 250-yr colour (darkest green)

Usage:
    # From project root, using the project venv:
    .venv/bin/python scripts/visualization/export_age_tiles.py

    # Or with conda env:
    conda run -n yew_pytorch python scripts/visualization/export_age_tiles.py

Output:
    docs/tiles/{slug}_age.png    — one per tile with VRI coverage
    docs/tiles/tiles.json        — updated with "age_png" field
    results/analysis/cwh_spot_comparisons/tile_cache/{slug}_age.npy  — cache
"""

import json
import sys
from datetime import datetime
from math import cos, radians
from pathlib import Path

import geopandas as gpd
import numpy as np
from PIL import Image
from pyproj import Transformer as ProjTransformer
from rasterio.features import rasterize as rio_rasterize
from rasterio.transform import from_bounds as rio_bounds

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
VEG_COMP_GDB   = ROOT / 'data' / 'VEG_COMP_LYR_R1_POLY_2024.gdb'
VEG_COMP_LAYER = 'VEG_COMP_LYR_R1_POLY'
CACHE_DIR = ROOT / 'results' / 'analysis' / 'cwh_spot_comparisons' / 'tile_cache'
OUT_DIR   = ROOT / 'docs' / 'tiles'
TILES_JSON = OUT_DIR / 'tiles.json'

AREA_KM = 10

# ── VRI age class midpoints ────────────────────────────────────────────────────
_AGE_CLASS_MIDPOINT = {
    '1': 10, '2': 30, '3': 50, '4': 70, '5': 90,
    '6': 110, '7': 130, '8': 195, '9': 300,
}

# Special age-raster sentinels (uint16)
AGE_TRANSPARENT = 0    # water / alpine / no-data → transparent
AGE_OLD_UNKNOWN = 500  # old forest with no age record → darkest green

# Display cap − ages ≥ this display as the maximum "old-growth" colour
AGE_DISPLAY_MAX = 250

# ── Study areas ───────────────────────────────────────────────────────────────
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
    # 40 new tiles (March 2026)
    (50.400, -126.050, "Sayward Forest",         "NE VI CWHmm, upper valley forests"),
    (50.600, -128.000, "Cape Scott Lowlands",    "NW VI CWHvh1, inland plateau"),
    (50.540, -127.760, "Holberg Inlet",          "NW VI CWHvh1, remote"),
    (48.970, -125.250, "Barkley Sound Slopes",   "SW VI CWHmm1, Kennedy Lake area"),
    (49.450, -124.700, "Courtenay Uplands",      "Central-east VI CWHmm1"),
    (49.050, -125.450, "Ucluelet Peninsula",     "West VI CWHvh1, Ucluelet forest"),
    (49.650, -126.700, "Tahsis Narrows",         "West-central VI inner CWH"),
    (50.480, -126.000, "Kelsey Bay Forest",      "NE VI CWHmm1 valley"),
    (50.050, -122.850, "Whistler Callaghan",     "Coast Mtns CWH/MH, upper slopes"),
    (49.700, -122.300, "Coquitlam Watershed",    "Lower mainland CWHvm1"),
    (50.420, -122.600, "Lillooet R Corridor",    "Transition CWH-IDF upper valley"),
    (49.050, -122.050, "Harrison Lowlands",      "Fraser CWHdm, logged valley"),
    (49.980, -124.500, "Theodosia Inlet",        "Upper Sunshine Coast CWHvm2"),
    (50.600, -123.800, "Lillooet Lake Slopes",   "Pemberton-Squamish CWHvm1"),
    (50.380, -124.650, "Loughborough Inlet",     "Northern Sunshine Coast fjord CWH"),
    (51.450, -124.650, "Homathko Canyon",        "Upper Bute tributary CWHxm2"),
    (51.750, -125.750, "Klinaklini Valley",      "Central coast fjord CWHmm1"),
    (52.300, -126.650, "Dean River Lower",       "Lower Dean valley CWH, alluvial flats"),
    (53.000, -127.500, "Gardner Canal Slopes",   "Kemano valley inner CWH"),
    (53.350, -129.000, "Porcher Island",         "Outer north coast CWHvh2"),
    (51.650, -128.050, "Calvert Island",         "Outer CWHvh2, Rivers Inlet area"),
    (52.150, -128.100, "Bella Bella Forest",     "Denny Island CWHvh2, outer mid-coast"),
    (52.800, -128.500, "Laredo Sound East",      "Inner passage mainland CWHvh"),
    (53.200, -130.050, "Banks Island NE",        "North coast outer CWHvh"),
    (51.850, -127.300, "Roscoe Inlet",           "Deep mainland fjord CWHvm2"),
    (52.250, -127.750, "Khutze Inlet",           "Mid-coast fjord CWHvm2"),
    (54.350, -130.350, "Tsimpsean Peninsula",    "Prince Rupert mainland CWHvh"),
    (54.350, -130.150, "Chatham Sound Slopes",   "Near Prince Rupert mainland CWHvh"),
    (53.350, -132.050, "Haida Gwaii Central",    "Central Moresby Island CWHvh3"),
    (53.680, -132.450, "Haida Gwaii E Graham",   "Graham Island east coast CWH"),
    (54.000, -132.050, "Tow Hill Area",          "N Graham Island CWHvh3"),
    (53.550, -132.000, "Skidegate Flats",        "Graham-Moresby narrows CWH"),
    (51.950, -128.200, "Seaforth Channel",       "Mid-coast outer passage CWH"),
    (52.850, -127.800, "Mucha Inlet",            "Dean-Kimsquit corridor CWH"),
    (54.650, -130.350, "Observatory Inlet",      "North fjord CWHvh3"),
    (50.950, -127.350, "Blunden Harbour",        "Broughton area CWHvh inner"),
    (51.350, -126.700, "Tribune Channel",        "Remote outer-coast fjord CWH"),
    (52.150, -126.200, "Tweedsmuir South",       "Atnarko CWH valley bottom"),
    (54.100, -130.100, "Skeena Estuary",         "Outer coast Prince Rupert area CWHvh"),
    (54.650, -130.450, "Work Channel",           "N of Prince Rupert outer coast CWHvh3"),
]


def slugify(name):
    return name.lower().replace(' ', '_').replace('-', '_').replace('/', '_')


def centre_to_bbox(lat, lon, km=AREA_KM):
    half_lat = (km * 1000 / 2) / 111320.0
    half_lon = (km * 1000 / 2) / (111320.0 * cos(radians(lat)))
    return lat - half_lat, lat + half_lat, lon - half_lon, lon + half_lon


# ── Age extraction ─────────────────────────────────────────────────────────────

_CUR_YEAR_2D = datetime.now().year % 100
_CUR_CENTURY = (datetime.now().year // 100) * 100


def _decode_7b_year(code_2d):
    y = int(code_2d)
    return _CUR_CENTURY + y if y <= _CUR_YEAR_2D else (_CUR_CENTURY - 100) + y


def _parse_7b_min_age(field_val, current_year):
    """Minimum age (years) from LINE_7B_DISTURBANCE_HISTORY, or None."""
    if not field_val:
        return None
    min_age = None
    for part in str(field_val).split(';'):
        part = part.strip()
        if len(part) >= 3 and part[0].isalpha() and part[1:3].isdigit():
            try:
                event_year = _decode_7b_year(part[1:3])
                age = current_year - event_year
                if age >= 0:
                    min_age = age if min_age is None else min(min_age, age)
            except Exception:
                pass
    return min_age


def _vri_row_to_age(row, current_year):
    """
    Return the best age estimate (int years) and a land-cover flag.

    Returns:
        (age, flag)
        flag == 'water'  → transparent pixel
        flag == 'alpine' → transparent pixel
        flag == 'forest' → draw with age colour
        flag == 'old'    → old forest, age unknown (draw as AGE_OLD_UNKNOWN)
    """
    bclcs1 = str(row.get('BCLCS_LEVEL_1') or '').strip()
    bclcs2 = str(row.get('BCLCS_LEVEL_2') or '').strip()

    if bclcs1 == 'W' or (bclcs1 == 'N' and bclcs2 == 'W'):
        return 0, 'water'
    if bclcs1 == 'N' and bclcs2 == 'L':
        return 0, 'alpine'
    if str(row.get('ALPINE_DESIGNATION') or '').strip() == 'A':
        return 0, 'alpine'

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
        osrc = int(opening_src) if opening_src is not None else None
    except (ValueError, TypeError):
        osrc = None
    if (osrc in {3, 4, 7, 11}) and not ages:
        ages.append(0)

    if not ages:
        # No age info at all — treed land with no harvest record
        if bclcs2 == 'N':
            return 0, 'alpine'   # non-veg, no age = alpine/barren
        return 0, 'old'          # assume old forest

    age = max(0, min(ages))      # most recent disturbance, clamp ≥ 0
    return age, 'forest'


def extract_age_raster(south, north, west, east, H, W, cache_path, current_year):
    """
    Read VRI GDB for the tile bbox and return a uint16 age raster.
    Uses cache_path if it already exists.
    """
    if cache_path and Path(cache_path).exists():
        return np.load(str(cache_path))

    if not VEG_COMP_GDB.exists():
        print(f'    ⚠  VEG_COMP GDB not found: {VEG_COMP_GDB}')
        return None

    t4326_3005 = ProjTransformer.from_crs('EPSG:4326', 'EPSG:3005', always_xy=True)
    x_min, y_min = t4326_3005.transform(west, south)
    x_max, y_max = t4326_3005.transform(east, north)

    try:
        gdf = gpd.read_file(
            str(VEG_COMP_GDB),
            bbox=(x_min, y_min, x_max, y_max),
            layer=VEG_COMP_LAYER,
            columns=[
                'BCLCS_LEVEL_1', 'BCLCS_LEVEL_2',
                'PROJ_AGE_1', 'PROJ_AGE_CLASS_CD_1',
                'HARVEST_DATE', 'LINE_7B_DISTURBANCE_HISTORY',
                'OPENING_IND', 'OPENING_SOURCE', 'ALPINE_DESIGNATION',
                'geometry',
            ],
        )
    except Exception as e:
        print(f'    ⚠  VRI read failed: {e}')
        return None

    if gdf.empty:
        print(f'    No VRI polygons in bbox')
        return np.zeros((H, W), dtype=np.uint16)

    gdf = gdf.to_crs('EPSG:4326')

    # Build encoded-age column
    encoded_ages = []
    for _, row in gdf.iterrows():
        age, flag = _vri_row_to_age(row, current_year)
        if flag in ('water', 'alpine'):
            encoded_ages.append(AGE_TRANSPARENT)
        elif flag == 'old':
            encoded_ages.append(AGE_OLD_UNKNOWN)
        else:
            encoded_ages.append(min(499, age))   # 1-499 years
    gdf['enc_age'] = encoded_ages

    transform = rio_bounds(west, south, east, north, W, H)
    shapes = [
        (geom, int(enc))
        for geom, enc in zip(gdf.geometry, gdf['enc_age'])
        if geom is not None and not geom.is_empty
    ]
    raster = rio_rasterize(
        shapes,
        out_shape=(H, W),
        transform=transform,
        fill=AGE_TRANSPARENT,
        dtype='uint16',
    )

    # Pixels rasterised as AGE_TRANSPARENT (0) may be genuine old-forest with
    # no age data AND no harvest.  Correct: any cell that ended up 0 and is
    # surrounded by non-zero should stay 0 (no-data boundary), but we cannot
    # distinguish that here cheaply; leave as-is.

    if cache_path:
        np.save(str(cache_path), raster)

    classified = int((raster > 0).sum())
    print(f'    {len(gdf):,} VRI polygons → {classified:,} classified pixels')
    return raster


# ── Colormap ──────────────────────────────────────────────────────────────────
# Breakpoints: (normalised_age, R, G, B)
# normalised_age = age / AGE_DISPLAY_MAX, clamped [0, 1]
# AGE_OLD_UNKNOWN is treated as normalised = 1.0
_CMAP_POINTS = [
    (0.00, 180,  20,  20),   # 0 yr:   deep red   (bare cutblock)
    (0.08, 230,  80,  10),   # 20 yr:  orange
    (0.16, 240, 180,  20),   # 40 yr:  yellow
    (0.32, 170, 210,  50),   # 80 yr:  yellow-green
    (0.60, 40,  155,  50),   # 150 yr: medium green
    (1.00, 10,   70,  20),   # 250+ yr: dark forest green
]


def _age_to_rgba(age_norm, alpha=200):
    """Interpolate CMAP_POINTS for a normalised age in [0, 1]."""
    for i in range(len(_CMAP_POINTS) - 1):
        t0, r0, g0, b0 = _CMAP_POINTS[i]
        t1, r1, g1, b1 = _CMAP_POINTS[i + 1]
        if t0 <= age_norm <= t1:
            f = (age_norm - t0) / (t1 - t0)
            return (
                round(r0 + f * (r1 - r0)),
                round(g0 + f * (g1 - g0)),
                round(b0 + f * (b1 - b0)),
                alpha,
            )
    return (_CMAP_POINTS[-1][1], _CMAP_POINTS[-1][2], _CMAP_POINTS[-1][3], alpha)


def build_age_lut():
    """Pre-build a 501-entry RGBA lookup table (index = encoded age value)."""
    lut = np.zeros((501, 4), dtype=np.uint8)
    for enc in range(1, 500):          # 1-499 yr: real ages
        norm = min(1.0, enc / AGE_DISPLAY_MAX)
        lut[enc] = _age_to_rgba(norm, alpha=210)
    lut[AGE_OLD_UNKNOWN] = _age_to_rgba(1.0, alpha=210)   # index 500
    # index 0 remains (0,0,0,0) = transparent
    return lut


AGE_LUT = build_age_lut()


def age_raster_to_png(age_raster, out_path):
    """Colorise uint16 age raster and write transparent RGBA PNG."""
    H, W = age_raster.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)

    # Clamp indices into LUT
    idx = np.clip(age_raster, 0, 500).astype(np.uint16)
    rgba[:, :, :] = AGE_LUT[idx]

    img = Image.fromarray(rgba, 'RGBA')
    img.save(str(out_path), format='PNG', optimize=True)
    return out_path.stat().st_size


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not VEG_COMP_GDB.exists():
        sys.exit(f'ERROR: VEG_COMP GDB not found at {VEG_COMP_GDB}')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    current_year = datetime.now().year

    print('=' * 60)
    print('EXPORT FOREST-AGE TILE OVERLAYS')
    print('=' * 60)
    print(f'  GDB         : {VEG_COMP_GDB.name}')
    print(f'  Output dir  : {OUT_DIR}')
    print(f'  Cache dir   : {CACHE_DIR}')
    print(f'  Display cap : {AGE_DISPLAY_MAX} yr (anything older → darkest green)')
    print()

    # Load existing tiles.json manifest
    if not TILES_JSON.exists():
        sys.exit(f'ERROR: {TILES_JSON} not found — run export_tiles_for_web.py first')
    with open(TILES_JSON) as f:
        manifest = json.load(f)
    manifest_by_slug = {t['slug']: t for t in manifest}

    total_bytes = 0
    processed = 0
    skipped = 0

    for lat, lon, name, desc in STUDY_AREAS:
        slug = slugify(name)
        grid_path = CACHE_DIR / f'{slug}_grid.npy'
        age_cache  = CACHE_DIR / f'{slug}_age.npy'
        age_png    = OUT_DIR / f'{slug}_age.png'

        if not grid_path.exists():
            print(f'  SKIP {name}: no grid cache')
            skipped += 1
            continue

        # Reference grid shape to size the raster
        ref_grid = np.load(str(grid_path))
        H, W = ref_grid.shape
        south, north, west, east = centre_to_bbox(lat, lon)

        print(f'  [{processed + skipped + 1:2d}] {name:30s} ({H}×{W}) ...', flush=True)

        age_raster = extract_age_raster(
            south, north, west, east, H, W, age_cache, current_year
        )
        if age_raster is None:
            print(f'      ↳ skipped (VRI unavailable)')
            skipped += 1
            continue

        size = age_raster_to_png(age_raster, age_png)
        total_bytes += size

        # Update tiles.json entry
        if slug in manifest_by_slug:
            manifest_by_slug[slug]['age_png'] = f'{slug}_age.png'

        old_pct = float(np.sum(age_raster >= 80)) / age_raster.size * 100
        young_pct = float(np.sum((age_raster > 0) & (age_raster < 80))) / age_raster.size * 100
        print(f'      ↳ {size / 1024:.0f} KB · forest≥80yr: {old_pct:.1f}% · <80yr: {young_pct:.1f}%')
        processed += 1

    # Write updated tiles.json
    with open(TILES_JSON, 'w') as f:
        json.dump(list(manifest_by_slug.values()), f, indent=2)

    print()
    print('=' * 60)
    print('SUMMARY')
    print('=' * 60)
    print(f'  Processed : {processed}')
    print(f'  Skipped   : {skipped}')
    print(f'  Total PNG size : {total_bytes / 1024 / 1024:.1f} MB')
    print(f'  tiles.json updated with "age_png" field')


if __name__ == '__main__':
    main()
