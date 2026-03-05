#!/usr/bin/env python3
"""
yew_in_protected_areas.py
─────────────────────────
For each study-area tile, rasterize BC Parks / Ecological Reserves / Protected
Areas (TA_PARK_ECORES_PA_SVW) onto the 10m probability grid and count what
fraction of yew habitat (P ≥ 0.5, after logging suppression) falls inside
protected areas.

Results broken down by:
  - Overall protected vs unprotected
  - Designation type: Provincial Park / Ecological Reserve / Protected Area
  - Park class: Class A / Class C

Usage:
    conda run -n yew_pytorch python scripts/analysis/yew_in_protected_areas.py
"""

import sys
from math import cos, radians
from pathlib import Path

import geopandas as gpd
import numpy as np
import requests
import rasterio.features
from rasterio.transform import from_bounds as rio_bounds

ROOT       = Path(__file__).resolve().parents[2]
TILE_CACHE = ROOT / "results" / "analysis" / "cwh_spot_comparisons" / "tile_cache"

# Province-wide parks data via BC Data Catalogue WFS (the local GDB is a
# northern-BC subset only; the WFS covers all of BC down to ~48°N).
PARKS_WFS = (
    "https://openmaps.gov.bc.ca/geo/pub/wfs"
    "?service=WFS&version=2.0.0&request=GetFeature"
    "&typeName=WHSE_TANTALIS.TA_PARK_ECORES_PA_SVW"
    "&outputFormat=application/json"
    "&srsName=EPSG:4326"
    "&count=5000"
)

SCALE_M   = 10
HA_PER_PX = (SCALE_M ** 2) / 10_000
THRESHOLD = 0.5

LOG_SUPPRESS = {
    1: 0.00, 2: 0.00, 3: 0.00,
    4: 0.20, 5: 0.35, 6: 0.00, 7: 1.00,
}

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
    (53.819, -132.435, "Haida Gwaii South"),
    (49.250, -121.750, "Chilliwack Uplands"),
    (55.250, -130.750, "Stewart Lowlands"),
    (51.250, -127.250, "Smith Sound"),
    (49.250, -125.250, "Alberni Valley"),
    (49.750, -123.750, "Sechelt Peninsula"),
    (52.750, -128.250, "Klemtu Forest"),
    (51.750, -127.750, "Namu Lowlands"),
]


def slugify(name):
    return name.lower().replace(' ', '_').replace('-', '_').replace('/', '_')


def centre_to_bbox(lat, lon, km=10):
    half_lat = (km * 1000 / 2) / 111320.0
    half_lon = (km * 1000 / 2) / (111320.0 * cos(radians(lat)))
    return lat - half_lat, lat + half_lat, lon - half_lon, lon + half_lon


def apply_suppression(grid, log_grid):
    out = grid.copy()
    for cat, factor in LOG_SUPPRESS.items():
        out[log_grid == cat] *= factor
    return out


def main():
    print("Loading parks / protected areas from BC Data Catalogue WFS...")
    try:
        r = requests.get(PARKS_WFS, timeout=90)
        r.raise_for_status()
        feats = r.json().get('features', [])
        parks = gpd.GeoDataFrame.from_features(feats, crs='EPSG:4326')
    except Exception as exc:
        sys.exit(f"Failed to fetch parks via WFS: {exc}")
    print(f"  {len(parks):,} protected area polygons loaded (province-wide)")
    print(f"  Designations: {parks['PROTECTED_LANDS_DESIGNATION'].value_counts().to_dict()}")
    print()

    # Designation codes → short label mapping
    DESIG_LABELS = {
        'PROVINCIAL PARK':    'Prov. Park',
        'ECOLOGICAL RESERVE': 'Ecol. Reserve',
        'PROTECTED AREA':     'Protected Area',
    }

    # Accumulators
    total_yew_px     = 0
    protected_px     = 0
    desig_px   = {d: 0 for d in DESIG_LABELS}
    class_px   = {'Class A': 0, 'Class C': 0}

    tile_rows = []
    tiles_found = 0

    for lat, lon, name in STUDY_AREAS:
        slug = slugify(name)
        grid_path = TILE_CACHE / f"{slug}_grid.npy"
        log_path  = TILE_CACHE / f"{slug}_logging.npy"
        if not grid_path.exists() or not log_path.exists():
            continue
        tiles_found += 1

        grid     = np.load(str(grid_path))
        log_grid = np.load(str(log_path))
        grid_h, grid_w = grid.shape

        suppressed = apply_suppression(grid, log_grid)
        yew_mask   = suppressed >= THRESHOLD   # bool, True = yew pixel

        south, north, west, east = centre_to_bbox(lat, lon)
        transform = rio_bounds(west, south, east, north, grid_w, grid_h)

        # Clip parks to tile bbox
        from shapely.geometry import box as sbox
        tile_box = sbox(west, south, east, north)
        local_parks = parks[parks.intersects(tile_box)].copy()

        tile_yew_px = int(yew_mask.sum())
        total_yew_px += tile_yew_px

        if local_parks.empty or tile_yew_px == 0:
            tile_rows.append({
                'name': name, 'yew_ha': tile_yew_px * HA_PER_PX,
                'protected_ha': 0.0, 'pct': 0.0
            })
            print(f"  {name:30s}  yew={tile_yew_px*HA_PER_PX:>7.1f} ha  protected=0 ha (no parks)")
            continue

        # Rasterize all parks to a single protected mask
        all_shapes = [
            (geom, 1)
            for geom in local_parks.geometry
            if geom is not None and not geom.is_empty
        ]
        park_raster = rasterio.features.rasterize(
            all_shapes,
            out_shape=(grid_h, grid_w),
            transform=transform,
            fill=0,
            dtype=np.uint8,
        ).astype(bool)

        in_park = yew_mask & park_raster
        tile_protected_px = int(in_park.sum())
        protected_px += tile_protected_px

        # Per-designation breakdown
        desig_tile = {}
        for desig, label in DESIG_LABELS.items():
            subset = local_parks[local_parks['PROTECTED_LANDS_DESIGNATION'] == desig]
            if subset.empty:
                desig_tile[desig] = 0
                continue
            shapes = [
                (geom, 1) for geom in subset.geometry
                if geom is not None and not geom.is_empty
            ]
            rast = rasterio.features.rasterize(
                shapes, out_shape=(grid_h, grid_w),
                transform=transform, fill=0, dtype=np.uint8,
            ).astype(bool)
            px = int((yew_mask & rast).sum())
            desig_tile[desig] = px
            desig_px[desig] += px

        # Per-class breakdown
        class_tile = {}
        for cls in ['Class A', 'Class C']:
            subset = local_parks[local_parks['PARK_CLASS'] == cls]
            if subset.empty:
                class_tile[cls] = 0
                continue
            shapes = [
                (geom, 1) for geom in subset.geometry
                if geom is not None and not geom.is_empty
            ]
            rast = rasterio.features.rasterize(
                shapes, out_shape=(grid_h, grid_w),
                transform=transform, fill=0, dtype=np.uint8,
            ).astype(bool)
            px = int((yew_mask & rast).sum())
            class_tile[cls] = px
            class_px[cls] += px

        pct = tile_protected_px / tile_yew_px * 100 if tile_yew_px > 0 else 0
        tile_rows.append({
            'name': name,
            'yew_ha': tile_yew_px * HA_PER_PX,
            'protected_ha': tile_protected_px * HA_PER_PX,
            'pct': pct,
            **{f'{d}_ha': desig_tile.get(d, 0) * HA_PER_PX for d in DESIG_LABELS},
        })

        desig_str = ' | '.join(
            f"{DESIG_LABELS[d]}={desig_tile[d]*HA_PER_PX:.0f}ha"
            for d in DESIG_LABELS if desig_tile.get(d, 0) > 0
        )
        print(f"  {name:30s}  yew={tile_yew_px*HA_PER_PX:>7.1f} ha  "
              f"protected={tile_protected_px*HA_PER_PX:>7.1f} ha ({pct:4.1f}%)  {desig_str}")

    # ── Summary ────────────────────────────────────────────────────────────
    total_yew_ha  = total_yew_px  * HA_PER_PX
    total_prot_ha = protected_px  * HA_PER_PX
    overall_pct   = protected_px / total_yew_px * 100 if total_yew_px else 0

    print()
    print("=" * 75)
    print(f"  Total yew habitat (P≥0.5, suppressed):  {total_yew_ha:>10,.1f} ha")
    print(f"  Inside protected areas:                  {total_prot_ha:>10,.1f} ha  ({overall_pct:.1f}%)")
    print(f"  Outside protected areas:                 {total_yew_ha-total_prot_ha:>10,.1f} ha  ({100-overall_pct:.1f}%)")
    print()
    print("  By designation:")
    for desig, label in DESIG_LABELS.items():
        ha  = desig_px[desig] * HA_PER_PX
        pct = desig_px[desig] / total_yew_px * 100 if total_yew_px else 0
        print(f"    {label:<20}  {ha:>8,.1f} ha  ({pct:.1f}% of all yew)")
    print()
    print("  By park class:")
    for cls in ['Class A', 'Class C']:
        ha  = class_px[cls] * HA_PER_PX
        pct = class_px[cls] / total_yew_px * 100 if total_yew_px else 0
        print(f"    {cls:<20}  {ha:>8,.1f} ha  ({pct:.1f}% of all yew)")
    print()

    # ── Per-tile table sorted by protected ha ──────────────────────────────
    print("  Per-tile breakdown (sorted by yew in protected areas):")
    print(f"  {'Tile':<30}  {'Yew ha':>8}  {'Prot. ha':>9}  {'%':>6}")
    print("  " + "-" * 58)
    for r in sorted(tile_rows, key=lambda x: x['protected_ha'], reverse=True):
        if r['yew_ha'] < 1:
            continue
        print(f"  {r['name']:<30}  {r['yew_ha']:>8.1f}  {r['protected_ha']:>9.1f}  {r['pct']:>5.1f}%")

    print()
    print(f"  Tiles processed: {tiles_found}")


if __name__ == "__main__":
    main()
