#!/usr/bin/env python3
"""
water_buffer_by_type.py
───────────────────────
Model 10 m and 20 m riparian/coastal buffer expansions on yew habitat,
broken out by water type:

  OC = ocean / saltwater coastline  (BCLCS_LEVEL_5 = 'OC')
  RI = rivers and streams            (BCLCS_LEVEL_5 = 'RI')
  LA = lakes                         (BCLCS_LEVEL_5 = 'LA')

Re-reads VRI GDB polygons for each tile to extract L5 water codes, builds
separate distance-transform masks, then applies 10 m and 20 m buffers for
each type independently and in combination.

Usage:
    conda run -n yew_pytorch python scripts/analysis/water_buffer_by_type.py
"""

import sys
from math import cos, radians
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio.features
from pyproj import Transformer as ProjTransformer
from rasterio.transform import from_bounds as rio_bounds
from scipy.ndimage import distance_transform_edt

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[2]
TILE_CACHE   = ROOT / "results" / "analysis" / "cwh_spot_comparisons" / "tile_cache"
VEG_COMP_GDB = ROOT / "data" / "VEG_COMP_LYR_R1_POLY_2024.gdb"
VEG_LAYER    = "VEG_COMP_LYR_R1_POLY"

# ── Constants ──────────────────────────────────────────────────────────────
SCALE_M   = 10
HA_PER_PX = (SCALE_M ** 2) / 10_000
THRESHOLD = 0.5
BUFFERS_M = [10, 20]

LOG_SUPPRESS = {
    1: 0.00, 2: 0.00, 3: 0.00,
    4: 0.20, 5: 0.35, 6: 0.00, 7: 1.00,
}

# ── Study areas ────────────────────────────────────────────────────────────
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


def yew_ha(prob_grid):
    return float((prob_grid >= THRESHOLD).sum()) * HA_PER_PX


def build_water_type_masks(south, north, west, east, grid_h, grid_w):
    """
    Re-read VRI GDB for this tile and return three boolean masks (all in grid
    pixel space):
      ocean_mask  — BCLCS_LEVEL_5 = 'OC'
      river_mask  — BCLCS_LEVEL_5 = 'RI'
      lake_mask   — BCLCS_LEVEL_5 = 'LA'

    Returns (ocean_mask, river_mask, lake_mask) all shape (grid_h, grid_w).
    """
    t = ProjTransformer.from_crs('EPSG:4326', 'EPSG:3005', always_xy=True)
    x_min, y_min = t.transform(west, south)
    x_max, y_max = t.transform(east, north)

    try:
        gdf = gpd.read_file(
            str(VEG_COMP_GDB),
            bbox=(x_min, y_min, x_max, y_max),
            layer=VEG_LAYER,
            columns=['BCLCS_LEVEL_1', 'BCLCS_LEVEL_2', 'BCLCS_LEVEL_5', 'geometry'],
        )
    except Exception as e:
        print(f"    ⚠  VRI read failed: {e}")
        empty = np.zeros((grid_h, grid_w), dtype=bool)
        return empty, empty, empty

    if gdf.empty:
        empty = np.zeros((grid_h, grid_w), dtype=bool)
        return empty, empty, empty

    gdf = gdf.to_crs('EPSG:4326')
    # Keep only water polygons (L1=W or L1=N,L2=W)
    water = gdf[
        (gdf['BCLCS_LEVEL_1'] == 'W') |
        ((gdf['BCLCS_LEVEL_1'] == 'N') & (gdf['BCLCS_LEVEL_2'] == 'W'))
    ].copy()

    transform = rio_bounds(west, south, east, north, grid_w, grid_h)

    def rasterize_type(code):
        subset = water[water['BCLCS_LEVEL_5'] == code]
        if subset.empty:
            return np.zeros((grid_h, grid_w), dtype=np.uint8)
        shapes = [
            (geom, 1)
            for geom in subset.geometry
            if geom is not None and not geom.is_empty
        ]
        if not shapes:
            return np.zeros((grid_h, grid_w), dtype=np.uint8)
        return rasterio.features.rasterize(
            shapes,
            out_shape=(grid_h, grid_w),
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )

    ocean_mask = rasterize_type('OC').astype(bool)
    river_mask = rasterize_type('RI').astype(bool)
    lake_mask  = rasterize_type('LA').astype(bool)

    return ocean_mask, river_mask, lake_mask


def dist_from_mask(mask, grid_h, grid_w):
    """Return Euclidean distance (metres) from every pixel to nearest True pixel."""
    if mask.any():
        return distance_transform_edt(~mask) * SCALE_M
    return np.full((grid_h, grid_w), 9999.0, dtype=np.float32)


def main():
    if not VEG_COMP_GDB.exists():
        sys.exit(f"VRI GDB not found: {VEG_COMP_GDB}")

    print("Water-buffer sensitivity by water type (ocean / river / lake)")
    print(f"Buffers: {BUFFERS_M} m   |   pixel = {SCALE_M} m   |   P threshold = {THRESHOLD}")
    print("=" * 80)

    # Scenario keys: individual types + combinations
    scenarios = [
        ("ocean",       ["OC"]),
        ("river",       ["RI"]),
        ("lake",        ["LA"]),
        ("river+lake",  ["RI", "LA"]),
        ("all water",   ["OC", "RI", "LA"]),
    ]

    # totals[scenario_name][buf_m] = total yew ha
    totals = {"baseline": 0.0}
    for name, _ in scenarios:
        for b in BUFFERS_M:
            totals[f"{name}_{b}m"] = 0.0

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

        suppressed_base = apply_suppression(grid, log_grid)
        base_ha = yew_ha(suppressed_base)
        totals["baseline"] += base_ha

        south, north, west, east = centre_to_bbox(lat, lon)

        print(f"  [{tiles_found:2d}] {name:30s} baseline={base_ha:,.0f} ha — reading VRI...", end="", flush=True)
        ocean_mask, river_mask, lake_mask = build_water_type_masks(
            south, north, west, east, grid_h, grid_w
        )

        oc_cnt = ocean_mask.sum()
        ri_cnt = river_mask.sum()
        la_cnt = lake_mask.sum()
        print(f" OC={oc_cnt:,} RI={ri_cnt:,} LA={la_cnt:,} px")

        # Distance transforms (metres) from each water type
        type_masks = {"OC": ocean_mask, "RI": river_mask, "LA": lake_mask}
        type_dists = {
            code: dist_from_mask(mask, grid_h, grid_w)
            for code, mask in type_masks.items()
        }

        # Existing water mask (cat=1 in log_grid) — never re-suppressed
        existing_water = (log_grid == 1)

        row = {"name": name, "slug": slug, "baseline_ha": base_ha}

        for scen_name, codes in scenarios:
            # Merge distance fields: a pixel is in-buffer if within buf_m of
            # ANY of the specified water types
            for b in BUFFERS_M:
                in_buf = np.zeros((grid_h, grid_w), dtype=bool)
                for code in codes:
                    in_buf |= (type_dists[code] <= b)
                # Exclude pixels already classified as water
                in_buf &= ~existing_water

                suppressed_buf = suppressed_base.copy()
                suppressed_buf[in_buf] = 0.0
                new_ha = yew_ha(suppressed_buf)
                lost_ha = base_ha - new_ha
                buf_zone_ha = float(in_buf.sum()) * HA_PER_PX

                key = f"{scen_name}_{b}m"
                totals[key] += new_ha
                row[f"{scen_name}_{b}m_yew"]  = new_ha
                row[f"{scen_name}_{b}m_lost"]  = lost_ha
                row[f"{scen_name}_{b}m_zone"]  = buf_zone_ha

        tile_rows.append(row)

    if tiles_found == 0:
        sys.exit("No tiles found.")

    base = totals["baseline"]
    print()
    print("=" * 80)
    print(f"  BASELINE yew habitat (no extra buffer): {base:>10,.1f} ha")
    print()

    # ── Summary table ──────────────────────────────────────────────────────
    col_w = 14
    print(f"  {'Scenario':<20} {'Buffer':>8}   {'Yew ha':>10}   {'Lost ha':>9}   {'Loss %':>7}   {'Zone ha':>9}")
    print("  " + "-" * 72)
    for scen_name, _ in scenarios:
        for b in BUFFERS_M:
            key = f"{scen_name}_{b}m"
            yew  = totals[key]
            lost = base - yew
            pct  = lost / base * 100 if base > 0 else 0
            # average zone ha across tiles
            zone_total = sum(r.get(f"{scen_name}_{b}m_zone", 0) for r in tile_rows)
            print(f"  {scen_name:<20} {b:>6} m   {yew:>10,.1f}   {lost:>9,.1f}   {pct:>6.2f}%   {zone_total:>9,.0f}")
        print()

    # ── Per-tile detail for the most ecologically relevant scenario ─────────
    print("─" * 80)
    print("  Per-tile detail: river+lake buffers only")
    scen = "river+lake"
    header_parts = "  " + f"{'Tile':<30}"
    for b in BUFFERS_M:
        header_parts += f"  {b:>2}m lost ha"
    print(header_parts)
    print("  " + "-" * 60)
    for r in sorted(tile_rows, key=lambda x: x["baseline_ha"], reverse=True):
        if r["baseline_ha"] < 1:
            continue
        parts = f"  {r['name']:<30}"
        for b in BUFFERS_M:
            lost = r.get(f"{scen}_{b}m_lost", 0)
            parts += f"  {lost:>10.1f}"
        print(parts)

    print()
    print(f"  Tiles processed: {tiles_found}")


if __name__ == "__main__":
    main()
