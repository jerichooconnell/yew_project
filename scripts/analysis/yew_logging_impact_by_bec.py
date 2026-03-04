#!/usr/bin/env python3
"""
yew_logging_impact_by_bec.py
─────────────────────────────
For each of the 45 study-area tiles, cross-tabulates yew habitat probability,
VRI logging category, and BEC subzone to estimate:

  1. Original yew habitat per BEC subzone (ha)
     — assumes logged areas historically had the same yew prevalence as
       surviving old-growth in the same BEC subzone within the same tile.
  2. Logging-destroyed yew habitat per BEC subzone (ha)
     — difference between estimated original and currently remaining habitat.

Outputs
-------
  results/analysis/yew_logging_impact_by_bec.csv   — per-BEC-subzone table
  results/analysis/yew_logging_impact_by_bec.txt   — human-readable summary
"""

import sys, warnings
from pathlib import Path
from math import cos, radians

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio.features
from rasterio import transform as rtransform
from rasterio.transform import from_bounds as rio_bounds
from pyproj import Transformer as ProjTransformer
from shapely.geometry import box as sbox

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
TILE_CACHE = ROOT / "results" / "analysis" / "cwh_spot_comparisons" / "tile_cache"
BEC_GDB    = ROOT / "data" / "BEC_BIOGEOCLIMATIC_POLY.gdb"
OUT_DIR    = ROOT / "results" / "analysis"

SCALE_M  = 10          # 10 m per pixel
HA_PER_PX = (SCALE_M ** 2) / 10_000   # 0.01 ha per pixel
THRESHOLD = 0.50       # P ≥ this to count as "yew habitat"

# ── Fire modifier constants ────────────────────────────────────────────────────
FIRE_CONTOURS = ROOT / "docs" / "tiles" / "fire_contours.geojson"
FIRE_YEAR_NEW = 2024
FIRE_SPAN     = 124    # 2024 − 1900

# ── Study areas (identical to classify_cwh_spots.py) ─────────────────────────
STUDY_AREAS = [
    (48.440, -124.160, "Carmanah-Walbran",        "South VI old-growth CWH"),
    (48.600, -123.800, "Sooke Hills",             "South VI montane CWH"),
    (49.315, -124.980, "Clayoquot Sound",          "West central VI CWH"),
    (50.020, -125.240, "Campbell River Uplands",   "North-central VI CWH"),
    (50.700, -127.100, "Quatsino Sound",           "Northern VI CWH"),
    (49.700, -123.150, "Squamish Highlands",       "Coast Mountains south CWH"),
    (50.720, -124.000, "Desolation Sound",         "Sunshine Coast north CWH"),
    (52.330, -126.600, "Bella Coola Valley",       "Central coast CWH"),
    (54.150, -129.700, "Prince Rupert Hills",      "North coast CWH"),
    (53.500, -128.600, "Kitimat Ranges",           "Skeena CWH fringe"),
    (49.900, -125.550, "Strathcona Highlands",     "Central VI CWH/MH boundary"),
    (49.860, -122.680, "Garibaldi Foothills",      "Mainland coast CWH near Whistler"),
    (50.830, -124.920, "Bute Inlet Slopes",        "Deep fjord CWH, Coast Mountains"),
    (49.020, -124.200, "Nanaimo Lakes",            "South VI mid-elevation CWH"),
    (51.400, -127.700, "Rivers Inlet",             "Central BC outer coast CWH"),
    (48.550, -124.420, "Port Renfrew",             "SW VI Pacific Rim old-growth CWH"),
    (48.820, -124.050, "Cowichan Uplands",         "South VI lower-elevation CWH"),
    (49.620, -125.100, "Comox Uplands",            "Central VI mid-elevation CWH"),
    (49.780, -126.020, "Gold River Forest",        "West-central VI inner CWH"),
    (50.720, -127.500, "Port Hardy Forest",        "North VI CWH valley bottoms"),
    (49.400, -123.720, "Sunshine Coast South",     "Lower Sunshine Coast CWH"),
    (49.520, -123.420, "Howe Sound East",          "Howe Sound montane CWH"),
    (49.780, -124.550, "Powell River Forest",      "Upper Sunshine Coast CWH"),
    (50.100, -124.060, "Jervis Inlet Slopes",      "Jervis Inlet fjord CWH"),
    (51.020, -124.480, "Toba Inlet Slopes",        "Remote fjord CWH, n. Sunshine Coast"),
    (51.080, -125.680, "Knight Inlet",             "Deep fjord, mainland Coast Mountains CWH"),
    (50.760, -126.480, "Broughton Archipelago",    "Outer archipelago CWH"),
    (51.220, -126.020, "Kingcome Inlet",           "Remote fjord valley, inner CWH"),
    (51.640, -126.520, "Owikeno Lake",             "Rivers Inlet drainage, interior CWH"),
    (52.090, -126.840, "Burke Channel",            "Outer fjord near Bella Coola"),
    (52.380, -127.680, "Ocean Falls",              "Outer coast CWH, high precipitation"),
    (52.720, -126.560, "Dean Channel",             "Dean River CWH, coast-interior edge"),
    (52.900, -128.700, "Princess Royal Island",    "Outer island CWH, spirit bear range"),
    (52.510, -128.580, "Milbanke Sound",           "Outer mid-coast CWH"),
    (54.820, -130.120, "Portland Inlet",           "Far north coast CWH near Nisga'a"),
    (50.250, -125.750, "Muchalat Valley",          "Central VI CWHmm1/xm2 valley"),
    (49.250, -122.250, "Stave Lake",               "Lower Fraser Valley CWHvm1/dm"),
    (53.750, -131.750, "Haida Gwaii South",        "Moresby Island CWHvh3 outer coast"),
    (49.250, -121.750, "Chilliwack Uplands",       "Fraser Valley east CWHdm/ms1"),
    (55.250, -130.750, "Stewart Lowlands",         "Far north CWHvh3 near Stewart BC"),
    (51.250, -127.250, "Smith Sound",              "Mid-coast CWHvm2 mainland fjord"),
    (49.250, -125.250, "Alberni Valley",           "South-central VI CWHmm1/vm2"),
    (49.750, -123.750, "Sechelt Peninsula",        "Central Sunshine Coast CWHvm2/dm"),
    (52.750, -128.250, "Klemtu Forest",            "Mid-coast CWHvh2/vm2 inner islands"),
    (51.750, -127.750, "Namu Lowlands",            "Central coast CWHvh2 old-growth"),
]


def slugify(name):
    return name.lower().replace(" ", "_").replace("-", "_").replace("'", "")


def centre_to_bbox(lat, lon, km=10):
    half_lat = (km * 1000 / 2) / 111320.0
    half_lon = (km * 1000 / 2) / (111320.0 * cos(radians(lat)))
    return lat - half_lat, lat + half_lat, lon - half_lon, lon + half_lon


# ── Fire modifier helpers ─────────────────────────────────────────────────────

def load_fire_gdf():
    """Load fire contours GeoJSON. Returns GeoDataFrame or None."""
    if not FIRE_CONTOURS.exists():
        print(f"  WARNING: {FIRE_CONTOURS} not found — fire modifier skipped")
        return None
    gdf = gpd.read_file(str(FIRE_CONTOURS))
    gdf = gdf.dropna(subset=["FIRE_YEAR"])
    gdf["FIRE_YEAR"] = gdf["FIRE_YEAR"].astype(int)
    print(f"  Loaded {len(gdf)} fire polygons for fire modifier")
    return gdf


def make_fire_modifier(H, W, west, south, east, north, fires_gdf):
    """Return per-pixel fire modifier [0, 1].
    modifier = (2024 − fire_year) / 124 — 0 = 2024 fire, 1 = unburned / pre-1900.
    Most recent fire wins for overlapping polygons.
    """
    if fires_gdf is None:
        return np.ones((H, W), dtype=np.float32)
    tile_box = sbox(west, south, east, north)
    local = fires_gdf[fires_gdf.intersects(tile_box)].copy()
    if len(local) == 0:
        return np.ones((H, W), dtype=np.float32)
    transform = rio_bounds(west, south, east, north, W, H)
    local = local.sort_values("FIRE_YEAR", ascending=True)   # newer overwrites older
    shapes = [
        (geom, int(yr))
        for geom, yr in zip(local.geometry, local["FIRE_YEAR"])
        if geom is not None and not geom.is_empty
    ]
    fire_year_raster = rasterio.features.rasterize(
        shapes, out_shape=(H, W), transform=transform, fill=0, dtype="int16",
    )
    modifier = np.where(
        fire_year_raster > 0,
        np.clip((FIRE_YEAR_NEW - fire_year_raster) / FIRE_SPAN, 0.0, 1.0),
        1.0,
    ).astype(np.float32)
    return modifier


def rasterize_bec_zones(bbox, grid_h, grid_w):
    """
    Read BEC_BIOGEOCLIMATIC_POLY.gdb clipped to *bbox* and rasterize
    MAP_LABEL strings onto a grid.  Returns (label_grid, label_lookup) where
    label_grid is int16 (0 = no data) and label_lookup maps int→MAP_LABEL.
    """
    south, north, west, east = bbox

    # Spatial filter — BEC GDB is in EPSG:3005
    t4326_3005 = ProjTransformer.from_crs("EPSG:4326", "EPSG:3005", always_xy=True)
    x_min, y_min = t4326_3005.transform(west, south)
    x_max, y_max = t4326_3005.transform(east, north)

    gdf = gpd.read_file(
        str(BEC_GDB),
        bbox=(x_min, y_min, x_max, y_max),
        columns=["MAP_LABEL", "ZONE", "SUBZONE", "geometry"],
    )
    if gdf.empty:
        return np.zeros((grid_h, grid_w), dtype=np.int16), {}

    gdf = gdf.to_crs("EPSG:4326")

    # Build integer code for each unique MAP_LABEL
    unique_labels = sorted(gdf["MAP_LABEL"].dropna().unique())
    lbl2code = {lbl: i + 1 for i, lbl in enumerate(unique_labels)}
    code2lbl = {v: k for k, v in lbl2code.items()}

    shapes = []
    for _, row in gdf.iterrows():
        lbl = row["MAP_LABEL"]
        geom = row.geometry
        if lbl and geom is not None and not geom.is_empty:
            shapes.append((geom, lbl2code[lbl]))

    transform = rtransform.from_bounds(west, south, east, north, grid_w, grid_h)
    raster = rasterio.features.rasterize(
        shapes,
        out_shape=(grid_h, grid_w),
        transform=transform,
        fill=0,
        dtype=np.int16,
    )
    return raster, code2lbl


# ── Logging category labels ──────────────────────────────────────────────────
LOG_LABELS = {
    0: "No data",
    1: "Water / non-forest",
    2: "Logged <20 yr",
    3: "Logged 20-40 yr",
    4: "Logged 40-80 yr",
    5: "Forest >80 yr",
    6: "Alpine / barren",
}

LOG_SUPPRESS = {
    1: 0.00,
    2: 0.00,
    3: 0.08,
    4: 0.50,
    5: 1.00,
    6: 0.00,
}


def main():
    if not BEC_GDB.exists():
        sys.exit(f"ERROR: BEC GDB not found: {BEC_GDB}")

    print(f"Yew Logging Impact by BEC Zone")
    print(f"{'='*60}")
    print(f"Tiles:     {len(STUDY_AREAS)}")
    print(f"Threshold: P ≥ {THRESHOLD}")
    print(f"BEC GDB:   {BEC_GDB.name}")
    print()

    # Load fire contours for fire modifier
    fires_gdf = load_fire_gdf()
    print()

    # Accumulators per BEC subzone
    # For each MAP_LABEL we track:
    #   total_px           – total pixels in that BEC zone across all tiles
    #   oldgrowth_px       – cat 5 pixels
    #   oldgrowth_yew_px   – cat 5 pixels with P ≥ threshold
    #   logged_px          – cat 2+3+4 pixels (all logged categories)
    #   logged_cat2_px     – cat 2 pixels specifically
    #   logged_cat3_px     – cat 3 pixels specifically
    #   logged_cat4_px     – cat 4 pixels specifically
    #   water_px           – cat 1 pixels
    #   alpine_px          – cat 6 pixels
    #   nodata_px          – cat 0 pixels
    #   current_yew_px     – pixels with suppressed P ≥ threshold
    #   raw_yew_px         – pixels with raw P ≥ threshold (regardless of logging)
    from collections import defaultdict
    stats = defaultdict(lambda: defaultdict(float))

    tiles_processed = 0
    tiles_skipped = 0

    for lat, lon, name, desc in STUDY_AREAS:
        slug = slugify(name)
        grid_path = TILE_CACHE / f"{slug}_grid.npy"
        log_path  = TILE_CACHE / f"{slug}_logging.npy"

        if not grid_path.exists() or not log_path.exists():
            print(f"  SKIP {name}: missing cache files")
            tiles_skipped += 1
            continue

        grid = np.load(str(grid_path))       # raw probability grid
        log_grid = np.load(str(log_path))     # VRI logging categories
        grid_h, grid_w = grid.shape

        bbox = centre_to_bbox(lat, lon)

        # Compute logging-suppressed grid
        suppressed = grid.copy()
        for cat, factor in LOG_SUPPRESS.items():
            suppressed[log_grid == cat] *= factor

        # Compute fire modifier and apply on top of logging suppression
        south_t, north_t, west_t, east_t = bbox
        fire_modifier = make_fire_modifier(grid_h, grid_w, west_t, south_t, east_t, north_t, fires_gdf)
        suppressed_with_fire = (suppressed * fire_modifier).astype(np.float32)

        print(f"  [{tiles_processed+1:2d}] {name:30s} ({grid_h}×{grid_w}) ...", end=" ", flush=True)

        # Rasterize BEC zones
        bec_grid, code2lbl = rasterize_bec_zones(bbox, grid_h, grid_w)

        if not code2lbl:
            print("no BEC data — skip")
            tiles_skipped += 1
            continue

        # Cross-tabulate
        bec_labels_in_tile = set()
        for code, lbl in code2lbl.items():
            bec_mask = bec_grid == code
            if not bec_mask.any():
                continue

            bec_labels_in_tile.add(lbl)
            s = stats[lbl]

            s["total_px"]         += int(bec_mask.sum())
            s["oldgrowth_px"]     += int(((log_grid == 5) & bec_mask).sum())
            s["oldgrowth_yew_px"] += int(((log_grid == 5) & bec_mask & (grid >= THRESHOLD)).sum())
            s["logged_cat2_px"]   += int(((log_grid == 2) & bec_mask).sum())
            s["logged_cat3_px"]   += int(((log_grid == 3) & bec_mask).sum())
            s["logged_cat4_px"]   += int(((log_grid == 4) & bec_mask).sum())
            s["logged_px"]        += int((((log_grid == 2) | (log_grid == 3) | (log_grid == 4)) & bec_mask).sum())
            s["water_px"]         += int(((log_grid == 1) & bec_mask).sum())
            s["alpine_px"]        += int(((log_grid == 6) & bec_mask).sum())
            s["nodata_px"]        += int(((log_grid == 0) & bec_mask).sum())
            s["current_yew_px"]         += int((bec_mask & (suppressed >= THRESHOLD)).sum())
            s["current_yew_fire_px"]    += int((bec_mask & (suppressed_with_fire >= THRESHOLD)).sum())
            s["fire_suppressed_yew_px"] += int(
                (bec_mask & (suppressed >= THRESHOLD) & (suppressed_with_fire < THRESHOLD)).sum()
            )
            s["raw_yew_px"]             += int((bec_mask & (grid >= THRESHOLD)).sum())

        bec_str = ", ".join(sorted(bec_labels_in_tile))
        print(f"{len(bec_labels_in_tile)} zones: {bec_str}")
        tiles_processed += 1

    print(f"\nProcessed {tiles_processed} tiles, skipped {tiles_skipped}")
    print()

    # ── Build results table ───────────────────────────────────────────────────
    rows = []
    for lbl in sorted(stats.keys()):
        s = stats[lbl]
        total        = s["total_px"]
        og           = s["oldgrowth_px"]
        og_yew       = s["oldgrowth_yew_px"]
        logged       = s["logged_px"]
        logged_2     = s["logged_cat2_px"]
        logged_3     = s["logged_cat3_px"]
        logged_4     = s["logged_cat4_px"]
        water        = s["water_px"]
        alpine       = s["alpine_px"]
        nodata       = s["nodata_px"]
        current_yew          = s["current_yew_px"]
        current_yew_fire     = s["current_yew_fire_px"]
        fire_suppressed_yew  = s["fire_suppressed_yew_px"]
        raw_yew              = s["raw_yew_px"]

        # Yew prevalence in old-growth for this BEC subzone
        yew_rate_og = og_yew / og if og > 0 else 0.0

        # Forested area = old-growth + logged (areas that were/are forest)
        forested = og + logged

        # Estimated original yew habitat = yew_rate_og × total forested area
        est_original_yew_px = yew_rate_og * forested

        # Destroyed yew = original - current remaining
        # Current remaining yew in old-growth (suppressed grid ≥ threshold in cat 5 or cat 3/4)
        est_destroyed_yew_px = est_original_yew_px - current_yew

        rows.append({
            "bec_subzone":           lbl,
            "bec_zone":              lbl[:3] if len(lbl) >= 3 else lbl,
            "total_ha":              round(total * HA_PER_PX, 1),
            "oldgrowth_ha":          round(og * HA_PER_PX, 1),
            "logged_ha":             round(logged * HA_PER_PX, 1),
            "logged_lt20yr_ha":      round(logged_2 * HA_PER_PX, 1),
            "logged_20_40yr_ha":     round(logged_3 * HA_PER_PX, 1),
            "logged_40_80yr_ha":     round(logged_4 * HA_PER_PX, 1),
            "water_ha":              round(water * HA_PER_PX, 1),
            "alpine_ha":             round(alpine * HA_PER_PX, 1),
            "yew_rate_oldgrowth":    round(yew_rate_og, 4),
            "est_original_yew_ha":   round(est_original_yew_px * HA_PER_PX, 1),
            "current_yew_ha":        round(current_yew * HA_PER_PX, 1),
            "current_yew_with_fire_ha": round(current_yew_fire * HA_PER_PX, 1),
            "fire_suppressed_ha":    round(fire_suppressed_yew * HA_PER_PX, 1),
            "destroyed_yew_ha":      round(max(0, est_destroyed_yew_px * HA_PER_PX), 1),
            "raw_model_yew_ha":      round(raw_yew * HA_PER_PX, 1),
        })

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "yew_logging_impact_by_bec.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    # ── Human-readable summary ────────────────────────────────────────────────
    txt_path = OUT_DIR / "yew_logging_impact_by_bec.txt"
    lines = []
    lines.append("=" * 90)
    lines.append("YEW LOGGING IMPACT BY BEC SUBZONE")
    lines.append(f"Threshold: P ≥ {THRESHOLD}  |  {tiles_processed} tiles  |  {HA_PER_PX} ha/pixel")
    lines.append("=" * 90)
    lines.append("")

    # ── Aggregate by BEC ZONE (e.g. CWH, CDF, MH) ───────────────────────────
    lines.append("─" * 90)
    lines.append("SUMMARY BY BEC ZONE (aggregated from subzones)")
    lines.append("─" * 90)
    lines.append(f"{'Zone':<6} {'Old-Growth':>12} {'Logged':>12} {'Yew Rate OG':>12} "
                 f"{'Orig Yew':>12} {'Current':>12} {'Destroyed':>12}")
    lines.append(f"{'':6} {'(ha)':>12} {'(ha)':>12} {'(%)':>12} "
                 f"{'(ha)':>12} {'(ha)':>12} {'(ha)':>12}")
    lines.append("-" * 90)

    zone_agg = df.groupby("bec_zone").agg({
        "total_ha":            "sum",
        "oldgrowth_ha":        "sum",
        "logged_ha":           "sum",
        "est_original_yew_ha": "sum",
        "current_yew_ha":      "sum",
        "destroyed_yew_ha":    "sum",
    }).sort_values("total_ha", ascending=False)

    # Compute weighted yew rate for zone-level
    for zone in zone_agg.index:
        z = zone_agg.loc[zone]
        # weighted yew rate from subzone data
        sub = df[df["bec_zone"] == zone]
        og_total = sub["oldgrowth_ha"].sum()
        og_yew   = (sub["yew_rate_oldgrowth"] * sub["oldgrowth_ha"]).sum()
        wt_rate  = og_yew / og_total if og_total > 0 else 0

        lines.append(f"{zone:<6} {z['oldgrowth_ha']:>12,.0f} {z['logged_ha']:>12,.0f} "
                      f"{wt_rate*100:>11.1f}% {z['est_original_yew_ha']:>12,.0f} "
                      f"{z['current_yew_ha']:>12,.0f} {z['destroyed_yew_ha']:>12,.0f}")

    totals = zone_agg.sum()
    sub_all = df
    og_all = sub_all["oldgrowth_ha"].sum()
    og_yew_all = (sub_all["yew_rate_oldgrowth"] * sub_all["oldgrowth_ha"]).sum()
    wt_rate_all = og_yew_all / og_all if og_all > 0 else 0
    lines.append("-" * 90)
    lines.append(f"{'TOTAL':<6} {totals['oldgrowth_ha']:>12,.0f} {totals['logged_ha']:>12,.0f} "
                  f"{wt_rate_all*100:>11.1f}% {totals['est_original_yew_ha']:>12,.0f} "
                  f"{totals['current_yew_ha']:>12,.0f} {totals['destroyed_yew_ha']:>12,.0f}")
    lines.append("")

    # ── Detail by subzone ─────────────────────────────────────────────────────
    lines.append("─" * 90)
    lines.append("DETAIL BY BEC SUBZONE")
    lines.append("─" * 90)
    lines.append(f"{'Subzone':<12} {'Old-Growth':>10} {'Logged':>10} {'Yew%OG':>8} "
                 f"{'Orig Yew':>10} {'Current':>10} {'Destroyed':>10} {'Logged%':>8}")
    lines.append("-" * 90)

    df_sorted = df.sort_values("total_ha", ascending=False)
    for _, row in df_sorted.iterrows():
        forested = row["oldgrowth_ha"] + row["logged_ha"]
        log_pct = (row["logged_ha"] / forested * 100) if forested > 0 else 0
        lines.append(
            f"{row['bec_subzone']:<12} "
            f"{row['oldgrowth_ha']:>10,.0f} "
            f"{row['logged_ha']:>10,.0f} "
            f"{row['yew_rate_oldgrowth']*100:>7.1f}% "
            f"{row['est_original_yew_ha']:>10,.0f} "
            f"{row['current_yew_ha']:>10,.0f} "
            f"{row['destroyed_yew_ha']:>10,.0f} "
            f"{log_pct:>7.1f}%"
        )

    lines.append("")
    lines.append("─" * 90)
    lines.append("METHODOLOGY")
    lines.append("─" * 90)
    lines.append(f"1. Raw yew probability grid (_grid.npy) from XGBoost model on 2024 embeddings")
    lines.append(f"2. VRI logging categories (_logging.npy) from VEG_COMP_LYR_R1_POLY_2024.gdb")
    lines.append(f"3. BEC zones rasterized from BEC_BIOGEOCLIMATIC_POLY.gdb per tile")
    lines.append(f"4. Yew prevalence rate = fraction of old-growth (cat 5) pixels with P ≥ {THRESHOLD}")
    lines.append(f"5. Estimated original yew = yew_rate × (old-growth + logged area)")
    lines.append(f"6. Destroyed yew = estimated original − current remaining (after suppression)")
    lines.append(f"7. Logging suppression: <20yr → ×0, 20-40yr → ×0.08, 40-80yr → ×0.50, >80yr → ×1")
    lines.append(f"8. Sample covers {tiles_processed} tiles × 10×10 km = ~{tiles_processed * 100:,} km²")
    lines.append("")

    report = "\n".join(lines)
    txt_path.write_text(report)
    print(f"Saved {txt_path}")
    print()
    print(report)


if __name__ == "__main__":
    main()
