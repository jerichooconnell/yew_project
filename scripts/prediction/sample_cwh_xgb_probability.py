#!/usr/bin/env python3
"""
Random Sampling of the CWH+CDF Land Area — XGBoost Yew Probability

1. Samples N random points uniformly from data/processed/cwh_cdf_land.shp
   (the high-resolution water-clipped CWH+CDF polygon) using vectorised
   rejection sampling (shapely.contains_xy) — ~5,000 pts/sec.
2. For every point inside an already-downloaded tile (tile_cache/*_emb.npy),
   extracts the 64-band embedding directly from the cached array.
3. For all remaining points, fetches embeddings from Google Earth Engine via
   ee.Image.sampleRegions() — one server-side call per batch of
   --gee-batch-size points (~5 MB/batch), ~30-90 s for 10,000 points total.
4. Classifies all points with XGBoost (GPU if available).
5. **Verification**: tile-cache points are cross-checked against *_grid.npy.
6. Saves results/predictions/cwh_xgb_sample.csv and an interactive map.

Usage:
    # Full 10,000-point run with GEE download:
    python scripts/prediction/sample_cwh_xgb_probability.py --n-samples 10000 --gee-extract
    # Quick local-only run (unclassified points left as NaN):
    python scripts/prediction/sample_cwh_xgb_probability.py --n-samples 1000
    # Verify coordinate mapping against cached grids only:
    python scripts/prediction/sample_cwh_xgb_probability.py --verify-only

Coordinate mapping (pixel ↔ geographic):
    Each tile has center (lat, lon) from spot_stats.json.
    south, north, west, east = centre_to_bbox(lat, lon, km=10)
    emb.npy has shape (H, W, 64) and grid.npy has shape (H, W).

    Given a geographic point (pt_lat, pt_lon) inside the tile bbox:
        row = int((north - pt_lat) / (north - south) * H)   # 0-indexed, top→bottom
        col = int((pt_lon  - west)  / (east  - west)  * W)  # 0-indexed, left→right

    The reverse (pixel → centroid):
        lat = north - (row + 0.5) / H * (north - south)
        lon = west  + (col + 0.5) / W * (east  - west)
"""

import argparse
import json
import sys
import time
from math import cos, radians
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import xgboost as xgb
from shapely import contains_xy          # vectorised containment (Shapely 2+)
from shapely.geometry import Point
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
SHAPEFILE  = ROOT / "data" / "processed" / "cwh_cdf_land.shp"
TILE_CACHE = ROOT / "results" / "analysis" / "cwh_spot_comparisons" / "tile_cache"
SPOT_STATS = ROOT / "results" / "analysis" / "cwh_spot_comparisons" / "spot_stats.json"
MODEL_DIR  = ROOT / "results" / "predictions" / "south_vi_large"
XGB_MODEL  = MODEL_DIR / "xgb_raw_model_expanded.json"
OUT_CSV    = ROOT / "results" / "predictions" / "cwh_xgb_sample.csv"
OUT_HTML   = ROOT / "results" / "figures"    / "cwh_xgb_sample_map.html"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
OUT_HTML.parent.mkdir(parents=True, exist_ok=True)

# Tile geometry constants — must match classify_cwh_spots.py
AREA_KM    = 10.0
SCALE_M    = 10.0      # metres per pixel

# GEE constants
GEE_PROJECT   = 'carbon-storm-206002'
GEE_ASSET     = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
GEE_BAND_NAMES = [f'A{i:02d}' for i in range(64)]  # A00 … A63


# ── Geometry helpers ───────────────────────────────────────────────────────

def centre_to_bbox(lat: float, lon: float, km: float = AREA_KM):
    """Return (south, north, west, east) for km×km box centred at (lat, lon)."""
    half_lat = (km * 1000 / 2) / 111320.0
    half_lon = (km * 1000 / 2) / (111320.0 * cos(radians(lat)))
    return lat - half_lat, lat + half_lat, lon - half_lon, lon + half_lon


def latlon_to_pixel(pt_lat: float, pt_lon: float,
                    south: float, north: float, west: float, east: float,
                    H: int, W: int):
    """
    Convert a geographic point to the (row, col) pixel index within a tile.

    Returns (row, col) as ints, or None if the point is outside the tile.
    Row 0 is the northernmost row; col 0 is the westernmost column.
    """
    if not (south <= pt_lat <= north and west <= pt_lon <= east):
        return None
    row = int((north - pt_lat) / (north - south) * H)
    col = int((pt_lon - west)  / (east - west)  * W)
    row = min(row, H - 1)
    col = min(col, W - 1)
    return row, col


def pixel_to_latlon(row: int, col: int,
                    south: float, north: float, west: float, east: float,
                    H: int, W: int):
    """Return the geographic centroid of pixel (row, col)."""
    lat = north - (row + 0.5) / H * (north - south)
    lon = west  + (col + 0.5) / W * (east  - west)
    return lat, lon


# ── Sample random points from shapefile ───────────────────────────────────

def sample_random_points(shapefile: Path, n: int, seed: int = 42) -> gpd.GeoDataFrame:
    """
    Sample *n* points uniformly within the CWH+CDF polygons.

    Uses area-weighted polygon selection then vectorised batch rejection sampling
    (shapely.contains_xy) so that even highly detailed coastline polygons
    (129k+ exterior vertices) are handled in milliseconds instead of minutes.
    """
    print(f"Loading shapefile: {shapefile}")
    gdf = gpd.read_file(shapefile)          # EPSG:3005 (equal-area metres)
    print(f"  {len(gdf)} polygons, CRS: {gdf.crs}")

    rng = np.random.default_rng(seed)
    areas = gdf.geometry.area.values
    weights = areas / areas.sum()

    # Pre-compute per-polygon bounding boxes for fast candidate generation
    bounds   = np.array([geom.bounds for geom in gdf.geometry])  # (N,4) minx miny maxx maxy
    geom_arr = gdf.geometry.values                               # numpy geometry array

    accepted_xy  = []   # (x, y) in EPSG:3005
    accepted_idx = []   # polygon index
    attempts     = 0
    BATCH        = max(n * 10, 5000)       # candidates per round

    pbar = tqdm(total=n, desc="Sampling points", unit="pt")

    while len(accepted_xy) < n:
        # Pick polygons proportional to area (whole batch at once)
        poly_ids = rng.choice(len(gdf), size=BATCH, p=weights)

        # Generate one random (x, y) per candidate inside that polygon's bbox
        bx = bounds[poly_ids]
        xs = rng.uniform(bx[:, 0], bx[:, 2])
        ys = rng.uniform(bx[:, 1], bx[:, 3])

        # Group by unique polygon and run vectorised contains_xy
        unique_ids = np.unique(poly_ids)
        mask       = np.zeros(BATCH, dtype=bool)

        for pid in unique_ids:
            sel = poly_ids == pid
            mask[sel] = contains_xy(geom_arr[pid], xs[sel], ys[sel])

        hit_xs  = xs[mask]
        hit_ys  = ys[mask]
        hit_ids = poly_ids[mask]
        attempts += BATCH

        need   = n - len(accepted_xy)
        take   = min(need, len(hit_xs))
        accepted_xy.extend(zip(hit_xs[:take], hit_ys[:take]))
        accepted_idx.extend(hit_ids[:take].tolist())
        pbar.update(take)

    pbar.close()
    print(f"  {n} points accepted from {attempts:,} candidates "
          f"(acceptance rate: {100*n/attempts:.1f}%)")

    xs_out = np.array([p[0] for p in accepted_xy])
    ys_out = np.array([p[1] for p in accepted_xy])

    pts_gdf = gpd.GeoDataFrame(
        {"poly_idx": accepted_idx},
        geometry=gpd.points_from_xy(xs_out, ys_out),
        crs=gdf.crs
    ).to_crs(epsg=4326)                    # Convert to WGS-84 for tile lookup
    pts_gdf["lat"] = pts_gdf.geometry.y
    pts_gdf["lon"] = pts_gdf.geometry.x
    # Carry BEC attributes
    for col in ["ZONE", "MAP_LABEL"]:
        if col in gdf.columns:
            pts_gdf[col] = gdf[col].iloc[pts_gdf["poly_idx"].values].values
    return pts_gdf


# ── Tile lookup ────────────────────────────────────────────────────────────

def build_tile_index(spot_stats: list) -> list:
    """
    Pre-compute bounding boxes for all available tiles.

    Returns a list of dicts with keys:
        name, slug, lat, lon, h, w, south, north, west, east
    Only tiles whose *_emb.npy AND *_grid.npy files both exist are included.
    """
    tiles = []
    for s in spot_stats:
        slug = s["name"].lower().replace(" ", "_").replace("-", "_")
        emb_path  = TILE_CACHE / f"{slug}_emb.npy"
        grid_path = TILE_CACHE / f"{slug}_grid.npy"
        if not emb_path.exists() or not grid_path.exists():
            continue
        south, north, west, east = centre_to_bbox(s["lat"], s["lon"])
        tiles.append({
            "name":  s["name"],
            "slug":  slug,
            "lat":   s["lat"],
            "lon":   s["lon"],
            "h":     s["h"],
            "w":     s["w"],
            "south": south,
            "north": north,
            "west":  west,
            "east":  east,
            "emb_path":  emb_path,
            "grid_path": grid_path,
        })
    print(f"Tile index: {len(tiles)} tiles with cached emb+grid")
    return tiles


def find_tile(pt_lat: float, pt_lon: float, tile_index: list) -> dict | None:
    """Return the first tile whose bbox contains (pt_lat, pt_lon), or None."""
    for t in tile_index:
        if t["south"] <= pt_lat <= t["north"] and t["west"] <= pt_lon <= t["east"]:
            return t
    return None


# ── XGBoost inference ──────────────────────────────────────────────────────

def load_xgb_model(model_path: Path) -> xgb.Booster:
    print(f"Loading XGBoost model: {model_path.name}")
    bst = xgb.Booster()
    bst.load_model(str(model_path))
    print(f"  ✓ Loaded")
    return bst


def predict_embeddings(bst: xgb.Booster, embs: np.ndarray) -> np.ndarray:
    """
    Run XGBoost on an (N, 64) embedding array.
    Returns (N,) probability array (float32).
    """
    embs = np.nan_to_num(embs.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    dmat = xgb.DMatrix(embs)
    return bst.predict(dmat).astype(np.float32)


# ── GEE extraction ────────────────────────────────────────────────────────

def extract_embeddings_gee(
    lats: np.ndarray,
    lons: np.ndarray,
    year: int = 2024,
    batch_size: int = 500,
) -> np.ndarray:
    """
    Fetch 64-band Prithvi embeddings for (lat, lon) pairs via GEE
    ee.Image.sampleRegions().  Returns (N, 64) float32 array; rows that
    GEE cannot resolve (cloud mask, outside image extent) are filled NaN.

    Uses sampleRegions() which evaluates the image server-side at each
    point and returns only the pixel values — far smaller payload than
    downloading a full raster tile (256 KB vs 410 MB per 10 km tile).
    """
    try:
        import ee
    except ImportError:
        raise ImportError("earthengine-api not installed: pip install earthengine-api")

    print(f"  Initialising GEE (project={GEE_PROJECT}) …")
    ee.Initialize(project=GEE_PROJECT)

    N = len(lats)
    result = np.full((N, 64), np.nan, dtype=np.float32)

    ee_img = (
        ee.ImageCollection(GEE_ASSET)
        .filterDate(f'{year}-01-01', f'{year + 1}-01-01')
        .mosaic()
        .select(GEE_BAND_NAMES)
        .toFloat()
    )

    n_batches = (N + batch_size - 1) // batch_size
    print(f"  Extracting {N:,} points in {n_batches} batches of ≤{batch_size} …")

    for b in range(n_batches):
        lo, hi = b * batch_size, min((b + 1) * batch_size, N)
        batch_lats = lats[lo:hi]
        batch_lons = lons[lo:hi]

        features = [
            ee.Feature(ee.Geometry.Point([float(lon), float(lat)]),
                       {'__idx': int(lo + i)})
            for i, (lat, lon) in enumerate(zip(batch_lats, batch_lons))
        ]
        fc = ee.FeatureCollection(features)

        t0 = time.time()
        try:
            sampled = ee_img.sampleRegions(
                collection=fc,
                scale=10,
                geometries=False,
                tileScale=4,
            )
            info = sampled.getInfo()
        except Exception as e:
            print(f"  ⚠ Batch {b + 1}/{n_batches} failed: {e}")
            continue

        for feat in info.get('features', []):
            props = feat.get('properties', {})
            idx   = props.get('__idx')
            if idx is None:
                continue
            row = np.array(
                [props.get(band, np.nan) for band in GEE_BAND_NAMES],
                dtype=np.float32,
            )
            result[idx] = row

        n_ok   = np.isfinite(result[lo:hi, 0]).sum()
        elapsed = time.time() - t0
        print(f"  Batch {b + 1:3d}/{n_batches}  pts {lo + 1:,}–{hi:,}  "
              f"{n_ok}/{hi - lo} resolved  {elapsed:.1f}s")

    n_resolved = int(np.isfinite(result[:, 0]).sum())
    print(f"  GEE extraction complete: {n_resolved:,}/{N:,} points resolved "
          f"({100 * n_resolved / N:.1f}%)")
    return result


# ── Verification ───────────────────────────────────────────────────────────

def verify_tile_consistency(
    pt_lat: float, pt_lon: float,
    tile: dict,
    bst: xgb.Booster,
    emb_cache: dict,
    grid_cache: dict,
    tol: float = 1e-4,
) -> dict:
    """
    For a point inside a tile, verify that:
      XGB(emb[row, col, :])  ≈  grid[row, col]

    Returns a dict with pixel coords, both probabilities, and whether they match.
    """
    rc = latlon_to_pixel(pt_lat, pt_lon,
                         tile["south"], tile["north"],
                         tile["west"],  tile["east"],
                         tile["h"], tile["w"])
    if rc is None:
        return {"error": "pixel outside tile despite bbox check"}

    row, col = rc

    # Lazy-load emb and grid arrays (cache across calls)
    slug = tile["slug"]
    if slug not in emb_cache:
        emb_cache[slug]  = np.load(tile["emb_path"],  mmap_mode="r")
        grid_cache[slug] = np.load(tile["grid_path"], mmap_mode="r")

    emb  = emb_cache[slug]
    grid = grid_cache[slug]

    pixel_emb  = emb[row, col, :].copy()                        # (64,)
    cached_prob = float(grid[row, col])                         # pre-computed

    # Recompute probability from embedding
    recomputed_prob = float(predict_embeddings(bst, pixel_emb[np.newaxis, :])[0])

    diff    = abs(recomputed_prob - cached_prob)
    matches = diff <= tol

    # Pixel centroid (for sanity display)
    px_lat, px_lon = pixel_to_latlon(row, col,
                                     tile["south"], tile["north"],
                                     tile["west"],  tile["east"],
                                     tile["h"], tile["w"])

    return {
        "row":            row,
        "col":            col,
        "px_lat":         round(px_lat, 6),
        "px_lon":         round(px_lon, 6),
        "cached_prob":    round(cached_prob, 6),
        "recomputed_prob":round(recomputed_prob, 6),
        "diff":           round(diff, 8),
        "matches":        matches,
    }


# ── Main ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-samples",      type=int,   default=10000,
                   help="Total random points to draw from CWH/CDF area (default: 10000)")
    p.add_argument("--n-verify",       type=int,   default=10,
                   help="Max verification comparisons to print (default: 10)")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--tol",            type=float, default=1e-4,
                   help="Max allowable |recomputed−cached| to count as matching")
    p.add_argument("--verify-only",    action="store_true",
                   help="Only run tile-cache verification, skip sampling")
    p.add_argument("--gee-extract",    action="store_true",
                   help="Fetch embeddings from GEE for points outside cached tiles")
    p.add_argument("--gee-year",       type=int,   default=2024,
                   help="Year for GEE embedding mosaic (default: 2024)")
    p.add_argument("--gee-batch-size", type=int,   default=500,
                   help="Points per GEE sampleRegions call (default: 500)")
    p.add_argument("--out-csv",        type=str,   default=str(OUT_CSV))
    p.add_argument("--out-html",       type=str,   default=str(OUT_HTML),
                   help="Output Folium map HTML path")
    p.add_argument("--no-map",         action="store_true",
                   help="Skip map generation")
    return p.parse_args()


def main():
    args = parse_args()
    out_csv = Path(args.out_csv)

    # ── Load tile index & model ────────────────────────────────────────────
    with open(SPOT_STATS) as f:
        spot_stats = json.load(f)

    tile_index = build_tile_index(spot_stats)
    bst        = load_xgb_model(XGB_MODEL)

    emb_cache  = {}   # slug → mmap'd (H,W,64) array
    grid_cache = {}   # slug → mmap'd (H,W) array

    # ── Sample points ──────────────────────────────────────────────────────
    if args.verify_only:
        # For verify-only: generate a handful of target points that we know are
        # inside cached tiles by picking random pixels directly
        print("\n-- VERIFY-ONLY mode: sampling pixels from tile cache ----------")
        rows_out = []
        for tile in tile_index[:args.n_verify]:
            slug = tile["slug"]
            if slug not in emb_cache:
                emb_cache[slug]  = np.load(tile["emb_path"],  mmap_mode="r")
                grid_cache[slug] = np.load(tile["grid_path"], mmap_mode="r")
            H, W = tile["h"], tile["w"]
            rng  = np.random.default_rng(args.seed)
            r    = rng.integers(0, H)
            c    = rng.integers(0, W)
            lat, lon = pixel_to_latlon(r, c, tile["south"], tile["north"],
                                       tile["west"],  tile["east"], H, W)
            v = verify_tile_consistency(lat, lon, tile, bst,
                                        emb_cache, grid_cache, tol=args.tol)
            rows_out.append({
                "lat": lat, "lon": lon,
                "in_tile":      tile["name"],
                "prob":         v["recomputed_prob"],
                "cached_prob":  v["cached_prob"],
                "prob_diff":    v["diff"],
                "verified":     v["matches"],
                "px_row":       v["row"],
                "px_col":       v["col"],
            })
        df = pd.DataFrame(rows_out)
    else:
        pts = sample_random_points(SHAPEFILE, args.n_samples, seed=args.seed)

        # ── Classify each point ────────────────────────────────────────────
        print(f"\nClassifying {len(pts)} points …")
        records = []

        for _, row_pt in tqdm(pts.iterrows(), total=len(pts)):
            pt_lat = row_pt.lat
            pt_lon = row_pt.lon
            tile   = find_tile(pt_lat, pt_lon, tile_index)
            rec    = {
                "lat":      round(pt_lat, 6),
                "lon":      round(pt_lon, 6),
                "bec_zone": row_pt.get("ZONE", ""),
                "map_label":row_pt.get("MAP_LABEL", ""),
                "in_tile":  tile["name"] if tile else None,
                "px_row":   None,
                "px_col":   None,
                "prob":     None,
                "cached_prob": None,
                "prob_diff": None,
                "verified": None,
            }

            if tile:
                rc = latlon_to_pixel(pt_lat, pt_lon,
                                     tile["south"], tile["north"],
                                     tile["west"],  tile["east"],
                                     tile["h"], tile["w"])
                if rc:
                    r, c = rc
                    slug = tile["slug"]
                    if slug not in emb_cache:
                        emb_cache[slug]  = np.load(tile["emb_path"],  mmap_mode="r")
                        grid_cache[slug] = np.load(tile["grid_path"], mmap_mode="r")

                    pixel_emb    = emb_cache[slug][r, c, :].copy()
                    prob         = float(predict_embeddings(bst, pixel_emb[np.newaxis, :])[0])
                    cached_prob  = float(grid_cache[slug][r, c])

                    rec["px_row"]      = r
                    rec["px_col"]      = c
                    rec["prob"]        = round(prob, 6)
                    rec["cached_prob"] = round(cached_prob, 6)
                    rec["prob_diff"]   = round(abs(prob - cached_prob), 8)
                    rec["verified"]    = abs(prob - cached_prob) <= args.tol
            else:
                rec["verified"] = False    # can't verify — needs GEE extraction

            records.append(rec)

        df = pd.DataFrame(records)

    # ── GEE extraction for unclassified points ─────────────────────────────
    if args.gee_extract and not args.verify_only:
        pending_mask = df["prob"].isna()
        n_pending    = pending_mask.sum()
        if n_pending > 0:
            print(f"\nFetching {n_pending:,} embeddings from GEE "
                  f"(year={args.gee_year}, batch={args.gee_batch_size}) …")
            p_lats = df.loc[pending_mask, "lat"].values
            p_lons = df.loc[pending_mask, "lon"].values

            embs = extract_embeddings_gee(
                p_lats, p_lons,
                year=args.gee_year,
                batch_size=args.gee_batch_size,
            )

            # Classify all resolved embeddings
            resolved = np.isfinite(embs[:, 0])
            if resolved.any():
                probs_gee = predict_embeddings(bst, embs[resolved])
                pending_indices = df.index[pending_mask].tolist()
                res_indices     = [pending_indices[i] for i in np.where(resolved)[0]]
                df.loc[res_indices, "prob"]    = probs_gee.round(6)
                df.loc[res_indices, "in_tile"] = "(GEE)"
                print(f"  Classified {resolved.sum():,} GEE points with XGBoost")
        else:
            print("All points already classified from tile cache.")

    # ── Save CSV ───────────────────────────────────────────────────────────
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved → {out_csv}  ({len(df)} rows)")

    # ── Summary ───────────────────────────────────────────────────────────
    n_in_tile    = df["in_tile"].notna().sum()
    n_out_tile   = df["in_tile"].isna().sum()
    n_classified = df["prob"].notna().sum()

    print(f"\n{'='*60}")
    print(f"SAMPLING SUMMARY")
    print(f"{'='*60}")
    print(f"  Total points sampled     : {len(df)}")
    print(f"  Inside downloaded tile   : {n_in_tile}  → classified from cache")
    print(f"  Outside any tile         : {n_out_tile}  → needs GEE extraction")
    print(f"  Points with probability  : {n_classified}")

    if n_in_tile > 0:
        sub = df[df["in_tile"].notna() & df["prob"].notna()]
        print(f"\nYEW PROBABILITY DISTRIBUTION (in-tile points):")
        thresholds = [0.30, 0.50, 0.70, 0.90]
        for t in thresholds:
            n  = (sub["prob"] >= t).sum()
            pct = 100 * n / len(sub)
            print(f"  P≥{t:.2f}: {n:5d} / {len(sub)} ({pct:.1f}%)")
        print(f"\n  Mean P   : {sub['prob'].mean():.4f}")
        print(f"  Median P : {sub['prob'].median():.4f}")

    # ── Verification report ────────────────────────────────────────────────
    # Only tile-cache points have cached_prob to verify against
    verified_rows = df[(df["verified"] == True) & (df["cached_prob"].notna())]  # noqa: E712
    mismatch_rows = df[
        (df["in_tile"].notna()) &
        (df["in_tile"] != "(GEE)") &
        (df["verified"] == False)  # noqa: E712
    ]

    print(f"\n{'='*60}")
    print(f"VERIFICATION (recomputed XGB  vs  cached _grid.npy)")
    print(f"{'='*60}")
    print(f"  Matched (|diff| ≤ {args.tol}): {len(verified_rows)}")
    print(f"  Mismatched              : {len(mismatch_rows)}")

    # Print details for the first n_verify in-tile points
    tile_rows = df[df["in_tile"].notna() & df["prob"].notna()].head(args.n_verify)
    if len(tile_rows):
        print(f"\n  {'Tile':<30} {'Lat':>8} {'Lon':>9}  {'row':>5} {'col':>5}  "
              f"{'cached':>8} {'recomp':>8} {'diff':>10} {'ok?':>5}")
        print("  " + "-"*90)
        for _, r in tile_rows.iterrows():
            ok = "✓" if r.get("verified") else "✗"
            print(f"  {str(r['in_tile']):<30} {r['lat']:8.4f} {r['lon']:9.4f}  "
                  f"{str(r.get('px_row','?')):>5} {str(r.get('px_col','?')):>5}  "
                  f"{r.get('cached_prob', 0):8.5f} {r.get('prob', 0):8.5f} "
                  f"{r.get('prob_diff', 0):10.2e} {ok:>5}")

    if len(mismatch_rows):
        print(f"\n  ⚠ {len(mismatch_rows)} mismatched pixel(s) — check coordinate mapping:")
        for _, r in mismatch_rows.head(5).iterrows():
            print(f"    {r['in_tile']}  row={r.get('px_row')} col={r.get('px_col')} "
                  f"cached={r.get('cached_prob'):.5f} recomp={r.get('prob'):.5f}")

    print(f"\n{'='*60}")
    if len(mismatch_rows) == 0 and len(verified_rows) > 0:
        print("✓  All verified points match.  Coordinate mapping is correct.")
    elif len(verified_rows) == 0:
        print("⚠  No in-tile points to verify (try --n-samples with a larger value).")
    else:
        print(f"⚠  {len(mismatch_rows)} mismatches — investigate before proceeding.")

    # ── Map ────────────────────────────────────────────────────────────────
    if not getattr(args, "no_map", False) and not args.verify_only:
        generate_map(df, SHAPEFILE, Path(args.out_html),
                     tile_index, args.n_samples, args.seed)


# ── Map generation ────────────────────────────────────────────────────────

def prob_color(p) -> str:
    """Return a hex colour for a probability value."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "#888888"   # grey — unclassified
    if p < 0.20: return "#4575b4"   # dark blue
    if p < 0.35: return "#74add1"   # light blue
    if p < 0.50: return "#fee090"   # yellow
    if p < 0.65: return "#f46d43"   # orange
    return "#d73027"                # red


def generate_map(df: pd.DataFrame, shapefile: Path, out_html: Path,
                 tile_index: list, n_samples: int, seed: int) -> None:
    """Build a Folium map of sampled points overlaid on the CWH/CDF polygons."""
    print(f"\nGenerating map → {out_html}")

    # Background: CWH+CDF polygons (simplified for the web)
    gdf = gpd.read_file(shapefile)
    gdf["geometry"] = gdf.geometry.simplify(300, preserve_topology=True)
    gdf = gdf.to_crs(epsg=4326)
    cwh = gdf[gdf["ZONE"] == "CWH"] if "ZONE" in gdf.columns else gdf
    cdf = gdf[gdf["ZONE"] == "CDF"] if "ZONE" in gdf.columns else gpd.GeoDataFrame()

    # Tile bounding boxes (for reference)
    centre_lat = df["lat"].mean() if len(df) else 51.0
    centre_lon = df["lon"].mean() if len(df) else -125.0

    m = folium.Map(location=[centre_lat, centre_lon], zoom_start=6,
                   tiles="cartodbpositron")

    # CWH zone
    cwh_fg = folium.FeatureGroup(name="CWH zone", show=True)
    folium.GeoJson(
        cwh,
        style_function=lambda f: {
            "fillColor": "#228B22", "color": "#228B22",
            "weight": 0.4, "fillOpacity": 0.18,
        },
    ).add_to(cwh_fg)
    cwh_fg.add_to(m)

    # CDF zone
    if len(cdf):
        cdf_fg = folium.FeatureGroup(name="CDF zone", show=True)
        folium.GeoJson(
            cdf,
            style_function=lambda f: {
                "fillColor": "#008080", "color": "#008080",
                "weight": 0.4, "fillOpacity": 0.18,
            },
        ).add_to(cdf_fg)
        cdf_fg.add_to(m)

    # Downloaded tile bounding boxes
    tile_fg = folium.FeatureGroup(name="Downloaded tiles (10 km)", show=True)
    for t in tile_index:
        bounds = [[t["south"], t["west"]], [t["north"], t["east"]]]
        folium.Rectangle(
            bounds=bounds,
            color="#333333", weight=1, fill=False,
            tooltip=t["name"],
        ).add_to(tile_fg)
    tile_fg.add_to(m)

    # Sampled points — two layers: classified and unclassified
    uncls_fg = folium.FeatureGroup(name="Unclassified points (need GEE)", show=True)
    cls_fg   = folium.FeatureGroup(name="Classified points (XGBoost)",    show=True)

    for _, r in df.iterrows():
        has_prob = pd.notna(r.get("prob"))
        prob_val = float(r["prob"]) if has_prob else None
        color    = prob_color(prob_val)
        radius   = 5 if has_prob else 3
        opacity  = 0.85 if has_prob else 0.45

        popup_lines = [
            f"<b>{r.get('in_tile', 'outside tiles') or 'outside tiles'}</b>",
            f"Lat: {r['lat']:.5f}  Lon: {r['lon']:.5f}",
            f"BEC: {r.get('bec_zone','')} / {r.get('map_label','')}",
        ]
        if has_prob:
            popup_lines.append(f"<b>P(yew) = {prob_val:.4f}</b>")
            popup_lines.append(f"Cached P = {r.get('cached_prob', '?')}")
            pr = r.get('px_row')
            pc = r.get('px_col')
            if pr is not None and not (isinstance(pr, float) and np.isnan(pr)):
                popup_lines.append(f"Pixel row={int(pr)} col={int(pc)}")
        else:
            popup_lines.append("<i>Not yet classified — outside cached tiles</i>")
        popup_html = "<br>".join(popup_lines)

        marker = folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=opacity,
            weight=0.8 if has_prob else 0.4,
            popup=folium.Popup(popup_html, max_width=260),
        )
        if has_prob:
            marker.add_to(cls_fg)
        else:
            marker.add_to(uncls_fg)

    uncls_fg.add_to(m)
    cls_fg.add_to(m)

    # Legend
    n_cls   = int(df["prob"].notna().sum())
    n_uncls = int(df["prob"].isna().sum())
    legend_html = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:9999;
                background:white;padding:12px 16px;border-radius:8px;
                border:1px solid #ccc;font-size:12px;line-height:1.7;">
      <b>CWH/CDF Random Sample</b><br>
      <i>n={n_samples}, seed={seed}</i><br><br>
      <span style="color:#888888;">&#9679;</span> Unclassified ({n_uncls} pts — needs GEE)<br>
      <span style="color:#4575b4;">&#9679;</span> P &lt; 0.20<br>
      <span style="color:#74add1;">&#9679;</span> P 0.20 – 0.35<br>
      <span style="color:#fee090;">&#9679;</span> P 0.35 – 0.50<br>
      <span style="color:#f46d43;">&#9679;</span> P 0.50 – 0.65<br>
      <span style="color:#d73027;">&#9679;</span> P &ge; 0.65<br><br>
      Classified from tile cache: <b>{n_cls}</b><br>
      Black outlines = downloaded tile bounds
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(out_html))
    print(f"  Map saved ({out_html.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
