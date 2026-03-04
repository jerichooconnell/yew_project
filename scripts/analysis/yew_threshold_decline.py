"""
yew_threshold_decline.py
========================
Standalone pipeline for estimating Pacific yew population decline using a
P ≥ THRESH presence threshold.

Steps
-----
1. Load the 10,000-point CWH/CDF sample CSV
2. Query the provincial VRI GDB to assign a logging category to every point
   (same 6-class scheme used by tile_before_after.py)
3. Zero the probability of all non-cat-5 points (VRI suppression)
4. Build counterfactual "historic" probabilities for logged points using three
   methods:
     A  Nearest geographic neighbour (same BEC subzone, haversine)
     B  BEC subzone median of unlogged pool
     C  Embedding-space cosine KNN (tile-cache points only — requires _emb.npy)
5. Apply P ≥ THRESH threshold to count "yew-present" points in current and
   historic scenarios, then extrapolate to CWH area
6. Save enriched point CSV + summary JSON

Usage
-----
    python scripts/analysis/yew_threshold_decline.py [--thresh 0.95] [--k 10]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import geopandas as gpd
from shapely.geometry import Point
from sklearn.neighbors import BallTree
from tqdm.auto import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
CSV        = ROOT / "results" / "predictions" / "cwh_xgb_sample.csv"
GDB        = str(ROOT / "data" / "VEG_COMP_LYR_R1_POLY_2024.gdb")
LAYER      = "VEG_COMP_LYR_R1_POLY"
TILE_CACHE = ROOT / "results" / "analysis" / "cwh_spot_comparisons" / "tile_cache"
SPOT_STATS = ROOT / "results" / "analysis" / "cwh_spot_comparisons" / "spot_stats.json"
OUT_DIR    = ROOT / "results" / "predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CWH_AREA_HA = 3_507_194   # total CWH+CDF land area from cwh_cdf_land.shp

# ── VRI constants ─────────────────────────────────────────────────────────
LOG_LABELS = {
    1: "water/non-forest", 2: "logged <20yr",   3: "logged 20-40yr",
    4: "logged 40-80yr",   5: "forest >80yr",   6: "alpine/barren",
}
LOG_SUPPRESS_FULL = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 1.0, 6: 0.0}
VRI_COLS = [
    "BCLCS_LEVEL_1", "BCLCS_LEVEL_2", "PROJ_AGE_1", "PROJ_AGE_CLASS_CD_1",
    "HARVEST_DATE", "LINE_7B_DISTURBANCE_HISTORY", "OPENING_IND",
    "OPENING_SOURCE", "ALPINE_DESIGNATION", "geometry",
]
_AGE_CLASS_MID = {
    "1": 10, "2": 30, "3": 50, "4": 70, "5": 90,
    "6": 110, "7": 130, "8": 195, "9": 300,
}
CURRENT_YEAR = 2025


# ── VRI helpers ───────────────────────────────────────────────────────────
def _parse_7b_min_age(val):
    if not val or not isinstance(val, str):
        return None
    min_age = None
    for seg in val.split(";"):
        parts = seg.strip().split()
        if len(parts) >= 2:
            try:
                age = CURRENT_YEAR - int(parts[1])
                if min_age is None or age < min_age:
                    min_age = age
            except ValueError:
                pass
    return min_age


def classify_vri_row(row):
    l1 = str(row.get("BCLCS_LEVEL_1") or "").strip().upper()
    l2 = str(row.get("BCLCS_LEVEL_2") or "").strip().upper()
    if l1 in ("W", "N", "") or l2 == "W":
        return 1
    alp = str(row.get("ALPINE_DESIGNATION") or "").strip().upper()
    if alp == "A" or l2 in ("A", "E", "S"):
        return 6

    age = None
    ac = str(row.get("PROJ_AGE_CLASS_CD_1") or "").strip()
    if ac in _AGE_CLASS_MID:
        age = _AGE_CLASS_MID[ac]
    a1 = row.get("PROJ_AGE_1")
    if a1 is not None and not (isinstance(a1, float) and np.isnan(a1)):
        try:
            age = int(float(a1))
        except (ValueError, TypeError):
            pass
    harv = str(row.get("HARVEST_DATE") or "").strip()
    if harv:
        try:
            ha = CURRENT_YEAR - int(harv[:4])
            if age is None or ha < age:
                age = ha
        except ValueError:
            pass
    d7b_age = _parse_7b_min_age(row.get("LINE_7B_DISTURBANCE_HISTORY"))
    if d7b_age is not None and (age is None or d7b_age < age):
        age = d7b_age

    op_src = str(row.get("OPENING_SOURCE") or "").strip().upper()
    op_ind = str(row.get("OPENING_IND") or "").strip().upper()
    is_logged = (
        op_src in ("FELLER_BUNCHER", "HANDFALL", "HARVEST", "CABLE", "PARTIAL")
        or op_ind == "Y"
    )
    if is_logged or (age is not None and age < 80):
        if age is None or age < 20:
            return 2
        elif age < 40:
            return 3
        elif age < 80:
            return 4
    return 5


def query_vri(df: pd.DataFrame) -> list:
    """Return a list of VRI log_cat for each row in df."""
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3005", always_xy=True)
    buf = 50
    cats = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="VRI GDB query"):
        x, y = transformer.transform(row["lon"], row["lat"])
        try:
            gdf = gpd.read_file(
                GDB, layer=LAYER,
                bbox=(x - buf, y - buf, x + buf, y + buf),
                columns=VRI_COLS,
            )
            if len(gdf) == 0:
                cats.append(1)
                continue
            gdf = gdf.set_crs("EPSG:3005", allow_override=True)
            within = gdf[gdf.geometry.contains(Point(x, y))]
            if len(within) == 0:
                within = gdf
            cats.append(classify_vri_row(within.iloc[0].to_dict()))
        except Exception:
            cats.append(np.nan)
    return cats


# ── Embedding helpers ──────────────────────────────────────────────────────
def _build_tile_index():
    with open(SPOT_STATS) as f:
        spot_stats = json.load(f)
    idx = {}
    for t in spot_stats:
        slug = t.get("slug") or t.get("name", "").lower().replace(" ", "_")
        emb_path = TILE_CACHE / f"{slug}_emb.npy"
        if emb_path.exists():
            idx[t.get("name", slug)] = {"slug": slug, "emb_path": emb_path}
    return idx


def _load_emb(row, tile_index, emb_cache):
    tile = tile_index.get(row.get("in_tile", ""))
    if tile is None:
        return np.zeros(64, dtype=np.float32)
    slug = tile["slug"]
    if slug not in emb_cache:
        emb_cache[slug] = np.load(tile["emb_path"], mmap_mode="r")
    try:
        return emb_cache[slug][int(row["px_row"]), int(row["px_col"]), :]
    except Exception:
        return np.zeros(64, dtype=np.float32)


def l2norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True).clip(1e-8)
    return X / n


# ── Main ───────────────────────────────────────────────────────────────────
def run(thresh: float = 0.95, k: int = 10):
    print(f"Loading sample CSV …")
    df = pd.read_csv(CSV)
    df["source"] = df["in_tile"].apply(
        lambda v: "tile_cache" if (pd.notna(v) and v != "(GEE)") else
                  "gee" if v == "(GEE)" else "unresolved"
    )
    print(f"  {len(df):,} points  ({(df.source=='tile_cache').sum()} tile-cache, "
          f"{(df.source=='gee').sum()} GEE)")

    # ── 1. VRI query ──────────────────────────────────────────────────────
    print(f"\nQuerying VRI GDB (~1 min) …")
    t0 = time.time()
    df["log_cat"]   = query_vri(df)
    df["log_label"] = df["log_cat"].map(LOG_LABELS)
    print(f"  Done in {time.time()-t0:.0f} s")
    print(df["log_label"].value_counts().to_string())

    # ── 2. Apply full VRI suppression ─────────────────────────────────────
    df["prob_raw"] = df["prob"].copy()
    df["prob"]     = df["prob_raw"] * df["log_cat"].map(LOG_SUPPRESS_FULL).fillna(0.0)
    print(f"\nLogged pixels zeroed. Raw mean={df['prob_raw'].mean():.4f}  "
          f"masked mean={df['prob'].mean():.4f}")

    # ── 3. Logged / unlogged pools ────────────────────────────────────────
    logged        = df[df["log_cat"].isin([2, 3, 4])].copy()
    unlogged_pool = df[df["log_cat"] == 5].copy()
    LOGGED_FRAC   = len(logged) / len(df)
    print(f"\nLogged (cat 2–4): {len(logged):,}  Unlogged (cat 5): {len(unlogged_pool):,}")
    print(f"Logged fraction : {LOGGED_FRAC:.1%}")

    # ── 4a. Method A: geographic KNN ─────────────────────────────────────
    print(f"\nMethod A: geographic KNN (K={k}) …")
    logged["cf_geo_prob"] = np.nan
    for bec, grp in logged.groupby("map_label"):
        pool = unlogged_pool[unlogged_pool["map_label"] == bec]
        if len(pool) == 0:
            pool = unlogged_pool[unlogged_pool["bec_zone"] == grp["bec_zone"].iloc[0]]
        if len(pool) == 0:
            pool = unlogged_pool
        tree = BallTree(np.radians(pool[["lat", "lon"]].values), metric="haversine")
        ka = min(k, len(pool))
        _, idx = tree.query(np.radians(grp[["lat", "lon"]].values), k=ka)
        logged.loc[grp.index, "cf_geo_prob"] = pool["prob"].values[idx].mean(axis=1)
    logged["loss_geo"] = np.maximum(0, logged["cf_geo_prob"] - logged["prob"])
    print(f"  Mean cf_geo = {logged['cf_geo_prob'].mean():.4f}")

    # ── 4b. Method B: BEC median ──────────────────────────────────────────
    print("Method B: BEC subzone median …")
    bec_med  = unlogged_pool.groupby("map_label")["prob"].median()
    zone_med = unlogged_pool.groupby("bec_zone")["prob"].median()
    logged["cf_bec_prob"] = logged["map_label"].map(bec_med).fillna(
        logged["bec_zone"].map(zone_med)
    )
    logged["loss_bec"] = np.maximum(0, logged["cf_bec_prob"] - logged["prob"])
    print(f"  Mean cf_bec = {logged['cf_bec_prob'].mean():.4f}")

    # ── 4c. Method C: embedding KNN (tile-cache only) ─────────────────────
    print("Method C: embedding cosine KNN (tile-cache points) …")
    tc_logged   = logged[logged["source"] == "tile_cache"].copy()
    tc_unlogged = unlogged_pool[unlogged_pool["source"] == "tile_cache"].copy()

    if len(tc_logged) > 0 and len(tc_unlogged) > 0:
        tile_index = _build_tile_index()
        emb_cache  = {}
        logged_embs   = np.vstack([_load_emb(r, tile_index, emb_cache)
                                    for _, r in tc_logged.iterrows()])
        unlogged_embs = np.vstack([_load_emb(r, tile_index, emb_cache)
                                    for _, r in tc_unlogged.iterrows()])
        unlogged_probs = tc_unlogged["prob"].values

        sims  = l2norm(logged_embs) @ l2norm(unlogged_embs).T
        ke    = min(k, len(tc_unlogged))
        top_k = np.argsort(-sims, axis=1)[:, :ke]
        cf_emb = unlogged_probs[top_k].mean(axis=1)
        logged.loc[tc_logged.index, "cf_emb_prob"] = cf_emb
        print(f"  Tile-cache logged: {len(tc_logged)}   unlogged: {len(tc_unlogged)}")
        print(f"  Mean cf_emb (tile-cache) = {cf_emb.mean():.4f}")
    else:
        logged["cf_emb_prob"] = np.nan
        print("  ⚠ Not enough tile-cache points — Method C skipped")

    logged["loss_emb"] = np.maximum(0, logged["cf_emb_prob"] - logged["prob"])

    # ── 5. P ≥ THRESH threshold analysis ─────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"P ≥ {thresh} threshold analysis")
    print(f"{'─'*60}")

    n_current   = int((df["prob"] >= thresh).sum())
    n_hist_geo  = n_current + int((logged["cf_geo_prob"] >= thresh).sum())
    n_hist_bec  = n_current + int((logged["cf_bec_prob"] >= thresh).sum())
    n_hist_emb  = (n_current
                   + int((logged["cf_emb_prob"] >= thresh).sum()))

    def pct_loss(cur, hist):
        return 100 * (hist - cur) / hist if hist > 0 else float("nan")

    print(f"Current points ≥ {thresh}: {n_current:,}  "
          f"({100*n_current/len(df):.3f}% of 10k sample)")
    print()
    print(f"{'Method':<20}  {'Historic':>10}  {'Lost':>8}  {'% decline':>10}")
    print("-" * 54)
    for mname, n_hist in [
        ("Geo-KNN (A)",    n_hist_geo),
        ("BEC median (B)", n_hist_bec),
        ("Emb-KNN (C)*",   n_hist_emb),
    ]:
        lost = n_hist - n_current
        print(f"{mname:<20}  {n_hist:>10,}  {lost:>8,}  {pct_loss(n_current, n_hist):>9.1f}%")
    print("* Method C restores only tile-cache logged points; loss is conservative")

    # Extrapolate to CWH area
    pct_curr     = n_current   / len(df)
    pct_hist_emb = n_hist_emb  / len(df)
    pct_hist_geo = n_hist_geo  / len(df)
    pct_hist_bec = n_hist_bec  / len(df)

    print(f"\nExtrapolated to CWH+CDF ({CWH_AREA_HA:,} ha):")
    print(f"  Current  yew-present area  : {pct_curr     * CWH_AREA_HA:>12,.0f} ha")
    print(f"  Historic area (Geo-KNN A)  : {pct_hist_geo * CWH_AREA_HA:>12,.0f} ha  "
          f"(−{pct_loss(n_current, n_hist_geo):.1f}%)")
    print(f"  Historic area (BEC med  B) : {pct_hist_bec * CWH_AREA_HA:>12,.0f} ha  "
          f"(−{pct_loss(n_current, n_hist_bec):.1f}%)")
    print(f"  Historic area (Emb-KNN C*) : {pct_hist_emb * CWH_AREA_HA:>12,.0f} ha  "
          f"(−{pct_loss(n_current, n_hist_emb):.1f}%)")

    # ── 6. Save ───────────────────────────────────────────────────────────
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Merge counterfactual columns back into df
    cf_cols = ["cf_geo_prob", "cf_bec_prob", "cf_emb_prob",
               "loss_geo", "loss_bec", "loss_emb"]
    for col in cf_cols:
        df[col] = np.nan
    for col in cf_cols:
        df.loc[logged.index, col] = logged[col].values

    enriched_path = OUT_DIR / f"cwh_xgb_enriched_{ts}.csv"
    df.to_csv(enriched_path, index=False)
    print(f"\nSaved enriched CSV → {enriched_path}")

    summary = {
        "threshold": thresh,
        "k": k,
        "n_sample": len(df),
        "n_logged": len(logged),
        "n_unlogged": len(unlogged_pool),
        "logged_frac": round(LOGGED_FRAC, 4),
        "cwh_area_ha": CWH_AREA_HA,
        "current_above_thresh": n_current,
        "current_area_ha": round(pct_curr * CWH_AREA_HA),
        "methods": {
            "geo_knn_A": {
                "historic_count": n_hist_geo,
                "lost_count": n_hist_geo - n_current,
                "pct_decline": round(pct_loss(n_current, n_hist_geo), 2),
                "historic_area_ha": round(pct_hist_geo * CWH_AREA_HA),
            },
            "bec_median_B": {
                "historic_count": n_hist_bec,
                "lost_count": n_hist_bec - n_current,
                "pct_decline": round(pct_loss(n_current, n_hist_bec), 2),
                "historic_area_ha": round(pct_hist_bec * CWH_AREA_HA),
            },
            "emb_knn_C": {
                "historic_count": n_hist_emb,
                "lost_count": n_hist_emb - n_current,
                "pct_decline": round(pct_loss(n_current, n_hist_emb), 2),
                "historic_area_ha": round(pct_hist_emb * CWH_AREA_HA),
                "note": "tile-cache logged points only — conservative",
            },
        },
    }
    summary_path = OUT_DIR / f"cwh_threshold_decline_{ts}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON → {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yew threshold-based population decline estimate")
    parser.add_argument("--thresh", type=float, default=0.95,
                        help="Presence threshold (default: 0.95)")
    parser.add_argument("--k", type=int, default=10,
                        help="KNN neighbours for Methods A and C (default: 10)")
    args = parser.parse_args()
    run(thresh=args.thresh, k=args.k)
