#!/usr/bin/env python3
"""
P0.2 (extended) — Tile-independent point validation of the yew habitat surface.

The original map_validation_boyce.py can only score presence/absence points that
fall inside one of the 98 rendered study tiles, so it matched just ~10 points and
the Boyce index was uncomputable. This script removes the tile-footprint
constraint by sampling the AlphaEarth embedding *directly at every point* from GEE,
classifying it with the production XGBoost model, and applying the same VRI
suppression the published map uses (via the local VEG_COMP GDB, point-in-polygon).
Every presence/absence point therefore yields a probability, tile or no tile.

Two surfaces are validated:
  - RAW        : production classifier probability (validates the habitat model)
  - SUPPRESSED : raw × VRI suppression factor (validates the published map surface)

Point sets:
  - FAIB PSP (independent — never used in classifier training):
        presence = sites with a TW (Pacific yew) record
        absence  = CWH/ICH/CDF sites with no TW record (sampled)
  - iNat held-out val positives are added in the full run (not this pilot).

Embeddings and VRI categories are cached under data/processed/point_validation/
so re-runs do not re-hit GEE or re-read the GDB.

Run (pilot, FAIB only):
    /home/jericho/anaconda3/envs/yew_pytorch/bin/python scripts/analysis/point_map_validation.py
"""
import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("/home/jericho/yew_project")
sys.path.insert(0, str(ROOT / "scripts/prediction"))
import classify_cwh_spots as C   # noqa: E402  (VRI classify + suppression, source of truth)

CACHE_DIR = ROOT / "data/processed/point_validation"
OUT_JSON  = ROOT / "results/analysis/point_map_validation.json"
MODEL     = ROOT / "results/predictions/south_vi_large/xgb_raw_model_expanded.json"
YEW_ZONES = {"CWH", "ICH", "CDF"}
VRI_COLS  = ['BCLCS_LEVEL_1', 'BCLCS_LEVEL_2', 'PROJ_AGE_1', 'PROJ_AGE_CLASS_CD_1',
             'HARVEST_DATE', 'LINE_7B_DISTURBANCE_HISTORY', 'OPENING_IND',
             'OPENING_SOURCE', 'ALPINE_DESIGNATION', 'geometry']


# ── Build the unified presence / absence point set (FAIB + optional iNat) ──────
def build_points(n_absence, seed, include_inat):
    td = pd.read_csv(ROOT / "data/raw/faib_tree_detail.csv",
                     usecols=["SITE_IDENTIFIER", "SPECIES"], low_memory=False)
    hdr = pd.read_csv(ROOT / "data/raw/faib_header.csv", low_memory=False)
    hdr["Latitude"]  = pd.to_numeric(hdr["Latitude"], errors="coerce")
    hdr["Longitude"] = pd.to_numeric(hdr["Longitude"], errors="coerce")
    hdr = hdr.dropna(subset=["Latitude", "Longitude"]).drop_duplicates("SITE_IDENTIFIER")
    hdr = hdr[hdr.BEC_ZONE.isin(YEW_ZONES)].copy()

    tw_sites = set(td[td.SPECIES == "TW"].SITE_IDENTIFIER.unique())
    hdr["has_yew"] = hdr.SITE_IDENTIFIER.isin(tw_sites).astype(int)

    pres = hdr[hdr.has_yew == 1]
    absn = hdr[hdr.has_yew == 0]
    if n_absence and n_absence < len(absn):
        absn = absn.sample(n=n_absence, random_state=seed)
    faib = pd.concat([pres, absn], ignore_index=True)[
        ["SITE_IDENTIFIER", "Latitude", "Longitude", "BEC_ZONE", "has_yew"]]
    faib = faib.rename(columns={"SITE_IDENTIFIER": "site", "Latitude": "lat",
                                "Longitude": "lon", "BEC_ZONE": "bec"})
    faib["site"] = "faib_" + faib.site.astype(str)
    faib["dataset"] = "faib"
    print(f"FAIB points: {len(faib)}  (presence={faib.has_yew.sum()}, "
          f"absence={(faib.has_yew==0).sum()})  zones={dict(faib.bec.value_counts())}")
    frames = [faib]

    if include_inat:
        v = pd.read_csv(ROOT / "data/processed/val_split_balanced_max.csv", low_memory=False)
        v = v[(v.lat > 47.5) & (v.lat < 60.5) & (v.lon > -140) & (v.lon < -114)].copy()
        v = v.reset_index(drop=True)
        inat = pd.DataFrame({"site": ["inat_%d" % i for i in range(len(v))],
                             "lat": v.lat.values, "lon": v.lon.values, "bec": "",
                             "has_yew": v.has_yew.astype(int).values, "dataset": "inat"})
        print(f"iNat val points: {len(inat)}  (presence={inat.has_yew.sum()}, "
              f"absence={(inat.has_yew==0).sum()})")
        frames.append(inat)

    return pd.concat(frames, ignore_index=True)


# ── GEE embedding extraction at points (batched sampleRegions) ────────────────
def extract_embeddings(pts, year, gee_project, batch_size=200):
    cache = CACHE_DIR / f"faib_point_embeddings_{year}.csv"
    if cache.exists():
        emb = pd.read_csv(cache)
        have = set(emb.site.astype(str))
        missing = pts[~pts.site.astype(str).isin(have)]
        if missing.empty:
            print(f"  Embedding cache hit ({len(emb)} pts): {cache}")
            return emb
        print(f"  Cache has {len(emb)}; extracting {len(missing)} new points")
        pts = missing
    else:
        emb = None

    import ee
    ee.Initialize(project=gee_project)
    band_names = [f"A{i:02d}" for i in range(64)]
    lats = pts.lat.values.astype(float); lons = pts.lon.values.astype(float)
    sites = pts.site.astype(str).values
    rows, failed = [], 0

    for start in range(0, len(pts), batch_size):
        end = min(start + batch_size, len(pts))
        blat, blon, bsite = lats[start:end], lons[start:end], sites[start:end]
        pad = 0.05
        bbox = ee.Geometry.Rectangle([float(blon.min())-pad, float(blat.min())-pad,
                                      float(blon.max())+pad, float(blat.max())+pad])
        try:
            img = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
                   .filterDate(f"{year}-01-01", f"{year+1}-01-01")
                   .filterBounds(bbox).mosaic().select(band_names).toFloat())
            fc = ee.FeatureCollection([
                ee.Feature(ee.Geometry.Point([float(blon[i]), float(blat[i])]),
                           {"idx": int(i)}) for i in range(len(blat))])
            sampled = img.sampleRegions(collection=fc, scale=10,
                                        geometries=False, tileScale=4)
            offset = 0
            while True:
                page = sampled.toList(100, offset).getInfo()
                if not page:
                    break
                for f in page:
                    p = f.get("properties", {})
                    idx = p.get("idx")
                    vals = [p.get(f"A{b:02d}") for b in range(64)]
                    if idx is not None and all(v is not None for v in vals):
                        rows.append({"site": bsite[idx],
                                     **{f"emb_{k}": vals[k] for k in range(64)}})
                    else:
                        failed += 1
                if len(page) < 100:
                    break
                offset += 100
        except Exception as e:
            print(f"    batch {start}:{end} failed: {repr(e)[:120]}")
            failed += len(blat)
        time.sleep(0.5)
        print(f"    extracted {len(rows)}/{len(pts)} ...", end="\r")

    print(f"\n  Extracted {len(rows)} embeddings (failed/no-coverage={failed})")
    new = pd.DataFrame(rows)
    out = new if emb is None else pd.concat([emb, new], ignore_index=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    print(f"  Saved embedding cache: {cache}")
    return out


# ── VRI suppression factor per point (local GDB, point-in-polygon) ────────────
def vri_factors(pts):
    cache = CACHE_DIR / "faib_point_vri.csv"
    if cache.exists():
        vc = pd.read_csv(cache)
        if set(pts.site.astype(str)).issubset(set(vc.site.astype(str))):
            print(f"  VRI cache hit: {cache}")
            return vc
    import geopandas as gpd
    from shapely.geometry import Point
    from pyproj import Transformer
    tf = Transformer.from_crs("EPSG:4326", "EPSG:3005", always_xy=True)
    cy = datetime.datetime.now().year
    recs = []
    t0 = time.time()
    for j, (_, r) in enumerate(pts.iterrows()):
        x, y = tf.transform(float(r.lon), float(r.lat))
        try:
            g = gpd.read_file(str(C.VEG_COMP_GDB), layer=C.VEG_COMP_LAYER,
                              bbox=(x-50, y-50, x+50, y+50), columns=VRI_COLS)
        except Exception:
            g = None
        if g is None or len(g) == 0:
            cat = 0                         # no VRI coverage → left unchanged on the map
        else:
            pt = Point(x, y)
            hit = g[g.contains(pt)]
            row = hit.iloc[0] if len(hit) else g.iloc[0]
            cat = C._classify_vri_row(row, cy)
        recs.append({"site": str(r.site), "vri_cat": int(cat),
                     "suppress": float(C.LOG_SUPPRESS.get(cat, 1.0))})
        if (j+1) % 50 == 0:
            print(f"    VRI {j+1}/{len(pts)}  ({time.time()-t0:.0f}s)", end="\r")
    vc = pd.DataFrame(recs)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    vc.to_csv(cache, index=False)
    print(f"\n  Saved VRI cache: {cache}")
    return vc


# ── Continuous Boyce index (Hirzel et al. 2006), point-background variant ──────
def continuous_boyce(pres, avail, n_bins=10):
    pres  = np.asarray(pres, float);  pres  = pres[~np.isnan(pres)]
    avail = np.asarray(avail, float); avail = avail[~np.isnan(avail)]
    lo, hi = float(avail.min()), float(avail.max())
    if hi <= lo or len(pres) < 10:
        return None, None
    edges = np.linspace(lo, hi, n_bins + 1)
    mids  = (edges[:-1] + edges[1:]) / 2
    F = []
    for i in range(n_bins):
        a, b = edges[i], edges[i+1]
        last = (i == n_bins - 1)
        Pn = ((pres >= a) & (pres <= b)).mean() if last else ((pres >= a) & (pres < b)).mean()
        En = ((avail >= a) & (avail <= b)).mean() if last else ((avail >= a) & (avail < b)).mean()
        F.append(Pn / En if En > 0 else np.nan)
    F = np.array(F)
    m = ~np.isnan(F)
    if m.sum() < 3:
        return None, None
    rho, pval = spearmanr(mids[m], F[m])
    return float(rho), {"bin_mids": mids[m].round(3).tolist(),
                        "F": np.round(F[m], 3).tolist(), "p": round(float(pval), 4)}


def score(y, s, label):
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 f1_score, precision_score, recall_score)
    s = np.where(np.isnan(s), 0.0, s)
    pred = (s >= 0.5).astype(int)
    tp = int(((pred==1)&(y==1)).sum()); fp = int(((pred==1)&(y==0)).sum())
    tn = int(((pred==0)&(y==0)).sum()); fn = int(((pred==0)&(y==1)).sum())
    boyce, detail = continuous_boyce(s[y==1], s)
    r = {"surface": label,
         "auc": round(roc_auc_score(y, s), 4),
         "avg_precision": round(average_precision_score(y, s), 4),
         "f1": round(f1_score(y, pred, zero_division=0), 4),
         "precision": round(precision_score(y, pred, zero_division=0), 4),
         "recall": round(recall_score(y, pred, zero_division=0), 4),
         "boyce": None if boyce is None else round(boyce, 3),
         "tp": tp, "fp": fp, "tn": tn, "fn": fn}
    print(f"  {label:<11} AUC={r['auc']:.4f}  AP={r['avg_precision']:.4f}  "
          f"F1={r['f1']:.3f}  Boyce={r['boyce']}  (TP={tp} FP={fp} TN={tn} FN={fn})")
    return r


def report_group(df, name):
    y = df.has_yew.values.astype(int)
    if len(set(y)) < 2:
        print(f"\n  [{name}] single-class (n={len(df)}); skipping")
        return None
    print(f"\n  [{name}] n={len(df)} (presence={int(y.sum())}, absence={int((y==0).sum())})")
    raw  = score(y, df.p_raw.values,  "RAW")
    supp = score(y, df.p_supp.values, "SUPPRESSED")
    return {"n": int(len(df)), "n_pres": int(y.sum()), "n_abs": int((y == 0).sum()),
            "raw": raw, "suppressed": supp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-absence", type=int, default=0, help="0 = all FAIB absence sites")
    ap.add_argument("--no-inat", action="store_true", help="exclude iNat val points")
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gee-project", default="carbon-storm-206002")
    args = ap.parse_args()

    print("STEP 1 — build point set")
    pts = build_points(args.n_absence, args.seed, include_inat=not args.no_inat)

    print("\nSTEP 2 — extract AlphaEarth embeddings at points (GEE)")
    emb = extract_embeddings(pts, args.year, args.gee_project)

    print("\nSTEP 3 — VRI suppression factor per point (local GDB)")
    vc = vri_factors(pts)

    print("\nSTEP 4 — classify + validate")
    for d in (pts, emb, vc):
        d["site"] = d["site"].astype(str)
    df = pts.merge(emb, on="site", how="inner").merge(vc, on="site", how="left")
    df["suppress"] = df["suppress"].fillna(1.0)

    import xgboost as xgb
    model = xgb.XGBClassifier(); model.load_model(str(MODEL))
    X = df[[f"emb_{i}" for i in range(64)]].values
    df["p_raw"] = model.predict_proba(X)[:, 1]
    df["p_supp"] = df["p_raw"] * df["suppress"]

    # ── Water / no-coverage suppression verification ──────────────────────────
    water = df[df.vri_cat == 1]; nocov = df[df.vri_cat == 0]
    n_water_bad = int((water.p_supp > 1e-9).sum())
    print("\n  WATER / SUPPRESSION CHECK")
    print(f"    cat 1 (water/non-forest): {len(water)} pts; suppressed>0: {n_water_bad} "
          f"(expect 0); max p_supp={float(water.p_supp.max()) if len(water) else 0:.4g}")
    print(f"    cat 0 (no VRI coverage — left unchanged, as the map does): {len(nocov)} pts; "
          f"raw p>0.5: {int((nocov.p_raw > 0.5).sum())}; presences among them: {int(nocov.has_yew.sum())}")
    assert n_water_bad == 0, "Water (cat 1) points were not fully suppressed!"

    faib = df[df.dataset == "faib"]
    pv = faib[faib.has_yew == 1].vri_cat.value_counts().sort_index()
    print("\n  FAIB presence VRI breakdown: " +
          ", ".join(f"cat{int(k)}={int(v)}" for k, v in pv.items()))

    # ── Per-dataset + combined validation ─────────────────────────────────────
    results = {}
    for name, g in {"FAIB (independent)": faib,
                    "iNat val (quasi-independent)": df[df.dataset == "inat"],
                    "Combined": df}.items():
        r = report_group(g, name)
        if r:
            results[name] = r

    # ── FAIB raw AUC stratified by stand condition ────────────────────────────
    from sklearn.metrics import roc_auc_score
    print("\n  FAIB raw AUC stratified by stand condition:")
    strat = {}
    for lab, cats in [("old-growth (cat7)", {7}), ("undisturbed (cat5/7)", {5, 7}),
                      ("logged/young (cat1-4)", {1, 2, 3, 4}), ("all", None)]:
        sub = faib if cats is None else faib[faib.vri_cat.isin(cats)]
        if sub.has_yew.nunique() == 2:
            auc = float(roc_auc_score(sub.has_yew, sub.p_raw))
            strat[lab] = {"n": int(len(sub)), "n_pres": int(sub.has_yew.sum()), "auc": round(auc, 4)}
            print(f"    {lab:<24} n={len(sub):>4} pres={int(sub.has_yew.sum()):>3}  AUC={auc:.3f}")

    out = {"n_points": int(len(df)), "year": args.year,
           "water_check": {"cat1_water": int(len(water)), "cat1_suppressed_gt0": n_water_bad,
                           "cat0_nocov": int(len(nocov)), "cat0_presences": int(nocov.has_yew.sum())},
           "faib_presence_vri": {int(k): int(v) for k, v in pv.items()},
           "groups": results, "faib_stratified_raw": strat,
           "note": ("Point-sampled AlphaEarth embeddings (tile-independent). FAIB PSP "
                    "presence/absence is genuinely independent of classifier training; "
                    "iNat val is quasi-independent (positives held out from training but "
                    "iNat was the training-positive source). Water (VRI cat 1) is suppressed "
                    "to 0 exactly as the published map; cat 0 (no VRI coverage) is left "
                    "unchanged, matching apply_logging_mask. Boyce uses point-set "
                    "predictions as available background.")}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()
