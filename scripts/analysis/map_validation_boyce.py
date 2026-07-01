#!/usr/bin/env python3
"""
P0.2 — Validate the habitat-probability map against independent presence data.

Uses two independent presence/absence datasets (neither used for habitat map training):
  1. Held-out iNaturalist validation set (val_split_balanced_max.csv, 208 BC+ positives)
  2. FAIB PSP sites (never used in training; results from faib_psp_validation.py)

Computes:
  - Confusion matrix at P≥0.5 operational threshold
  - Continuous Boyce index (Hirzel et al. 2006) using presence-only points
  - AUC-ROC and average precision where absence data allows

Map predictions come from the per-tile probability PNGs (same rendering as the published
habitat maps). This independently validates the final suppressed-probability surface, not
just the raw classifier output.

Run:
    /home/jericho/anaconda3/envs/yew_pytorch/bin/python scripts/analysis/map_validation_boyce.py
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr

ROOT       = Path("/home/jericho/yew_project")
TILES_JSON = ROOT / "docs/tiles/tiles.json"
TILES_DIR  = ROOT / "docs/tiles"
OUT_DIR    = ROOT / "results/analysis"

# ── Probability PNG colormap (YEWCMAP, matches classify_cwh_spots.py) ────────
YEWCMAP = LinearSegmentedColormap.from_list("yew", [
    (0.00, (0.20, 0.70, 0.20, 0.70)), (0.17, (0.45, 0.85, 0.05, 0.80)),
    (0.33, (1.00, 0.90, 0.00, 0.88)), (0.50, (1.00, 0.60, 0.00, 0.90)),
    (0.67, (0.90, 0.40, 0.10, 0.93)), (0.83, (0.80, 0.15, 0.30, 0.95)),
    (1.00, (0.65, 0.00, 0.45, 0.96))], N=256)
_PROB_LUT = (YEWCMAP(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.float32)
_PROB_VAL = np.linspace(0, 1, 256)

# ── Logging category colormap ─────────────────────────────────────────────────
LOG_RGBA  = {1:(30,100,220,180), 2:(220,50,50,170), 3:(230,120,30,150),
             4:(220,200,50,110), 5:(180,220,70,70), 6:(175,155,125,160), 7:(20,100,40,70)}
_LOG_LUT  = np.array([[v[0],v[1],v[2]] for v in LOG_RGBA.values()], np.float32)
_LOG_CATS = np.array(list(LOG_RGBA.keys()), np.int8)
CAT_LABEL = {0:"no data",1:"water",2:"logged<20yr",3:"logged 20–40yr",
             4:"logged 40–80yr",5:"forest 80–150yr",6:"alpine",7:"old-growth >150yr"}


def pixel_prob(rgba, row, col, win=2):
    """Return mean yew probability in a (2*win+1)^2 window (NaN = transparent)."""
    H, W = rgba.shape[:2]
    patch = rgba[max(0,row-win):min(H,row+win+1), max(0,col-win):min(W,col+win+1)]
    vis   = (patch[:,:,3] >= 13).ravel()
    if not vis.any():
        return np.nan
    px = patch[:,:,:3].reshape(-1,3).astype(np.float32)[vis]
    best_d = np.full(len(px), np.inf, np.float32)
    best_i = np.zeros(len(px), np.int16)
    for i in range(256):
        d = ((px - _PROB_LUT[i])**2).sum(1)
        m = d < best_d; best_d[m] = d[m]; best_i[m] = i
    return float(_PROB_VAL[best_i].mean())


def pixel_cat(rgba, row, col, win=2):
    """Return modal VRI logging category in a (2*win+1)^2 window."""
    H, W = rgba.shape[:2]
    patch = rgba[max(0,row-win):min(H,row+win+1), max(0,col-win):min(W,col+win+1)]
    vis   = (patch[:,:,3] >= 13).ravel()
    if not vis.any():
        return 0
    px   = patch[:,:,:3].reshape(-1,3).astype(np.float32)[vis]
    dists = ((px[:,None,:] - _LOG_LUT[None,:,:])**2).sum(2)
    cats  = _LOG_CATS[dists.argmin(1)]
    return int(Counter(cats).most_common(1)[0][0])


def match_to_tiles(df_pts, lat_col, lon_col, tiles, img_cache):
    """Match DataFrame rows to tile pixels; return rows with prob and vri_cat."""
    rows = []
    for _, r in df_pts.iterrows():
        lat, lon = r[lat_col], r[lon_col]
        for t in tiles:
            b = t["bbox"]
            if not (b["west"] <= lon <= b["east"] and b["south"] <= lat <= b["north"]):
                continue
            slug = t["slug"]
            pp = TILES_DIR / t["png"]
            lp = TILES_DIR / f"{slug}_logging.png"
            if not pp.exists() or not lp.exists():
                break
            if slug not in img_cache:
                img_cache[slug] = {
                    "prob": np.asarray(Image.open(pp).convert("RGBA")),
                    "log":  np.asarray(Image.open(lp).convert("RGBA")),
                }
            imgs = img_cache[slug]
            H, W = imgs["prob"].shape[:2]
            col_ = int((lon - b["west"])  / (b["east"]  - b["west"])  * W)
            row_ = int((b["north"] - lat) / (b["north"] - b["south"]) * H)
            if not (0 <= row_ < H and 0 <= col_ < W):
                break
            rows.append(dict(r) | {
                "tile": slug,
                "prob": pixel_prob(imgs["prob"], row_, col_),
                "vri_cat": pixel_cat(imgs["log"],  row_, col_),
            })
            break
    return pd.DataFrame(rows)


def continuous_boyce(probs, bins=10, fill=0.0):
    """
    Continuous Boyce index (Hirzel et al. 2006) from presence-only probabilities.

    probs  : predicted probability at presence locations (1-D array)
    bins   : number of equal-width bins across [0,1]
    fill   : probability assigned to transparent pixels (model ≈ 0)
    Returns: (boyce_index, spearman_rho, p_value)
    """
    p = np.where(np.isnan(probs), fill, probs)
    edges = np.linspace(0, 1, bins + 1)
    obs   = np.array([((p >= edges[i]) & (p < edges[i+1])).sum() for i in range(bins)], float)
    # Expected under uniform: bin width / 1.0 = 1/bins (all bins equal)
    exp   = np.full(bins, 1.0 / bins)
    # F_i = obs_i / expected_i (boyce ratio per bin)
    F     = obs / (exp * len(p))
    # Use bin midpoints as x-axis
    mids  = (edges[:-1] + edges[1:]) / 2
    # Only include bins with nonzero expected; filter bins with obs=0 as per Boyce
    mask  = obs > 0
    if mask.sum() < 3:
        return np.nan, np.nan, np.nan
    rho, pval = spearmanr(mids[mask], F[mask])
    return float(rho), float(pval), {"bin_mids": mids.tolist(), "F": F.tolist(), "obs": obs.tolist()}


def main():
    tiles      = json.loads(TILES_JSON.read_text())
    img_cache  = {}
    results    = {}

    # ── Dataset 1: iNaturalist held-out validation set ────────────────────────
    print("── Dataset 1: iNaturalist held-out validation set ──")
    val = pd.read_csv(ROOT / "data/processed/val_split_balanced_max.csv", low_memory=False)
    # Filter to BC lat/lon range
    val_bc = val[(val.lat > 47.5) & (val.lat < 60.5) &
                 (val.lon > -140)  & (val.lon < -114)].copy()
    print(f"  BC records in val split: {len(val_bc)}  (pos={val_bc.has_yew.sum()}, neg={(~val_bc.has_yew).sum()})")

    mv = match_to_tiles(val_bc, "lat", "lon", tiles, img_cache)
    mv["prob_filled"] = mv["prob"].fillna(0.0)
    print(f"  Matched to study tiles:  {len(mv)}  (pos={mv.has_yew.sum()}, neg={(~mv.has_yew).sum()})")

    if len(mv) >= 5 and mv.has_yew.nunique() == 2:
        from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
        y = mv.has_yew.values.astype(int)
        s = mv.prob_filled.values
        pred = (s >= 0.5).astype(int)
        auc  = roc_auc_score(y, s)
        ap   = average_precision_score(y, s)
        f1   = f1_score(y, pred, zero_division=0)
        prec = precision_score(y, pred, zero_division=0)
        rec  = recall_score(y, pred, zero_division=0)
        tp = ((pred==1)&(y==1)).sum(); fp = ((pred==1)&(y==0)).sum()
        tn = ((pred==0)&(y==0)).sum(); fn = ((pred==0)&(y==1)).sum()
        print(f"  AUC-ROC={auc:.3f}  AvgPrec={ap:.3f}  F1={f1:.3f}  Prec={prec:.3f}  Rec={rec:.3f}")
        print(f"  Confusion (P≥0.5): TP={tp} FP={fp} TN={tn} FN={fn}")

        # Boyce index on presence points only
        pres_probs = mv[mv.has_yew == 1].prob_filled.values
        bi, pval, detail = continuous_boyce(pres_probs)
        print(f"  Boyce index (presence-only, n={len(pres_probs)}): rho={bi:.3f}  p={pval:.4f}")
        results["inat_val"] = {"auc": round(auc, 3), "avg_precision": round(ap, 3),
                               "f1": round(f1, 3), "n_pos": int(tp+fn), "n_neg": int(tn+fp),
                               "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
                               "boyce_rho": round(bi, 3) if not np.isnan(bi) else None,
                               "boyce_pval": round(pval, 4) if not np.isnan(pval) else None,
                               "boyce_detail": detail}
    else:
        print("  Insufficient points for full validation.")
        results["inat_val"] = {"n_matched": len(mv)}

    # ── Dataset 2: FAIB PSP sites ─────────────────────────────────────────────
    print("\n── Dataset 2: FAIB PSP sites (independent — never used in classifier training) ──")
    td  = pd.read_csv(ROOT / "data/raw/faib_tree_detail.csv", low_memory=False)
    hdr = pd.read_csv(ROOT / "data/raw/faib_header.csv", low_memory=False)

    tw = (td[td.SPECIES == "TW"]
            .sort_values("VISIT_NUMBER")
            .groupby(["SITE_IDENTIFIER","PLOT","TREE_NO"], as_index=False).last())
    yew_sites = set(tw.SITE_IDENTIFIER.unique())

    hh = (hdr[["SITE_IDENTIFIER","Longitude","Latitude","BEC_ZONE","BEC_SBZ"]]
            .drop_duplicates("SITE_IDENTIFIER")
            .dropna(subset=["Longitude","Latitude"]))
    hh = hh[hh.BEC_ZONE.isin({"CWH","ICH","CDF"})].copy()
    hh["has_yew"] = hh.SITE_IDENTIFIER.isin(yew_sites).astype(int)
    print(f"  FAIB sites in CWH/ICH/CDF: {len(hh)}  (yew={hh.has_yew.sum()})")

    mf = match_to_tiles(hh, "Latitude", "Longitude", tiles, img_cache)
    mf["prob_filled"] = mf["prob"].fillna(0.0)
    print(f"  Matched to study tiles: {len(mf)}  (yew={mf.has_yew.sum()})")

    if len(mf) >= 5:
        for cat_set, label in [({5, 7}, "undisturbed (cat 5/7)"),
                                ({2, 3, 4}, "logged (cat 2/3/4)"),
                                (set(range(8)), "all categories")]:
            sub = mf[mf.vri_cat.isin(cat_set)].copy() if cat_set != set(range(8)) else mf.copy()
            if len(sub) < 3:
                continue
            n_y = sub.has_yew.sum(); n_a = len(sub) - n_y
            print(f"\n  {label}: n={len(sub)}  yew={n_y}  absent={n_a}")
            if n_y > 0:
                bi, pval, _ = continuous_boyce(sub[sub.has_yew==1].prob_filled.values)
                print(f"    Boyce (presence-only n={n_y}): rho={bi:.3f}  p={pval:.4f}" if not np.isnan(bi) else f"    Boyce: insufficient bins (n={n_y})")
            if sub.has_yew.nunique() == 2:
                from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
                y = sub.has_yew.values.astype(int)
                s = sub.prob_filled.values
                pred = (s >= 0.5).astype(int)
                auc = roc_auc_score(y, s)
                tp = ((pred==1)&(y==1)).sum(); fp = ((pred==1)&(y==0)).sum()
                tn = ((pred==0)&(y==0)).sum(); fn = ((pred==0)&(y==1)).sum()
                print(f"    AUC={auc:.3f}  TP={tp} FP={fp} TN={tn} FN={fn}")

    # ── Combined: all independent presence points ─────────────────────────────
    print("\n── Combined independent presence points (iNat val + FAIB PSP) ──")
    pres_all = pd.concat([
        mv[mv.has_yew == 1][["prob_filled"]],
        mf[mf.has_yew == 1][["prob_filled"]],
    ], ignore_index=True)
    print(f"  Total independent presence points in tiles: {len(pres_all)}")
    bi, pval, detail = continuous_boyce(pres_all.prob_filled.values)
    print(f"  Boyce index (all independent presence): rho={bi:.3f}  p={pval:.4f}" if not np.isnan(bi) else f"  Boyce: insufficient bins (n={len(pres_all)})")
    results["combined_presence_boyce"] = {
        "n": len(pres_all),
        "boyce_rho": round(bi, 3) if not np.isnan(bi) else None,
        "boyce_pval": round(pval, 4) if not np.isnan(pval) else None,
    }

    out = OUT_DIR / "map_validation_boyce.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")

    print("""
LIMITATIONS
───────────
- iNat val-set positives: the validation split (208 BC positives) is held out from the
  CLASSIFIER training but their tile locations could have influenced tile selection for
  the habitat map. Caveat: presence of any iNat positive in a tile increased its
  likelihood of being included in the 98-tile study set.
- FAIB PSP sites: genuinely independent (never used in any analysis until §3.3 / P0.2).
  Only 4 yew-present PSP sites fall inside study tiles — all are in VRI-logged areas,
  so the suppression pipeline correctly assigns low predictions to them. No undisturbed
  yew-present PSP sites fall within the study tiles, limiting sensitivity assessment.
- The Boyce index requires ≥10 presence points for reliable estimates; treat results with
  very few points as directional, not confirmatory.
""")


if __name__ == "__main__":
    main()
