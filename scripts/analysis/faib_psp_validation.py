#!/usr/bin/env python3
"""
Validate yew classifier accuracy at FAIB PSP sites in undisturbed forest.

For each PSP site falling within a study tile:
  1. Extract predicted yew probability from the tile's probability PNG
  2. Extract VRI logging category from the tile's logging PNG
  3. Filter to undisturbed sites (cat 5 = forest 80-150yr, cat 7 = old-growth)
  4. Compare predictions vs actual yew presence/absence

Yew-present : FAIB TW records, deduplicated to latest visit, site level
Yew-absent  : PSP sites in CWH/ICH/CDF with no TW recorded

Run: conda run -n yew_pytorch python scripts/analysis/faib_psp_validation.py
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path("/home/jericho/yew_project")
TILES_JSON = ROOT / "docs/tiles/tiles.json"
TILES_DIR  = ROOT / "docs/tiles"
OUT_DIR    = ROOT / "results/analysis"
FIG_DIR    = ROOT / "results/figures/paper"

# ── Probability PNG colormap (from protected_area_from_tiles.py) ──────────────
YEWCMAP = LinearSegmentedColormap.from_list('yew', [
    (0.00, (0.20, 0.70, 0.20, 0.70)), (0.17, (0.45, 0.85, 0.05, 0.80)),
    (0.33, (1.00, 0.90, 0.00, 0.88)), (0.50, (1.00, 0.60, 0.00, 0.90)),
    (0.67, (0.90, 0.40, 0.10, 0.93)), (0.83, (0.80, 0.15, 0.30, 0.95)),
    (1.00, (0.65, 0.00, 0.45, 0.96))], N=256)
_PROB_LUT = (YEWCMAP(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.float32)
_PROB_VAL = np.linspace(0, 1, 256)

# ── Logging PNG colormap (from classify_cwh_spots.py LOG_RGBA) ────────────────
LOG_RGBA = {
    1: (30,  100, 220, 180),
    2: (220,  50,  50, 170),
    3: (230, 120,  30, 150),
    4: (220, 200,  50, 110),
    5: (180, 220,  70,  70),
    6: (175, 155, 125, 160),
    7: ( 20, 100,  40,  70),
}
_LOG_LUT  = np.array([[v[0], v[1], v[2]] for v in LOG_RGBA.values()], np.float32)
_LOG_CATS = np.array(list(LOG_RGBA.keys()), np.int8)

CAT_LABEL = {0:'no data',1:'water',2:'logged<20yr',3:'logged 20-40yr',
             4:'logged 40-80yr',5:'forest 80-150yr',6:'alpine',7:'old-growth >150yr'}


def pixel_coords(lat, lon, bbox, H, W):
    col = int((lon - bbox["west"]) / (bbox["east"] - bbox["west"]) * W)
    row = int((bbox["north"] - lat) / (bbox["north"] - bbox["south"]) * H)
    return row, col


def extract_prob(rgba, row, col, win=2):
    """Mean yew probability in a (2*win+1)^2 window around (row, col)."""
    H, W = rgba.shape[:2]
    patch = rgba[max(0, row-win):min(H, row+win+1),
                 max(0, col-win):min(W, col+win+1)]
    alpha = patch[:, :, 3]
    vis = (alpha >= 13).ravel()
    if not vis.any():
        return np.nan
    px = patch[:, :, :3].reshape(-1, 3).astype(np.float32)[vis]
    # iterate over 256 LUT entries (memory-light, same as protected_area script)
    best_d = np.full(len(px), np.inf, np.float32)
    best_i = np.zeros(len(px), np.int16)
    for i in range(256):
        d = ((px - _PROB_LUT[i]) ** 2).sum(1)
        upd = d < best_d
        best_d[upd] = d[upd]
        best_i[upd] = i
    return float(_PROB_VAL[best_i].mean())


def extract_log_cat(rgba, row, col, win=2):
    """Modal VRI logging category in a (2*win+1)^2 window."""
    H, W = rgba.shape[:2]
    patch = rgba[max(0, row-win):min(H, row+win+1),
                 max(0, col-win):min(W, col+win+1)]
    alpha = patch[:, :, 3]
    vis = (alpha >= 13).ravel()
    if not vis.any():
        return 0
    px = patch[:, :, :3].reshape(-1, 3).astype(np.float32)[vis]
    dists = np.zeros((len(px), len(_LOG_CATS)), np.float32)
    for i in range(len(_LOG_CATS)):
        dists[:, i] = ((px - _LOG_LUT[i]) ** 2).sum(1)
    cat_idx = dists.argmin(1)
    cats = _LOG_CATS[cat_idx]
    return int(Counter(cats).most_common(1)[0][0])


def main():
    # ── FAIB data ─────────────────────────────────────────────────────────────
    td = pd.read_csv(ROOT / "data/raw/faib_tree_detail.csv",
                     usecols=["SITE_IDENTIFIER", "VISIT_NUMBER", "PLOT", "TREE_NO",
                              "SPECIES", "LV_D"], low_memory=False)
    h  = pd.read_csv(ROOT / "data/raw/faib_header.csv", low_memory=False)

    # Deduplicate TW to latest visit per tree; yew-present sites
    tw = (td[td.SPECIES == "TW"]
            .sort_values("VISIT_NUMBER")
            .groupby(["SITE_IDENTIFIER", "PLOT", "TREE_NO"], as_index=False).last())
    yew_sites = set(tw["SITE_IDENTIFIER"].unique())

    hh = (h[["SITE_IDENTIFIER", "Longitude", "Latitude", "BEC_ZONE", "BEC_SBZ"]]
            .drop_duplicates("SITE_IDENTIFIER")
            .dropna(subset=["Longitude", "Latitude"]))
    hh = hh[hh.BEC_ZONE.isin({"CWH", "ICH", "CDF"})].copy()
    hh["yew_present"] = hh.SITE_IDENTIFIER.isin(yew_sites).astype(int)
    print(f"PSP sites in CWH/ICH/CDF with coords: {len(hh)}")
    print(f"  Yew-present: {hh.yew_present.sum()}  |  Yew-absent: {(~hh.yew_present.astype(bool)).sum()}")

    # ── Tile matching ─────────────────────────────────────────────────────────
    tiles = json.loads(Path(TILES_JSON).read_text())
    img_cache = {}
    rows = []

    for _, site in hh.iterrows():
        lat, lon = site.Latitude, site.Longitude
        for t in tiles:
            b = t["bbox"]
            if not (b["west"] <= lon <= b["east"] and b["south"] <= lat <= b["north"]):
                continue
            slug = t["slug"]
            prob_path = TILES_DIR / t["png"]
            log_path  = TILES_DIR / f"{slug}_logging.png"
            if not prob_path.exists() or not log_path.exists():
                break
            if slug not in img_cache:
                img_cache[slug] = {
                    "prob": np.asarray(Image.open(prob_path).convert("RGBA")),
                    "log":  np.asarray(Image.open(log_path).convert("RGBA")),
                }
            imgs = img_cache[slug]
            H, W = imgs["prob"].shape[:2]
            row, col = pixel_coords(lat, lon, b, H, W)
            if not (0 <= row < H and 0 <= col < W):
                break
            rows.append({
                "site": site.SITE_IDENTIFIER,
                "bec_zone": site.BEC_ZONE, "bec_sbz": site.BEC_SBZ,
                "yew_present": int(site.yew_present),
                "tile": slug,
                "prob": extract_prob(imgs["prob"], row, col),
                "vri_cat": extract_log_cat(imgs["log"], row, col),
            })
            break  # matched; move to next site

    df = pd.DataFrame(rows)
    print(f"\nMatched to study tiles: {len(df)} PSP sites")
    cat_counts = df.vri_cat.map(CAT_LABEL).value_counts()
    print("VRI category breakdown:")
    for label, n in cat_counts.items():
        print(f"  {label:22s} {n:4d}")

    df.to_csv(OUT_DIR / "faib_psp_validation.csv", index=False)
    print(f"\nSaved raw results to {OUT_DIR/'faib_psp_validation.csv'}")

    # ── Filter to undisturbed forest ──────────────────────────────────────────
    # NaN probability = transparent pixel = model predicted ~0 (below rendering threshold)
    df["prob"] = df["prob"].fillna(0.0)
    df_ud = df[df.vri_cat.isin({5, 7})].copy()
    n_yew = df_ud.yew_present.sum()
    n_abs = len(df_ud) - n_yew
    print(f"\n── Undisturbed forest (VRI cat 5 or 7): {len(df_ud)} sites ──")
    print(f"   Yew-present: {n_yew}   |   Yew-absent: {n_abs}")

    if len(df_ud) < 5 or df_ud.yew_present.nunique() < 2:
        print("Insufficient data for ROC; descriptive stats only.")
    else:
        from sklearn.metrics import roc_auc_score, roc_curve
        y = df_ud.yew_present.values
        s = df_ud.prob.values

        auc = roc_auc_score(y, s)
        print(f"\nAUC-ROC: {auc:.3f}")

        # Stats at P ≥ 0.5
        pred = (s >= 0.5).astype(int)
        tp = ((pred==1) & (y==1)).sum()
        fp = ((pred==1) & (y==0)).sum()
        tn = ((pred==0) & (y==0)).sum()
        fn = ((pred==0) & (y==1)).sum()
        acc  = (tp+tn) / len(y)
        prec = tp/(tp+fp) if (tp+fp) > 0 else 0.0
        rec  = tp/(tp+fn) if (tp+fn) > 0 else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
        print(f"At P≥0.5:  Acc={acc:.2f}  Prec={prec:.2f}  Rec={rec:.2f}  F1={f1:.2f}")
        print(f"           TP={tp}  FP={fp}  TN={tn}  FN={fn}")

        # By VRI category
        for cat in [5, 7]:
            sub = df_ud[df_ud.vri_cat == cat]
            if len(sub) < 3:
                continue
            label = CAT_LABEL[cat]
            n_sub = len(sub); n_y = sub.yew_present.sum()
            print(f"\n{label} (n={n_sub}, yew={n_y}):")
            if sub.yew_present.nunique() == 2:
                a = roc_auc_score(sub.yew_present, sub.prob)
                print(f"  AUC = {a:.3f}")
            print(f"  Mean prob | yew-present : {sub[sub.yew_present==1].prob.mean():.3f}")
            print(f"  Mean prob | yew-absent  : {sub[sub.yew_present==0].prob.mean():.3f}")

        # By BEC zone
        for zone in df_ud.bec_zone.unique():
            sub = df_ud[df_ud.bec_zone == zone]
            if len(sub) < 3 or sub.yew_present.nunique() < 2:
                continue
            a = roc_auc_score(sub.yew_present, sub.prob)
            print(f"\n{zone} undisturbed (n={len(sub)}, yew={sub.yew_present.sum()}):  AUC={a:.3f}")

        # ── Figure: ROC + probability distributions ────────────────────────
        fpr, tpr, _ = roc_curve(y, s)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Yew classifier validation — FAIB PSP sites in undisturbed forest\n"
                     "(VRI cat 5: forest 80-150yr or cat 7: old-growth >150yr)",
                     y=1.01, fontsize=11)

        ax = axes[0]
        ax.plot(fpr, tpr, color="#0072B2", lw=2, label=f"ROC (AUC = {auc:.3f})")
        ax.plot([0,1],[0,1], "k--", lw=0.8)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC curve  (n={len(df_ud)}: {n_yew} yew-present, {n_abs} yew-absent)")
        ax.legend(fontsize=10)

        ax = axes[1]
        pres = df_ud[df_ud.yew_present==1]["prob"]
        abse = df_ud[df_ud.yew_present==0]["prob"]
        bins = np.linspace(0, 1, 21)
        ax.hist(abse, bins=bins, alpha=0.65, color="#D55E00", label=f"Yew-absent (n={n_abs})")
        ax.hist(pres, bins=bins, alpha=0.65, color="#009E73", label=f"Yew-present (n={n_yew})")
        ax.axvline(0.5, color="k", lw=1, ls="--", label="P=0.5 threshold")
        ax.set_xlabel("Predicted yew probability"); ax.set_ylabel("PSP site count")
        ax.set_title("Predicted probability by presence/absence")
        ax.legend(fontsize=9)

        plt.tight_layout()
        out_fig = FIG_DIR / "fig_psp_validation.png"
        plt.savefig(out_fig, dpi=180, bbox_inches="tight")
        print(f"\nSaved figure: {out_fig}")

    # ── Also report: all matched sites (include disturbed) ──────────────────
    print("\n── All matched sites by VRI category ──")
    for cat in sorted(df.vri_cat.unique()):
        sub = df[df.vri_cat == cat]
        if len(sub) < 2:
            continue
        n_y = sub.yew_present.sum()
        mp = sub[sub.yew_present==1].prob.mean() if n_y > 0 else float("nan")
        ma = sub[sub.yew_present==0].prob.mean() if (len(sub)-n_y) > 0 else float("nan")
        print(f"  cat {cat} {CAT_LABEL[cat]:22s} n={len(sub):3d}  yew={n_y:3d}  "
              f"mean_p(present)={mp:.3f}  mean_p(absent)={ma:.3f}")


if __name__ == "__main__":
    main()
