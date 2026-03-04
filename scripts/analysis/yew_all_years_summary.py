#!/usr/bin/env python
"""
yew_all_years_summary.py
------------------------
Fetches Google Satellite Embeddings for all available years (2017–2024),
classifies each with the XGBoost yew model, and produces:

  • Console summary table (mean / median / std / P≥0.20 / P≥0.50 / P≥0.95)
  • results/tables/yew_prob_all_years_summary.csv
  • results/figures/yew_prob_all_years.png  (4-panel figure)

Caches embeddings as  results/predictions/cwh_xgb_sample_{year}_emb.npy
so subsequent runs skip the GEE download.

Usage
-----
    python scripts/analysis/yew_all_years_summary.py
    python scripts/analysis/yew_all_years_summary.py --years 2019 2020 2021
    python scripts/analysis/yew_all_years_summary.py --batch-size 250
"""
import argparse
import sys
import time
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
SAMPLE    = ROOT / "results" / "predictions" / "cwh_xgb_sample.csv"
XGB_MODEL = ROOT / "results" / "predictions" / "south_vi_large" / "xgb_raw_model_expanded.json"
PRED_DIR  = ROOT / "results" / "predictions"
FIG_DIR   = ROOT / "results" / "figures"
TABLE_DIR = ROOT / "results" / "tables"

# ── Constants ─────────────────────────────────────────────────────────────────
GEE_PROJECT    = "carbon-storm-206002"
GEE_ASSET      = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
GEE_BAND_NAMES = [f"A{i:02d}" for i in range(64)]
ALL_YEARS      = list(range(2017, 2025))   # 2017–2024 inclusive
CWH_AREA_HA    = 3_507_194

# One colour per year — interpolated across a blue→orange palette
_CMAP = plt.colormaps["plasma"]
YEAR_COLOUR = {
    yr: mcolors.to_hex(_CMAP(i / max(len(ALL_YEARS) - 1, 1)))
    for i, yr in enumerate(ALL_YEARS)
}


# ── GEE fetch + classify ──────────────────────────────────────────────────────

def fetch_and_classify(
    year: int,
    bst: xgb.Booster,
    lats: np.ndarray,
    lons: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Return (N,) float32 probability array for *year*. Uses cache if available."""
    N     = len(lats)
    cache = PRED_DIR / f"cwh_xgb_sample_{year}_emb.npy"

    if cache.exists():
        print(f"  [{year}] cache hit — loading {cache.name}")
        embs = np.load(cache)
    else:
        print(f"  [{year}] fetching from GEE …")
        ee.Initialize(project=GEE_PROJECT)
        ee_img = (
            ee.ImageCollection(GEE_ASSET)
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .mosaic()
            .select(GEE_BAND_NAMES)
            .toFloat()
        )
        embs      = np.full((N, 64), np.nan, dtype=np.float32)
        n_batches = (N + batch_size - 1) // batch_size
        t0        = time.time()

        for b in tqdm(range(n_batches), desc=f"    {year}", leave=False):
            lo, hi   = b * batch_size, min((b + 1) * batch_size, N)
            features = [
                ee.Feature(
                    ee.Geometry.Point([float(lons[i]), float(lats[i])]),
                    {"__idx": int(i)},
                )
                for i in range(lo, hi)
            ]
            try:
                sampled = ee_img.sampleRegions(
                    collection=ee.FeatureCollection(features),
                    scale=10, geometries=False, tileScale=4,
                )
                for feat in sampled.getInfo().get("features", []):
                    props = feat.get("properties", {})
                    idx   = props.get("__idx")
                    if idx is not None:
                        embs[idx] = [
                            props.get(band, np.nan) for band in GEE_BAND_NAMES
                        ]
            except Exception as exc:
                print(f"\n    ⚠ batch {b + 1}/{n_batches} failed: {exc}")

        n_ok = int(np.isfinite(embs[:, 0]).sum())
        elapsed = (time.time() - t0) / 60
        print(f"  [{year}] done in {elapsed:.1f} min — {n_ok:,}/{N:,} resolved")
        np.save(cache, embs)
        print(f"  [{year}] saved → {cache.name}")

    # ── Classify ──────────────────────────────────────────────────────────────
    valid = np.isfinite(embs[:, 0])
    probs = np.full(N, np.nan, dtype=np.float32)
    if valid.any():
        X            = np.nan_to_num(embs[valid], nan=0.0, posinf=0.0, neginf=0.0)
        probs[valid] = bst.predict(xgb.DMatrix(X)).astype(np.float32)

    n_valid = int(valid.sum())
    print(
        f"  [{year}] classified {n_valid:,} pts  "
        f"mean={np.nanmean(probs):.4f}  P≥0.95={int((probs >= 0.95).sum()):,}"
    )
    return probs


# ── Summary table ─────────────────────────────────────────────────────────────

def build_summary(df: pd.DataFrame, years: list) -> pd.DataFrame:
    thresholds = [0.20, 0.50, 0.80, 0.95]
    rows = []
    for yr in years:
        col = f"prob_{yr}"
        s   = df[col].dropna()
        row = {
            "year":    yr,
            "n_valid": len(s),
            "mean":    s.mean(),
            "median":  s.median(),
            "std":     s.std(),
            "p10":     s.quantile(0.10),
            "p90":     s.quantile(0.90),
        }
        for t in thresholds:
            row[f"pct_ge_{int(t*100):02d}"] = 100 * (s >= t).mean()
            row[f"ha_ge_{int(t*100):02d}"]  = (s >= t).mean() * CWH_AREA_HA
        rows.append(row)
    return pd.DataFrame(rows).set_index("year")


# ── Figure ────────────────────────────────────────────────────────────────────

def make_figure(df: pd.DataFrame, summary: pd.DataFrame, years: list, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        f"Western Yew Probability — All Available Years ({years[0]}–{years[-1]})\n"
        f"N = {len(df):,} CWH random sample points",
        fontsize=13, fontweight="bold", y=0.98,
    )

    colours = [YEAR_COLOUR[yr] for yr in years]

    # ── A: Year-over-year trend lines ─────────────────────────────────────────
    ax = axes[0, 0]
    metrics = {
        "Mean":        "mean",
        "Median":      "median",
        "P≥0.50 (%)":  "pct_ge_50",
        "P≥0.95 (%)":  "pct_ge_95",
    }
    ax2 = ax.twinx()
    lns = []
    for label, col in metrics.items():
        target     = ax2 if "%" in label else ax
        ls         = "--" if "%" in label else "-"
        lns.append(
            target.plot(
                years, summary[col].values,
                marker="o", lw=2, ls=ls, label=label,
            )[0]
        )
    ax.set_ylabel("Probability (mean / median)", fontsize=9)
    ax2.set_ylabel("% of points above threshold", fontsize=9, color="#555")
    ax.set_xlabel("Year")
    ax.set_title("(A) Year-over-year trend")
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], rotation=45)
    ax.legend(lns, [l.get_label() for l in lns], fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    # ── B: Histogram overlay ─────────────────────────────────────────────────
    ax = axes[0, 1]
    bins_h = np.linspace(0, 1, 51)
    for yr, c in zip(years, colours):
        p = df[f"prob_{yr}"].dropna().values
        ax.hist(p, bins=bins_h, color=c, alpha=0.45, label=str(yr), density=True)
    ax.axvline(0.95, color="red", lw=1.2, ls="--", label="P=0.95")
    ax.set_xlabel("Yew probability")
    ax.set_ylabel("Density")
    ax.set_title("(B) Probability distributions")
    ax.set_yscale("log")
    ax.legend(fontsize=8, ncol=2)

    # ── C: CDFs ───────────────────────────────────────────────────────────────
    ax = axes[1, 0]
    for yr, c in zip(years, colours):
        p_sorted = np.sort(df[f"prob_{yr}"].dropna().values)
        cdf      = np.linspace(0, 1, len(p_sorted))
        ax.plot(p_sorted, cdf, color=c, lw=2, label=str(yr))
    ax.axvline(0.95, color="red", lw=1.2, ls="--", label="P=0.95")
    ax.set_xlabel("Yew probability")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("(C) Cumulative distribution functions")
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)

    # ── D: BEC subzone heatmap (mean prob) ───────────────────────────────────
    ax = axes[1, 1]
    bec_means = {}
    for yr in years:
        bec_means[yr] = (
            df.dropna(subset=[f"prob_{yr}"])
            .groupby("map_label")[f"prob_{yr}"]
            .agg(["mean", "count"])
            .query("count >= 30")["mean"]
        )
    bec_df = pd.DataFrame(bec_means).dropna()
    # rank by range (most variable subzones at top)
    bec_df["_range"] = bec_df.max(axis=1) - bec_df.min(axis=1)
    top_bec = bec_df.sort_values("_range", ascending=False).head(20).drop(columns="_range")

    vmin = top_bec.values.min()
    vmax = top_bec.values.max()
    im   = ax.imshow(
        top_bec.values.T, aspect="auto",
        cmap="YlOrRd", vmin=vmin, vmax=vmax,
    )
    ax.set_xticks(range(len(top_bec)))
    ax.set_xticklabels(top_bec.index, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels([str(y) for y in years], fontsize=8)
    fig.colorbar(im, ax=ax, label="Mean yew probability", shrink=0.8)
    ax.set_title("(D) BEC subzone mean prob — top-20 by inter-year range")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out_path}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--years", nargs="+", type=int, default=ALL_YEARS,
        metavar="YEAR",
        help=f"Years to include (default: {ALL_YEARS[0]}–{ALL_YEARS[-1]})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=500,
        help="GEE sampleRegions batch size (default: 500)",
    )
    parser.add_argument(
        "--skip-missing", action="store_true",
        help="Skip years whose cache is absent instead of fetching from GEE",
    )
    args = parser.parse_args()

    years = sorted(args.years)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load sample + model ───────────────────────────────────────────────────
    print("Loading sample CSV …")
    df   = pd.read_csv(SAMPLE)
    lats = df["lat"].values
    lons = df["lon"].values
    N    = len(lats)
    print(f"  {N:,} points  |  columns: {list(df.columns)}")

    print("\nLoading XGBoost model …")
    bst = xgb.Booster()
    bst.load_model(str(XGB_MODEL))
    print("  ✓ loaded")

    # ── Process each year ─────────────────────────────────────────────────────
    print(f"\nProcessing {len(years)} year(s): {years}\n")
    t_total = time.time()

    for yr in years:
        cache = PRED_DIR / f"cwh_xgb_sample_{yr}_emb.npy"

        if args.skip_missing and not cache.exists():
            print(f"  [{yr}] no cache — skipping (--skip-missing)")
            continue

        # Special-case 2024: use the pre-computed prob column if no cache yet
        if yr == 2024 and "prob" in df.columns and not cache.exists():
            print(f"  [2024] using pre-computed prob column from CSV")
            df["prob_2024"] = df["prob"].astype(np.float32)
            n_valid = df["prob_2024"].notna().sum()
            print(
                f"  [2024] {n_valid:,} pts  "
                f"mean={df['prob_2024'].mean():.4f}  "
                f"P≥0.95={int((df['prob_2024'] >= 0.95).sum()):,}"
            )
        else:
            df[f"prob_{yr}"] = fetch_and_classify(
                yr, bst, lats, lons, args.batch_size
            )

    elapsed_total = (time.time() - t_total) / 60
    print(f"\nAll years processed in {elapsed_total:.1f} min")

    # Keep only years that were actually computed
    years_ok = [yr for yr in years if f"prob_{yr}" in df.columns]
    if not years_ok:
        print("No years succeeded — nothing to plot.", file=sys.stderr)
        sys.exit(1)

    # ── Summary table ─────────────────────────────────────────────────────────
    summary = build_summary(df, years_ok)

    print("\n─── Summary statistics ───────────────────────────────────────────")
    display_cols = ["n_valid", "mean", "median", "std",
                    "pct_ge_20", "pct_ge_50", "pct_ge_95"]
    print(summary[display_cols].round(4).to_string())

    print("\n─── Estimated yew-present area (P ≥ 0.95) ───────────────────────")
    for yr in years_ok:
        ha = summary.loc[yr, "ha_ge_95"]
        pct = summary.loc[yr, "pct_ge_95"]
        print(f"  {yr}: {ha:>12,.0f} ha  ({pct:.3f}% of CWH)")

    out_csv = TABLE_DIR / "yew_prob_all_years_summary.csv"
    summary.to_csv(out_csv)
    print(f"\nSummary table saved → {out_csv}")

    # ── Figure ────────────────────────────────────────────────────────────────
    out_fig = FIG_DIR / "yew_prob_all_years.png"
    make_figure(df, summary, years_ok, out_fig)


if __name__ == "__main__":
    main()
