#!/usr/bin/env python
"""
yew_normalized_trend.py
-----------------------
Apples-to-apples comparison of yew habitat across 2017-2024.

Problem
~~~~~~~
The XGBoost model was trained on **2024 embeddings only**.  Applying it to
other years' embeddings injects a per-year calibration bias (weather,
phenology, compositing differences all shift the embedding distribution).
Raw probabilities are NOT comparable across years — the 7× swing between
2022 (12.8%  P≥0.95) and 2024 (1.8%) is mostly instrument noise, not
real yew change.

Solution: three complementary normalisation strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. **Percentile-rank normalisation**
   Convert each year's probabilities to within-year percentile ranks
   (0–100).  Rank 95 = "top 5% of that year."  This factors out the
   year-level mean & variance shift.

2. **Paired-point Theil-Sen trends**
   For each of the 10,000 points, fit a robust (Theil-Sen) slope across
   the 8 years of percentile-normalised scores.  Negative slope =
   declining relative suitability.

3. **Spatial concordance**
   For the top-K% points in each year, compute the Jaccard overlap with
   every other year.  High overlap ≈ real signal; low overlap ≈ noise.
   The "consistent core" = points that appear in the top K% of ≥ 6/8 years.

Outputs
~~~~~~~
• Console summary with normalised metrics
• results/tables/yew_normalized_summary.csv
• results/tables/yew_point_trends.csv       (per-point trend slopes)
• results/figures/yew_normalized_trend.png   (6-panel figure)

Usage
-----
    python scripts/analysis/yew_normalized_trend.py
    python scripts/analysis/yew_normalized_trend.py --top-k 5
    python scripts/analysis/yew_normalized_trend.py --top-k 3 --min-years 7
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import folium
from scipy import stats

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
SAMPLE    = ROOT / "results" / "predictions" / "cwh_xgb_sample.csv"
XGB_MODEL = ROOT / "results" / "predictions" / "south_vi_large" / "xgb_raw_model_expanded.json"
PRED_DIR  = ROOT / "results" / "predictions"
FIG_DIR   = ROOT / "results" / "figures"
TABLE_DIR = ROOT / "results" / "tables"

CWH_AREA_HA = 3_507_194
ALL_YEARS   = list(range(2017, 2025))


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_probs(years: list[int]) -> pd.DataFrame:
    """Load sample CSV, classify every cached year, return DataFrame."""
    df = pd.read_csv(SAMPLE)
    N  = len(df)

    bst = xgb.Booster()
    bst.load_model(str(XGB_MODEL))

    for yr in years:
        cache = PRED_DIR / f"cwh_xgb_sample_{yr}_emb.npy"

        # 2024 has pre-computed probs in CSV
        if yr == 2024 and "prob" in df.columns and not cache.exists():
            df["prob_2024"] = df["prob"].astype(np.float32)
            continue

        if not cache.exists():
            print(f"  ⚠  {cache.name} missing — skipping {yr}", file=sys.stderr)
            continue

        embs  = np.load(cache)
        valid = np.isfinite(embs[:, 0])
        probs = np.full(N, np.nan, dtype=np.float32)
        if valid.any():
            X = np.nan_to_num(embs[valid], nan=0.0, posinf=0.0, neginf=0.0)
            probs[valid] = bst.predict(xgb.DMatrix(X)).astype(np.float32)
        df[f"prob_{yr}"] = probs

    return df


def percentile_rank(series: pd.Series) -> pd.Series:
    """Return within-series percentile rank (0–100).  NaN stays NaN."""
    return series.rank(pct=True, na_option="keep") * 100


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


# ── Analysis ─────────────────────────────────────────────────────────────────

def run_analysis(df: pd.DataFrame, years: list[int], top_k: int, min_years: int):
    N = len(df)
    prob_cols = [f"prob_{yr}" for yr in years]
    rank_cols = [f"rank_{yr}" for yr in years]

    # ── 1. Percentile-rank normalisation ──────────────────────────────────────
    for yr in years:
        df[f"rank_{yr}"] = percentile_rank(df[f"prob_{yr}"])

    # ── 2. Per-year summary (normalised) ──────────────────────────────────────
    rows = []
    for yr in years:
        raw = df[f"prob_{yr}"].dropna()
        rnk = df[f"rank_{yr}"].dropna()
        rows.append({
            "year":              yr,
            "raw_mean":          raw.mean(),
            "raw_median":        raw.median(),
            "raw_std":           raw.std(),
            "raw_pct_ge_95":     100 * (raw >= 0.95).mean(),
            # These next ones should be ~constant if ranking works
            "rank_mean":         rnk.mean(),
            "rank_median":       rnk.median(),
            # Top-K count (should be identical by definition)
            "top_k_threshold":   raw.quantile(1 - top_k / 100),
            "top_k_count":       int((rnk >= (100 - top_k)).sum()),
        })
    summary = pd.DataFrame(rows).set_index("year")

    # ── 3. Spatial concordance matrix ─────────────────────────────────────────
    top_k_sets = {}
    for yr in years:
        cutoff = 100 - top_k
        top_k_sets[yr] = set(df.index[df[f"rank_{yr}"] >= cutoff])

    n_years = len(years)
    concordance = pd.DataFrame(
        np.zeros((n_years, n_years)),
        index=years, columns=years, dtype=float,
    )
    for i, y1 in enumerate(years):
        for j, y2 in enumerate(years):
            concordance.iloc[i, j] = jaccard(top_k_sets[y1], top_k_sets[y2])

    # ── 4. Consistent core ────────────────────────────────────────────────────
    in_top_k_count = pd.Series(0, index=df.index, dtype=int)
    for yr in years:
        in_top_k_count += (df[f"rank_{yr}"] >= (100 - top_k)).astype(int)

    df["top_k_years"]    = in_top_k_count
    df["consistent_core"] = in_top_k_count >= min_years

    n_core = df["consistent_core"].sum()
    core_ha = (n_core / N) * CWH_AREA_HA

    # ── 5. Per-point Theil-Sen trend on percentile ranks ──────────────────────
    rank_matrix = df[rank_cols].values          # (N, n_years)
    year_arr    = np.array(years, dtype=float)

    slopes     = np.full(N, np.nan, dtype=np.float32)
    intercepts = np.full(N, np.nan, dtype=np.float32)
    pvalues    = np.full(N, np.nan, dtype=np.float32)

    for i in range(N):
        y = rank_matrix[i, :]
        valid = np.isfinite(y)
        if valid.sum() >= 3:
            res          = stats.theilslopes(y[valid], year_arr[valid])
            slopes[i]    = res.slope
            intercepts[i] = res.intercept
            # Also get a p-value via Kendall's tau
            tau_result   = stats.kendalltau(year_arr[valid], y[valid])
            pvalues[i]   = tau_result.pvalue

    df["trend_slope"]     = slopes       # rank-units per year
    df["trend_intercept"] = intercepts
    df["trend_pvalue"]    = pvalues

    # Classify trend
    sig = df["trend_pvalue"] < 0.10
    df["trend_class"] = "stable"
    df.loc[sig & (df["trend_slope"] < -0.5), "trend_class"] = "declining"
    df.loc[sig & (df["trend_slope"] >  0.5), "trend_class"] = "increasing"

    return summary, concordance, n_core, core_ha


# ── Figure ────────────────────────────────────────────────────────────────────

def make_figure(
    df: pd.DataFrame, summary: pd.DataFrame, concordance: pd.DataFrame,
    years: list[int], top_k: int, n_core: int, core_ha: float, out_path: Path,
):
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(
        "Western Yew — Normalised Trend Analysis  (2017–2024)\n"
        "Percentile-rank normalisation removes per-year calibration bias",
        fontsize=13, fontweight="bold", y=0.99,
    )

    cmap = plt.colormaps["plasma"]
    colours = [mcolors.to_hex(cmap(i / max(len(years) - 1, 1)))
               for i in range(len(years))]

    # ── A: Raw vs normalised means ─────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(years, summary["raw_mean"], "o-", lw=2, color="#e74c3c",
            label="Raw mean probability")
    ax.fill_between(
        years,
        summary["raw_mean"] - summary["raw_std"],
        summary["raw_mean"] + summary["raw_std"],
        color="#e74c3c", alpha=0.12, label="Raw ±1 SD",
    )
    ax.set_ylabel("Raw probability", fontsize=9)
    ax.set_xlabel("Year")
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], rotation=45)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_title("(A) Raw probability — year-level bias visible", fontsize=10)

    # ── B: Per-year probability threshold for top K% ────────────────────────
    ax = axes[0, 1]
    ax.bar(years, summary["top_k_threshold"], color=colours, alpha=0.8,
           edgecolor="black", lw=0.5)
    ax.set_ylabel(f"Raw p threshold for top-{top_k}%", fontsize=9)
    ax.set_xlabel("Year")
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], rotation=45)
    ax.set_title(
        f"(B) Top-{top_k}% threshold per year\n"
        "(calibration drift — why raw P is misleading)",
        fontsize=10,
    )
    ax.grid(axis="y", alpha=0.3)

    # ── C: Concordance heatmap ────────────────────────────────────────────────
    ax = axes[0, 2]
    im = ax.imshow(concordance.values, cmap="YlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([str(y) for y in years], fontsize=8, rotation=45)
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels([str(y) for y in years], fontsize=8)
    for i in range(len(years)):
        for j in range(len(years)):
            v = concordance.values[i, j]
            colour = "white" if v < 0.5 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color=colour)
    fig.colorbar(im, ax=ax, label="Jaccard overlap", shrink=0.8)
    ax.set_title(
        f"(C) Spatial concordance — top-{top_k}%\n"
        "(high = same points identified each year)",
        fontsize=10,
    )

    # ── D: Distribution of per-point trend slopes ────────────────────────────
    ax = axes[1, 0]
    slopes = df["trend_slope"].dropna()
    ax.hist(slopes, bins=80, color="#3498db", alpha=0.7, edgecolor="none")
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.axvline(slopes.median(), color="#e74c3c", lw=2,
               label=f"Median = {slopes.median():+.2f} rank/yr")
    n_dec = (df["trend_class"] == "declining").sum()
    n_inc = (df["trend_class"] == "increasing").sum()
    n_stb = (df["trend_class"] == "stable").sum()
    ax.set_xlabel("Theil-Sen slope (rank-points per year)")
    ax.set_ylabel("# points")
    ax.set_title(
        f"(D) Per-point rank trends (N={len(slopes):,})\n"
        f"Declining: {n_dec:,}  |  Stable: {n_stb:,}  |  Increasing: {n_inc:,}",
        fontsize=10,
    )
    ax.legend(fontsize=9)

    # ── E: Consistent core histogram ──────────────────────────────────────────
    ax = axes[1, 1]
    counts = df["top_k_years"].value_counts().sort_index()
    ax.bar(counts.index, counts.values, color="#2ecc71", alpha=0.8,
           edgecolor="black", lw=0.5)
    ax.set_xlabel(f"# years in top-{top_k}%")
    ax.set_ylabel("# points")
    ax.set_title(
        f"(E) Point persistence in top-{top_k}%\n"
        f"Consistent core (≥ {len(years)-2}/{len(years)} yrs): "
        f"{n_core:,} pts → {core_ha:,.0f} ha",
        fontsize=10,
    )
    ax.grid(axis="y", alpha=0.3)

    # ── F: BEC subzone consistency ────────────────────────────────────────────
    ax = axes[1, 2]
    rank_cols = [f"rank_{yr}" for yr in years]
    bec_rank_mean = (
        df.dropna(subset=rank_cols)
        .groupby("map_label")[rank_cols]
        .mean()
    )
    # Inter-year standard deviation per subzone (lower = more consistent)
    bec_rank_mean["inter_year_sd"]   = bec_rank_mean[rank_cols].std(axis=1)
    bec_rank_mean["overall_rank"]    = bec_rank_mean[rank_cols].mean(axis=1)
    bec_rank_mean["n_pts"] = df.groupby("map_label").size()
    bec_plot = (
        bec_rank_mean.query("n_pts >= 30")
        .sort_values("overall_rank", ascending=False)
        .head(20)
    )

    x = range(len(bec_plot))
    ax.barh(
        list(x), bec_plot["overall_rank"],
        xerr=bec_plot["inter_year_sd"],
        color="#9b59b6", alpha=0.7, ecolor="#555", capsize=3,
    )
    ax.set_yticks(list(x))
    ax.set_yticklabels(bec_plot.index, fontsize=7)
    ax.set_xlabel("Mean percentile rank (± inter-year SD)")
    ax.set_title(
        "(F) BEC subzone mean rank ± year-to-year SD\n"
        "(small bars = consistent across years)",
        fontsize=10,
    )
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out_path}")
    plt.close()


# ── Interactive map ───────────────────────────────────────────────────────────

def make_map(
    df: pd.DataFrame, years: list[int], top_k: int, out_path: Path,
):
    """Build a folium map with layers for trend class, consistent core, and rank."""

    # ── Subsample for browser performance ─────────────────────────────────────
    plot_df = df.copy()
    if len(plot_df) > 6000:
        # Always keep interesting points; random-sample the rest
        interesting = plot_df[
            (plot_df["trend_class"] != "stable") | plot_df["consistent_core"]
        ]
        remaining = plot_df.drop(interesting.index)
        n_fill = min(6000 - len(interesting), len(remaining))
        plot_df = pd.concat([
            interesting,
            remaining.sample(n_fill, random_state=42) if n_fill > 0 else remaining.iloc[:0],
        ])
    print(f"  Map: plotting {len(plot_df):,} points ({len(df):,} total)")

    centre = [plot_df["lat"].mean(), plot_df["lon"].mean()]
    m = folium.Map(location=centre, zoom_start=7, tiles=None)

    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri WorldImagery",
        name="Satellite (Esri)",
    ).add_to(m)

    # ── Colour helpers ────────────────────────────────────────────────────────
    trend_colour = {
        "declining":  "#d73027",
        "stable":     "#999999",
        "increasing": "#4575b4",
    }

    rdbu = mcolors.LinearSegmentedColormap.from_list(
        "RdBu", ["#d73027", "#f7f7f7", "#4575b4"]
    )
    slope_norm = mcolors.TwoSlopeNorm(vmin=-3, vcenter=0.0, vmax=3)

    def slope_hex(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "#888888"
        return mcolors.to_hex(rdbu(slope_norm(float(val))))

    def tooltip(row):
        parts = [
            f"<b>{row.get('map_label', '?')}</b>",
            f"Trend: <b>{row['trend_class']}</b> (slope={row['trend_slope']:+.2f})",
            f"p-value: {row['trend_pvalue']:.3f}",
            f"Core ({top_k}% in ≥6/8 yr): {'YES' if row['consistent_core'] else 'no'}",
            f"Top-{top_k}% count: {int(row['top_k_years'])}/8 years",
            "---",
        ]
        for yr in years:
            p = row.get(f'prob_{yr}', float('nan'))
            r = row.get(f'rank_{yr}', float('nan'))
            parts.append(f"{yr}: P={p:.3f}  rank={r:.1f}")
        return "<br>".join(parts)

    # ── Layer 1: Trend classification ─────────────────────────────────────────
    fg_trend = folium.FeatureGroup(name="Trend class (declining / stable / increasing)", show=True)
    for _, row in plot_df.iterrows():
        colour = trend_colour.get(row["trend_class"], "#888")
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4 if row["trend_class"] == "stable" else 6,
            color=colour, fill=True, fill_color=colour,
            fill_opacity=0.75, weight=0.4,
            tooltip=tooltip(row),
        ).add_to(fg_trend)
    fg_trend.add_to(m)

    # ── Layer 2: Trend slope (continuous) ─────────────────────────────────────
    fg_slope = folium.FeatureGroup(name="Trend slope (red=declining, blue=increasing)", show=False)
    for _, row in plot_df.iterrows():
        colour = slope_hex(row["trend_slope"])
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4,
            color=colour, fill=True, fill_color=colour,
            fill_opacity=0.80, weight=0.3,
            tooltip=tooltip(row),
        ).add_to(fg_slope)
    fg_slope.add_to(m)

    # ── Layer 3: Consistent core ──────────────────────────────────────────────
    fg_core = folium.FeatureGroup(name=f"Consistent core (top-{top_k}% in ≥6/8 yrs)", show=False)
    core_pts = plot_df[plot_df["consistent_core"]]
    for _, row in core_pts.iterrows():
        n_yr = int(row["top_k_years"])
        opacity = 0.6 + 0.05 * n_yr
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=6,
            color="#e74c3c", fill=True, fill_color="#e74c3c",
            fill_opacity=min(opacity, 1.0), weight=0.5,
            tooltip=tooltip(row),
        ).add_to(fg_core)
    fg_core.add_to(m)

    # ── Layer 4: Declining only ───────────────────────────────────────────────
    fg_dec = folium.FeatureGroup(name="Declining points only", show=False)
    dec_pts = plot_df[plot_df["trend_class"] == "declining"]
    for _, row in dec_pts.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=5,
            color="#d73027", fill=True, fill_color="#d73027",
            fill_opacity=0.85, weight=0.5,
            tooltip=tooltip(row),
        ).add_to(fg_dec)
    fg_dec.add_to(m)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_html = f"""
    <div style="position:fixed; bottom:40px; left:40px; z-index:9999;
                background:white; padding:10px 14px; border-radius:8px;
                border:1px solid #aaa; font-size:12px; line-height:1.8;
                max-width:220px;">
    <b>Normalised Trend</b><br>
    <span style="color:#d73027;">&#9679;</span> Declining (slope&lt;-0.5, p&lt;0.10)<br>
    <span style="color:#999;">&#9679;</span> Stable<br>
    <span style="color:#4575b4;">&#9679;</span> Increasing (slope&gt;+0.5, p&lt;0.10)<br>
    <hr style="margin:4px 0;">
    <span style="color:#e74c3c;">&#9679;</span> Consistent core (top-{top_k}% ≥6/8 yr)<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(m)

    m.save(str(out_path))
    print(f"  Map saved → {out_path}")


# ── Console report ────────────────────────────────────────────────────────────

def print_report(
    df: pd.DataFrame, summary: pd.DataFrame, concordance: pd.DataFrame,
    years: list[int], top_k: int, n_core: int, core_ha: float, min_years: int,
):
    N = len(df)
    n_dec = (df["trend_class"] == "declining").sum()
    n_inc = (df["trend_class"] == "increasing").sum()
    n_stb = (df["trend_class"] == "stable").sum()
    med_slope = df["trend_slope"].median()

    print()
    print("=" * 72)
    print("  NORMALISED TREND ANALYSIS — Western Yew Habitat (2017–2024)")
    print("=" * 72)

    print(f"""
WHY RAW PROBABILITIES ARE MISLEADING
  The XGBoost model was trained on 2024 embeddings only.  Applying it to
  other years' embeddings introduces per-year calibration bias: the same
  forest patch gets different raw scores depending on weather, phenology,
  and sensor compositing.  The 2022 spike to 12.8% is NOT 7× more yew;
  it's a year whose image characteristics happen to trigger the model.

APPROACH
  1. Convert raw probs to within-year percentile ranks (0–100)
     → removes year-level bias; rank 95 = "top 5% of THAT year"
  2. Track each point's rank trajectory over 8 years (Theil-Sen slope)
  3. Identify "consistent core" habitat = top-{top_k}% in ≥ {min_years}/{len(years)} years
""")

    print("─── Raw vs Normalised (showing the calibration problem) ────────────")
    disp = summary[["raw_mean", "raw_std", "raw_pct_ge_95", "top_k_threshold"]].copy()
    disp.columns = ["Raw Mean", "Raw SD", "Raw P≥0.95 (%)", f"Top-{top_k}% Threshold"]
    print(disp.round(4).to_string())

    print(f"\n  Note: the 'Top-{top_k}% Threshold' column shows the raw probability")
    print(f"  that defines the top {top_k}% in each year — these vary from")
    print(f"  {summary['top_k_threshold'].min():.3f} to {summary['top_k_threshold'].max():.3f}, "
          f"proving raw P thresholds aren't comparable.\n")

    print(f"─── Spatial Concordance (top-{top_k}% Jaccard overlap) ────────────────")
    print(concordance.round(3).to_string())
    # Mean off-diagonal Jaccard
    mask = np.ones(concordance.shape, dtype=bool)
    np.fill_diagonal(mask, False)
    mean_j = concordance.values[mask].mean()
    print(f"\n  Mean off-diagonal Jaccard: {mean_j:.3f}")
    if mean_j > 0.50:
        print(f"  → GOOD spatial consistency — the same areas rank high across years")
    elif mean_j > 0.30:
        print(f"  → MODERATE consistency — some stable signal, some noise")
    else:
        print(f"  → LOW consistency — raw model struggles to identify the same areas")

    print(f"\n─── Consistent Core (top-{top_k}% in ≥ {min_years}/{len(years)} years) ────────")
    print(f"  Points in core:  {n_core:,} / {N:,}  ({100*n_core/N:.2f}%)")
    print(f"  Estimated area:  {core_ha:,.0f} ha  ({100*core_ha/CWH_AREA_HA:.3f}% of CWH)")

    top_bec = (
        df[df["consistent_core"]]
        .groupby("map_label")
        .size()
        .sort_values(ascending=False)
        .head(10)
    )
    if not top_bec.empty:
        print(f"\n  Top BEC subzones in the consistent core:")
        for bec, cnt in top_bec.items():
            pct_of_bec = 100 * cnt / (df["map_label"] == bec).sum()
            print(f"    {bec:15s}  {cnt:>5,} pts  ({pct_of_bec:.1f}% of subzone sample)")

    print(f"\n─── Per-Point Rank Trends (Theil-Sen slope) ────────────────────────")
    print(f"  Declining (slope < -0.5, p < 0.10):   {n_dec:>5,} points  ({100*n_dec/N:.1f}%)")
    print(f"  Stable:                                {n_stb:>5,} points  ({100*n_stb/N:.1f}%)")
    print(f"  Increasing (slope > +0.5, p < 0.10):  {n_inc:>5,} points  ({100*n_inc/N:.1f}%)")
    print(f"  Median slope: {med_slope:+.3f} rank-points/year")

    if n_dec > n_inc * 1.5:
        print(f"\n  ⚠  More declining than increasing points — suggests real habitat loss")
    elif n_inc > n_dec * 1.5:
        print(f"\n  ↑  More increasing than declining — possibly habitat expansion")
    else:
        print(f"\n  ≈  Roughly balanced — no strong directional trend detected")

    # Translate decline to area
    declining_ha = (n_dec / N) * CWH_AREA_HA
    increasing_ha = (n_inc / N) * CWH_AREA_HA
    print(f"\n  Extrapolated (crude) to CWH land base:")
    print(f"    Declining suitability:  ~{declining_ha:,.0f} ha")
    print(f"    Increasing suitability: ~{increasing_ha:,.0f} ha")
    print(f"    Net:                    ~{increasing_ha - declining_ha:+,.0f} ha")

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=ALL_YEARS, metavar="YEAR",
        help=f"Years to include (default: {ALL_YEARS[0]}–{ALL_YEARS[-1]})",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Top-K%% for concordance & core analysis (default: 5)",
    )
    parser.add_argument(
        "--min-years", type=int, default=None,
        help="Min years in top-K%% to qualify as 'consistent core' "
             "(default: n_years - 2)",
    )
    args = parser.parse_args()

    years     = sorted(args.years)
    top_k     = args.top_k
    min_years = args.min_years if args.min_years is not None else max(len(years) - 2, 1)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load & classify ──────────────────────────────────────────────────────
    print("Loading sample + classifying …")
    t0 = time.time()
    df = load_probs(years)
    print(f"  {len(df):,} points, {len(years)} years  ({time.time()-t0:.1f}s)")

    # Only keep years we actually have
    years = [yr for yr in years if f"prob_{yr}" in df.columns]
    if len(years) < 3:
        print("Need at least 3 years for trend analysis.", file=sys.stderr)
        sys.exit(1)

    # ── Run analysis ──────────────────────────────────────────────────────────
    summary, concordance, n_core, core_ha = run_analysis(df, years, top_k, min_years)

    # ── Print report ──────────────────────────────────────────────────────────
    print_report(df, summary, concordance, years, top_k, n_core, core_ha, min_years)

    # ── Save tables ───────────────────────────────────────────────────────────
    out_summary = TABLE_DIR / "yew_normalized_summary.csv"
    summary.to_csv(out_summary)
    print(f"Summary table  → {out_summary}")

    out_trends = TABLE_DIR / "yew_point_trends.csv"
    save_cols = [
        "lat", "lon", "bec_zone", "map_label",
        *[f"prob_{yr}" for yr in years],
        *[f"rank_{yr}" for yr in years],
        "trend_slope", "trend_pvalue", "trend_class",
        "top_k_years", "consistent_core",
    ]
    df[[c for c in save_cols if c in df.columns]].to_csv(out_trends, index=False)
    print(f"Point trends   → {out_trends}")

    # ── Figure ────────────────────────────────────────────────────────────────
    out_fig = FIG_DIR / "yew_normalized_trend.png"
    make_figure(df, summary, concordance, years, top_k, n_core, core_ha, out_fig)

    # ── Interactive map ───────────────────────────────────────────────────────
    out_map = FIG_DIR / "yew_normalized_trend_map.html"
    make_map(df, years, top_k, out_map)


if __name__ == "__main__":
    main()
