#!/usr/bin/env python3
"""
Compare observed Pacific yew (Taxus brevifolia) size distribution to the
expected reverse-J (de Liocourt) model for a stable, self-sustaining population.

Field measurements are circumference at breast height (CBH, cm);
converted to DBH (cm) via DBH = CBH / π before analysis.

Tests the large-diameter (>30 cm DBH) tail for a deficit relative to the stable
de Liocourt expectation (binomial + chi-square, robust across q = 1.4-1.6) and
fits a whole-population reverse-J slope (q with bootstrap CI). A steeper-than-
stable q together with a large-tree deficit is consistent with selective removal
of large, bark-rich individuals (e.g. the 1989-1993 taxol harvest); a recruitment
bottleneck would instead deplete the smallest classes (not observed here).

Literature references:
  - de Liocourt, F. (1898). De l'aménagement des sapinières. Bull. Soc. For.
    Franche-Comté et Belfort, July: 396-409.  [q-ratio model]
  - Meyer, H.A. (1952). Structure, growth, and drain in balanced uneven-aged
    forests. J. Forestry 50(2): 85-92.
  - Graham, R.T. (1994). Taxus brevifolia Nutt. In: Burns & Honkala (eds.)
    Silvics of North America vol. 1. USDA FS Agric. Handb. 654, pp. 573-579.
  - Bolsinger, C.L. & Jaramillo, A.E. (1990). Taxus of the Pacific Coast.
    USDA FS Resource Bull. PNW-RB-172. Portland, OR.
  - Pacific Yew Recovery Team (1995). Recovery Plan for Pacific Yew in British
    Columbia. BC Ministry of Environment, Victoria.

Run:
    conda run -n yew_pytorch python scripts/analysis/tree_size_distribution.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import binomtest, chisquare

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH = "data/sample_tree_size.csv"
OUT_PNG   = "results/figures/tree_size_distribution.png"
OUT_PDF   = "results/figures/tree_size_distribution.pdf"

# ── Load data ──────────────────────────────────────────────────────────────────
df  = pd.read_csv(DATA_PATH)
cbh = df["Size (cm)"].values          # field measurement: circumference (cm)
d   = cbh / np.pi                      # convert to DBH for de Liocourt analysis
n   = len(d)

print(f"Sample (n={n}):")
print(f"  CBH — min={cbh.min():.0f} cm, max={cbh.max():.0f} cm, "
      f"mean={cbh.mean():.1f} cm, median={np.median(cbh):.1f} cm")
print(f"  DBH — min={d.min():.1f} cm, max={d.max():.1f} cm, "
      f"mean={d.mean():.1f} cm, median={np.median(d):.1f} cm")

# ── Bin into 10-cm diameter classes (standard BC forestry convention) ──────────
BIN_W   = 10
bins    = np.arange(0, d.max() + BIN_W + 1, BIN_W)
centers = (bins[:-1] + bins[1:]) / 2
obs, _  = np.histogram(d, bins=bins)

# ── Literature reference distribution (de Liocourt stable population) ──────────
# Bolsinger & Jaramillo (1990) and Graham (1994) document that Pacific yew in
# intact old-growth CWH/ICH forests follows a reverse-J with q ≈ 1.4–1.6 per
# 10-cm class. We use q = 1.5 as the central reference and report the large-tree
# result across the full 1.4–1.6 range so it does not hinge on a single value.
Q_LIT    = 1.5
def expected_counts(q):
    raw = q ** (-np.arange(len(centers)))
    return raw * (n / raw.sum()), raw / raw.sum()
lit_exp, lit_p = expected_counts(Q_LIT)

# ── Large-tree deficit: the a-priori prediction concerns the >30 cm tail ───────
THRESH   = 30
large    = centers > THRESH
obs_lg   = int(obs[large].sum())
print(f"\nLarge-tree (>{THRESH} cm) deficit — observed {obs_lg} stems")
print("  expected >30 cm by reference q:")
for q in (1.4, 1.5, 1.6):
    e, _ = expected_counts(q)
    print(f"    q={q}: {e[large].sum():5.1f}")

# Exact binomial test (one-sided deficit) of the >30 cm count vs each reference q
print("\nBinomial test (one-sided, deficit) of >30 cm count:")
for q in (1.4, 1.5, 1.6):
    _, p = expected_counts(q)
    bt = binomtest(obs_lg, n, p[large].sum(), alternative="less")
    print(f"    q={q}: P(>30cm)={p[large].sum():.3f}  p={bt.pvalue:.2e}")
_, p15 = expected_counts(Q_LIT)
binom_p = binomtest(obs_lg, n, p15[large].sum(), alternative="less").pvalue

# χ² goodness-of-fit on binned counts (3 classes <30 cm + pooled ≥30 cm)
obs_pool = np.array([obs[0], obs[1], obs[2], obs[centers >= THRESH].sum()], float)
exp_pool = np.array([lit_exp[0], lit_exp[1], lit_exp[2],
                     lit_exp[centers >= THRESH].sum()], float)
exp_pool *= obs_pool.sum() / exp_pool.sum()
chi2_stat, chi2_p = chisquare(obs_pool, exp_pool)
print(f"\nχ² goodness-of-fit (pooled, q={Q_LIT}): "
      f"χ²={chi2_stat:.1f}, df={len(obs_pool)-1}, p={chi2_p:.2e}")

# ── Whole-population reverse-J slope (well-constrained; direction interpretable) ─
def fit_pop_q(counts):
    """Empirical de Liocourt q from an OLS log-linear fit over occupied classes."""
    idx = np.arange(len(counts))
    m   = counts > 0
    if m.sum() < 2:
        return np.nan
    slope = np.polyfit(idx[m], np.log(counts[m].astype(float)), 1)[0]
    return float(np.exp(-slope))           # q per 10-cm class

q_pop = fit_pop_q(obs)
rng_boot = np.random.default_rng(0)
boot_q   = [fit_pop_q(np.histogram(rng_boot.choice(d, size=n, replace=True),
                                   bins=bins)[0]) for _ in range(2000)]
boot_q   = np.array([q for q in boot_q if np.isfinite(q)])
q_lo, q_hi = np.percentile(boot_q, [2.5, 97.5])
print(f"\nWhole-population fitted q = {q_pop:.2f} "
      f"(95% bootstrap CI {q_lo:.2f}–{q_hi:.2f}); literature stable q = {Q_LIT}")
print("  q_pop > q_lit ⇒ distribution declines faster toward large classes "
      "(large-adult depletion)")

# Reverse-J curve at the fitted population q, scaled to n, for plotting
fit_raw   = q_pop ** (-np.arange(len(centers)))
fit_curve = fit_raw * (n / fit_raw.sum())

# ── Per-class table ───────────────────────────────────────────────────────────
print("\n10-cm class  |  Observed  |  Expected (q=1.5)  |  Surplus/Deficit")
print("-" * 60)
for i in range(len(centers)):
    if obs[i] > 0 or lit_exp[i] > 0.5:
        print(f"  {int(bins[i]):4d}–{int(bins[i+1]):3d} cm  |"
              f"    {obs[i]:3d}     |     {lit_exp[i]:6.1f}        |  {obs[i]-lit_exp[i]:+6.1f}")

# ── Figure ─────────────────────────────────────────────────────────────────────
OLIVE   = "#606c38"
RED     = "#bc3425"
BLUE    = "#264653"
AMBER   = "#e9c46a"

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                         gridspec_kw={"width_ratios": [2, 1]})

fig.suptitle(
    "Pacific Yew   DBH Distribution (converted from CBH): Observed vs. Expected Stable Population",
    fontsize=13, fontweight="bold", y=1.02)

# ── Left panel: bar chart ──────────────────────────────────────────────────────
ax = axes[0]

ax.bar(centers, obs, width=BIN_W * 0.7, color=OLIVE, alpha=0.85,
       label=f"Observed (n = {n})", edgecolor="white", linewidth=0.4, zorder=3)

ax.plot(centers, lit_exp, "--o", color=RED, ms=5, lw=1.9,
        label=f"Expected: de Liocourt q = {Q_LIT} per 10 cm\n"
              "(Bolsinger & Jaramillo 1990; Graham 1994)")

ax.plot(centers, fit_curve, "-.s", color=BLUE, ms=4, lw=1.5,
        label=f"Reverse-J fitted to whole sample\n"
              f"(q = {q_pop:.1f}, 95% CI {q_lo:.1f}–{q_hi:.1f})")

# Shade the large-tree deficit (>30 cm: observed below the stable expectation)
lg_mask = centers > THRESH
ax.fill_between(centers[lg_mask],
                obs[lg_mask].astype(float),
                lit_exp[lg_mask],
                where=(lit_exp[lg_mask] > obs[lg_mask]),
                alpha=0.30, color=RED,
                label=f"Large-tree deficit > {THRESH} cm "
                      f"({obs_lg} obs vs ~{lit_exp[large].sum():.0f} expected)")

ax.axvline(THRESH, color=AMBER, lw=1.8, ls=":", alpha=0.9,
           label=f"{THRESH} cm threshold")

ax.set_xlabel("Diameter at breast height (cm)", fontsize=11)
ax.set_ylabel("Number of trees", fontsize=11)
ax.set_xlim(-5, bins[-1] + 5)
ax.set_ylim(0, max(obs.max(), lit_exp.max()) * 1.18)
ax.legend(fontsize=8.5, loc="upper right")
ax.grid(axis="y", alpha=0.25, zorder=0)

# Annotation box
ax.text(0.02, 0.97,
        f"Observed > {THRESH} cm:  {obs_lg} stems\n"
        f"Expected > {THRESH} cm:  {lit_exp[large].sum():.0f} (q=1.5; 27–38 over q=1.4–1.6)\n"
        f"Binomial p = {binom_p:.1e}\n"
        f"χ² = {chi2_stat:.0f}, df = {len(obs_pool)-1}, p = {chi2_p:.1e}\n"
        f"Population q = {q_pop:.1f} (CI {q_lo:.1f}–{q_hi:.1f}) vs stable {Q_LIT}",
        transform=ax.transAxes, va="top", ha="left", fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.85))

# ── Right panel: cumulative distribution ──────────────────────────────────────
ax2 = axes[1]

obs_order  = np.sort(d)
obs_cdf    = np.arange(1, n + 1) / n

# Expected CDF from the theoretical q=1.5 class probabilities (step at bin edges)
exp_cdf_x  = np.repeat(bins[1:], 2)[:-1]
exp_cdf_y  = np.repeat(np.cumsum(lit_p), 2)[1:]

ax2.plot(obs_order, obs_cdf, "-", color=OLIVE, lw=2, label="Observed CDF")
ax2.plot(exp_cdf_x, exp_cdf_y, "--", color=RED, lw=2,
         label=f"Expected CDF (q = {Q_LIT})")

obs_med = np.median(d)
ax2.axvline(obs_med, color=OLIVE, ls=":", lw=1.5,
            label=f"Observed median = {obs_med:.0f} cm")
ax2.axhline(0.5, color="lightgray", ls="-", lw=0.8, zorder=0)

ax2.set_xlabel("Diameter at breast height (cm)", fontsize=11)
ax2.set_ylabel("Cumulative proportion", fontsize=11)
ax2.set_xlim(0, 80)
ax2.set_title("Cumulative distribution (0–80 cm)", fontsize=9)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.25)

plt.tight_layout()

os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")
print(f"\nSaved: {OUT_PNG}")
print(f"Saved: {OUT_PDF}")
plt.show()
