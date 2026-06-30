#!/usr/bin/env python3
"""
FAIB PSP de Liocourt analysis for Pacific yew (TW), compared to the
n=120 field sample from data/sample_tree_size.csv.

Steps:
  1. Deduplicate FAIB tree records — same tree (SITE+PLOT+TREE_NO) may appear
     across multiple VISIT_NUMBERs in separate CLSTR_IDs; keep latest visit.
  2. Run the same reverse-J (de Liocourt) analysis as tree_size_distribution.py
     on the deduplicated live TW stems.
  3. Compute conditional density (where TW is present) from PHF_TREE expansion
     factors — the correct metric for habitat-to-population conversion.
  4. Produce a two-panel comparison figure:
       Left  — size-class histograms (FAIB + field, both vs q=1.5 reference)
       Right — per-plot TW density distribution (all & mature ≥10 cm DBH)

Run:
    conda run -n yew_pytorch python scripts/analysis/population_estimate.py
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
ROOT          = "/home/jericho/yew_project"
TREE_CSV      = f"{ROOT}/data/raw/faib_tree_detail.csv"
FIELD_CSV     = f"{ROOT}/data/sample_tree_size.csv"
OUT_PNG       = f"{ROOT}/results/figures/faib_size_distribution.png"
OUT_PDF       = f"{ROOT}/results/figures/faib_size_distribution.pdf"
PROB_MASS_HA  = 47_534.0   # headline continuous probability-mass habitat (ha)

# ── Load & deduplicate FAIB ────────────────────────────────────────────────────
print("Loading FAIB data …")
df_raw = pd.read_csv(TREE_CSV,
                     usecols=['SITE_IDENTIFIER','CLSTR_ID','VISIT_NUMBER','PLOT',
                               'TREE_NO','SPECIES','DBH','LV_D','PHF_TREE'],
                     low_memory=False)

tw_all = df_raw[df_raw['SPECIES'] == 'TW'].copy()

# Deduplicate: (SITE_IDENTIFIER, PLOT, TREE_NO) is a unique tree across PSP
# visits; keep the most recent VISIT_NUMBER so each tree appears once.
tw = (tw_all.sort_values('VISIT_NUMBER')
            .groupby(['SITE_IDENTIFIER', 'PLOT', 'TREE_NO'], as_index=False)
            .last())

print(f"  Raw TW records : {len(tw_all)}")
print(f"  Unique trees   : {len(tw)}  (removed {len(tw_all)-len(tw)} repeat visits)")
print(f"  Live           : {(tw['LV_D']=='L').sum()}")
print(f"  Dead           : {(tw['LV_D']=='D').sum()}")

live = tw[tw['LV_D'] == 'L'].copy()
live['PHF_TREE'] = pd.to_numeric(live['PHF_TREE'], errors='coerce')
live_dbh = live['DBH'].dropna().values

# ── Load field sample ─────────────────────────────────────────────────────────
field = pd.read_csv(FIELD_CSV)
cbh_field = field["Size (cm)"].values
dbh_field = cbh_field / np.pi          # CBH → DBH

# ── Binning (same 10-cm scheme for both) ─────────────────────────────────────
BIN_W  = 10
MAX_D  = max(live_dbh.max(), dbh_field.max()) + BIN_W
bins   = np.arange(0, MAX_D + 1, BIN_W)
ctrs   = (bins[:-1] + bins[1:]) / 2

obs_faib,  _ = np.histogram(live_dbh,  bins=bins)
obs_field, _ = np.histogram(dbh_field, bins=bins)

n_faib  = int(obs_faib.sum())
n_field = int(obs_field.sum())

# ── de Liocourt reference & tests (q = 1.5) ───────────────────────────────────
Q_LIT  = 1.5
THRESH = 30   # cm — large-tree deficit threshold

def expected_counts(q, n_total):
    raw = q ** (-np.arange(len(ctrs)))
    return raw * (n_total / raw.sum()), raw / raw.sum()

def large_tree_test(obs, n_total, q=Q_LIT):
    lit_exp, lit_p = expected_counts(q, n_total)
    large = ctrs > THRESH
    obs_lg = int(obs[large].sum())
    bt = binomtest(obs_lg, n_total, lit_p[large].sum(), alternative="less")
    # chi2 on 3 occupied classes + pooled ≥30 cm
    obs_pool = np.array([obs[0], obs[1], obs[2], obs[ctrs >= THRESH].sum()], float)
    exp_pool = np.array([lit_exp[0], lit_exp[1], lit_exp[2],
                         lit_exp[ctrs >= THRESH].sum()], float)
    exp_pool *= obs_pool.sum() / exp_pool.sum()
    chi2_s, chi2_p = chisquare(obs_pool, exp_pool)
    return obs_lg, lit_exp, lit_p, bt.pvalue, chi2_s, chi2_p

def fit_pop_q(obs_counts):
    """Whole-population OLS q on occupied 10-cm classes."""
    idx = np.arange(len(obs_counts))
    m = obs_counts > 0
    if m.sum() < 2:
        return np.nan
    slope = np.polyfit(idx[m], np.log(obs_counts[m].astype(float)), 1)[0]
    return float(np.exp(-slope))

def bootstrap_q(d, bins, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    qs = [fit_pop_q(np.histogram(rng.choice(d, size=len(d), replace=True), bins=bins)[0])
          for _ in range(n_boot)]
    qs = np.array([v for v in qs if np.isfinite(v)])
    return np.percentile(qs, [2.5, 97.5])

# FAIB tests
obs_lg_f, lit_exp_f, lit_p_f, bpv_f, chi2_f, chi2pv_f = large_tree_test(obs_faib, n_faib)
q_pop_f  = fit_pop_q(obs_faib)
q_lo_f, q_hi_f = bootstrap_q(live_dbh, bins)

# Field tests
obs_lg_s, lit_exp_s, lit_p_s, bpv_s, chi2_s, chi2pv_s = large_tree_test(obs_field, n_field)
q_pop_s  = fit_pop_q(obs_field)
q_lo_s, q_hi_s = bootstrap_q(dbh_field, bins)

print(f"\n=== FAIB de Liocourt (n = {n_faib} live, deduplicated) ===")
print(f"  >30 cm observed: {obs_lg_f}  expected(q=1.5): {lit_exp_f[ctrs>THRESH].sum():.1f}")
print(f"  Binomial p      : {bpv_f:.2e}")
print(f"  χ² stat         : {chi2_f:.1f}  p = {chi2pv_f:.2e}")
print(f"  Population q    : {q_pop_f:.2f}  (95% CI {q_lo_f:.2f}–{q_hi_f:.2f})  vs stable {Q_LIT}")

print(f"\n=== Field sample de Liocourt (n = {n_field}) ===")
print(f"  >30 cm observed: {obs_lg_s}  expected(q=1.5): {lit_exp_s[ctrs>THRESH].sum():.1f}")
print(f"  Binomial p      : {bpv_s:.2e}")
print(f"  χ² stat         : {chi2_s:.1f}  p = {chi2pv_s:.2e}")
print(f"  Population q    : {q_pop_s:.2f}  (95% CI {q_lo_s:.2f}–{q_hi_s:.2f})  vs stable {Q_LIT}")

# ── Per-plot conditional density ───────────────────────────────────────────────
plot_density_all = (live.groupby(['SITE_IDENTIFIER','PLOT'])['PHF_TREE']
                        .sum().reset_index())
plot_density_all.columns = ['SITE','PLOT','dens_all']

mature = live[live['DBH'] >= 10.0]
plot_density_mat = (mature.groupby(['SITE_IDENTIFIER','PLOT'])['PHF_TREE']
                          .sum().reset_index())
plot_density_mat.columns = ['SITE','PLOT','dens_mat']

d_all = plot_density_all['dens_all'].values
d_mat = plot_density_mat['dens_mat'].values

med_all = np.median(d_all)
med_mat = np.median(d_mat)
mean_all = np.mean(d_all)
mean_mat = np.mean(d_mat)
p25_mat, p75_mat = np.percentile(d_mat, [25, 75])

print(f"\n=== Conditional density (plots with TW present, deduplicated) ===")
print(f"  Plots with TW (any size) : {len(d_all)}")
print(f"  All TW  — median {med_all:.1f}  mean {mean_all:.1f}  stems/ha")
print(f"  Plots with mature TW     : {len(d_mat)}")
print(f"  Mature  — median {med_mat:.1f}  mean {mean_mat:.1f}  "
      f"IQR {p25_mat:.1f}–{p75_mat:.1f}  stems/ha")

# ── Population estimate ────────────────────────────────────────────────────────
# Using median conditional mature density as the central estimate.
# Occupancy (~proportion of habitat with TW present) is already embedded: the
# 47,534 ha figure is probability MASS (Σp × 0.01 ha), so it discounts
# low-probability pixels. We apply density only to the high-confidence portion.
# Conservative: use only P≥0.5 habitat (37,885 ha) × median mature density.
PROB_MASS_P50 = 37_885.0   # P≥0.5 threshold area (ha)

N_low  = PROB_MASS_HA  * med_mat
N_high = PROB_MASS_HA  * mean_mat
N_p50  = PROB_MASS_P50 * med_mat

print(f"\n=== Population estimate (mature trees, DBH ≥ 10 cm) ===")
print(f"  Using prob-mass 47,534 ha × median {med_mat:.1f}/ha  → N ≈ {N_low:,.0f}")
print(f"  Using prob-mass 47,534 ha × mean   {mean_mat:.1f}/ha  → N ≈ {N_high:,.0f}")
print(f"  Using P≥0.5    37,885 ha × median {med_mat:.1f}/ha  → N ≈ {N_p50:,.0f}")

# ── Figure ─────────────────────────────────────────────────────────────────────
OLIVE  = "#606c38"
RED    = "#bc3425"
BLUE   = "#264653"
AMBER  = "#e9c46a"
TEAL   = "#2a9d8f"
GREY   = "#888888"

fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
fig.suptitle(
    "Pacific Yew  ·  Size-Structure Comparison: FAIB PSP (deduplicated) vs Field Sample",
    fontsize=13, fontweight="bold", y=1.01)

# ─── Panel A: FAIB size classes ────────────────────────────────────────────────
ax = axes[0]
ax.set_title(f"A · FAIB PSP inventory  (n = {n_faib} live stems)", fontsize=10, pad=5)

ax.bar(ctrs, obs_faib, width=BIN_W * 0.72, color=TEAL, alpha=0.80,
       label=f"Observed FAIB (n={n_faib})", edgecolor="white", lw=0.4, zorder=3)
ax.plot(ctrs, lit_exp_f, "--o", color=RED, ms=4, lw=1.8,
        label=f"Expected  q = {Q_LIT}", zorder=4)

# fitted curve
fit_raw_f  = q_pop_f ** (-np.arange(len(ctrs)))
fit_curv_f = fit_raw_f * (n_faib / fit_raw_f.sum())
ax.plot(ctrs, fit_curv_f, "-.s", color=BLUE, ms=3, lw=1.4,
        label=f"Fitted  q = {q_pop_f:.2f} (CI {q_lo_f:.2f}–{q_hi_f:.2f})", zorder=4)

lg = ctrs > THRESH
ax.fill_between(ctrs[lg], obs_faib[lg].astype(float), lit_exp_f[lg],
                where=(lit_exp_f[lg] > obs_faib[lg]),
                alpha=0.28, color=RED, label=f"Large-tree deficit >30 cm")
ax.axvline(THRESH, color=AMBER, lw=1.8, ls=":", alpha=0.9)

ax.text(0.97, 0.96,
        f"Binomial p = {bpv_f:.1e}\n"
        f"χ² = {chi2_f:.0f},  p = {chi2pv_f:.1e}\n"
        f"q = {q_pop_f:.2f} (CI {q_lo_f:.2f}–{q_hi_f:.2f})",
        transform=ax.transAxes, va="top", ha="right", fontsize=7.8,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#aaa", alpha=0.88))

ax.set_xlabel("DBH (cm)", fontsize=10)
ax.set_ylabel("Number of stems", fontsize=10)
ax.set_xlim(-3, bins[-1] + 3)
ax.set_ylim(0, max(obs_faib.max(), lit_exp_f.max()) * 1.18)
ax.legend(fontsize=7.5, loc="upper left")
ax.grid(axis="y", alpha=0.22, zorder=0)

# ─── Panel B: Field sample ──────────────────────────────────────────────────────
ax2 = axes[1]
ax2.set_title(f"B · Field sample  (n = {n_field} stems)", fontsize=10, pad=5)

ax2.bar(ctrs, obs_field, width=BIN_W * 0.72, color=OLIVE, alpha=0.80,
        label=f"Observed field (n={n_field})", edgecolor="white", lw=0.4, zorder=3)
ax2.plot(ctrs, lit_exp_s, "--o", color=RED, ms=4, lw=1.8,
         label=f"Expected  q = {Q_LIT}", zorder=4)

fit_raw_s  = q_pop_s ** (-np.arange(len(ctrs)))
fit_curv_s = fit_raw_s * (n_field / fit_raw_s.sum())
ax2.plot(ctrs, fit_curv_s, "-.s", color=BLUE, ms=3, lw=1.4,
         label=f"Fitted  q = {q_pop_s:.2f} (CI {q_lo_s:.2f}–{q_hi_s:.2f})", zorder=4)

ax2.fill_between(ctrs[lg], obs_field[lg].astype(float), lit_exp_s[lg],
                 where=(lit_exp_s[lg] > obs_field[lg]),
                 alpha=0.28, color=RED, label="Large-tree deficit >30 cm")
ax2.axvline(THRESH, color=AMBER, lw=1.8, ls=":", alpha=0.9)

ax2.text(0.97, 0.96,
         f"Binomial p = {bpv_s:.1e}\n"
         f"χ² = {chi2_s:.0f},  p = {chi2pv_s:.1e}\n"
         f"q = {q_pop_s:.2f} (CI {q_lo_s:.2f}–{q_hi_s:.2f})",
         transform=ax2.transAxes, va="top", ha="right", fontsize=7.8,
         bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#aaa", alpha=0.88))

ax2.set_xlabel("DBH (cm)", fontsize=10)
ax2.set_ylabel("Number of stems", fontsize=10)
ax2.set_xlim(-3, bins[-1] + 3)
ax2.set_ylim(0, max(obs_field.max(), lit_exp_s.max()) * 1.18)
ax2.legend(fontsize=7.5, loc="upper left")
ax2.grid(axis="y", alpha=0.22, zorder=0)

# ─── Panel C: Per-plot conditional density ─────────────────────────────────────
ax3 = axes[2]
ax3.set_title("C · Conditional density (FAIB plots with TW)", fontsize=10, pad=5)

bins_d = np.linspace(0, min(d_mat.max() * 1.05, 110), 22)
ax3.hist(d_all, bins=bins_d, color=TEAL, alpha=0.55, label=f"All TW (n={len(d_all)} plots)",
         edgecolor="white", lw=0.4, zorder=3)
ax3.hist(d_mat, bins=bins_d, color=BLUE, alpha=0.65,
         label=f"Mature ≥10 cm DBH\n(n={len(d_mat)} plots)",
         edgecolor="white", lw=0.4, zorder=4)

ax3.axvline(med_all, color=TEAL, lw=2, ls="--",
            label=f"Median all = {med_all:.0f} stems/ha")
ax3.axvline(med_mat, color=BLUE, lw=2, ls="--",
            label=f"Median mature = {med_mat:.0f} stems/ha")

ax3.text(0.97, 0.96,
         f"All TW\n  median {med_all:.0f}  mean {mean_all:.0f} /ha\n\n"
         f"Mature (≥10 cm)\n  median {med_mat:.0f}  mean {mean_mat:.0f} /ha\n"
         f"  IQR {p25_mat:.0f}–{p75_mat:.0f} /ha\n\n"
         f"Population estimate\n"
         f"  47,534 ha × {med_mat:.0f}/ha\n"
         f"  ≈ {N_low:,.0f} mature trees",
         transform=ax3.transAxes, va="top", ha="right", fontsize=7.8,
         bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#aaa", alpha=0.88))

ax3.set_xlabel("TW stems / ha (PHF_TREE expansion)", fontsize=10)
ax3.set_ylabel("Number of plots", fontsize=10)
ax3.legend(fontsize=7.8, loc="upper left")
ax3.grid(axis="y", alpha=0.22, zorder=0)

plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")
print(f"\nSaved: {OUT_PNG}")
print(f"Saved: {OUT_PDF}")
