#!/usr/bin/env python3
"""
Predictive Pacific yew population scenarios, 2024–2224.

Combines the Busing & Spies (1995, PNW-RN-515) stage-projection matrix for
old-growth yew with this study's mapped habitat (47,534 ha continuous mass) and
FAIB-calibrated density to project province-scale yew abundance under best- and
worst-case management.

Panel A — Stand-level dynamics from the Busing matrix (λ = 1.02), density-capped
          at the carrying capacity Busing observed for old-growth stands
          (~500 stems/ha total, ~50/ha >5 cm DBH). Shows that an intact stand
          sits near carrying capacity, whereas a cl- cut/harvested stand needs
          >2 centuries to rebuild the large-stem cohort — justifying the
          treatment of logged land as effectively lost over the planning horizon.

Panel B — Province-scale total mature yew (DBH ≥ 10 cm), driven by the dominant
          lever (habitat area):
            • Best case   — remaining 47,534 ha protected; size structure recovers
                            from the current depleted state toward the Busing
                            old-growth stable distribution.
            • Status quo  — recent old-growth attrition continues at a modest rate.
            • Worst case  — the 89 % of habitat currently outside protected areas
                            is logged over the next century; only the protected
                            fraction persists.
          Shaded bands span the density uncertainty (FAIB median 10/ha →
          Busing old-growth ~40/ha >5 cm).

This is an illustrative scenario analysis, not a calibrated forecast; all
assumptions are stated in the caption and here.

Run:
    conda run -n yew_pytorch python scripts/analysis/population_projection.py
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from figure_style import apply_style, ROLE, PALETTE

apply_style()

OUTDIR = Path("/home/jericho/yew_project/results/figures/paper")

# ── Busing & Spies (1995) Table 1 stage-projection matrix A[recipient, donor] ──
# Stages: 1 <0.5 m · 2 0.5–1.0 m · 3 ≤1 cm · 4 >1–4.9 cm · 5 5–14.9 cm
#         6 15–24.9 cm · 7 >24.9 cm DBH
A = np.array([
    [0.88, 0.00, 0.00, 0.12,  0.16,  0.20,  0.24 ],
    [0.09, 0.88, 0.00, 0.00,  0.00,  0.00,  0.00 ],
    [0.00, 0.08, 0.89, 0.00,  0.00,  0.00,  0.00 ],
    [0.00, 0.00, 0.07, 0.97,  0.00,  0.00,  0.00 ],
    [0.00, 0.00, 0.00, 0.025, 0.99,  0.00,  0.00 ],
    [0.00, 0.00, 0.00, 0.00,  0.005, 0.99,  0.00 ],
    [0.00, 0.00, 0.00, 0.00,  0.00,  0.009, 0.993],
])
BIG = slice(4, 7)                       # stages 5–7 = >5 cm DBH ("harvestable")
K_TOTAL = 500.0                         # Busing old-growth carrying capacity (stems/ha)

eval_, evec = np.linalg.eig(A)
k = int(np.argmax(eval_.real))
LAMBDA = eval_[k].real
STABLE = np.abs(evec[:, k].real); STABLE /= STABLE.sum()


LARGE = 6        # stage 7 = >24.9 cm DBH (the bark-harvest / dominant-reproducer cohort)

def project_capped(n0, steps, K=K_TOTAL):
    """Project the matrix, capping TOTAL density at K by proportional thinning.

    Rescaling the whole vector preserves the evolving stage structure (and holds
    a stand already at the stable distribution flat at carrying capacity), which
    is the behaviour Busing's density-regulated old-growth simulations show.
    Returns per-step totals for all stems, >5 cm (stages 5–7) and >24.9 cm (st 7).
    """
    n = n0.astype(float).copy()
    tot, big, lg = [n.sum()], [n[BIG].sum()], [n[LARGE]]
    for _ in range(steps):
        n = A @ n
        if n.sum() > K:
            n *= K / n.sum()
        tot.append(n.sum()); big.append(n[BIG].sum()); lg.append(n[LARGE])
    return np.array(tot), np.array(big), np.array(lg)


# ── Panel A trajectories ──────────────────────────────────────────────────────
# Busing's two scenarios: an intact old-growth stand at the stable distribution,
# and the same stand after harvest of all stems >5 cm DBH (stages 5–7 set to 0).
# Recovery is shown as a fraction of the intact old-growth level for two cohorts:
# all harvestable stems (>5 cm) and the large-tree cohort (>24.9 cm DBH).
YEARS = 200
t = np.arange(YEARS + 1)
tot_i, big_i, lg_i = project_capped(STABLE * K_TOTAL, YEARS)
n_harv = (STABLE * K_TOTAL).copy(); n_harv[BIG] = 0   # remove all >5 cm (Busing)
tot_l, big_l, lg_l = project_capped(n_harv, YEARS)

big_frac   = big_l / big_i[-1]        # >5 cm recovery as fraction of old-growth
large_frac = lg_l / lg_i[-1]          # >24.9 cm recovery as fraction of old-growth

# ── Panel B province-scale scenarios ──────────────────────────────────────────
HABITAT_HA   = 47_534.0           # continuous probability-mass habitat
PROT_FRAC    = 0.11               # 11 % currently protected
DENS_FAIB    = 10.0               # FAIB median mature (≥10 cm) stems/ha
DENS_BUSING  = 40.0               # Busing old-growth >5 cm stems/ha (upper anchor)
YRS = np.arange(2024, 2225)
H = len(YRS)

# Best: area held; size structure recovers, lifting mature density from the
# current depleted FAIB median toward the Busing old-growth value along the
# matrix recovery shape.
recov = np.interp(np.linspace(0, YEARS, H), t, big_frac)   # >5 cm recovery shape
recov = (recov - recov[0]) / (recov.max() - recov[0])      # normalise 0→1
dens_best = DENS_FAIB + (DENS_BUSING - DENS_FAIB) * recov
area_best = np.full(H, HABITAT_HA)

# Worst: unprotected 89 % logged linearly over 100 yr; protected fraction remains
frac_lost = np.clip((YRS - 2024) / 100.0, 0, 1) * (1 - PROT_FRAC)
area_worst = HABITAT_HA * (1 - frac_lost)
dens_worst = np.full(H, DENS_FAIB)

# Status quo: slow attrition (0.4 %/yr of remaining), density flat
area_sq = HABITAT_HA * (0.996 ** (YRS - 2024))
dens_sq = np.full(H, DENS_FAIB)

N_best  = area_best  * dens_best
N_worst = area_worst * dens_worst
N_sq    = area_sq    * dens_sq
# uncertainty bands: density held at FAIB low (10/ha) vs Busing high (50/ha)
N_best_lo  = area_best  * DENS_FAIB
N_best_hi  = area_best  * 50.0
N_worst_lo = area_worst * 5.0

print(f"λ = {LAMBDA:.4f}; stable >5cm fraction = {STABLE[BIG].sum()*100:.1f}%")
print(f"2024 N (FAIB 10/ha): {HABITAT_HA*DENS_FAIB:,.0f} mature")
print(f"2224 best:  {N_best[-1]:,.0f}   worst: {N_worst[-1]:,.0f}   status-quo: {N_sq[-1]:,.0f}")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
fig.suptitle("Predicted Pacific yew trajectories under best- and worst-case scenarios",
             y=1.00)

# Panel A — recovery of two cohorts after large-tree harvest (fraction of old-growth)
ax = axes[0]
ax.set_title("(a) Post-harvest recovery (Busing & Spies 1995 matrix, λ = 1.02)")
ax.axhline(1.0, color=ROLE["neutral"], lw=1, ls=":")
ax.annotate("intact old-growth level", xy=(110, 1.0), xytext=(70, 1.04),
            fontsize=8.2, color=ROLE["neutral"])
ax.plot(t, big_frac, color=ROLE["best"], lw=2.4,
        label="All harvestable stems (>5 cm DBH)")
ax.plot(t, large_frac, color=ROLE["worst"], lw=2.4, ls="--",
        label="Large trees (>24.9 cm DBH) — bark/seed cohort")
for frac, yr_lbl, col, y in ((big_frac, ">5 cm", ROLE["best"], 0.9),
                             (large_frac, ">24.9 cm", ROLE["worst"], 0.9)):
    yr90 = np.interp(0.9, frac, t)
    if yr90 < YEARS:
        ax.annotate(f"{yr_lbl}: ~{yr90:.0f} yr to 90%", xy=(yr90, 0.9),
                    xytext=(yr90 + 6, 0.62 if col == ROLE["worst"] else 0.78),
                    fontsize=8, color=col,
                    arrowprops=dict(arrowstyle="->", color=col, lw=0.9))
ax.set_xlabel("Years since large-tree harvest")
ax.set_ylabel("Cohort density (fraction of old-growth)")
ax.set_xlim(0, YEARS)
ax.set_ylim(0, 1.15)
ax.legend(loc="lower right")

# Panel B
ax = axes[1]
ax.set_title("(b) Province-scale mature yew (DBH ≥ 10 cm), 2024–2224")
ax.fill_between(YRS, N_best_lo / 1e6, N_best_hi / 1e6, color=ROLE["best"], alpha=0.13)
ax.plot(YRS, N_best / 1e6, color=ROLE["best"], lw=2.6,
        label="Best case — full protection + structural recovery")
ax.plot(YRS, N_sq / 1e6, color=ROLE["status_quo"], lw=2.2, ls="-.",
        label="Status quo — 0.4 %/yr old-growth attrition")
ax.fill_between(YRS, N_worst_lo / 1e6, N_worst / 1e6, color=ROLE["worst"], alpha=0.15)
ax.plot(YRS, N_worst / 1e6, color=ROLE["worst"], lw=2.6, ls="--",
        label="Worst case — unprotected habitat logged by 2124")
ax.scatter([2024], [HABITAT_HA * DENS_FAIB / 1e6], color=PALETTE["black"],
           zorder=5, s=35)
ax.annotate("2024 estimate\n~0.48 M (FAIB) – 1.9 M (Busing)",
            xy=(2024, HABITAT_HA * DENS_FAIB / 1e6),
            xytext=(2055, 1.35), fontsize=8.2,
            arrowprops=dict(arrowstyle="->", color=PALETTE["black"], lw=0.9))
ax.set_xlabel("Year")
ax.set_ylabel("Mature yew individuals (millions)")
ax.set_xlim(2024, 2224)
ax.set_ylim(0, N_best_hi.max() / 1e6 * 1.1)
ax.legend(loc="upper left")

plt.tight_layout(rect=[0, 0, 1, 0.97])
for ext in ("png", "pdf"):
    plt.savefig(OUTDIR / f"fig_population_projection.{ext}")
print(f"\nSaved: {OUTDIR/'fig_population_projection.png'}")
