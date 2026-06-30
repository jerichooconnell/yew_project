#!/usr/bin/env python3
"""
Stand-level population viability analysis (PVA) for Pacific yew, parameterised
from this study's observed size structure and the Busing & Spies (1995)
stage-projection matrix.

This complements the landscape habitat-area projection (Fig 11b) with a
within-stand demographic model: given the vital rates of old-growth yew, is a
stand viable, and how do the documented ongoing stressors — ungulate browse
suppressing recruitment (60–80% seedling/sapling mortality) and continued
removal of large reproductive trees — drive the growth rate λ below 1 and raise
quasi-extinction risk?

Outputs:
  - λ, stable stage distribution, reproductive values, stage elasticities
  - deterministic projections of the observed (depleted) structure under
    baseline / browse / harvest / combined scenarios
  - stochastic quasi-extinction probabilities (à la Busing's PVA)
  - figure: results/figures/paper/fig_pva.png

Run:
    conda run -n yew_pytorch python scripts/analysis/yew_pva.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from figure_style import apply_style, ROLE, PALETTE
apply_style()

ROOT = Path("/home/jericho/yew_project")
OUT  = ROOT / "results/figures/paper"

# ── Busing & Spies (1995) stage-projection matrix (stages 1–7) ────────────────
# 1 <0.5 m · 2 0.5–1 m · 3 ≤1 cm · 4 >1–4.9 cm · 5 5–14.9 cm · 6 15–24.9 cm · 7 >24.9 cm
A = np.array([
    [0.88, 0.00, 0.00, 0.12,  0.16,  0.20,  0.24 ],
    [0.09, 0.88, 0.00, 0.00,  0.00,  0.00,  0.00 ],
    [0.00, 0.08, 0.89, 0.00,  0.00,  0.00,  0.00 ],
    [0.00, 0.00, 0.07, 0.97,  0.00,  0.00,  0.00 ],
    [0.00, 0.00, 0.00, 0.025, 0.99,  0.00,  0.00 ],
    [0.00, 0.00, 0.00, 0.00,  0.005, 0.99,  0.00 ],
    [0.00, 0.00, 0.00, 0.00,  0.00,  0.009, 0.993],
])
FECUNDITY = (0, slice(3, 7))      # recruitment terms: row 0, donor stages 4–7
SEEDLING  = [0, 1]                # stages 1–2 (browse-vulnerable seedlings/saplings)
LARGE     = [5, 6]                # stages 6–7 (>15 cm, harvest-removable big trees)


def lam(M):
    w = np.linalg.eigvals(M)
    return float(w.real[np.argmax(w.real)])


def demography(M):
    val, vec = np.linalg.eig(M)
    i = int(np.argmax(val.real))
    l = val[i].real
    w = np.abs(vec[:, i].real); w /= w.sum()                 # stable stage dist
    vl, ve = np.linalg.eig(M.T)
    j = int(np.argmax(vl.real))
    v = np.abs(ve[:, j].real); v /= v[0]                     # reproductive values
    s = np.outer(v, w) / (v @ w)                              # sensitivities
    e = (M / l) * s                                           # elasticities
    return l, w, v, e


def scenario_matrix(browse=0.0, harvest=0.0):
    """browse = fractional reduction of recruitment + seedling survival;
    harvest = annual fractional removal of large trees (stages 6–7)."""
    M = A.copy()
    M[FECUNDITY] *= (1 - browse)                              # fewer recruits
    for s in SEEDLING:                                       # higher seedling mortality
        M[:, s] *= (1 - browse)
    for s in LARGE:                                          # ongoing large-tree loss
        M[s, s] *= (1 - harvest)
    return M


def project(M, n0, steps):
    n = n0.astype(float).copy()
    out = [n.sum()]
    for _ in range(steps):
        n = M @ n
        out.append(n.sum())
    return np.array(out)


def observed_state():
    """Initial stage vector from the field + FAIB DBH structure; seedling stages
    1–3 filled from the matrix stable distribution (not measured in the samples)."""
    td = pd.read_csv(ROOT / "data/raw/faib_tree_detail.csv",
                     usecols=["SITE_IDENTIFIER", "VISIT_NUMBER", "PLOT", "TREE_NO",
                              "SPECIES", "DBH", "LV_D"], low_memory=False)
    tw = (td[td.SPECIES == "TW"].sort_values("VISIT_NUMBER")
            .groupby(["SITE_IDENTIFIER", "PLOT", "TREE_NO"], as_index=False).last())
    dbh = tw[tw.LV_D == "L"]["DBH"].values
    counts, _ = np.histogram(dbh, bins=[1, 5, 15, 25, 999])   # stages 4–7
    _, w, _, _ = demography(A)
    n = np.zeros(7)
    n[3:] = counts                                            # observed DBH stages
    # scale seedling stages 1–3 to the stable ratio relative to stage 4
    if w[3] > 0:
        n[:3] = w[:3] / w[3] * n[3]
    return n, int(counts.sum())


def main():
    l0, w, v, e = demography(A)
    print(f"Baseline λ = {l0:.4f}")
    print(f"Stable stage distribution (%): {np.round(w*100,1)}")
    print(f"Reproductive values: {np.round(v,2)}")
    print(f"Stage elasticities (Σ={e.sum():.2f}): {np.round(e.sum(axis=1),3)}  (per recipient stage)")

    # Scenario growth rates
    scen = {
        "Baseline (protected OG)":      scenario_matrix(),
        "Browse (70% recruit. loss)":   scenario_matrix(browse=0.70),
        "Large-tree harvest (2%/yr)":   scenario_matrix(harvest=0.02),
        "Browse + harvest":             scenario_matrix(browse=0.70, harvest=0.02),
    }
    print("\nScenario growth rates:")
    for k, M in scen.items():
        print(f"  λ = {lam(M):.4f}   {k}")

    n0, n_obs = observed_state()
    print(f"\nObserved initial structure (n={n_obs} measured stems): "
          f"{np.round(n0,1)}")
    # The large trees (stages 6–7) carry the highest reproductive value, so their
    # depletion erodes seed output even though their survival elasticity is low.
    print(f"Reproductive value of large trees (st6–7): {v[5]:.1f}, {v[6]:.1f} "
          f"(highest of all stages — dominant seed producers)")

    # ── deterministic projections ────────────────────────────────────────────
    YEARS = 200
    t = np.arange(YEARS + 1)
    traj = {k: project(M, n0, YEARS) for k, M in scen.items()}

    # ── stochastic quasi-extinction (lognormal noise on vital rates) ──────────
    rng = np.random.default_rng(0)
    N_REP, QE_FRAC, CV = 500, 0.10, 0.15
    qe = {}
    for k, M in scen.items():
        below = np.zeros(YEARS + 1)
        thresh = n0.sum() * QE_FRAC
        ext_time = []
        for _ in range(N_REP):
            n = n0.copy(); hit = None
            for yr in range(1, YEARS + 1):
                Mt = M * rng.lognormal(0, CV, M.shape)
                Mt = np.clip(Mt, 0, None)
                n = Mt @ n
                if n.sum() < thresh and hit is None:
                    hit = yr
            if hit is not None:
                ext_time.append(hit)
        qe[k] = len(ext_time) / N_REP
    print(f"\nQuasi-extinction probability (<{QE_FRAC:.0%} of current in {YEARS} yr, "
          f"{N_REP} reps):")
    for k in scen:
        print(f"  {qe[k]*100:4.0f}%   {k}")

    # ── figure ────────────────────────────────────────────────────────────────
    cols = {"Baseline (protected OG)": ROLE["best"],
            "Browse (70% recruit. loss)": ROLE["status_quo"],
            "Large-tree harvest (2%/yr)": PALETTE["purple"],
            "Browse + harvest": ROLE["worst"]}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    fig.suptitle("Stand-level population viability analysis (Busing & Spies 1995 matrix, "
                 "observed yew structure)", y=1.00)

    ax = axes[0]
    ax.set_title("(a) Projected stand population (relative to today)")
    for k, y in traj.items():
        ax.plot(t, y / y[0], color=cols[k], lw=2.3,
                ls="--" if "harvest" in k and "Browse" not in k else "-",
                label=f"{k}  (λ={lam(scen[k]):.3f})")
    ax.axhline(1, color=PALETTE["grey"], lw=0.9, ls=":")
    ax.set_xlabel("Years"); ax.set_ylabel("Population / current")
    ax.set_yscale("log"); ax.set_xlim(0, YEARS)
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    ax.set_title(f"(b) Quasi-extinction probability\n(<{QE_FRAC:.0%} of current, "
                 f"{N_REP} stochastic reps, CV={CV})")
    names = list(scen); probs = [qe[k] * 100 for k in names]
    ax.barh(range(len(names)), probs, color=[cols[k] for k in names],
            edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace(" (", "\n(") for n in names], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Quasi-extinction probability in 200 yr (%)")
    ax.set_xlim(0, 100)
    for i, p in enumerate(probs):
        ax.text(p + 1.5, i, f"{p:.0f}%", va="center", fontsize=8.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        plt.savefig(OUT / f"fig_pva.{ext}")
    print(f"\nSaved: {OUT/'fig_pva.png'}")


if __name__ == "__main__":
    main()
