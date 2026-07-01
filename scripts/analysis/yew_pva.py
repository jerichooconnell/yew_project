#!/usr/bin/env python3
"""
Stand-level population viability analysis (PVA) for Pacific yew.

A single 7-stage Busing & Spies (1995) stage-projection matrix with FOUR
independent, biologically-placed threat knobs representing the *ongoing*
(not historical bark-rush) pressures on BC yew:

  browse (b, 0-1)   Ungulate browse. Deer/elk strip seedlings and saplings, so
                    browse multiplies recruitment (fecundity) AND stage 1-2
                    (seedling/sapling) survival by (1-b). 60-80% seedling/sapling
                    mortality is documented above ~10 deer/km^2 (paper 2.6.4).

  mite (m, 0-1)     Yew big-bud mite (Cecidophyopsis psilaspis). Suppresses shoot
                    extension and aril set, so growth-class advancement is scaled
                    by (1-0.20 m) and fecundity by (1-0.25 m). These dose-response
                    multipliers are illustrative sensitivity bounds, not fitted.

  fire (phi>=1)     Wildfire stand hazard. Annual burn probability = P_BURN0*phi;
                    a burn year kills 75% of the stand and leaves 25% in unburned
                    refugia (paper 2.5.2). P_BURN0 = 0.000794/yr from BC fire
                    history (1900-2024; fit_demographic_rates.py).

  log (p_log)       Industrial logging stand-clearing hazard. Annual probability
                    p_log of near-total removal: yew is shade-obligate, so a
                    clearcut locally extirpates it (~2% leave-tree retention, no
                    in-place recovery until canopy returns). The mapped 69.2%
                    decline over ~100 yr implies a historical average
                    p_log ~= 0.0117/yr; p_log ~= 0 inside protected areas.

Large-tree (bark) harvest is deliberately NOT a forward scenario: it is a 1990s
taxol-rush legacy (large trees are no longer harvested) and is the weakest lever
in isolation (lambda 1.022 -> 1.021, quasi-extinction 0%). The large-tree deficit
is treated elsewhere as a record of past harvest and a seed-source brake on
recovery, not an ongoing viability threat.

Catastrophic factors (fire, logging) are read primarily from the quasi-extinction
panel and the *median* stochastic trajectory: their ensemble mean is dominated by
the minority of stands that escape disturbance, so the median stand is the honest
summary.

Outputs:
  - lambda, stable stage distribution, reproductive values, stage elasticities
  - per-factor effective growth rate and quasi-extinction probability
  - figure: results/figures/paper/fig_pva.png

Run:
    /home/jericho/anaconda3/envs/yew_pytorch/bin/python scripts/analysis/yew_pva.py
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
FECUNDITY  = (0, slice(3, 7))            # recruitment terms: row 0, donor stages 4–7
SEEDLING   = [0, 1]                       # stages 1–2 (browse-vulnerable seedlings/saplings)
GROWTH_IDX = [(i + 1, i) for i in range(6)]  # subdiagonal growth-advancement entries

# ── Threat parameters (grounded; see module docstring) ────────────────────────
P_BURN0   = 0.000794     # baseline annual stand-burn probability (BC fire history)
FIRE_SURV = 0.25         # fraction surviving a burn year (25% refugia; 75% killed)
P_LOG_HIST = 0.0117      # historical avg annual logging hazard (69.2% over ~100 yr)
LOG_SURV  = 0.02         # leave-tree retention after clearcut (shade-obligate loss)
BROWSE_HI = 0.70         # 60–80% seedling/sapling mortality at high deer density
FIRE_PHI  = 10           # climate-elevated fire multiplier (2020s burn ~1.7× 2010s)


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


def scenario_matrix(browse=0.0, mite=0.0):
    """Chronic (within-stand) modifiers.

    browse : ungulate browse — recruitment + seedling/sapling survival ×(1-browse)
    mite   : big-bud mite — growth advancement ×(1-0.20 mite), fecundity ×(1-0.25 mite)
    (Fire and logging are catastrophic hazards, applied outside the matrix.)
    """
    M = A.copy()
    # Bud mite: slow growth-class advancement.
    for (r, c) in GROWTH_IDX:
        M[r, c] *= (1 - 0.20 * mite)
    # Fecundity reduced by browse (fewer recruits survive) and mite (less aril set).
    M[0, 3:7] *= (1 - browse) * (1 - 0.25 * mite)
    # Browse also raises seedling/sapling mortality (stage 1–2 columns).
    for s in SEEDLING:
        M[:, s] *= (1 - browse)
    return M


def cat_multiplier(p_fire=0.0, p_log=0.0):
    """Expected annual survival multiplier from catastrophic hazards (ensemble mean)."""
    return (1 - p_fire * (1 - FIRE_SURV)) * (1 - p_log * (1 - LOG_SURV))


def lam_eff(browse=0.0, mite=0.0, p_fire=0.0, p_log=0.0):
    """Effective growth rate combining chronic matrix modifiers and catastrophe hazards."""
    return lam(scenario_matrix(browse, mite)) * cat_multiplier(p_fire, p_log)


def stochastic(M, n0, years, p_fire=0.0, p_log=0.0,
               n_rep=500, cv=0.15, qe_frac=0.10, seed=0):
    """Stochastic projection: lognormal vital-rate noise + Bernoulli fire/logging.

    Returns quasi-extinction probability (population ever < qe_frac of current)
    and the 10th / 50th / 90th percentile relative-population trajectories.
    """
    rng = np.random.default_rng(seed)
    thresh = n0.sum() * qe_frac
    traj = np.zeros((n_rep, years + 1))
    hits = 0
    for r in range(n_rep):
        n = n0.astype(float).copy()
        traj[r, 0] = n.sum()
        hit = False
        for yr in range(1, years + 1):
            Mt = np.clip(M * rng.lognormal(0, cv, M.shape), 0, None)
            n = Mt @ n
            if rng.random() < p_fire:
                n = n * FIRE_SURV
            if rng.random() < p_log:
                n = n * LOG_SURV
            traj[r, yr] = n.sum()
            if n.sum() < thresh and not hit:
                hit = True
        hits += int(hit)
    rel = traj / traj[:, [0]]
    return (hits / n_rep,
            np.percentile(rel, 10, axis=0),
            np.median(rel, axis=0),
            np.percentile(rel, 90, axis=0))


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


# ── Four-factor scenario definitions ──────────────────────────────────────────
SCENARIOS = {
    "Protected old-growth":             dict(),
    "Bud mite (full dose)":             dict(mite=1.0),
    "Ungulate browse (70%)":            dict(browse=BROWSE_HI),
    "Wildfire ×10 (climate)":           dict(p_fire=FIRE_PHI * P_BURN0),
    "Logging (unprotected, 1.2%/yr)":   dict(p_log=P_LOG_HIST),
    "Unprotected (all four)":           dict(browse=BROWSE_HI, mite=1.0,
                                             p_fire=FIRE_PHI * P_BURN0, p_log=P_LOG_HIST),
}
COLS = {
    "Protected old-growth":            ROLE["best"],
    "Bud mite (full dose)":            PALETTE["blue"],
    "Ungulate browse (70%)":           PALETTE["purple"],
    "Wildfire ×10 (climate)":          ROLE["fire"],
    "Logging (unprotected, 1.2%/yr)":  ROLE["worst"],
    "Unprotected (all four)":          PALETTE["black"],
}


def main():
    l0, w, v, e = demography(A)
    print(f"Baseline λ = {l0:.4f}")
    print(f"Stable stage distribution (%): {np.round(w*100,1)}")
    print(f"Reproductive values: {np.round(v,2)}")
    print(f"Stage elasticities (Σ={e.sum():.2f}): {np.round(e.sum(axis=1),3)}  (per recipient stage)")

    n0, n_obs = observed_state()
    print(f"\nObserved initial structure (n={n_obs} measured stems): {np.round(n0,1)}")

    YEARS = 200
    t = np.arange(YEARS + 1)

    # ── per-scenario deterministic λ_eff + stochastic QE / percentile bands ────
    print(f"\n{'Scenario':<34} {'λ (chronic)':>11} {'λ_eff':>8} {'QE 200yr':>9}")
    results, bands = {}, {}
    for name, kw in SCENARIOS.items():
        chronic = lam(scenario_matrix(kw.get("browse", 0.0), kw.get("mite", 0.0)))
        leff    = lam_eff(**kw)
        qe, lo, med, hi = stochastic(scenario_matrix(kw.get("browse", 0.0), kw.get("mite", 0.0)),
                                     n0, YEARS,
                                     p_fire=kw.get("p_fire", 0.0),
                                     p_log=kw.get("p_log", 0.0))
        results[name] = dict(chronic=chronic, leff=leff, qe=qe)
        bands[name]   = (lo, med, hi)
        print(f"{name:<34} {chronic:>11.4f} {leff:>8.4f} {qe*100:>8.0f}%")

    print(f"\nParameters: P_BURN0={P_BURN0}/yr, fire×{FIRE_PHI}; "
          f"p_log={P_LOG_HIST}/yr (69.2% over ~100 yr); "
          f"browse={BROWSE_HI}; mite dose −20% growth / −25% fecundity.")
    print(f"Reproductive value of large trees (st6–7): {v[5]:.1f}, {v[6]:.1f} "
          f"(highest of all stages — dominant seed producers)")

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    fig.suptitle("Stand-level viability under four ongoing threats "
                 "(Busing & Spies 1995 matrix, observed yew structure)", y=1.00)

    ax = axes[0]
    ax.set_title("(a) Median stand trajectory (relative to today)")
    for name in SCENARIOS:
        lo, med, hi = bands[name]
        ax.plot(t, med, color=COLS[name], lw=2.3,
                label=f"{name}  (λ_eff={results[name]['leff']:.3f})")
    # spread band only on the two anchor scenarios to avoid clutter
    for name in ("Protected old-growth", "Unprotected (all four)"):
        lo, med, hi = bands[name]
        ax.fill_between(t, lo, hi, color=COLS[name], alpha=0.12, lw=0)
    ax.axhline(1, color=PALETTE["grey"], lw=0.9, ls=":")
    ax.axhline(0.10, color=PALETTE["grey"], lw=0.9, ls="--")
    ax.text(YEARS * 0.98, 0.105, "quasi-extinction (10%)", ha="right", va="bottom",
            fontsize=7.5, color=PALETTE["grey"])
    ax.set_xlabel("Years"); ax.set_ylabel("Population / current (median rep)")
    ax.set_yscale("log"); ax.set_xlim(0, YEARS)
    ax.legend(fontsize=7.8, loc="lower left")

    ax = axes[1]
    ax.set_title("(b) Quasi-extinction probability\n(<10% of current in 200 yr, "
                 "500 stochastic reps, CV=0.15)")
    names = list(SCENARIOS)
    probs = [results[k]["qe"] * 100 for k in names]
    ax.barh(range(len(names)), probs, color=[COLS[k] for k in names],
            edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace(" (", "\n(") for n in names], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Quasi-extinction probability in 200 yr (%)")
    ax.set_xlim(0, 100)
    for i, p in enumerate(probs):
        ax.text(p + 1.5, i, f"{p:.0f}%", va="center", fontsize=8.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("png", "pdf"):
        plt.savefig(OUT / f"fig_pva.{ext}")
    print(f"\nSaved: {OUT/'fig_pva.png'}")


if __name__ == "__main__":
    main()
