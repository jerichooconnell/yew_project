#!/usr/bin/env python3
"""
Physically-grounded post-fire yew recovery, replacing the arbitrary linear
(2024 − fire_year)/124 suppression modifier.

Two ingredients:
  1. Burn severity — within a mapped fire perimeter we assume 75% of the yew
     habitat is killed (BURN_FRAC = 0.75) and 25% survives in unburned refugia
     and skips/islands. Pacific yew has very thin bark (~0.5 cm) and essentially
     no fire tolerance, so burned area implies near-total local mortality.
  2. Recovery — the killed fraction regrows following the Busing & Spies (1995)
     stage-projection matrix for old-growth yew (λ = 1.02). Recovery is tracked as
     the old-growth large-tree cohort (stages 6–7, >15 cm DBH) rebuilding from a
     reseeded seedling stand toward old-growth density, since those large trees
     define old-forest yew habitat and set Busing's "centuries to recover"
     timescale. Recovery reaches ~90% only at ~105 yr post-fire.

The resulting modifier is the surviving fraction of pre-fire habitat at
t = 2024 − fire_year years post-fire:

    modifier(t) = (1 − BURN_FRAC) + BURN_FRAC · R(t)

where R(t) ∈ [0, 1] is the Busing recovery fraction of the burned portion
(R(0) = 0, R(∞) = 1). modifier(0) = 0.25 (75% just killed); modifier → 1 as the
stand recovers. This is more defensible than the linear ramp: it encodes a real
burn severity and a yew-specific demographic recovery rate, and it is *slower*
than the linear model for recent fires yet *faster* for century-old ones.

Run (print the curve):
    conda run -n yew_pytorch python scripts/analysis/fire_recovery.py
"""
import numpy as np

BURN_FRAC = 0.75          # fraction of yew within a perimeter killed by fire
K_TOTAL   = 500.0         # Busing old-growth carrying capacity (stems/ha, total)
# Recovery is tracked on the old-growth-defining large-tree cohort (stages 6–7,
# >15 cm DBH). Busing & Spies conclude disturbed stands need "centuries to recover
# the population size and structure characteristic of old-forest stands"; that
# slow timescale is set by the large trees, not the fast-rebuilding small stems,
# so basing recovery on >5 cm would understate the loss and contradict Busing.
OLDGROWTH = slice(5, 7)   # stages 6–7 = >15 cm DBH (old-growth structural cohort)

# Busing & Spies (1995) PNW-RN-515 Table 1 stage-projection matrix A[recipient, donor]
A = np.array([
    [0.88, 0.00, 0.00, 0.12,  0.16,  0.20,  0.24 ],
    [0.09, 0.88, 0.00, 0.00,  0.00,  0.00,  0.00 ],
    [0.00, 0.08, 0.89, 0.00,  0.00,  0.00,  0.00 ],
    [0.00, 0.00, 0.07, 0.97,  0.00,  0.00,  0.00 ],
    [0.00, 0.00, 0.00, 0.025, 0.99,  0.00,  0.00 ],
    [0.00, 0.00, 0.00, 0.00,  0.005, 0.99,  0.00 ],
    [0.00, 0.00, 0.00, 0.00,  0.00,  0.009, 0.993],
])

_eval, _evec = np.linalg.eig(A)
_k = int(np.argmax(_eval.real))
STABLE = np.abs(_evec[:, _k].real); STABLE /= STABLE.sum()


def _recovery_curve(n_years=400):
    """Busing recovery fraction R(t) of the old-growth large-tree cohort.

    The burned 75% of the stand is destroyed to bare ground; recovery starts from
    a reseeded seedling cohort (stage 1 only, at the stable stage-1 density, fed by
    surviving refugia and neighbours) and the large-tree structural cohort
    (stages 6–7) is tracked as it slowly rebuilds toward the old-growth density.
    Because large trees can only arrive by decades of growth through the smaller
    stages, R rises slowly — the "centuries to recover" timescale Busing reports.
    Returns R[t] for t = 0..n_years, R∈[0,1].
    """
    bigtree_K = STABLE[OLDGROWTH].sum() * K_TOTAL          # old-growth large-tree density
    n = np.zeros(7)
    n[0] = STABLE[0] * K_TOTAL                             # reseeded seedling cohort only
    big = [n[OLDGROWTH].sum()]
    for _ in range(n_years):
        n = A @ n
        if n.sum() > K_TOTAL:
            n *= K_TOTAL / n.sum()                         # density cap (proportional)
        big.append(n[OLDGROWTH].sum())
    big = np.array(big)
    R = big / bigtree_K                                    # 0 at burn → 1 at old-growth structure
    return np.clip(R, 0.0, 1.0)


_R = _recovery_curve()


def busing_fire_modifier(years_since_fire):
    """Surviving fraction of pre-fire yew habitat, t years after a fire.

    Vectorised over an int array; t<0 (future-dated / clipped) returns 1.0.
    """
    t = np.asarray(years_since_fire)
    t_clip = np.clip(t, 0, len(_R) - 1).astype(int)
    mod = (1 - BURN_FRAC) + BURN_FRAC * _R[t_clip]
    return np.where(t < 0, 1.0, mod).astype(np.float32)


if __name__ == "__main__":
    print(f"λ = {_eval[_k].real:.4f}   burn fraction = {BURN_FRAC}")
    print(f"Old-growth large-tree (>15 cm) density = {STABLE[OLDGROWTH].sum()*K_TOTAL:.0f}/ha\n")
    print(f"{'yrs since fire':>14} | {'Busing mod':>10} | {'linear mod':>10}")
    print("-" * 42)
    for t in (0, 5, 10, 20, 40, 60, 80, 100, 124):
        lin = min(t / 124.0, 1.0)
        print(f"{t:>14} | {float(busing_fire_modifier(t)):>10.3f} | {lin:>10.3f}")
    # time to 90% habitat recovery
    mod = busing_fire_modifier(np.arange(len(_R)))
    t90 = int(np.argmax(mod >= 0.9))
    print(f"\nTime to 90% habitat recovery: ~{t90} yr "
          f"(modifier reaches {float(busing_fire_modifier(t90)):.2f})")
