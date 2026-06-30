#!/usr/bin/env python3
"""
Tile-level bootstrap confidence interval for the headline yew-habitat decline.

The 69.2% decline is an aggregate over a purposive sample of study tiles (~0.26%
of the CWH+ICH+CDF extent), reported without uncertainty. Here we resample the
per-tile (historical, current) habitat pairs with replacement to propagate the
tile-sampling variance into a 95% CI on the decline percentage and on the
aggregate hectare figures.

Source: results/analysis/yew_decline_tile_matched.json (per-tile historical_yew_ha
and current_yew_ha; their aggregate ratio reproduces the 69.2% headline).

Run:
    conda run -n yew_pytorch python scripts/analysis/decline_bootstrap_ci.py
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path("/home/jericho/yew_project")
SRC  = ROOT / "results/analysis/yew_decline_tile_matched.json"
N_BOOT = 10000
SEED = 0


def main():
    d = json.load(open(SRC))
    pt = d["per_tile"]
    hist = np.array([t["historical_yew_ha"] for t in pt], float)
    curr = np.array([t["current_yew_ha"]    for t in pt], float)
    n = len(pt)

    point = (1 - curr.sum() / hist.sum()) * 100
    print(f"Per-tile records: {n}")
    print(f"Point estimate (aggregate decline): {point:.1f}%")
    print(f"  Σ historical = {hist.sum():,.0f} ha, Σ current = {curr.sum():,.0f} ha\n")

    rng = np.random.default_rng(SEED)
    boot_decl, boot_hist, boot_curr = [], [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)                       # resample tiles w/ replacement
        h, c = hist[idx].sum(), curr[idx].sum()
        boot_decl.append((1 - c / h) * 100)
        boot_hist.append(h); boot_curr.append(c)
    boot_decl = np.array(boot_decl)

    lo, hi = np.percentile(boot_decl, [2.5, 97.5])
    print(f"=== Tile bootstrap ({N_BOOT:,} resamples of {n} tiles) ===")
    print(f"Decline: {point:.1f}%  (95% CI {lo:.1f}–{hi:.1f}%)")
    print(f"  bootstrap mean {boot_decl.mean():.1f}%, SD {boot_decl.std():.1f}%")

    # Scale the CI onto the headline 154,483 ha baseline for reporting
    base = 154_483.0
    rem_lo = base * (1 - hi / 100)
    rem_hi = base * (1 - lo / 100)
    print(f"\nApplied to the 154,483 ha modelled baseline:")
    print(f"  remaining habitat 47,534 ha  (95% CI {rem_lo:,.0f}–{rem_hi:,.0f} ha)")
    print(f"\nReporting string:")
    print(f"  69.2% decline (95% CI {lo:.0f}–{hi:.0f}%, tile bootstrap, n={n})")


if __name__ == "__main__":
    main()
