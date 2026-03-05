"""
water_buffer_sensitivity.py
───────────────────────────
Model the effect of expanding water/riparian buffers by 5, 10, and 15 m on
estimated yew habitat across all study-area tiles.

At 10 m/pixel resolution, the buffer radii in pixels are:
  5 m  → 0.5 px
  10 m → 1.0 px
  15 m → 1.5 px

Method
------
1. Load each tile's yew-probability grid and VRI logging-category grid.
2. Build water mask (category 1).
3. Use scipy.ndimage.distance_transform_edt to get distance (in metres) from
   every non-water pixel to the nearest water pixel.
4. For each buffer scenario, zero out yew probabilities within that distance
   band and recount habitat area above P=0.5.
5. Compare against the baseline (no additional buffer).

Output
------
Prints a summary table + per-tile details for each buffer scenario.
"""

from pathlib import Path
import sys

import numpy as np
try:
    from scipy.ndimage import distance_transform_edt
except ImportError:
    sys.exit("scipy not found — run:  conda run -n yew_pytorch pip install scipy")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
TILE_CACHE = ROOT / "results" / "analysis" / "cwh_spot_comparisons" / "tile_cache"

# ── Constants ─────────────────────────────────────────────────────────────────
SCALE_M    = 10            # metres per pixel
HA_PER_PX  = (SCALE_M**2) / 10_000   # 0.01 ha / px
THRESHOLD  = 0.5           # P ≥ this = "yew present"

# Buffer distances to test (metres)
BUFFERS_M  = [5, 10, 15]

# Suppression factors (current scheme)
LOG_SUPPRESS = {
    1: 0.00,  # water
    2: 0.00,  # logged <20 yr
    3: 0.00,  # logged 20–40 yr
    4: 0.20,  # logged 40–80 yr
    5: 0.35,  # forest 80–150 yr
    6: 0.00,  # alpine
    7: 1.00,  # old-growth >150 yr
}

# ── Study areas (slug only; matched to *_grid.npy / *_logging.npy) ─────────
STUDY_AREAS = [
    "alberni_valley",
    "bella_coola_valley",
    "broughton_archipelago",
    "burke_channel",
    "bute_inlet_slopes",
    "campbell_river_uplands",
    "carmanah_walbran",
    "chilliwack_uplands",
    "clayoquot_sound",
    "comox_uplands",
    "cowichan_uplands",
    "dean_channel",
    "desolation_sound",
    "garibaldi_foothills",
    "gold_river_forest",
    "haida_gwaii_south",
    "howe_sound_east",
    "jervis_inlet_slopes",
    "kingcome_inlet",
    "kitimat_ranges",
    "klemtu_forest",
    "knight_inlet",
    "milbanke_sound",
    "muchalat_valley",
    "namu_lowlands",
    "nanaimo_lakes",
    "ocean_falls",
    "owikeno_lake",
    "port_hardy_forest",
    "port_renfrew",
    "portland_inlet",
    "powell_river_forest",
    "prince_rupert_hills",
    "princess_royal_island",
    "quatsino_sound",
    "rivers_inlet",
    "sechelt_peninsula",
    "smith_sound",
    "sooke_hills",
    "squamish_highlands",
    "stave_lake",
    "stewart_lowlands",
    "strathcona_highlands",
    "sunshine_coast_south",
    "toba_inlet_slopes",
]


def apply_suppression(grid, log_grid):
    """Apply LOG_SUPPRESS to get suppressed probability grid."""
    out = grid.copy()
    for cat, factor in LOG_SUPPRESS.items():
        out[log_grid == cat] *= factor
    return out


def yew_ha(prob_grid):
    """Count ha where P >= THRESHOLD."""
    return float((prob_grid >= THRESHOLD).sum()) * HA_PER_PX


def main():
    print("Water-buffer sensitivity analysis")
    print(f"Buffers tested: {BUFFERS_M} m  (pixel size = {SCALE_M} m)")
    print(f"Threshold: P ≥ {THRESHOLD}")
    print("=" * 75)

    # Accumulators: baseline + one per buffer distance
    totals = {"baseline": 0.0}
    for b in BUFFERS_M:
        totals[b] = 0.0

    tile_rows = []

    tiles_found = 0
    for slug in STUDY_AREAS:
        grid_path = TILE_CACHE / f"{slug}_grid.npy"
        log_path  = TILE_CACHE / f"{slug}_logging.npy"
        if not grid_path.exists() or not log_path.exists():
            continue
        tiles_found += 1

        grid     = np.load(str(grid_path))
        log_grid = np.load(str(log_path))

        # Baseline suppressed grid
        suppressed_base = apply_suppression(grid, log_grid)
        base_ha = yew_ha(suppressed_base)
        totals["baseline"] += base_ha

        # Water mask and distance transform (metres)
        water_mask = (log_grid == 1)
        if water_mask.any():
            # distance_transform_edt: distance of each non-water pixel to
            # nearest water pixel, in pixels; multiply by SCALE_M for metres
            dist_m = distance_transform_edt(~water_mask) * SCALE_M
        else:
            dist_m = np.full(log_grid.shape, 9999.0, dtype=np.float32)

        row = {"slug": slug, "baseline_ha": base_ha}

        for buf_m in BUFFERS_M:
            # Pixels within buf_m of water and not already water → expand water
            in_buffer = (~water_mask) & (dist_m <= buf_m)
            buf_ha = float(in_buffer.sum()) * HA_PER_PX

            # Zero out yew probability in buffer zone
            suppressed_buf = suppressed_base.copy()
            suppressed_buf[in_buffer] = 0.0
            new_ha = yew_ha(suppressed_buf)

            lost_ha = base_ha - new_ha
            totals[buf_m] += new_ha
            row[f"buf_{buf_m}m_ha"]   = new_ha
            row[f"lost_{buf_m}m_ha"]  = lost_ha
            row[f"zone_{buf_m}m_ha"]  = buf_ha   # total area of new buffer zone

        tile_rows.append(row)

    if tiles_found == 0:
        sys.exit("No tile cache files found.")

    # ── Per-tile table ─────────────────────────────────────────────────────
    col = 26
    header = (f"{'Tile':<{col}}  {'Baseline':>9} "
              + "".join(f"  {b:>6}m±ha" for b in BUFFERS_M))
    print(header)
    print("-" * len(header))
    for r in sorted(tile_rows, key=lambda x: x["baseline_ha"], reverse=True):
        parts = f"  {r['baseline_ha']:>9.1f}"
        for b in BUFFERS_M:
            lost = r[f'lost_{b}m_ha']
            parts += f"  {r[f'buf_{b}m_ha']:>6.1f} ({lost:+.1f})"
        print(f"  {r['slug']:<{col}}{parts}")

    # ── Summary ────────────────────────────────────────────────────────────
    print()
    print("=" * 75)
    base = totals["baseline"]
    print(f"  Baseline yew habitat (no extra buffer):     {base:>10,.1f} ha")
    for b in BUFFERS_M:
        new  = totals[b]
        lost = base - new
        pct  = lost / base * 100 if base > 0 else 0
        print(f"  +{b:>2} m water buffer  → yew = {new:>10,.1f} ha  "
              f"(−{lost:,.1f} ha, −{pct:.1f}%)")

    # ── Buffer-zone area summary ───────────────────────────────────────────
    print()
    print("  Area newly classified as water buffer:")
    for b in BUFFERS_M:
        total_zone = sum(r[f"zone_{b}m_ha"] for r in tile_rows)
        print(f"  +{b:>2} m buffer zone:  {total_zone:>10,.0f} ha of non-water land")

    print()
    print(f"  Tiles processed: {tiles_found}")


if __name__ == "__main__":
    main()
