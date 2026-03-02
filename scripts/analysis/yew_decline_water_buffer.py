#!/usr/bin/env python3
"""
Yew Historical Population — Tile-Matched Controls + Water Buffer
================================================================
Extension of yew_decline_tile_matched.py that dilates the VRI water mask
(category 1) by a configurable buffer to simulate increased stream erosion
from logging + climate-driven streamflow changes.

See docs/STREAM_EROSION_BUFFER.md for the hydrological rationale.

Usage:
    python scripts/analysis/yew_decline_water_buffer.py               # default 30 m buffer
    python scripts/analysis/yew_decline_water_buffer.py --buffer 20   # 20 m buffer
    python scripts/analysis/yew_decline_water_buffer.py --sweep        # run 0–50 m sensitivity
"""
import argparse
import json
import numpy as np
from pathlib import Path
from scipy.ndimage import binary_dilation, generate_binary_structure

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Tile-matched yew decline with water buffer')
parser.add_argument('--buffer', type=int, default=30,
                    help='Water buffer distance in metres (default: 30)')
parser.add_argument('--sweep', action='store_true',
                    help='Run sensitivity analysis across 0–50 m buffers')
args = parser.parse_args()

# ── Constants ─────────────────────────────────────────────────────────────────
PIXEL_SIZE_M = 10
PX_AREA_M2   = PIXEL_SIZE_M * PIXEL_SIZE_M
PX_PER_HA    = 10_000 / PX_AREA_M2   # 100
CWH_AREA_HA  = 3_595_194
THRESH       = 0.95

# ── Load data ─────────────────────────────────────────────────────────────────
with open('results/analysis/cwh_spot_comparisons/spot_stats.json') as f:
    stats = json.load(f)

cache = Path('results/analysis/cwh_spot_comparisons/tile_cache')

# Dilation structuring element: diamond (connectivity=1) for isotropic expansion
STRUCT = generate_binary_structure(2, 1)


def run_analysis(buffer_m: int, verbose: bool = True):
    """Run the full tile-matched analysis with a given water buffer.

    Parameters
    ----------
    buffer_m : int
        Water buffer in metres. Converted to pixels (buffer_m / PIXEL_SIZE_M).
    verbose : bool
        Print detailed output.

    Returns
    -------
    dict : Full results dictionary.
    """
    buffer_px = max(0, round(buffer_m / PIXEL_SIZE_M))

    # ── Per-tile analysis ─────────────────────────────────────────────────────
    tiles = []
    total_new_water_forest = 0
    total_new_water_logged = 0
    total_new_water_yew    = 0

    for s in stats:
        slug = s['name'].lower().replace(' ', '_').replace('-', '_')
        log_path  = cache / f'{slug}_logging.npy'
        grid_path = cache / f'{slug}_grid.npy'
        if not log_path.exists() or not grid_path.exists():
            continue

        log  = np.load(log_path)     # VRI categories
        grid = np.load(grid_path)    # production model probabilities

        # Original water mask
        water_mask = (log == 1)
        original_water_px = int(water_mask.sum())

        # Dilate water
        if buffer_px > 0:
            dilated_water = binary_dilation(water_mask, structure=STRUCT,
                                            iterations=buffer_px)
        else:
            dilated_water = water_mask

        new_water = dilated_water & ~water_mask  # newly classified as water

        # Build masks AFTER water expansion
        # Pixels that become water are removed from their original category
        forest_mask = (log == 5) & ~new_water
        logged_mask = ((log == 2) | (log == 3) | (log == 4)) & ~new_water

        # Track what was lost to water expansion
        nw_forest = int(((log == 5) & new_water).sum())
        nw_logged = int((((log == 2) | (log == 3) | (log == 4)) & new_water).sum())
        nw_yew    = int(((log == 5) & new_water & (grid >= THRESH)).sum())
        total_new_water_forest += nw_forest
        total_new_water_logged += nw_logged
        total_new_water_yew    += nw_yew

        n_forest = int(forest_mask.sum())
        n_logged = int(logged_mask.sum())
        n_log_lt20  = int(((log == 2) & ~new_water).sum())
        n_log_20_40 = int(((log == 3) & ~new_water).sum())
        n_log_40_80 = int(((log == 4) & ~new_water).sum())

        # Yew in (post-buffer) mature forest
        p95_forest = int((grid[forest_mask] >= THRESH).sum())
        frac_yew   = p95_forest / n_forest if n_forest > 0 else 0

        # Historical: apply THIS tile's rate to its (post-buffer) logged area
        hist_in_logged = n_logged * frac_yew
        hist_total     = p95_forest + hist_in_logged
        curr_total     = p95_forest

        decline = (1 - curr_total / hist_total) * 100 if hist_total > 0 else 0

        forested_px = n_forest + n_logged
        curr_density = (curr_total / forested_px) if forested_px > 0 else 0
        hist_density = (hist_total / forested_px) if forested_px > 0 else 0
        loss_ratio   = (hist_in_logged / curr_total) if curr_total > 0 else (
            float('inf') if hist_in_logged > 0 else 0)

        tiles.append({
            'name':         s['name'],
            'lat':          s['lat'],
            'lon':          s['lon'],
            'forest_px':    n_forest,
            'logged_px':    n_logged,
            'forested_px':  forested_px,
            'log_lt20_px':  n_log_lt20,
            'log_20_40_px': n_log_20_40,
            'log_40_80_px': n_log_40_80,
            'water_px':     original_water_px + int(new_water.sum()),
            'new_water_px': int(new_water.sum()),
            'p95_forest':   p95_forest,
            'frac_yew':     frac_yew,
            'hist_in_logged': hist_in_logged,
            'hist_total':   hist_total,
            'curr_total':   curr_total,
            'decline_pct':  decline,
            'curr_density': curr_density,
            'hist_density': hist_density,
            'loss_ratio':   loss_ratio,
        })

    # ── Aggregation ───────────────────────────────────────────────────────────
    N = len(tiles)
    tot_forest   = sum(t['forest_px'] for t in tiles)
    tot_logged   = sum(t['logged_px'] for t in tiles)
    tot_forested = tot_forest + tot_logged
    tot_water    = sum(t['water_px'] for t in tiles)
    tot_new_water = sum(t['new_water_px'] for t in tiles)
    tot_px       = sum(s['h'] * s['w'] for s in stats)

    curr_total_px = sum(t['curr_total'] for t in tiles)
    hist_total_px = sum(t['hist_total'] for t in tiles)
    lost_px       = hist_total_px - curr_total_px
    decline_pct   = (1 - curr_total_px / hist_total_px) * 100 if hist_total_px > 0 else 0

    global_frac   = curr_total_px / tot_forest if tot_forest > 0 else 0

    curr_density_pct = (curr_total_px / tot_forested) * 100 if tot_forested > 0 else 0
    hist_density_pct = (hist_total_px / tot_forested) * 100 if tot_forested > 0 else 0
    loss_ratio_agg   = lost_px / curr_total_px if curr_total_px > 0 else 0

    # CWH extrapolation
    sample_ha = tot_px / PX_PER_HA
    expansion = CWH_AREA_HA / sample_ha if sample_ha > 0 else 0
    cwh_current_ha    = curr_total_px / PX_PER_HA * expansion
    cwh_historical_ha = hist_total_px / PX_PER_HA * expansion
    cwh_lost_ha       = lost_px / PX_PER_HA * expansion
    cwh_decline       = (1 - cwh_current_ha / cwh_historical_ha) * 100 if cwh_historical_ha > 0 else 0

    # ── Latitude zones ────────────────────────────────────────────────────────
    zones_def = [
        (48, 50, 'South (48–50°N)', 'Vancouver Island + Sunshine Coast'),
        (50, 52, 'Central (50–52°N)', 'North VI + mainland fjords'),
        (52, 56, 'North (52–56°N)', 'Central + north coast'),
    ]
    zone_results = {}
    for lat_lo, lat_hi, zname, zdesc in zones_def:
        zt = [t for t in tiles if lat_lo <= t['lat'] < lat_hi]
        if not zt:
            continue
        z_for  = sum(t['forest_px'] for t in zt)
        z_log  = sum(t['logged_px'] for t in zt)
        z_curr = sum(t['curr_total'] for t in zt)
        z_hist = sum(t['hist_total'] for t in zt)
        z_dec  = (1 - z_curr / z_hist) * 100 if z_hist > 0 else 0
        z_frac = z_curr / z_for if z_for > 0 else 0
        z_forested = z_for + z_log
        z_curr_dens = (z_curr / z_forested) * 100 if z_forested > 0 else 0
        z_hist_dens = (z_hist / z_forested) * 100 if z_forested > 0 else 0
        z_loss_ratio = (z_hist - z_curr) / z_curr if z_curr > 0 else 0

        zone_results[zname] = {
            'description': zdesc,
            'n_sites': len(zt),
            'forest_ha': round(z_for / PX_PER_HA),
            'logged_ha': round(z_log / PX_PER_HA),
            'forested_ha': round(z_forested / PX_PER_HA),
            'p95_forest_frac': round(z_frac, 4),
            'current_yew_ha': round(z_curr / PX_PER_HA),
            'historical_yew_ha': round(z_hist / PX_PER_HA),
            'lost_ha': round((z_hist - z_curr) / PX_PER_HA),
            'decline_pct': round(z_dec, 1),
            'current_density_pct': round(z_curr_dens, 3),
            'historical_density_pct': round(z_hist_dens, 3),
            'loss_to_remaining_ratio': round(z_loss_ratio, 2),
        }

    # ── Per-tile table ────────────────────────────────────────────────────────
    tile_table = []
    for t in sorted(tiles, key=lambda x: -x['frac_yew']):
        tile_table.append({
            'name': t['name'],
            'lat': t['lat'],
            'forest_ha': round(t['forest_px'] / PX_PER_HA),
            'logged_ha': round(t['logged_px'] / PX_PER_HA),
            'forested_ha': round(t['forested_px'] / PX_PER_HA),
            'water_ha': round(t['water_px'] / PX_PER_HA),
            'new_water_ha': round(t['new_water_px'] / PX_PER_HA),
            'p95_frac': round(t['frac_yew'], 4),
            'current_yew_ha': round(t['curr_total'] / PX_PER_HA),
            'historical_yew_ha': round(t['hist_total'] / PX_PER_HA),
            'lost_ha': round((t['hist_total'] - t['curr_total']) / PX_PER_HA),
            'decline_pct': round(t['decline_pct'], 1),
            'current_density_pct': round(t['curr_density'] * 100, 3),
            'historical_density_pct': round(t['hist_density'] * 100, 3),
            'loss_to_remaining': round(t['loss_ratio'], 2),
        })

    # ── Build output ──────────────────────────────────────────────────────────
    result = {
        'method': 'Tile-matched controls + water buffer',
        'description': (
            f'Tile-matched controls with {buffer_m} m ({buffer_px} px) water buffer. '
            'VRI water mask (category 1) is dilated to simulate increased stream '
            'erosion from logging and climate change.'
        ),
        'water_buffer_m': buffer_m,
        'water_buffer_px': buffer_px,
        'threshold': THRESH,
        'n_sites': N,
        'sample_area_ha': round(sample_ha),
        'cwh_zone_area_ha': CWH_AREA_HA,
        'water_buffer_impact': {
            'total_new_water_ha': round(tot_new_water / PX_PER_HA),
            'forest_lost_to_water_ha': round(total_new_water_forest / PX_PER_HA),
            'logged_lost_to_water_ha': round(total_new_water_logged / PX_PER_HA),
            'yew_pixels_lost_to_water_ha': round(total_new_water_yew / PX_PER_HA),
            'pct_forest_lost_to_water': round(
                total_new_water_forest / (tot_forest + total_new_water_forest) * 100, 3
            ) if (tot_forest + total_new_water_forest) > 0 else 0,
        },
        'sample_totals': {
            'forest_px': tot_forest,
            'logged_px': tot_logged,
            'forested_px': tot_forested,
            'water_px': tot_water,
            'new_water_px': tot_new_water,
            'current_yew_px': round(curr_total_px),
            'historical_yew_px': round(hist_total_px),
            'lost_px': round(lost_px),
            'decline_pct': round(decline_pct, 1),
        },
        'normalized_density': {
            'description': 'Yew ha per 100 ha of forested land (mature forest + logged)',
            'current_density_pct': round(curr_density_pct, 3),
            'historical_density_pct': round(hist_density_pct, 3),
            'density_loss_pct': round(hist_density_pct - curr_density_pct, 3),
            'loss_to_remaining_ratio': round(loss_ratio_agg, 2),
        },
        'cwh_extrapolation': {
            'current_yew_ha': round(cwh_current_ha),
            'historical_yew_ha': round(cwh_historical_ha),
            'lost_ha': round(cwh_lost_ha),
            'decline_pct': round(cwh_decline, 1),
        },
        'latitude_zones': zone_results,
        'per_tile': tile_table,
    }

    # ── Save JSON ─────────────────────────────────────────────────────────────
    if buffer_m == 0:
        out_name = 'yew_decline_tile_matched_no_buffer.json'
    else:
        out_name = f'yew_decline_tile_matched_buf{buffer_m}m.json'
    out_path = Path('results/analysis') / out_name
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    if verbose:
        print(f"\nSaved: {out_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    if verbose:
        print()
        print("=" * 78)
        print("  WESTERN YEW — TILE-MATCHED DECLINE + WATER BUFFER")
        print(f"  Buffer: {buffer_m} m ({buffer_px} pixels at {PIXEL_SIZE_M} m resolution)")
        print("=" * 78)
        print(f"""
WATER BUFFER IMPACT:
  New water area (from dilation): {tot_new_water/PX_PER_HA:>10,.0f} ha
  Forest pixels → water:          {total_new_water_forest/PX_PER_HA:>10,.0f} ha
  Logged pixels → water:          {total_new_water_logged/PX_PER_HA:>10,.0f} ha
  Yew pixels (P≥{THRESH}) → water:  {total_new_water_yew/PX_PER_HA:>10,.0f} ha

LAND COMPOSITION (after buffer):
  Mature forest:   {tot_forest/PX_PER_HA:>10,.0f} ha
  Logged (<80 yr): {tot_logged/PX_PER_HA:>10,.0f} ha
  Water:           {tot_water/PX_PER_HA:>10,.0f} ha (original + buffer)

TILE-MATCHED RESULTS (P≥{THRESH}, {buffer_m} m buffer):
  Current yew:     {curr_total_px/PX_PER_HA:>10,.0f} ha
  Historical yew:  {hist_total_px/PX_PER_HA:>10,.0f} ha
  Lost to logging: {lost_px/PX_PER_HA:>10,.0f} ha
  ▸ Sample decline: {decline_pct:.1f}%

NORMALIZED YEW DENSITY (per 100 ha forested):
  Historical:  {hist_density_pct:.3f}%
  Current:     {curr_density_pct:.3f}%
  L:R =        {loss_ratio_agg:.2f} : 1

CWH ZONE EXTRAPOLATION:
  Current:         {cwh_current_ha:>10,.0f} ha
  Historical:      {cwh_historical_ha:>10,.0f} ha
  Lost:            {cwh_lost_ha:>10,.0f} ha
  ▸ Decline:       {cwh_decline:.1f}%
""")

        # Latitude zones
        for lat_lo, lat_hi, zname, zdesc in zones_def:
            z = zone_results.get(zname)
            if not z:
                continue
            print(f"  {zname} — {zdesc}")
            print(f"    Sites: {z['n_sites']}  Forest: {z['forest_ha']:,} ha  "
                  f"Logged: {z['logged_ha']:,} ha")
            print(f"    Density: {z['historical_density_pct']:.3f} → "
                  f"{z['current_density_pct']:.3f} ha yew / 100 ha forested")
            print(f"    Lost: {z['lost_ha']:,} ha  ▸ Decline: {z['decline_pct']:.1f}%  "
                  f"(L:R = {z['loss_to_remaining_ratio']:.2f}:1)")
            print()

    return result


def run_sweep():
    """Run sensitivity analysis across buffer distances 0–50 m."""
    print("=" * 78)
    print("  WATER BUFFER SENSITIVITY ANALYSIS")
    print("=" * 78)
    print()
    print(f"  {'Buffer':>8s}  {'Curr yew':>10s}  {'Hist yew':>10s}  "
          f"{'Lost':>10s}  {'Decline':>8s}  {'L:R':>6s}  "
          f"{'New water':>10s}  {'Yew→water':>10s}")
    print(f"  {'(m)':>8s}  {'(ha)':>10s}  {'(ha)':>10s}  "
          f"{'(ha)':>10s}  {'(%)':>8s}  {'':>6s}  "
          f"{'(ha)':>10s}  {'(ha)':>10s}")
    print("  " + "─" * 76)

    sweep_results = []
    for buf_m in [0, 10, 20, 30, 40, 50]:
        r = run_analysis(buf_m, verbose=False)
        s = r['sample_totals']
        d = r['normalized_density']
        w = r['water_buffer_impact']

        curr_ha = s['current_yew_px'] / PX_PER_HA
        hist_ha = s['historical_yew_px'] / PX_PER_HA
        lost_ha = s['lost_px'] / PX_PER_HA

        print(f"  {buf_m:>6d} m  {curr_ha:>10,.0f}  {hist_ha:>10,.0f}  "
              f"{lost_ha:>10,.0f}  {s['decline_pct']:>7.1f}%  "
              f"{d['loss_to_remaining_ratio']:>5.2f}  "
              f"{w['total_new_water_ha']:>10,}  "
              f"{w['yew_pixels_lost_to_water_ha']:>10,}")

        sweep_results.append({
            'buffer_m': buf_m,
            'current_yew_ha': round(curr_ha),
            'historical_yew_ha': round(hist_ha),
            'lost_ha': round(lost_ha),
            'decline_pct': s['decline_pct'],
            'loss_to_remaining': d['loss_to_remaining_ratio'],
            'new_water_ha': w['total_new_water_ha'],
            'yew_lost_to_water_ha': w['yew_pixels_lost_to_water_ha'],
        })

    # Save sweep results
    sweep_path = Path('results/analysis/yew_decline_water_buffer_sweep.json')
    with open(sweep_path, 'w') as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\n  Saved sweep results: {sweep_path}")

    return sweep_results


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if args.sweep:
        run_sweep()
    else:
        run_analysis(args.buffer)
