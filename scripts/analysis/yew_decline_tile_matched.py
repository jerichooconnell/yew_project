#!/usr/bin/env python3
"""
Yew Historical Population — Tile-Matched Controls (Option 5)
=============================================================
For each 10×10 km tile, compute the P≥THRESH yew rate in mature forest
within THAT tile, then apply it to the logged area in the same tile.

This avoids the previous method's assumption that all logged areas share
a single global yew density. Instead, each tile acts as its own local
control — capturing the strong north–south gradient (e.g., Carmanah at
31% vs Kitimat at 0%).

Current yew:     classifier P≥THRESH in mature forest only (logged = 0)
Historical yew:  per-tile mature forest rate × (mature forest + logged area)

Supports any classifier whose grids are cached in tile_cache/{slug}_grid.npy.
Default threshold adapts to the classifier:
  - MLP (sigmoid):  P≥0.95 (concentrated near 1.0)
  - RF/kNN/logistic: P≥0.50 (tree-vote / class-probability scale)

Usage:
    python scripts/analysis/yew_decline_tile_matched.py                # P>=0.50 (default for RF grids)
    python scripts/analysis/yew_decline_tile_matched.py --threshold 0.60
    python scripts/analysis/yew_decline_tile_matched.py --threshold 0.95  # MLP-era threshold

Outputs:
    results/analysis/yew_decline_tile_matched.json
    Printed summary to stdout
"""
import argparse
import json
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser(description='Tile-matched yew decline analysis')
parser.add_argument('--threshold', type=float, default=0.50,
                    help='Probability threshold for yew-present (default: 0.50)')
args = parser.parse_args()

# ── Load data ─────────────────────────────────────────────────────────────────
with open('results/analysis/cwh_spot_comparisons/spot_stats.json') as f:
    stats = json.load(f)

cache = Path('results/analysis/cwh_spot_comparisons/tile_cache')

CWH_AREA_HA = 3_595_194
PX_AREA_M2  = 10 * 10
PX_PER_HA   = 10_000 / PX_AREA_M2   # 100
THRESH      = args.threshold

# ── Per-tile analysis ─────────────────────────────────────────────────────────
tiles = []

for s in stats:
    slug = s['name'].lower().replace(' ', '_').replace('-', '_')
    log_path  = cache / f'{slug}_logging.npy'
    grid_path = cache / f'{slug}_grid.npy'
    if not log_path.exists() or not grid_path.exists():
        continue

    log  = np.load(log_path)     # VRI categories
    grid = np.load(grid_path)    # production model probabilities (unmodified)

    forest_mask = (log == 5)
    logged_mask = (log == 2) | (log == 3) | (log == 4)

    n_forest = int(forest_mask.sum())
    n_logged = int(logged_mask.sum())
    n_log_lt20  = int((log == 2).sum())
    n_log_20_40 = int((log == 3).sum())
    n_log_40_80 = int((log == 4).sum())

    # Yew in mature forest (current)
    p95_forest = int((grid[forest_mask] >= THRESH).sum())
    frac_yew   = p95_forest / n_forest if n_forest > 0 else 0

    # Historical: apply THIS tile's mature-forest rate to its logged area
    hist_in_logged = n_logged * frac_yew
    hist_total     = p95_forest + hist_in_logged
    curr_total     = p95_forest   # logged = 0

    decline = (1 - curr_total / hist_total) * 100 if hist_total > 0 else 0

    # Normalized densities (yew ha per ha of forested land)
    forested_px = n_forest + n_logged
    curr_density = (curr_total / forested_px) if forested_px > 0 else 0   # current yew / all forested
    hist_density = (hist_total / forested_px) if forested_px > 0 else 0   # historical yew / all forested
    loss_ratio   = (hist_in_logged / curr_total) if curr_total > 0 else float('inf') if hist_in_logged > 0 else 0
    # loss_ratio = lost / remaining: >1 means more was lost than survives

    t = {
        'name':        s['name'],
        'lat':         s['lat'],
        'lon':         s['lon'],
        'forest_px':   n_forest,
        'logged_px':   n_logged,
        'forested_px': forested_px,
        'log_lt20_px': n_log_lt20,
        'log_20_40_px': n_log_20_40,
        'log_40_80_px': n_log_40_80,
        'p95_forest':  p95_forest,
        'frac_yew':    frac_yew,
        'hist_in_logged': hist_in_logged,
        'hist_total':  hist_total,
        'curr_total':  curr_total,
        'decline_pct': decline,
        'curr_density': curr_density,
        'hist_density': hist_density,
        'loss_ratio':   loss_ratio,
    }
    tiles.append(t)

# ── Aggregate (tile-matched: sum per-tile historical estimates) ───────────────
N = len(tiles)
tot_forest   = sum(t['forest_px'] for t in tiles)
tot_logged   = sum(t['logged_px'] for t in tiles)
tot_forested = tot_forest + tot_logged
tot_water    = sum(
    int(np.sum(np.load(cache / f'{s["name"].lower().replace(" ","_").replace("-","_")}_logging.npy') == 1))
    for s in stats
    if (cache / f'{s["name"].lower().replace(" ","_").replace("-","_")}_logging.npy').exists()
)
tot_alpine = sum(
    int(np.sum(np.load(cache / f'{s["name"].lower().replace(" ","_").replace("-","_")}_logging.npy') == 6))
    for s in stats
    if (cache / f'{s["name"].lower().replace(" ","_").replace("-","_")}_logging.npy').exists()
)
tot_px = sum(s['h'] * s['w'] for s in stats)

# Tile-matched totals
curr_total_px = sum(t['curr_total'] for t in tiles)
hist_total_px = sum(t['hist_total'] for t in tiles)
lost_px       = hist_total_px - curr_total_px
decline_pct   = (1 - curr_total_px / hist_total_px) * 100 if hist_total_px > 0 else 0

# Global rate for reference (previous method)
global_frac = curr_total_px / tot_forest if tot_forest > 0 else 0

# Weighted-average tile yew fraction (for display)
weighted_frac = sum(t['frac_yew'] * t['forest_px'] for t in tiles) / tot_forest if tot_forest > 0 else 0

# ── Normalized density metrics ────────────────────────────────────────────────
# Yew density = ha of yew per 100 ha of forested land (forest + logged)
curr_density_pct = (curr_total_px / tot_forested) * 100 if tot_forested > 0 else 0
hist_density_pct = (hist_total_px / tot_forested) * 100 if tot_forested > 0 else 0
loss_ratio_agg   = lost_px / curr_total_px if curr_total_px > 0 else 0

# ── Extrapolation to CWH zone ────────────────────────────────────────────────
sample_ha = tot_px / PX_PER_HA
frac_forest_of_land = tot_forest / (tot_px - tot_water) if (tot_px - tot_water) > 0 else 0
frac_logged_of_land = tot_logged / (tot_px - tot_water) if (tot_px - tot_water) > 0 else 0

cwh_forest_ha = CWH_AREA_HA * frac_forest_of_land
cwh_logged_ha = CWH_AREA_HA * frac_logged_of_land

# Tile-matched CWH estimates: use per-tile rate weighted up
cwh_current_ha    = cwh_forest_ha * global_frac
cwh_historical_ha = (curr_total_px + lost_px) / tot_forest * cwh_forest_ha if tot_forest > 0 else 0
# Simpler: scale sample directly
expansion = CWH_AREA_HA / sample_ha if sample_ha > 0 else 0
cwh_current_ha    = curr_total_px / PX_PER_HA * expansion
cwh_historical_ha = hist_total_px / PX_PER_HA * expansion
cwh_lost_ha       = lost_px / PX_PER_HA * expansion
cwh_decline       = (1 - cwh_current_ha / cwh_historical_ha) * 100 if cwh_historical_ha > 0 else 0

# ── Latitude zone analysis ───────────────────────────────────────────────────
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

# ── Per-tile detail table ────────────────────────────────────────────────────
tile_table = []
for t in sorted(tiles, key=lambda x: -x['frac_yew']):
    tile_table.append({
        'name': t['name'],
        'lat': t['lat'],
        'forest_ha': round(t['forest_px'] / PX_PER_HA),
        'logged_ha': round(t['logged_px'] / PX_PER_HA),
        'forested_ha': round(t['forested_px'] / PX_PER_HA),
        'p95_frac': round(t['frac_yew'], 4),
        'current_yew_ha': round(t['curr_total'] / PX_PER_HA),
        'historical_yew_ha': round(t['hist_total'] / PX_PER_HA),
        'lost_ha': round((t['hist_total'] - t['curr_total']) / PX_PER_HA),
        'decline_pct': round(t['decline_pct'], 1),
        'current_density_pct': round(t['curr_density'] * 100, 3),
        'historical_density_pct': round(t['hist_density'] * 100, 3),
        'loss_to_remaining': round(t['loss_ratio'], 2),
    })

# ── Save JSON ─────────────────────────────────────────────────────────────────
output = {
    'method': 'Tile-matched controls (Option 5)',
    'description': (
        'For each 10×10 km tile, the P≥0.95 yew rate in mature forest '
        'within that tile is applied to logged area in the same tile. '
        'Each tile is its own local control.'
    ),
    'threshold': THRESH,
    'n_sites': N,
    'sample_area_ha': round(sample_ha),
    'cwh_zone_area_ha': CWH_AREA_HA,
    'sample_totals': {
        'forest_px': tot_forest,
        'logged_px': tot_logged,
        'forested_px': tot_forested,
        'current_yew_px': round(curr_total_px),
        'historical_yew_px': round(hist_total_px),
        'lost_px': round(lost_px),
        'decline_pct': round(decline_pct, 1),
        'global_p95_frac_forest': round(global_frac, 4),
        'weighted_avg_tile_frac': round(weighted_frac, 4),
    },
    'normalized_density': {
        'description': 'Yew ha per 100 ha of forested land (mature forest + logged)',
        'current_density_pct': round(curr_density_pct, 3),
        'historical_density_pct': round(hist_density_pct, 3),
        'density_loss_pct': round(hist_density_pct - curr_density_pct, 3),
        'loss_to_remaining_ratio': round(loss_ratio_agg, 2),
        'note': 'loss_to_remaining = lost yew / current yew. '
                'A ratio of 1.0 means as much was lost as remains.',
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

out_path = Path('results/analysis/yew_decline_tile_matched.json')
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"Saved: {out_path}")

# ── Print summary ─────────────────────────────────────────────────────────────
print()
print("=" * 78)
print("  WESTERN YEW — HISTORICAL DECLINE (Tile-Matched Controls)")
print("  Each tile's mature forest yew rate applied to logged area in same tile")
print("=" * 78)

print(f"""
METHOD:
  For each 10×10 km tile, compute P≥{THRESH} yew rate in mature forest
  (>80 yr), then assume logged area within that tile had the same rate
  before harvest. Current yew = mature forest only; logged = 0.

  This naturally captures the north–south gradient: tiles in south VI
  (Carmanah: {[t['frac_yew'] for t in tiles if 'Carmanah' in t['name']][0]*100:.1f}% yew) get a high rate applied to their logged
  area, while northern tiles (Kitimat: {[t['frac_yew'] for t in tiles if 'Kitimat' in t['name']][0]*100:.2f}%) get almost zero.

SAMPLE: {N} tiles × 10×10 km = {sample_ha:,.0f} ha

LAND COMPOSITION:
  Mature forest:  {tot_forest/PX_PER_HA:>10,.0f} ha
  Logged (<80 yr): {tot_logged/PX_PER_HA:>10,.0f} ha
  Water:           {tot_water/PX_PER_HA:>10,.0f} ha
  Alpine:          {tot_alpine/PX_PER_HA:>10,.0f} ha

TILE-MATCHED RESULTS (P≥{THRESH}):
  Current yew (mature forest only): {curr_total_px/PX_PER_HA:>10,.0f} ha
  Historical yew (tile-matched):     {hist_total_px/PX_PER_HA:>10,.0f} ha
  Lost to logging:                   {lost_px/PX_PER_HA:>10,.0f} ha
  ▸ Sample decline: {decline_pct:.1f}%

NORMALIZED YEW DENSITY (per 100 ha of forested land):
  Total forested land in sample:  {tot_forested/PX_PER_HA:>10,.0f} ha
  Historical density:  {hist_density_pct:.3f} ha yew per 100 ha forested
  Current density:     {curr_density_pct:.3f} ha yew per 100 ha forested
  Density lost:        {hist_density_pct - curr_density_pct:.3f} ha yew per 100 ha forested
  Loss : remaining =   {loss_ratio_agg:.2f} : 1
  (For every ha of yew that remains, {loss_ratio_agg:.2f} ha were lost)

CWH ZONE EXTRAPOLATION:
  Total forested land:  {cwh_forest_ha + cwh_logged_ha:>10,.0f} ha
  Current:         {cwh_current_ha:>10,.0f} ha  ({cwh_current_ha/(cwh_forest_ha+cwh_logged_ha)*100:.3f} ha / 100 ha)
  Historical:      {cwh_historical_ha:>10,.0f} ha  ({cwh_historical_ha/(cwh_forest_ha+cwh_logged_ha)*100:.3f} ha / 100 ha)
  Lost to logging: {cwh_lost_ha:>10,.0f} ha
  ▸ Decline:       {cwh_decline:.1f}%
""")

# Latitude zones
for lat_lo, lat_hi, zname, zdesc in zones_def:
    z = zone_results.get(zname)
    if not z:
        continue
    print(f"  {zname} — {zdesc}")
    print(f"    Sites: {z['n_sites']}  Forest: {z['forest_ha']:,} ha  Logged: {z['logged_ha']:,} ha")
    print(f"    P≥{THRESH} in mature forest: {z['p95_forest_frac']*100:.2f}%")
    print(f"    Density: {z['historical_density_pct']:.3f} → {z['current_density_pct']:.3f} ha yew / 100 ha forested")
    print(f"    Historical: {z['historical_yew_ha']:,} ha → Current: {z['current_yew_ha']:,} ha")
    print(f"    Lost: {z['lost_ha']:,} ha  ▸ Decline: {z['decline_pct']:.1f}%  (loss:remaining = {z['loss_to_remaining_ratio']:.2f}:1)")
    print()

# Per-tile detail
print("PER-TILE DETAIL (sorted by yew density):")
print(f"  {'Tile':<26s} {'Lat':>5s}  {'Forested':>8s}  {'Yew%':>6s}  "
      f"{'Hist.dens':>9s}  {'Curr.dens':>9s}  {'Lost':>6s}  {'L:R':>5s}  {'Decl.':>6s}")
print(f"  {'':─<26s} {'':─>5s}  {'(ha)':─>8s}  {'':─>6s}  "
      f"{'(/100ha)':─>9s}  {'(/100ha)':─>9s}  {'(ha)':─>6s}  {'':─>5s}  {'':─>6s}")
for t in tile_table:
    lr = t.get('loss_to_remaining', 0)
    lr_str = f"{lr:5.2f}" if lr < 999 else "  inf"
    print(f"  {t['name']:<26s} {t['lat']:5.1f}  {t['forested_ha']:>8,}  "
          f"{t['p95_frac']*100:>5.2f}%  "
          f"{t['historical_density_pct']:>8.3f}%  {t['current_density_pct']:>8.3f}%  "
          f"{t['lost_ha']:>6,}  {lr_str}  {t['decline_pct']:>5.1f}%")

print(f"""
DENSITY INTERPRETATION:
  Historical density represents the fraction of forested land that would
  have supported yew (P≥{THRESH}) before logging. Current density is what
  remains. The loss:remaining ratio (L:R) shows severity — a ratio of
  1.0 means as much yew was lost as survives; >1 means more was destroyed
  than remains.

  Province-wide yew density: ~{hist_density_pct:.2f}% of forested land historically,
  reduced to ~{curr_density_pct:.2f}% currently.

CAVEATS:
  1. Assumes logged areas within a tile had similar yew density to remaining
     mature forest in the same tile (reasonable at 10 km scale)
  2. Tiles with very little mature forest have noisy rate estimates
  3. Model trained on South VI — may underpredict in central/north BC
  4. 35 tiles sample ~1.5% of CWH zone
  5. Logging categories from BC VRI (2024)
""")
