#!/usr/bin/env python3
"""
Habitat loss statistics: XGBoost potential habitat vs logging/water/alpine mask.
Reports how much predicted yew habitat has been lost to each VRI land cover category.
Uses zero yew presence for all logged areas (cats 2,3,4 → 0).
"""
import numpy as np
from pathlib import Path

CACHE  = Path('results/analysis/cwh_spot_comparisons/tile_cache')
PX_HA  = 0.01   # 10m x 10m pixels → 0.01 ha each
THRESH = 0.5    # probability threshold for "yew present"

# VRI categories in logging grid:
# 0 = no data, 1 = water/non-forest, 2 = logged <20 yr,
# 3 = logged 20-40 yr, 4 = logged 40-80 yr, 5 = forest >80 yr, 6 = alpine/barren

slugs = sorted([p.stem.replace('_grid', '') for p in CACHE.glob('*_grid.npy')])

totals = dict(potential=0, lt20=0, yr20_40=0, yr40_80=0, water=0, alpine=0,
              forest=0, nodata=0)

HDR = ('Tile', 'Potential', 'Log<20', 'Log20-40', 'Log40-80',
       'Water', 'Alpine', 'Forest>80', 'Loss%')
fmt = '{:<30} {:>10} {:>9} {:>10} {:>10} {:>8} {:>8} {:>10} {:>7}'
print(fmt.format(*HDR))
print('-' * 110)

rows = []
for slug in slugs:
    grid_path = CACHE / (slug + '_grid.npy')
    log_path  = CACHE / (slug + '_logging.npy')
    if not log_path.exists():
        continue

    raw = np.load(grid_path)
    log = np.load(log_path)
    pos = raw >= THRESH   # pixels where XGBoost says yew present

    # Zeroed: water/non-forest (cat 1) are ocean false-positives — excluded from
    # potential habitat.  Reported separately for transparency.
    water      = int((pos & (log == 1)).sum()) * PX_HA  # excluded
    pos_land   = pos & (log != 1)                        # land-only positives

    potential  = pos_land.sum() * PX_HA
    lt20       = int((pos_land & (log == 2)).sum()) * PX_HA
    yr20_40    = int((pos_land & (log == 3)).sum()) * PX_HA
    yr40_80    = int((pos_land & (log == 4)).sum()) * PX_HA
    alpine     = int((pos_land & (log == 6)).sum()) * PX_HA
    forest     = int((pos_land & (log == 5)).sum()) * PX_HA
    nodata     = int((pos_land & (log == 0)).sum()) * PX_HA
    loss_pct   = (potential - forest) / potential * 100 if potential > 0 else 0.0

    name = slug.replace('_', ' ').title()
    print(fmt.format(name,
                     f'{potential:.0f}', f'{lt20:.0f}', f'{yr20_40:.0f}',
                     f'{yr40_80:.0f}', f'{water:.0f}', f'{alpine:.0f}',
                     f'{forest:.0f}', f'{loss_pct:.1f}%'))

    for k, v in (('potential', potential), ('lt20', lt20), ('yr20_40', yr20_40),
                 ('yr40_80', yr40_80), ('water', water), ('alpine', alpine),
                 ('forest', forest), ('nodata', nodata)):
        totals[k] += v

    rows.append(dict(slug=slug, potential=potential, lt20=lt20,
                     yr20_40=yr20_40, yr40_80=yr40_80, water=water,
                     alpine=alpine, forest=forest))

print('-' * 110)
tp = totals['potential']
tl = tp - totals['forest']
print(fmt.format('TOTAL',
                 f'{tp:.0f}', f'{totals["lt20"]:.0f}', f'{totals["yr20_40"]:.0f}',
                 f'{totals["yr40_80"]:.0f}', f'{totals["water"]:.0f}',
                 f'{totals["alpine"]:.0f}', f'{totals["forest"]:.0f}',
                 f'{tl/tp*100:.1f}%'))

total_logging = totals['lt20'] + totals['yr20_40'] + totals['yr40_80']
print()
print('=' * 60)
print('SUMMARY  (all 35 tiles, XGBoost GPU model)')
print('Water/non-forest pixels zeroed — not counted as habitat')
print('=' * 60)
print(f'  Potential land habitat (P>=0.5, water zeroed): {tp:>8,.0f} ha')
print(f'  Excluded ocean/water false-positives:          {totals["water"]:>8,.0f} ha  (zeroed, not in totals)')
print()
print(f'  Lost to logging < 20 yr:              {totals["lt20"]:>8,.0f} ha  '
      f'({totals["lt20"]/tp*100:4.1f}%)')
print(f'  Lost to logging 20-40 yr:             {totals["yr20_40"]:>8,.0f} ha  '
      f'({totals["yr20_40"]/tp*100:4.1f}%)')
print(f'  Lost to logging 40-80 yr:             {totals["yr40_80"]:>8,.0f} ha  '
      f'({totals["yr40_80"]/tp*100:4.1f}%)')
print(f'  ─ Total logging loss:                 {total_logging:>8,.0f} ha  '
      f'({total_logging/tp*100:4.1f}%)')
print()
print(f'  Lost to alpine/barren:                {totals["alpine"]:>8,.0f} ha  '
      f'({totals["alpine"]/tp*100:4.1f}%)')
print()
print(f'  Remaining (forest >80 yr only):       {totals["forest"]:>8,.0f} ha  '
      f'({totals["forest"]/tp*100:4.1f}%)')
print(f'  Total logging+alpine loss:            {tl:>8,.0f} ha  '
      f'({tl/tp*100:4.1f}%)')
print('=' * 60)

# Top tiles by logging loss
rows_sorted = sorted(rows, key=lambda r: r['lt20'] + r['yr20_40'] + r['yr40_80'],
                     reverse=True)
print()
print('Top 10 tiles by logging habitat loss:')
for r in rows_sorted[:10]:
    log_loss = r['lt20'] + r['yr20_40'] + r['yr40_80']
    pct = log_loss / r['potential'] * 100 if r['potential'] > 0 else 0
    name = r['slug'].replace('_', ' ').title()
    print(f'  {name:<30}  {log_loss:>6,.0f} ha lost  ({pct:.1f}% of potential)')
