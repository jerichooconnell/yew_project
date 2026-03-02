"""
Yew Population & Historical Decline Analysis
=============================================
Uses 35 spot comparisons (10×10 km tiles, pixel-dense at 10m) with BC VRI
logging overlays to estimate current yew habitat and historical decline
from forestry activity across the BC CWH biogeoclimatic zone.
"""
import json, numpy as np
from pathlib import Path

# ── Load data ─────────────────────────────────────────────────────────────────
with open('results/analysis/cwh_spot_comparisons/spot_stats.json') as f:
    stats = json.load(f)

cache = Path('results/analysis/cwh_spot_comparisons/tile_cache')

CWH_AREA_HA = 3_595_194   # from BEC zone polygon
PX_AREA_M2  = 10 * 10     # 10m pixels
PX_PER_HA   = 10_000 / PX_AREA_M2  # 100 px per ha

# ── Per-site analysis ─────────────────────────────────────────────────────────
results = []
for s in stats:
    slug = s['name'].lower().replace(' ','_').replace('-','_')
    log_path = cache / f'{slug}_logging.npy'
    grid_path = cache / f'{slug}_grid.npy'
    if not log_path.exists() or not grid_path.exists():
        continue
    
    log = np.load(log_path)
    grid = np.load(grid_path)
    
    # Yew does not persist in cut forests — zero out logged pixels
    grid[(log == 2) | (log == 3) | (log == 4)] = 0.0
    
    r = {'name': s['name'], 'lat': s['lat'], 'lon': s['lon'], 'desc': s.get('desc','')}
    r['total_px'] = s['h'] * s['w']
    
    for cat_id, cat_name in [(0,'nodata'),(1,'water'),(2,'log_lt20'),
                              (3,'log_20_40'),(4,'log_40_80'),
                              (5,'forest_gt80'),(6,'alpine')]:
        mask = (log == cat_id)
        r[cat_name] = int(mask.sum())
        r[f'{cat_name}_p95'] = int((grid[mask] >= 0.95).sum())
        r[f'{cat_name}_mean'] = float(grid[mask].mean()) if mask.sum() > 0 else 0.0
    
    r['logged_total'] = r['log_lt20'] + r['log_20_40'] + r['log_40_80']
    r['logged_p95'] = r['log_lt20_p95'] + r['log_20_40_p95'] + r['log_40_80_p95']
    r['forested_land'] = r['forest_gt80'] + r['logged_total']
    results.append(r)

# ── Aggregate ─────────────────────────────────────────────────────────────────
def agg(key):
    return sum(r[key] for r in results)

N_SITES = len(results)
tot_px = agg('total_px')
tot_forest = agg('forest_gt80')
tot_logged = agg('logged_total')
tot_forested = agg('forested_land')
tot_water = agg('water')
tot_alpine = agg('alpine')
tot_nodata = agg('nodata')
tot_log_lt20 = agg('log_lt20')
tot_log_20_40 = agg('log_20_40')
tot_log_40_80 = agg('log_40_80')

# Current yew (P≥0.95 threshold)
p95_forest = agg('forest_gt80_p95')
p95_logged = agg('logged_p95')

p95_log_lt20 = agg('log_lt20_p95')
p95_log_20_40 = agg('log_20_40_p95')
p95_log_40_80 = agg('log_40_80_p95')

frac_yew_forest = p95_forest / tot_forest if tot_forest > 0 else 0
frac_logged = tot_logged / tot_forested if tot_forested > 0 else 0

# Historical: assume logged areas had same yew density as current mature forest
hist_yew_in_logged = int(tot_logged * frac_yew_forest)

hist_total = p95_forest + hist_yew_in_logged
curr_total = p95_forest  # yew does not persist in cut forests

decline_pct = (1 - curr_total / hist_total) * 100 if hist_total > 0 else 0

# ── Extrapolation to full CWH zone ───────────────────────────────────────────
# Sample represents 5,555 km² = 555,500 ha out of 3,595,194 ha CWH
sample_ha = tot_px / PX_PER_HA
expansion_factor = CWH_AREA_HA / sample_ha

# Land composition in sample
frac_forest_of_land = tot_forest / (tot_px - tot_water - tot_nodata) if (tot_px - tot_water - tot_nodata) > 0 else 0
frac_logged_of_land = tot_logged / (tot_px - tot_water - tot_nodata) if (tot_px - tot_water - tot_nodata) > 0 else 0

cwh_forest_ha = CWH_AREA_HA * frac_forest_of_land
cwh_logged_ha = CWH_AREA_HA * frac_logged_of_land
cwh_forested_ha = cwh_forest_ha + cwh_logged_ha

cwh_yew_current_ha = cwh_forest_ha * frac_yew_forest  # yew only in mature forest
cwh_yew_historical_ha = (cwh_forest_ha + cwh_logged_ha) * frac_yew_forest
cwh_decline = (1 - cwh_yew_current_ha / cwh_yew_historical_ha) * 100 if cwh_yew_historical_ha > 0 else 0

# ── Latitude zone analysis ───────────────────────────────────────────────────
zones = [
    (48, 50, 'South (48–50°N)', 'Vancouver Island + Sunshine Coast'),
    (50, 52, 'Central (50–52°N)', 'North VI + mainland fjords'),
    (52, 56, 'North (52–56°N)', 'Central + north coast'),
]

# ── Output ────────────────────────────────────────────────────────────────────
output = {
    'summary': {
        'n_sites': N_SITES,
        'sample_area_ha': round(sample_ha),
        'cwh_zone_area_ha': CWH_AREA_HA,
        'expansion_factor': round(expansion_factor, 1),
    },
    'land_composition_sample': {
        'total_px': tot_px,
        'forest_gt80_px': tot_forest,
        'logged_lt80_px': tot_logged,
        'water_px': tot_water,
        'alpine_px': tot_alpine,
        'frac_forest_of_land': round(frac_forest_of_land, 4),
        'frac_logged_of_land': round(frac_logged_of_land, 4),
    },
    'current_yew_sample': {
        'threshold': 0.95,
        'p95_in_forest_px': p95_forest,
        'p95_frac_forest': round(frac_yew_forest, 4),
        'note': 'Yew assumed absent in all logged areas',
    },
    'historical_reconstruction_sample': {
        'historical_px': hist_total,
        'current_px': curr_total,
        'decline_pct': round(decline_pct, 1),
    },
    'cwh_zone_extrapolation': {
        'threshold': 0.95,
        'current_ha': round(cwh_yew_current_ha),
        'historical_ha': round(cwh_yew_historical_ha),
        'decline_pct': round(cwh_decline, 1),
        'lost_ha': round(cwh_yew_historical_ha - cwh_yew_current_ha),
        'forest_area_ha': round(cwh_forest_ha),
        'logged_area_ha': round(cwh_logged_ha),
    },
    'logged_areas': {
        'assumption': 'Yew does not persist in cut forests — probability set to zero',
        'logged_lt20_ha': round(tot_log_lt20 / PX_PER_HA),
        'logged_20_40_ha': round(tot_log_20_40 / PX_PER_HA),
        'logged_40_80_ha': round(tot_log_40_80 / PX_PER_HA),
        'total_logged_ha': round(tot_logged / PX_PER_HA),
    },
    'latitude_zones': {},
}

for lat_lo, lat_hi, zname, zdesc in zones:
    zone = [r for r in results if lat_lo <= r['lat'] < lat_hi]
    if not zone:
        continue
    z_for = sum(r['forest_gt80'] for r in zone)
    z_log = sum(r['logged_total'] for r in zone)
    z_p95f = sum(r['forest_gt80_p95'] for r in zone)
    z_frac = z_p95f / z_for if z_for > 0 else 0
    z_hist = z_p95f + int(z_log * z_frac)
    z_curr = z_p95f  # yew only in mature forest
    z_dec = (1 - z_curr/z_hist)*100 if z_hist > 0 else 0
    output['latitude_zones'][zname] = {
        'description': zdesc,
        'n_sites': len(zone),
        'forest_ha': round(z_for / PX_PER_HA),
        'logged_ha': round(z_log / PX_PER_HA),
        'p95_forest_frac': round(z_frac, 4),
        'current_yew_ha': round(z_curr / PX_PER_HA),
        'historical_yew_ha': round(z_hist / PX_PER_HA),
        'decline_pct': round(z_dec, 1),
    }

# Save
out_path = Path('results/analysis/yew_population_decline.json')
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"Saved: {out_path}")

# ── Print summary ─────────────────────────────────────────────────────────────
print()
print("=" * 78)
print("  WESTERN YEW (Taxus brevifolia) — POPULATION & DECLINE ESTIMATE")
print("  Based on 35 spot comparisons across BC CWH zone, 10m pixel resolution")
print("=" * 78)

print(f"""
SAMPLE: {N_SITES} sites × 10×10 km = {sample_ha:,.0f} ha ({sample_ha/1e4:.0f} km²)
CWH Zone total: {CWH_AREA_HA:,} ha ({CWH_AREA_HA/1e4:,.0f} km²)
Expansion factor: {expansion_factor:.1f}×

LAND COMPOSITION (in sample):
  Mature forest (>80 yr):   {tot_forest/PX_PER_HA:>10,.0f} ha  ({frac_forest_of_land*100:.1f}% of land)
  Logged (<80 yr):          {tot_logged/PX_PER_HA:>10,.0f} ha  ({frac_logged_of_land*100:.1f}% of land)
  Water / non-forest:       {tot_water/PX_PER_HA:>10,.0f} ha
  Alpine / barren:          {tot_alpine/PX_PER_HA:>10,.0f} ha

CURRENT YEW HABITAT (P≥0.95, mature forest only):
  Yew assumed absent in all logged/cut forests.
  In sample:           {p95_forest/PX_PER_HA:>10,.0f} ha ({frac_yew_forest*100:.2f}% of mature forest)
  CWH zone estimate:   {cwh_yew_current_ha:>10,.0f} ha

LOGGED AREAS (yew = 0):
  Logged <20 yr:        {tot_log_lt20/PX_PER_HA:>10,.0f} ha
  Logged 20–40 yr:      {tot_log_20_40/PX_PER_HA:>10,.0f} ha
  Logged 40–80 yr:      {tot_log_40_80/PX_PER_HA:>10,.0f} ha
  Total logged:         {tot_logged/PX_PER_HA:>10,.0f} ha

HISTORICAL RECONSTRUCTION (pre-logging, P≥0.95):
  Assumes logged land had equivalent yew density to adjacent mature forest
  before harvesting; all yew lost when forest was cut.
    Historical:    {cwh_yew_historical_ha:>10,.0f} ha
    Current:       {cwh_yew_current_ha:>10,.0f} ha
    Lost to logging: {cwh_yew_historical_ha - cwh_yew_current_ha:>10,.0f} ha
    ▸ Decline:     {cwh_decline:.1f}%
""")

for lat_lo, lat_hi, zname, zdesc in zones:
    z = output['latitude_zones'].get(zname)
    if z:
        print(f"  {zname} — {zdesc}")
        print(f"    Sites: {z['n_sites']}  Forest: {z['forest_ha']:,} ha  Logged: {z['logged_ha']:,} ha")
        print(f"    P≥0.95 in forest: {z['p95_forest_frac']*100:.2f}%")
        print(f"    Historical: {z['historical_yew_ha']:,} ha → Current: {z['current_yew_ha']:,} ha")
        print(f"    ▸ Decline: {z['decline_pct']:.1f}%")
        print()

print("CAVEATS:")
print("  1. Model trained on South Vancouver Island — may over/underpredict elsewhere")
print("  2. 'Historical' assumes yew density in logged areas matched adjacent mature forest")
print("  3. 35 tiles sample ~1.5% of CWH zone — extrapolation has wide confidence intervals")
print("  4. Yew assumed completely absent in all logged/cut areas (VRI categories 2-4)")
print("  5. Logging categories from BC VRI (2024); some polygons may have age errors")
