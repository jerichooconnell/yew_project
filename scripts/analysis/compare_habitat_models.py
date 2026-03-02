#!/usr/bin/env python3
"""
Compare production model vs habitat-suitability model across 35 CWH spot tiles.

The habitat model was trained WITHOUT FAIB forestry negatives — it predicts
"would this location support yew based on terrain/environment alone?" versus
the production model which also learned "is this location currently a managed
forest with no yew?"

By applying the habitat model to logged areas, we can estimate what yew
density WOULD have been there pre-disturbance. The difference between
habitat predictions and current (production) predictions is the impact
of logging.

Outputs:
    results/analysis/habitat_comparison.json
    Printed summary to stdout
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'prediction'))
from classify_tiled_gpu import YewMLP

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE / 'results' / 'predictions' / 'south_vi_large'
SPOT_DIR  = BASE / 'results' / 'analysis' / 'cwh_spot_comparisons'
CACHE     = SPOT_DIR / 'tile_cache'

CWH_AREA_HA = 3_595_194
PX_AREA_M2  = 10 * 10
PX_PER_HA   = 10_000 / PX_AREA_M2  # 100


def load_model(model_path, scaler_path, device):
    """Load a YewMLP + scaler."""
    model = YewMLP(input_dim=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler


@torch.no_grad()
def classify_grid(emb_flat, model, scaler, device, batch_size=500_000):
    """Classify a flattened embedding array. Returns probability array."""
    scaled = scaler.transform(emb_flat).astype(np.float32)
    probs = np.zeros(len(scaled), dtype=np.float32)
    for i in range(0, len(scaled), batch_size):
        end = min(i + batch_size, len(scaled))
        batch = torch.from_numpy(scaled[i:end]).to(device)
        probs[i:end] = torch.sigmoid(model(batch)).cpu().numpy()
    return probs


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 78)
    print("  HABITAT MODEL vs PRODUCTION MODEL — 35 CWH SPOT TILES")
    print("=" * 78)
    print(f"Device: {device}")

    # Load both models
    print("\nLoading models...")
    prod_model, prod_scaler = load_model(
        MODEL_DIR / 'mlp_model.pth', MODEL_DIR / 'mlp_scaler.pkl', device)
    hab_model, hab_scaler = load_model(
        MODEL_DIR / 'habitat_model.pth', MODEL_DIR / 'habitat_scaler.pkl', device)
    print("  ✓ Production model (with FAIB negatives)")
    print("  ✓ Habitat model (alpine negatives only)")

    # Load spot stats for site metadata
    with open(SPOT_DIR / 'spot_stats.json') as f:
        stats = json.load(f)

    # ── Per-site analysis ─────────────────────────────────────────────────────
    THRESH = 0.95
    results = []

    for s in stats:
        slug = s['name'].lower().replace(' ', '_').replace('-', '_')
        emb_path = CACHE / f'{slug}_emb.npy'
        log_path = CACHE / f'{slug}_logging.npy'
        if not emb_path.exists() or not log_path.exists():
            continue

        emb = np.load(emb_path)   # (H, W, 64)
        log = np.load(log_path)   # (H, W) — VRI categories
        H, W, C = emb.shape
        flat = emb.reshape(-1, C).astype(np.float32)
        flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)

        # Classify with both models
        prod_probs = classify_grid(flat, prod_model, prod_scaler, device).reshape(H, W)
        hab_probs = classify_grid(flat, hab_model, hab_scaler, device).reshape(H, W)

        r = {'name': s['name'], 'lat': s['lat'], 'lon': s['lon'],
             'total_px': H * W}

        # Per VRI category
        for cat_id, cat_name in [(1, 'water'), (2, 'log_lt20'), (3, 'log_20_40'),
                                  (4, 'log_40_80'), (5, 'forest_gt80'), (6, 'alpine')]:
            mask = (log == cat_id)
            n = int(mask.sum())
            r[cat_name] = n
            r[f'{cat_name}_prod_p95']  = int((prod_probs[mask] >= THRESH).sum())
            r[f'{cat_name}_hab_p95']   = int((hab_probs[mask] >= THRESH).sum())
            r[f'{cat_name}_prod_mean'] = float(prod_probs[mask].mean()) if n > 0 else 0
            r[f'{cat_name}_hab_mean']  = float(hab_probs[mask].mean()) if n > 0 else 0

        r['logged_total'] = r['log_lt20'] + r['log_20_40'] + r['log_40_80']
        r['logged_prod_p95'] = r['log_lt20_prod_p95'] + r['log_20_40_prod_p95'] + r['log_40_80_prod_p95']
        r['logged_hab_p95'] = r['log_lt20_hab_p95'] + r['log_20_40_hab_p95'] + r['log_40_80_hab_p95']
        r['forested_land'] = r['forest_gt80'] + r['logged_total']

        results.append(r)
        print(f"  {s['name']:25s}  forest: prod={r['forest_gt80_prod_p95']/max(r['forest_gt80'],1)*100:5.2f}% "
              f"hab={r['forest_gt80_hab_p95']/max(r['forest_gt80'],1)*100:5.2f}%  "
              f"logged: prod={r['logged_prod_p95']/max(r['logged_total'],1)*100:5.2f}% "
              f"hab={r['logged_hab_p95']/max(r['logged_total'],1)*100:5.2f}%")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    def agg(key):
        return sum(r[key] for r in results)

    N = len(results)
    tot_forest = agg('forest_gt80')
    tot_logged = agg('logged_total')
    tot_forested = tot_forest + tot_logged
    tot_water = agg('water')
    tot_alpine = agg('alpine')
    tot_log_lt20 = agg('log_lt20')
    tot_log_20_40 = agg('log_20_40')
    tot_log_40_80 = agg('log_40_80')

    # Production model: current yew (mature forest only — logged = 0)
    prod_forest_p95 = agg('forest_gt80_prod_p95')
    frac_prod_forest = prod_forest_p95 / tot_forest if tot_forest > 0 else 0

    # Habitat model: predicted suitability in ALL land categories
    hab_forest_p95 = agg('forest_gt80_hab_p95')
    hab_logged_p95 = agg('logged_hab_p95')
    frac_hab_forest = hab_forest_p95 / tot_forest if tot_forest > 0 else 0
    frac_hab_logged = hab_logged_p95 / tot_logged if tot_logged > 0 else 0

    hab_log_lt20_p95 = agg('log_lt20_hab_p95')
    hab_log_20_40_p95 = agg('log_20_40_hab_p95')
    hab_log_40_80_p95 = agg('log_40_80_hab_p95')

    # ── Historical estimate using HABITAT model ──────────────────────────────
    # The habitat model tells us: "what fraction of logged land had suitable
    # conditions for yew?" This is our best estimate of pre-logging yew density
    # in those areas — the terrain/environment supported yew, but logging removed it.
    
    # Historical yew = habitat model P≥0.95 in mature forest + logged areas
    hist_total_px = hab_forest_p95 + hab_logged_p95

    # Current yew = production model P≥0.95 in mature forest only (logged = 0)
    curr_total_px = prod_forest_p95

    decline_sample = (1 - curr_total_px / hist_total_px) * 100 if hist_total_px > 0 else 0

    # ── Extrapolation to CWH zone ────────────────────────────────────────────
    sample_ha = agg('total_px') / PX_PER_HA
    tot_land = agg('total_px') - tot_water - agg('water')  # approximate
    frac_forest_of_land = tot_forest / (agg('total_px') - tot_water) if (agg('total_px') - tot_water) > 0 else 0
    frac_logged_of_land = tot_logged / (agg('total_px') - tot_water) if (agg('total_px') - tot_water) > 0 else 0

    cwh_forest_ha = CWH_AREA_HA * frac_forest_of_land
    cwh_logged_ha = CWH_AREA_HA * frac_logged_of_land

    # Current yew (production model, mature forest only)
    cwh_current_ha = cwh_forest_ha * frac_prod_forest

    # Historical yew (habitat model applied to all forested land)
    cwh_hist_ha = cwh_forest_ha * frac_hab_forest + cwh_logged_ha * frac_hab_logged

    cwh_decline = (1 - cwh_current_ha / cwh_hist_ha) * 100 if cwh_hist_ha > 0 else 0
    cwh_lost_ha = cwh_hist_ha - cwh_current_ha

    # ── Latitude zones ────────────────────────────────────────────────────────
    zones_def = [
        (48, 50, 'South (48–50°N)', 'Vancouver Island + Sunshine Coast'),
        (50, 52, 'Central (50–52°N)', 'North VI + mainland fjords'),
        (52, 56, 'North (52–56°N)', 'Central + north coast'),
    ]
    zone_results = {}
    for lat_lo, lat_hi, zname, zdesc in zones_def:
        zone = [r for r in results if lat_lo <= r['lat'] < lat_hi]
        if not zone:
            continue
        z_for = sum(r['forest_gt80'] for r in zone)
        z_log = sum(r['logged_total'] for r in zone)
        z_prod = sum(r['forest_gt80_prod_p95'] for r in zone)
        z_hab_f = sum(r['forest_gt80_hab_p95'] for r in zone)
        z_hab_l = sum(r['logged_hab_p95'] for r in zone)
        z_hist = z_hab_f + z_hab_l
        z_curr = z_prod  # logged = 0 for current
        z_dec = (1 - z_curr / z_hist) * 100 if z_hist > 0 else 0
        zone_results[zname] = {
            'description': zdesc,
            'n_sites': len(zone),
            'forest_ha': round(z_for / PX_PER_HA),
            'logged_ha': round(z_log / PX_PER_HA),
            'prod_p95_forest_frac': round(z_prod / z_for, 4) if z_for > 0 else 0,
            'hab_p95_forest_frac': round(z_hab_f / z_for, 4) if z_for > 0 else 0,
            'hab_p95_logged_frac': round(z_hab_l / z_log, 4) if z_log > 0 else 0,
            'historical_ha': round(z_hist / PX_PER_HA),
            'current_ha': round(z_curr / PX_PER_HA),
            'decline_pct': round(z_dec, 1),
        }

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        'method': 'Habitat model (no forestry negatives) vs Production model comparison',
        'threshold': THRESH,
        'n_sites': N,
        'sample_area_ha': round(sample_ha),
        'models': {
            'production': 'YewMLP trained with iNat + annotations + FAIB + alpine negatives',
            'habitat': 'YewMLP trained with iNat + annotations + alpine negatives ONLY',
        },
        'sample_statistics': {
            'total_forest_px': tot_forest,
            'total_logged_px': tot_logged,
            'prod_p95_in_forest': prod_forest_p95,
            'prod_p95_frac_forest': round(frac_prod_forest, 4),
            'hab_p95_in_forest': hab_forest_p95,
            'hab_p95_frac_forest': round(frac_hab_forest, 4),
            'hab_p95_in_logged': hab_logged_p95,
            'hab_p95_frac_logged': round(frac_hab_logged, 4),
        },
        'historical_estimate': {
            'note': 'Historical yew = habitat model P≥0.95 in (mature forest + logged areas). '
                    'Current yew = production model P≥0.95 in mature forest only (logged=0).',
            'historical_sample_px': hist_total_px,
            'current_sample_px': curr_total_px,
            'decline_sample_pct': round(decline_sample, 1),
        },
        'cwh_extrapolation': {
            'cwh_area_ha': CWH_AREA_HA,
            'forest_ha': round(cwh_forest_ha),
            'logged_ha': round(cwh_logged_ha),
            'current_yew_ha': round(cwh_current_ha),
            'historical_yew_ha': round(cwh_hist_ha),
            'lost_ha': round(cwh_lost_ha),
            'decline_pct': round(cwh_decline, 1),
        },
        'habitat_suitability_in_logged_areas': {
            'log_lt20': {
                'area_ha': round(tot_log_lt20 / PX_PER_HA),
                'hab_p95_frac': round(hab_log_lt20_p95 / tot_log_lt20, 4) if tot_log_lt20 > 0 else 0,
            },
            'log_20_40': {
                'area_ha': round(tot_log_20_40 / PX_PER_HA),
                'hab_p95_frac': round(hab_log_20_40_p95 / tot_log_20_40, 4) if tot_log_20_40 > 0 else 0,
            },
            'log_40_80': {
                'area_ha': round(tot_log_40_80 / PX_PER_HA),
                'hab_p95_frac': round(hab_log_40_80_p95 / tot_log_40_80, 4) if tot_log_40_80 > 0 else 0,
            },
        },
        'latitude_zones': zone_results,
    }

    out_path = BASE / 'results' / 'analysis' / 'habitat_comparison.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

    # ── Printed summary ──────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("  HISTORICAL YEW POPULATION — HABITAT MODEL APPROACH")
    print("  Production model → current yew | Habitat model → pre-disturbance suitability")
    print("=" * 78)

    print(f"""
APPROACH:
  • Production model: trained with FAIB forestry negatives — predicts where
    yew currently EXISTS. Applied only to mature forest (>80 yr); logged = 0.
  • Habitat model: trained WITHOUT forestry negatives — predicts where the
    environment SUPPORTS yew (terrain, climate, vegetation structure).
    Applied to ALL forested land including logged areas.
  • Historical yew = habitat model P≥{THRESH} across all forested land
  • Current yew = production model P≥{THRESH} in mature forest only

SAMPLE: {N} sites × 10×10 km = {sample_ha:,.0f} ha

MODEL COMPARISON ON MATURE FOREST (>80 yr):
  Production model P≥{THRESH}: {frac_prod_forest*100:.2f}% of mature forest
  Habitat model P≥{THRESH}:    {frac_hab_forest*100:.2f}% of mature forest
  (Habitat model is more liberal — it doesn't penalize forest types
   that look similar to managed stands)

HABITAT SUITABILITY IN LOGGED AREAS:
  The habitat model says this fraction of logged land has conditions
  that would support yew if undisturbed:
  Logged <20 yr:    {hab_log_lt20_p95/max(tot_log_lt20,1)*100:.2f}%
  Logged 20–40 yr:  {hab_log_20_40_p95/max(tot_log_20_40,1)*100:.2f}%
  Logged 40–80 yr:  {hab_log_40_80_p95/max(tot_log_40_80,1)*100:.2f}%
  All logged:       {frac_hab_logged*100:.2f}%

CWH ZONE EXTRAPOLATION:
  Historical yew (habitat model): {cwh_hist_ha:>10,.0f} ha
  Current yew (production model):  {cwh_current_ha:>10,.0f} ha
  Lost to logging:                 {cwh_lost_ha:>10,.0f} ha
  ▸ Decline: {cwh_decline:.1f}%
""")

    for zname, z in zone_results.items():
        print(f"  {zname} — {z['description']}")
        print(f"    Sites: {z['n_sites']}  Forest: {z['forest_ha']:,} ha  Logged: {z['logged_ha']:,} ha")
        print(f"    Production P≥{THRESH} in forest: {z['prod_p95_forest_frac']*100:.2f}%")
        print(f"    Habitat P≥{THRESH} in forest:    {z['hab_p95_forest_frac']*100:.2f}%")
        print(f"    Habitat P≥{THRESH} in logged:    {z['hab_p95_logged_frac']*100:.2f}%")
        print(f"    Historical: {z['historical_ha']:,} ha → Current: {z['current_ha']:,} ha")
        print(f"    ▸ Decline: {z['decline_pct']:.1f}%")
        print()

    print("INTERPRETATION:")
    print("  The habitat model estimates what yew density WOULD exist in logged")
    print("  areas if they had never been cut. Combined with the production model's")
    print("  current estimates in mature forest, this gives a data-driven historical")
    print("  baseline — no assumption that logged areas matched adjacent forest.")
    print()
    print("CAVEATS:")
    print("  1. Habitat model has fewer negatives → more false positives expected")
    print("  2. Satellite embeddings in logged areas reflect current land cover,")
    print("     not pre-disturbance — habitat model may under/over-predict")
    print("  3. Model trained on South VI — performance varies by latitude")
    print("  4. 35 tiles sample ~1.5% of CWH zone")


if __name__ == '__main__':
    main()
