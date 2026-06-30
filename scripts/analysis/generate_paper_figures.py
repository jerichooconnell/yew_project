#!/usr/bin/env python3
"""
Generate publication-quality figures for Pacific Yew habitat decline paper.
Reads BEC analysis CSV + tile cache to produce multiple figure panels.

Output: results/figures/paper/
"""

import csv
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import ticker

sys.path.insert(0, str(Path(__file__).parent))
from figure_style import apply_style, PALETTE
apply_style()

# ── paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "results" / "analysis" / "yew_logging_impact_by_bec.csv"
TILE_CACHE = ROOT / "results" / "analysis" / "cwh_spot_comparisons" / "tile_cache"
TILES_JSON = ROOT / "docs" / "tiles" / "tiles.json"
FIRE_STATS = ROOT / "docs" / "tiles" / "fire_stats.json"
OUT_DIR = ROOT / "results" / "figures" / "paper"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── color palettes ─────────────────────────────────────────────────────
# Zone colours drawn from the shared Okabe-Ito palette (colourblind-safe)
ZONE_COLORS = {
    'CWH': PALETTE['green'],
    'ICH': PALETTE['vermillion'],
    'CDF': PALETTE['orange'],
    'ESSF': PALETTE['blue'],
    'MH':  PALETTE['purple'],
    'IDF': PALETTE['yellow'],
    'CMA': PALETTE['grey'],
    'MS':  PALETTE['sky'],
    'IMA': '#bbbbbb',
    'SBS': '#5a7d2a',
    'BAF': '#b08968',
}

# Yew probability colormap matching web map
YEW_CMAP = LinearSegmentedColormap.from_list('yew', [
    (0.0, '#00000000'),
    (0.02, '#0072B240'),
    (0.33, '#228B2280'),
    (0.50, '#F0E442B0'),
    (0.83, '#E69F00D0'),
    (1.0, '#CC79A7E0'),
])

LOG_CMAP = LinearSegmentedColormap.from_list('logging', [
    (0.0, '#0000FF80'),  # water
    (0.143, '#D55E0090'),  # <20yr
    (0.286, '#D55E0090'),  # 20-40yr
    (0.429, '#E69F0080'),  # 40-80yr
    (0.571, '#7FD1B080'),  # 80-150yr
    (0.714, '#0072B200'),  # alpine (transparent)
    (1.0, '#0072B2B0'),    # old-growth
])


def read_csv():
    """Read BEC analysis CSV and return list of dicts with float conversion."""
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            d = {'bec_subzone': r['bec_subzone'], 'bec_zone': r['bec_zone']}
            for k, v in r.items():
                if k not in ('bec_subzone', 'bec_zone'):
                    try:
                        d[k] = float(v)
                    except ValueError:
                        d[k] = 0.0
            rows.append(d)
    return rows


def aggregate_by_zone(rows):
    """Aggregate to major BEC zone level."""
    zones = defaultdict(lambda: defaultdict(float))
    for r in rows:
        z = r['bec_zone']
        for k, v in r.items():
            if isinstance(v, (int, float)):
                zones[z][k] += v
    # Compute zone-level yew rate
    for z in zones:
        og = zones[z]['oldgrowth_ha']
        if og > 0:
            # Weighted average: sum(raw_model in OG) / sum(OG pixels)
            # We approximate from est_original / (OG + logged)
            pass
    return dict(zones)


# ══════════════════════════════════════════════════════════════════════
# FIGURE 1: Bar chart — estimated original vs current yew by BEC zone
# ══════════════════════════════════════════════════════════════════════
def fig1_zone_comparison(rows):
    """Dumbbell chart (log x) of original→current yew habitat by major zone.

    A log x-axis keeps both the dominant CWH (~111k ha) and the smallest zones
    (~hundreds of ha) legible in one view; the gap between the faded (original)
    and solid (remaining) dot encodes the loss, annotated with % decline.
    """
    zones_data = aggregate_by_zone(rows)
    # ascending so the largest zone sits at the top of a horizontal layout;
    # >100 ha keeps the eight zones reported in Table 1 and drops marginal noise
    zone_list = sorted([z for z in zones_data if zones_data[z]['est_original_yew_ha'] > 100],
                       key=lambda z: zones_data[z]['est_original_yew_ha'])

    fig, ax = plt.subplots(figsize=(11, 6.5))
    y = np.arange(len(zone_list))
    for i, z in enumerate(zone_list):
        orig = zones_data[z]['est_original_yew_ha']
        curr = max(zones_data[z]['current_yew_ha'], 1.0)        # avoid log(0)
        c = ZONE_COLORS.get(z, '#888888')
        ax.plot([curr, orig], [i, i], color='#bbbbbb', lw=2.5, zorder=1, solid_capstyle='round')
        ax.scatter(orig, i, s=95, color=c, alpha=0.40, edgecolor='black', linewidth=0.5, zorder=2)
        ax.scatter(curr, i, s=95, color=c, edgecolor='black', linewidth=0.5, zorder=3)
        pct = (orig - zones_data[z]['current_yew_ha']) / orig * 100 if orig > 0 else 0
        ax.text(orig * 1.18, i, f'−{pct:.0f}%', va='center', fontsize=9.5,
                color='#A33A00', fontweight='bold')

    ax.set_xscale('log')
    ax.set_yticks(y)
    ax.set_yticklabels(zone_list, fontsize=11)
    ax.set_xlim(20, 500_000)
    ax.set_xlabel('Yew habitat (ha, probability mass — log scale)', fontsize=12)
    ax.set_title('Pacific Yew Habitat: Estimated Historical → Current Remaining\n'
                 'by Major BEC Zone (% = decline)', fontsize=14)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.grid(axis='x', alpha=0.3, which='both')

    handles = [
        Line2D([], [], marker='o', ls='', ms=10, mfc='#888', mec='black', alpha=0.4,
               label='Estimated original habitat'),
        Line2D([], [], marker='o', ls='', ms=10, mfc='#444', mec='black',
               label='Current remaining habitat'),
    ]
    ax.legend(handles=handles, fontsize=10, loc='lower right', framealpha=0.95)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig1_zone_comparison.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'fig1_zone_comparison.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Figure 1: Zone comparison (dumbbell, log x)")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 2: Stacked bar — CWH subzone breakdown
# ══════════════════════════════════════════════════════════════════════
def fig2_cwh_subzone_detail(rows):
    """Stacked bar of destroyed vs remaining yew for each CWH subzone."""
    cwh = [r for r in rows if r['bec_zone'] == 'CWH']
    cwh.sort(key=lambda r: r['destroyed_yew_ha'], reverse=True)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    labels = [r['bec_subzone'] for r in cwh]
    destroyed = [r['destroyed_yew_ha'] for r in cwh]
    remaining = [r['current_yew_ha'] for r in cwh]
    
    x = np.arange(len(labels))
    ax.bar(x, remaining, label='Current remaining', color='#009E73', edgecolor='black', linewidth=0.5)
    ax.bar(x, destroyed, bottom=remaining, label='Destroyed by logging', color='#D55E00', alpha=0.8,
           edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('CWH Subzone', fontsize=12)
    ax.set_ylabel('Yew habitat (ha)', fontsize=12)
    ax.set_title('Pacific Yew Habitat Loss in Coastal Western Hemlock (CWH) Subzones', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig2_cwh_subzones.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'fig2_cwh_subzones.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Figure 2: CWH subzone detail")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 3: ICH subzone breakdown (same style)
# ══════════════════════════════════════════════════════════════════════
def fig3_ich_subzone_detail(rows):
    """Stacked bar of destroyed vs remaining yew for each ICH subzone."""
    ich = [r for r in rows if r['bec_zone'] == 'ICH']
    ich.sort(key=lambda r: r['destroyed_yew_ha'], reverse=True)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    labels = [r['bec_subzone'] for r in ich]
    destroyed = [r['destroyed_yew_ha'] for r in ich]
    remaining = [r['current_yew_ha'] for r in ich]
    
    x = np.arange(len(labels))
    ax.bar(x, remaining, label='Current remaining', color='#D2691E', edgecolor='black', linewidth=0.5)
    ax.bar(x, destroyed, bottom=remaining, label='Destroyed by logging', color='#D55E00', alpha=0.8,
           edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('ICH Subzone', fontsize=12)
    ax.set_ylabel('Yew habitat (ha)', fontsize=12)
    ax.set_title('Pacific Yew Habitat Loss in Interior Cedar–Hemlock (ICH) Subzones', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig3_ich_subzones.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'fig3_ich_subzones.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Figure 3: ICH subzone detail")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 4: CDF analysis (single subzone but important)
# ══════════════════════════════════════════════════════════════════════
def fig4_cdf_detail(rows):
    """Bar chart showing CDF habitat breakdown."""
    cdf = [r for r in rows if r['bec_zone'] == 'CDF']
    if not cdf:
        print("  ⚠ No CDF data, skipping Figure 4")
        return
    
    r = cdf[0]  # CDFmm is the only subzone typically
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: land use breakdown
    categories = ['Old-growth\n(>150 yr)', 'Logged\n(<80 yr)', 'Water', 'Alpine/\nBarren']
    values = [r['oldgrowth_ha'], r['logged_ha'], r['water_ha'], r['alpine_ha']]
    colors = ['#0072B2', '#D55E00', '#0072B2', '#A9A9A9']
    ax1.bar(categories, values, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Area (ha)', fontsize=12)
    ax1.set_title('CDF Zone Land Cover', fontsize=13)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax1.grid(axis='y', alpha=0.3)
    
    # Right: yew habitat
    cats2 = ['Estimated\nOriginal', 'Current\nRemaining', 'Destroyed\nby Logging']
    vals2 = [r['est_original_yew_ha'], r['current_yew_ha'], r['destroyed_yew_ha']]
    cols2 = ['#E69F00', '#009E73', '#D55E00']
    bars = ax2.bar(cats2, vals2, color=cols2, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Yew habitat (ha)', fontsize=12)
    ax2.set_title('CDF Yew Habitat: 99% Decline', fontsize=13)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax2.grid(axis='y', alpha=0.3)
    
    # Annotate remaining
    if r['est_original_yew_ha'] > 0:
        pct_remaining = r['current_yew_ha'] / r['est_original_yew_ha'] * 100
        ax2.annotate(f'{pct_remaining:.1f}% remaining',
                    xy=(1, r['current_yew_ha']), xytext=(0, 15),
                    textcoords='offset points', ha='center', fontsize=11,
                    color='red', fontweight='bold')
    
    fig.suptitle('Coastal Douglas-fir (CDF) Zone — Pacific Yew Habitat Analysis', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig4_cdf_detail.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'fig4_cdf_detail.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Figure 4: CDF detail")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 5: Percent decline comparison across all zones
# ══════════════════════════════════════════════════════════════════════
def fig5_percent_decline_all_zones(rows):
    """Horizontal bar chart of % yew decline by major BEC zone."""
    zones_data = aggregate_by_zone(rows)
    
    zone_list = [(z, d) for z, d in zones_data.items() if d['est_original_yew_ha'] > 10]
    zone_list.sort(key=lambda x: (x[1]['est_original_yew_ha'] - x[1]['current_yew_ha']) / max(x[1]['est_original_yew_ha'], 1), reverse=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = [z for z, _ in zone_list]
    pct_decline = [(d['est_original_yew_ha'] - d['current_yew_ha']) / d['est_original_yew_ha'] * 100
                   for _, d in zone_list]
    colors = [ZONE_COLORS.get(z, '#888888') for z in names]
    
    bars = ax.barh(names, pct_decline, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Estimated Yew Habitat Decline (%)', fontsize=12)
    ax.set_title('Pacific Yew Habitat Decline by BEC Zone\n(logging-driven, fire-adjusted)', fontsize=14)
    ax.set_xlim(0, 105)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, pct in zip(bars, pct_decline):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{pct:.0f}%', va='center', fontsize=10, fontweight='bold')
    
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig5_percent_decline.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'fig5_percent_decline.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Figure 5: Percent decline all zones")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 6: Pie chart — overall habitat status
# ══════════════════════════════════════════════════════════════════════
def fig6_overall_pie(rows):
    """Pie chart showing overall habitat breakdown."""
    zones_data = aggregate_by_zone(rows)
    
    total_original = sum(d['est_original_yew_ha'] for d in zones_data.values())
    total_remaining = sum(d['current_yew_ha'] for d in zones_data.values())
    total_fire = sum(d['fire_suppressed_ha'] for d in zones_data.values())
    total_elev = sum(d['elev_suppressed_ha'] for d in zones_data.values())
    total_logging = total_original - total_remaining - total_fire
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: overall status
    labels1 = ['Remaining habitat', 'Destroyed by logging', 'Fire suppression', 'Elevation suppression']
    sizes1 = [total_remaining, max(0, total_logging), total_fire, total_elev]
    colors1 = ['#009E73', '#D55E00', '#E69F00', '#0072B2']
    explode1 = (0.05, 0.02, 0.02, 0.02)
    
    wedges1, texts1, autotexts1 = ax1.pie(
        sizes1, explode=explode1, labels=labels1, colors=colors1,
        autopct=lambda pct: f'{pct:.1f}%' if pct > 2 else '',
        shadow=False, startangle=90, textprops={'fontsize': 10})
    ax1.set_title(f'Overall Yew Habitat Status\n(Est. original: {total_original:,.0f} ha)', fontsize=13)
    
    # Right: remaining by zone
    zone_remaining = [(z, d['current_yew_ha']) for z, d in zones_data.items() if d['current_yew_ha'] > 50]
    zone_remaining.sort(key=lambda x: x[1], reverse=True)
    
    labels2 = [z for z, _ in zone_remaining]
    sizes2 = [v for _, v in zone_remaining]
    colors2 = [ZONE_COLORS.get(z, '#888888') for z in labels2]
    
    wedges2, texts2, autotexts2 = ax2.pie(
        sizes2, labels=labels2, colors=colors2,
        autopct=lambda pct: f'{pct:.1f}%' if pct > 3 else '',
        shadow=False, startangle=90, textprops={'fontsize': 10})
    ax2.set_title(f'Remaining Yew Habitat by Zone\n(Total: {total_remaining:,.0f} ha)', fontsize=13)
    
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig6_overall_pie.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'fig6_overall_pie.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Figure 6: Overall pie charts")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 7: CWH vs ICH vs CDF three-panel comparison
# ══════════════════════════════════════════════════════════════════════
def fig7_three_zone_comparison(rows):
    """Three-panel comparison of CWH, ICH, and CDF zones."""
    zones_data = aggregate_by_zone(rows)
    target_zones = ['CWH', 'ICH', 'CDF']
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    for ax, zone in zip(axes, target_zones):
        if zone not in zones_data:
            ax.set_visible(False)
            continue
        d = zones_data[zone]
        
        categories = ['Old-growth', 'Logged', 'Water', 'Alpine']
        values = [d['oldgrowth_ha'], d['logged_ha'], d['water_ha'], d['alpine_ha']]
        colors = ['#0072B2', '#D55E00', '#0072B2', '#A9A9A9']
        
        ax.bar(categories, values, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_title(f'{zone} Zone\n({d["total_ha"]:,.0f} ha total)', fontsize=13,
                    color=ZONE_COLORS.get(zone, 'black'))
        ax.set_ylabel('Area (ha)')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=30)
        
        # Add yew stats text
        if d['est_original_yew_ha'] > 0:
            pct = (1 - d['current_yew_ha'] / d['est_original_yew_ha']) * 100
            ax.text(0.5, 0.95, f'Yew decline: {pct:.0f}%\n'
                   f'Original: {d["est_original_yew_ha"]:,.0f} ha\n'
                   f'Remaining: {d["current_yew_ha"]:,.0f} ha',
                   transform=ax.transAxes, ha='center', va='top',
                   fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    fig.suptitle('Land Cover and Yew Decline: CWH vs ICH vs CDF', fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig7_three_zone_comparison.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'fig7_three_zone_comparison.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Figure 7: Three-zone comparison")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 8: Logging intensity vs yew decline scatter
# ══════════════════════════════════════════════════════════════════════
def fig8_logging_vs_decline(rows):
    """Scatter of logging fraction vs yew % decline per subzone."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for r in rows:
        total = r['total_ha']
        est_orig = r['est_original_yew_ha']
        if total < 100 or est_orig < 5:
            continue
        
        log_frac = r['logged_ha'] / total * 100
        decline = (est_orig - r['current_yew_ha']) / est_orig * 100
        zone = r['bec_zone']
        color = ZONE_COLORS.get(zone, '#888888')
        
        ax.scatter(log_frac, decline, c=color, s=max(20, min(200, est_orig/50)),
                  edgecolors='black', linewidth=0.5, alpha=0.7, zorder=3)
    
    # Legend for zones
    handles = [mpatches.Patch(color=ZONE_COLORS.get(z, '#888'), label=z)
               for z in ['CWH', 'ICH', 'CDF', 'ESSF', 'MH', 'IDF', 'MS']]
    ax.legend(handles=handles, loc='lower right', fontsize=10)
    
    ax.set_xlabel('Logged Area (% of subzone)', fontsize=12)
    ax.set_ylabel('Estimated Yew Habitat Decline (%)', fontsize=12)
    ax.set_title('Logging Intensity vs. Pacific Yew Decline by BEC Subzone\n(bubble size ∝ original habitat area)', fontsize=14)
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='1:1 line')
    ax.grid(alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig8_logging_vs_decline.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'fig8_logging_vs_decline.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Figure 8: Logging vs decline scatter")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 9: Old-growth yew prevalence rate by subzone
# ══════════════════════════════════════════════════════════════════════
def fig9_og_yew_rate(rows):
    """Horizontal bar of yew prevalence in old-growth by subzone."""
    filtered = [r for r in rows if r['oldgrowth_ha'] > 100 and r['yew_rate_oldgrowth'] > 0.001]
    filtered.sort(key=lambda r: r['yew_rate_oldgrowth'])
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    names = [r['bec_subzone'] for r in filtered]
    rates = [r['yew_rate_oldgrowth'] * 100 for r in filtered]
    colors = [ZONE_COLORS.get(r['bec_zone'], '#888888') for r in filtered]
    
    ax.barh(names, rates, color=colors, edgecolor='black', linewidth=0.3)
    ax.set_xlabel('Mean Yew Probability in Old-Growth (%)', fontsize=12)
    ax.set_title('Pacific Yew Prevalence in Old-Growth Forest\nby BEC Subzone', fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    
    # Add zone legend
    handles = [mpatches.Patch(color=ZONE_COLORS.get(z, '#888'), label=z)
               for z in ['CWH', 'ICH', 'CDF', 'ESSF', 'MH', 'IDF', 'MS']]
    ax.legend(handles=handles, loc='lower right', fontsize=9)
    
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig9_og_yew_rate.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'fig9_og_yew_rate.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Figure 9: Old-growth yew rate")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 10: Fire suppression by decade (if fire_stats.json exists)
# ══════════════════════════════════════════════════════════════════════
def fig10_fire_by_decade():
    """Bar chart of fire suppression by decade."""
    if not FIRE_STATS.exists():
        print("  ⚠ fire_stats.json not found, skipping Figure 10")
        return
    
    with open(FIRE_STATS) as f:
        data = json.load(f)
    
    if not data or not isinstance(data, dict) or 'decades' not in data:
        print("  ⚠ fire_stats.json empty or unexpected format, skipping Figure 10")
        return
    
    decade_list = data['decades']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    decades = [d['decade_label'] for d in decade_list]
    burned = [d.get('burned_ha', 0) for d in decade_list]
    suppressed = [d.get('yew_suppressed_ha', 0) for d in decade_list]
    
    x = np.arange(len(decades))
    ax.bar(x, burned, label='Total burned area (ha)', color='#E69F00', alpha=0.5,
           edgecolor='black', linewidth=0.5)
    ax.bar(x, suppressed, label='Yew habitat suppressed (ha)', color='#D55E00',
           edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Decade', fontsize=12)
    ax.set_ylabel('Area (ha)', fontsize=12)
    ax.set_title('Wildfire Impact on Pacific Yew Habitat by Decade', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(decades, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig10_fire_decades.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'fig10_fire_decades.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Figure 10: Fire by decade")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 11: Example tile — yew probability + logging overlay
# ══════════════════════════════════════════════════════════════════════
def fig11_example_tiles():
    """4-panel showing example tiles with yew probability and logging."""
    # Pick representative tiles from available cache
    example_slugs = [
        ('port_renfrew', 'Port Renfrew (CWHvm1)'),
        ('sooke_hills', 'Sooke Hills (CWHxm1)'),
        ('seaforth_channel', 'Seaforth Channel (CWHvh2)'),
        ('muchalat_valley', 'Muchalat Valley (CWHmm)'),
    ]
    
    available = []
    for slug, label in example_slugs:
        grid_path = TILE_CACHE / f"{slug}_grid.npy"
        log_path = TILE_CACHE / f"{slug}_logging.npy"
        if grid_path.exists() and log_path.exists():
            available.append((slug, label, grid_path, log_path))
    
    if len(available) < 2:
        print("  ⚠ Not enough tile data for example tiles, skipping Figure 11")
        return
    
    n = min(4, len(available))
    fig, axes = plt.subplots(2, n, figsize=(4*n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)
    
    for i, (slug, label, grid_path, log_path) in enumerate(available[:n]):
        grid = np.load(grid_path)
        log = np.load(log_path)
        
        # Top row: yew probability
        ax_yew = axes[0, i] if n > 1 else axes[0, 0]
        im1 = ax_yew.imshow(grid, cmap='YlOrRd', vmin=0, vmax=1, interpolation='nearest')
        ax_yew.set_title(f'{label}\nYew probability', fontsize=10)
        ax_yew.axis('off')
        
        # Bottom row: logging
        ax_log = axes[1, i] if n > 1 else axes[1, 0]
        im2 = ax_log.imshow(log, cmap='RdYlGn', vmin=1, vmax=7, interpolation='nearest')
        ax_log.set_title('Logging status', fontsize=10)
        ax_log.axis('off')
    
    # Colorbars
    cbar1 = fig.colorbar(im1, ax=axes[0, :] if n > 1 else axes[0, 0], shrink=0.8, pad=0.02)
    cbar1.set_label('P(yew)')
    
    # Logging legend
    log_labels = {1: 'Water', 2: '<20 yr', 3: '20-40 yr', 4: '40-80 yr',
                  5: '80-150 yr', 6: 'Alpine', 7: 'Old-growth'}
    cbar2 = fig.colorbar(im2, ax=axes[1, :] if n > 1 else axes[1, 0], shrink=0.8, pad=0.02,
                        ticks=list(log_labels.keys()))
    cbar2.ax.set_yticklabels(list(log_labels.values()), fontsize=8)
    
    fig.suptitle('Example Study Tiles: Yew Probability and Forest Age', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig11_example_tiles.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'fig11_example_tiles.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Figure 11: Example tiles")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 12: Suppression waterfall chart
# ══════════════════════════════════════════════════════════════════════
def fig12_suppression_waterfall(rows):
    """Waterfall chart showing raw model → logging suppression → fire → elevation → final."""
    zones_data = aggregate_by_zone(rows)
    
    total_raw = sum(d['raw_model_yew_ha'] for d in zones_data.values())
    total_current = sum(d['current_yew_ha'] for d in zones_data.values())
    total_fire = sum(d['fire_suppressed_ha'] for d in zones_data.values())
    total_elev = sum(d['elev_suppressed_ha'] for d in zones_data.values())
    total_logging_suppression = total_raw - total_current - total_fire - total_elev
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    steps = ['Raw model\nprediction', 'After logging\nsuppression', 'After fire\nsuppression', 
             'After elevation\nsuppression', 'Final\ncurrent habitat']
    vals = [total_raw, 
            total_raw - total_logging_suppression,
            total_raw - total_logging_suppression - total_fire,
            total_raw - total_logging_suppression - total_fire - total_elev,
            total_current]
    
    colors = ['#0072B2', '#E69F00', '#D55E00', '#8B4513', '#009E73']
    
    bars = ax.bar(range(len(steps)), vals, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(steps, fontsize=10)
    ax.set_ylabel('Predicted yew habitat (ha)', fontsize=12)
    ax.set_title('Post-Classification Suppression Pipeline\n(all zones combined)', fontsize=14)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.grid(axis='y', alpha=0.3)
    
    # Annotate reductions
    for i in range(1, len(vals)):
        diff = vals[i-1] - vals[i]
        if diff > 0:
            mid_y = (vals[i] + vals[i-1]) / 2
            ax.annotate(f'−{diff:,.0f} ha', xy=(i, vals[i]),
                       xytext=(0, 10), textcoords='offset points',
                       ha='center', fontsize=9, color='red')
    
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig12_suppression_waterfall.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'fig12_suppression_waterfall.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Figure 12: Suppression waterfall")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 13: Logging age class distribution for CWH / ICH / CDF
# ══════════════════════════════════════════════════════════════════════
def fig13_logging_age_classes(rows):
    """Grouped bar: recent vs older logging by zone."""
    target = ['CWH', 'ICH', 'CDF']
    zones_data = aggregate_by_zone(rows)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(target))
    w = 0.2
    
    lt20 = [zones_data.get(z, {}).get('logged_lt20yr_ha', 0) for z in target]
    a20_40 = [zones_data.get(z, {}).get('logged_20_40yr_ha', 0) for z in target]
    a40_80 = [zones_data.get(z, {}).get('logged_40_80yr_ha', 0) for z in target]
    
    ax.bar(x - w, lt20, w, label='<20 yr (clearcut)', color='#D55E00', alpha=0.8,
           edgecolor='black', linewidth=0.5)
    ax.bar(x, a20_40, w, label='20–40 yr', color='#D55E00', alpha=0.8,
           edgecolor='black', linewidth=0.5)
    ax.bar(x + w, a40_80, w, label='40–80 yr', color='#E69F00', alpha=0.8,
           edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('BEC Zone', fontsize=12)
    ax.set_ylabel('Logged area (ha)', fontsize=12)
    ax.set_title('Logging Age Classes in CWH, ICH, and CDF Zones', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(target, fontsize=12)
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig13_logging_age_classes.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'fig13_logging_age_classes.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Figure 13: Logging age classes")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 14: Model performance summary
# ══════════════════════════════════════════════════════════════════════
def fig14_model_performance():
    """Grouped bar chart comparing classifiers (AUC-ROC, Accuracy)."""
    models = [
        ('XGBoost\n(production)', 0.9957, 0.989),
        ('MLP +\nStandardScaler', 0.9961, 0.986),
        ('MLP raw', 0.9962, 0.976),
        ('Random\nForest', 0.9896, 0.984),
        ('kNN (k=3)', 0.9909, 0.911),
        ('Logistic\nRegression', 0.9165, 0.813),
    ]
    names = [m[0] for m in models]
    metrics = ['AUC-ROC', 'Accuracy']
    colors = ['#009E73', '#0072B2']
    vals = np.array([[m[1], m[2]] for m in models])

    x = np.arange(len(models))
    w = 0.36
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for k, (metric, c) in enumerate(zip(metrics, colors)):
        bars = ax.bar(x + (k - 0.5) * w, vals[:, k], w, label=metric, color=c,
                      edgecolor='white', linewidth=0.5, zorder=3)
        for b, v in zip(bars, vals[:, k]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=7.5, rotation=90)

    # Zoom y to 0.5–1.0 so the near-ceiling differences are visible; leave
    # headroom above the rotated value labels for the production-model marker
    ax.set_ylim(0.5, 1.11)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Classifier Performance on 64-band AlphaEarth Embeddings\n'
                 '(held-out validation set; y-axis truncated at 0.5)', fontsize=13)
    ax.axvline(0.5, color='#009E73', ls=':', lw=1.2, alpha=0.7, zorder=1)
    ax.annotate('production model', xy=(0, 1.085), fontsize=8.5, color='#009E73',
                ha='center', style='italic')
    ax.legend(fontsize=10, loc='lower left', framealpha=0.95)
    ax.grid(axis='y', alpha=0.3, zorder=0)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig14_model_performance.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'fig14_model_performance.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Figure 14: Model performance bar chart")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 15: Threats summary infographic
# ══════════════════════════════════════════════════════════════════════
def fig15_threats_summary(rows):
    """Two-panel threats figure: direct habitat loss (log ha) + demographic impacts.

    The original single linear bar made logging (~107k ha) dwarf every other
    threat into invisibility and left three threats as empty bars. Here the
    area-quantified threats use a log x-axis (so fire/erosion/sea-level remain
    legible) and the demographic threats are shown on their own % axis with
    literature ranges, since they are not expressible in hectares.
    """
    zones_data = aggregate_by_zone(rows)
    total_original = sum(d['est_original_yew_ha'] for d in zones_data.values())
    total_remaining = sum(d['current_yew_ha'] for d in zones_data.values())
    total_fire = sum(d['fire_suppressed_ha'] for d in zones_data.values())
    logging_loss = total_original - total_remaining

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(11, 8.5),
                                   gridspec_kw={'height_ratios': [1, 0.85]})

    # ── Panel A: area-quantified threats (log x) ──────────────────────────
    quant = [
        ('Clear-cut logging\n(modelled)', logging_loss, '#D55E00'),
        ('Stream erosion, 30 m buffer\n(modelled, 42-tile subset)', 1717, '#E69F00'),
        ('Wildfire, historical\n(modelled)', total_fire, '#E69F00'),
        ('Sea-level rise\n(projected, <1%)', 240, '#0072B2'),
    ]
    names = [q[0] for q in quant]
    vals = [q[1] for q in quant]
    yA = range(len(quant))
    axA.barh(list(yA), vals, color=[q[2] for q in quant], edgecolor='black', linewidth=0.5)
    axA.set_yticks(list(yA))
    axA.set_yticklabels(names, fontsize=10)
    axA.set_xscale('log')
    axA.set_xlim(50, 300_000)
    axA.set_xlabel('Direct habitat loss (ha of yew habitat — log scale)', fontsize=11)
    axA.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    axA.grid(axis='x', alpha=0.3, which='both')
    axA.invert_yaxis()
    axA.set_title('Direct habitat loss (area-quantified)', fontsize=12)
    for i, v in enumerate(vals):
        axA.text(v * 1.25, i, f'{v:,.0f} ha', va='center', fontsize=9.5)

    # ── Panel B: demographic / fecundity threats (% impact ranges) ────────
    demo = [
        ('Ungulate browsing\n(seedling/sapling mortality)', 60, 80, '#8B4513'),
        ('Yew big bud mite\n(seed/aril production loss)', 20, 25, '#9370DB'),
        ('Yew big bud mite\n(terminal bud mortality)', 20, 25, '#B19CD9'),
    ]
    yB = range(len(demo))
    for i, (lab, lo, hi, c) in enumerate(demo):
        axB.barh(i, hi - lo, left=lo, color=c, edgecolor='black', linewidth=0.5, height=0.5)
        axB.text(hi + 1.5, i, f'{lo}–{hi}%', va='center', fontsize=9.5)
    axB.set_yticks(list(yB))
    axB.set_yticklabels([d[0] for d in demo], fontsize=10)
    axB.set_xlim(0, 100)
    axB.set_xlabel('Literature-estimated impact (% of affected cohort)', fontsize=11)
    axB.grid(axis='x', alpha=0.3)
    axB.invert_yaxis()
    axB.set_title('Demographic / fecundity threats (not expressible in ha)', fontsize=12)
    axB.text(0.99, -0.32, 'Historical taxol bark harvest (1989–1993): ~360,000 mature '
             'trees/yr felled across the Pacific Northwest — not spatially documented.',
             transform=axB.transAxes, ha='right', va='top', fontsize=8.5,
             style='italic', color='#555')

    fig.suptitle('Threats to Pacific Yew Populations in British Columbia', fontsize=14, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig15_threats_summary.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'fig15_threats_summary.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Figure 15: Threats summary (2-panel)")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 16: Heatmap of yew probability by subzone
# ══════════════════════════════════════════════════════════════════════
def fig16_heatmap(rows):
    """Standardised (z-score) heatmap of complementary subzone metrics.

    The original max-normalised three habitat-area columns were near-duplicates
    (large subzones large everywhere) and colours were not comparable across
    columns of different units. Here the columns are chosen to be complementary
    — prevalence, original size, % decline, % logged — and each is standardised
    (z-score) so colour means the same thing in every column: how far above (red)
    or below (blue) the cross-subzone average a subzone sits. Printed cells show
    the raw values.
    """
    filtered = [r for r in rows if r['est_original_yew_ha'] > 10]
    filtered.sort(key=lambda r: r['destroyed_yew_ha'], reverse=True)
    filtered = filtered[:25]

    def pct_decline(r):
        return r['destroyed_yew_ha'] / r['est_original_yew_ha'] * 100 if r['est_original_yew_ha'] > 0 else 0

    def pct_logged(r):
        denom = r['total_ha'] - r.get('water_ha', 0) - r.get('alpine_ha', 0)
        return r['logged_ha'] / denom * 100 if denom > 0 else 0

    cols = [
        ('OG yew\nprevalence', [r['yew_rate_oldgrowth'] for r in filtered], '{:.2f}'),
        ('Est. original\nhabitat (ha)', [r['est_original_yew_ha'] for r in filtered], '{:,.0f}'),
        ('Habitat\ndecline (%)', [pct_decline(r) for r in filtered], '{:.0f}'),
        ('Forest area\nlogged (%)', [pct_logged(r) for r in filtered], '{:.0f}'),
    ]
    labels = [r['bec_subzone'] for r in filtered]
    raw = np.array([c[1] for c in cols], dtype=float).T          # (n_subzone, n_col)

    # z-score per column
    z = np.zeros_like(raw)
    for j in range(raw.shape[1]):
        mu, sd = raw[:, j].mean(), raw[:, j].std()
        z[:, j] = (raw[:, j] - mu) / sd if sd > 0 else 0.0
    vmax = np.abs(z).max()

    fig, ax = plt.subplots(figsize=(8.5, 10))
    im = ax.imshow(z, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[0] for c in cols], fontsize=10)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(len(labels)):
        for j in range(len(cols)):
            ax.text(j, i, cols[j][2].format(raw[i, j]), ha='center', va='center',
                    fontsize=7.5, color='white' if abs(z[i, j]) > vmax * 0.55 else '#222')

    ax.set_title('Top 25 BEC Subzones by Yew Habitat Loss\n'
                 '(colour = z-score within column; cells = raw values)', fontsize=12)
    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label('Standard deviations from cross-subzone mean', fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig16_heatmap.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'fig16_heatmap.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Figure 16: Heatmap (z-score, complementary metrics)")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 17: Old-growth vs logged area ratio by zone
# ══════════════════════════════════════════════════════════════════════
def fig17_og_logged_ratio(rows):
    """Show ratio of old-growth to logged for CWH, ICH, CDF."""
    zones_data = aggregate_by_zone(rows)
    target = ['CWH', 'ICH', 'CDF']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(target))
    w = 0.35
    
    og_vals = [zones_data.get(z, {}).get('oldgrowth_ha', 0) for z in target]
    log_vals = [zones_data.get(z, {}).get('logged_ha', 0) for z in target]
    
    ax.bar(x - w/2, og_vals, w, label='Old-growth (>150 yr)', color='#0072B2',
           edgecolor='black', linewidth=0.5)
    ax.bar(x + w/2, log_vals, w, label='Logged (<150 yr)', color='#D55E00',
           edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('BEC Zone', fontsize=12)
    ax.set_ylabel('Area (ha)', fontsize=12)
    ax.set_title('Old-Growth vs. Logged Forest: CWH, ICH, CDF', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(target, fontsize=12)
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.grid(axis='y', alpha=0.3)
    
    # Add ratios
    for i, z in enumerate(target):
        if log_vals[i] > 0:
            ratio = og_vals[i] / log_vals[i]
            ax.text(i, max(og_vals[i], log_vals[i]) * 1.05,
                   f'OG:Log = {ratio:.2f}', ha='center', fontsize=10, fontweight='bold')
    
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig17_og_logged_ratio.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'fig17_og_logged_ratio.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Figure 17: OG vs logged ratio")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 18: Study area map (text-based, from tiles.json)
# ══════════════════════════════════════════════════════════════════════
def fig18_study_area_map():
    """Simple lat/lon scatter of study tile locations."""
    if not TILES_JSON.exists():
        print("  ⚠ tiles.json not found, skipping Figure 18")
        return
    
    with open(TILES_JSON) as f:
        tiles = json.load(f)
    
    fig, ax = plt.subplots(figsize=(10, 9))

    # Land backdrop: simplified BC boundary for geographic context
    bc_path = ROOT / "results" / "analysis" / "cwh_yew_population_100k" / "bc_boundary_simplified.geojson"
    if bc_path.exists():
        try:
            import geopandas as gpd
            bc = gpd.read_file(bc_path).to_crs(4326)
            bc.plot(ax=ax, color='#eef2ee', edgecolor='#9bb09b', linewidth=0.6, zorder=1)
        except Exception as exc:
            print(f"    (BC boundary not drawn: {exc})")

    n_coastal = n_ich = 0
    for t in tiles:
        lat = t.get('lat', 0)
        lon = t.get('lon', t.get('lng', 0))          # tiles.json uses 'lon'
        slug = t.get('slug', '')

        is_ich = slug.lower().startswith('ich')      # ICH tiles are 'ich_ptNN'
        if is_ich:
            color, marker, s = ZONE_COLORS['ICH'], '^', 70
            n_ich += 1
        else:
            color, marker, s = ZONE_COLORS['CWH'], 'o', 45
            n_coastal += 1

        ax.scatter(lon, lat, c=color, marker=marker, s=s,
                   edgecolors='black', linewidth=0.5, zorder=3)

    ax.set_xlabel('Longitude (°W)', fontsize=12)
    ax.set_ylabel('Latitude (°N)', fontsize=12)
    ax.set_title(f'Study Area Locations\n{len(tiles)} tiles across British Columbia', fontsize=14)
    ax.grid(alpha=0.25, zorder=0)

    handles = [
        plt.scatter([], [], c=ZONE_COLORS['CWH'], marker='o', s=45, edgecolors='black',
                    label=f'Coastal tiles ({n_coastal})'),
        plt.scatter([], [], c=ZONE_COLORS['ICH'], marker='^', s=70, edgecolors='black',
                    label=f'ICH tiles ({n_ich})'),
    ]
    ax.legend(handles=handles, fontsize=11, loc='lower left')

    # Frame on the populated coastal+interior extent (with a small margin)
    ax.set_xlim(-134, -114.5)
    ax.set_ylim(48, 56)
    ax.set_aspect(1.0 / np.cos(np.radians(52)))      # approx. equal-distance at BC latitudes
    
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig18_study_area_map.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'fig18_study_area_map.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Figure 18: Study area map")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 19: Combined CWH+ICH+CDF subzone decline waterfall
# ══════════════════════════════════════════════════════════════════════
def fig19_zone_waterfall(rows):
    """Three-panel waterfall: original → remaining for each focus zone."""
    zones = ['CWH', 'ICH', 'CDF']
    zones_data = aggregate_by_zone(rows)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    for ax, zone in zip(axes, zones):
        d = zones_data.get(zone, {})
        if not d:
            ax.set_visible(False)
            continue
        
        est_orig = d.get('est_original_yew_ha', 0)
        current = d.get('current_yew_ha', 0)
        fire = d.get('fire_suppressed_ha', 0)
        elev = d.get('elev_suppressed_ha', 0)
        logging_loss = est_orig - current - fire - elev
        
        cats = ['Original', 'Logging\nloss', 'Fire\nloss', 'Elev.\nloss', 'Remaining']
        vals = [est_orig, -logging_loss, -fire, -elev, current]
        cumulative = [est_orig, est_orig - logging_loss, est_orig - logging_loss - fire,
                     est_orig - logging_loss - fire - elev, current]
        
        colors = [ZONE_COLORS.get(zone, '#888'), '#D55E00', '#E69F00', '#0072B2', '#009E73']
        
        ax.bar(range(len(cats)), cumulative, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats, fontsize=9)
        ax.set_ylabel('Yew habitat (ha)')
        ax.set_title(f'{zone} Zone', fontsize=13, color=ZONE_COLORS.get(zone, 'black'))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
        ax.grid(axis='y', alpha=0.3)
        
        if est_orig > 0:
            pct = (1 - current/est_orig) * 100
            ax.text(0.5, 0.95, f'{pct:.0f}% decline',
                   transform=ax.transAxes, ha='center', va='top',
                   fontsize=12, fontweight='bold', color='red',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.suptitle('Yew Habitat Decline Pathway: CWH vs ICH vs CDF', fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig19_zone_waterfall.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'fig19_zone_waterfall.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Figure 19: Zone waterfall comparison")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print(f"Reading data from {CSV_PATH}")
    rows = read_csv()
    print(f"  {len(rows)} BEC subzones loaded")
    print(f"\nGenerating figures in {OUT_DIR}/\n")
    
    fig1_zone_comparison(rows)
    fig2_cwh_subzone_detail(rows)
    fig3_ich_subzone_detail(rows)
    fig4_cdf_detail(rows)
    fig5_percent_decline_all_zones(rows)
    fig6_overall_pie(rows)
    fig7_three_zone_comparison(rows)
    fig8_logging_vs_decline(rows)
    fig9_og_yew_rate(rows)
    fig10_fire_by_decade()
    fig11_example_tiles()
    fig12_suppression_waterfall(rows)
    fig13_logging_age_classes(rows)
    fig14_model_performance()
    fig15_threats_summary(rows)
    fig16_heatmap(rows)
    fig17_og_logged_ratio(rows)
    fig18_study_area_map()
    fig19_zone_waterfall(rows)
    
    print(f"\n✓ All figures saved to {OUT_DIR}/")
    
    # Print summary stats for the paper
    zones_data = aggregate_by_zone(rows)
    total_orig = sum(d['est_original_yew_ha'] for d in zones_data.values())
    total_curr = sum(d['current_yew_ha'] for d in zones_data.values())
    total_fire = sum(d['fire_suppressed_ha'] for d in zones_data.values())
    total_elev = sum(d['elev_suppressed_ha'] for d in zones_data.values())
    
    print("\n" + "="*70)
    print("PAPER STATISTICS SUMMARY")
    print("="*70)
    print(f"Study tiles:              99 (85 coastal + 14 interior)")
    print(f"Total analyzed area:      ~9,900 km²")
    print(f"BEC zones covered:        {len(zones_data)}")
    print(f"BEC subzones analyzed:    {len(rows)}")
    print(f"Estimated original yew:   {total_orig:,.0f} ha")
    print(f"Current remaining yew:    {total_curr:,.0f} ha")
    print(f"Overall decline:          {(1 - total_curr/total_orig)*100:.1f}%")
    print(f"Destroyed by logging:     {total_orig - total_curr:,.0f} ha")
    print(f"Fire suppression:         {total_fire:,.0f} ha")
    print(f"Elevation suppression:    {total_elev:,.0f} ha")
    print()
    
    for zone in ['CWH', 'ICH', 'CDF']:
        d = zones_data.get(zone, {})
        if d and d.get('est_original_yew_ha', 0) > 0:
            pct = (1 - d['current_yew_ha']/d['est_original_yew_ha'])*100
            print(f"{zone}: original={d['est_original_yew_ha']:,.0f} ha, "
                  f"remaining={d['current_yew_ha']:,.0f} ha, "
                  f"decline={pct:.1f}%, "
                  f"fire={d['fire_suppressed_ha']:,.0f} ha, "
                  f"logged={d['logged_ha']:,.0f} ha")


if __name__ == '__main__':
    main()
