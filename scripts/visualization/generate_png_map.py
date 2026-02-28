"""
Generate static PNG map of 100k CWH yew predictions.
No cartopy required — uses matplotlib + geopandas.
"""
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
import shapely.geometry as sg

# ── 1. Load predictions ────────────────────────────────────────────────────────
csv_path = 'results/analysis/cwh_yew_population_100k/sample_predictions_cwh.csv'
df = pd.read_csv(csv_path)
print(f"Loaded {len(df):,} prediction points")
print(df[['lat','lon','prob']].describe().to_string())

# ── 2. Rasterize points into a probability grid ───────────────────────────────
resolution = 0.04   # ~4 km at mid-latitudes
lon_min, lon_max = df['lon'].min(), df['lon'].max()
lat_min, lat_max = df['lat'].min(), df['lat'].max()
pad = 0.4
lon_min -= pad;  lon_max += pad
lat_min -= pad;  lat_max += pad

nx = int((lon_max - lon_min) / resolution) + 1
ny = int((lat_max - lat_min) / resolution) + 1
print(f"Grid: {nx} × {ny} cells")

col_idx = np.floor((df['lon'] - lon_min) / resolution).astype(int).clip(0, nx - 1)
row_idx = np.floor((df['lat'] - lat_min) / resolution).astype(int).clip(0, ny - 1)

grid_sum   = np.zeros((ny, nx), dtype=np.float32)
grid_count = np.zeros((ny, nx), dtype=np.int32)
np.add.at(grid_sum,   (row_idx, col_idx), df['prob'].values)
np.add.at(grid_count, (row_idx, col_idx), 1)

grid_prob = np.where(grid_count > 0, grid_sum / grid_count, np.nan)

# Gentle smooth to reduce stippling
valid_mask = ~np.isnan(grid_prob)
filled     = np.where(valid_mask, grid_prob, 0.0)
smoothed   = gaussian_filter(filled, sigma=0.9)
grid_prob  = np.where(valid_mask, smoothed, np.nan)

# Flip: imshow row-0 = top = north
grid_prob_plot = np.flipud(grid_prob)
extent = [lon_min, lon_max, lat_min, lat_max]
print(f"Prob range: {np.nanmin(grid_prob):.3f}–{np.nanmax(grid_prob):.3f}, "
      f"valid cells: {valid_mask.sum():,}")

# ── 3. Geographic layers ──────────────────────────────────────────────────────
print("Loading geographic layers …")
ne_land_path = '/tmp/ne_10m_land.geojson'
cwh_path     = 'data/processed/cwh_negatives/cwh_boundary_forestry.gpkg'

land = gpd.read_file(ne_land_path)
cwh  = gpd.read_file(cwh_path)

bbox_shp  = sg.box(lon_min, lat_min, lon_max, lat_max)
bbox_gs   = gpd.GeoDataFrame(geometry=[bbox_shp], crs='EPSG:4326')
land_clip = gpd.clip(land, bbox_gs)

# Also try NaturalEarth countries for border lines
try:
    countries = gpd.read_file('/tmp/ne_10m_admin_0_countries.geojson')
    countries_clip = gpd.clip(countries, bbox_gs)
    has_countries = len(countries_clip) > 0
except Exception:
    has_countries = False

print(f"  Land polygons: {len(land_clip)}, CWH: {len(cwh)}, "
      f"Countries: {has_countries}")

# ── 4. Custom colormap ────────────────────────────────────────────────────────
# YlOrRd, but transparent for very low probabilities
cmap_base  = plt.cm.YlOrRd
colors_arr = cmap_base(np.linspace(0, 1, 256))
cutoff     = 0.10   # fade in below 10% probability
for i in range(int(cutoff * 256)):
    alpha = (i / (cutoff * 256)) ** 0.8 * 0.5
    colors_arr[i, 3] = alpha
cmap_custom = mcolors.ListedColormap(colors_arr)
cmap_custom.set_bad(alpha=0.0)          # NaN → transparent
norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

# ── 5. Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 12), dpi=200)
fig.patch.set_facecolor('#c8dff0')
ax.set_facecolor('#c8dff0')   # ocean

# Land background
land_clip.plot(ax=ax, facecolor='#f2efe9', edgecolor='#999999',
               linewidth=0.4, zorder=1)

# Probability raster (bilinear so grid edges are soft)
im = ax.imshow(grid_prob_plot, extent=extent,
               cmap=cmap_custom, norm=norm,
               aspect='auto', zorder=2,
               interpolation='bilinear', origin='upper')

# Country / province borders on top of raster but under CWH
if has_countries:
    countries_clip.boundary.plot(ax=ax, edgecolor='#555555',
                                  linewidth=0.7, zorder=3)
else:
    land_clip.boundary.plot(ax=ax, edgecolor='#666666',
                             linewidth=0.4, zorder=3)

# CWH boundary — prominent dashed outline
cwh.plot(ax=ax, facecolor='none', edgecolor='#111111',
         linewidth=1.8, linestyle='--', zorder=5)

# ── 6. Colorbar ───────────────────────────────────────────────────────────────
cax = fig.add_axes([0.14, 0.075, 0.52, 0.022])
cb  = fig.colorbar(im, cax=cax, orientation='horizontal')
cb.set_label('Predicted occurrence probability', fontsize=10)
cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cb.ax.axvline(x=0.5, color='black', linewidth=1.8, linestyle='--')
# Arrow annotation pointing at threshold line
cb.ax.annotate('P = 0.5\n(threshold)',
               xy=(0.5, 1.0), xytext=(0.5, 2.9),
               xycoords='axes fraction', textcoords='axes fraction',
               ha='center', va='bottom', fontsize=8,
               arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

# ── 7. Legend ─────────────────────────────────────────────────────────────────
legend_elements = [
    Line2D([0], [0], linestyle='--', color='#111111',
           linewidth=1.8, label='CWH zone boundary'),
    Patch(facecolor='#f2efe9', edgecolor='#999999', label='Land'),
    Patch(facecolor='#c8dff0', label='Ocean'),
]
ax.legend(handles=legend_elements, loc='lower left',
          fontsize=9, framealpha=0.92, edgecolor='#aaaaaa',
          title='Map layers', title_fontsize=9)

# ── 8. Title & axes ───────────────────────────────────────────────────────────
ax.set_title(
    'Predicted Western Yew  (Taxus brevifolia)  Occurrence Probability\n'
    'Coastal Western Hemlock (CWH) Biogeoclimatic Zone — British Columbia',
    fontsize=13, fontweight='bold', pad=16
)
ax.set_xlabel('Longitude (°)', fontsize=9)
ax.set_ylabel('Latitude (°)', fontsize=9)
ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)
ax.tick_params(labelsize=8)
ax.grid(True, linestyle=':', linewidth=0.4, color='#aaaaaa', alpha=0.6, zorder=0)

# ── 9. Stats box ──────────────────────────────────────────────────────────────
stats = (
    "n = 99,869 sample points\n"
    "P ≥ 0.5:  314,416 ha  (95% CI: 308–321 k ha)\n"
    "P ≥ 0.3:  411,758 ha\n"
    "P ≥ 0.7:  232,482 ha\n"
    "CWH area: 3,595,194 ha\n"
    "Model: YewMLP  AUC 0.998  F1 0.947"
)
ax.text(0.982, 0.982, stats,
        transform=ax.transAxes, fontsize=8,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                  edgecolor='#aaaaaa', alpha=0.93),
        family='monospace')

# ── 10. North arrow ───────────────────────────────────────────────────────────
ax.annotate('', xy=(0.036, 0.145), xytext=(0.036, 0.09),
            xycoords='axes fraction',
            arrowprops=dict(arrowstyle='->', color='black', lw=2.2))
ax.text(0.036, 0.155, 'N', transform=ax.transAxes,
        ha='center', va='bottom', fontsize=14, fontweight='bold')

# ── 11. Save ──────────────────────────────────────────────────────────────────
plt.tight_layout(rect=[0, 0.12, 1, 1])

out_path = 'results/analysis/cwh_yew_population_100k/cwh_yew_100k_map.png'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='#c8dff0')
size_kb = os.path.getsize(out_path) // 1024
print(f"\n✓ Saved: {out_path}  ({size_kb} KB)")
