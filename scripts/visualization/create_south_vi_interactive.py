#!/usr/bin/env python3
"""
Create interactive HTML map for South Vancouver Island yew predictions.
Uses folium ImageOverlay to display the full probability raster, plus
training data points and basemap toggle.
"""

import io
import base64
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image as PILImage
import folium
from folium import plugins
from matplotlib.colors import LinearSegmentedColormap


# ── Colormap ─────────────────────────────────────────────────────────────────
YEWCMAP = LinearSegmentedColormap.from_list(
    'yew_visible',
    [
        (0.00, (0.90, 0.90, 0.90, 0.00)),
        (0.05, (0.60, 0.80, 0.60, 0.55)),
        (0.10, (0.20, 0.70, 0.20, 0.70)),
        (0.30, (0.45, 0.85, 0.05, 0.80)),
        (0.50, (1.00, 0.90, 0.00, 0.88)),
        (0.70, (0.90, 0.40, 0.10, 0.93)),
        (1.00, (0.65, 0.00, 0.45, 0.96)),
    ],
    N=256,
)


def grid_to_png_bytes(grid, cmap, downsample=4, threshold=0.02):
    """Convert probability grid to PNG bytes via colormap."""
    # Downsample
    h, w = grid.shape
    new_h, new_w = h // downsample, w // downsample
    small = grid[::downsample, ::downsample]  # fast stride downsample
    small = small[:new_h, :new_w]

    # Apply colormap → RGBA float [0,1]
    rgba = cmap(small)  # shape (H, W, 4)

    # Zero-out near-zero probs fully
    mask = small < threshold
    rgba[mask] = [0, 0, 0, 0]

    # Convert to uint8
    rgba_u8 = (rgba * 255).astype(np.uint8)

    img = PILImage.fromarray(rgba_u8, mode='RGBA')
    buf = io.BytesIO()
    img.save(buf, format='PNG', compress_level=6)
    return buf.getvalue()


def prob_to_color(prob):
    """Scalar probability → hex colour string."""
    r, g, b, _ = YEWCMAP(prob)
    return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'


def add_training_layers(m, base_dir='.'):
    """Add iNat yew positives and FAIB negatives as toggleable layers."""
    base_dir = Path(base_dir)

    # ── Positives ──────────────────────────────────────────────────────────
    fg_pos = folium.FeatureGroup(name='Training: Yew positives (iNat)', show=False)

    for csv_path, label, colour in [
        (base_dir / 'data/processed/inat_yew_positives_train.csv', 'Train positive', '#00dd00'),
        (base_dir / 'data/processed/inat_yew_positives_val.csv',   'Val positive',   '#66ff66'),
    ]:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=5,
                    color=colour,
                    fill=True, fillColor=colour, fillOpacity=0.75,
                    opacity=0.9,
                    popup=f'{label}<br>Lat:{row["lat"]:.4f}<br>Lon:{row["lon"]:.4f}',
                    tooltip=label,
                ).add_to(fg_pos)

    fg_pos.add_to(m)

    # ── Negatives ──────────────────────────────────────────────────────────
    neg_path = base_dir / 'data/processed/faib_negatives/faib_negative_embeddings.csv'
    if neg_path.exists():
        df_neg = pd.read_csv(neg_path)
        # Sample to keep page snappy; prioritise sites inside south VI bbox
        bbox = dict(south=48.27, north=48.72, west=-124.55, east=-123.15)
        in_bbox = (
            df_neg['lat'].between(bbox['south'], bbox['north']) &
            df_neg['lon'].between(bbox['west'], bbox['east'])
        )
        df_in   = df_neg[in_bbox]
        df_rest = df_neg[~in_bbox].sample(n=min(200, (~in_bbox).sum()), random_state=42)
        df_show = pd.concat([df_in, df_rest])

        fg_neg = folium.FeatureGroup(name=f'Training: Non-yew FAIB ({len(df_show):,} shown)', show=False)
        for _, row in df_show.iterrows():
            bec = row.get('bec_zone', '?')
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=4,
                color='#cc0000',
                fill=True, fillColor='#cc0000', fillOpacity=0.45,
                opacity=0.65,
                popup=f'FAIB non-yew<br>BEC: {bec}<br>Lat:{row["lat"]:.4f}<br>Lon:{row["lon"]:.4f}',
                tooltip=f'FAIB: {bec}',
            ).add_to(fg_neg)
        fg_neg.add_to(m)


def create_south_vi_map(output_html='results/analysis/south_vi_interactive.html',
                        downsample=4):
    pred_dir  = Path('results/predictions/south_vi_large')
    output    = Path(output_html)
    output.parent.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    print('Loading probability grid…')
    grid = np.load(pred_dir / 'prob_grid.npy')
    print(f'  Grid: {grid.shape}  mean={grid.mean():.3f}  max={grid.max():.3f}')

    with open(pred_dir / 'metadata.json') as f:
        meta = json.load(f)
    bbox = meta['bbox']
    south, north, west, east = bbox['south'], bbox['north'], bbox['west'], bbox['east']
    print(f'  Bbox: {south}–{north}°N, {west}–{east}°E')

    # ── Build PNG overlay ──────────────────────────────────────────────────
    print(f'Rendering PNG (1/{downsample} scale)…')
    png_bytes = grid_to_png_bytes(grid, YEWCMAP, downsample=downsample)
    png_b64   = base64.b64encode(png_bytes).decode()
    print(f'  PNG size: {len(png_bytes)/1024:.0f} KB')

    # ── Base map ───────────────────────────────────────────────────────────
    center = [(north + south) / 2, (west + east) / 2]
    m = folium.Map(location=center, zoom_start=10, control_scale=True,
                   tiles='OpenStreetMap')

    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='Satellite', overlay=False, control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='Terrain', overlay=False, control=True,
    ).add_to(m)

    # ── Probability raster overlay ─────────────────────────────────────────
    img_url = f'data:image/png;base64,{png_b64}'
    folium.raster_layers.ImageOverlay(
        image=img_url,
        bounds=[[south, west], [north, east]],
        origin='upper',
        opacity=1.0,
        name='Yew Probability Raster',
        show=True,
        interactive=False,
        zindex=1,
    ).add_to(m)

    # ── Training data ──────────────────────────────────────────────────────
    print('Adding training data…')
    add_training_layers(m, base_dir='.')

    # ── Layer control ──────────────────────────────────────────────────────
    folium.LayerControl(collapsed=False).add_to(m)

    # ── Legend ────────────────────────────────────────────────────────────
    model_stats = meta.get('model', {})
    acc  = model_stats.get('validation_accuracy', 0) * 100
    f1   = model_stats.get('validation_f1', 0)
    auc  = model_stats.get('validation_roc_auc', 0)
    n_tr = model_stats.get('training_samples', 0)
    p50_ha = meta.get('statistics', {}).get('pixels_above_50', 0) * 100 / 1e4  # 10m px → ha

    legend_html = f'''
    <div style="position:fixed;bottom:30px;right:10px;width:230px;
                background:#ffffffee;border:2px solid #555;z-index:9999;
                font-size:13px;padding:12px;border-radius:6px;">
      <b>Yew Probability</b>
      <div style="margin:6px 0 2px">
        <span style="background:#99cc99;padding:1px 10px;">▇</span> 0.05 – 0.10
      </div>
      <div style="margin:2px 0">
        <span style="background:#33b233;padding:1px 10px;">▇</span> 0.10 – 0.30
      </div>
      <div style="margin:2px 0">
        <span style="background:#73d90d;padding:1px 10px;">▇</span> 0.30 – 0.50
      </div>
      <div style="margin:2px 0">
        <span style="background:#ffe600;padding:1px 10px;">▇</span> 0.50 – 0.70
      </div>
      <div style="margin:2px 0">
        <span style="background:#e66619;padding:1px 10px;">▇</span> 0.70 – 1.00
      </div>
      <b style="display:block;margin-top:8px;">Training Data</b>
      <div style="margin:4px 0 2px">
        <span style="color:#00dd00;font-size:16px;">●</span> Yew observations
      </div>
      <div style="margin:2px 0">
        <span style="color:#cc0000;font-size:16px;">●</span> FAIB non-yew sites
      </div>
      <hr style="margin:8px 0">
      <span style="font-size:11px;color:#555;">
        Model: acc={acc:.1f}% F1={f1:.3f} AUC={auc:.3f}<br>
        Training samples: {n_tr:,}<br>
        P≥0.5 area: ~{p50_ha:,.0f} ha ({p50_ha/1e6*1e4:.1f} km²)
      </span>
    </div>'''
    m.get_root().html.add_child(folium.Element(legend_html))

    # ── Title ─────────────────────────────────────────────────────────────
    title_html = '''
    <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
                width:520px;background:#ffffffee;border:2px solid #555;
                z-index:9999;font-size:15px;padding:10px;text-align:center;
                border-radius:6px;">
      <b>Pacific Yew Habitat Probability — South Vancouver Island</b><br>
      <span style="font-size:11px;color:#555;">
        FAIB-negatives model · 10 m GEE embeddings · 2024
      </span>
    </div>'''
    m.get_root().html.add_child(folium.Element(title_html))

    # ── Save ──────────────────────────────────────────────────────────────
    print(f'Saving to {output}…')
    m.save(str(output))
    size_mb = output.stat().st_size / 1024 / 1024
    print(f'✓ Done — {size_mb:.1f} MB')
    print(f'Open: file://{output.absolute()}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/analysis/south_vi_interactive.html')
    parser.add_argument('--downsample', type=int, default=4,
                        help='Pixel downsample factor (4 = 1/16th area, good for web)')
    args = parser.parse_args()
    create_south_vi_map(output_html=args.output, downsample=args.downsample)
