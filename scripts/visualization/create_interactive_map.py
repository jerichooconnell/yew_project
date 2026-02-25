#!/usr/bin/env python3
"""
Create Interactive Yew Probability Map with zoom/pan and optional forestry overlay.

Features:
- Zoom (scroll wheel) and pan (click+drag) navigation
- Overlay opacity and probability threshold sliders
- Toggle forestry data overlay showing logged/unsuitable areas
- Cursor position showing lat/lon and probability value
- Statistics sidebar with model performance, threshold table, histogram

Usage:
    python scripts/visualization/create_interactive_map.py \
        --input-dir results/predictions/south_vi_large_forestry
"""

import argparse
import base64
import json
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


def array_to_base64(arr, format='PNG'):
    """Convert numpy array to base64 encoded image string."""
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    buffer = BytesIO()
    img.save(buffer, format=format, optimize=True)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


def create_probability_colormap():
    """Create blue-white-red colormap for probability values."""
    colors = [
        (33, 102, 172),
        (146, 197, 222),
        (247, 247, 247),
        (244, 165, 130),
        (214, 96, 77),
    ]
    cmap = np.zeros((256, 3), dtype=np.uint8)
    positions = [0, 64, 128, 192, 255]
    for i in range(256):
        for j in range(len(positions) - 1):
            if positions[j] <= i <= positions[j + 1]:
                t = (i - positions[j]) / (positions[j + 1] - positions[j])
                c1 = np.array(colors[j])
                c2 = np.array(colors[j + 1])
                cmap[i] = (c1 * (1 - t) + c2 * t).astype(np.uint8)
                break
    return cmap


def apply_colormap(prob_grid, cmap):
    """Apply colormap to probability grid."""
    indices = (np.clip(prob_grid, 0, 1) * 255).astype(np.uint8)
    return cmap[indices.flatten()].reshape(*prob_grid.shape, 3)


def create_forestry_overlay(suitability_grid):
    """Create RGBA overlay showing forestry/land-use categories."""
    h, w = suitability_grid.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)

    # Water/non-vegetated (0.0) → blue
    mask = suitability_grid == 0.0
    overlay[mask] = [30, 100, 220, 180]

    # Recently logged <20yr (0.05) → red
    mask = (suitability_grid > 0) & (suitability_grid <= 0.08)
    overlay[mask] = [220, 50, 50, 170]

    # Non-treed veg / young stands (0.1-0.2) → orange
    mask = (suitability_grid > 0.08) & (suitability_grid <= 0.25)
    overlay[mask] = [230, 120, 30, 150]

    # Logged 40-80yr / medium stand (0.25-0.65) → yellow
    mask = (suitability_grid > 0.25) & (suitability_grid <= 0.65)
    overlay[mask] = [220, 200, 50, 110]

    # Logged >80yr (0.65-0.99) → light green
    mask = (suitability_grid > 0.65) & (suitability_grid < 1.0)
    overlay[mask] = [100, 200, 100, 70]

    return overlay


def get_bbox(metadata):
    """Extract bbox handling both metadata formats."""
    bbox = metadata.get('bbox', {})
    if 'lat_min' in bbox:
        return bbox['lat_min'], bbox['lat_max'], bbox['lon_min'], bbox['lon_max']
    elif 'south' in bbox:
        return bbox['south'], bbox['north'], bbox['west'], bbox['east']
    return 0, 0, 0, 0


def generate_html(rgb_base64, prob_base64, metadata, prob_grid,
                  forestry_base64=None):
    """Generate the interactive HTML map with zoom/pan."""
    h, w = prob_grid.shape
    lat_min, lat_max, lon_min, lon_max = get_bbox(metadata)
    model_info = metadata.get('model', {})
    stats = metadata.get('statistics', {})
    scale = metadata.get('scale_m', 10)
    pixel_area_ha = (scale ** 2) / 10000

    # Downsample prob_grid for JS (hover/threshold)
    max_js = 500
    if h > max_js or w > max_js:
        sf = max_js / max(h, w)
        new_h, new_w = int(h * sf), int(w * sf)
        prob_small = np.zeros((new_h, new_w), dtype=np.float32)
        for i in range(new_h):
            for j in range(new_w):
                prob_small[i, j] = prob_grid[min(int(i / sf), h - 1), min(int(j / sf), w - 1)]
        prob_json = prob_small.tolist()
        dh, dw = new_h, new_w
    else:
        prob_json = prob_grid.tolist()
        dh, dw = h, w

    # Threshold statistics
    prob_flat = prob_grid.flatten()
    thresh_rows = ''
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        cnt = int((prob_flat >= t).sum())
        area = cnt * pixel_area_ha
        pct = 100 * cnt / len(prob_flat)
        thresh_rows += (
            f'<tr><td>{t:.1f}</td><td>{cnt:,}</td>'
            f'<td>{area:,.0f} ha</td><td>{pct:.1f}%</td></tr>\n'
        )

    # Histogram
    hist_counts, hist_edges = np.histogram(prob_flat, bins=50, range=(0, 1))
    max_count = max(hist_counts) if hist_counts.max() > 0 else 1
    hist_bars = ''
    for i, cnt in enumerate(hist_counts):
        pct = 100 * cnt / max_count
        hist_bars += f'<div class="hbar" style="height:{max(pct, 0.5):.1f}%" title="{hist_edges[i]:.2f}-{hist_edges[i+1]:.2f}: {cnt:,}"></div>\n'

    # Percentiles
    percs = {p: float(np.percentile(prob_flat, p)) for p in [10, 25, 50, 75, 90, 95, 99]}
    perc_rows = ''.join(f'<tr><td>P{k}</td><td>{v:.4f}</td></tr>' for k, v in percs.items())

    has_forestry = forestry_base64 is not None
    forestry_canvas_html = '<canvas id="forestryCanvas"></canvas>' if has_forestry else ''
    forestry_btn_html = (
        '<div class="sep"></div>'
        '<button class="tbtn" id="btnForestry" onclick="toggleForestry()" '
        'title="Toggle forestry overlay">🌳 Forestry</button>'
    ) if has_forestry else ''
    forestry_js_init = f'''
        const forestryImg = new Image();
        forestryImg.onload = onImageLoad;
        forestryImg.src = 'data:image/png;base64,{forestry_base64}';
    ''' if has_forestry else '''
        // No forestry data
        imagesNeeded--;
    '''
    forestry_legend_html = '''
    <div class="info-box">
        <h3>Forestry Legend</h3>
        <div class="legend-item"><span class="lswatch" style="background:rgba(30,100,220,0.7)"></span> Water / Non-vegetated</div>
        <div class="legend-item"><span class="lswatch" style="background:rgba(220,50,50,0.67)"></span> Logged &lt;20 yr</div>
        <div class="legend-item"><span class="lswatch" style="background:rgba(230,120,30,0.59)"></span> Logged 20-40 yr / Young</div>
        <div class="legend-item"><span class="lswatch" style="background:rgba(220,200,50,0.43)"></span> Logged 40-80 yr / Medium</div>
        <div class="legend-item"><span class="lswatch" style="background:rgba(100,200,100,0.27)"></span> Logged &gt;80 yr (recovering)</div>
        <div class="legend-item"><span class="lswatch" style="background:transparent;border:1px dashed #555"></span> Old growth / Suitable (1.0)</div>
    </div>
    ''' if has_forestry else ''

    val_acc = model_info.get('validation_accuracy', 0)
    val_f1 = model_info.get('validation_f1', 0)
    val_auc = model_info.get('validation_roc_auc', model_info.get('validation_auc', 0))
    mean_p = stats.get('mean', 0)
    p70 = stats.get('pixels_above_70', 0)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Interactive Yew Probability Map</title>
<style>
*{{ box-sizing:border-box; margin:0; padding:0; }}
body{{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
       background:#1a1a2e; color:#eee; overflow:hidden; height:100vh; }}

/* Toolbar */
.toolbar{{
    position:fixed; top:0; left:0; right:0; height:52px;
    background:#16213e; border-bottom:2px solid #0f3460;
    display:flex; align-items:center; padding:0 12px; gap:8px; z-index:1000;
}}
.toolbar h1{{ font-size:15px; color:#4ecdc4; white-space:nowrap; }}
.sep{{ width:1px; height:28px; background:#0f3460; flex-shrink:0; }}
.tbtn{{
    padding:5px 12px; border:2px solid #555; border-radius:5px;
    background:#1a1a2e; color:#ccc; cursor:pointer; font-size:12px;
    font-weight:600; transition:all .15s; white-space:nowrap;
}}
.tbtn:hover{{ background:#2a2a4e; }}
.tbtn.active{{ border-color:#4ecdc4; color:#4ecdc4; background:#1a2e3e; }}
.tbtn.forestry-on{{ border-color:#66bb6a; color:#66bb6a; background:#1a3e1a; }}
.view-group{{ display:flex; gap:4px; }}
.slbl{{ font-size:11px; color:#999; white-space:nowrap; }}
.sval{{ font-size:11px; color:#ccc; min-width:36px; text-align:right; font-family:monospace; }}
.toolbar input[type="range"]{{ width:90px; accent-color:#4ecdc4; }}
.spacer{{ flex:1; }}
.cursor-box{{
    background:#0f3460; border-radius:5px; padding:4px 10px;
    font-family:monospace; font-size:12px; color:#4ecdc4; white-space:nowrap;
}}

/* Sidebar */
.sidebar{{
    position:fixed; top:52px; right:0; bottom:0; width:320px;
    background:#16213e; overflow-y:auto; padding:16px;
    border-left:2px solid #0f3460; z-index:500;
    transition:transform .3s;
}}
.sidebar.hidden{{ transform:translateX(100%); }}
.info-box{{
    background:#0f3460; border-radius:8px; padding:12px; margin-bottom:14px;
}}
.info-box h3{{
    color:#4ecdc4; margin-bottom:8px; font-size:12px;
    text-transform:uppercase; letter-spacing:1px;
}}
.stats-grid{{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }}
.stat-card{{ background:#1a1a2e; border-radius:6px; padding:8px; text-align:center; }}
.stat-val{{ font-size:16px; font-weight:bold; color:#4ecdc4; }}
.stat-lbl{{ font-size:10px; color:#888; margin-top:2px; }}
.stbl{{ width:100%; border-collapse:collapse; font-size:12px; }}
.stbl th{{ background:#1a1a2e; padding:8px; text-align:left; color:#4ecdc4; font-size:11px; }}
.stbl td{{ padding:6px 8px; border-top:1px solid #1a1a2e; }}
.stbl tr:hover{{ background:#16213e; }}
.histogram{{ display:flex; align-items:flex-end; height:120px; gap:1px; margin-top:8px; }}
.hbar{{
    flex:1; background:linear-gradient(to top,#4ecdc4,#e94560);
    border-radius:1px 1px 0 0; min-width:2px;
}}
.hbar:hover{{ opacity:.7; }}
.legend-item{{ display:flex; align-items:center; gap:8px; margin:4px 0; font-size:12px; }}
.lswatch{{ width:18px; height:14px; border-radius:3px; flex-shrink:0; }}

/* Colorbar at bottom */
.colorbar-wrap{{
    position:fixed; bottom:12px; left:50%; transform:translateX(-50%);
    background:rgba(22,33,62,.92); border:1px solid #0f3460; border-radius:8px;
    padding:6px 14px; z-index:600; display:flex; align-items:center; gap:10px;
}}
.colorbar{{
    width:200px; height:14px; border-radius:4px;
    background:linear-gradient(to right,rgb(33,102,172),rgb(146,197,222),rgb(247,247,247),rgb(244,165,130),rgb(214,96,77));
}}
.cblbl{{ font-size:10px; color:#888; }}

/* Map area */
.map-area{{
    position:fixed; top:52px; left:0; bottom:0;
    overflow:hidden; cursor:grab; background:#111;
}}
.map-area.has-sidebar{{ right:320px; }}
.map-area.no-sidebar{{ right:0; }}
.map-area:active{{ cursor:grabbing; }}
.map-viewport{{ position:absolute; transform-origin:0 0; }}
.map-viewport canvas{{ position:absolute; top:0; left:0; image-rendering:pixelated; }}

/* Threshold indicator */
.thresh-ind{{
    margin-top:8px; padding:8px; background:#1a1a2e; border-radius:6px; text-align:center;
}}
.thresh-ind .cnt{{ font-size:18px; font-weight:bold; color:#e94560; }}
.thresh-ind .desc{{ font-size:10px; color:#888; margin-top:2px; }}
</style>
</head>
<body>

<!-- TOOLBAR -->
<div class="toolbar">
    <h1>🌲 Yew Probability Map</h1>
    <div class="sep"></div>
    <div class="view-group">
        <button class="tbtn active" id="btnOverlay" onclick="setView('overlay')">Overlay</button>
        <button class="tbtn" id="btnRgb" onclick="setView('rgb')">RGB</button>
        <button class="tbtn" id="btnProb" onclick="setView('prob')">Probability</button>
    </div>
    <div class="sep"></div>
    <span class="slbl">Opacity:</span>
    <input type="range" id="alphaSlider" min="0" max="100" value="50"
           oninput="setAlpha(this.value)">
    <span class="sval" id="alphaVal">50%</span>
    <div class="sep"></div>
    <span class="slbl">Threshold:</span>
    <input type="range" id="threshSlider" min="0" max="100" value="0"
           oninput="setThreshold(this.value)">
    <span class="sval" id="threshVal">0.00</span>
    {forestry_btn_html}
    <div class="sep"></div>
    <button class="tbtn active" id="btnSidebar" onclick="toggleSidebar()">📊</button>
    <div class="spacer"></div>
    <div class="cursor-box" id="cursorInfo">Hover over map</div>
</div>

<!-- SIDEBAR -->
<div class="sidebar" id="sidebar">
    <div class="info-box">
        <h3>Model Performance</h3>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-val">{val_acc*100:.1f}%</div>
                <div class="stat-lbl">Val Accuracy</div>
            </div>
            <div class="stat-card">
                <div class="stat-val">{val_f1*100:.1f}%</div>
                <div class="stat-lbl">F1 Score</div>
            </div>
            <div class="stat-card">
                <div class="stat-val">{mean_p:.3f}</div>
                <div class="stat-lbl">Mean Prob</div>
            </div>
            <div class="stat-card">
                <div class="stat-val">{p70:,}</div>
                <div class="stat-lbl">P &gt; 0.7</div>
            </div>
        </div>
    </div>

    <div class="info-box">
        <h3>Detection by Threshold</h3>
        <table class="stbl">
            <thead><tr><th>Threshold</th><th>Pixels</th><th>Area</th><th>%</th></tr></thead>
            <tbody>{thresh_rows}</tbody>
        </table>
        <div class="thresh-ind">
            <div class="cnt" id="pixelCount">-</div>
            <div class="desc">pixels above current threshold</div>
        </div>
    </div>

    <div class="info-box">
        <h3>Probability Distribution</h3>
        <div class="histogram">{hist_bars}</div>
    </div>

    <div class="info-box">
        <h3>Percentiles</h3>
        <table class="stbl">
            <thead><tr><th>Percentile</th><th>Value</th></tr></thead>
            <tbody>{perc_rows}</tbody>
        </table>
    </div>

    {forestry_legend_html}

    <div class="info-box">
        <h3>Coverage</h3>
        <div style="font-size:12px; line-height:1.6;">
            {lat_min:.4f}°N to {lat_max:.4f}°N<br>
            {abs(lon_max):.4f}°W to {abs(lon_min):.4f}°W<br>
            {w:,} × {h:,} pixels ({w*h:,} total)<br>
            ~{w*h*pixel_area_ha:,.0f} ha
        </div>
    </div>
</div>

<!-- MAP -->
<div class="map-area has-sidebar" id="mapArea">
    <div class="map-viewport" id="viewport">
        <canvas id="rgbCanvas"></canvas>
        <canvas id="probCanvas"></canvas>
        {forestry_canvas_html}
    </div>
</div>

<!-- COLORBAR -->
<div class="colorbar-wrap">
    <span class="cblbl">0.0 Low</span>
    <div class="colorbar"></div>
    <span class="cblbl">1.0 High</span>
</div>

<script>
// ============================================================
// CONSTANTS
// ============================================================
const IMG_W = {w};
const IMG_H = {h};
const DW = {dw};
const DH = {dh};
const LAT_MIN = {lat_min};
const LAT_MAX = {lat_max};
const LON_MIN = {lon_min};
const LON_MAX = {lon_max};
const ORIG_PIXELS = {w * h};
const DISPLAY_PIXELS = {dw * dh};
const HAS_FORESTRY = {'true' if has_forestry else 'false'};

const probData = {json.dumps(prob_json)};

// ============================================================
// STATE
// ============================================================
let vpX = 0, vpY = 0, vpScale = 1;
let isPanning = false, panSX = 0, panSY = 0, panVX = 0, panVY = 0;
let currentView = 'overlay';
let alpha = 0.5;
let threshold = 0;
let showForestry = false;
let showSidebar = true;

// ============================================================
// DOM REFS
// ============================================================
const mapArea = document.getElementById('mapArea');
const viewport = document.getElementById('viewport');
const rgbCanvas = document.getElementById('rgbCanvas');
const probCanvas = document.getElementById('probCanvas');
const forestryCanvas = HAS_FORESTRY ? document.getElementById('forestryCanvas') : null;
const rgbCtx = rgbCanvas.getContext('2d');
const probCtx = probCanvas.getContext('2d');

// ============================================================
// IMAGE LOADING
// ============================================================
let imagesNeeded = HAS_FORESTRY ? 3 : 2;
let imagesLoaded = 0;

const rgbImg = new Image();
const probImg = new Image();

function onImageLoad() {{
    imagesLoaded++;
    if (imagesLoaded >= imagesNeeded) initMap();
}}

rgbImg.onload = onImageLoad;
probImg.onload = onImageLoad;
rgbImg.src = 'data:image/png;base64,{rgb_base64}';
probImg.src = 'data:image/png;base64,{prob_base64}';

{forestry_js_init}

// ============================================================
// INITIALIZATION
// ============================================================
function initMap() {{
    // Set canvas sizes
    rgbCanvas.width = IMG_W; rgbCanvas.height = IMG_H;
    probCanvas.width = IMG_W; probCanvas.height = IMG_H;
    if (HAS_FORESTRY && forestryCanvas) {{
        forestryCanvas.width = IMG_W; forestryCanvas.height = IMG_H;
        const fCtx = forestryCanvas.getContext('2d');
        fCtx.drawImage(forestryImg, 0, 0, IMG_W, IMG_H);
        forestryCanvas.style.display = 'none';
    }}

    // Draw base layers
    rgbCtx.drawImage(rgbImg, 0, 0, IMG_W, IMG_H);
    drawProbLayer();

    // Fit to screen
    const cw = mapArea.clientWidth;
    const ch = mapArea.clientHeight;
    vpScale = Math.min(cw / IMG_W, ch / IMG_H) * 0.95;
    vpX = (cw - IMG_W * vpScale) / 2;
    vpY = (ch - IMG_H * vpScale) / 2;
    updateViewport();
    applyView();
    updatePixelCount();
}}

function drawProbLayer() {{
    probCtx.clearRect(0, 0, IMG_W, IMG_H);
    probCtx.drawImage(probImg, 0, 0, IMG_W, IMG_H);

    if (threshold > 0) {{
        const imgData = probCtx.getImageData(0, 0, IMG_W, IMG_H);
        const d = imgData.data;
        const sx = DW / IMG_W, sy = DH / IMG_H;
        for (let y = 0; y < IMG_H; y++) {{
            const gi = Math.min(Math.floor(y * sy), DH - 1);
            const row = probData[gi];
            for (let x = 0; x < IMG_W; x++) {{
                const gj = Math.min(Math.floor(x * sx), DW - 1);
                if (row[gj] < threshold) {{
                    d[(y * IMG_W + x) * 4 + 3] = 0;
                }}
            }}
        }}
        probCtx.putImageData(imgData, 0, 0);
    }}
}}

function updateViewport() {{
    viewport.style.transform = `translate(${{vpX}}px,${{vpY}}px) scale(${{vpScale}})`;
}}

// ============================================================
// VIEW MODES
// ============================================================
function applyView() {{
    const v = currentView;
    rgbCanvas.style.display = (v === 'prob') ? 'none' : '';
    probCanvas.style.display = (v === 'rgb') ? 'none' : '';
    probCanvas.style.opacity = (v === 'prob') ? 1.0 : alpha;
}}

function setView(v) {{
    currentView = v;
    document.querySelectorAll('.view-group .tbtn').forEach(b => b.classList.remove('active'));
    document.getElementById('btn' + v.charAt(0).toUpperCase() + v.slice(1)).classList.add('active');
    applyView();
}}

function setAlpha(val) {{
    alpha = val / 100;
    document.getElementById('alphaVal').textContent = val + '%';
    if (currentView === 'overlay') probCanvas.style.opacity = alpha;
}}

function setThreshold(val) {{
    threshold = val / 100;
    document.getElementById('threshVal').textContent = threshold.toFixed(2);
    drawProbLayer();
    updatePixelCount();
}}

function updatePixelCount() {{
    let cnt = 0;
    for (let y = 0; y < DH; y++)
        for (let x = 0; x < DW; x++)
            if (probData[y][x] >= threshold) cnt++;
    const est = Math.round(cnt * ORIG_PIXELS / DISPLAY_PIXELS);
    const el = document.getElementById('pixelCount');
    if (el) el.textContent = est.toLocaleString();
}}

function toggleForestry() {{
    if (!HAS_FORESTRY || !forestryCanvas) return;
    showForestry = !showForestry;
    forestryCanvas.style.display = showForestry ? '' : 'none';
    document.getElementById('btnForestry').classList.toggle('forestry-on', showForestry);
    document.getElementById('btnForestry').classList.toggle('active', false);
}}

function toggleSidebar() {{
    showSidebar = !showSidebar;
    document.getElementById('sidebar').classList.toggle('hidden', !showSidebar);
    mapArea.classList.toggle('has-sidebar', showSidebar);
    mapArea.classList.toggle('no-sidebar', !showSidebar);
    document.getElementById('btnSidebar').classList.toggle('active', showSidebar);
    // Re-fit on toggle (optional: keep current zoom)
}}

// ============================================================
// ZOOM / PAN
// ============================================================
mapArea.addEventListener('wheel', (e) => {{
    e.preventDefault();
    const rect = mapArea.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    const ns = vpScale * factor;
    vpX = mx - (mx - vpX) * (ns / vpScale);
    vpY = my - (my - vpY) * (ns / vpScale);
    vpScale = ns;
    updateViewport();
}}, {{ passive: false }});

mapArea.addEventListener('mousedown', (e) => {{
    if (e.button === 0) {{
        isPanning = true;
        panSX = e.clientX; panSY = e.clientY;
        panVX = vpX; panVY = vpY;
    }}
}});

window.addEventListener('mousemove', (e) => {{
    if (isPanning) {{
        vpX = panVX + (e.clientX - panSX);
        vpY = panVY + (e.clientY - panSY);
        updateViewport();
    }}
    // Cursor info
    updateCursor(e);
}});

window.addEventListener('mouseup', () => {{ isPanning = false; }});

// Double-click to reset view
mapArea.addEventListener('dblclick', () => {{
    const cw = mapArea.clientWidth;
    const ch = mapArea.clientHeight;
    vpScale = Math.min(cw / IMG_W, ch / IMG_H) * 0.95;
    vpX = (cw - IMG_W * vpScale) / 2;
    vpY = (ch - IMG_H * vpScale) / 2;
    updateViewport();
}});

// ============================================================
// CURSOR INFO
// ============================================================
function screenToImage(sx, sy) {{
    const rect = mapArea.getBoundingClientRect();
    const cx = sx - rect.left;
    const cy = sy - rect.top;
    return {{ ix: (cx - vpX) / vpScale, iy: (cy - vpY) / vpScale }};
}}

function updateCursor(e) {{
    const {{ ix, iy }} = screenToImage(e.clientX, e.clientY);
    if (ix < 0 || ix >= IMG_W || iy < 0 || iy >= IMG_H) {{
        document.getElementById('cursorInfo').textContent = 'Hover over map';
        return;
    }}
    const lon = LON_MIN + (ix / IMG_W) * (LON_MAX - LON_MIN);
    const lat = LAT_MAX - (iy / IMG_H) * (LAT_MAX - LAT_MIN);
    const gi = Math.min(Math.floor(iy / IMG_H * DH), DH - 1);
    const gj = Math.min(Math.floor(ix / IMG_W * DW), DW - 1);
    const prob = (gi >= 0 && gj >= 0) ? probData[gi][gj] : null;
    const pStr = prob !== null ? prob.toFixed(3) : '--';
    document.getElementById('cursorInfo').textContent =
        `${{lat.toFixed(5)}}°N, ${{Math.abs(lon).toFixed(5)}}°W  |  P = ${{pStr}}`;
}}

// Keyboard: F = toggle forestry, S = toggle sidebar
document.addEventListener('keydown', (e) => {{
    if (e.key === 'f' || e.key === 'F') toggleForestry();
    if (e.key === 's' || e.key === 'S') toggleSidebar();
}});
</script>
</body>
</html>'''

    return html


def main():
    parser = argparse.ArgumentParser(description='Create interactive yew probability map')
    parser.add_argument('--input-dir', type=str, default='results/predictions/large_area',
                        help='Directory with prob_grid.npy, rgb_image.npy, metadata.json')
    parser.add_argument('--output', type=str, default=None,
                        help='Output HTML file path')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    # Load prediction data
    print("Loading data...")
    prob_grid = np.load(input_dir / 'prob_grid.npy')
    print(f"  Probability grid: {prob_grid.shape}")

    rgb_path = input_dir / 'rgb_image.npy'
    if rgb_path.exists():
        rgb_image = np.load(rgb_path)
        print(f"  RGB image: {rgb_image.shape}")
    else:
        print("  No RGB image found, creating grayscale placeholder")
        rgb_image = np.stack([prob_grid] * 3, axis=-1)

    metadata_path = input_dir / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print("  Metadata loaded")
    else:
        metadata = {}

    # Encode images
    print("Encoding images...")
    rgb_base64 = array_to_base64(rgb_image)
    cmap = create_probability_colormap()
    prob_rgb = apply_colormap(prob_grid, cmap)
    prob_base64 = array_to_base64(prob_rgb)

    # Load forestry data if available
    forestry_base64 = None
    suit_path = input_dir / 'suitability_grid.npy'
    if suit_path.exists():
        print("  Loading forestry suitability data...")
        suit_grid = np.load(suit_path)
        overlay = create_forestry_overlay(suit_grid)
        forestry_base64 = array_to_base64(overlay)
        print(f"  Forestry overlay encoded ({suit_grid.shape})")

    # Generate HTML
    output_path = Path(args.output) if args.output else input_dir / 'interactive_map.html'
    print("Generating interactive HTML...")
    html = generate_html(rgb_base64, prob_base64, metadata, prob_grid,
                         forestry_base64=forestry_base64)

    with open(output_path, 'w') as f:
        f.write(html)

    print(f"✓ Created interactive map: {output_path}")
    print(f"\nOpen in browser: file://{output_path.absolute()}")


if __name__ == '__main__':
    main()
