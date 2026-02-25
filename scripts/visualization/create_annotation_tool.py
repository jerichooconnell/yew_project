#!/usr/bin/env python3
"""
Create an interactive annotation tool for yew probability maps.

Features:
- Zoom (scroll wheel) and pan (click+drag) navigation
- Click to mark points as YEW (positive) or NOT YEW (negative)
- Toggle forestry overlay showing logged/unsuitable areas
- Export annotations as CSV for retraining
- Auto-save to localStorage

Usage:
    python scripts/visualization/create_annotation_tool.py \
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
    """Create blue-white-red colormap."""
    colors = [
        (33, 102, 172), (146, 197, 222), (247, 247, 247),
        (244, 165, 130), (214, 96, 77),
    ]
    positions = [0, 64, 128, 192, 255]
    cmap = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        for j in range(len(positions) - 1):
            if positions[j] <= i <= positions[j + 1]:
                t = (i - positions[j]) / (positions[j + 1] - positions[j])
                c1 = np.array(colors[j])
                c2 = np.array(colors[j + 1])
                cmap[i] = (c1 * (1 - t) + c2 * t).astype(np.uint8)
                break
    return cmap


def create_forestry_overlay(suitability_grid):
    """Create RGBA overlay showing forestry/land-use categories."""
    h, w = suitability_grid.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    mask = suitability_grid == 0.0
    overlay[mask] = [30, 100, 220, 180]
    mask = (suitability_grid > 0) & (suitability_grid <= 0.08)
    overlay[mask] = [220, 50, 50, 170]
    mask = (suitability_grid > 0.08) & (suitability_grid <= 0.25)
    overlay[mask] = [230, 120, 30, 150]
    mask = (suitability_grid > 0.25) & (suitability_grid <= 0.65)
    overlay[mask] = [220, 200, 50, 110]
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


def generate_annotation_html(rgb_base64, prob_base64, prob_grid, metadata,
                             output_path, forestry_base64=None):
    """Generate the annotation tool HTML."""
    h, w = prob_grid.shape
    lat_min, lat_max, lon_min, lon_max = get_bbox(metadata)

    # Downsample prob_grid for JS hover values
    max_size = 800
    if h > max_size or w > max_size:
        sf = max_size / max(h, w)
        new_h, new_w = int(h * sf), int(w * sf)
        prob_small = np.zeros((new_h, new_w), dtype=np.float32)
        for i in range(new_h):
            for j in range(new_w):
                prob_small[i, j] = prob_grid[min(int(i / sf), h - 1), min(int(j / sf), w - 1)]
        prob_json = [[round(float(v), 4) for v in row] for row in prob_small]
        display_h, display_w = new_h, new_w
    else:
        prob_json = [[round(float(v), 4) for v in row] for row in prob_grid]
        display_h, display_w = h, w

    has_forestry = forestry_base64 is not None
    forestry_canvas_html = '<canvas id="forestryCanvas"></canvas>' if has_forestry else ''
    forestry_btn_html = (
        '<div class="sep"></div>'
        '<button class="tool-btn forestry-btn" id="btnForestry" '
        'onclick="toggleForestry()" title="Toggle forestry overlay (F)">🌳 Forestry</button>'
    ) if has_forestry else ''
    forestry_js_load = f'''
        const forestryImg = new Image();
        forestryImg.onload = onImageLoad;
        forestryImg.src = 'data:image/png;base64,{forestry_base64}';
    ''' if has_forestry else '''
        imagesNeeded--;
    '''

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Yew Annotation Tool</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
            overflow: hidden;
        }}

        .toolbar {{
            position: fixed;
            top: 0; left: 0; right: 0;
            height: 56px;
            background: #16213e;
            border-bottom: 2px solid #0f3460;
            display: flex;
            align-items: center;
            padding: 0 16px;
            gap: 10px;
            z-index: 1000;
        }}
        .toolbar h1 {{
            font-size: 16px;
            color: #4ecdc4;
            white-space: nowrap;
        }}
        .sep {{
            width: 1px;
            height: 30px;
            background: #0f3460;
        }}
        .tool-group {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .tool-btn {{
            padding: 6px 14px;
            border: 2px solid #555;
            border-radius: 6px;
            background: #1a1a2e;
            color: #ccc;
            cursor: pointer;
            font-size: 13px;
            font-weight: 600;
            transition: all 0.15s;
            white-space: nowrap;
        }}
        .tool-btn:hover {{ background: #2a2a4e; }}
        .tool-btn.active {{ border-color: #4ecdc4; color: #4ecdc4; background: #1a2e3e; }}
        .tool-btn.yew-btn.active {{ border-color: #4CAF50; color: #4CAF50; background: #1a3e1a; }}
        .tool-btn.notyew-btn.active {{ border-color: #f44336; color: #f44336; background: #3e1a1a; }}
        .tool-btn.pan-btn.active {{ border-color: #2196F3; color: #2196F3; background: #1a1a3e; }}
        .tool-btn.forestry-btn.forestry-on {{ border-color: #66bb6a; color: #66bb6a; background: #1a3e1a; }}

        .tool-btn.action {{
            background: #0f3460;
            border-color: #0f3460;
            color: #ccc;
        }}
        .tool-btn.action:hover {{ background: #1a4a80; }}
        .tool-btn.export {{ border-color: #4ecdc4; color: #4ecdc4; }}
        .tool-btn.export:hover {{ background: #1a3e3e; }}
        .tool-btn.danger {{ border-color: #f44336; color: #f44336; }}
        .tool-btn.danger:hover {{ background: #3e1a1a; }}

        .slider-group {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .slider-group label {{
            font-size: 12px;
            color: #999;
            white-space: nowrap;
        }}
        .slider-group input[type="range"] {{
            width: 100px;
            accent-color: #4ecdc4;
        }}

        .count-badge {{
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        }}
        .count-yew {{ background: #1a3e1a; color: #4CAF50; border: 1px solid #4CAF50; }}
        .count-notyew {{ background: #3e1a1a; color: #f44336; border: 1px solid #f44336; }}

        .map-container {{
            position: fixed;
            top: 56px; left: 0; right: 0; bottom: 0;
            overflow: hidden;
            cursor: crosshair;
        }}
        .map-container.panning {{
            cursor: grab;
        }}
        .map-container.panning:active {{
            cursor: grabbing;
        }}

        .map-viewport {{
            position: absolute;
            transform-origin: 0 0;
        }}
        .map-viewport canvas {{
            position: absolute;
            top: 0; left: 0;
            image-rendering: pixelated;
        }}

        .annotation-marker {{
            position: absolute;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            border: 2px solid white;
            transform: translate(-50%, -50%);
            pointer-events: all;
            cursor: pointer;
            z-index: 100;
            box-shadow: 0 0 4px rgba(0,0,0,0.8);
            transition: transform 0.1s;
        }}
        .annotation-marker:hover {{
            transform: translate(-50%, -50%) scale(1.4);
            z-index: 200;
        }}
        .annotation-marker.yew {{ background: rgba(76, 175, 80, 0.9); }}
        .annotation-marker.notyew {{ background: rgba(244, 67, 54, 0.9); }}

        .info-panel {{
            position: fixed;
            bottom: 16px; left: 16px;
            background: rgba(22, 33, 62, 0.95);
            border: 1px solid #0f3460;
            border-radius: 8px;
            padding: 12px;
            font-size: 12px;
            z-index: 500;
            min-width: 200px;
        }}
        .info-panel .coord {{
            color: #4ecdc4;
            font-family: monospace;
            font-size: 13px;
        }}
        .info-panel .prob {{
            font-family: monospace;
            font-size: 13px;
            margin-top: 4px;
        }}

        .help-panel {{
            position: fixed;
            bottom: 16px; right: 16px;
            background: rgba(22, 33, 62, 0.95);
            border: 1px solid #0f3460;
            border-radius: 8px;
            padding: 12px;
            font-size: 11px;
            z-index: 500;
            line-height: 1.8;
            color: #aaa;
        }}
        .help-panel kbd {{
            background: #2a2a4e;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 1px 5px;
            font-size: 11px;
            color: #ddd;
        }}

        .forestry-legend {{
            position: fixed;
            bottom: 16px; left: 240px;
            background: rgba(22, 33, 62, 0.95);
            border: 1px solid #0f3460;
            border-radius: 8px;
            padding: 10px 14px;
            font-size: 11px;
            z-index: 500;
            display: none;
        }}
        .forestry-legend.visible {{ display: block; }}
        .forestry-legend h4 {{ color: #4ecdc4; margin-bottom: 6px; font-size: 11px; }}
        .fleg-item {{ display: flex; align-items: center; gap: 6px; margin: 2px 0; }}
        .fleg-sw {{ width: 14px; height: 10px; border-radius: 2px; flex-shrink: 0; }}

        .toast {{
            position: fixed;
            top: 70px; left: 50%;
            transform: translateX(-50%);
            background: #16213e;
            border: 1px solid #4ecdc4;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 14px;
            z-index: 2000;
            opacity: 0;
            transition: opacity 0.3s;
            pointer-events: none;
        }}
        .toast.show {{ opacity: 1; }}
    </style>
</head>
<body>

<div class="toolbar">
    <h1>🌲 Yew Annotation Tool</h1>
    <div class="sep"></div>

    <div class="tool-group">
        <button class="tool-btn pan-btn active" id="btnPan" onclick="setMode('pan')" title="Pan mode (Space)">
            ✋ Pan
        </button>
        <button class="tool-btn yew-btn" id="btnYew" onclick="setMode('yew')" title="Mark as Yew (1)">
            🌲 Yew
        </button>
        <button class="tool-btn notyew-btn" id="btnNotYew" onclick="setMode('notyew')" title="Mark as Not Yew (2)">
            ❌ Not Yew
        </button>
    </div>

    <div class="sep"></div>

    <div class="slider-group">
        <label>Overlay:</label>
        <input type="range" id="overlaySlider" min="0" max="100" value="50" oninput="updateOverlay(this.value)">
    </div>
    <div class="slider-group">
        <label>Threshold:</label>
        <input type="range" id="thresholdSlider" min="0" max="100" value="0" oninput="updateThreshold(this.value)">
        <span id="threshVal" style="font-size:11px; color:#999;">0.00</span>
    </div>

    {forestry_btn_html}

    <div class="sep"></div>

    <span class="count-badge count-yew" id="yewCount">🌲 0</span>
    <span class="count-badge count-notyew" id="notyewCount">❌ 0</span>

    <div class="sep"></div>

    <button class="tool-btn action" onclick="undoLast()" title="Undo (Ctrl+Z)">↩ Undo</button>
    <button class="tool-btn export" onclick="exportCSV()" title="Export (Ctrl+S)">📥 Export CSV</button>
    <button class="tool-btn action" onclick="importCSV()" title="Import previous annotations">📤 Import</button>
    <button class="tool-btn danger" onclick="clearAll()" title="Clear all annotations">🗑 Clear</button>
</div>

<div class="map-container" id="mapContainer">
    <div class="map-viewport" id="mapViewport">
        <canvas id="rgbCanvas"></canvas>
        <canvas id="overlayCanvas"></canvas>
        {forestry_canvas_html}
        <div id="markerLayer"></div>
    </div>
</div>

<div class="info-panel" id="infoPanel">
    <div>Cursor Position:</div>
    <div class="coord" id="coordDisplay">--</div>
    <div class="prob" id="probDisplay">P(yew): --</div>
    <div class="prob" id="pixelDisplay">Pixel: --</div>
</div>

<div class="help-panel">
    <div><kbd>1</kbd> Yew mode &nbsp; <kbd>2</kbd> Not Yew mode &nbsp; <kbd>Space</kbd> Pan mode</div>
    <div><kbd>Click</kbd> Place marker &nbsp; <kbd>Right-click</kbd> on marker to delete</div>
    <div><kbd>Scroll</kbd> Zoom &nbsp; <kbd>Ctrl+Z</kbd> Undo &nbsp; <kbd>Ctrl+S</kbd> Export</div>
    <div><kbd>F</kbd> Toggle forestry &nbsp; <kbd>Dbl-click</kbd> Reset view</div>
</div>

<div class="forestry-legend" id="forestryLegend">
    <h4>🌳 Forestry Overlay</h4>
    <div class="fleg-item"><span class="fleg-sw" style="background:rgba(30,100,220,0.7)"></span> Water / Non-vegetated</div>
    <div class="fleg-item"><span class="fleg-sw" style="background:rgba(220,50,50,0.67)"></span> Logged &lt;20 yr</div>
    <div class="fleg-item"><span class="fleg-sw" style="background:rgba(230,120,30,0.59)"></span> Logged 20-40 yr / Young</div>
    <div class="fleg-item"><span class="fleg-sw" style="background:rgba(220,200,50,0.43)"></span> Logged 40-80 yr / Medium</div>
    <div class="fleg-item"><span class="fleg-sw" style="background:rgba(100,200,100,0.27)"></span> Logged &gt;80 yr</div>
</div>

<div class="toast" id="toast"></div>

<input type="file" id="fileInput" accept=".csv" style="display:none" onchange="handleImport(event)">

<script>
// ============================================================
// DATA
// ============================================================
const IMG_W = {w};
const IMG_H = {h};
const DISPLAY_W = {display_w};
const DISPLAY_H = {display_h};
const LAT_MIN = {lat_min};
const LAT_MAX = {lat_max};
const LON_MIN = {lon_min};
const LON_MAX = {lon_max};
const HAS_FORESTRY = {'true' if has_forestry else 'false'};

const probGrid = {json.dumps(prob_json)};

// ============================================================
// STATE
// ============================================================
let mode = 'pan';
let annotations = [];
let nextId = 1;

let vpX = 0, vpY = 0, vpScale = 1;
let isPanning = false;
let panStartX = 0, panStartY = 0;
let panStartVpX = 0, panStartVpY = 0;

let overlayOpacity = 0.5;
let threshold = 0;
let showForestry = false;

// ============================================================
// DOM REFS
// ============================================================
const mapContainer = document.getElementById('mapContainer');
const mapViewport = document.getElementById('mapViewport');
const rgbCanvas = document.getElementById('rgbCanvas');
const overlayCanvas = document.getElementById('overlayCanvas');
const forestryCanvas = HAS_FORESTRY ? document.getElementById('forestryCanvas') : null;
const markerLayer = document.getElementById('markerLayer');
const rgbCtx = rgbCanvas.getContext('2d');
const overlayCtx = overlayCanvas.getContext('2d');

// ============================================================
// IMAGE LOADING
// ============================================================
let imagesNeeded = HAS_FORESTRY ? 3 : 2;
let imagesLoaded = 0;

const rgbImg = new Image();
const probImg = new Image();

function onImageLoad() {{
    imagesLoaded++;
    if (imagesLoaded >= imagesNeeded) initCanvas();
}}

rgbImg.onload = onImageLoad;
probImg.onload = onImageLoad;
rgbImg.src = 'data:image/png;base64,{rgb_base64}';
probImg.src = 'data:image/png;base64,{prob_base64}';

{forestry_js_load}

function initCanvas() {{
    rgbCanvas.width = IMG_W;
    rgbCanvas.height = IMG_H;
    overlayCanvas.width = IMG_W;
    overlayCanvas.height = IMG_H;

    rgbCtx.drawImage(rgbImg, 0, 0, IMG_W, IMG_H);
    drawOverlay();

    if (HAS_FORESTRY && forestryCanvas) {{
        forestryCanvas.width = IMG_W;
        forestryCanvas.height = IMG_H;
        const fCtx = forestryCanvas.getContext('2d');
        fCtx.drawImage(forestryImg, 0, 0, IMG_W, IMG_H);
        forestryCanvas.style.display = 'none';
    }}

    // Fit to screen
    const cw = mapContainer.clientWidth;
    const ch = mapContainer.clientHeight;
    vpScale = Math.min(cw / IMG_W, ch / IMG_H) * 0.95;
    vpX = (cw - IMG_W * vpScale) / 2;
    vpY = (ch - IMG_H * vpScale) / 2;
    updateViewport();
}}

function drawOverlay() {{
    overlayCtx.clearRect(0, 0, IMG_W, IMG_H);
    overlayCtx.globalAlpha = 1.0;
    overlayCtx.drawImage(probImg, 0, 0, IMG_W, IMG_H);

    if (threshold > 0) {{
        const imageData = overlayCtx.getImageData(0, 0, IMG_W, IMG_H);
        const data = imageData.data;
        const scaleX = DISPLAY_W / IMG_W;
        const scaleY = DISPLAY_H / IMG_H;
        for (let y = 0; y < IMG_H; y++) {{
            const gi = Math.min(Math.floor(y * scaleY), DISPLAY_H - 1);
            const row = probGrid[gi];
            for (let x = 0; x < IMG_W; x++) {{
                const gj = Math.min(Math.floor(x * scaleX), DISPLAY_W - 1);
                if (row[gj] < threshold) {{
                    data[(y * IMG_W + x) * 4 + 3] = 0;
                }}
            }}
        }}
        overlayCtx.putImageData(imageData, 0, 0);
    }}

    overlayCanvas.style.opacity = overlayOpacity;
}}

function updateViewport() {{
    mapViewport.style.transform = `translate(${{vpX}}px, ${{vpY}}px) scale(${{vpScale}})`;
}}

// ============================================================
// COORDINATE MAPPING
// ============================================================
function pixelToLatLon(px, py) {{
    const lon = LON_MIN + (px / IMG_W) * (LON_MAX - LON_MIN);
    const lat = LAT_MAX - (py / IMG_H) * (LAT_MAX - LAT_MIN);
    return {{ lat, lon }};
}}

function latLonToPixel(lat, lon) {{
    const px = ((lon - LON_MIN) / (LON_MAX - LON_MIN)) * IMG_W;
    const py = ((LAT_MAX - lat) / (LAT_MAX - LAT_MIN)) * IMG_H;
    return {{ px, py }};
}}

function screenToImage(sx, sy) {{
    const rect = mapContainer.getBoundingClientRect();
    const cx = sx - rect.left;
    const cy = sy - rect.top;
    const ix = (cx - vpX) / vpScale;
    const iy = (cy - vpY) / vpScale;
    return {{ ix, iy }};
}}

function getProbAtPixel(ix, iy) {{
    const gi = Math.min(Math.floor(iy / IMG_H * DISPLAY_H), DISPLAY_H - 1);
    const gj = Math.min(Math.floor(ix / IMG_W * DISPLAY_W), DISPLAY_W - 1);
    if (gi >= 0 && gi < DISPLAY_H && gj >= 0 && gj < DISPLAY_W) {{
        return probGrid[gi][gj];
    }}
    return null;
}}

// ============================================================
// ANNOTATION MANAGEMENT
// ============================================================
function addAnnotation(ix, iy, isYew) {{
    const {{ lat, lon }} = pixelToLatLon(ix, iy);
    const ann = {{
        id: nextId++,
        lat: lat,
        lon: lon,
        has_yew: isYew ? 1 : 0,
        px_row: Math.round(iy),
        px_col: Math.round(ix),
        prob: getProbAtPixel(ix, iy)
    }};
    annotations.push(ann);
    createMarkerElement(ann);
    updateCounts();
    showToast(`${{isYew ? '🌲 Yew' : '❌ Not Yew'}} at ${{lat.toFixed(5)}}°N, ${{Math.abs(lon).toFixed(5)}}°W`);
}}

function removeAnnotation(id) {{
    annotations = annotations.filter(a => a.id !== id);
    const el = document.getElementById('marker-' + id);
    if (el) el.remove();
    updateCounts();
}}

function createMarkerElement(ann) {{
    const div = document.createElement('div');
    div.className = 'annotation-marker ' + (ann.has_yew ? 'yew' : 'notyew');
    div.id = 'marker-' + ann.id;
    div.style.left = ann.px_col + 'px';
    div.style.top = ann.px_row + 'px';
    div.title = `${{ann.has_yew ? 'YEW' : 'NOT YEW'}} | ${{ann.lat.toFixed(5)}}°N, ${{Math.abs(ann.lon).toFixed(5)}}°W | P=${{ann.prob !== null ? ann.prob.toFixed(3) : '?'}}`;

    div.addEventListener('contextmenu', (e) => {{
        e.preventDefault();
        e.stopPropagation();
        removeAnnotation(ann.id);
    }});

    markerLayer.appendChild(div);
}}

function rebuildMarkers() {{
    markerLayer.innerHTML = '';
    annotations.forEach(a => createMarkerElement(a));
}}

function updateCounts() {{
    const yewN = annotations.filter(a => a.has_yew === 1).length;
    const notyewN = annotations.filter(a => a.has_yew === 0).length;
    document.getElementById('yewCount').textContent = '🌲 ' + yewN;
    document.getElementById('notyewCount').textContent = '❌ ' + notyewN;
}}

function undoLast() {{
    if (annotations.length === 0) return;
    const last = annotations[annotations.length - 1];
    removeAnnotation(last.id);
}}

function clearAll() {{
    if (annotations.length === 0) return;
    if (!confirm(`Delete all ${{annotations.length}} annotations?`)) return;
    annotations = [];
    markerLayer.innerHTML = '';
    updateCounts();
    showToast('All annotations cleared');
}}

// ============================================================
// EXPORT / IMPORT
// ============================================================
function exportCSV() {{
    if (annotations.length === 0) {{
        showToast('No annotations to export');
        return;
    }}
    let csv = 'lat,lon,has_yew,px_row,px_col,prob\\n';
    annotations.forEach(a => {{
        csv += `${{a.lat.toFixed(6)}},${{a.lon.toFixed(6)}},${{a.has_yew}},${{a.px_row}},${{a.px_col}},${{(a.prob !== null ? a.prob.toFixed(4) : '')}}\\n`;
    }});

    const blob = new Blob([csv], {{ type: 'text/csv' }});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'yew_annotations.csv';
    a.click();
    URL.revokeObjectURL(url);
    showToast(`Exported ${{annotations.length}} annotations`);
}}

function importCSV() {{
    document.getElementById('fileInput').click();
}}

function handleImport(event) {{
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = function(e) {{
        const lines = e.target.result.trim().split('\\n');
        let imported = 0;
        for (let i = 1; i < lines.length; i++) {{
            const parts = lines[i].split(',');
            if (parts.length >= 3) {{
                const lat = parseFloat(parts[0]);
                const lon = parseFloat(parts[1]);
                const has_yew = parseInt(parts[2]);
                const {{ px, py }} = latLonToPixel(lat, lon);
                const ann = {{
                    id: nextId++,
                    lat: lat,
                    lon: lon,
                    has_yew: has_yew,
                    px_row: Math.round(py),
                    px_col: Math.round(px),
                    prob: getProbAtPixel(px, py)
                }};
                annotations.push(ann);
                createMarkerElement(ann);
                imported++;
            }}
        }}
        updateCounts();
        showToast(`Imported ${{imported}} annotations`);
    }};
    reader.readAsText(file);
    event.target.value = '';
}}

// ============================================================
// MODE & UI
// ============================================================
function setMode(m) {{
    mode = m;
    document.getElementById('btnPan').classList.toggle('active', m === 'pan');
    document.getElementById('btnYew').classList.toggle('active', m === 'yew');
    document.getElementById('btnNotYew').classList.toggle('active', m === 'notyew');
    mapContainer.classList.toggle('panning', m === 'pan');
}}

function updateOverlay(val) {{
    overlayOpacity = val / 100;
    overlayCanvas.style.opacity = overlayOpacity;
}}

function updateThreshold(val) {{
    threshold = val / 100;
    document.getElementById('threshVal').textContent = threshold.toFixed(2);
    drawOverlay();
}}

function toggleForestry() {{
    if (!HAS_FORESTRY || !forestryCanvas) return;
    showForestry = !showForestry;
    forestryCanvas.style.display = showForestry ? '' : 'none';
    const btn = document.getElementById('btnForestry');
    btn.classList.toggle('forestry-on', showForestry);
    const legend = document.getElementById('forestryLegend');
    if (legend) legend.classList.toggle('visible', showForestry);
}}

function showToast(msg) {{
    const toast = document.getElementById('toast');
    toast.textContent = msg;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 2000);
}}

// ============================================================
// EVENT HANDLERS
// ============================================================

// Mouse move for coordinates
mapContainer.addEventListener('mousemove', (e) => {{
    const {{ ix, iy }} = screenToImage(e.clientX, e.clientY);
    if (ix >= 0 && ix < IMG_W && iy >= 0 && iy < IMG_H) {{
        const {{ lat, lon }} = pixelToLatLon(ix, iy);
        const prob = getProbAtPixel(ix, iy);
        document.getElementById('coordDisplay').textContent =
            `${{lat.toFixed(5)}}°N, ${{Math.abs(lon).toFixed(5)}}°W`;
        document.getElementById('probDisplay').textContent =
            `P(yew): ${{prob !== null ? prob.toFixed(4) : '--'}}`;
        document.getElementById('pixelDisplay').textContent =
            `Pixel: (${{Math.round(iy)}}, ${{Math.round(ix)}})`;
    }}

    // Panning
    if (isPanning) {{
        vpX = panStartVpX + (e.clientX - panStartX);
        vpY = panStartVpY + (e.clientY - panStartY);
        updateViewport();
    }}
}});

// Mouse down
mapContainer.addEventListener('mousedown', (e) => {{
    if (e.button === 0) {{
        if (mode === 'pan') {{
            isPanning = true;
            panStartX = e.clientX;
            panStartY = e.clientY;
            panStartVpX = vpX;
            panStartVpY = vpY;
        }} else if (mode === 'yew' || mode === 'notyew') {{
            const {{ ix, iy }} = screenToImage(e.clientX, e.clientY);
            if (ix >= 0 && ix < IMG_W && iy >= 0 && iy < IMG_H) {{
                addAnnotation(ix, iy, mode === 'yew');
            }}
        }}
    }}
}});

// Mouse up
window.addEventListener('mouseup', () => {{
    isPanning = false;
}});

// Scroll to zoom
mapContainer.addEventListener('wheel', (e) => {{
    e.preventDefault();
    const rect = mapContainer.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const zoomFactor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    const newScale = vpScale * zoomFactor;

    vpX = mx - (mx - vpX) * (newScale / vpScale);
    vpY = my - (my - vpY) * (newScale / vpScale);
    vpScale = newScale;
    updateViewport();
}}, {{ passive: false }});

// Double-click to reset view
mapContainer.addEventListener('dblclick', (e) => {{
    // Only reset if in pan mode (avoid interfering with annotation placement)
    if (mode === 'pan') {{
        const cw = mapContainer.clientWidth;
        const ch = mapContainer.clientHeight;
        vpScale = Math.min(cw / IMG_W, ch / IMG_H) * 0.95;
        vpX = (cw - IMG_W * vpScale) / 2;
        vpY = (ch - IMG_H * vpScale) / 2;
        updateViewport();
    }}
}});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {{
    if (e.key === '1') setMode('yew');
    else if (e.key === '2') setMode('notyew');
    else if (e.key === ' ') {{ e.preventDefault(); setMode('pan'); }}
    else if (e.key === 'z' && (e.ctrlKey || e.metaKey)) {{ e.preventDefault(); undoLast(); }}
    else if (e.key === 's' && (e.ctrlKey || e.metaKey)) {{ e.preventDefault(); exportCSV(); }}
    else if (e.key === 'f' || e.key === 'F') {{ toggleForestry(); }}
}});

// Prevent context menu on map
mapContainer.addEventListener('contextmenu', (e) => e.preventDefault());

// Auto-save to localStorage every 10 seconds
setInterval(() => {{
    if (annotations.length > 0) {{
        localStorage.setItem('yew_annotations_backup', JSON.stringify(annotations));
    }}
}}, 10000);

// Restore from localStorage on load
window.addEventListener('load', () => {{
    const backup = localStorage.getItem('yew_annotations_backup');
    if (backup) {{
        try {{
            const saved = JSON.parse(backup);
            if (saved.length > 0 && confirm(`Restore ${{saved.length}} annotations from previous session?`)) {{
                saved.forEach(a => {{
                    a.id = nextId++;
                    annotations.push(a);
                    createMarkerElement(a);
                }});
                updateCounts();
                showToast(`Restored ${{saved.length}} annotations`);
            }}
        }} catch(err) {{}}
    }}
}});
</script>
</body>
</html>'''

    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"✓ Created annotation tool: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Create annotation tool for yew maps')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory with prob_grid.npy, rgb_image.npy, metadata.json')
    parser.add_argument('--output', type=str, default=None,
                        help='Output HTML file path')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    # Load data
    print("Loading data...")
    prob_grid = np.load(input_dir / 'prob_grid.npy')
    rgb_image = np.load(input_dir / 'rgb_image.npy')
    print(f"  Probability grid: {prob_grid.shape}")
    print(f"  RGB image: {rgb_image.shape}")

    with open(input_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    # Encode images
    print("Encoding images...")
    h, w = prob_grid.shape

    if rgb_image.dtype != np.uint8:
        rgb_u8 = (np.clip(rgb_image, 0, 1) * 255).astype(np.uint8)
    else:
        rgb_u8 = rgb_image
    rgb_base64 = array_to_base64(rgb_u8)

    cmap = create_probability_colormap()
    indices = (np.clip(prob_grid, 0, 1) * 255).astype(np.uint8)
    prob_rgb = cmap[indices.flatten()].reshape(h, w, 3)
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
    output_path = args.output or str(input_dir / 'annotation_tool.html')
    print("Generating annotation tool HTML...")
    generate_annotation_html(rgb_base64, prob_base64, prob_grid, metadata,
                             output_path, forestry_base64=forestry_base64)

    print(f"\nOpen in browser: file://{Path(output_path).resolve()}")
    print("\nWorkflow:")
    print("  1. Open the HTML file in a web browser")
    print("  2. Press 1 to enter YEW mode, 2 for NOT YEW mode, Space for Pan")
    print("  3. Click on the map to place markers")
    print("  4. Right-click markers to delete them")
    print("  5. Press F to toggle forestry overlay")
    print("  6. Click 'Export CSV' when done")
    print("  7. Run the retrain script with the exported CSV")


if __name__ == '__main__':
    main()
