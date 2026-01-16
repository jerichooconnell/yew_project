#!/usr/bin/env python3
"""
Create Interactive Yew Probability Map

Generates an HTML file with interactive sliders for:
- Overlay opacity (alpha)
- Probability threshold filtering

Usage:
    python scripts/visualization/create_interactive_map.py \
        --input-dir results/predictions/large_area \
        --output interactive_yew_map.html
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
    img.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


def create_probability_colormap():
    """Create a colormap for probability values (blue to red)."""
    # Blue -> Light Blue -> White -> Light Red -> Red
    colors = [
        (33, 102, 172),    # 0.0 - Dark blue
        (146, 197, 222),   # 0.25 - Light blue
        (247, 247, 247),   # 0.5 - White
        (244, 165, 130),   # 0.75 - Light red
        (214, 96, 77),     # 1.0 - Red
    ]
    
    # Interpolate to 256 colors
    cmap = np.zeros((256, 3), dtype=np.uint8)
    positions = [0, 64, 128, 192, 255]
    
    for i in range(256):
        # Find which segment we're in
        for j in range(len(positions) - 1):
            if positions[j] <= i <= positions[j + 1]:
                t = (i - positions[j]) / (positions[j + 1] - positions[j])
                c1 = np.array(colors[j])
                c2 = np.array(colors[j + 1])
                cmap[i] = (c1 * (1 - t) + c2 * t).astype(np.uint8)
                break
    
    return cmap


def apply_colormap(prob_grid, cmap):
    """Apply colormap to probability grid, returning RGBA image."""
    h, w = prob_grid.shape
    
    # Convert probabilities to indices
    indices = (np.clip(prob_grid, 0, 1) * 255).astype(np.uint8)
    
    # Create RGB image
    rgb = cmap[indices.flatten()].reshape(h, w, 3)
    
    return rgb


def generate_html(rgb_base64, prob_grid, metadata, output_path):
    """Generate interactive HTML file."""
    
    # Get dimensions
    h, w = prob_grid.shape
    
    # Create colormap
    cmap = create_probability_colormap()
    
    # Apply colormap to get colored probability image
    prob_rgb = apply_colormap(prob_grid, cmap)
    prob_base64 = array_to_base64(prob_rgb)
    
    # Compute additional statistics
    prob_flat = prob_grid.flatten()
    
    # Histogram bins
    hist_bins = 50
    hist_counts, hist_edges = np.histogram(prob_flat, bins=hist_bins, range=(0, 1))
    hist_data = {
        'counts': hist_counts.tolist(),
        'edges': hist_edges.tolist()
    }
    
    # Percentiles
    percentiles = {
        'p10': float(np.percentile(prob_flat, 10)),
        'p25': float(np.percentile(prob_flat, 25)),
        'p50': float(np.percentile(prob_flat, 50)),
        'p75': float(np.percentile(prob_flat, 75)),
        'p90': float(np.percentile(prob_flat, 90)),
        'p95': float(np.percentile(prob_flat, 95)),
        'p99': float(np.percentile(prob_flat, 99)),
    }
    
    # Area calculations (assuming 10m pixels)
    scale = metadata.get('scale_m', 10)
    pixel_area_m2 = scale ** 2
    pixel_area_ha = pixel_area_m2 / 10000
    
    threshold_stats = []
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        count = int((prob_flat >= thresh).sum())
        area_ha = count * pixel_area_ha
        pct = 100 * count / len(prob_flat)
        threshold_stats.append({
            'threshold': thresh,
            'count': count,
            'area_ha': area_ha,
            'percent': pct
        })
    
    # Encode probability grid as JSON (downsampled if too large)
    # For interactivity, we need the raw values
    max_size = 500
    if h > max_size or w > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        # Simple downsampling
        prob_small = np.zeros((new_h, new_w), dtype=np.float32)
        for i in range(new_h):
            for j in range(new_w):
                src_i = int(i / scale)
                src_j = int(j / scale)
                prob_small[i, j] = prob_grid[src_i, src_j]
        prob_json = prob_small.tolist()
        display_h, display_w = new_h, new_w
    else:
        prob_json = prob_grid.tolist()
        display_h, display_w = h, w
    
    # Get bounding box from metadata
    bbox = metadata.get('bbox', {})
    lat_min = bbox.get('lat_min', 0)
    lat_max = bbox.get('lat_max', 0)
    lon_min = bbox.get('lon_min', 0)
    lon_max = bbox.get('lon_max', 0)
    
    # Model info
    model_info = metadata.get('model', {})
    stats = metadata.get('statistics', {})
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Yew Probability Map</title>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        h1 {{
            text-align: center;
            margin-bottom: 10px;
            color: #4ecdc4;
        }}
        
        .subtitle {{
            text-align: center;
            color: #888;
            margin-bottom: 20px;
        }}
        
        .main-content {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}
        
        .map-container {{
            flex: 1;
            min-width: 600px;
            background: #16213e;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }}
        
        .canvas-wrapper {{
            position: relative;
            width: 100%;
            overflow: hidden;
            border-radius: 8px;
        }}
        
        canvas {{
            display: block;
            width: 100%;
            height: auto;
            cursor: crosshair;
        }}
        
        .controls {{
            width: 320px;
            background: #16213e;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }}
        
        .control-group {{
            margin-bottom: 20px;
        }}
        
        .control-group label {{
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #4ecdc4;
        }}
        
        .slider-container {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        input[type="range"] {{
            flex: 1;
            -webkit-appearance: none;
            height: 8px;
            border-radius: 4px;
            background: #0f3460;
            outline: none;
        }}
        
        input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #4ecdc4;
            cursor: pointer;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }}
        
        .slider-value {{
            min-width: 50px;
            text-align: right;
            font-family: monospace;
            font-size: 14px;
            color: #fff;
        }}
        
        .info-box {{
            background: #0f3460;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
        }}
        
        .info-box h3 {{
            color: #4ecdc4;
            margin-bottom: 10px;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .info-row {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
            font-size: 13px;
        }}
        
        .info-row .label {{
            color: #888;
        }}
        
        .info-row .value {{
            color: #fff;
            font-family: monospace;
        }}
        
        .colorbar {{
            height: 20px;
            border-radius: 4px;
            background: linear-gradient(to right, 
                rgb(33, 102, 172), 
                rgb(146, 197, 222), 
                rgb(247, 247, 247), 
                rgb(244, 165, 130), 
                rgb(214, 96, 77));
            margin: 10px 0;
        }}
        
        .colorbar-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: #888;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }}
        
        .stat-card {{
            background: #1a1a2e;
            border-radius: 6px;
            padding: 10px;
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 18px;
            font-weight: bold;
            color: #4ecdc4;
        }}
        
        .stat-label {{
            font-size: 11px;
            color: #888;
            margin-top: 4px;
        }}
        
        .cursor-info {{
            background: #0f3460;
            border-radius: 8px;
            padding: 15px;
            margin-top: 15px;
            min-height: 80px;
        }}
        
        .cursor-info h3 {{
            color: #4ecdc4;
            margin-bottom: 10px;
            font-size: 14px;
        }}
        
        #cursor-details {{
            font-family: monospace;
            font-size: 13px;
            line-height: 1.6;
        }}
        
        .threshold-indicator {{
            margin-top: 10px;
            padding: 10px;
            background: #1a1a2e;
            border-radius: 6px;
            text-align: center;
        }}
        
        .threshold-indicator .count {{
            font-size: 24px;
            font-weight: bold;
            color: #e94560;
        }}
        
        .threshold-indicator .desc {{
            font-size: 12px;
            color: #888;
            margin-top: 4px;
        }}
        
        .view-buttons {{
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }}
        
        .view-btn {{
            flex: 1;
            padding: 10px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.2s;
            background: #0f3460;
            color: #888;
        }}
        
        .view-btn.active {{
            background: #4ecdc4;
            color: #1a1a2e;
        }}
        
        .view-btn:hover {{
            opacity: 0.8;
        }}
        
        .tabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid #0f3460;
        }}
        
        .tab-btn {{
            padding: 12px 24px;
            border: none;
            background: none;
            color: #888;
            cursor: pointer;
            font-weight: 600;
            border-bottom: 3px solid transparent;
            transition: all 0.2s;
        }}
        
        .tab-btn.active {{
            color: #4ecdc4;
            border-bottom-color: #4ecdc4;
        }}
        
        .tab-btn:hover {{
            color: #fff;
        }}
        
        .tab-content {{
            display: none;
        }}
        
        .tab-content.active {{
            display: block;
        }}
        
        .stats-table {{
            width: 100%;
            border-collapse: collapse;
            background: #0f3460;
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 20px;
        }}
        
        .stats-table th {{
            background: #1a1a2e;
            padding: 12px;
            text-align: left;
            color: #4ecdc4;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .stats-table td {{
            padding: 10px 12px;
            border-top: 1px solid #1a1a2e;
            font-size: 13px;
        }}
        
        .stats-table tr:hover {{
            background: #16213e;
        }}
        
        .histogram-container {{
            background: #0f3460;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        
        .histogram-title {{
            color: #4ecdc4;
            margin-bottom: 15px;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .histogram {{
            display: flex;
            align-items: flex-end;
            height: 200px;
            gap: 2px;
        }}
        
        .histogram-bar {{
            flex: 1;
            background: linear-gradient(to top, #4ecdc4, #e94560);
            border-radius: 2px 2px 0 0;
            transition: opacity 0.2s;
        }}
        
        .histogram-bar:hover {{
            opacity: 0.7;
        }}
        
        .section-title {{
            color: #4ecdc4;
            margin: 25px 0 15px 0;
            font-size: 16px;
            text-transform: uppercase;
            letter-spacing: 1px;
            border-bottom: 2px solid #0f3460;
            padding-bottom: 8px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🌲 Yew Probability Map</h1>
        <p class="subtitle">
            {lat_min:.4f}°N to {lat_max:.4f}°N, {abs(lon_max):.4f}°W to {abs(lon_min):.4f}°W
        </p>
        
        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('map')">Map View</button>
            <button class="tab-btn" onclick="switchTab('statistics')">Statistics & Analysis</button>
        </div>
        
        <div class="main-content">
            <!-- Map Tab -->
            <div id="tab-map" class="tab-content active">
                <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                    <div class="map-container">
                        <div class="canvas-wrapper">
                            <canvas id="mapCanvas"></canvas>
                        </div>
                    </div>
                    
                    <div class="controls">
                        <div class="view-buttons">
                            <button class="view-btn active" id="btn-overlay" onclick="setView('overlay')">Overlay</button>
                            <button class="view-btn" id="btn-rgb" onclick="setView('rgb')">RGB Only</button>
                            <button class="view-btn" id="btn-prob" onclick="setView('prob')">Probability</button>
                        </div>
                
                <div class="control-group">
                    <label>Overlay Opacity</label>
                    <div class="slider-container">
                        <input type="range" id="alphaSlider" min="0" max="100" value="50">
                        <span class="slider-value" id="alphaValue">50%</span>
                    </div>
                </div>
                
                <div class="control-group">
                    <label>Probability Threshold</label>
                    <div class="slider-container">
                        <input type="range" id="thresholdSlider" min="0" max="100" value="0">
                        <span class="slider-value" id="thresholdValue">0.00</span>
                    </div>
                    <div class="threshold-indicator">
                        <div class="count" id="pixelCount">-</div>
                        <div class="desc">pixels above threshold</div>
                    </div>
                </div>
                
                <div class="info-box">
                    <h3>Colormap</h3>
                    <div class="colorbar"></div>
                    <div class="colorbar-labels">
                        <span>0.0 (Low)</span>
                        <span>0.5</span>
                        <span>1.0 (High)</span>
                    </div>
                </div>
                
                <div class="info-box">
                    <h3>Model Performance</h3>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-value">{model_info.get('validation_accuracy', 0)*100:.1f}%</div>
                            <div class="stat-label">Val Accuracy</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{model_info.get('validation_f1', 0)*100:.1f}%</div>
                            <div class="stat-label">F1 Score</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{stats.get('mean', 0):.3f}</div>
                            <div class="stat-label">Mean Prob</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{stats.get('pixels_above_70', 0):,}</div>
                            <div class="stat-label">P > 0.7</div>
                        </div>
                    </div>
                </div>
                
                <div class="cursor-info">
                    <h3>Cursor Position</h3>
                    <div id="cursor-details">
                        Hover over the map to see details
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Image data
        const rgbBase64 = "{rgb_base64}";
        const probBase64 = "{prob_base64}";
        const probData = {json.dumps(prob_json)};
        
        const origWidth = {w};
        const origHeight = {h};
        const displayWidth = {display_w};
        const displayHeight = {display_h};
        
        const latMin = {lat_min};
        const latMax = {lat_max};
        const lonMin = {lon_min};
        const lonMax = {lon_max};
        
        // Canvas setup
        const canvas = document.getElementById('mapCanvas');
        const ctx = canvas.getContext('2d');
        
        // Load images
        const rgbImg = new Image();
        const probImg = new Image();
        let imagesLoaded = 0;
        
        let currentView = 'overlay';
        let alpha = 0.5;
        let threshold = 0;
        
        function onImageLoad() {{
            imagesLoaded++;
            if (imagesLoaded === 2) {{
                canvas.width = rgbImg.width;
                canvas.height = rgbImg.height;
                render();
            }}
        }}
        
        rgbImg.onload = onImageLoad;
        probImg.onload = onImageLoad;
        rgbImg.src = 'data:image/png;base64,' + rgbBase64;
        probImg.src = 'data:image/png;base64,' + probBase64;
        
        function getColorForProb(p) {{
            // Blue to red colormap
            const colors = [
                [33, 102, 172],
                [146, 197, 222],
                [247, 247, 247],
                [244, 165, 130],
                [214, 96, 77]
            ];
            const positions = [0, 0.25, 0.5, 0.75, 1];
            
            for (let i = 0; i < positions.length - 1; i++) {{
                if (p >= positions[i] && p <= positions[i + 1]) {{
                    const t = (p - positions[i]) / (positions[i + 1] - positions[i]);
                    return [
                        Math.round(colors[i][0] * (1 - t) + colors[i + 1][0] * t),
                        Math.round(colors[i][1] * (1 - t) + colors[i + 1][1] * t),
                        Math.round(colors[i][2] * (1 - t) + colors[i + 1][2] * t)
                    ];
                }}
            }}
            return colors[colors.length - 1];
        }}
        
        function render() {{
            if (imagesLoaded < 2) return;
            
            const w = canvas.width;
            const h = canvas.height;
            
            ctx.clearRect(0, 0, w, h);
            
            if (currentView === 'rgb') {{
                ctx.drawImage(rgbImg, 0, 0);
            }} else if (currentView === 'prob') {{
                drawFilteredProb(w, h, 1.0);
            }} else {{
                // Overlay
                ctx.drawImage(rgbImg, 0, 0);
                drawFilteredProb(w, h, alpha);
            }}
            
            updatePixelCount();
        }}
        
        function drawFilteredProb(w, h, opacity) {{
            // Draw probability image to a temporary canvas first
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = w;
            tempCanvas.height = h;
            const tempCtx = tempCanvas.getContext('2d');
            
            tempCtx.drawImage(probImg, 0, 0);
            
            // Apply threshold filter if needed
            if (threshold > 0) {{
                const imageData = tempCtx.getImageData(0, 0, w, h);
                const data = imageData.data;
                
                const scaleX = displayWidth / w;
                const scaleY = displayHeight / h;
                
                for (let y = 0; y < h; y++) {{
                    for (let x = 0; x < w; x++) {{
                        const probY = Math.min(Math.floor(y * scaleY), displayHeight - 1);
                        const probX = Math.min(Math.floor(x * scaleX), displayWidth - 1);
                        const prob = probData[probY][probX];
                        
                        if (prob < threshold) {{
                            const idx = (y * w + x) * 4;
                            // Make below-threshold pixels fully transparent
                            data[idx + 3] = 0;
                        }}
                    }}
                }}
                
                tempCtx.putImageData(imageData, 0, 0);
            }}
            
            // Draw the filtered probability layer on main canvas
            ctx.globalAlpha = opacity;
            ctx.drawImage(tempCanvas, 0, 0);
            ctx.globalAlpha = 1.0;
        }}
        
        function updatePixelCount() {{
            let count = 0;
            for (let y = 0; y < displayHeight; y++) {{
                for (let x = 0; x < displayWidth; x++) {{
                    if (probData[y][x] >= threshold) {{
                        count++;
                    }}
                }}
            }}
            
            // Scale to original resolution
            const scale = (origWidth * origHeight) / (displayWidth * displayHeight);
            const estimatedCount = Math.round(count * scale);
            
            document.getElementById('pixelCount').textContent = estimatedCount.toLocaleString();
        }}
        
        function setView(view) {{
            currentView = view;
            document.querySelectorAll('.view-btn').forEach(btn => btn.classList.remove('active'));
            document.getElementById('btn-' + view).classList.add('active');
            render();
        }}
        
        // Slider events
        document.getElementById('alphaSlider').addEventListener('input', function(e) {{
            alpha = e.target.value / 100;
            document.getElementById('alphaValue').textContent = e.target.value + '%';
            render();
        }});
        
        document.getElementById('thresholdSlider').addEventListener('input', function(e) {{
            threshold = e.target.value / 100;
            document.getElementById('thresholdValue').textContent = threshold.toFixed(2);
            render();
        }});
        
        // Mouse hover
        canvas.addEventListener('mousemove', function(e) {{
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            
            const canvasX = (e.clientX - rect.left) * scaleX;
            const canvasY = (e.clientY - rect.top) * scaleY;
            
            const probX = Math.floor(canvasX * displayWidth / canvas.width);
            const probY = Math.floor(canvasY * displayHeight / canvas.height);
            
            if (probX >= 0 && probX < displayWidth && probY >= 0 && probY < displayHeight) {{
                const prob = probData[probY][probX];
                
                // Calculate lat/lon
                const lon = lonMin + (canvasX / canvas.width) * (lonMax - lonMin);
                const lat = latMax - (canvasY / canvas.height) * (latMax - latMin);
                
                const color = getColorForProb(prob);
                const colorStr = `rgb(${{color[0]}}, ${{color[1]}}, ${{color[2]}})`;
                
                document.getElementById('cursor-details').innerHTML = `
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                        <div style="width: 20px; height: 20px; border-radius: 4px; background: ${{colorStr}};"></div>
                        <span style="font-size: 18px; font-weight: bold;">P = ${{prob.toFixed(3)}}</span>
                    </div>
                    <div>Lat: ${{lat.toFixed(5)}}°N</div>
                    <div>Lon: ${{Math.abs(lon).toFixed(5)}}°W</div>
                `;
            }}
        }});
        
        canvas.addEventListener('mouseleave', function() {{
            document.getElementById('cursor-details').innerHTML = 'Hover over the map to see details';
        }});
    </script>
</body>
</html>'''
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"✓ Created interactive map: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Create interactive yew probability map')
    parser.add_argument('--input-dir', type=str, default='results/predictions/large_area',
                        help='Directory with prob_grid.npy, rgb_image.npy, and metadata.json')
    parser.add_argument('--output', type=str, default=None,
                        help='Output HTML file path (default: input_dir/interactive_map.html)')
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    
    # Load data
    print("Loading data...")
    
    prob_path = input_dir / 'prob_grid.npy'
    rgb_path = input_dir / 'rgb_image.npy'
    metadata_path = input_dir / 'metadata.json'
    
    if not prob_path.exists():
        raise FileNotFoundError(f"Probability grid not found: {prob_path}")
    
    prob_grid = np.load(prob_path)
    print(f"  Probability grid: {prob_grid.shape}")
    
    if rgb_path.exists():
        rgb_image = np.load(rgb_path)
        print(f"  RGB image: {rgb_image.shape}")
    else:
        # Create placeholder RGB from probability
        print("  No RGB image found, creating grayscale placeholder")
        rgb_image = np.stack([prob_grid] * 3, axis=-1)
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"  Metadata loaded")
    else:
        metadata = {}
        print("  No metadata found, using defaults")
    
    # Convert RGB to base64
    print("Encoding images...")
    rgb_base64 = array_to_base64(rgb_image)
    
    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_dir / 'interactive_map.html'
    
    # Generate HTML
    print("Generating interactive HTML...")
    generate_html(rgb_base64, prob_grid, metadata, output_path)
    
    print(f"\nOpen in browser: file://{output_path.absolute()}")


if __name__ == '__main__':
    main()
