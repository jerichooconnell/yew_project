#!/usr/bin/env python3
"""
Generate interactive Leaflet heatmap overlaying yew probability predictions
on the coastal BC study region boundary.
"""

import json, csv, argparse, os


def load_predictions(csv_path, threshold=0.0):
    """Load all predictions, optionally filtering by threshold."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = float(row['prob'])
            if p >= threshold:
                rows.append((float(row['lat']), float(row['lon']), p))
    return rows



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', default='results/analysis/coastal_region_100k/sample_predictions_coastal.csv')
    ap.add_argument('--boundary',    default='data/processed/coastal_study_region.geojson')
    ap.add_argument('--output',      default='results/analysis/coastal_region_100k/coastal_yew_overlay.html')
    ap.add_argument('--threshold',   type=float, default=0.0,
                    help='Only show points with prob >= threshold (default 0 = all)')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Loading boundary: {args.boundary}")
    with open(args.boundary) as f:
        boundary_geojson = json.load(f)

    print(f"Loading predictions: {args.predictions}")
    points = load_predictions(args.predictions, threshold=args.threshold)
    print(f"  {len(points):,} points to render (threshold={args.threshold})")

    # Encode as compact JS array [lat, lon, intensity]
    heat_data = ",\n".join(f"[{lat:.5f},{lon:.5f},{p:.4f}]" for lat, lon, p in points)

    boundary_json_str = json.dumps(boundary_geojson)

    # Stats
    n_total = len(points)
    n_05  = sum(1 for _, _, p in points if p >= 0.5)
    n_07  = sum(1 for _, _, p in points if p >= 0.7)
    pct05 = 100 * n_05 / n_total if n_total else 0
    pct07 = 100 * n_07 / n_total if n_total else 0

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>Coastal BC — Yew Probability Heatmap</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
<style>
  html,body,#map{{margin:0;padding:0;height:100%;width:100%;background:#1a1a2e;}}
  #legend{{
    position:absolute;bottom:30px;right:10px;z-index:1000;
    background:rgba(10,10,30,0.88);color:#eee;padding:12px 16px;
    border-radius:8px;font:13px/1.6 sans-serif;min-width:210px;
    box-shadow:0 2px 10px rgba(0,0,0,0.5);
  }}
  #legend h4{{margin:0 0 8px;font-size:14px;color:#fff;}}
  .grad-bar{{
    height:12px;width:100%;border-radius:4px;margin:6px 0 2px;
    background:linear-gradient(to right,
      #00007f 0%,#0000ff 15%,#00ffff 35%,
      #00ff00 50%,#ffff00 65%,#ff8000 80%,#ff0000 100%);
  }}
  .grad-labels{{display:flex;justify-content:space-between;font-size:10px;color:#aaa;}}
  #info{{
    position:absolute;top:10px;left:50%;transform:translateX(-50%);
    z-index:1000;background:rgba(10,10,30,0.88);color:#eee;
    padding:6px 16px;border-radius:20px;font:13px sans-serif;
    box-shadow:0 2px 8px rgba(0,0,0,0.4);white-space:nowrap;
  }}
  #controls{{
    position:absolute;top:50px;left:50%;transform:translateX(-50%);
    z-index:1000;background:rgba(10,10,30,0.88);color:#eee;
    padding:6px 16px;border-radius:20px;font:12px sans-serif;
    box-shadow:0 2px 8px rgba(0,0,0,0.4);display:flex;gap:16px;align-items:center;
  }}
  input[type=range]{{width:100px;}}
</style>
</head>
<body>
<div id="map"></div>
<div id="info">Coastal BC · {n_total:,} pts · P≥0.5: {n_05:,} ({pct05:.1f}%) · P≥0.7: {n_07:,} ({pct07:.1f}%)</div>
<div id="controls">
  Radius: <input type="range" id="radius" min="5" max="40" value="18" oninput="updateHeat()"> <span id="rval">18</span>px &nbsp;
  Blur: <input type="range" id="blur" min="5" max="40" value="20" oninput="updateHeat()"> <span id="bval">20</span>px &nbsp;
  Opacity: <input type="range" id="opacity" min="1" max="10" value="7" oninput="updateHeat()"> <span id="oval">0.7</span>
</div>
<div id="legend">
  <h4>Western Yew Probability</h4>
  <div class="grad-bar"></div>
  <div class="grad-labels"><span>0</span><span>0.25</span><span>0.5</span><span>0.75</span><span>1.0</span></div>
  <hr style="margin:8px 0;border-color:#333"/>
  <div style="font-size:11px;color:#aaa">
    Study area: 227,652 km²<br>
    Sample: 100,000 pts · GEE 300 m<br>
    Model: YewMLP (acc 98.85%, AUC 0.998)
  </div>
</div>
<script>
var map = L.map('map',{{zoomControl:true}}).setView([51.5, -126.5], 6);

L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png',
  {{attribution:'© OpenStreetMap © CARTO',maxZoom:18}}).addTo(map);

// Study region boundary
var boundaryData = {boundary_json_str};
L.geoJSON(boundaryData, {{
  style: {{color:'#60a5fa', weight:1.2, fill:false, opacity:0.6}}
}}).addTo(map);

// Heatmap data: [lat, lon, intensity]
var heatData = [
{heat_data}
];

var heat = L.heatLayer(heatData, {{
  radius: 18,
  blur: 20,
  maxZoom: 12,
  max: 1.0,
  minOpacity: 0.0,
  gradient: {{
    0.0:  '#00007f',
    0.15: '#0000ff',
    0.35: '#00ffff',
    0.50: '#00ff00',
    0.65: '#ffff00',
    0.80: '#ff8000',
    1.0:  '#ff0000'
  }}
}}).addTo(map);

function updateHeat() {{
  var r = +document.getElementById('radius').value;
  var b = +document.getElementById('blur').value;
  var o = +document.getElementById('opacity').value / 10;
  document.getElementById('rval').textContent = r;
  document.getElementById('bval').textContent = b;
  document.getElementById('oval').textContent = o.toFixed(1);
  heat.setOptions({{radius:r, blur:b, minOpacity:0}});
  heat._canvas.style.opacity = o;
}}
</script>
</body>
</html>
"""

    with open(args.output, 'w') as f:
        f.write(html)

    print(f"✓ Saved: {args.output}")


if __name__ == '__main__':
    main()
