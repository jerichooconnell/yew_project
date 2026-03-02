#!/usr/bin/env python3
"""
Step 1 of land-only coastal sampling:
  1. Download Natural Earth 10m land polygons (cached after first run)
  2. Intersect with coastal_study_region.geojson to get land-only boundary
  3. Sample N random points within that land boundary
  4. Write an HTML preview map so you can visually verify the points
     before committing to a GEE extraction.

Run:
    python scripts/prediction/prepare_land_boundary.py

Then open results/analysis/coastal_land_100k/sample_preview.html, check the
points look good, and run the extraction:
    python scripts/prediction/sample_coastal_region.py \
        --region-boundary data/processed/coastal_land_only.geojson \
        --output-dir results/analysis/coastal_land_100k --scale 10

"""

import argparse
import json
import os
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.ops import unary_union

# ── Paths ─────────────────────────────────────────────────────────────────────
STUDY_REGION   = Path('data/processed/coastal_study_region.geojson')
LAND_CACHE_DIR = Path('data/lookup_tables')
LAND_ZIP       = LAND_CACHE_DIR / 'ne_10m_land.zip'
LAND_SHP       = LAND_CACHE_DIR / 'ne_10m_land' / 'ne_10m_land.shp'
LAND_GEOJSON   = Path('data/processed/coastal_land_only.geojson')
OUTPUT_DIR     = Path('results/analysis/coastal_land_100k')
PREVIEW_HTML   = OUTPUT_DIR / 'sample_preview.html'
POINTS_CSV     = OUTPUT_DIR / 'preview_points.csv'

NE_10M_LAND_URL = (
    'https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip'
)


def download_ne_land():
    """Download + unzip Natural Earth 10m land polygons if not cached."""
    if LAND_SHP.exists():
        print(f'Using cached land shapefile: {LAND_SHP}')
        return
    LAND_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f'Downloading Natural Earth 10m land from {NE_10M_LAND_URL} …')
    urllib.request.urlretrieve(NE_10M_LAND_URL, LAND_ZIP)
    print(f'Extracting to {LAND_CACHE_DIR / "ne_10m_land"} …')
    with zipfile.ZipFile(LAND_ZIP) as z:
        z.extractall(LAND_CACHE_DIR / 'ne_10m_land')
    print('Done.')


def build_land_boundary():
    """Intersect NE land with study region, save land-only GeoJSON."""
    if LAND_GEOJSON.exists():
        print(f'Using cached land boundary: {LAND_GEOJSON}')
        return gpd.read_file(LAND_GEOJSON)

    print('Loading study region …')
    study = gpd.read_file(STUDY_REGION)

    print('Loading Natural Earth 10m land polygons …')
    land = gpd.read_file(LAND_SHP)
    land = land.to_crs(study.crs)

    # Clip land to study region bounding box first (speed), then intersect
    bbox = study.total_bounds          # [minx, miny, maxx, maxy]
    land_clip = land.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    print(f'  Land polygons in bbox: {len(land_clip)}')

    print('Intersecting land with study region …')
    study_union  = unary_union(study.geometry)
    land_union   = unary_union(land_clip.geometry)
    land_only    = study_union.intersection(land_union)

    land_gdf = gpd.GeoDataFrame(geometry=[land_only], crs=study.crs)
    land_gdf = land_gdf.explode(index_parts=False).reset_index(drop=True)

    # Remove tiny slivers (< 1 ha = 0.01 km²)
    land_gdf = land_gdf.to_crs('EPSG:3005')
    land_gdf = land_gdf[land_gdf.area > 10_000]  # 1 ha in m²
    land_gdf = land_gdf.to_crs(study.crs)

    total_km2 = land_gdf.to_crs('EPSG:3005').area.sum() / 1e6
    print(f'  Land-only area: {total_km2:,.0f} km² ({len(land_gdf)} parts)')

    LAND_GEOJSON.parent.mkdir(parents=True, exist_ok=True)
    land_gdf.to_file(LAND_GEOJSON, driver='GeoJSON')
    print(f'  Saved: {LAND_GEOJSON}')
    return land_gdf


def sample_points(land_gdf, n_samples=100_000, seed=42):
    """
    Sample n_samples random points within the land boundary.
    Uses vectorized shapely batch-contains for speed (~seconds vs minutes).
    """
    import numpy as np
    from shapely.ops import unary_union
    from shapely import contains_xy   # shapely >= 2.0

    rng = np.random.default_rng(seed)

    # Dissolve to a single geometry for fast contains test
    land_proj = land_gdf.to_crs('EPSG:3005')
    dissolved = unary_union(land_proj.geometry)

    minx, miny, maxx, maxy = dissolved.bounds
    print(f'Sampling {n_samples:,} points inside land boundary '
          f'({dissolved.area/1e6:,.0f} km²) …')

    collected_x = []
    collected_y = []
    batch = n_samples * 4     # oversample — expect ~25% acceptance
    iters = 0

    while len(collected_x) < n_samples:
        xs = rng.uniform(minx, maxx, batch)
        ys = rng.uniform(miny, maxy, batch)
        mask = contains_xy(dissolved, xs, ys)
        collected_x.extend(xs[mask])
        collected_y.extend(ys[mask])
        iters += 1
        if iters > 20:
            print('  Warning: low acceptance rate, stopping early')
            break

    # Trim to exactly n_samples
    collected_x = np.array(collected_x[:n_samples])
    collected_y = np.array(collected_y[:n_samples])

    # Back-project to WGS84
    from pyproj import Transformer
    t = Transformer.from_crs('EPSG:3005', 'EPSG:4326', always_xy=True)
    lons, lats = t.transform(collected_x, collected_y)

    points = list(zip(lats.tolist(), lons.tolist()))
    print(f'  Total sampled: {len(points):,}  (acceptance rate: '
          f'{len(points)/(iters*batch)*100:.1f}%)')
    return points


def build_preview_html(points, land_gdf, boundary_gdf):
    """Generate an HTML Leaflet preview map of sampled points."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    boundary_json = json.loads(boundary_gdf.to_json())
    land_json     = json.loads(land_gdf.to_json())

    # Thin points for JS (max 20k for browser performance)
    display = points
    if len(display) > 20_000:
        idx = np.random.default_rng(0).choice(len(display), 20_000, replace=False)
        display = [display[i] for i in idx]

    pts_js = ','.join(f'[{lat:.5f},{lon:.5f}]' for lat, lon in display)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>Coastal BC — Land-Only Sample Preview</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html,body,#map{{margin:0;padding:0;height:100%;width:100%;}}
  #info{{
    position:absolute;top:10px;left:50%;transform:translateX(-50%);
    z-index:1000;background:rgba(255,255,255,0.93);
    padding:7px 16px;border-radius:20px;font:13px sans-serif;
    box-shadow:0 2px 8px rgba(0,0,0,0.2);white-space:nowrap;
  }}
  #legend{{
    position:absolute;bottom:30px;right:10px;z-index:1000;
    background:rgba(255,255,255,0.93);padding:10px 14px;
    border-radius:8px;font:12px/1.7 sans-serif;
    box-shadow:0 2px 8px rgba(0,0,0,0.2);
  }}
</style>
</head>
<body>
<div id="map"></div>
<div id="info">
  Preview: {len(points):,} sample points on land-only boundary
  · showing {len(display):,}
</div>
<div id="legend">
  <b>Coastal BC — Land-Only Sampling Preview</b><br>
  <span style="color:#2563eb">■</span> Original study region boundary<br>
  <span style="color:#16a34a">■</span> Land-only boundary (NE 10m)<br>
  <span style="color:#dc2626">●</span> Sample points ({len(points):,} total)
</div>
<script>
var map = L.map('map').setView([51.5,-126.5],6);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png',
  {{attribution:'© OpenStreetMap © CARTO',maxZoom:18}}).addTo(map);

// Original boundary
L.geoJSON({json.dumps(boundary_json)},{{
  style:{{color:'#2563eb',weight:1,fill:false,opacity:0.5}}
}}).addTo(map);

// Land-only boundary
L.geoJSON({json.dumps(land_json)},{{
  style:{{color:'#16a34a',weight:1.5,fillColor:'#16a34a',fillOpacity:0.07,opacity:0.8}}
}}).addTo(map);

// Sample points
var pts = [{pts_js}];
var layer = L.layerGroup().addTo(map);
for(var i=0;i<pts.length;i++){{
  L.circleMarker(pts[i],{{radius:1.5,color:'#dc2626',fillColor:'#dc2626',
    weight:0,fillOpacity:0.6}}).addTo(layer);
}}
</script>
</body>
</html>"""

    with open(PREVIEW_HTML, 'w') as f:
        f.write(html)
    print(f'Preview map: {PREVIEW_HTML}')


def save_points_csv(points):
    """Save sampled lat/lon to CSV for use by the extraction script."""
    import csv
    POINTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(POINTS_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['lat', 'lon'])
        w.writerows(points)
    print(f'Points CSV:   {POINTS_CSV}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-samples', type=int, default=100_000)
    ap.add_argument('--seed',      type=int, default=42)
    ap.add_argument('--force',     action='store_true',
                    help='Re-build land boundary even if cached')
    args = ap.parse_args()

    if args.force and LAND_GEOJSON.exists():
        LAND_GEOJSON.unlink()

    download_ne_land()
    land_gdf    = build_land_boundary()
    study_gdf   = gpd.read_file(STUDY_REGION)

    points = sample_points(land_gdf, n_samples=args.n_samples, seed=args.seed)

    save_points_csv(points)
    build_preview_html(points, land_gdf, study_gdf)

    total_km2 = land_gdf.to_crs('EPSG:3005').area.sum() / 1e6
    print(f'\n✓ Ready to review:')
    print(f'  - Land boundary : {LAND_GEOJSON}  ({total_km2:,.0f} km²)')
    print(f'  - Sample points : {POINTS_CSV}  ({len(points):,} pts)')
    print(f'  - Preview map   : {PREVIEW_HTML}')
    print(f'\nIf the preview looks good, extract from GEE with:')
    print(f'  python scripts/prediction/sample_coastal_region.py \\')
    print(f'    --region-boundary {LAND_GEOJSON} \\')
    print(f'    --output-dir results/analysis/coastal_land_100k --scale 10')


if __name__ == '__main__':
    main()
