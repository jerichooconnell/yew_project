#!/usr/bin/env python3
"""
Create an interactive Folium map showing:
  - Training samples (positive / negative, coloured by split)
  - Validation samples
  - The 35 analysis tiles as 10×10 km boxes
  - iNaturalist BC yew observations (optional layer)
"""

import csv, json, pathlib, math
import folium
from folium.plugins import MarkerCluster

ROOT = pathlib.Path(__file__).resolve().parents[2]

# ── helpers ──────────────────────────────────────────────────────────────────

def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def tile_bounds(lat, lon, size_km=10):
    """Return SW/NE corners for a square tile centred on (lat, lon)."""
    half_lat = (size_km / 2) / 111.0
    half_lon = (size_km / 2) / (111.0 * math.cos(math.radians(lat)))
    return [[lat - half_lat, lon - half_lon],
            [lat + half_lat, lon + half_lon]]


# ── load data ────────────────────────────────────────────────────────────────

train = load_csv(ROOT / "data/processed/train_split_balanced_max.csv")
val   = load_csv(ROOT / "data/processed/val_split_balanced_max.csv")

spot_stats_path = ROOT / "results/analysis/cwh_spot_comparisons/spot_stats.json"
tiles = json.loads(spot_stats_path.read_text()) if spot_stats_path.exists() else []

inat_path = ROOT / "data/inat_observations/observations-558049.csv"
inat = load_csv(inat_path) if inat_path.exists() else []

# ── build map ────────────────────────────────────────────────────────────────

centre = [50.5, -125.5]
m = folium.Map(location=centre, zoom_start=6, tiles="OpenStreetMap")
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    name="Satellite",
).add_to(m)

# --- Layer: 35 analysis tiles (10×10 km boxes) ---
tile_layer = folium.FeatureGroup(name="Analysis tiles (35 × 10 km)", show=True)
for t in tiles:
    bounds = tile_bounds(t["lat"], t["lon"])
    folium.Rectangle(
        bounds=bounds,
        color="#2222ff",
        weight=2,
        fill=True,
        fill_color="#2222ff",
        fill_opacity=0.10,
        popup=f"<b>{t['name']}</b><br>{t['lat']:.2f}°N, {t['lon']:.2f}°W",
    ).add_to(tile_layer)
tile_layer.add_to(m)

# --- Layer: Training positives ---
train_pos_layer = folium.FeatureGroup(name=f"Train positive (yew) — {sum(1 for r in train if r['has_yew']=='True')}", show=True)
for r in train:
    if r["has_yew"] != "True":
        continue
    try:
        lat, lon = float(r["lat"]), float(r["lon"])
    except (ValueError, KeyError):
        continue
    folium.CircleMarker(
        location=[lat, lon],
        radius=4,
        color="#00aa00",
        fill=True,
        fill_color="#00ee00",
        fill_opacity=0.7,
        weight=1,
        popup=f"TRAIN +yew<br>{lat:.3f}, {lon:.3f}<br>src: {r.get('source','')}",
    ).add_to(train_pos_layer)
train_pos_layer.add_to(m)

# --- Layer: Training negatives ---
train_neg_layer = folium.FeatureGroup(name=f"Train negative (no yew) — {sum(1 for r in train if r['has_yew']=='False')}", show=True)
for r in train:
    if r["has_yew"] != "False":
        continue
    try:
        lat, lon = float(r["lat"]), float(r["lon"])
    except (ValueError, KeyError):
        continue
    folium.CircleMarker(
        location=[lat, lon],
        radius=3,
        color="#cc0000",
        fill=True,
        fill_color="#ff4444",
        fill_opacity=0.5,
        weight=1,
        popup=f"TRAIN −yew<br>{lat:.3f}, {lon:.3f}<br>src: {r.get('source','')}",
    ).add_to(train_neg_layer)
train_neg_layer.add_to(m)

# --- Layer: Validation positives ---
val_pos_layer = folium.FeatureGroup(name=f"Val positive (yew) — {sum(1 for r in val if r['has_yew']=='True')}", show=True)
for r in val:
    if r["has_yew"] != "True":
        continue
    try:
        lat, lon = float(r["lat"]), float(r["lon"])
    except (ValueError, KeyError):
        continue
    folium.CircleMarker(
        location=[lat, lon],
        radius=4,
        color="#006600",
        fill=True,
        fill_color="#00cc00",
        fill_opacity=0.7,
        weight=1,
        popup=f"VAL +yew<br>{lat:.3f}, {lon:.3f}<br>src: {r.get('source','')}",
    ).add_to(val_pos_layer)
val_pos_layer.add_to(m)

# --- Layer: Validation negatives ---
val_neg_layer = folium.FeatureGroup(name=f"Val negative (no yew) — {sum(1 for r in val if r['has_yew']=='False')}", show=True)
for r in val:
    if r["has_yew"] != "False":
        continue
    try:
        lat, lon = float(r["lat"]), float(r["lon"])
    except (ValueError, KeyError):
        continue
    folium.CircleMarker(
        location=[lat, lon],
        radius=3,
        color="#880000",
        fill=True,
        fill_color="#cc3333",
        fill_opacity=0.5,
        weight=1,
        popup=f"VAL −yew<br>{lat:.3f}, {lon:.3f}<br>src: {r.get('source','')}",
    ).add_to(val_neg_layer)
val_neg_layer.add_to(m)

# --- Layer: ALL iNaturalist observations (clustered) ---
all_inat_with_coords = []
for r in inat:
    try:
        lat, lon = float(r["latitude"]), float(r["longitude"])
        all_inat_with_coords.append((lat, lon, r))
    except (ValueError, KeyError):
        continue

# Split into BC vs non-BC
bc_inat = [(lat, lon, r) for lat, lon, r in all_inat_with_coords
           if r.get("place_state_name") == "British Columbia"]
non_bc_inat = [(lat, lon, r) for lat, lon, r in all_inat_with_coords
               if r.get("place_state_name") != "British Columbia"]

# BC layer
inat_bc_cluster = MarkerCluster(name=f"iNat BC yew — {len(bc_inat)}", show=True)
for lat, lon, r in bc_inat:
    folium.CircleMarker(
        location=[lat, lon],
        radius=3,
        color="#ff8800",
        fill=True,
        fill_color="#ffaa00",
        fill_opacity=0.7,
        weight=1,
        popup=(f"iNat BC<br>{lat:.3f}, {lon:.3f}"
               f"<br>{r.get('place_guess','')}"
               f"<br>{r.get('observed_on','')}"),
    ).add_to(inat_bc_cluster)
inat_bc_cluster.add_to(m)

# Non-BC layer (WA, OR, CA, ID, MT, etc.)
inat_other_cluster = MarkerCluster(name=f"iNat outside BC — {len(non_bc_inat)}", show=True)
for lat, lon, r in non_bc_inat:
    folium.CircleMarker(
        location=[lat, lon],
        radius=3,
        color="#9944cc",
        fill=True,
        fill_color="#bb66ee",
        fill_opacity=0.6,
        weight=1,
        popup=(f"iNat {r.get('place_state_name','?')}<br>{lat:.3f}, {lon:.3f}"
               f"<br>{r.get('place_guess','')}"
               f"<br>{r.get('observed_on','')}"),
    ).add_to(inat_other_cluster)
inat_other_cluster.add_to(m)

# --- Layer control ---
folium.LayerControl(collapsed=False).add_to(m)

# --- Legend ---
legend_html = f"""
<div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
     background: white; padding: 12px 16px; border: 2px solid #888;
     border-radius: 6px; font-size: 13px; line-height: 1.6;">
  <b>Training Samples</b><br>
  <span style="color:#00ee00;">&#9679;</span> Train positive (yew)<br>
  <span style="color:#ff4444;">&#9679;</span> Train negative<br>
  <span style="color:#00cc00;">&#9679;</span> Val positive (yew)<br>
  <span style="color:#cc3333;">&#9679;</span> Val negative<br>
  <b>iNaturalist</b><br>
  <span style="color:#ffaa00;">&#9679;</span> iNat BC ({len(bc_inat):,})<br>
  <span style="color:#bb66ee;">&#9679;</span> iNat other ({len(non_bc_inat):,})<br>
  <span style="color:#2222ff;">&#9632;</span> Analysis tile (10 km)
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# ── save ─────────────────────────────────────────────────────────────────────

out = ROOT / "results/figures/training_samples_map.html"
out.parent.mkdir(parents=True, exist_ok=True)
m.save(str(out))
print(f"Saved: {out}")
print(f"  Train: {len(train)} ({sum(1 for r in train if r['has_yew']=='True')} pos, "
      f"{sum(1 for r in train if r['has_yew']=='False')} neg)")
print(f"  Val:   {len(val)} ({sum(1 for r in val if r['has_yew']=='True')} pos, "
      f"{sum(1 for r in val if r['has_yew']=='False')} neg)")
print(f"  Tiles: {len(tiles)}")
print(f"  iNat BC: {len(bc_inat)}")
print(f"  iNat other: {len(non_bc_inat)}")
print(f"  iNat total: {len(all_inat_with_coords)}")
