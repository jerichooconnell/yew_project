#!/usr/bin/env python3
"""
Visualize CWH and CDF BEC zones with historical wildfire polygons on an
interactive Folium map of British Columbia.

Data sources (BC Data Catalogue, Open Government Licence – BC):
  - BEC_BIOGEOCLIMATIC_POLY.gdb  → ZONE in ('CWH', 'CDF')
  - PROT_HISTORICAL_FIRE_POLYS_SP.gdb → all historical fire perimeters

Output:
  results/figures/bec_fire_map.html
"""

import warnings, time
warnings.filterwarnings("ignore")

import geopandas as gpd
import folium
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
BEC_GDB  = ROOT / "data" / "BEC_BIOGEOCLIMATIC_POLY.gdb"
FIRE_GDB = ROOT / "data" / "PROT_HISTORICAL_FIRE_POLYS_SP.gdb"
OUT_HTML = ROOT / "results" / "figures" / "bec_fire_map.html"
OUT_HTML.parent.mkdir(parents=True, exist_ok=True)

# ── 1. Load BEC zones (CWH + CDF only) ──────────────────────────────────
print("Loading BEC zones …")
t0 = time.time()
bec = gpd.read_file(
    BEC_GDB,
    columns=["ZONE", "ZONE_NAME", "SUBZONE", "MAP_LABEL"],
)
bec = bec[bec["ZONE"].isin(["CWH", "CDF"])].copy()
print(f"  Loaded {len(bec)} CWH/CDF polygons in {time.time()-t0:.1f}s")

# Fix any invalid geometries (e.g. CWHvh2, CWHvm2 covering Haida Gwaii / coast)
from shapely.validation import make_valid
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

def repair_geometry(geom):
    """Repair invalid geometry and ensure it stays a Polygon/MultiPolygon."""
    if not geom.is_valid:
        geom = make_valid(geom)
    # make_valid can produce GeometryCollections; extract only polygon parts
    if geom.geom_type == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        if polys:
            geom = unary_union(polys)
        else:
            geom = Polygon()  # fallback empty
    return geom

n_invalid = (~bec.geometry.is_valid).sum()
bec["geometry"] = bec.geometry.apply(repair_geometry)
print(f"  Repaired {n_invalid} invalid geometries")

# Simplify geometry for faster rendering (tolerance in metres, CRS is EPSG:3005)
bec["geometry"] = bec.geometry.simplify(tolerance=200, preserve_topology=True)
# Reproject to WGS-84 for Folium
bec = bec.to_crs(epsg=4326)

# ── 2. Load historical fire perimeters ───────────────────────────────────
print("Loading historical fire perimeters …")
t0 = time.time()
fires = gpd.read_file(
    FIRE_GDB,
    columns=["FIRE_YEAR", "FIRE_CAUSE", "FIRE_SIZE_HECTARES", "FIRE_LABEL"],
)
print(f"  Loaded {len(fires)} fire polygons in {time.time()-t0:.1f}s")

# Simplify for rendering
fires["geometry"] = fires.geometry.simplify(tolerance=200, preserve_topology=True)
fires = fires.to_crs(epsg=4326)

# ── 3. Build Folium map ─────────────────────────────────────────────────
print("Building map …")

# Centre on BC
BC_CENTRE = [54.0, -125.0]
m = folium.Map(location=BC_CENTRE, zoom_start=6, tiles="cartodbpositron")

# --- CWH layer (green) ---
cwh = bec[bec["ZONE"] == "CWH"]
cwh_fg = folium.FeatureGroup(name="Coastal Western Hemlock (CWH)", show=True)
folium.GeoJson(
    cwh,
    style_function=lambda f: {
        "fillColor": "#228B22",
        "color": "#228B22",
        "weight": 0.5,
        "fillOpacity": 0.35,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["ZONE_NAME", "MAP_LABEL"],
        aliases=["Zone:", "Subzone:"],
    ),
).add_to(cwh_fg)
cwh_fg.add_to(m)

# --- CDF layer (teal) ---
cdf = bec[bec["ZONE"] == "CDF"]
cdf_fg = folium.FeatureGroup(name="Coastal Douglas-fir (CDF)", show=True)
folium.GeoJson(
    cdf,
    style_function=lambda f: {
        "fillColor": "#008080",
        "color": "#008080",
        "weight": 0.5,
        "fillOpacity": 0.35,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["ZONE_NAME", "MAP_LABEL"],
        aliases=["Zone:", "Subzone:"],
    ),
).add_to(cdf_fg)
cdf_fg.add_to(m)

# --- Fire perimeters (red/orange) ---
fire_fg = folium.FeatureGroup(name="Historical Wildfire Perimeters", show=True)
folium.GeoJson(
    fires,
    style_function=lambda f: {
        "fillColor": "#FF4500",
        "color": "#CC3300",
        "weight": 0.4,
        "fillOpacity": 0.40,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["FIRE_LABEL", "FIRE_YEAR", "FIRE_CAUSE", "FIRE_SIZE_HECTARES"],
        aliases=["Fire:", "Year:", "Cause:", "Size (ha):"],
    ),
).add_to(fire_fg)
fire_fg.add_to(m)

# --- Layer control ---
folium.LayerControl(collapsed=False).add_to(m)

# ── 4. Save ──────────────────────────────────────────────────────────────
m.save(str(OUT_HTML))
print(f"\nMap saved → {OUT_HTML}")
print("Open the HTML file in a browser to explore interactively.")
