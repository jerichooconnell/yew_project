#!/usr/bin/env python3
"""
1. Fill the Haida Gwaii BEC gap: land areas on Haida Gwaii not covered by any
   BEC polygon are added as CWHvh3 (Very Wet Hypermaritime, the dominant zone
   for that island group).
2. Clip the full CWH + CDF dataset to BC land (removing open-water areas).
3. Save as a Shapefile for downstream use.
4. Regenerate the interactive Folium map.

Outputs:
  data/processed/cwh_cdf_land.shp  (+ .dbf .shx .prj)
  results/figures/bec_fire_map.html
"""

import warnings, time
warnings.filterwarnings("ignore")

import geopandas as gpd
import pandas as pd
from pathlib import Path
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.validation import make_valid
from shapely.ops import unary_union
import folium

# ── paths ─────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
BEC_GDB   = ROOT / "data" / "BEC_BIOGEOCLIMATIC_POLY.gdb"
FIRE_GDB  = ROOT / "data" / "PROT_HISTORICAL_FIRE_POLYS_SP.gdb"
# High-resolution Natural Earth 10m land polygons (443k vertices vs 3.7k in simplified)
NE_LAND   = ROOT / "data" / "lookup_tables" / "ne_10m_land" / "ne_10m_land.shp"
# Fallback simplified boundary (used only for the gap-fill step)
BC_BOUND  = ROOT / "results" / "analysis" / "cwh_yew_population_100k" / "bc_boundary_simplified.geojson"
OUT_SHP   = ROOT / "data" / "processed" / "cwh_cdf_land.shp"
OUT_HTML  = ROOT / "results" / "figures" / "bec_fire_map.html"
OUT_SHP.parent.mkdir(parents=True, exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────
def repair(geom):
    """Make valid and ensure Polygon/MultiPolygon output."""
    if not geom.is_valid:
        geom = make_valid(geom)
    if geom.geom_type == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        geom = unary_union(polys) if polys else Polygon()
    return geom

# ── 1. Load & repair BEC polygons (CWH + CDF) ────────────────────────────
print("Loading BEC data …")
t0 = time.time()
bec = gpd.read_file(BEC_GDB, columns=["ZONE", "ZONE_NAME", "SUBZONE",
                                       "MAP_LABEL", "BGC_LABEL"])
bec = bec[bec["ZONE"].isin(["CWH", "CDF"])].copy()
bec["geometry"] = bec.geometry.apply(repair)
print(f"  {len(bec)} CWH/CDF polygons loaded & repaired ({time.time()-t0:.1f}s)")

# ── 2. Load BC land boundary (Natural Earth 10m, clipped to BC extent) ───
print("Loading Natural Earth 10m land, clipping to BC …")
# BC bounding box in WGS-84 (add a small buffer to capture coastal islands)
bc_bbox = box(-141.0, 47.5, -113.5, 60.5)
ne_land = gpd.read_file(NE_LAND)
bc_land = ne_land[ne_land.geometry.intersects(bc_bbox)].copy()
bc_land["geometry"] = bc_land.geometry.apply(
    lambda g: repair(g.intersection(bc_bbox))
)
bc_land = bc_land[~bc_land.geometry.is_empty].to_crs(epsg=3005)
bc_geom = repair(unary_union(bc_land.geometry.tolist()))
print(f"  {len(bc_land)} NE 10m land feature(s) covering BC")

# ── 3. Fill the Haida Gwaii BEC gap ──────────────────────────────────────
# Haida Gwaii bounding box in EPSG:3005
HG_BOX = box(490000, 700000, 720000, 1160000)

print("Finding unclassified land on Haida Gwaii …")
# Land on Haida Gwaii
hg_land = bc_geom.intersection(HG_BOX)

# All BEC polygons that intersect Haida Gwaii
bec_hg_union = unary_union(
    bec[bec.geometry.intersects(HG_BOX)].geometry.tolist()
)

# Land not covered by any BEC polygon
hg_gap = hg_land.difference(bec_hg_union)
hg_gap = repair(hg_gap)

if not hg_gap.is_empty:
    total_gap_km2 = hg_gap.area / 1e6
    print(f"  Gap area on Haida Gwaii: {total_gap_km2:.1f} km²")

    # Build a new row matching BEC schema, classified as CWHvh3
    gap_row = pd.DataFrame([{
        "ZONE":       "CWH",
        "ZONE_NAME":  "Coastal Western Hemlock",
        "SUBZONE":    "vh",
        "MAP_LABEL":  "CWHvh3",
        "BGC_LABEL":  "CWHvh3",
        "geometry":   hg_gap,
    }])
    gap_gdf = gpd.GeoDataFrame(gap_row, geometry="geometry", crs="EPSG:3005")
    bec = pd.concat([bec, gap_gdf], ignore_index=True)
    print(f"  Added Haida Gwaii gap as CWHvh3 ({total_gap_km2:.1f} km²)")
else:
    print("  No gap found (already fully covered).")

# ── 4. Clip entire CWH+CDF by BC land boundary (remove water) ────────────
print("Clipping CWH/CDF to BC land boundary (removing water) …")
t0 = time.time()
bec_land = gpd.clip(bec, bc_land)
bec_land = bec_land[~bec_land.geometry.is_empty].copy()
print(f"  {len(bec_land)} polygons after land clip ({time.time()-t0:.1f}s)")

# ── 5. Save shapefile ─────────────────────────────────────────────────────
print(f"Saving shapefile → {OUT_SHP}")
bec_land.to_file(OUT_SHP, driver="ESRI Shapefile")
print(f"  Saved ({OUT_SHP.stat().st_size/1024:.0f} KB)")

# ── 6. Regenerate the Folium map ──────────────────────────────────────────
print("Building map …")
t0 = time.time()

# Simplify for web rendering
bec_map = bec_land.copy()
bec_map["geometry"] = bec_map.geometry.simplify(tolerance=200, preserve_topology=True)
bec_map = bec_map.to_crs(epsg=4326)

fires = gpd.read_file(FIRE_GDB, columns=["FIRE_YEAR", "FIRE_CAUSE",
                                          "FIRE_SIZE_HECTARES", "FIRE_LABEL"])
fires["geometry"] = fires.geometry.simplify(tolerance=200, preserve_topology=True)
fires = fires.to_crs(epsg=4326)

m = folium.Map(location=[54.0, -125.0], zoom_start=6, tiles="cartodbpositron")

# CWH layer
cwh_fg = folium.FeatureGroup(name="Coastal Western Hemlock (CWH)", show=True)
folium.GeoJson(
    bec_map[bec_map["ZONE"] == "CWH"],
    style_function=lambda f: {"fillColor": "#228B22", "color": "#228B22",
                               "weight": 0.5, "fillOpacity": 0.35},
    tooltip=folium.GeoJsonTooltip(
        fields=["ZONE_NAME", "MAP_LABEL"],
        aliases=["Zone:", "Subzone:"],
    ),
).add_to(cwh_fg)
cwh_fg.add_to(m)

# CDF layer
cdf_fg = folium.FeatureGroup(name="Coastal Douglas-fir (CDF)", show=True)
folium.GeoJson(
    bec_map[bec_map["ZONE"] == "CDF"],
    style_function=lambda f: {"fillColor": "#008080", "color": "#008080",
                               "weight": 0.5, "fillOpacity": 0.35},
    tooltip=folium.GeoJsonTooltip(
        fields=["ZONE_NAME", "MAP_LABEL"],
        aliases=["Zone:", "Subzone:"],
    ),
).add_to(cdf_fg)
cdf_fg.add_to(m)

# Fire perimeters
fire_fg = folium.FeatureGroup(name="Historical Wildfire Perimeters", show=True)
folium.GeoJson(
    fires,
    style_function=lambda f: {"fillColor": "#FF4500", "color": "#CC3300",
                               "weight": 0.4, "fillOpacity": 0.40},
    tooltip=folium.GeoJsonTooltip(
        fields=["FIRE_LABEL", "FIRE_YEAR", "FIRE_CAUSE", "FIRE_SIZE_HECTARES"],
        aliases=["Fire:", "Year:", "Cause:", "Size (ha):"],
    ),
).add_to(fire_fg)
fire_fg.add_to(m)

folium.LayerControl(collapsed=False).add_to(m)
m.save(str(OUT_HTML))
print(f"  Map saved ({time.time()-t0:.1f}s) → {OUT_HTML}")
print("\nDone.")
