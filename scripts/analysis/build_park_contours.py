#!/usr/bin/env python3
"""
build_park_contours.py
─────────────────────────────────────────────────────────────────────────────
Regenerates docs/tiles/park_contours.geojson at high resolution from three
authoritative BC Data Catalogue WFS layers (all returned full-resolution,
EPSG:4326):

  1. Provincial parks / ecological reserves / protected areas / recreation areas
     → WHSE_TANTALIS.TA_PARK_ECORES_PA_SVW          (~930 features)
  2. Conservancies
     → WHSE_TANTALIS.TA_CONSERVANCY_AREAS_SVW        (~169 features)
  3. National parks (federal boundaries within BC)
     → WHSE_ADMIN_BOUNDARIES.CLAB_NATIONAL_PARKS     (~102 polygons)

The previous version of this layer was sourced from a generalised WFS response
(and OSM Overpass for national parks) and simplified to ~11 m, leaving ~38
vertices/feature — visibly blocky. These layers are now pulled at full
resolution (~1,300 vertices/feature for parks) and simplified with a finer,
topology-preserving tolerance so contours stay crisp while the file remains
web-friendly.

Usage:
    conda run -n yew_pytorch python scripts/analysis/build_park_contours.py
    # optional: --tol 0.00005   (simplify tolerance in degrees, ~5.5 m)
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "docs" / "tiles" / "park_contours.geojson"

# Default simplify tolerance in degrees (~17 m at BC latitudes). ~1–2 px at the
# map's working zoom — far crisper than the previous ~38-vertex/feature layer —
# while keeping the file ~8 MB (≈2.4 MB gzipped, as served by GitHub Pages).
DEFAULT_TOL = 0.00015

WFS_BASE = (
    "https://openmaps.gov.bc.ca/geo/pub/wfs"
    "?service=WFS&version=2.0.0&request=GetFeature"
    "&outputFormat=application/json&srsName=EPSG:4326&count=10000"
)

# (typeName, name_field, designation, class_field, area_field, area_is_sqm)
SOURCES = [
    ("WHSE_TANTALIS.TA_PARK_ECORES_PA_SVW",
     "PROTECTED_LANDS_NAME", None, "PARK_CLASS", "OFFICIAL_AREA_HA", False),
    ("WHSE_TANTALIS.TA_CONSERVANCY_AREAS_SVW",
     "CONSERVANCY_AREA_NAME", "CONSERVANCY", None, "OFFICIAL_AREA_HA", False),
    ("WHSE_ADMIN_BOUNDARIES.CLAB_NATIONAL_PARKS",
     "ENGLISH_NAME", "NATIONAL PARK", None, "FEATURE_AREA_SQM", True),
]


def fetch_layer(type_name):
    print(f"  Fetching {type_name} …")
    r = requests.get(f"{WFS_BASE}&typeName={type_name}", timeout=240)
    r.raise_for_status()
    feats = r.json().get("features", [])
    if not feats:
        print(f"    ⚠ no features returned for {type_name}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    gdf = gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
    print(f"    → {len(gdf):,} features")
    return gdf


def normalise(gdf, name_field, designation, class_field, area_field, area_is_sqm):
    """Map a source layer onto the common park_contours property schema."""
    out = gpd.GeoDataFrame(index=gdf.index, geometry=gdf.geometry, crs=gdf.crs)
    out["PROTECTED_LANDS_NAME"] = gdf.get(name_field)
    if designation is not None:
        out["PROTECTED_LANDS_DESIGNATION"] = designation
    else:
        out["PROTECTED_LANDS_DESIGNATION"] = gdf.get("PROTECTED_LANDS_DESIGNATION")
    out["PARK_CLASS"] = gdf.get(class_field) if class_field else None
    if area_field and area_field in gdf:
        area = gdf[area_field].astype("float64")
        out["OFFICIAL_AREA_HA"] = (area / 10_000.0) if area_is_sqm else area
    else:
        out["OFFICIAL_AREA_HA"] = None
    out["SOURCE"] = "BC Data Catalogue"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL,
                    help="simplify tolerance in degrees (default %.5f)" % DEFAULT_TOL)
    args = ap.parse_args()

    print("Building park_contours.geojson (high resolution)")
    print("=" * 60)

    parts = []
    for type_name, *cfg in SOURCES:
        gdf = fetch_layer(type_name)
        if len(gdf):
            parts.append(normalise(gdf, *cfg))

    merged = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:4326")

    # Topology-preserving simplification
    merged["geometry"] = merged.geometry.simplify(args.tol, preserve_topology=True)
    merged = merged[~merged.geometry.is_empty & merged.geometry.notna()]

    geojson = json.loads(merged.to_json())
    with open(OUT_PATH, "w") as f:
        json.dump(geojson, f, separators=(",", ":"))

    # Report
    desig = Counter(ft["properties"]["PROTECTED_LANDS_DESIGNATION"]
                    for ft in geojson["features"])

    def verts(g):
        def walk(x):
            return 1 if isinstance(x[0], (int, float)) else sum(walk(i) for i in x)
        return walk(g["coordinates"])
    tv = sum(verts(ft["geometry"]) for ft in geojson["features"])

    print("\n  Written:", OUT_PATH)
    print(f"  Features: {len(geojson['features']):,}  |  vertices: {tv:,}"
          f"  ({tv // max(len(geojson['features']), 1)}/feature)")
    print(f"  File size: {OUT_PATH.stat().st_size / 1024:.0f} KB  (tol={args.tol})")
    print("  Designations:", dict(desig))


if __name__ == "__main__":
    main()
