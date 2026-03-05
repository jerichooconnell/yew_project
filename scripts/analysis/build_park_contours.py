#!/usr/bin/env python3
"""
build_park_contours.py
─────────────────────────────────────────────────────────────────────────────
Regenerates docs/tiles/park_contours.geojson from two sources:

  1. BC provincial parks / ecological reserves / protected areas
     → BC Data Catalogue WFS (WHSE_TANTALIS.TA_PARK_ECORES_PA_SVW)
       Full province-wide coverage, 930 features.

  2. Federal / national parks not in BC Data Catalogue
     → OpenStreetMap Overpass API (targeted queries per park)
       Covers Gwaii Haanas NP Reserve, Pacific Rim NP Reserve,
       Gulf Islands NP Reserve, Yoho NP, Glacier NP, etc. in BC.

Geometry is simplified to ~10 m tolerance (0.0001°) so the resulting
file stays small for the web map but contours remain crisp at zoom 13.

Usage:
    conda run -n yew_pytorch python scripts/analysis/build_park_contours.py
"""

import json
import time
from pathlib import Path

import geopandas as gpd
import requests
from shapely.geometry import (
    MultiPolygon,
    Polygon,
    mapping,
    shape,
)
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "docs" / "tiles" / "park_contours.geojson"

# Simplify tolerance in degrees (~10 m at BC latitudes)
SIMPLIFY_TOL = 0.0001

# ── Province-wide BC parks via WFS ────────────────────────────────────────────
BC_WFS = (
    "https://openmaps.gov.bc.ca/geo/pub/wfs"
    "?service=WFS&version=2.0.0&request=GetFeature"
    "&typeName=WHSE_TANTALIS.TA_PARK_ECORES_PA_SVW"
    "&outputFormat=application/json"
    "&srsName=EPSG:4326"
    "&count=5000"
)

# ── Federal parks to fetch from OSM Overpass ──────────────────────────────────
# Each entry: (name, designation, bbox as (south, west, north, east))
FEDERAL_PARKS = [
    ("Gwaii Haanas National Park Reserve and Haida Heritage Site",
     "National Park Reserve",
     (51.5, -132.5, 53.3, -130.5)),
    ("Pacific Rim National Park Reserve of Canada",
     "National Park Reserve",
     (48.6, -126.2, 49.55, -125.0)),
    ("Gulf Islands National Park Reserve of Canada",
     "National Park Reserve",
     (48.55, -123.7, 48.95, -123.0)),
    ("Mount Revelstoke National Park of Canada",
     "National Park",
     (51.0, -118.5, 51.4, -117.8)),
    ("Glacier National Park of Canada",
     "National Park",
     (51.0, -118.5, 51.7, -117.0)),
    ("Kootenay National Park of Canada",
     "National Park",
     (50.6, -116.6, 51.4, -115.8)),
    ("Yoho National Park of Canada",
     "National Park",
     (51.1, -116.7, 51.7, -116.0)),
]

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def fetch_bc_parks():
    """Download all BC provincial parks from the BC Data Catalogue WFS."""
    print("  Fetching BC parks from WFS…")
    r = requests.get(BC_WFS, timeout=120)
    r.raise_for_status()
    feats = r.json().get("features", [])
    gdf = gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
    print(f"    → {len(gdf):,} BC features")
    return gdf


def osm_relation_to_polygon(element):
    """
    Convert an OSM relation element (from Overpass `out geom`) to a Shapely
    geometry by assembling member ways into polygon rings.
    Returns a Polygon or MultiPolygon, or None on failure.
    """
    if element.get("type") != "relation":
        return None

    outer_rings = []
    inner_rings = []

    for member in element.get("members", []):
        role = member.get("role", "outer")
        geom_pts = member.get("geometry", [])
        if len(geom_pts) < 2:
            continue
        coords = [(p["lon"], p["lat"]) for p in geom_pts]
        if len(coords) < 3:
            continue
        if role == "inner":
            inner_rings.append(coords)
        else:
            outer_rings.append(coords)

    if not outer_rings:
        return None

    polys = []
    for outer in outer_rings:
        holes = []
        # Simple heuristic: assign all inner rings (will improve if needed)
        try:
            shell = Polygon(outer)
            for inner in inner_rings:
                hole = Polygon(inner)
                if shell.contains(hole):
                    holes.append(inner)
            polys.append(Polygon(outer, holes))
        except Exception:
            continue

    if not polys:
        return None
    if len(polys) == 1:
        return polys[0]
    return MultiPolygon(polys)


def fetch_federal_park(name, designation, bbox):
    """
    Query Overpass for relations with boundary=national_park in a small bbox.
    Returns a (geojson_feature, name, designation) tuple or None.
    """
    s, w, n, e = bbox
    q = (
        f"[out:json][timeout:30];"
        f"(relation[\"boundary\"=\"national_park\"]({s},{w},{n},{e});"
        f"relation[\"protection_title\"~\"National Park\"]({s},{w},{n},{e}););"
        f"out geom;"
    )
    try:
        r = requests.post(OVERPASS_URL, data={"data": q}, timeout=35)
        if r.status_code != 200:
            print(f"    ✗ {name}: HTTP {r.status_code}")
            return None
        elems = r.json().get("elements", [])
        if not elems:
            print(f"    ✗ {name}: no OSM relations found in bbox")
            return None

        # Pick the best matching element (prefer exact name match)
        best = None
        for el in elems:
            el_name = el.get("tags", {}).get("name", "")
            if name.lower().split()[0] in el_name.lower():
                best = el
                break
        if best is None:
            best = elems[0]  # fallback to first

        geom = osm_relation_to_polygon(best)
        if geom is None or geom.is_empty:
            print(f"    ✗ {name}: geometry assembly failed")
            return None

        tags = best.get("tags", {})
        feat = {
            "type": "Feature",
            "geometry": mapping(geom),
            "properties": {
                "PROTECTED_LANDS_NAME": name,
                "PROTECTED_LANDS_DESIGNATION": "NATIONAL PARK",
                "PARK_CLASS": designation,
                "OFFICIAL_AREA_HA": tags.get("Wikipedia", ""),
                "SOURCE": "OpenStreetMap",
            },
        }
        print(f"    ✓ {name}  ({geom.geom_type}, {geom.area*1e4:.0f} km² approx)")
        return feat
    except Exception as exc:
        print(f"    ✗ {name}: {exc}")
        return None


def simplify_gdf(gdf, tol=SIMPLIFY_TOL):
    """Simplify geometry while preserving validity."""
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.simplify(tol, preserve_topology=True)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
    return gdf


def main():
    print("Building park_contours.geojson")
    print("=" * 60)

    # ── 1. BC Parks (WFS) ───────────────────────────────────────────────────
    bc_parks = fetch_bc_parks()

    # Keep only the columns we need
    keep_cols = [
        "PROTECTED_LANDS_NAME",
        "PROTECTED_LANDS_DESIGNATION",
        "PARK_CLASS",
        "OFFICIAL_AREA_HA",
        "geometry",
    ]
    bc_parks = bc_parks[[c for c in keep_cols if c in bc_parks.columns]]

    # Simplify geometry
    bc_parks = simplify_gdf(bc_parks)
    print(f"  BC parks after simplification: {len(bc_parks):,}")

    # ── 2. Federal parks (OSM) ──────────────────────────────────────────────
    print("\n  Fetching federal parks from OpenStreetMap Overpass API…")
    federal_features = []
    for name, designation, bbox in FEDERAL_PARKS:
        feat = fetch_federal_park(name, designation, bbox)
        if feat:
            federal_features.append(feat)
        time.sleep(1)  # be polite to Overpass

    print(f"\n  Federal parks fetched: {len(federal_features)}/{len(FEDERAL_PARKS)}")

    # ── 3. Combine & write GeoJSON ──────────────────────────────────────────
    print("\n  Building combined GeoJSON…")

    # Convert BC GDF to features
    bc_features = json.loads(bc_parks.to_json())["features"]

    # Merge, putting federal parks first so they render below provinces
    all_features = federal_features + bc_features

    geojson = {
        "type": "FeatureCollection",
        "features": all_features,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(geojson, f, separators=(",", ":"))

    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"\n  Written: {OUT_PATH}")
    print(f"  Total features: {len(all_features):,}")
    print(f"  File size: {size_kb:.0f} KB")

    # Vertex count check
    total_verts = 0
    for feat in all_features[:20]:
        geom = feat["geometry"]
        def count_verts(g):
            if g["type"] in ("Polygon",):
                return sum(len(r) for r in g["coordinates"])
            elif g["type"] in ("MultiPolygon",):
                return sum(len(r) for poly in g["coordinates"] for r in poly)
            return 0
        total_verts += count_verts(geom)
    print(f"  Avg vertices (first 20 features): {total_verts // 20}")


if __name__ == "__main__":
    main()
