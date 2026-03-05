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
)
from shapely.validation import make_valid

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
# For parks that are container relations (with subarea members), list explicit
# OSM relation IDs for the unit sub-relations under "unit_ids" instead of a bbox.
FEDERAL_PARKS = [
    ("Gwaii Haanas National Park Reserve and Haida Heritage Site",
     "National Park Reserve",
     (51.5, -132.5, 53.3, -130.5)),
    # Pacific Rim is a container relation — use unit sub-relation IDs directly
    # IDs: 2017388 (Broken Group), 2017389 (Long Beach), 2100172 (West Coast Trail)
    ("Pacific Rim National Park Reserve of Canada",
     "National Park Reserve",
     None,  # bbox=None signals use of unit_ids below
     [2017388, 2017389, 2100172]),
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


def build_polys_from_relation(element):
    """
    Assemble Shapely Polygons from a single OSM relation element.
    Uses make_valid to handle self-intersections; returns list of Polygons.
    """
    outer_coords, inner_coords = [], []
    for member in element.get("members", []):
        pts = member.get("geometry", [])
        if len(pts) < 3:
            continue
        coords = [(p["lon"], p["lat"]) for p in pts]
        role = member.get("role", "outer")
        (inner_coords if role == "inner" else outer_coords).append(coords)

    polys = []
    for outer in outer_coords:
        try:
            shell = Polygon(outer)
            holes = [h for h in inner_coords if shell.contains(Polygon(h))]
            p = make_valid(Polygon(outer, holes))
            if p.is_empty:
                continue
            if p.geom_type == "Polygon":
                polys.append(p)
            elif p.geom_type == "MultiPolygon":
                polys.extend(p.geoms)
        except Exception:
            continue
    return polys


def elems_to_geojson_feature(elems, name, designation):
    """
    Convert a list of OSM relation elements to a single GeoJSON Feature.
    Each element is assembled into polygons; all polygons are combined as
    a MultiPolygon (no union, so topology issues are avoided).
    Returns the Feature dict or None.
    """
    all_polys = []
    for elem in elems:
        for p in build_polys_from_relation(elem):
            s = p.simplify(SIMPLIFY_TOL, preserve_topology=True)
            if not s.is_empty and s.geom_type == "Polygon":
                all_polys.append(s)

    if not all_polys:
        return None

    geom = MultiPolygon(all_polys) if len(all_polys) > 1 else all_polys[0]
    print(f"    ✓ {name}  ({geom.geom_type}, {len(all_polys)} parts)")
    return {
        "type": "Feature",
        "geometry": mapping(geom),
        "properties": {
            "PROTECTED_LANDS_NAME": name,
            "PROTECTED_LANDS_DESIGNATION": "NATIONAL PARK",
            "PARK_CLASS": designation,
            "SOURCE": "OpenStreetMap",
        },
    }


def fetch_federal_park(name, designation, bbox, unit_ids=None):
    """
    Fetch an OSM national park and return a GeoJSON Feature.

    If unit_ids is given (list of OSM relation IDs), those sub-relations are
    fetched directly — this handles parks like Pacific Rim that are container
    relations whose sub-area members carry the actual way geometry.

    Otherwise, a bbox query is used.
    """
    if unit_ids:
        ids_str = ",".join(str(i) for i in unit_ids)
        q = f"[out:json][timeout:60];relation(id:{ids_str});out geom;"
        timeout = 65
    else:
        s, w, n, e = bbox
        q = (
            f"[out:json][timeout:40];"
            f"(relation[\"boundary\"=\"national_park\"]({s},{w},{n},{e});"
            f"relation[\"protection_title\"~\"National Park\"]({s},{w},{n},{e}););"
            f"out geom;"
        )
        timeout = 45

    try:
        r = requests.post(OVERPASS_URL, data={"data": q}, timeout=timeout)
        if r.status_code != 200:
            print(f"    ✗ {name}: HTTP {r.status_code}")
            return None
        elems = r.json().get("elements", [])
        if not elems:
            print(f"    ✗ {name}: no OSM relations found")
            return None
        if not unit_ids:
            # Pick best-matching element from bbox result
            best = next(
                (el for el in elems
                 if name.lower().split()[0] in el.get("tags", {}).get("name", "").lower()),
                elems[0]
            )
            elems = [best]
        return elems_to_geojson_feature(elems, name, designation)
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
    for entry in FEDERAL_PARKS:
        name, designation = entry[0], entry[1]
        bbox = entry[2]
        unit_ids = entry[3] if len(entry) > 3 else None
        feat = fetch_federal_park(name, designation, bbox, unit_ids)
        if feat:
            federal_features.append(feat)
        time.sleep(2)

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
