#!/usr/bin/env python3
"""
Export fire and protected-area (park) contours from BC GDB files
as simplified GeoJSON for the Leaflet web map.

Clips to a generous coastal-BC bounding box, simplifies geometries
for small file size, and writes to docs/tiles/.

Usage:
    conda run -n yew_pytorch python scripts/visualization/export_overlays_geojson.py
"""

import json
import geopandas as gpd
from shapely.geometry import box

# ── Config ────────────────────────────────────────────────────────────────────
# Bounding box covering all of BC
STUDY_BBOX = box(-139.5, 48.2, -114.0, 60.0)   # west, south, east, north

FIRE_GDB  = 'data/PROT_HISTORICAL_FIRE_POLYS_SP.gdb'
PARKS_GDB = 'data/TA_PARK_ECORES_PA_SVW.gdb'

OUT_FIRE  = 'docs/tiles/fire_contours.geojson'
OUT_PARKS = 'docs/tiles/park_contours.geojson'

# Simplification tolerance (degrees) — ~300 m at these latitudes
SIMPLIFY_TOL = 0.003


def export_fire():
    """Export historical fire polygons clipped to study area."""
    print("Reading fire GDB...")
    fire = gpd.read_file(FIRE_GDB)
    print(f"  {len(fire)} total fire records, CRS={fire.crs}")

    # Reproject to WGS84
    fire = fire.to_crs(4326)

    # Clip to study area
    fire = fire[fire.intersects(STUDY_BBOX)].copy()
    print(f"  {len(fire)} records after spatial clip")

    # Keep only larger fires (>= 100 ha) for web display — all-BC coverage
    if 'FIRE_SIZE_HECTARES' in fire.columns:
        fire = fire[fire['FIRE_SIZE_HECTARES'] >= 100].copy()
        print(f'  {len(fire)} records after filtering >= 100 ha')

    # Simplify geometries
    fire['geometry'] = fire.geometry.simplify(SIMPLIFY_TOL, preserve_topology=True)

    # Keep only essential columns
    keep_cols = ['FIRE_YEAR', 'FIRE_CAUSE', 'FIRE_SIZE_HECTARES', 'FIRE_LABEL', 'geometry']
    keep_cols = [c for c in keep_cols if c in fire.columns]
    fire = fire[keep_cols].copy()

    # Convert types for JSON serialization
    if 'FIRE_YEAR' in fire.columns:
        fire['FIRE_YEAR'] = fire['FIRE_YEAR'].astype(int)
    if 'FIRE_SIZE_HECTARES' in fire.columns:
        fire['FIRE_SIZE_HECTARES'] = fire['FIRE_SIZE_HECTARES'].round(1)

    # Write GeoJSON with reduced coordinate precision
    fire.to_file(OUT_FIRE, driver='GeoJSON', coordinate_precision=5)
    import os
    size_mb = os.path.getsize(OUT_FIRE) / 1024 / 1024
    print(f"  ✓ Wrote {OUT_FIRE} ({size_mb:.1f} MB, {len(fire)} features)")
    return size_mb


def export_parks():
    """Export protected areas / parks clipped to study area."""
    print("\nReading parks GDB...")
    parks = gpd.read_file(PARKS_GDB)
    print(f"  {len(parks)} total park records, CRS={parks.crs}")

    # Reproject to WGS84
    parks = parks.to_crs(4326)

    # Clip to study area
    parks = parks[parks.intersects(STUDY_BBOX)].copy()
    print(f"  {len(parks)} records after spatial clip")

    # Simplify geometries
    parks['geometry'] = parks.geometry.simplify(SIMPLIFY_TOL, preserve_topology=True)

    # Keep only essential columns
    keep_cols = ['PROTECTED_LANDS_NAME', 'PROTECTED_LANDS_DESIGNATION',
                 'PARK_CLASS', 'OFFICIAL_AREA_HA', 'geometry']
    keep_cols = [c for c in keep_cols if c in parks.columns]
    parks = parks[keep_cols].copy()

    if 'OFFICIAL_AREA_HA' in parks.columns:
        parks['OFFICIAL_AREA_HA'] = parks['OFFICIAL_AREA_HA'].round(1)

    # Write GeoJSON with reduced coordinate precision
    parks.to_file(OUT_PARKS, driver='GeoJSON', coordinate_precision=5)
    import os
    size_mb = os.path.getsize(OUT_PARKS) / 1024 / 1024
    print(f"  ✓ Wrote {OUT_PARKS} ({size_mb:.1f} MB, {len(parks)} features)")
    return size_mb


if __name__ == '__main__':
    fire_mb = export_fire()
    parks_mb = export_parks()
    print(f"\nDone. Combined: {fire_mb + parks_mb:.1f} MB")
