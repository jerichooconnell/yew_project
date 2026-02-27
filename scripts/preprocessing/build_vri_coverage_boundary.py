#!/usr/bin/env python3
"""
Build a VRI-coverage boundary for the CWH zone.

Checks each 0.25-degree tile within the CWH extent for the presence of VRI
polygons (fast spatial index bbox query), then intersects the resulting
coverage grid with the existing CWH forestry boundary.

Output: data/processed/cwh_negatives/vri_coverage_boundary.gpkg
"""

import os
os.environ.setdefault('PROJ_DATA',
    '/home/jericho/anaconda3/envs/yew_pytorch/share/proj')

import fiona
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from pyproj import Transformer
from pathlib import Path

GDB = '/home/jericho/Downloads/firefox-downloads/VEG_COMP_LYR_R1_POLY_2024.gdb'
OUT = 'data/processed/cwh_negatives/vri_coverage_boundary.gpkg'
CWH_BOUNDARY = 'data/processed/cwh_negatives/cwh_boundary_forestry.gpkg'
TILE_DEG = 0.5   # 0.5° (~50 km) — reduces total checks from 1700 to ~425

# Bounding box slightly larger than the CWH zone
LAT_MIN, LAT_MAX = 48.0, 56.5
LON_MIN, LON_MAX = -133.0, -120.5


def main():
    t = Transformer.from_crs('EPSG:4326', 'EPSG:3005', always_xy=True)
    lats = np.arange(LAT_MIN, LAT_MAX, TILE_DEG)
    lons = np.arange(LON_MIN, LON_MAX, TILE_DEG)
    total = len(lats) * len(lons)
    print(f"Checking {total} tiles ({TILE_DEG} deg) for VRI coverage ...")

    covered = []
    with fiona.open(GDB, layer='VEG_COMP_LYR_R1_POLY') as src:
        for i, lat in enumerate(lats):
            for lon in lons:
                x0, y0 = t.transform(lon,             lat)
                x1, y1 = t.transform(lon + TILE_DEG,  lat + TILE_DEG)
                # Stop at first hit — we only need to know if ANY polygon exists
                has_data = next(iter(src.items(bbox=(x0, y0, x1, y1))), None) is not None
                if has_data:
                    covered.append(box(lon, lat, lon + TILE_DEG, lat + TILE_DEG))
            if (i + 1) % 5 == 0:
                done = (i + 1) * len(lons)
                print(f"  {done}/{total} ({100*done/total:.0f}%),"
                      f" covered so far: {len(covered)}")

    print(f"\nCovered tiles: {len(covered)} / {total}")

    vri_gdf = gpd.GeoDataFrame(geometry=covered, crs='EPSG:4326').dissolve()
    area_ha = vri_gdf.to_crs('EPSG:3005').area.sum() / 10000
    print(f"VRI coverage area (grid): {area_ha:,.0f} ha")

    # Intersect with existing CWH forestry boundary
    cwh = gpd.read_file(CWH_BOUNDARY).to_crs('EPSG:4326')
    clipped = gpd.overlay(vri_gdf.reset_index(), cwh, how='intersection')
    area_clipped = clipped.to_crs('EPSG:3005').area.sum() / 10000
    print(f"VRI coverage intersected with CWH boundary: {area_clipped:,.0f} ha")
    print(f"(Original CWH boundary: ~3,595,194 ha)")

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    clipped.to_file(OUT, driver='GPKG')
    print(f"Saved: {OUT}")


if __name__ == '__main__':
    main()
