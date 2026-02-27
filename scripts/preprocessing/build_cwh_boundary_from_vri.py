#!/usr/bin/env python3
"""
Build CWH BEC zone boundary directly from VRI polygon attributes.

Reads all VRI polygons where BEC_ZONE_CODE='CWH' and saves as GeoPackage.
Computes total area by summing individual polygon areas (no slow dissolve).

This is more accurate than the previous FAIB-derived convex hull approach,
because it uses the actual VRI polygon classification of biogeoclimatic zones.

Usage:
    python scripts/preprocessing/build_cwh_boundary_from_vri.py
"""

import os
os.environ.setdefault('PROJ_DATA',
    '/home/jericho/anaconda3/envs/yew_pytorch/share/proj')

import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np

GDB_PATH = '/home/jericho/Downloads/firefox-downloads/VEG_COMP_LYR_R1_POLY_2024.gdb'
LAYER = 'VEG_COMP_LYR_R1_POLY'
OUTPUT = 'data/processed/cwh_boundary_vri.gpkg'


def main():
    out_path = Path(OUTPUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BUILD CWH BOUNDARY FROM VRI BEC_ZONE_CODE")
    print("=" * 60)
    print(f"GDB: {GDB_PATH}")
    print(f"Reading all polygons where BEC_ZONE_CODE = 'CWH'...")

    t0 = time.time()

    # Use SQL WHERE filter — reads only CWH polygons from the 6.8M total
    gdf = gpd.read_file(
        GDB_PATH,
        layer=LAYER,
        where="BEC_ZONE_CODE = 'CWH'",
        columns=['BEC_ZONE_CODE', 'BEC_SUBZONE'],
    )

    elapsed_read = time.time() - t0
    print(f"  Read {len(gdf):,} CWH polygons in {elapsed_read:.0f}s")
    print(f"  CRS: {gdf.crs}")

    # -------------------------------------------------------------------------
    # Compute area by summing individual polygon areas (EPSG:3005 = metres)
    # This avoids the very slow unary_union / dissolve step entirely.
    # -------------------------------------------------------------------------
    print("  Computing total area from individual polygon areas...")
    area_ha = gdf.geometry.area.sum() / 10000
    print(f"\nCWH zone area (VRI-derived): {area_ha:,.0f} ha ({area_ha/1e6:.2f} million ha)")

    # Add area column per polygon
    gdf['area_ha'] = gdf.geometry.area / 10000

    # Save in native EPSG:3005 (fast, no reprojection)
    out_3005 = out_path.with_name('cwh_boundary_vri_3005.gpkg')
    print(f"  Saving {len(gdf):,} polygons to {out_3005}...")
    gdf.to_file(out_3005, driver='GPKG')
    print(f"  Saved (EPSG:3005): {out_3005}")

    # Convert to WGS84 and save
    print(f"  Reprojecting to EPSG:4326...")
    gdf_4326 = gdf.to_crs('EPSG:4326')
    print(f"  Saving {len(gdf_4326):,} polygons to {out_path}...")
    gdf_4326.to_file(out_path, driver='GPKG')
    print(f"  Saved (EPSG:4326): {out_path}")

    bounds = gdf_4326.total_bounds
    print(f"\nBounds (WGS84): lon [{bounds[0]:.2f}, {bounds[2]:.2f}], "
          f"lat [{bounds[1]:.2f}, {bounds[3]:.2f}]")

    # Save metadata JSON (area + bounds) for quick loading by other scripts
    meta = {
        'bec_zone': 'CWH',
        'n_polygons': len(gdf),
        'area_ha': float(area_ha),
        'bounds_4326': {
            'lon_min': float(bounds[0]), 'lat_min': float(bounds[1]),
            'lon_max': float(bounds[2]), 'lat_max': float(bounds[3]),
        },
        'source': 'VRI VEG_COMP_LYR_R1_POLY_2024.gdb, BEC_ZONE_CODE=CWH',
        'gpkg_4326': str(out_path),
        'gpkg_3005': str(out_3005),
    }
    meta_path = out_path.with_suffix('.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved metadata: {meta_path}")

    # Subzone breakdown
    sub_stats = (gdf.groupby('BEC_SUBZONE')
                 .agg(n_polys=('area_ha', 'size'), total_ha=('area_ha', 'sum'))
                 .sort_values('total_ha', ascending=False))
    print(f"\nCWH subzone breakdown:")
    print(f"  {'Subzone':>8} {'Polygons':>10} {'Area (ha)':>14}")
    for sz, row in sub_stats.head(15).iterrows():
        print(f"  {str(sz):>8} {row['n_polys']:>10,} {row['total_ha']:>14,.0f}")

    elapsed_total = time.time() - t0
    print(f"\nTotal time: {elapsed_total:.0f}s")


if __name__ == '__main__':
    main()
