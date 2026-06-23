#!/usr/bin/env python3
"""
protected_area_from_tiles.py
────────────────────────────
Recompute the fraction of mapped Pacific-yew habitat inside protected areas
across ALL study tiles, using the published per-tile probability PNGs as the
habitat source (raw .npy grids survive for only 42 of the 99 tiles, but the
suppressed-probability PNGs exist for all 99 and are losslessly invertible via
the export colormap — validated to within ~2% of the stored p50_ha).

Habitat = suppressed yew probability ≥ 0.5 (the same surface shown on the web
map). For each tile the habitat mask is recovered from docs/tiles/<slug>.png,
the protected-area layer (docs/tiles/park_contours.geojson) is rasterised onto
the tile's pixel grid using its bbox, and pixels are tallied overall and per
designation. Protected *fraction* is a pixel ratio and is independent of the
per-pixel area constant; areas use the export's 0.01 ha/px convention.

Usage:
    conda run -n yew_pytorch python scripts/analysis/protected_area_from_tiles.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import geopandas as gpd
import rasterio.features
from rasterio.transform import from_bounds as rio_bounds
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]
TILES_JSON = ROOT / "docs" / "tiles" / "tiles.json"
TILES_DIR = ROOT / "docs" / "tiles"
PARKS = ROOT / "docs" / "tiles" / "park_contours.geojson"

HA_PER_PX = 0.01          # export convention (10 m pixel); cancels in fractions
THRESH = 0.50

# Provincial designations that the paper's original 4.6% figure counted
PROVINCIAL = {"PROVINCIAL PARK", "ECOLOGICAL RESERVE", "PROTECTED AREA",
              "RECREATION AREA"}

# Same colormap the export used to render the probability PNGs
YEWCMAP = LinearSegmentedColormap.from_list('yew', [
    (0.00, (0.20, 0.70, 0.20, 0.70)), (0.17, (0.45, 0.85, 0.05, 0.80)),
    (0.33, (1.00, 0.90, 0.00, 0.88)), (0.50, (1.00, 0.60, 0.00, 0.90)),
    (0.67, (0.90, 0.40, 0.10, 0.93)), (0.83, (0.80, 0.15, 0.30, 0.95)),
    (1.00, (0.65, 0.00, 0.45, 0.96))], N=256)
_LUT = (YEWCMAP(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.float32)
_VAL = np.linspace(0, 1, 256)


def png_to_habitat_mask(png_path):
    """Invert the probability PNG → boolean habitat mask (suppressed P ≥ 0.5).

    Only non-transparent pixels (≈7% of a tile) are inverted, via a memory-light
    running-argmin over the 256-entry colormap LUT.
    """
    im = np.asarray(Image.open(png_path).convert("RGBA"), dtype=np.float32)
    rgb, alpha = im[:, :, :3], im[:, :, 3]
    mask = np.zeros(alpha.shape, bool)
    vis = alpha >= 13                       # transparent => below 0.02, skip
    if not vis.any():
        return mask
    px = rgb[vis]                           # (Nvis, 3)
    best_d = np.full(len(px), np.inf, np.float32)
    best_i = np.zeros(len(px), np.int16)
    for i in range(256):                    # 256 cheap passes over visible px
        d = ((px - _LUT[i]) ** 2).sum(1)
        upd = d < best_d
        best_d[upd] = d[upd]
        best_i[upd] = i
    mask[vis] = _VAL[best_i] >= THRESH
    return mask


def main():
    tiles = json.loads(TILES_JSON.read_text())
    parks = gpd.read_file(PARKS).to_crs("EPSG:4326")
    parks = parks[parks.geometry.notna() & ~parks.geometry.is_empty]
    print(f"{len(tiles)} tiles | {len(parks)} protected-area polygons "
          f"({parks['PROTECTED_LANDS_DESIGNATION'].nunique()} designations)\n")

    tot_hab = tot_prot_all = tot_prot_prov = 0
    desig_px = defaultdict(int)
    rows = []
    from shapely.geometry import box as sbox

    for t in tiles:
        png = TILES_DIR / t["png"]
        if not png.exists():
            print(f"  SKIP {t['name']}: {png.name} missing")
            continue
        hab = png_to_habitat_mask(png)
        H, W = hab.shape
        b = t["bbox"]
        transform = rio_bounds(b["west"], b["south"], b["east"], b["north"], W, H)
        tile_box = sbox(b["west"], b["south"], b["east"], b["north"])
        local = parks[parks.intersects(tile_box)]

        hab_px = int(hab.sum())
        tot_hab += hab_px
        if hab_px == 0 or local.empty:
            rows.append((t["name"], hab_px * HA_PER_PX, 0.0, 0.0))
            continue

        def rasterize(gdf):
            shapes = [(g, 1) for g in gdf.geometry if g and not g.is_empty]
            if not shapes:
                return np.zeros((H, W), bool)
            return rasterio.features.rasterize(
                shapes, out_shape=(H, W), transform=transform, fill=0,
                dtype=np.uint8).astype(bool)

        prot_all = hab & rasterize(local)
        prot_prov = hab & rasterize(
            local[local["PROTECTED_LANDS_DESIGNATION"].isin(PROVINCIAL)])
        tot_prot_all += int(prot_all.sum())
        tot_prot_prov += int(prot_prov.sum())

        for desig, sub in local.groupby("PROTECTED_LANDS_DESIGNATION"):
            desig_px[desig] += int((hab & rasterize(sub)).sum())

        rows.append((t["name"], hab_px * HA_PER_PX,
                     int(prot_all.sum()) * HA_PER_PX,
                     int(prot_all.sum()) / hab_px * 100))

    print("=" * 68)
    hab_ha = tot_hab * HA_PER_PX
    print(f"Total mapped habitat (P≥0.5, suppressed):  {hab_ha:>10,.0f} ha "
          f"(across {len(tiles)} tiles)")
    print(f"Inside ALL protected areas:                {tot_prot_all*HA_PER_PX:>10,.0f} ha "
          f"({tot_prot_all/tot_hab*100:.1f}%)")
    print(f"Inside provincial parks only:              {tot_prot_prov*HA_PER_PX:>10,.0f} ha "
          f"({tot_prot_prov/tot_hab*100:.1f}%)")
    print("\nBy designation (% of all mapped habitat):")
    for desig, px in sorted(desig_px.items(), key=lambda x: -x[1]):
        print(f"  {desig:<20} {px*HA_PER_PX:>9,.0f} ha  ({px/tot_hab*100:4.1f}%)")

    print("\nTop tiles by protected habitat:")
    for name, hab, prot, pct in sorted(rows, key=lambda r: -r[2])[:8]:
        print(f"  {name:<26} habitat={hab:>7.0f} ha  protected={prot:>6.0f} ha ({pct:4.1f}%)")

    # Persist a reproducible summary for the paper / future runs
    out = {
        "n_tiles": len(tiles),
        "total_habitat_ha": round(hab_ha, 1),
        "protected_all_ha": round(tot_prot_all * HA_PER_PX, 1),
        "protected_all_pct": round(tot_prot_all / tot_hab * 100, 1),
        "protected_provincial_ha": round(tot_prot_prov * HA_PER_PX, 1),
        "protected_provincial_pct": round(tot_prot_prov / tot_hab * 100, 1),
        "by_designation_ha": {d: round(px * HA_PER_PX, 1) for d, px in desig_px.items()},
        "per_tile": [{"name": n, "habitat_ha": round(h, 1),
                      "protected_ha": round(p, 1), "pct": round(c, 1)}
                     for n, h, p, c in rows],
    }
    out_path = ROOT / "results" / "analysis" / "protected_area_all_tiles.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
