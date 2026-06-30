#!/usr/bin/env python3
"""
Fire-driven yew-habitat loss computed INDEPENDENTLY of logging.

In the production pipeline the fire modifier is applied to the logging-suppressed
grid, so the headline 692 ha fire figure only captures fire's effect on the
old-growth that survived logging. This script instead applies the fire modifier
to the RAW probability grid over the full original-habitat footprint (old-growth
+ logged forest, i.e. VRI categories 2–5 and 7, excluding water and alpine),
answering: how much yew habitat would historical fires have suppressed if no
logging had occurred first?

Method (per tile, using the 41 tiles with cached raw grids):
  original_footprint = raw_prob over forested pixels (cats 2,3,4,5,7)
  fire_loss          = Σ raw_prob × (1 − fire_modifier) over that footprint
  fire_modifier      = clip((2024 − fire_year)/124, 0, 1)   (pipeline definition)
Loss is split into the old-growth (cat 7) and already-logged (cats 2–5) portions,
then the 41-tile total is scaled to all 98 study tiles by the ratio of historical
habitat (154,483 ha province / cached-tile historical mass).

Run:
    conda run -n yew_pytorch python scripts/analysis/fire_independent_loss.py
"""
import json
from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio.features
from rasterio.transform import from_bounds as rio_bounds
from shapely.geometry import box as sbox

ROOT       = Path("/home/jericho/yew_project")
CACHE      = ROOT / "results/analysis/cwh_spot_comparisons/tile_cache"
TILES_JSON = ROOT / "docs/tiles/tiles.json"
FIRE_JSON  = ROOT / "docs/tiles/fire_contours.geojson"
HA_PER_PX  = 0.01
FIRE_YEAR_NOW, FIRE_SPAN = 2024, 124.0
PROVINCE_HISTORICAL_HA = 154_483.0       # headline original-habitat estimate (98 tiles)

OG_CAT      = 7
LOGGED_CATS = (2, 3, 4, 5)               # non-old-growth forest = "logged"


def fire_modifier(H, W, bb, fires):
    """Pipeline fire modifier raster: clip((2024 − fire_year)/124, 0, 1)."""
    box = sbox(bb["west"], bb["south"], bb["east"], bb["north"])
    local = fires[fires.intersects(box)].sort_values("FIRE_YEAR")  # newer overwrites older
    if local.empty:
        return np.ones((H, W), np.float32)
    transform = rio_bounds(bb["west"], bb["south"], bb["east"], bb["north"], W, H)
    shapes = [(r.geometry, int(r.FIRE_YEAR)) for _, r in local.iterrows()
              if r.geometry and not r.geometry.is_empty]
    fy = rasterio.features.rasterize(shapes, (H, W), transform=transform,
                                     fill=0, dtype="int32")
    return np.where(fy > 0, np.clip((FIRE_YEAR_NOW - fy) / FIRE_SPAN, 0, 1),
                    1.0).astype(np.float32)


def main():
    tiles = {t["slug"]: t for t in json.load(open(TILES_JSON))}
    fires = gpd.read_file(str(FIRE_JSON))
    print(f"{len(tiles)} tiles, {len(fires)} fire perimeters\n")

    hist_mass = 0.0          # original-habitat footprint mass (cached tiles)
    fire_og   = 0.0          # fire loss in old-growth pixels
    fire_log  = 0.0          # fire loss in already-logged pixels
    pipe_og   = 0.0          # pipeline-style fire loss (OG only, after logging) — for comparison
    n = 0
    per_tile = []

    for slug, t in tiles.items():
        gp, lp = CACHE / f"{slug}_grid.npy", CACHE / f"{slug}_logging.npy"
        if not (gp.exists() and lp.exists()):
            continue
        grid = np.load(gp).astype(np.float32)
        lg   = np.load(lp)
        H, W = grid.shape
        fm   = fire_modifier(H, W, t["bbox"], fires)

        og_mask  = (lg == OG_CAT)
        log_mask = np.isin(lg, LOGGED_CATS)

        og_raw   = float((grid * og_mask).sum())
        log_raw  = float((grid * log_mask).sum())
        og_loss  = float((grid * og_mask  * (1 - fm)).sum())
        log_loss = float((grid * log_mask * (1 - fm)).sum())

        hist_mass += og_raw + log_raw
        fire_og   += og_loss
        fire_log  += log_loss
        pipe_og   += og_loss        # OG-only loss == pipeline fire effect on surviving OG
        n += 1
        per_tile.append((t["name"], (og_loss + log_loss) * HA_PER_PX))

    scale = PROVINCE_HISTORICAL_HA / (hist_mass * HA_PER_PX)
    fire_indep_41 = (fire_og + fire_log) * HA_PER_PX
    fire_indep_98 = fire_indep_41 * scale
    fire_og_98    = fire_og * HA_PER_PX * scale
    fire_log_98   = fire_log * HA_PER_PX * scale

    print(f"Cached tiles analysed: {n}")
    print(f"Original-habitat footprint (41 tiles): {hist_mass*HA_PER_PX:,.0f} ha")
    print(f"Scale factor to 98-tile province total: ×{scale:.3f}\n")

    print("=== Fire loss INDEPENDENT of logging (fire applied to raw grid) ===")
    print(f"  41-tile fire loss:        {fire_indep_41:,.0f} ha")
    print(f"  Scaled to 98 tiles:       {fire_indep_98:,.0f} ha "
          f"({fire_indep_98/PROVINCE_HISTORICAL_HA*100:.1f}% of 154,483 ha original)")
    print(f"    – in old-growth pixels: {fire_og_98:,.0f} ha")
    print(f"    – in logged pixels:     {fire_log_98:,.0f} ha "
          f"({fire_log_98/fire_indep_98*100:.0f}% — dual-threatened)")
    print(f"\nFor comparison, pipeline fire (after logging, all 98 tiles): 692 ha")
    print(f"Independent estimate is ~{fire_indep_98/692:.1f}× the pipeline figure.")

    per_tile.sort(key=lambda r: -r[1])
    print("\nTop cached tiles by independent fire loss (ha):")
    for name, ha in per_tile[:8]:
        print(f"  {name:28s} {ha:6.1f}")


if __name__ == "__main__":
    main()
