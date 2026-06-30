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
import sys
import json
from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio.features
from rasterio.transform import from_bounds as rio_bounds
from shapely.geometry import box as sbox

sys.path.insert(0, str(Path(__file__).parent))
from fire_recovery import busing_fire_modifier, BURN_FRAC

ROOT       = Path("/home/jericho/yew_project")
CACHE      = ROOT / "results/analysis/cwh_spot_comparisons/tile_cache"
TILES_JSON = ROOT / "docs/tiles/tiles.json"
FIRE_JSON  = ROOT / "docs/tiles/fire_contours.geojson"
HA_PER_PX  = 0.01
FIRE_YEAR_NOW, FIRE_SPAN = 2024, 124.0
PROVINCE_HISTORICAL_HA = 154_483.0       # headline original-habitat estimate (98 tiles)
PIPELINE_FIRE_HA       = 692.0           # current linear-model pipeline fire loss (98 tiles)

OG_CAT      = 7
LOGGED_CATS = (2, 3, 4, 5)               # non-old-growth forest = "logged"


def fire_year_raster(H, W, bb, fires):
    """Year of most recent fire per pixel (0 = unburned)."""
    box = sbox(bb["west"], bb["south"], bb["east"], bb["north"])
    local = fires[fires.intersects(box)].sort_values("FIRE_YEAR")  # newer overwrites older
    if local.empty:
        return np.zeros((H, W), np.int32)
    transform = rio_bounds(bb["west"], bb["south"], bb["east"], bb["north"], W, H)
    shapes = [(r.geometry, int(r.FIRE_YEAR)) for _, r in local.iterrows()
              if r.geometry and not r.geometry.is_empty]
    return rasterio.features.rasterize(shapes, (H, W), transform=transform,
                                       fill=0, dtype="int32")


def linear_modifier(fy):
    """Old arbitrary modifier: clip((2024 − fire_year)/124, 0, 1)."""
    return np.where(fy > 0, np.clip((FIRE_YEAR_NOW - fy) / FIRE_SPAN, 0, 1),
                    1.0).astype(np.float32)


def busing_modifier(fy):
    """New modifier: 75% burn + Busing-matrix recovery of the mature cohort."""
    yrs = np.where(fy > 0, FIRE_YEAR_NOW - fy, -1)
    return busing_fire_modifier(yrs)


def main():
    tiles = {t["slug"]: t for t in json.load(open(TILES_JSON))}
    fires = gpd.read_file(str(FIRE_JSON))
    print(f"{len(tiles)} tiles, {len(fires)} fire perimeters")
    print(f"New fire model: {BURN_FRAC:.0%} burn within perimeter + Busing recovery\n")

    hist_mass = 0.0
    # accumulators: [linear, busing] × [og, logged]
    acc = {("lin", "og"): 0.0, ("lin", "log"): 0.0,
           ("bus", "og"): 0.0, ("bus", "log"): 0.0}
    n = 0
    per_tile = []

    for slug, t in tiles.items():
        gp, lp = CACHE / f"{slug}_grid.npy", CACHE / f"{slug}_logging.npy"
        if not (gp.exists() and lp.exists()):
            continue
        grid = np.load(gp).astype(np.float32)
        lg   = np.load(lp)
        H, W = grid.shape
        fy   = fire_year_raster(H, W, t["bbox"], fires)
        mods = {"lin": linear_modifier(fy), "bus": busing_modifier(fy)}

        og_mask  = (lg == OG_CAT)
        log_mask = np.isin(lg, LOGGED_CATS)
        hist_mass += float((grid * (og_mask | log_mask)).sum())

        for key, fm in mods.items():
            acc[(key, "og")]  += float((grid * og_mask  * (1 - fm)).sum())
            acc[(key, "log")] += float((grid * log_mask * (1 - fm)).sum())
        n += 1
        per_tile.append((t["name"],
                         float((grid * (og_mask | log_mask) * (1 - mods["bus"])).sum()) * HA_PER_PX))

    scale = PROVINCE_HISTORICAL_HA / (hist_mass * HA_PER_PX)

    def ha(key, part):  # scaled to 98 tiles
        return acc[(key, part)] * HA_PER_PX * scale

    print(f"Cached tiles analysed: {n}")
    print(f"Original-habitat footprint (cached): {hist_mass*HA_PER_PX:,.0f} ha")
    print(f"Scale factor to 98-tile province total: ×{scale:.3f}\n")

    # Pipeline-style fire loss = OG-only (after logging). Calibrate the new model
    # to the known 692 ha linear pipeline figure via the OG-only ratio.
    ratio_og = acc[("bus", "og")] / acc[("lin", "og")]
    new_pipeline = PIPELINE_FIRE_HA * ratio_og

    print("=== Pipeline-style fire loss (old-growth only, after logging) ===")
    print(f"  Linear model (= 692 ha calibration):  OG {ha('lin','og'):,.0f} ha (cached-scaled)")
    print(f"  Busing model:                         OG {ha('bus','og'):,.0f} ha (cached-scaled)")
    print(f"  New-model / linear OG ratio: {ratio_og:.2f}")
    print(f"  → New pipeline fire estimate (98 tiles): {new_pipeline:,.0f} ha "
          f"(was {PIPELINE_FIRE_HA:.0f} ha)\n")

    print("=== Fire loss INDEPENDENT of logging (fire on raw original footprint) ===")
    for key, lab in (("lin", "Linear model "), ("bus", "Busing model ")):
        tot = ha(key, "og") + ha(key, "log")
        print(f"  {lab}: {tot:,.0f} ha total  "
              f"(OG {ha(key,'og'):,.0f} + logged {ha(key,'log'):,.0f}; "
              f"{tot/PROVINCE_HISTORICAL_HA*100:.1f}% of original)")

    per_tile.sort(key=lambda r: -r[1])
    print("\nTop cached tiles by Busing-model fire loss (ha):")
    for name, hav in per_tile[:8]:
        print(f"  {name:28s} {hav:6.1f}")


if __name__ == "__main__":
    main()
