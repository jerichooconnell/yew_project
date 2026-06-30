#!/usr/bin/env python3
"""
Single-tile diagnostic panels for Pacific yew habitat.

For a handful of information-rich CWH tiles, render a six-panel composite that
makes the per-tile threat story legible at the scale of one ~10×10 km tile:

  (a) RGB satellite context
  (b) Yew habitat probability (raw XGBoost output, YEWCMAP)
  (c) Logging — VRI stand-age categories (old-growth vs logged classes)
  (d) Fire — historical fire perimeters coloured by recency, over a habitat backdrop
  (e) Riparian erosion risk — 30 m water buffer intersected with old-growth habitat
  (f) Protected areas — park/conservancy/national-park polygons over habitat

These complement the province-wide figures by showing the actual model output
and threat geometry for a single zone, which reviewers of the companion
distribution report specifically asked to see at native resolution.

Run:
    conda run -n yew_pytorch python scripts/analysis/single_tile_panels.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from pathlib import Path
from scipy.ndimage import binary_dilation
import geopandas as gpd
import rasterio.features
from rasterio.transform import from_bounds as rio_bounds
from shapely.geometry import box as sbox

ROOT   = Path("/home/jericho/yew_project")
CACHE  = ROOT / "results/analysis/cwh_spot_comparisons/tile_cache"
OUTDIR = ROOT / "results/figures/paper"
HA_PER_PX = 0.01

# Tiles to render (cached, all four threats present; one zone each)
TILES_TO_PLOT = [
    "sechelt_peninsula",     # logging + fire + park + riparian, high habitat
    "squamish_highlands",    # most parks (5), strong logging, fire
    "theodosia_inlet",       # highest water fraction — riparian/erosion story
]

YEWCMAP = LinearSegmentedColormap.from_list("yew", [
    (0.00, (0.85, 0.90, 0.85)), (0.25, (0.40, 0.75, 0.55)),
    (0.50, (0.95, 0.78, 0.20)), (0.75, (0.84, 0.37, 0.00)),
    (1.00, (0.55, 0.10, 0.40))], N=256)

# VRI category → colour (cats 0–7); 0 = nodata
LOG_CATS = {
    0: ("No data",            (0.90, 0.90, 0.90)),
    1: ("Water / non-forest", (0.12, 0.39, 0.86)),
    2: ("Logged <20 yr",      (0.86, 0.20, 0.20)),
    3: ("Logged 20–40 yr",    (0.90, 0.47, 0.12)),
    4: ("Logged 40–80 yr",    (0.86, 0.78, 0.20)),
    5: ("Forest 80–150 yr",   (0.71, 0.86, 0.27)),
    6: ("Alpine / barren",    (0.69, 0.61, 0.49)),
    7: ("Old-growth >150 yr", (0.08, 0.39, 0.16)),
}
LOG_LISTED = ListedColormap([LOG_CATS[i][1] for i in range(8)])
LOG_NORM   = BoundaryNorm(np.arange(-0.5, 8.5, 1), LOG_LISTED.N)

FIRE_YEAR_NOW = 2024


def load_tile(slug):
    g  = np.load(CACHE / f"{slug}_grid.npy").astype(np.float32)
    lg = np.load(CACHE / f"{slug}_logging.npy")
    rgb_path = CACHE / f"{slug}_rgb.npy"
    rgb = np.load(rgb_path) if rgb_path.exists() else None
    return g, lg, rgb


def fire_year_raster(H, W, bb, fires):
    box = sbox(bb["west"], bb["south"], bb["east"], bb["north"])
    local = fires[fires.intersects(box)].sort_values("FIRE_YEAR")
    transform = rio_bounds(bb["west"], bb["south"], bb["east"], bb["north"], W, H)
    if local.empty:
        return np.zeros((H, W), np.int16), 0
    shapes = [(r.geometry, int(r.FIRE_YEAR)) for _, r in local.iterrows()
              if r.geometry and not r.geometry.is_empty]
    fy = rasterio.features.rasterize(shapes, (H, W), transform=transform,
                                     fill=0, dtype="int32").astype(np.int16)
    return fy, len(local)


def park_raster(H, W, bb, parks):
    box = sbox(bb["west"], bb["south"], bb["east"], bb["north"])
    local = parks[parks.intersects(box)]
    transform = rio_bounds(bb["west"], bb["south"], bb["east"], bb["north"], W, H)
    if local.empty:
        return np.zeros((H, W), bool), 0
    shapes = [(g, 1) for g in local.geometry if g and not g.is_empty]
    pr = rasterio.features.rasterize(shapes, (H, W), transform=transform,
                                     fill=0, dtype=np.uint8).astype(bool)
    return pr, len(local)


def _norm_rgb(rgb):
    """Normalise an HxWx3 array to 0–1 for display, robust to value range."""
    if rgb is None:
        return None
    arr = rgb.astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] in (3, 4):      # C,H,W → H,W,C
        arr = np.transpose(arr, (1, 2, 0))
    arr = arr[..., :3]
    lo, hi = np.percentile(arr, [2, 98])
    if hi <= lo:
        hi = lo + 1
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def make_panel(slug, t, fires, parks):
    g, lg, rgb = load_tile(slug)
    H, W = g.shape
    bb = t["bbox"]
    extent = [bb["west"], bb["east"], bb["south"], bb["north"]]

    og_mask  = (lg == 7)
    supp     = g * og_mask                       # habitat surviving in old-growth
    raw_yew_ha  = float((g * og_mask).sum()) * HA_PER_PX
    hab_px      = int((supp >= 0.5).sum())

    fy, n_fire = fire_year_raster(H, W, bb, fires)
    pr, n_park = park_raster(H, W, bb, parks)

    # riparian: 30 m (3 px) dilation of water, intersected with old-growth habitat
    water_dil = binary_dilation(lg == 1, iterations=3)
    erosion   = water_dil & og_mask & (g >= 0.5)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10.6))
    fig.suptitle(f"{t['name']}  ·  single-tile threat diagnostics "
                 f"(~{(bb['east']-bb['west'])*111*np.cos(np.radians(bb['south'])):.0f} × "
                 f"{(bb['north']-bb['south'])*111:.0f} km, CWH zone)",
                 fontsize=14, fontweight="bold", y=0.98)

    rgb_disp = _norm_rgb(rgb)

    # (a) RGB context
    ax = axes[0, 0]
    if rgb_disp is not None:
        ax.imshow(rgb_disp, extent=extent, origin="upper", interpolation="nearest")
    else:
        ax.text(0.5, 0.5, "RGB not cached", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_title("(a) Sentinel/AlphaEarth RGB context", fontsize=10.5)

    # (b) yew probability
    ax = axes[0, 1]
    im = ax.imshow(g, extent=extent, origin="upper", cmap=YEWCMAP,
                   vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"(b) Yew habitat probability (raw)\n"
                 f"old-growth mass = {raw_yew_ha:,.0f} ha", fontsize=10.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("P(yew)", fontsize=8)

    # (c) logging categories
    ax = axes[0, 2]
    ax.imshow(lg, extent=extent, origin="upper", cmap=LOG_LISTED, norm=LOG_NORM,
              interpolation="nearest")
    logged_pct = float(np.isin(lg, [2, 3, 4, 5]).sum()) / lg.size * 100
    ax.set_title(f"(c) VRI stand age (logging)\n{logged_pct:.0f}% of tile logged <150 yr",
                 fontsize=10.5)
    handles = [Patch(facecolor=LOG_CATS[i][1], label=LOG_CATS[i][0])
               for i in (7, 5, 4, 3, 2, 1)]
    ax.legend(handles=handles, fontsize=6.5, loc="lower left", framealpha=0.85)

    # (d) fire over faded habitat
    ax = axes[1, 0]
    ax.imshow(np.ma.masked_less(supp, 0.15), extent=extent, origin="upper",
              cmap=YEWCMAP, vmin=0, vmax=1, alpha=0.5, interpolation="nearest")
    if n_fire:
        recency = np.where(fy > 0, (FIRE_YEAR_NOW - fy), np.nan)
        im = ax.imshow(np.ma.masked_invalid(recency), extent=extent, origin="upper",
                       cmap="inferno_r", vmin=0, vmax=124, interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("years since fire", fontsize=8)
    ax.set_title(f"(d) Historical fire perimeters\n{n_fire} fire(s) intersect tile",
                 fontsize=10.5)

    # (e) riparian erosion risk
    ax = axes[1, 1]
    ax.imshow(np.ma.masked_less(supp, 0.15), extent=extent, origin="upper",
              cmap=YEWCMAP, vmin=0, vmax=1, alpha=0.45, interpolation="nearest")
    ax.imshow(np.ma.masked_where(~(lg == 1), lg == 1), extent=extent, origin="upper",
              cmap=ListedColormap([(0.12, 0.39, 0.86)]), alpha=0.55,
              interpolation="nearest")
    ax.imshow(np.ma.masked_where(~erosion, erosion), extent=extent, origin="upper",
              cmap=ListedColormap([(0.85, 0.10, 0.10)]), interpolation="nearest")
    er_ha = float((g * erosion).sum()) * HA_PER_PX
    ax.set_title(f"(e) Riparian erosion risk (30 m buffer)\n"
                 f"{er_ha:,.0f} ha old-growth habitat at risk", fontsize=10.5)
    ax.legend(handles=[Patch(facecolor=(0.12, 0.39, 0.86), label="Water"),
                       Patch(facecolor=(0.85, 0.10, 0.10), label="Habitat in buffer")],
              fontsize=7, loc="lower left", framealpha=0.85)

    # (f) protected areas
    ax = axes[1, 2]
    ax.imshow(np.ma.masked_less(supp, 0.15), extent=extent, origin="upper",
              cmap=YEWCMAP, vmin=0, vmax=1, alpha=0.55, interpolation="nearest")
    if n_park:
        prot_hab = pr & (supp >= 0.5)
        ax.imshow(np.ma.masked_where(~pr, pr), extent=extent, origin="upper",
                  cmap=ListedColormap([(0.20, 0.50, 0.95)]), alpha=0.28,
                  interpolation="nearest")
        ax.imshow(np.ma.masked_where(~prot_hab, prot_hab), extent=extent, origin="upper",
                  cmap=ListedColormap([(0.10, 0.45, 0.20)]), interpolation="nearest")
        prot_ha = prot_hab.sum() * HA_PER_PX
        pct = prot_hab.sum() / max(hab_px, 1) * 100
        ttl = f"(f) Protected areas\n{prot_ha:,.0f} ha ({pct:.0f}%) of habitat protected"
    else:
        ttl = "(f) Protected areas\nnone intersect tile (0% protected)"
    ax.set_title(ttl, fontsize=10.5)
    ax.legend(handles=[Patch(facecolor=(0.20, 0.50, 0.95), alpha=0.4, label="Protected area"),
                       Patch(facecolor=(0.10, 0.45, 0.20), label="Habitat protected")],
              fontsize=7, loc="lower left", framealpha=0.85)

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = OUTDIR / f"tile_panel_{slug}.png"
    out_pdf = OUTDIR / f"tile_panel_{slug}.pdf"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {slug}: yew {raw_yew_ha:,.0f} ha, {n_fire} fires, "
          f"{n_park} parks → {out_png.name}")


def main():
    tiles = {t["slug"]: t for t in json.load(open(ROOT / "docs/tiles/tiles.json"))}
    fires = gpd.read_file(str(ROOT / "docs/tiles/fire_contours.geojson"))
    parks = gpd.read_file(str(ROOT / "docs/tiles/park_contours.geojson"))
    print(f"Loaded {len(fires)} fire perimeters, {len(parks)} protected polygons\n")
    for slug in TILES_TO_PLOT:
        if not (CACHE / f"{slug}_grid.npy").exists():
            print(f"  SKIP {slug}: no cached grid")
            continue
        make_panel(slug, tiles[slug], fires, parks)


if __name__ == "__main__":
    main()
