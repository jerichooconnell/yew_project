#!/usr/bin/env python3
"""
figures_enhanced.py
───────────────────
Higher-data-density figures that use per-tile and protected-area results which
the original figure set under-exploited:

  figA  Geographic map of yew habitat across BC, each tile sized by mapped
        habitat (P≥0.5) and coloured by % inside protected areas — shows WHERE
        habitat is, how much, and how exposed it is, in one view.
  figB  Where the 37,885 ha of mapped habitat sits: protected by designation
        vs unprotected, plus the share in tiles with zero protection.

Inputs (all committed, no GEE):
  docs/tiles/tiles.json
  results/analysis/protected_area_all_tiles.json
  results/analysis/cwh_yew_population_100k/bc_boundary_simplified.geojson

Run:  conda run -n yew_pytorch python scripts/analysis/figures_enhanced.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm, colors

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "figures" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

TILES = {t["name"]: t for t in json.loads((ROOT / "docs/tiles/tiles.json").read_text())}
PROT = json.loads((ROOT / "results/analysis/protected_area_all_tiles.json").read_text())
PT = {t["name"]: t for t in PROT["per_tile"]}
BC = ROOT / "results/analysis/cwh_yew_population_100k/bc_boundary_simplified.geojson"


def figA_habitat_protection_map():
    rows = []
    for name, t in TILES.items():
        hab = PT[name]["habitat_ha"]
        if hab <= 0:
            continue
        rows.append((t["lon"], t["lat"], hab, PT[name]["pct"]))
    lon, lat, hab, pct = (np.array(x) for x in zip(*rows))

    fig, ax = plt.subplots(figsize=(10, 9))
    if BC.exists():
        try:
            import geopandas as gpd
            gpd.read_file(BC).to_crs(4326).plot(
                ax=ax, color="#eef2ee", edgecolor="#9bb09b", linewidth=0.6, zorder=1)
        except Exception as exc:
            print(f"    (BC boundary not drawn: {exc})")

    sizes = 30 + hab / hab.max() * 620          # area ∝ mapped habitat
    cmap = cm.get_cmap("RdYlGn")
    norm = colors.Normalize(0, 100)
    sc = ax.scatter(lon, lat, s=sizes, c=pct, cmap=cmap, norm=norm,
                    edgecolors="black", linewidth=0.5, alpha=0.9, zorder=3)

    cb = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.02)
    cb.set_label("% of tile habitat inside protected areas", fontsize=11)

    # size legend
    for h in [250, 1000, 2500]:
        ax.scatter([], [], s=30 + h / hab.max() * 620, c="lightgray",
                   edgecolors="black", linewidth=0.5, label=f"{h:,} ha")
    leg = ax.legend(title="Mapped yew habitat", loc="lower left",
                    labelspacing=1.4, borderpad=1.0, framealpha=0.95, fontsize=9)
    leg.get_title().set_fontsize(10)

    ax.set_xlabel("Longitude (°W)", fontsize=12)
    ax.set_ylabel("Latitude (°N)", fontsize=12)
    ax.set_title("Pacific Yew Habitat and Protection Across British Columbia\n"
                 "(tile area ∝ mapped habitat; colour = % protected)", fontsize=13)
    ax.set_xlim(-134, -114.5)
    ax.set_ylim(48, 56)
    ax.set_aspect(1.0 / np.cos(np.radians(52)))
    ax.grid(alpha=0.25, zorder=0)
    fig.tight_layout()
    fig.savefig(OUT / "figA_habitat_protection_map.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "figA_habitat_protection_map.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ figA habitat/protection map")


def figB_protection_breakdown():
    total = PROT["total_habitat_ha"]
    d = PROT["by_designation_ha"]
    prov = d.get("PROVINCIAL PARK", 0) + d.get("ECOLOGICAL RESERVE", 0) + d.get("PROTECTED AREA", 0)
    consv = d.get("CONSERVANCY", 0)
    natl = d.get("NATIONAL PARK", 0)
    unprot = total - (prov + consv + natl)
    zero_ha = sum(t["habitat_ha"] for t in PROT["per_tile"] if t["pct"] == 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6),
                                   gridspec_kw={"width_ratios": [2.3, 1]})

    # ── left: composition of all mapped habitat (single stacked bar) ──
    segs = [("Provincial parks", prov, "#2e8b57"),
            ("Conservancies", consv, "#8e44ad"),
            ("National parks", natl, "#d35400"),
            ("Unprotected", unprot, "#b0b0b0")]
    left = 0
    handles = []
    for label, val, c in segs:
        ax1.barh(0, val, left=left, color=c, edgecolor="white", height=0.55)
        handles.append(plt.Rectangle((0, 0), 1, 1, color=c,
                                     label=f"{label}: {val:,.0f} ha ({val/total*100:.1f}%)"))
        if label == "Unprotected":      # only the wide segment is labelled inline
            ax1.text(left + val / 2, 0, f"Unprotected\n{val:,.0f} ha ({val/total*100:.1f}%)",
                     ha="center", va="center", fontsize=11, color="#222", fontweight="bold")
        left += val
    # leader annotation for the protected cluster (left end)
    prot_total = prov + consv + natl
    ax1.annotate(f"Protected: {prot_total:,.0f} ha ({prot_total/total*100:.1f}%)",
                 xy=(prot_total, 0.30), xytext=(total * 0.18, 0.46),
                 fontsize=10, fontweight="bold", color="#14532d",
                 arrowprops=dict(arrowstyle="->", color="#14532d", lw=1.2))
    ax1.set_xlim(0, total)
    ax1.set_ylim(-0.6, 0.6)
    ax1.set_yticks([])
    ax1.set_xlabel("Mapped yew habitat (ha, P≥0.5; total = %s ha)" % f"{total:,.0f}", fontsize=11)
    ax1.set_title("Only 11.0% of mapped yew habitat is protected (5.6% in provincial parks)",
                  fontsize=12)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax1.legend(handles=handles, loc="lower center", ncol=2, fontsize=9,
               framealpha=0.95, bbox_to_anchor=(0.5, -0.42))

    # ── right: how much habitat sits in tiles with ZERO protection ──
    ax2.bar(0, total, color="#e8e8e8", edgecolor="#999", width=0.6, label="All habitat")
    ax2.bar(0, zero_ha, color="#c0392b", edgecolor="black", width=0.6,
            label=f"In tiles with 0% protection")
    ax2.text(0, zero_ha / 2, f"{zero_ha:,.0f} ha\n({zero_ha/total*100:.0f}%)",
             ha="center", va="center", color="white", fontweight="bold", fontsize=11)
    ax2.set_xlim(-0.6, 0.6)
    ax2.set_xticks([])
    ax2.set_ylabel("Mapped habitat (ha)", fontsize=11)
    ax2.set_title("Habitat in wholly\nunprotected tiles", fontsize=12)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    fig.tight_layout()
    fig.savefig(OUT / "figB_protection_breakdown.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "figB_protection_breakdown.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ figB protection breakdown")


def figC_geographic_panels():
    """2x2 small-multiples map: each tile coloured by a different variable, so
    the geography of habitat, logging pressure, old-growth refugia, and
    protection can be compared side by side. Marker area ∝ mapped habitat
    throughout, tying the panels together."""
    import geopandas as gpd
    bc = gpd.read_file(BC).to_crs(4326) if BC.exists() else None

    recs = []
    for name, t in TILES.items():
        hab = PT[name]["habitat_ha"]
        ls = t["logging_stats"]
        pct_logged = ls.get("logged_lt20_pct", 0) + ls.get("logged_20_40_pct", 0) + ls.get("logged_40_80_pct", 0)
        recs.append(dict(lon=t["lon"], lat=t["lat"], hab=hab,
                         logged=pct_logged, og=ls.get("oldgrowth_pct", 0),
                         prot=PT[name]["pct"]))
    lon = np.array([r["lon"] for r in recs])
    lat = np.array([r["lat"] for r in recs])
    hab = np.array([r["hab"] for r in recs])
    sizes = 12 + hab / max(hab.max(), 1) * 300

    panels = [
        ("Mapped habitat (ha, P≥0.5)", hab, "YlGn", None,
         lambda v: f"{v:,.0f}"),
        ("Forest area logged (%)", np.array([r["logged"] for r in recs]), "OrRd", (0, 100),
         lambda v: f"{v:.0f}"),
        ("Old-growth remaining (% of tile)", np.array([r["og"] for r in recs]), "Greens", None,
         lambda v: f"{v:.0f}"),
        ("Habitat inside protected areas (%)", np.array([r["prot"] for r in recs]), "RdYlGn", (0, 100),
         lambda v: f"{v:.0f}"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    for ax, (title, vals, cmap, vlim, _) in zip(axes.ravel(), panels):
        if bc is not None:
            bc.plot(ax=ax, color="#f0f3f0", edgecolor="#9bb09b", linewidth=0.5, zorder=1)
        vmin, vmax = (vlim if vlim else (float(vals.min()), float(vals.max())))
        sc = ax.scatter(lon, lat, c=vals, s=sizes, cmap=cmap,
                        vmin=vmin, vmax=vmax, edgecolors="black",
                        linewidth=0.4, alpha=0.9, zorder=3)
        cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
        cb.ax.tick_params(labelsize=8)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(-134, -114.5)
        ax.set_ylim(48, 56)
        ax.set_aspect(1.0 / np.cos(np.radians(52)))
        ax.set_xlabel("Longitude (°W)", fontsize=9)
        ax.set_ylabel("Latitude (°N)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.2, zorder=0)

    fig.suptitle("Geography of Pacific Yew Habitat, Logging, Old-Growth and Protection\n"
                 "(99 study tiles; marker area ∝ mapped habitat)", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "figC_geographic_panels.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "figC_geographic_panels.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ figC geographic small-multiples")


# ── shared helpers for tile-imagery maps ──────────────────────────────────────
from matplotlib.colors import LinearSegmentedColormap
YEWCMAP = LinearSegmentedColormap.from_list('yew', [
    (0.00, (0.20, 0.70, 0.20)), (0.17, (0.45, 0.85, 0.05)),
    (0.33, (1.00, 0.90, 0.00)), (0.50, (1.00, 0.60, 0.00)),
    (0.67, (0.90, 0.40, 0.10)), (0.83, (0.80, 0.15, 0.30)),
    (1.00, (0.65, 0.00, 0.45))], N=256)

# VRI logging-category colours (match export_tiles_for_web.py LOG_RGBA)
LOG_LEGEND = [
    ("Water / non-forest", (30, 100, 220)), ("Logged <20 yr", (220, 50, 50)),
    ("Logged 20–40 yr", (230, 120, 30)), ("Logged 40–80 yr", (220, 200, 50)),
    ("Forest 80–150 yr", (180, 220, 70)), ("Alpine / barren", (175, 155, 125)),
    ("Old-growth >150 yr", (20, 100, 40))]

REGIONS = {
    "Vancouver Island": dict(xlim=(-126.7, -123.3), ylim=(48.2, 50.5)),
    "Central & North Coast": dict(xlim=(-129.4, -125.6), ylim=(50.8, 53.8)),
}


def _bc():
    import geopandas as gpd
    return gpd.read_file(BC).to_crs(4326) if BC.exists() else None


def _region_imgs(png_key, region):
    from PIL import Image
    out = []
    for name, t in TILES.items():
        p = ROOT / "docs/tiles" / t[png_key]
        if not p.exists():
            continue
        b = t["bbox"]
        cx, cy = (b["west"] + b["east"]) / 2, (b["south"] + b["north"]) / 2
        if not (region["xlim"][0] < cx < region["xlim"][1]
                and region["ylim"][0] < cy < region["ylim"][1]):
            continue
        out.append((name, np.asarray(Image.open(p).convert("RGBA")),
                    [b["west"], b["east"], b["south"], b["north"]]))
    return out


def _draw_region(ax, imgs, region, bc, label=True):
    from matplotlib.patches import Rectangle
    if bc is not None:
        bc.plot(ax=ax, color="#eef2ee", edgecolor="#9bb09b", linewidth=0.7, zorder=1)
    for name, im, ext in imgs:
        ax.imshow(im, extent=ext, origin="upper", zorder=3, interpolation="nearest")
        ax.add_patch(Rectangle((ext[0], ext[2]), ext[1]-ext[0], ext[3]-ext[2],
                               fill=False, edgecolor="#222", linewidth=0.7, zorder=4))
        if label:
            ax.annotate(name, (ext[0], ext[3]), fontsize=6, va="bottom", ha="left",
                        color="#222", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6))
    ax.set_xlim(*region["xlim"])
    ax.set_ylim(*region["ylim"])
    ax.set_aspect(1.0 / np.cos(np.radians(np.mean(region["ylim"]))))


def _bc_locator(ax, bc, region):
    from matplotlib.patches import Rectangle
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axin = inset_axes(ax, width="26%", height="26%", loc="upper right")
    if bc is not None:
        bc.plot(ax=axin, color="#eef2ee", edgecolor="#9bb09b", linewidth=0.4)
    axin.add_patch(Rectangle((region["xlim"][0], region["ylim"][0]),
                             region["xlim"][1]-region["xlim"][0],
                             region["ylim"][1]-region["ylim"][0],
                             fill=False, edgecolor="red", linewidth=1.3))
    axin.set_xlim(-134, -114.5); axin.set_ylim(48, 56)
    axin.set_xticks([]); axin.set_yticks([])
    axin.set_aspect(1.0 / np.cos(np.radians(52)))


def _imagery_map(region_name, out_stub, fignum):
    """Single-region probability tile-imagery map."""
    region = REGIONS[region_name]
    bc = _bc()
    imgs = _region_imgs("png", region)
    fig, ax = plt.subplots(figsize=(12, 9))
    _draw_region(ax, imgs, region, bc)
    ax.set_xlabel("Longitude (°W)", fontsize=11)
    ax.set_ylabel("Latitude (°N)", fontsize=11)
    ax.set_title(f"Predicted Pacific Yew Probability — Tile Imagery, {region_name}\n"
                 "(each patch is a study tile's model output at its true location)", fontsize=13)
    _bc_locator(ax, bc, region)
    sm = plt.cm.ScalarMappable(cmap=YEWCMAP, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("Predicted yew probability (suppressed)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / f"{out_stub}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{out_stub}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_stub} ({region_name}, {len(imgs)} tiles)")


def figD_tile_imagery_map():
    _imagery_map("Vancouver Island", "figD_tile_imagery_map", 11)


def figE_tile_imagery_north():
    _imagery_map("Central & North Coast", "figE_tile_imagery_north", 12)


def figF_prob_vs_logging():
    """Side-by-side maps of the same tiles: predicted yew probability (left) vs
    VRI logging classification (right) — shows the decline driver directly."""
    from matplotlib.patches import Patch
    region = REGIONS["Vancouver Island"]
    bc = _bc()
    prob = _region_imgs("png", region)
    logg = _region_imgs("logging_png", region)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 8.5))
    _draw_region(axL, prob, region, bc, label=True)
    axL.set_title("Predicted yew probability", fontsize=13)
    _draw_region(axR, logg, region, bc, label=False)
    axR.set_title("VRI logging / forest-age classification", fontsize=13)
    for ax in (axL, axR):
        ax.set_xlabel("Longitude (°W)", fontsize=10)
    axL.set_ylabel("Latitude (°N)", fontsize=10)

    sm = plt.cm.ScalarMappable(cmap=YEWCMAP, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axL, shrink=0.55, pad=0.02)
    cb.set_label("Yew probability", fontsize=9)
    handles = [Patch(facecolor=np.array(c)/255, edgecolor="#444", label=lab)
               for lab, c in LOG_LEGEND]
    axR.legend(handles=handles, fontsize=7.5, loc="lower left", framealpha=0.95,
               title="Logging category", title_fontsize=8)

    fig.suptitle("Yew Probability vs. Logging Status, Vancouver Island Tiles\n"
                 "(old-growth tiles carry the highest predicted yew probability; "
                 "logged tiles are largely zeroed)", fontsize=14, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "figF_prob_vs_logging.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "figF_prob_vs_logging.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ figF prob-vs-logging ({len(prob)} tiles)")


if __name__ == "__main__":
    figA_habitat_protection_map()
    figB_protection_breakdown()
    figC_geographic_panels()
    figD_tile_imagery_map()
    figE_tile_imagery_north()
    figF_prob_vs_logging()
    print("done")
