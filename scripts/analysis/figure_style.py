#!/usr/bin/env python3
"""
Shared figure style for the Pacific yew paper.

A single colourblind-safe (Okabe-Ito) palette and a uniform set of matplotlib
rcParams so every figure in the manuscript shares fonts, sizing, gridlines and
semantic colours. Import and call `apply_style()` at the top of each figure
script; use the named colours from PALETTE / ROLE so the same concept (habitat,
loss, fire, protected …) is always drawn the same way.

    from figure_style import apply_style, ROLE, YEWCMAP, sequential
    apply_style()
    ax.bar(..., color=ROLE['habitat'])
"""
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# ── Okabe-Ito colourblind-safe base palette ───────────────────────────────────
PALETTE = {
    "black":      "#000000",
    "orange":     "#E69F00",
    "sky":        "#56B4E9",
    "green":      "#009E73",   # bluish green
    "yellow":     "#F0E442",
    "blue":       "#0072B2",
    "vermillion": "#D55E00",
    "purple":     "#CC79A7",
    "grey":       "#999999",
}

# ── Semantic role → colour (use these in figures, not raw names) ───────────────
ROLE = {
    "habitat":    PALETTE["green"],       # remaining / mapped yew habitat
    "remaining":  PALETTE["green"],
    "loss":       PALETTE["vermillion"],  # logging / destroyed habitat
    "logging":    PALETTE["vermillion"],
    "oldgrowth":  PALETTE["blue"],        # old-growth forest
    "fire":       PALETTE["orange"],      # wildfire
    "protected":  PALETTE["sky"],         # protected areas
    "water":      PALETTE["sky"],
    "neutral":    PALETTE["grey"],
    "highlight":  PALETTE["purple"],      # emphasis / deficit / observed
    "expected":   PALETTE["grey"],        # reference / expected distribution
    "best":       PALETTE["green"],       # best-case scenario
    "status_quo": PALETTE["orange"],      # status-quo scenario
    "worst":      PALETTE["vermillion"],  # worst-case scenario
}

# Ordered list for categorical series needing distinct colours
CYCLE = [PALETTE["blue"], PALETTE["vermillion"], PALETTE["green"],
         PALETTE["orange"], PALETTE["purple"], PALETTE["sky"],
         PALETTE["yellow"], PALETTE["grey"]]

# ── Continuous yew-probability colormap (kept green→purple, journal-tuned) ─────
YEWCMAP = LinearSegmentedColormap.from_list("yew", [
    (0.00, (0.85, 0.90, 0.85)), (0.25, (0.40, 0.75, 0.55)),
    (0.50, (0.95, 0.78, 0.20)), (0.75, (0.84, 0.37, 0.00)),
    (1.00, (0.55, 0.10, 0.40))], N=256)


def sequential(name="green"):
    """A light→saturated single-hue ramp for one semantic colour."""
    base = PALETTE[name] if name in PALETTE else name
    return LinearSegmentedColormap.from_list(f"seq_{name}", ["#f2f2f2", base], N=256)


def apply_style():
    """Apply uniform manuscript rcParams (call once at script start)."""
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#d9d9d9",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 8.8,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#cccccc",
        "figure.titlesize": 13.5,
        "figure.titleweight": "bold",
    })
