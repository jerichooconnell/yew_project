---
name: recompute-protected-area
description: Recompute the fraction of yew habitat inside protected areas across all 99 study tiles. Use when the parks layer changes or the protected-area statistic needs refreshing.
---

# Recompute protected-area coverage (all 99 tiles)

Only 42 of 99 tiles have raw `.npy` grids; the rest exist only as suppressed-
probability PNGs in `docs/tiles/`. The all-tiles recompute recovers the habitat mask
from those PNGs (inverts the export colormap, validated to ~1% of stored `p50_ha`),
intersects with the protected-area layer, and reports provincial-only vs
all-designation coverage.

```bash
# 1. (optional) refresh the parks layer from BC WFS
conda run -n yew_pytorch python scripts/analysis/build_park_contours.py

# 2. recompute over all 99 tiles -> results/analysis/protected_area_all_tiles.json
conda run -n yew_pytorch python scripts/analysis/protected_area_from_tiles.py
```

Notes:
- Use the `yew_pytorch` env (needs geopandas/rasterio/PIL/matplotlib).
- The colormap inversion only processes non-transparent pixels (memory-light); do
  NOT broadcast the full (N×256×3) distance array — it OOMs on ~1.5M-px tiles.
- Protected *fraction* is a pixel ratio (independent of the 0.01 ha/px convention).
- Inputs are all committed (PNGs + `tiles.json` + `park_contours.geojson`) — no GEE.
- `scripts/analysis/yew_in_protected_areas.py` is the OLD version (42-tile subset,
  biased) — prefer `protected_area_from_tiles.py`.
