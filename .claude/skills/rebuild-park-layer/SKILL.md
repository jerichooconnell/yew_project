---
name: rebuild-park-layer
description: Rebuild the web-map protected-areas layer (docs/tiles/park_contours.geojson) at high resolution from BC Data Catalogue WFS. Use to refresh parks/conservancies/national-park boundaries.
---

# Rebuild the protected-areas map layer

```bash
conda run -n yew_pytorch python scripts/analysis/build_park_contours.py
# optional: --tol 0.0001  (simplify tolerance in degrees; default 0.00015 ≈ 17 m)
```

Pulls three full-resolution BC Data Catalogue WFS layers and merges them into
`docs/tiles/park_contours.geojson`:
- `WHSE_TANTALIS.TA_PARK_ECORES_PA_SVW` (provincial parks / ecological reserves /
  protected areas / recreation areas)
- `WHSE_TANTALIS.TA_CONSERVANCY_AREAS_SVW` (conservancies)
- `WHSE_ADMIN_BOUNDARIES.CLAB_NATIONAL_PARKS` (national parks in BC)

Notes:
- ~1,200 features at tol 0.00015 → ~7 MB (≈2.4 MB gzipped). Lower `--tol` = sharper +
  bigger; the previous hand-built layer was ~38 verts/feature (visibly blocky).
- The web map (`docs/index.html`, `toggleParksLayer`) styles by
  `PROTECTED_LANDS_DESIGNATION`: national park = pumpkin, provincial = green,
  conservancy = purple, protected area = dark green, ecological reserve = teal.
  Keep the legend (`#parks-legend`) in sync if designations change.
- Overpass/OSM is unreachable in this environment — national parks come from the BC
  WFS layer above, not OSM.
- After rebuilding, refresh the protected-area statistic with the
  `recompute-protected-area` skill.
