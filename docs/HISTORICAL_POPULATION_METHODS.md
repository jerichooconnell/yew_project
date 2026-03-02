# Approaches for Estimating Historical Yew Population

## Problem Statement

We have a validated production model (YewMLP) that predicts current yew
presence at 10m resolution, and BC VRI logging overlays that classify each
pixel as mature forest (>80 yr), logged by age class, alpine, or water.

Yew does not persist in cut forests, so we need a way to estimate what
yew density *would have existed* in logged areas before harvest. This lets
us compare the current population to a hypothetical pre-disturbance baseline.

---

## Option 1: Spatially-Matched Mature Forest Controls

**Method:** For each logged pixel, find the nearest mature forest pixel
(or mean of k-nearest neighbours) and use its production-model probability
as the "what would have been" estimate.

**Strengths:**
- Uses the validated production model — no second model needed
- Captures fine-scale spatial variation (valley floor vs ridge)
- Ecologically defensible: adjacent undisturbed forest is the best proxy

**Weaknesses:**
- Computationally expensive (nearest-neighbour search at pixel level)
- Logged areas far from mature forest may get poor matches
- Edge effects: mature forest adjacent to clearcuts may itself be degraded

**Data needed:** Existing tile cache (grid + logging arrays)

---

## Option 2: Elevation × Latitude Stratification

**Method:** Bin all mature forest pixels by elevation band (e.g., 100m
intervals) and latitude zone. Compute the P≥0.95 rate for each bin, then
apply those rates to logged pixels in the same elevation/latitude bin.

**Strengths:**
- Captures the strong elevational and latitudinal gradient in yew density
- Simple to implement and interpret
- Avoids over-fitting to local spatial patterns

**Weaknesses:**
- Requires elevation data (DEM) for each pixel — would need a GEE extraction
- Assumes elevation is the main driver (ignores aspect, soil, moisture)
- Coarse bins may miss important within-bin variation

**Data needed:** DEM rasters (not currently in tile cache — new GEE extraction)

---

## Option 3: BEC Subzone Stratification

**Method:** Use BC's biogeoclimatic subzones (CWHvm, CWHxm, CWHdm, etc.)
as strata. Compute yew rate in mature forest per subzone, then apply to
logged area in the same subzone.

**Strengths:**
- Ecologically meaningful — BEC subzones are defined by climate and vegetation
- Widely used in BC forestry and ecology — results are interpretable
- BEC polygons available from DataBC at high resolution

**Weaknesses:**
- Requires BEC polygon overlay (new spatial join, but straightforward)
- Some subzones may have very few mature forest pixels → unreliable rates
- Ignores within-subzone variation (e.g., valley bottom vs slope)

**Data needed:** BEC subzone polygons (available from DataBC, ~50 MB shapefile)

---

## Option 4: Buffer-Ring Analysis Around Logged Polygons

**Method:** For each logged VRI polygon, measure yew density in a 1–5 km
ring of mature forest around it. Use that ring's P≥0.95 rate as the
pre-logging baseline for that polygon.

**Strengths:**
- Most ecologically defensible — adjacent undisturbed forest is the best
  control for what was there before logging
- Naturally accounts for all local environmental factors
- Individual polygon-level estimates

**Weaknesses:**
- Complex implementation (polygon buffering + spatial indexing)
- Some logged polygons may have no mature forest within the buffer
- Ring forests may not be representative if logging was selective

**Data needed:** VRI polygon geometries (already have GDB) + spatial ops

---

## Option 5: Tile-Level Matched Controls ★ IMPLEMENTED

**Method:** For each of the 35 spot tiles (10×10 km), compute the P≥0.95
yew rate in mature forest (>80 yr) within that tile, then apply it to the
logged area in the *same tile*. Each tile acts as its own local control.

**Strengths:**
- Uses the validated production model — no second model needed
- Naturally captures the north–south gradient (each tile gets its own rate)
- Simple, transparent, and reproducible
- All data already cached — no new extraction required
- Per-tile rates are verifiable against the spot comparison maps

**Weaknesses:**
- Assumes logged areas within a 10×10 km tile had similar yew density to
  the remaining mature forest in that tile (reasonable at this scale)
- Some tiles may have very little mature forest → noisy rate estimates
- 10×10 km resolution is coarser than pixel-level matching

**Data needed:** Existing tile cache (already have all 35 grids + logging)

**Implementation:** `scripts/analysis/yew_decline_tile_matched.py`

---

## Comparison of Approaches

| Approach | Spatial Resolution | New Data Needed | Complexity | Ecological Validity |
|----------|-------------------|-----------------|------------|---------------------|
| 1. Nearest-pixel | Pixel (10m) | None | High | Very high |
| 2. Elev × Lat | ~100m bins | DEM rasters | Medium | Medium |
| 3. BEC subzone | Subzone (~km) | BEC polygons | Medium | High |
| 4. Buffer rings | Polygon (~ha) | VRI geometry | High | Very high |
| 5. Tile-matched | Tile (10 km) | None | Low | Good |

## Recommendation

Start with **Option 5** (tile-matched) as the primary estimate — it's
simple, uses trusted data, and naturally handles the large latitudinal
gradient. If finer spatial resolution is needed, follow up with **Option 4**
(buffer rings) which is the most defensible ecologically but requires more
complex spatial operations.
