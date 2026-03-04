# CWH/CDF Random Sampling — XGBoost Yew Probability Estimation

**Script:** `scripts/prediction/sample_cwh_xgb_probability.py`  
**Model:** XGBoost (`xgb_raw_model_expanded`) — GPU-accelerated inference on raw 64-band Prithvi embeddings  
**Boundary:** `data/processed/cwh_cdf_land.shp` — water-clipped CWH + CDF land polygons (NE 10m coastline)

---

## 1. Purpose

Rather than classifying every 10 m pixel across the ~35,000 km² CWH/CDF area (computationally
prohibitive), this script draws a **stratified random sample** of *N* geographic points and
estimates yew probability at each one using the same XGBoost classifier that drives the
tile-matched methodology. With GEE extraction enabled (`--gee-extract`), all *N* points can
be fully classified in a single run — 10,000 points takes roughly 5–15 minutes of GEE time.

---

## 2. Sampling Strategy

### 2.1 Source polygon

Points are sampled from `cwh_cdf_land.shp`, the combined CWH + CDF land polygon produced by
`scripts/preprocessing/build_cwh_cdf_land_shapefile.py`. This polygon:

- Covers coastal BC's CWH and CDF biogeoclimatic zones
- Has been clipped to the Natural Earth 10 m land boundary (removing ocean and large water bodies)
- Includes a gap-filled Haida Gwaii extension (CWHvh3, ~11,700 km²) where the source GDB had
  geometry holes

### 2.2 Vectorised area-weighted rejection sampling

1. Each polygon's area (in EPSG:3005 equal-area metres) is computed.
2. A batch of candidate polygons is selected at once with probability proportional to area.
3. One candidate point per polygon is drawn uniformly within the polygon's bounding box.
4. All candidates are tested against their polygon in one call using `shapely.contains_xy()`
   (vectorised C kernel) rather than looping over individual `geom.contains(Point(...))` calls.

**Performance:** The largest CWH polygon has 129,687 exterior vertices. The old point-by-point
approach took ~56 ms per candidate (2 min for 1,000 points). The vectorised batch approach runs
at **~5,000 accepted points/sec** — a 600× speedup. 10,000 points are sampled in ~2 seconds.

---

## 3. Embedding Extraction

### 3.1 Path A — tile cache (instant)

If a sampled point falls within the bounding box of one of the 35 downloaded 10×10 km tiles,
its 64-band embedding is read directly from `tile_cache/{slug}_emb.npy` at the corresponding pixel.

Coordinate → pixel mapping (0-indexed, north-to-south rows, west-to-east columns):
```
south, north, west, east = centre_to_bbox(tile_lat, tile_lon, km=10)
row = int((north − pt_lat) / (north − south) × H)
col = int((pt_lon  − west)  / (east  − west)  × W)
```
These points are also cross-verified against `_grid.npy` (max observed diff: 1.8×10⁻⁷ — float32 rounding only).

### 3.2 Path B — GEE `sampleRegions` (fast, server-side)

For all remaining points the script calls `ee.Image.sampleRegions()` on the
`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` mosaic. This evaluates the 64-band image
**server-side** at each point and returns only the pixel values:

```python
ee_img = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
           .filterDate('2024-01-01', '2025-01-01')
           .mosaic()
           .select([f'A{i:02d}' for i in range(64)]))

sampled = ee_img.sampleRegions(collection=fc, scale=10, geometries=False, tileScale=4)
info    = sampled.getInfo()   # returns list of feature dicts with band values
```

**Why this is fast:** The old tile-download approach moved ~410 MB per 10×10 km tile (1000×1600×64×4
bytes) over the wire. `sampleRegions` for 500 points transfers only ~128 KB (500 × 64 × 4 bytes)
per batch — a **~3,200× reduction** in payload. GEE does all the spatial interpolation on its servers
before sending results.

Points are processed in batches of `--gee-batch-size` (default 500) to stay within GEE's
`getInfo()` response size limit (~10 MB). Each batch takes 15–45 seconds of GEE time, so
10,000 points (20 batches) completes in roughly **5–15 minutes**.

---

## 4. Classification

All resolved embeddings (tile cache + GEE) are classified with:

```python
bst   = xgb.Booster(); bst.load_model('xgb_raw_model_expanded.json')
dmat  = xgb.DMatrix(embs)     # (N, 64) float32
probs = bst.predict(dmat)     # (N,) float32 — P(yew present)
```

XGBoost uses GPU inference if CUDA is available, otherwise CPU. Classification of 10,000
points takes < 1 second regardless.

---

## 5. Output Files

| File | Description |
|------|-------------|
| `results/predictions/cwh_xgb_sample.csv` | One row per sampled point |
| `results/figures/cwh_xgb_sample_map.html` | Interactive Folium map |

### CSV columns

| Column | Description |
|--------|-------------|
| `lat`, `lon` | WGS-84 coordinates |
| `bec_zone` | `CWH` or `CDF` |
| `map_label` | BEC subzone (e.g. `CWHvh3`) |
| `in_tile` | Tile name, `"(GEE)"`, or `NaN` if unresolved |
| `px_row`, `px_col` | Pixel indices if from tile cache |
| `prob` | XGBoost P(yew), or `NaN` if GEE failed for this point |
| `cached_prob` | Pre-computed grid value (tile-cache points only, for verification) |
| `prob_diff` | `|prob − cached_prob|` |
| `verified` | `True` if diff ≤ 1×10⁻⁴ |

---

## 6. Map

`cwh_xgb_sample_map.html` shows:

- **Green fill:** CWH zone polygons
- **Teal fill:** CDF zone polygons
- **Black rectangles:** Bounding boxes of the 35 downloaded 10×10 km study tiles
- **Coloured circles (large):** Classified points, coloured by P(yew):
  - Dark blue (< 0.20) → light blue (0.20–0.35) → yellow (0.35–0.50) → orange (0.50–0.65) → red (≥ 0.65)
- **Grey circles (small):** Points where GEE returned no data (cloud cover, image gaps)
- Clickable popups show lat/lon, BEC label, probability, and pixel coordinates

---

## 7. Results (N = 10,000, seed = 42, GEE year = 2024)

| Metric | Value |
|--------|-------|
| Total points sampled | 10,000 |
| From tile cache | 217 (2.2%) |
| From GEE `sampleRegions` | 9,783 (97.8%) |
| GEE 100% resolved (0 missing) | ✓ |
| Total classified | **10,000 (100%)** |
| GEE extraction time | ~96 s (20 batches × 500 pts) |
| **Mean P(yew)** | **0.195** |
| **Median P(yew)** | **0.062** |
| P ≥ 0.30 | 2,426 / 10,000 (24.3%) |
| P ≥ 0.50 | 1,469 / 10,000 (14.7%) |
| P ≥ 0.70 | 853 / 10,000 (8.5%) |
| P ≥ 0.90 | 310 / 10,000 (3.1%) |
| Tile-cache verification | ✓ 217/217 matched (max diff 1.8×10⁻⁷) |

The skewed distribution (mean 0.195, median 0.062) is consistent with yew being a relatively
rare understory species: the majority of the CWH/CDF landscape scores < 0.10, with a long tail
of high-probability pixels corresponding to old-growth riparian and south-facing slope habitats.

---

## 8. Relationship to Tile-Matched Methodology

The tile-matched method (see `TILE_MATCHED_METHODOLOGY.md`) estimates historical and current
yew density at the **pixel** level within each of the 35 pre-selected 10×10 km tiles, using
the VRI logging mask to separate mature forest from logged areas.

| Aspect | Tile-matched | Random sampling |
|--------|-------------|-----------------|
| Spatial coverage | 35 fixed tiles (~3,500 km²) | Province-wide CWH+CDF (~35,000+ km²) |
| Sampling | Every pixel within each tile | N random points, area-weighted |
| Logging mask | Yes — VRI categories 2/3/4/5 | Not yet applied |
| Population estimate | Extrapolated from tile densities | Direct from sample mean × area |
| Bias risk | Tile selection bias | None (uniform random) |
| Embedding source | `tile_cache/*_emb.npy` (raster download) | `sampleRegions` (point extraction) |
| Time for 10k points | N/A (all pixels) | ~10 min GEE + <1 s XGBoost |

Once the full 10,000-point GEE extraction is done, the sample mean P(yew) × total CWH+CDF
land area gives a province-wide yew area estimate independent of the 35-tile extrapolation.

---

## 9. Usage

```bash
# Full 10,000-point run with GEE download (recommended):
python scripts/prediction/sample_cwh_xgb_probability.py \
    --n-samples 10000 --gee-extract --gee-year 2024

# Quick local-only run (no GEE — unclassified points left as NaN):
python scripts/prediction/sample_cwh_xgb_probability.py --n-samples 1000

# Verify coordinate mapping against cached grids only:
python scripts/prediction/sample_cwh_xgb_probability.py --verify-only

# Custom batch size or year:
python scripts/prediction/sample_cwh_xgb_probability.py \
    --n-samples 5000 --gee-extract --gee-batch-size 250 --gee-year 2023
```

---

## 10. Next Steps

1. **Apply VRI logging mask** — filter points to mature forest (VRI category 5) to match the
   tile-matched assumption that `P(yew) = 0` in logged areas < 80 yr.
2. **Stratify by latitude zone** — compare South/Central/North distributions with tile-matched zone results.
3. **Scale to province-wide estimate** — multiply `mean P(yew) × CWH_area_ha` corrected for
   logging fraction to get independent population estimate.
4. **Increase N to 30,000+** — reduces margin of error to ≤ 1% at 95% confidence.


**Script:** `scripts/prediction/sample_cwh_xgb_probability.py`  
**Model:** XGBoost (`xgb_raw_model_expanded`) — GPU-accelerated inference on raw 64-band Prithvi embeddings  
**Boundary:** `data/processed/cwh_cdf_land.shp` — water-clipped CWH + CDF land polygons (NE 10m coastline)

---

## 1. Purpose

Rather than classifying every 10 m pixel across the ~35,000 km² CWH/CDF area (computationally
prohibitive), this script draws a **stratified random sample** of *N* geographic points and
estimates yew probability at each one using the same XGBoost classifier that drives the
tile-matched methodology. It is the first step toward a province-wide yew density estimate
that is not anchored to the 35 pre-selected study tiles.

---

## 2. Sampling Strategy

### 2.1 Source polygon

Points are sampled from `cwh_cdf_land.shp`, the combined CWH + CDF land polygon produced by
`scripts/preprocessing/build_cwh_cdf_land_shapefile.py`. This polygon:

- Covers coastal BC's CWH and CDF biogeoclimatic zones
- Has been clipped to the Natural Earth 10 m land boundary (removing ocean and large water bodies)
- Includes a gap-filled Haida Gwaii extension (CWHvh3, ~11,700 km²) where the source GDB had
  geometry holes

### 2.2 Area-weighted random sampling

1. Each polygon's area (in EPSG:3005 equal-area metres) is computed.
2. A polygon is selected with probability proportional to its area, so large polygons receive
   proportionally more sample points.
3. A candidate point is drawn uniformly within the polygon's bounding box; it is accepted if
   `polygon.contains(point)` (rejection sampling). Typical acceptance rate: ~13%.
4. Accepted points are re-projected to WGS-84 (EPSG:4326) for tile lookup and GEE extraction.

This produces a **spatially representative** sample — the probability that any given hectare is
sampled is identical across the entire CWH/CDF area.

---

## 3. Classification

### 3.1 Model

The classifier is `xgb_raw_model_expanded.json` — a GPU-accelerated XGBoost Booster trained on
raw 64-dimensional Prithvi satellite embeddings (no scaler). This is the same model used in the
tile-matched pipeline (`--classifier xgb_raw_expanded` in `classify_cwh_spots.py`).

### 3.2 Two-path classification

**Path A — cached tile (fast):**  
If a sampled point falls within the bounding box of one of the 35 downloaded 10×10 km tiles,
its embedding is read directly from `tile_cache/{slug}_emb.npy` at the corresponding pixel.

Coordinate → pixel mapping:
```
south, north, west, east = centre_to_bbox(tile_lat, tile_lon, km=10)
row = int((north − pt_lat) / (north − south) × H)   # 0-indexed, north → south
col = int((pt_lon  − west)  / (east  − west)  × W)   # 0-indexed, west → east
```

XGBoost is then run on the single 64-element embedding vector via `xgb.DMatrix`.

**Path B — outside tile (pending GEE extraction):**  
Points that fall outside all 35 cached tiles are saved with `prob = NaN`. Their lat/lon
coordinates are written to `results/predictions/cwh_xgb_sample.csv` for a subsequent batch
Google Earth Engine extraction using `extract_ee_imagery_fast.py`.

### 3.3 Verification

For every in-tile point, the freshly recomputed XGBoost probability is compared against the
pre-stored value in `tile_cache/{slug}_grid.npy`. All discrepancies have been ≤ 1.8×10⁻⁷ —
purely float32 rounding noise — confirming that the coordinate-to-pixel mapping is exact.

---

## 4. Output Files

| File | Description |
|------|-------------|
| `results/predictions/cwh_xgb_sample.csv` | One row per sampled point |
| `results/figures/cwh_xgb_sample_map.html` | Interactive Folium map |

### CSV columns

| Column | Description |
|--------|-------------|
| `lat`, `lon` | WGS-84 coordinates of sampled point |
| `bec_zone` | BEC zone code (`CWH` or `CDF`) |
| `map_label` | BEC subzone label (e.g. `CWHvh3`) |
| `in_tile` | Name of the downloaded tile containing this point, or `NaN` |
| `px_row`, `px_col` | Pixel indices within the tile array (0-indexed) |
| `prob` | XGBoost P(yew) recomputed from embedding, or `NaN` if outside tile |
| `cached_prob` | P(yew) from pre-stored `_grid.npy`, for verification |
| `prob_diff` | \|prob − cached_prob\| |
| `verified` | `True` if diff ≤ 1×10⁻⁴, `False` or `NaN` otherwise |

---

## 5. Map

`cwh_xgb_sample_map.html` shows:

- **Green fill:** CWH zone polygons (Coastal Western Hemlock)
- **Teal fill:** CDF zone polygons (Coastal Douglas-fir)
- **Black rectangles:** Bounding boxes of the 35 downloaded 10×10 km study tiles
- **Coloured circles:** Classified points (larger, 85% opacity), coloured by P(yew):
  - Dark blue → light blue → yellow → orange → red (low → high)
- **Grey circles:** Unclassified points (smaller, 45% opacity) awaiting GEE extraction
- Clickable popups show lat/lon, BEC label, probability, and pixel coordinates

---

## 6. Current Results (N = 1000, seed = 42)

| Metric | Value |
|--------|-------|
| Total points sampled | 1,000 |
| Points inside cached tiles | 20 (2.0%) |
| Points needing GEE extraction | 980 (98.0%) |
| Acceptance rate (rejection sampling) | 13.1% |
| Mean P(yew) — in-tile only | 0.245 |
| Median P(yew) — in-tile only | 0.253 |
| P ≥ 0.30 | 7 / 20 (35%) |
| P ≥ 0.50 | 2 / 20 (10%) |
| Verification: all in-tile matched | ✓ 20 / 20 |

The low in-tile fraction (2%) reflects that the 35 study tiles cover roughly 3,500 km² of the
~35,000+ km² CWH/CDF area. A GEE batch extraction for the remaining 980 points will bring
the full 1,000-point sample to completion.

---

## 7. Relationship to Tile-Matched Methodology

The tile-matched method (see `TILE_MATCHED_METHODOLOGY.md`) estimates historical and current
yew density at the **pixel** level within each of the 35 pre-selected 10×10 km tiles, using
the VRI logging mask to separate mature forest from logged areas. It is well-suited to
computing decline ratios (L:R) and separating the logging impact.

This random sampling method is **complementary**:

| Aspect | Tile-matched | Random sampling |
|--------|-------------|-----------------|
| Spatial coverage | 35 fixed tiles (~3,500 km²) | Province-wide CWH+CDF |
| Sampling | Every pixel within each tile | N random points, area-weighted |
| Logging mask | Yes — VRI categories 2/3/4/5 | Not yet (all land) |
| Population estimate | Extrapolated from tile densities | Direct from sample mean |
| Bias risk | Tile selection bias | None after VRI filter |
| Status | Complete | Partially classified (2%) |

Once the full 1,000-point GEE extraction is done, the sample mean P(yew) can be multiplied by the
total CWH+CDF land area to give an independent province-wide yew area estimate that corroborates
(or tests) the tile-matched extrapolation.

---

## 8. Next Steps

1. **GEE batch extraction** for the 980 unclassified points:
   use `scripts/preprocessing/extract_ee_imagery_fast.py` with the lat/lon from
   `cwh_xgb_sample.csv` where `prob` is NaN.
2. **Apply VRI logging mask** to separate mature forest from logged pixels, matching the
   tile-matched assumption that `P(yew) = 0` in logged areas.
3. **Increase N** to 3,000–5,000 for a margin of error ≤ 1.8% at 95% confidence.
4. **Stratify by latitude zone** (South / Central / North) to compare with the tile-matched
   zone results.
