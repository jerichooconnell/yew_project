# CWH Yew Mapping Workflow

**Project:** Spatial prediction of Pacific yew (*Taxus brevifolia*) habitat in the Coastal Western Hemlock (CWH) BEC zone, British Columbia  
**Last updated:** February 2026  

---

## Overview

The pipeline uses Google satellite spectral embeddings as input features to a trained MLP classifier, with VRI forestry data applied as a post-classification mask. The result is a pixel-probability map of yew habitat suitability that accounts for stand age, harvest history, alpine/barren terrain, and water bodies.

---

## 1. Study Area

**Zone:** Coastal Western Hemlock (CWH) BEC zone  
**Area:** ~3.6 million ha  
**Extent:** Lat 48.4–55.4°N, Lon 122.0–132.6°W  
**Boundary file:** `data/processed/cwh_negatives/cwh_boundary_forestry.gpkg`  
- Single dissolved MultiPolygon (38 parts), EPSG:4326  
- Built from BC VRI GDB (`VEG_COMP_LYR_R1_POLY_2024.gdb`) BEC_ZONE_CODE = 'CWH'  
- Script: `scripts/preprocessing/build_cwh_boundary_from_vri.py`

> **Important:** The BC government WFS endpoint for BEC zones (`openmaps.gov.bc.ca`) currently returns a 400 error. Always pass `--boundary data/processed/cwh_negatives/cwh_boundary_forestry.gpkg` to avoid the approximate-polygon fallback.

---

## 2. Spectral Embeddings

**Source:** Google Earth Engine — `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`  
**Bands:** 64 floats (A00–A63) per pixel, derived from Sentinel-2 annual composites  
**Scale:** 10 m (tile inference) or 300 m (BC-wide sampling)  
**Year:** 2024

### 2a. South Vancouver Island training tiles (dense 10 m)

Downloaded as a 4×7 grid of tiles (28 total) covering south Vancouver Island:

```bash
conda run -n yew_pytorch python scripts/prediction/classify_tiled_area.py \
    --bbox 48.2728 48.7014 -124.5086 -123.1969 \
    --output-dir results/predictions/south_vi_large \
    --year 2024 --scale 10
```

Tiles are cached as `emb_R_C.npy` files in `results/predictions/south_vi_large/`.  
RGB mosaic saved as `rgb_image.npy`; full extent metadata in `metadata.json`.

### 2b. BC-wide CWH sample (300 m)

Use `resample_cwh_100k.py` which samples 100k points **within the actual CWH polygon**:

```bash
conda run -n yew_pytorch python scripts/prediction/resample_cwh_100k.py \
    --n-samples 100000 \
    --year 2024 \
    --model-dir results/predictions/south_vi_large \
    --output-dir results/analysis/cwh_yew_population_100k
```

Embeddings cached to `embeddings_cwh_100000_seed42.csv`. Add `--skip-extract` on reruns to skip GEE and reuse the cache.

---

## 3. Training Data

| Source | File | Count | Weight |
|---|---|---|---|
| iNaturalist yew positives (train) | `data/processed/inat_yew_positives_train.csv` | 834 | 1 |
| iNaturalist yew positives (val) | `data/processed/inat_yew_positives_val.csv` | 209 | — |
| Manual annotations (yew / not-yew) | `data/raw/yew_annotations_combined.csv` | 267 | 3 |
| FAIB forestry negatives | `data/processed/faib_negatives/faib_negative_embeddings.csv` | 6,194 | 1 |
| Alpine hard-negatives (cat-6 pixels, prob>0.2) | `data/processed/alpine_negatives/alpine_negative_embeddings.csv` | 2,800 | 1 |
| **Combined negatives** | `data/processed/combined_negative_embeddings.csv` | **8,994** | 1 |

Alpine negatives were extracted from the 15 CWH spot tile caches using:

```bash
conda run -n yew_pytorch python scripts/preprocessing/extract_alpine_negatives.py
```

---

## 4. Model

**Architecture:** YewMLP — fully connected, 64→128→64→32→1  
**Loss:** Binary cross-entropy with sigmoid output  
**Saved to:** `results/predictions/south_vi_large/mlp_model.pth`  
**Scaler:** `results/predictions/south_vi_large/mlp_scaler.pkl`

### Training command

```bash
conda run -n yew_pytorch python scripts/prediction/classify_tiled_gpu.py \
    --input-dir results/predictions/south_vi_large \
    --train-csv data/processed/inat_yew_positives_train.csv \
    --val-csv data/processed/inat_yew_positives_val.csv \
    --annotations data/raw/yew_annotations_combined.csv \
    --annotation-weight 3 \
    --gee-negatives data/processed/combined_negative_embeddings.csv \
    --gee-negatives-weight 1 \
    --epochs 100
```

### Current model metrics (February 2026)

| Metric | Value |
|---|---|
| Accuracy | 98.85% |
| F1 score | 0.9471 |
| AUC-ROC | 0.9980 |

---

## 5. VRI Logging Mask

After classification, raw pixel probabilities are multiplied element-wise by a VRI suitability factor derived from BC's Vegetation Resources Inventory (VEG_COMP_LYR_R1_POLY_2024.gdb).

**GDB file:** `data/VEG_COMP_LYR_R1_POLY_2024.gdb`  
**Layer:** `VEG_COMP_LYR_R1_POLY` (6.87M features, EPSG:3005)

### 5.1 VRI fields used

| Field | Purpose |
|---|---|
| `BCLCS_LEVEL_1` / `BCLCS_LEVEL_2` | Land cover class (water, non-vegetated, alpine) |
| `PROJ_AGE_CLASS_CD_1` | Stand age class 1–9 (92% polygon coverage) — **primary age source** |
| `PROJ_AGE_1` | Explicit projected stand age (58% coverage) |
| `HARVEST_DATE` | Recorded logging date |
| `LINE_7B_DISTURBANCE_HISTORY` | Coded disturbance events, e.g. `B23;L14` (burn 2023, logged 2014) |
| `OPENING_IND` | `'Y'` = aerially confirmed harvest via BC RESULTS system |
| `OPENING_SOURCE` | Cutblock database source: 3=RESULTS, 4=imagery, 7=harvest, 11=silviculture |
| `ALPINE_DESIGNATION` | `'A'` = officially designated alpine zone |

### 5.2 Age resolution logic

For each VRI polygon the minimum stand age is taken across all available sources (most recent disturbance wins):

```
age = min(
    PROJ_AGE_CLASS_CD_1 midpoint,
    PROJ_AGE_1,
    current_year − HARVEST_DATE.year,
    years_since_LINE_7B_event
)
```

`PROJ_AGE_CLASS_CD_1` uses midpoint values: 1→10 yr, 2→30 yr, 3→50 yr, 4→70 yr, 5→90 yr, 6→110 yr, 7→130 yr, 8→195 yr, 9→300 yr.

`LINE_7B_DISTURBANCE_HISTORY` is parsed for all event codes (B=burn, L=log, W=wind, I=insect, D=disease) with 2-digit year → 4-digit century-aware conversion.

If `OPENING_IND='Y'` or `OPENING_SOURCE` ∈ {3,4,7,11} but no age is available from any other field, the polygon is assigned age=0 (conservative: treat as recent disturbance).

### 5.3 Suppression categories

| Cat | Condition | Suppression factor | Visual colour |
|---|---|---|---|
| 1 | Water / non-forest | ×0.00 | Blue |
| 2 | Logged <20 yr | ×0.00 | Red |
| 3 | Logged 20–40 yr | ×0.08 | Orange |
| 4 | Logged 40–80 yr | ×0.30 | Yellow |
| 5 | Forest >80 yr / unlogged | ×1.00 | Green |
| 6 | Alpine / barren | ×0.00 | Tan |

Category 6 is assigned when: `BCLCS_LEVEL_1='N'` + `BCLCS_LEVEL_2='L'` (rock/rubble), `ALPINE_DESIGNATION='A'`, or `BCLCS_LEVEL_2='N'` with no harvest/age record.

### 5.4 Implementation

```bash
# Reclassify and regenerate all 15 CWH spot maps
conda run -n yew_pytorch python scripts/prediction/classify_cwh_spots.py \
    --force-reclassify
```

The logging raster for each spot is cached in `results/analysis/cwh_spot_comparisons/tile_cache/*_logging.npy`.  
Delete those files before running if VRI classification logic has changed.

---

## 6. CWH Spot Comparison Maps (15 areas)

`classify_cwh_spots.py` generates comparison maps for 15 known CWH localities:

| # | Area | Centre |
|---|---|---|
| 1 | Carmanah-Walbran | 48.44°N, 124.16°W |
| 2 | Sooke Hills | 48.60°N, 123.80°W |
| 3 | Clayoquot Sound | 49.32°N, 124.98°W |
| 4 | Campbell River Uplands | 50.02°N, 125.24°W |
| 5 | Quatsino Sound | 50.70°N, 127.10°W |
| 6 | Squamish Highlands | 49.70°N, 123.15°W |
| 7 | Desolation Sound | 50.72°N, 124.00°W |
| 8 | Bella Coola Valley | 52.33°N, 126.60°W |
| 9 | Prince Rupert Hills | 54.15°N, 129.70°W |
| 10 | Kitimat Ranges | 53.50°N, 128.60°W |
| 11 | Strathcona Highlands | 49.90°N, 125.55°W |
| 12 | Garibaldi Foothills | 49.86°N, 122.68°W |
| 13 | Bute Inlet Slopes | 50.83°N, 124.92°W |
| 14 | Nanaimo Lakes | 49.02°N, 124.20°W |
| 15 | Rivers Inlet | 51.40°N, 127.70°W |

Each area produces:
- An HTML comparison map with VRI logging overlay (`results/analysis/cwh_spot_comparisons/*.html`)
- A spot summary PNG thumbnail
- Stats: mean probability, max probability, area with P≥0.5 (ha)

---

## 7. BC-Wide KMZ Map

From sample predictions, a KMZ ground overlay is generated for Google Earth:

```bash
conda run -n yew_pytorch python scripts/visualization/create_cwh_kmz.py \
    --input results/analysis/cwh_yew_population_100k/sample_predictions_cwh.csv \
    --output results/analysis/cwh_yew_population_100k/cwh_yew_100k.kmz
```

The script rasterises point predictions to a 0.03° grid (~3 km), applies a NaturalEarth land mask to remove ocean points, and writes a KMZ with a colour-coded RGBA overlay.

---

## 8. Population Estimate

The BC-wide estimated yew habitat area (P≥0.5 threshold) as of February 2026 run:

| Threshold | Estimated area | 95% CI |
|---|---|---|
| P≥0.3 | ~330,000 ha | ±16,000 ha |
| **P≥0.5** | **~245,000 ha** | **±14,000 ha** |
| P≥0.7 | ~179,000 ha | ±12,000 ha |

These estimates are based on 16,139 CWH-filtered points from the 2024 300k sample. The 100k re-sample (run Feb 2026) will update these figures.

> **Note:** Estimates apply exclusively to the CWH BEC zone (~3.6M ha). Areas outside CWH (interior BC, Coast Mountains, Haida Gwaii alpine) are not included.

---

## 9. Key Files Reference

| File | Description |
|---|---|
| `data/processed/cwh_negatives/cwh_boundary_forestry.gpkg` | Dissolved CWH boundary, 38-part MultiPolygon |
| `data/VEG_COMP_LYR_R1_POLY_2024.gdb` | BC VRI geodatabase (6.87M polygons) |
| `results/predictions/south_vi_large/mlp_model.pth` | Trained MLP weights |
| `results/predictions/south_vi_large/mlp_scaler.pkl` | Feature scaler |
| `results/predictions/south_vi_large/metadata.json` | South VI tile extent |
| `data/processed/combined_negative_embeddings.csv` | 8,994 training negatives |
| `data/raw/yew_annotations_combined.csv` | 267 manual annotations |
| `results/analysis/cwh_yew_population_100k/` | 100k re-sample outputs |
| `results/analysis/cwh_spot_comparisons/` | 15-area spot comparison maps |

---

## 10. Conda Environment

All scripts use the `yew_pytorch` conda environment:

```bash
conda run -n yew_pytorch python <script>
```

Key packages: `torch 2.1.0+cu118`, `geopandas`, `rasterio`, `pyproj`, `earthengine-api`, `folium`.  
Environment file: `config/yew_pytorch_env.yml`

---

## 11. Known Issues / Limitations

1. **BC WFS download** — `openmaps.gov.bc.ca` BEC WFS returns a 400 error. Always use `--boundary data/processed/cwh_negatives/cwh_boundary_forestry.gpkg`.
2. **`sample_cwh_yew_population.py` tiled fallback** — when the WFS fails the old script falls back to an approximate bounding-box polygon, inflating the zone area ~7×. Use `resample_cwh_100k.py` instead for BC-wide estimates.
3. **PROJ_AGE_1 pre-disturbance bias** — for recently burned/logged stands, `PROJ_AGE_1` stores the pre-disturbance age. Always ensure `LINE_7B_DISTURBANCE_HISTORY` is parsed alongside it; the classifier takes the minimum age.
4. **Sample size vs precision** — at 300 m scale, 100k points within 3.6M ha gives one point per ~36 ha. The margin of error at P≥0.5 is approximately ±0.5% of CWH area (±18,000 ha).
