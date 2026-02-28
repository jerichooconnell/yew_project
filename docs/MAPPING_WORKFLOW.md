# CWH Yew Mapping Workflow

**Project:** Spatial prediction of Pacific yew (*Taxus brevifolia*) habitat in the coastal zone of British Columbia  
**Last updated:** February 2026  

---

## Overview

The pipeline uses Google satellite spectral embeddings as input features to a trained MLP classifier, with VRI forestry data applied as a post-classification mask. The result is a pixel-probability map of yew habitat suitability that accounts for stand age, harvest history, alpine/barren terrain, and water bodies.

---

## 1. Study Areas

Two study area definitions have been developed:

### 1a. CWH BEC Zone (original, patchy boundary)

**Zone:** Coastal Western Hemlock (CWH) Biogeoclimatic Ecosystem Classification zone  
**Area:** ~3.6 million ha  
**Extent:** Lat 48.4–55.4°N, Lon 121.0–132.6°W  
**Boundary file:** `data/processed/cwh_negatives/cwh_boundary_forestry.gpkg`  
- Single dissolved MultiPolygon (38 parts), EPSG:4326  
- Built from BC VRI GDB (`VEG_COMP_LYR_R1_POLY_2024.gdb`) BEC_ZONE_CODE = 'CWH'  
- Script: `scripts/preprocessing/build_cwh_boundary_from_vri.py`

> **Issue with this boundary:** The CWH zone is naturally patchy — many small isolated polygons scattered through the interior mountain ranges. This makes uniform sampling difficult and the 38-part boundary misses some coastal fringe areas.

> **WFS note:** The BC government WFS endpoint for BEC zones (`openmaps.gov.bc.ca`) currently returns a 400 error. Always pass `--boundary data/processed/cwh_negatives/cwh_boundary_forestry.gpkg` to avoid the approximate-polygon fallback.

**Population estimate (Feb 2026, 100k sample):**

| Threshold | Estimated area | 95% CI |
|---|---|---|
| P≥0.3 | 411,758 ha | 404,600–418,900 ha |
| **P≥0.5** | **314,416 ha** | **308,117–320,715 ha** |
| P≥0.7 | 232,482 ha | 226,900–238,100 ha |

---

### 1b. Coastal BC Study Region (current, preferred)

A more ecologically coherent boundary built by intersecting the BC provincial outline (Natural Earth 10m admin-1) with a bounding polygon defined by two user-specified corners:

**NE corner:** 54.7786°N, 127.8119°W  
**SE corner:** 49.0325°N, 119.5254°W  
**Area:** ~227,650 km² (22.8 million ha)  
**Boundary file:** `data/processed/coastal_study_region.geojson`  
- 87-part MultiPolygon, EPSG:4326  
- Includes all coastal islands: Haida Gwaii (23 polygons), Vancouver Island, Gulf Islands, Central Coast islands  
- Southern boundary follows actual Canada–US border through Strait of Juan de Fuca (extends to ~48.3°N)

**Boundary definition logic:**
```
East boundary : diagonal line from NE corner → SE corner
South boundary: US–Canada border (below 49th parallel for Vancouver Island)
West boundary : BC coastline including all islands (Natural Earth 10m)
North boundary: latitude of NE corner, westward to coast
```

**Regenerate boundary:**
```bash
conda run -n yew_pytorch python3 -c "
import geopandas as gpd; from shapely.geometry import Polygon; import os
gdf = gpd.read_file('https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip')
bc = gdf[gdf['name']=='British Columbia']
poly = Polygon([(-127.811936,54.778601),(-119.525404,49.032480),(-119.525404,48.0),(-140.0,48.0),(-140.0,54.778601),(-127.811936,54.778601)])
region = gpd.GeoDataFrame(geometry=[bc.geometry.iloc[0].intersection(poly)], crs=4326)
region.to_file('data/processed/coastal_study_region.geojson', driver='GeoJSON')
"
```

**Sample and classify:**
```bash
conda run -n yew_pytorch python scripts/prediction/sample_coastal_region.py \
    --n-samples 100000 \
    --gee-project <your-gee-project> \
    --output-dir results/analysis/coastal_region_100k
```

Add `--skip-extract` on reruns to reuse the cached embeddings CSV.

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

### 2b. CWH zone sample (300 m)

Use `resample_cwh_100k.py` which samples 100k points **within the actual CWH polygon**:

```bash
conda run -n yew_pytorch python scripts/prediction/resample_cwh_100k.py \
    --n-samples 100000 \
    --year 2024 \
    --model-dir results/predictions/south_vi_large \
    --output-dir results/analysis/cwh_yew_population_100k
```

Embeddings cached to `embeddings_cwh_100000_seed42.csv`. Add `--skip-extract` on reruns to skip GEE and reuse the cache.

### 2c. Coastal BC region sample (300 m) — preferred

Use `sample_coastal_region.py` which samples 100k points within the broader coastal BC study region (227,650 km², incl. Haida Gwaii and all of Vancouver Island):

```bash
conda run -n yew_pytorch python scripts/prediction/sample_coastal_region.py \
    --n-samples 100000 \
    --gee-project <your-gee-project> \
    --output-dir results/analysis/coastal_region_100k
```

Embeddings cached to `embeddings_coastal_100000_seed42.csv`. Add `--skip-extract` on reruns.

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

## 7. KMZ Map

From sample predictions, a KMZ ground overlay is generated for Google Earth:

```bash
conda run -n yew_pytorch python scripts/visualization/create_cwh_kmz.py \
    --input results/analysis/cwh_yew_population_100k/sample_predictions_cwh.csv \
    --output results/analysis/cwh_yew_population_100k/cwh_yew_100k.kmz
```

The script rasterises point predictions to a 0.03° grid (~3 km), applies a NaturalEarth land mask to remove ocean points, and writes a KMZ with a colour-coded RGBA overlay.

## 7b. Static PNG Map

```bash
conda run -n yew_pytorch python scripts/visualization/generate_png_map.py
```

Outputs a 200 dpi map to `results/analysis/cwh_yew_population_100k/cwh_yew_100k_map.png` with NaturalEarth coastlines, CWH boundary overlay, probability colorbar, and stats box.

---

## 8. Population Estimates (February 2026)

### CWH zone (100k sample, Feb 2026)

| Threshold | Estimated area | 95% CI |
|---|---|---|
| P≥0.3 | 411,758 ha | 404,600–418,900 ha |
| **P≥0.5** | **314,416 ha** | **308,117–320,715 ha** |
| P≥0.7 | 232,482 ha | 226,900–238,100 ha |

**Zone area:** 3,595,194 ha — **Model:** YewMLP AUC 0.998, F1 0.947

### Coastal BC region (100k sample, in progress Feb 2026)

Results pending extraction. Region area: 22,765,200 ha (227,652 km²).

> **Note (methodology):** Earlier estimates using `sample_cwh_yew_population.py` were inflated (~1.1M ha at P≥0.5) due to a WFS fallback bug that sampled the bounding-box rectangle rather than the actual CWH polygon. The corrected estimates above use `resample_cwh_100k.py` which passes `cwh_boundary_forestry.gpkg` directly to GEE.

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
| `results/analysis/cwh_yew_population_100k/` | CWH 100k re-sample outputs (predictions, KMZ, PNG) |
| `results/analysis/coastal_region_100k/` | Coastal BC region 100k sample outputs |
| `results/analysis/cwh_spot_comparisons/` | 15-area spot comparison maps |
| `data/processed/coastal_study_region.geojson` | Coastal BC study region boundary (87-part MultiPolygon) |
| `results/analysis/cwh_yew_population_100k/proposed_region.html` | Interactive Leaflet map of study region |

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
2. **`sample_cwh_yew_population.py` boundary bug** — when the WFS fails this old script falls back to an approximate bounding-box polygon, inflating the zone area ~7×. **Do not use it.** Use `resample_cwh_100k.py` (CWH zone) or `sample_coastal_region.py` (coastal region) instead.
3. **`/tmp` boundary file loss** — `sample_coastal_region.py` reads `data/processed/coastal_study_region.geojson`. Earlier runs stored the boundary in `/tmp/` which is cleared on reboot. The boundary is now committed to the project.
4. **PROJ_AGE_1 pre-disturbance bias** — for recently burned/logged stands, `PROJ_AGE_1` stores the pre-disturbance age. Always ensure `LINE_7B_DISTURBANCE_HISTORY` is parsed alongside it; the classifier takes the minimum age.
5. **Sample size vs precision** — at 300 m scale, 100k points within 3.6M ha gives one point per ~36 ha. The margin of error at P≥0.5 is approximately ±0.5% of CWH area (±18,000 ha).
