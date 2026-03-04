# Pacific Yew Habitat Mapping — Current Analysis Synopsis

**Project:** Spatial prediction of Pacific yew (*Taxus brevifolia*) habitat probability across coastal British Columbia  
**Last updated:** March 4, 2026  
**Interactive map:** [jerichooconnell.github.io/yew_project](https://jerichooconnell.github.io/yew_project/)

---

## Executive Summary

This project maps Pacific yew habitat suitability across British Columbia's Coastal Western Hemlock (CWH) zone using machine learning applied to satellite spectral embeddings. The analysis covers **45 study areas** spanning ~450 km² at 10 m resolution, integrating yew presence data from iNaturalist with Google Earth Engine satellite embeddings, BC forestry records, historical fire perimeters, and protected area boundaries.

**Key findings:**
- **~39,000 hectares** of high-probability yew habitat (P≥0.5) identified across 45 study tiles
- **~38,000 hectares** of yew habitat estimated destroyed by logging since 1920s in CWH zone
- Highest habitat concentrations in south-central Vancouver Island, Sunshine Coast, and mid-coast mainland fjords
- XGBoost classifier achieves **AUC 0.9957** on validation data

The interactive web map displays yew probability overlays with forestry/logging status, historical fire boundaries, and protected areas, allowing users to report field observations via GitHub.

---

## 1. Study Areas

**Coverage:** 45 study tiles (previously 15, expanded March 2026)  
**Geographic extent:** Coastal British Columbia, 48.3°N–55.3°N  
**Tile dimensions:** ~10 km × 10 km each (~100 km² per tile)  
**Total analyzed area:** ~450 km² at 10 m pixel resolution  
**Biogeoclimatic zones:** Primarily CWH (Coastal Western Hemlock), with representation of CDF, MH, ESSF, and CMA zones

### Study tile coverage (by region)

**Vancouver Island (16 tiles):**
- South VI: Carmanah-Walbran, Port Renfrew, Sooke Hills, Cowichan Uplands, Nanaimo Lakes
- Central VI: Clayoquot Sound, Comox Uplands, Alberni Valley, Gold River Forest, Strathcona Highlands, Muchalat Valley, Campbell River Uplands
- North VI: Quatsino Sound, Port Hardy Forest

**Mainland Coast (20 tiles):**
- South mainland: Squamish Highlands, Garibaldi Foothills, Howe Sound East, Stave Lake, Chilliwack Uplands
- Sunshine Coast: Sunshine Coast South, Sechelt Peninsula, Desolation Sound, Powell River Forest, Jervis Inlet Slopes
- Mid-coast: Bute Inlet Slopes, Toba Inlet Slopes, Knight Inlet, Kingcome Inlet, Broughton Archipelago
- Central coast: Rivers Inlet, Owikeno Lake, Burke Channel, Ocean Falls, Bella Coola Valley
- North coast: Dean Channel, Princess Royal Island, Milbanke Sound, Klemtu Forest, Namu Lowlands, Smith Sound

**Northern extremes (5 tiles):**
- Prince Rupert Hills
- Kitimat Ranges  
- Portland Inlet
- Stewart Lowlands

**Haida Gwaii (1 tile):**
- **Haida Gwaii South** (53.819°N, -132.435°W) — outer coast Moresby Island CWHvh3
  - *Relocated March 2026* to better capture CWH rainforest on western slopes

### Area selection rationale

Study areas were selected to:
1. **Cover the full latitudinal and precipitation gradient** of the CWH zone (south VI lowlands → north coast hypermaritime)
2. **Represent diverse stand ages and disturbance histories** (old-growth reserves, second-growth forests, recent clearcuts)
3. **Include iNaturalist observation clusters** where yew presence/absence is documented
4. **Sample remote areas** with minimal human access (deep fjords, outer islands)
5. **Focus computational resources** on ecologically coherent tiles rather than sparse sampling across the full 3.6M ha CWH zone

---

## 2. Model Architecture & Training

**Classifier:** XGBoost (Extreme Gradient Boosting)  
**Input features:** 64 spectral embedding dimensions from Google Earth Engine  
**Training data sources:**
- **834 iNaturalist yew positives** (training set)
- **209 iNaturalist yew positives** (validation set) 
- **267 manual field annotations** (weighted 3×)
- **8,994 negative samples** from FAIB forestry plots, alpine areas, and CWH non-yew zones

**Model performance (validation set):**
- **AUC-ROC:** 0.9957
- **Accuracy:** 98.9%
- **F1 score:** 0.947

**Key model insights:**
- Spectral embedding bands A03, A15, A27 (related to NIR vegetation indices, canopy moisture, texture) contribute most to classification
- Model successfully discriminates yew habitat from similar moist forest types (western hemlock, western redcedar)
- Performance degrades slightly in logged <40 yr stands where spectral signal is dominated by dense shrub/herb regeneration

---

## 3. Data Integration

### 3.1 Satellite Embeddings

**Source:** Google Earth Engine `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`  
**Year:** 2024  
**Resolution:** 10 m  
**Bands:** 64 spectral embedding dimensions (A00–A63) derived from Sentinel-2 annual composites

Embeddings are pre-computed by Google using a deep neural network trained on global satellite imagery. They capture complex spectral-spatial patterns related to vegetation structure, moisture, and land cover.

### 3.2 BC Vegetation Resources Inventory (VRI)

**Source:** BC Data Catalogue — `VEG_COMP_LYR_R1_POLY_2024.gdb`  
**Features:** 6.87 million forest polygons across BC  
**Key attributes used:**
- `PROJ_AGE_CLASS_CD_1`: Stand age class (1–9, mapped to years)
- `PROJ_AGE_1`: Explicit stand age (years)
- `HARVEST_DATE`: Year of recorded harvest
- `LINE_7B_DISTURBANCE_HISTORY`: Coded disturbance events (logging, fire, windthrow)
- `BCLCS_LEVEL_1/2`: Land cover classification (water, alpine, barren)
- `ALPINE_DESIGNATION`: Official alpine zone designation

**Post-classification suppression factors:**
| Category | Condition | Suppression | Rationale |
|---|---|---|---|
| Water/non-forest | BCLCS water/urban/agriculture | ×0.00 | Yew absent |
| Logged <40 yr | Stand age <40 yr | ×0.00 | Young second-growth, yew absent |
| Logged 40–80 yr | Stand age 40–80 yr | ×0.50 | Maturing second-growth, partial recovery |
| Forest 80–150 yr | Stand age 80–150 yr | ×0.35 | Established but regenerating; yew present but sparse |
| Old-growth >150 yr | Stand age >150 yr | ×1.00 | No suppression |
| Alpine/barren | Designated alpine, rock/ice | ×0.00 | Above treeline, yew absent |

This approach preserves raw model predictions in old-growth areas while suppressing probabilities in recently disturbed or unsuitable terrain.

### 3.3 Historical Fire Perimeters

**Source:** BC Data Catalogue — `PROT_HISTORICAL_FIRE_POLYS_SP.gdb`  
**Coverage:** 3,603 fires ≥50 ha in coastal study region, 1918–2024  
**Attributes:** Fire year, cause (lightning/person), size (hectares)  
**Web display:** Color-coded by age (recent fires = red, older fires = orange)  
**GeoJSON export:** `docs/tiles/fire_contours.geojson` (1.8 MB, simplified at ~300 m tolerance)

Fire history provides context for stand age and disturbance patterns, complementing VRI logging records.

### 3.4 Protected Areas

**Source:** BC Data Catalogue WFS — `WHSE_TANTALIS.TA_PARK_ECORES_PA_SVW` (live service) supplemented by local `TA_PARK_ECORES_PA_SVW.gdb` for northern BC  
**Coverage:** 512 provincial parks, ecological reserves, and conservancies in coastal BC from 47.5°N–56°N (including all of Vancouver Island, Sunshine Coast, Haida Gwaii)  
**Web display:** Green polygons, ecological reserves shown with dashed borders  
**GeoJSON export:** `docs/tiles/park_contours.geojson` (0.8 MB, created by combining WFS download with local GDB)

Protected areas highlight where yew habitat may be conserved from future logging.

---

## 4. Interactive Web Map

**URL:** [jerichooconnell.github.io/yew_project](https://jerichooconnell.github.io/yew_project/)  
**Technology:** Leaflet.js on GitHub Pages  
**Base layers:** Esri World Imagery (satellite), OpenStreetMap, terrain

### Map features

**Yew probability overlay (45 tiles):**
- Transparent PNG tiles draped over satellite imagery
- Color scale: green (0.02–0.33) → yellow (0.33–0.50) → orange (0.50–0.83) → magenta (0.83–1.0)
- Pixels with P<0.02 are fully transparent (noise floor)
- Toggle individual tiles on/off in sidebar

**Forestry/logging overlay:**
- Categorical raster showing VRI-derived stand age classes
- Color-coded by disturbance history (red = recent logging, green = old-growth >80 yr)
- Synchronized with yew probabilities (can view both simultaneously)

**Fire contours (🔥):**
- 3,603 historical fire perimeters ≥50 ha
- Clickable popups: fire year, cause, size
- Color gradient by age (bright red = 2020s, orange = pre-1950)

**Protected areas (🏞️):**
- 255 parks, ecological reserves, conservancies
- Clickable popups: name, designation, official area
- Green fill with dashed borders for ecological reserves

**Observation reporting:**
- Users can click "✔ Yew Present" or "✘ No Yew" then click the map to add field observations
- "Submit to GitHub" creates a GitHub Issue with observation coordinates in CSV format
- Allows crowdsourced validation and model improvement

**Logging impact statistics tab:**
- By-zone and by-subzone estimates of yew habitat destroyed by logging
- Covers 42 of 45 tiles (~4,200 km²) across CWH zone
- Methodology: yew prevalence in old-growth ×  (old-growth + logged area) − current remaining

---

## 5. Key Results

### 5.1 Habitat extent by tile (top 10)

| Tile | Region | Area P≥0.5 (ha) | Mean prob | Notes |
|---|---|---|---|---|
| Stave Lake | Lower Fraser | 4,357 | 0.362 | CWHvm1/dm, logged valleys + intact slopes |
| Alberni Valley | Central VI | 4,408 | 0.363 | CWHmm1/vm2, Somass River drainage |
| Howe Sound East | South mainland | 3,962 | 0.261 | Montane CWH, Squamish River |
| Rivers Inlet | Mid-coast | 3,114 | 0.188 | Outer coast CWH, hypermaritime |
| Port Renfrew | SW VI | 2,181 | 0.164 | Pacific Rim old-growth |
| Campbell River Uplands | Central VI | 2,184 | 0.172 | North-central VI CWH |
| Port Hardy Forest | North VI | 2,097 | 0.152 | Valley-bottom CWH |
| Clayoquot Sound | West VI | 1,773 | 0.171 | UNESCO biosphere reserve |
| Sooke Hills | South VI | 1,673 | 0.159 | Montane CWH, Victoria watershed |
| Cowichan Uplands | South VI | 1,652 | 0.173 | Lower-elevation CWH, mixed age |

**Total across 45 tiles:** ~39,000 ha with P≥0.5 (390 km²)

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
