# Pacific Yew (*Taxus brevifolia*) Habitat Detection — Analysis Methods

**Date:** February 2026  
**Project:** Remote sensing detection of Pacific yew across the Coastal Western Hemlock (CWH) BEC zone, British Columbia  
**Contact:** Jericho O'Connell

---

## 1. Overview

This analysis uses satellite-derived spectral embeddings and a machine learning classifier to estimate the spatial distribution of Pacific yew (*Taxus brevifolia*) habitat across the Coastal Western Hemlock (CWH) Biogeoclimatic Ecosystem Classification (BEC) zone in British Columbia. The goal is to produce a probability map and population area estimate that could guide field survey effort.

The approach has four main components:

1. **Satellite feature extraction** — 64-band spectral embeddings from Google Earth Engine's Satellite Embedding V1 product
2. **Supervised classification** — A GPU-accelerated Multi-Layer Perceptron (MLP) trained on iNaturalist yew observations as positive examples and multiple categories of confirmed non-yew locations as negative examples
3. **Forestry data integration** — BC Vegetation Resource Inventory (VRI) data used both to mask logged areas in the output and to generate logged-area training negatives
4. **Population estimation** — Stratified random sampling within the CWH BEC zone boundary, with 95% confidence intervals via Wilson score intervals

---

## 2. Study Area

**BEC Zone:** Coastal Western Hemlock (CWH)  
**Coverage area:** 14,684,432 ha (14.68 million ha)  
**Geographic extent:** Latitude 48.23°–59.77°N, Longitude 120.83°–136.54°W  
**Boundary source:** Derived directly from the BC VRI GDB (`VEG_COMP_LYR_R1_POLY_2024.gdb`) by selecting all polygons where `BEC_ZONE_CODE = 'CWH'` — this yielded 904,641 VRI polygons whose areas sum to 14,684,432 ha  

The CWH zone is the primary habitat for Pacific yew in BC — a humid, maritime-influenced forest type dominated by western hemlock (*Tsuga heterophylla*), western redcedar (*Thuja plicata*), and Douglas-fir (*Pseudotsuga menziesii*). Pacific yew is a shade-tolerant understorey species most commonly found in old-growth and mature unlogged stands.

The CWH boundary is stored as individual polygons (not dissolved) in `data/processed/cwh_boundary_vri.gpkg` (EPSG:4326) and `cwh_boundary_vri_3005.gpkg` (EPSG:3005). Area was computed by summing polygon areas in the BC Albers (EPSG:3005) projection.

---

## 3. Satellite Data

### 3.1 Product

**Collection:** `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`  
**Platform:** Google Earth Engine (GEE)  
**Year:** 2024  
**Spatial resolution:** 10 metres  
**Band count:** 64 (bands A00–A63)  

The Google Satellite Embedding product generates dense, spatially coherent 64-dimensional feature vectors at 10 m resolution by passing aerial/satellite imagery through a deep convolutional network pre-trained on a large corpus of global imagery. Each 64-band pixel encodes the local spectral, textural, and contextual characteristics of the landscape.

### 3.2 Extraction

For the south Vancouver Island high-resolution prediction (training + output), embeddings were extracted in a 4×7 tile grid (28 tiles, each ~1,194×2,087 pixels) covering:
- Latitude: 48.27°–48.70°N  
- Longitude: −124.51°–−123.20°W  
- Total pixels: ~69.8 million  
- Extraction time: ~67 seconds on NVIDIA RTX 4050 Laptop GPU

For the CWH-wide population estimate, a tiled random sampling approach extracted embeddings at 214,721 sampled points using GEE's `.sample()` method across a 12×18 geographic grid of tiles covering the full VRI-derived CWH boundary, avoiding GEE memory limits. Tiles with no valid data (ocean/no coverage) were excluded.

---

## 4. Training Data

### 4.1 Positive Examples (yew-present locations)

**Source:** iNaturalist observations, species *Taxus brevifolia*  
**Total iNat records available:** 6,964  
**Records used in training split:** 1,043 yew-present samples (train) + 208 (val)  

Each iNaturalist observation provides a GPS coordinate. The corresponding 64-band GEE embedding is extracted at the centre pixel. Quality filters applied:
- Minimum positional accuracy threshold
- Year 2024 imagery matched to most recent observation window
- Duplicate removal by spatial proximity

### 4.2 Negative Examples (yew-absent locations)

Negative (non-yew) examples were sourced from three complementary populations:

**a) iNaturalist background negatives (base dataset)**  
Observations from southern Vancouver Island and coastal BC of non-yew species in forested environments. These provide the baseline "what forests look like without yew" signal.  
- Training: 2,458 non-yew samples  
- Validation: 492 non-yew samples

**b) CWH-wide VRI logged-area negatives (the critical correction)**  
1,000 points selected from locations where:  
1. The current model predicted P(yew) ≥ 0.5 (i.e., spectral false positives)  
2. The BC VRI GDB (`VEG_COMP_LYR_R1_POLY_2024.gdb`) classifies the polygon as recently logged, non-vegetated, or non-treed vegetation (VRI suitability ≤ 0.15) and that point is over 100 meters from a water source.

The rationale: Pacific yew does not regenerate readily after clearcutting. High-probability predictions in confirmed logged areas are false positives the model should learn to suppress. A key constraint is ≥100 m from water, because yew does grow in riparian zones — even in logged areas, stream-side yew communities can survive logging operations, so these must be excluded from negatives.

The 1,000 points were selected from 1,221 qualified candidates across the CWH zone (48.4°–52.5°N), with 60% chosen as highest-probability false positives and 40% randomly sampled for diversity.

**VRI suitability categories included as negatives:**
| VRI Category | Suitability | Count |
|---|---|---|
| Non-vegetated / bare ground | 0.00 | 359 |
| Recently logged (<20 yr, HARVEST_DATE) | 0.05 | 123 |
| Non-treed / shrub vegetation | 0.10 | 334 |
| Logged 20–40 yr ago | 0.15 | 184 |

These CWH-wide negatives were weighted ×2 in training, giving a total of 2,000 negative samples from this source.

**Total training dataset:**  
| Split | Yew | Non-yew | Total |
|---|---|---|---|
| Train | 1,043 | 4,458 | 5,501 |
| Validation | 208 | 492 | 700 |

---

## 5. Classification Model

### 5.1 Architecture

A fully connected Multi-Layer Perceptron (MLP) with the following structure:

```
Input:  64 dimensions (satellite embedding bands A00–A63)
Layer 1: 64 → 128  +  BatchNorm  +  ReLU  +  Dropout(0.2)
Layer 2: 128 → 64  +  BatchNorm  +  ReLU  +  Dropout(0.2)
Layer 3: 64 → 32   +  BatchNorm  +  ReLU  +  Dropout(0.2)
Output:  32 → 1    +  Sigmoid
```

The output is a per-pixel probability P(yew ∈ [0, 1]).

### 5.2 Training

| Parameter | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 0.001 |
| Batch size | 512 |
| Epochs | 100 |
| Feature normalisation | StandardScaler (zero mean, unit variance) |
| Hardware | NVIDIA GeForce RTX 4050 Laptop GPU (CUDA) |

Features are standardised using a `StandardScaler` fitted on the full training set before input to the MLP. The scaler is saved alongside the model for consistent application at inference time.

### 5.3 Validation Performance

| Metric | Value |
|---|---|
| Validation accuracy | 97.86% |
| Validation F1 score | 0.9645 |
| Validation ROC-AUC | 0.9961 |

### 5.4 Inference

The trained model is applied to all 69.8 million pixels of the south Vancouver Island tile grid in GPU batches of 500,000 pixels, completing in ~98 seconds. The output is a full-resolution float32 probability grid at 10 m pixel size.

---

## 6. Forestry Mask (BC VRI Integration)

### 6.1 Data Source

**Geodatabase:** `VEG_COMP_LYR_R1_POLY_2024.gdb`  
**Layer:** `VEG_COMP_LYR_R1_POLY`  
**Projection:** BC Albers (EPSG:3005)  
**Total features:** 6,872,386 polygons covering 47.8°–59.7°N  
**File size:** 5.7 GB  

Key VRI fields used:
- `HARVEST_DATE` — recorded harvest/logging date
- `PROJ_AGE_1`, `PROJ_AGE_CLASS_CD_1` — projected stand age
- `OPENING_SOURCE` (3/4/7/11) — identifies cutblock openings
- `BCLCS_LEVEL_1`, `BCLCS_LEVEL_2` — land cover classification (non-vegetated, water, etc.)

### 6.2 Suitability Classification

Each VRI polygon is assigned a yew suitability score (0–1) based on logging history:

| Condition | Suitability | Rationale |
|---|---|---|
| Non-vegetated / rock | 0.00 | No forest habitat |
| Water body | 0.00 | Aquatic, not forest |
| Logged <20 yr ago (HARVEST_DATE) | 0.05 | Clearcut regeneration, no yew |
| Cutblock age class 1 (1–20 yr) | 0.05 | Same |
| Non-treed vegetation (shrub/herb) | 0.10 | No overstorey for yew |
| Logged 20–40 yr ago | 0.15 | Early regeneration, minimal yew |
| Logged 40–60 yr ago | 0.30 | Partial recovery |
| Logged 60–80 yr ago | 0.50 | Moderate recovery |
| Logged >80 yr ago | 0.70 | Substantial recovery |
| Old growth / unlogged (no HARVEST_DATE) | 1.00 | Maximum suitability |

For stand ages inferred from `PROJ_AGE_CLASS_CD_1` without a harvest record, similar breakpoints are applied.

### 6.3 Application

The suitability grid is rasterised to the same 10 m pixel grid as the model output and multiplied element-wise with the probability grid:

```
P_masked = P_raw × suitability
```

This suppresses high predictions in logged areas while leaving old-growth and unlogged stands unaffected.

---

## 7. Population Estimation — CWH Zone

### 7.1 Sampling Design

Random point sampling within the CWH BEC zone boundary (3,595,195 ha) using a tiled GEE extraction strategy:

- **Total attempted samples:** 300,000
- **Valid extracted samples:** 222,430 (21 ocean/no-data tiles excluded)
- **Sampling grid:** 10 columns × 15 rows geographic tiles
- **GEE extraction:** `.sample()` per tile with `numPixels` proportional to tile area
- **Seed:** 42 (reproducible)

### 7.2 Statistical Method

For each probability threshold θ ∈ {0.3, 0.5, 0.7}, the proportion of sample points with P ≥ θ is computed, and the estimated habitat area is:

$$\hat{A} = \hat{p} \times A_{\text{zone}}$$

where $A_{\text{zone}}$ = 3,595,195 ha.

**95% confidence intervals** are computed using the Wilson score interval, which is robust for proportions near 0 or 1 and small k:

$$\text{CI} = \frac{\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$

where $z = 1.96$ (95% level), $n$ = 222,430 sample points.

### 7.3 Results

| Threshold | Proportion | Area (ha) | 95% CI (ha) |
|---|---|---|---|
| P ≥ 0.3 | 2.18% | 78,343 | 76,191 – 80,555 |
| **P ≥ 0.5** | **1.40%** | **50,510** | **48,782 – 52,299** |
| P ≥ 0.7 | 0.87% | 31,276 | 29,919 – 32,694 |

**Recommended primary estimate:** P ≥ 0.5 threshold = **50,510 ha** (95% CI: 48,782–52,299 ha), representing approximately 1.4% of the CWH zone.

The margin of error at P ≥ 0.5 is ±0.05%  (227 ha absolute), reflecting the large sample size.

### 7.4 Caveats

- These estimates reflect areas where the spectral signature is *compatible with* Pacific yew habitat, not confirmed yew presence. Field validation is required.
- The VRI forestry mask reduces false positives in logged areas but cannot detect areas logged prior to the VRI database's temporal coverage.
- iNaturalist observations (positive training data) are biased toward accessible locations (roadsides, trails, parks). Yew in remote old-growth stands is likely under-represented in the training data.
- The model is trained primarily on south Vancouver Island spectral conditions. Performance may degrade in northern and coastal parts of the CWH zone where spectral characteristics differ.
- The 10 m pixel prediction map (south VI) and the CWH population estimate use the same classifier but different spatial extents.

---

## 8. Software and Reproducibility

| Component | Tool / Library | Version |
|---|---|---|
| Satellite data | Google Earth Engine Python API | — |
| GEE collection | GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL | 2024 |
| Spatial data | GeoPandas, Fiona, Shapely | current |
| CRS reprojection | PyProj | current |
| Machine learning | PyTorch | 2.1.0+cu118 |
| Feature scaling | scikit-learn StandardScaler | current |
| GPU | NVIDIA GeForce RTX 4050 Laptop | CUDA |
| Python environment | Conda (yew_pytorch), Python 3.10 | — |
| GEE project | carbon-storm-206002 | — |

### 8.1 Key Scripts

| Script | Purpose |
|---|---|
| `scripts/prediction/classify_tiled_gpu.py` | Train MLP, classify 28-tile south VI grid |
| `scripts/preprocessing/apply_forestry_mask.py` | Load VRI GDB, rasterise suitability grid |
| `scripts/preprocessing/extract_cwh_logged_negatives.py` | Extract VRI logged-area negatives with water buffer |
| `scripts/preprocessing/build_vri_coverage_boundary.py` | Verify VRI tile coverage over CWH zone |
| `scripts/prediction/sample_cwh_yew_population.py` | GEE tiled sampling, classification, statistics |
| `scripts/visualization/create_cwh_kmz.py` | Generate KMZ from sample predictions |

### 8.2 Iterative Retraining History

The model went through four training iterations with progressively better negative examples:

| Iteration | Negative Source | Val Acc | F1 | AUC |
|---|---|---|---|---|
| 1 | iNaturalist background only | 97.71% | 0.9624 | 0.9963 |
| 2 | + 500 distance-from-iNat negatives | 97.00% | 0.9499 | 0.9922 |
| 3 | + 1,000 south VI VRI logged negatives | 98.00% | 0.9668 | 0.9950 |
| **4** | **+ 1,000 CWH-wide VRI logged negatives (≥100m from water)** | **97.86%** | **0.9645** | **0.9961** |

Iteration 2 (distance-based) was rejected because Pacific yew does exist far from iNaturalist observation points — distance from known observations is not a valid proxy for absence. The correct approach (iterations 3–4) uses confirmed logged areas from the VRI where yew does not regenerate after clearcutting.

---

## 9. Output Files

| File | Description |
|---|---|
| `results/predictions/south_vi_large/prob_grid.npy` | 4,775×14,609 float32 probability map, south VI at 10 m |
| `results/predictions/south_vi_large/mlp_model.pth` | PyTorch model weights |
| `results/predictions/south_vi_large_forestry/suitability_grid.npy` | VRI suitability raster, same extent |
| `results/analysis/cwh_yew_forestry_300k/sample_predictions.csv` | 222,430 CWH sample points with probabilities |
| `results/analysis/cwh_yew_forestry_300k/population_statistics.json` | Area estimates and confidence intervals |
| `data/processed/cwh_logged_negatives/cwh_logged_negatives_google_maps.kml` | 1,000 logged-area negatives for Google Earth/Maps |
| `results/analysis/cwh_yew_forestry_300k/cwh_yew_forestry_300k.kmz` | CWH probability map for Google Earth |
