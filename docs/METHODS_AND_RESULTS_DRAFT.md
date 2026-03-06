# Satellite Embedding–Based Mapping of Pacific Yew (*Taxus brevifolia*) Habitat Decline Across British Columbia

**Draft — Methods and Results**

---

## Abstract

Pacific yew (*Taxus brevifolia*) is a slow-growing, shade-tolerant understory conifer historically considered a "trash tree" by the British Columbia logging industry. Its ecological importance — and vulnerability — became globally apparent in the 1960s when its bark was found to contain paclitaxel (Taxol), one of the most effective chemotherapy drugs ever discovered. Despite the subsequent development of semi-synthetic paclitaxel production, the species continues to face severe population decline driven primarily by industrial clear-cut logging, compounded by wildfire, stream erosion, sea-level rise, and ungulate browsing.

We present a machine-learning approach using Google Earth Engine satellite spectral embeddings to map Pacific yew habitat probability at 10 m resolution across 9,900 km² of British Columbia, spanning three major biogeoclimatic zones: Coastal Western Hemlock (CWH), Interior Cedar–Hemlock (ICH), and Coastal Douglas-fir (CDF). An XGBoost classifier trained on 64-band spectral embeddings from the Google AlphaEarth Foundation model achieves AUC-ROC 0.996 on held-out validation data. Post-classification suppression using BC Vegetation Resources Inventory (VRI) stand-age records, historical fire perimeters, and digital elevation data allows estimation of both current remaining and historically destroyed yew habitat.

Our analysis estimates that **154,483 ha** of yew habitat existed historically across the 99 study tiles, of which **47,534 ha (30.8%)** remains today — representing a **69.2% decline**. The Coastal Douglas-fir zone has suffered the most catastrophic loss (99.1% decline), followed by Interior Cedar–Hemlock (74.7%) and Coastal Western Hemlock (69.1%). Logging alone accounts for an estimated 106,949 ha of habitat destruction. Only 4.6% of remaining modelled yew habitat falls inside provincial protected areas.

---

## 1. Introduction

Pacific yew (*Taxus brevifolia* Nutt.) is a long-lived, dioecious conifer native to the Pacific Northwest of North America, ranging from southeastern Alaska to central California and inland to the Rocky Mountain foothills. In British Columbia, it occurs primarily as a scattered understory tree in moist, old-growth forests of the Coastal Western Hemlock (CWH), Interior Cedar–Hemlock (ICH), and Coastal Douglas-fir (CDF) biogeoclimatic zones (Pojar et al. 1991; Meidinger & Pojar 1991).

The species gained worldwide attention when the National Cancer Institute discovered in the 1960s that its bark contained paclitaxel, a potent mitotic inhibitor effective against ovarian, breast, and lung cancers (Wani et al. 1971). Because the yield of pure paclitaxel is extraordinarily low — approximately 1 kg per 9,080 kg of bark — and extraction required killing the tree, a "yew rush" in the early 1990s resulted in the destruction of hundreds of thousands of mature Pacific yew trees across the Pacific Northwest (Appendix: Taxol Extraction History). By 1991, Bristol-Myers Squibb was permitted to harvest over 825,000 pounds of bark from Pacific Northwest national forests in a single year. The crisis was largely averted by 1993 when semi-synthetic production from European yew (*Taxus baccata*) needles eliminated the need for wild bark harvest.

However, the primary ongoing threat to Pacific yew populations is industrial clear-cut logging. As a shade-obligate species requiring old-growth canopy conditions, Pacific yew is eliminated entirely from clearcut blocks and takes over a century to recolonise (Busing et al. 1995). In British Columbia, where industrial forestry has operated at scale since the 1920s, much of the CWH and ICH zones have been logged at least once, with devastating consequences for this understory species.

No comprehensive spatial assessment of Pacific yew habitat extent or decline has been attempted at the provincial scale. Existing range data consist of point observations (e.g., iNaturalist, herbarium records) supplemented by incidental mentions in forest inventory plots. This study addresses that gap using satellite-based spectral embeddings and machine learning.

---

## 2. Methods

### 2.1 Study Area

We analysed 99 study tiles (each ~10 × 10 km, ~100 km²) distributed across British Columbia (Figure 18), covering a total of approximately 9,900 km². Of these, 85 tiles were located in coastal British Columbia within the CWH zone and adjacent zones, and 14 tiles were placed within the Interior Cedar–Hemlock (ICH) zone in the BC interior (one additional ICH tile, ICH_pt10 at Valemount, failed to download due to Google Earth Engine timeout).

#### 2.1.1 Coastal Tile Selection

The 85 coastal tiles span 48.3°N–55.3°N and were selected to:
1. Cover the full latitudinal and precipitation gradient of the CWH zone, from south Vancouver Island lowlands to north coast hypermaritime rainforest
2. Represent diverse stand ages and disturbance histories
3. Include clusters of iNaturalist yew observations for training data proximity
4. Sample remote areas (deep fjords, outer islands) with minimal human access

#### 2.1.2 Interior Tile Selection

The 14 ICH tiles were generated by random stratified sampling within the ICH biogeoclimatic zone boundary polygon (from the BC BEC map, v12), using a fixed random seed (42) projected in BC Albers (EPSG:3005) and converted to WGS84. These points sample 13 distinct ICH subzones including ICHdw1, ICHmw2, ICHxw, ICHdw4, ICHvk1, ICHwk1, and others.

#### 2.1.3 Biogeoclimatic Zone Coverage

Across all 99 tiles, the analysis intersects 12 major BEC zones and 69 subzones:

| Zone | Full Name | Tiles with Coverage | Total Area Analysed (ha) |
|------|-----------|--------------------:|-------------------------:|
| CWH  | Coastal Western Hemlock | 85 | 885,783 |
| ICH  | Interior Cedar–Hemlock | 14 | 142,778 |
| CDF  | Coastal Douglas-fir | 5 | 36,055 |
| ESSF | Engelmann Spruce–Subalpine Fir | 25 | 70,798 |
| MH   | Mountain Hemlock | 20 | 99,030 |
| IDF  | Interior Douglas-fir | 5 | 8,595 |
| CMA  | Coastal Mountain-heather Alpine | 10 | 147,304 |
| MS   | Montane Spruce | 4 | 9,189 |
| IMA  | Interior Mountain-heather Alpine | 3 | 9,789 |
| BAF  | Boreal Altai Fescue Alpine | 1 | 5,901 |
| SBS  | Sub-Boreal Spruce | 1 | 793 |

### 2.2 Satellite Spectral Embeddings

#### 2.2.1 Google AlphaEarth Foundation Model

Per-pixel spectral features were extracted from the Google AlphaEarth Foundation model satellite embeddings (`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`), accessed through Google Earth Engine (GEE). This dataset provides 64-dimensional embedding vectors (bands A00–A63) at 10 m spatial resolution, derived from annual Sentinel-2 composites using a deep neural network trained on global satellite imagery. We used the 2024 annual composite for all tiles.

The embeddings capture complex spectral-spatial patterns related to vegetation structure, moisture regime, canopy density, and land cover type. Each embedding dimension is approximately unit-normalised, consistent with the AlphaEarth Foundation model design.

#### 2.2.2 Embedding Download Protocol

Embedding rasters were downloaded tile-by-tile via the GEE Python API in chunked 2-band increments to remain within GEE's ~50 MB download limit per request. A safety margin of 35% was applied to the limit calculation. Each tile's embeddings were cached as `{slug}_emb.npy` (float32, shape ~1000 × 1000 × 64) for reproducibility.

True-colour RGB composites were also downloaded from the Copernicus Sentinel-2 Surface Reflectance Harmonized collection (`COPERNICUS/S2_SR_HARMONIZED`), filtered to June–September with <20% cloud cover, and median-composited.

### 2.3 Training Data

#### 2.3.1 Positive Samples

Yew presence records were assembled from two sources:
- **834 iNaturalist observations** (research-grade *Taxus brevifolia* records in BC and adjacent Washington/Oregon), divided into 834 training and 209 validation samples
- **267 manual field annotations** from site visits and aerial image review, weighted 3× during training to prioritise expert knowledge

Centre-pixel 64-dimensional embeddings were extracted at each observation location. This yielded per-observation feature vectors of length 64, representing the spectral embedding at the pixel containing the GPS coordinate.

#### 2.3.2 Negative Samples

Negative (yew-absent) training data comprised **8,994 samples** drawn from:
- BC Forest Analysis and Inventory Branch (FAIB) forest plots in non-yew species assemblages
- Alpine and subalpine locations above the elevational range of yew
- CWH-zone locations with confirmed non-yew canopy dominance (western hemlock, Sitka spruce stands)
- GEE-extracted embeddings from confirmed non-yew locations, weighted 2× during training

The combined training set included **15,019 samples** (6,066 positives, 8,953 negatives).

### 2.4 Classifier Architecture

#### 2.4.1 Production XGBoost Model

The production classifier is an XGBoost gradient-boosted tree ensemble (`xgb_raw_model_expanded.json`) trained on the 64 raw embedding dimensions with the following hyperparameters:

| Parameter | Value |
|-----------|-------|
| Objective | `binary:logistic` |
| Eval metric | AUC |
| Max depth | 6 |
| Learning rate | 0.05 |
| Subsample | 0.8 |
| Column sample by tree | 0.8 |
| Scale positive weight | Auto (ratio of negatives to positives) |
| Number of boosting rounds | 500 (early stopping at 50) |
| Tree method | `hist` |
| Random state | 42 |

The model was trained with spatial cross-validation using K-means clustering (10 clusters) for train/validation/test splits (70/10/20%) to mitigate spatial autocorrelation.

#### 2.4.2 Comparison Models

Six additional classifiers were evaluated on identical features and splits (Table: Figure 14):

| Model | AUC-ROC | Accuracy | F1 Score |
|-------|---------|----------|----------|
| **XGBoost (production)** | **0.9957** | **0.989** | **0.947** |
| MLP + StandardScaler | 0.9961 | 0.986 | 0.977 |
| MLP raw embeddings | 0.9962 | 0.976 | 0.960 |
| Random Forest | 0.9896 | 0.984 | 0.974 |
| kNN (k=3) | 0.9909 | 0.911 | 0.833 |
| Logistic Regression | 0.9165 | 0.813 | 0.562 |

The MLP architecture used a 128→64→32→1 network with BatchNorm, ReLU activations, and 0.2 dropout, trained for 100 epochs with Adam (lr=0.001) and cosine annealing. XGBoost was selected as the production model for its comparable performance, interpretability, and computational efficiency for per-pixel inference across ~100 million pixels.

#### 2.4.3 Alternative Feature Sets

A separate XGBoost model was trained on 35 engineered features from the BC Vegetation Resources Inventory (basal area, stems/ha, volume, site index, height, age, and their interactions). This inventory-only model achieved AUC 0.82 — substantially lower than the satellite embedding approach — with only 26% recall at operational thresholds. A multi-modal model combining spectral embeddings with inventory features yielded only 5% recall due to scale mismatch between polygon-level inventory data and pixel-level embeddings. These results confirmed that satellite spectral embeddings alone provide superior classification performance for this application.

### 2.5 Post-Classification Suppression

Raw XGBoost probabilities represent spectral habitat suitability but do not account for disturbance history. We applied a sequential suppression pipeline to convert raw probabilities into ecologically realistic current habitat estimates.

#### 2.5.1 Logging Suppression

Stand-age information was extracted from the BC Vegetation Resources Inventory (VRI 2024, `VEG_COMP_LYR_R1_POLY_2024.gdb`, 6.87 million polygons). Each 10 m pixel was assigned to one of seven land-use categories based on projected stand age, harvest date, disturbance history, and land cover classification:

| Category | Description | Suppression Factor | Ecological Rationale |
|----------|-------------|-------------------:|----------------------|
| 1 | Water / non-forest | ×0.00 | Yew absent |
| 2 | Logged <20 yr | ×0.00 | Dense regeneration, yew excluded |
| 3 | Logged 20–40 yr | ×0.00 | Early second-growth, yew not yet established |
| 4 | Logged 40–80 yr | ×0.00 | Canopy closure incomplete; yew recovery negligible |
| 5 | Forest 80–150 yr | ×0.00 | Maturing second-growth; yew present but sparse |
| 6 | Alpine / barren | ×0.00 | Above treeline |
| 7 | Old-growth (>150 yr) | ×1.00 | Reference condition; full model confidence |

Note: For the analysis of habitat *loss*, all non-old-growth forested land (categories 2–5) is treated as "logged" to estimate what yew habitat would have existed under pre-logging old-growth conditions. The web map display uses a more graduated suppression (×0.20 for 40–80 yr, ×0.35 for 80–150 yr) to show partial recovery.

#### 2.5.2 Fire Suppression

Historical fire perimeters (1900–2024) were obtained from the BC historical fire polygon dataset (`PROT_HISTORICAL_FIRE_POLYS_SP.gdb`), filtered to fires ≥100 ha and clipped to the study extent. A time-dependent fire modifier was applied:

$$\text{fire\_modifier} = \frac{2024 - \text{fire\_year}}{124}, \quad \text{clamped to } [0, 1]$$

This assumes linear recovery over 124 years (1900–2024), with recent fires causing near-complete suppression and century-old fires having minimal residual impact. For overlapping fire polygons, the most recent fire year takes precedence.

Across all tiles, 5,700 fire features covering approximately 96,543 ha intersected the study area, resulting in 692 ha of estimated yew habitat suppression.

#### 2.5.3 Elevation Suppression

A Copernicus GLO-30 Digital Elevation Model was used to suppress yew predictions at very low elevations (coastal tiles), where sea-spray, salt intrusion, and tidal influence preclude yew establishment:

$$\text{elev\_factor} = \text{clip}\left(\frac{\text{elevation}}{30}, 0, 1\right)$$

This linear ramp from 0 at sea level to 1 at 30 m elevation removed 18,434 ha of false-positive predictions from intertidal and low-coastal zones.

#### 2.5.4 Habitat Loss Estimation

For each BEC subzone, the following metrics were computed:

1. **Yew prevalence rate** ($r$): mean raw XGBoost probability across all old-growth (category 7) pixels in the subzone
2. **Estimated original habitat**: $r \times (\text{old-growth pixels} + \text{logged pixels}) \times 0.01$ ha/pixel
3. **Current remaining habitat**: sum of suppressed probabilities across all old-growth pixels
4. **Destroyed habitat**: estimated original − current remaining

This approach uses continuous probability mass rather than binary thresholding, providing a more robust estimate of cumulative habitat area. Each pixel contributes its predicted probability (0–1) multiplied by its area (0.01 ha at 10 m resolution).

### 2.6 Secondary Threat Assessment

In addition to the quantitative logging and fire analysis, we reviewed the scientific literature to identify and characterise secondary threats to Pacific yew populations that are not directly modellable from remote sensing data.

#### 2.6.1 Stream Erosion and Riparian Habitat Loss

Pacific yew preferentially occupies moist riparian zones — stream banks, canyon bottoms, and shaded ravines (Busing et al. 1995). Logging-driven hydrological changes increase peak flows by 20–50% (Hartman & Scrivener 1990, Carnation Creek study), causing channel widening proportional to $W \propto Q^{0.5}$ (Leopold & Maddock 1953). Combined with climate-driven precipitation increases of 5–20% by 2050 (Schnorbus et al. 2012, PCIC), channels may widen by ~14% in logged watersheds. A sensitivity analysis applying a 30 m riparian buffer to all water features estimates an additional 2–3% habitat loss beyond static VRI classifications.

#### 2.6.2 Sea-Level Rise and Saltwater Intrusion

Coastal yew populations in low-elevation fjords and inlets face direct habitat loss from sea-level rise. Pacific yew is not salt-tolerant; saline groundwater intrusion kills root systems. Habitat below 1.0–1.5 m elevation in 50-year projections faces 100% loss, with an additional 15–20 m inland buffer for saltwater intrusion effects. In our study, elevation suppression (§2.5.3) captures current low-elevation effects but does not project future sea-level scenarios.

#### 2.6.3 Yew Big Bud Mite (*Cecidophyopsis psilaspis*)

This eriophyid mite causes bud galls on *Taxus* species, with terminal bud mortality averaging over 20% in infested coastal BC populations. Both vegetative and reproductive buds are affected, reducing growth rates by ~20% and seed (aril) production by ~25%. While this pest does not cause immediate mortality, its cumulative impact on population regeneration capacity is substantial — particularly in combination with logging-driven habitat fragmentation.

#### 2.6.4 Ungulate Browsing

Wild ungulates (moose, elk, black-tailed and white-tailed deer) preferentially browse Pacific yew foliage, especially during winter. Browsing pressure is considered a primary barrier to natural regeneration, with 60–80% seedling/sapling mortality in areas with high ungulate density (>10 deer/km²). This creates a "browsing ceiling" that prevents recruitment from the seedling to the established understory tree stage, accelerating the ageing of remaining populations without replacement.

#### 2.6.5 Wildfire Frequency Increase

Pacific yew has no fire tolerance — its thin, flaky bark provides no protection against even low-intensity ground fires. In the CWH zone, historical fire return intervals of 250–1,000+ years permitted the species' extremely slow reproductive cycle. Climate change is projected to reduce fire return intervals to 80–120 years in drier maritime subzones, potentially preventing populations from reaching reproductive maturity (~80–100 years) between successive fires. Our fire modifier (§2.5.2) captures the impact of historical fires but does not model projected future fire regime shifts.

#### 2.6.6 Historical Taxol Bark Harvest

Between 1989 and 1993, hundreds of thousands of mature Pacific yew trees were felled for paclitaxel extraction across the Pacific Northwest. At peak harvest (1991), the NCI estimated that 360,000 mature yew trees per year would need to be harvested to meet clinical demand. Treating a single cancer patient required the bark of six 100+-year-old trees. This harvest was not spatially documented in GIS and thus cannot be directly modelled, but its legacy is embedded in the current reduced old-growth yew density in some regions.

### 2.7 Interactive Web Map

Results are presented via an interactive Leaflet.js web map hosted on GitHub Pages ([jerichooconnell.github.io/yew_project](https://jerichooconnell.github.io/yew_project/)). The map displays:

- **Yew probability overlays**: 99 transparent PNG tiles with a green→yellow→orange→magenta colour scale (P = 0.02–1.0)
- **Forestry/logging overlays**: VRI-derived categorical rasters showing stand age classes
- **Historical fire perimeters**: 5,700 fire polygons ≥100 ha (1900–2024), colour-coded by age
- **Protected areas**: 480 provincial parks, ecological reserves, and conservancies across BC
- **Observation reporting**: Users can submit field observations (yew present/absent) directly to GitHub Issues for crowd-sourced validation

The sidebar presents per-BEC-subzone statistics including estimated original habitat, current remaining, logging loss, fire suppression, and percent decline.

---

## 3. Results

### 3.1 Overall Habitat Decline

Across all 99 study tiles (~9,900 km², 69 BEC subzones), we estimate that **154,483 ha** of yew habitat existed historically under pre-logging old-growth conditions, of which **47,534 ha (30.8%)** remains today (Table 1, Figure 1). This represents an overall decline of **69.2%**.

The dominant driver of decline is industrial logging, which accounts for an estimated **106,949 ha** of destroyed yew habitat. Fire suppression contributes an additional **691 ha** (0.4% of original), and low-elevation suppression removes **18,434 ha** (11.9%) of false-positive predictions from intertidal zones.

**Table 1. Summary statistics by major BEC zone (zones with >10 ha estimated original yew)**

| Zone | Est. Original (ha) | Current Remaining (ha) | Decline (%) | Logged Area (ha) | Fire Loss (ha) | Elevation Loss (ha) |
|------|--------------------:|-----------------------:|------------:|-----------------:|---------------:|--------------------:|
| CWH  | 111,407 | 34,386 | 69.1 | 312,128 | 460 | 17,060 |
| ICH  | 25,257 | 6,385 | 74.7 | 97,071 | 152 | 0 |
| CDF  | 3,889 | 36 | 99.1 | 9,986 | 0 | 45 |
| ESSF | 3,957 | 1,742 | 56.0 | 29,392 | 64 | 26 |
| MH   | 7,078 | 4,161 | 41.2 | 21,709 | 7 | 1,134 |
| IDF  | 1,697 | 463 | 72.7 | 5,768 | 0 | 0 |
| MS   | 552 | 114 | 79.4 | 6,511 | 6 | 0 |
| CMA  | 609 | 225 | 63.0 | 10,474 | 1 | 170 |
| **Total** | **154,483** | **47,534** | **69.2** | — | **691** | **18,434** |

### 3.2 Coastal Western Hemlock (CWH) Zone

The CWH zone contains the vast majority of British Columbia's yew habitat (72% of estimated original, 72% of current remaining). Across 14 CWH subzones and 885,783 ha of analysed area:

- **Estimated original yew habitat**: 111,407 ha
- **Current remaining**: 34,386 ha (**69.1% decline**)
- **Old-growth forest remaining**: 330,804 ha (37.3% of total CWH area in study tiles)
- **Logged forest**: 312,128 ha (35.2% of total CWH area)

The most severely impacted CWH subzones are (Figure 2):

| Subzone | Est. Original (ha) | Current (ha) | Decline (%) | Key Feature |
|---------|--------------------:|-------------:|------------:|-------------|
| CWHxm2 | 18,493 | 1,943 | 89.5 | Heavy logging, south mainland |
| CWHvm1 | 22,296 | 8,167 | 63.4 | Largest CWH subzone; 97,591 ha logged |
| CWHvh2 | 20,434 | 10,355 | 49.3 | Hypermaritime; much old-growth remains |
| CWHmm2 | 8,029 | 749 | 90.7 | Intensive second-growth management |
| CWHdm  | 8,633 | 1,392 | 83.9 | Dry maritime; heavy logging pressure |
| CWHxm1 | 8,300 | 1,318 | 84.1 | Very dry maritime; extensive forestry |
| CWHvh1 | 10,158 | 3,597 | 64.6 | Hypermaritime; 17,000 ha old-growth |

The CWHxm2 (very dry maritime 2) and CWHmm2 (moist maritime 2) subzones show the highest percentage decline (>89%), reflecting both high original yew prevalence and intensive logging history. The CWHvh2 (very wet hypermaritime 2) subzone retains the most yew habitat in absolute terms (10,355 ha) due to its extensive old-growth (121,647 ha) and relatively low logging pressure.

### 3.3 Interior Cedar–Hemlock (ICH) Zone

The ICH zone represents the interior counterpart to the coastal CWH, supporting Pacific yew in moist valley-bottom forests east of the Coast and Columbia Mountains. Our analysis of 14 ICH tiles across 142,778 ha reveals:

- **Estimated original yew habitat**: 25,257 ha
- **Current remaining**: 6,385 ha (**74.7% decline**)
- **Old-growth forest remaining**: 35,907 ha (25.1% of total ICH area)
- **Logged forest**: 97,071 ha (68.0% of total ICH area)

The ICH zone shows a **higher percentage decline** than the CWH (74.7% vs. 69.1%), primarily because a larger proportion of the ICH landscape has been logged (68.0% vs. 35.2%). This reflects the more accessible valley-bottom terrain in interior BC where yew habitat concentrates.

The most impacted ICH subzones are (Figure 3):

| Subzone | Est. Original (ha) | Current (ha) | Decline (%) |
|---------|--------------------:|-------------:|------------:|
| ICHdw1  | 5,368 | 704 | 86.9 |
| ICHdw4  | 2,956 | 661 | 77.6 |
| ICHmw5  | 2,628 | 498 | 81.0 |
| ICHmw2  | 2,254 | 988 | 56.2 |
| ICHxm1  | 3,303 | 1,256 | 62.0 |
| ICHxw   | 1,944 | 315 | 83.8 |
| ICHdm   | 1,349 | 198 | 85.3 |

The ICHxw (very wet warm) subzone has the highest mean yew probability in old-growth (0.489), making it the single most yew-rich subzone in our entire dataset. Despite this, 83.8% of its estimated original habitat has been lost to logging. The ICHdw1 (dry warm 1) subzone has lost the most habitat in absolute terms (4,664 ha destroyed).

Fire plays a somewhat larger relative role in the ICH than in the CWH (152 ha suppressed), consistent with the shorter fire return intervals in interior BC. Elevation suppression is zero across all ICH subzones because interior study tiles are located above the coastal tidal zone.

### 3.4 Coastal Douglas-fir (CDF) Zone

The CDF zone, restricted to the rain-shadow lowlands of southeastern Vancouver Island and the Gulf Islands, has experienced the most catastrophic yew habitat loss of any zone in our study (Figure 4):

- **Estimated original yew habitat**: 3,889 ha
- **Current remaining**: 36 ha (**99.1% decline**)
- **Old-growth forest remaining**: 214 ha (0.6% of total CDF area)
- **Logged forest**: 9,986 ha (27.7% of total CDF area)
- **Water/developed**: 25,163 ha (69.8% — reflecting urban and agrarian conversion)

The single CDF subzone (CDFmm — moist maritime) had the highest old-growth yew prevalence rate (0.381) of any subzone, indicating that the Coastal Douglas-fir zone historically supported dense yew populations. Today, only 214 ha of old-growth remain — surrounded by 9,986 ha of logged forest and 25,163 ha of water and developed land. The effective elimination of Pacific yew from the CDF zone represents a near-total regional extirpation within the 99.4% of the landscape that has been logged, urbanised, or converted to agriculture.

### 3.5 Old-Growth Yew Prevalence

Mean yew probability in old-growth pixels varies widely among subzones (Figure 9), reflecting genuine ecological variation in species prevalence:

**Highest prevalence subzones:**
| Subzone | Mean P(yew) in OG | Zone |
|---------|------------------:|------|
| ICHxw   | 0.489 | ICH |
| CWHvh1  | 0.411 | CWH |
| ICHdw1  | 0.410 | CWH |
| CDFmm   | 0.381 | CDF |
| ICHdw4  | 0.356 | ICH |
| CWHmm2  | 0.365 | CWH |
| ICHxm1  | 0.320 | ICH |
| CWHxm2  | 0.303 | CWH |
| CWHmm1  | 0.307 | CWH |

These prevalence rates serve as the basis for estimating historical habitat extent. The highest rates (>0.30) occur in warm, moist subzones at low to mid elevations — the ecological niche of Pacific yew.

### 3.6 Logging Intensity and Age Distribution

The relationship between logging intensity and yew decline is strongly positive (Figure 8), with subzones exceeding 50% logged area consistently showing >70% yew decline.

Across the three focal zones, the logged area is dominated by different age classes (Figure 13):

| Age Class | CWH (ha) | ICH (ha) | CDF (ha) |
|-----------|----------:|----------:|----------:|
| <20 yr (recent clearcut) | 46,662 | 22,579 | 476 |
| 20–40 yr | 59,544 | 19,812 | 1,131 |
| 40–80 yr | 125,262 | 11,333 | 2,545 |

In the CWH, the 40–80 yr age class dominates (125,262 ha), reflecting the peak of industrial logging in the 1950s–1980s. In the ICH, recent (<20 yr) logging is proportionally larger (22,579 ha, or 23% of total logged), indicating that interior forestry has remained active more recently. The CDF shows minimal recent logging — most of its conversion occurred decades ago.

### 3.7 Fire Impact

Historical fires (1900–2024) contribute a relatively small but non-negligible component of yew habitat suppression (Figure 10). Across all tiles:

- **Total burned area intersecting study tiles**: 96,543 ha
- **Total yew suppression from fire**: 692 ha
- **Fire modifier formula**: linear recovery over 124 years

The decade-by-decade breakdown reveals:

| Decade | Burned Area (ha) | Yew Suppressed (ha) | Mean Modifier |
|--------|------------------:|---------------------:|--------------:|
| 1910s  | 2,827 | 72 | 0.847 |
| 1920s  | 33,208 | 113 | 0.802 |
| 1930s  | 13,530 | 97 | 0.720 |
| 1950s  | 16,602 | 95 | 0.564 |
| 2000s  | 2,826 | 26 | 0.125 |
| 2010s  | 4,859 | 17 | 0.063 |
| 2020s  | 8,146 | 30 | 0.013 |

Older fires (1920s–1930s) show high yew suppression per hectare burned because the fire modifier assigns near-full recovery (time × recovery rate), while the most recent fires (2020s) show the highest suppression per unit recovered (modifier ≈ 0.013). The 2020s burned area (8,146 ha) exceeds the 2010s (4,859 ha), consistent with the accelerating fire season in BC driven by climate change.

### 3.8 Post-Classification Suppression Pipeline

The suppression pipeline transforms raw model predictions through sequential filters (Figure 12):

| Stage | Predicted Yew Habitat (ha) | Description |
|-------|---------------------------:|-------------|
| Raw model output | 213,165 | Spectral suitability only |
| After logging suppression | 66,659 | Removes predictions from logged/young forest |
| After fire suppression | 65,968 | Removes predictions from recently burned areas |
| After elevation suppression | 47,534 | Removes coastal low-elevation false positives |
| **Final current habitat** | **47,534** | Ecologically realistic estimate |

Logging suppression is by far the dominant filter, removing ~146,506 ha (68.7%) of raw predictions. This does not imply that the model is inaccurate — rather, that the spectral signature of potential yew habitat (moist closed-canopy forest) persists even in young second-growth stands where yew itself cannot yet survive.

### 3.9 Comparison of CWH, ICH, and CDF Decline Pathways

The three focal zones exhibit qualitatively different decline pathways (Figure 19):

**CWH: Gradual erosion by industrial forestry**
- Large absolute habitat base (111,407 ha original)
- 37% old-growth remaining → substantial yew still present
- Ongoing logging pressure across the landscape
- 69.1% decline, with most loss in recent decades

**ICH: Intensive logging in concentrated habitat**
- Yew habitat concentrated in valley bottoms and moist sites
- 68% of the landscape already logged
- Higher decline rate (74.7%) than CWH despite smaller total area
- Fire plays a larger role than in CWH

**CDF: Near-complete extirpation**
- Almost all old-growth eliminated (only 214 ha of 36,055 ha total)
- Urban, agricultural, and forestry conversion has removed 99.4% of natural forest
- 99.1% yew decline — effective regional extinction
- The CDFmm subzone represents one of the most imperilled yew habitats in BC

### 3.10 Secondary Threats — Quantitative Estimates

While logging dominates the estimated decline, secondary threats compound the population impact:

| Threat | Estimated Impact | Confidence | Type |
|--------|------------------|------------|------|
| Clear-cut logging | 106,949 ha destroyed | High (modelled) | Direct habitat destruction |
| Wildfire (historical) | 692 ha suppressed | High (modelled) | Direct mortality |
| Stream erosion buffer (30 m) | ~1,190 ha (2.5% of remaining) | Moderate (sensitivity analysis) | Riparian habitat loss |
| Sea-level rise (future) | ~240 ha (<0.5% of remaining) | Low (estimated) | Coastal inundation + salt intrusion |
| Yew big bud mite | 20–25% bud mortality | Moderate (literature) | Growth/fecundity reduction |
| Ungulate browsing | 60–80% seedling mortality | Moderate (literature) | Regeneration failure |
| Historical Taxol harvest | Unknown (hundreds of thousands of trees) | Low (no spatial data) | Historical direct mortality |
| Climate-driven fire increase | Potentially catastrophic | Low (projected) | Future 100% mortality in fire polygons |

The stream erosion estimate is derived from a sensitivity analysis applying a 30 m water buffer to all stream features: at 10 m resolution, this translates to 3 pixels of dilation, adding ~1,190 ha of effective habitat loss based on the riparian concentration of yew populations. The scientific basis for this buffer includes the Carnation Creek study (Hartman & Scrivener 1990) showing 20–50% peak flow increases post-logging, hydraulic geometry relationships predicting ~14% channel widening for a 30% flow increase, and PCIC climate projections indicating 5–15% additional winter discharge increases by 2050.

---

## 4. Discussion

### 4.1 Scale of Decline

Our estimate of 69.2% overall decline across three BEC zones is likely conservative. The study tiles cover approximately 9,900 km² of a total CWH+ICH+CDF zone area of ~3.8 million hectares. The tiles were not randomly selected — they include both heavily logged and well-preserved areas. Extrapolation to the full zonal extent, especially for the CWH and ICH zones where industrial forestry has operated at scale for a century, would require province-wide analysis.

The 99.1% decline in the CDF zone is particularly alarming given that this zone represents the driest, warmest edge of the yew's coastal range — conditions that may become more common under climate change. The near-complete elimination of old-growth forest (only 0.6% remaining) in the CDF leaves no refugia for yew recovery.

### 4.2 Interaction of Threats

The threats to Pacific yew are not independent. Logging creates the conditions for all other threats to intensify:

1. **Logging → stream erosion**: Removed canopy increases peak flows, eroding riparian zones where yew concentrates
2. **Logging → browsing pressure**: Clearcuts attract ungulates to forest edges, increasing browse pressure on remaining yew
3. **Logging → fire vulnerability**: Forest fragmentation creates drier microclimates and increases fire ignition points
4. **Logging → mite susceptibility**: Stressed, isolated yew populations may have reduced resistance to *Cecidophyopsis psilaspis*

The cumulative effect of these interacting threats is likely significantly greater than the sum of individually estimated impacts.

### 4.3 Limitations

1. **Spectral similarity**: The classifier cannot distinguish Pacific yew understorey from spectrally similar moist-forest conditions; the probability represents habitat suitability rather than confirmed presence
2. **VRI accuracy**: Stand-age assignments from the VRI have known errors, particularly for multi-cohort stands and post-fire regeneration
3. **Spatial coverage**: 99 tiles cover ~0.26% of BC's total area; extrapolation requires caution
4. **Static analysis**: The study represents a snapshot as of 2024; ongoing logging and climate change will alter the estimates
5. **No field validation**: Predictions have not been systematically validated against field surveys
6. **ICH coverage**: Only 14 ICH tiles were analysed, covering 13 subzones; the full ICH zone contains >20 subzones across a much larger area

### 4.4 Conservation Implications

The finding that **only 4.6% of modelled yew habitat falls inside protected areas** (2,244 ha out of 49,248 ha at P≥0.5 threshold) highlights the species' extreme vulnerability to ongoing forestry operations. The tiles with the most yew habitat — Blunden Harbour (4,609 ha), Alberni Valley (4,287 ha), Seaforth Channel (3,188 ha), Rivers Inlet (3,114 ha) — have **0% inside protected areas**.

Priority conservation actions include:
1. Inclusion of Pacific yew habitat considerations in forest stewardship plans for CWH and ICH operating areas
2. Riparian buffer widening to protect streamside yew populations from logging-driven erosion
3. Ungulate management in high-value yew stands to permit natural regeneration
4. Monitoring for *Cecidophyopsis psilaspis* spread in coastal populations
5. Long-term fire management planning that recognises yew's zero fire tolerance

---

## 5. Data Availability

The interactive web map is available at [jerichooconnell.github.io/yew_project](https://jerichooconnell.github.io/yew_project/). Source code, model weights, and analysis scripts are hosted on GitHub. Satellite embeddings were accessed via Google Earth Engine; the VRI 2024 dataset was obtained from the BC Data Catalogue. Users can contribute field observations directly through the web map interface.

---

## References

- Busing, R.T., Halpern, C.B., & Spies, T.A. (1995). Ecology of Pacific yew (*Taxus brevifolia*) in western Oregon and Washington. *Conservation Biology* 9(5): 1199-1207.
- Church, M. (2006). Bed material transport and the morphology of alluvial river channels. *Annual Review of Earth and Planetary Sciences* 34: 325-354.
- Green, K.C. & Alila, Y. (2012). A paradigm shift in understanding and quantifying the effects of forest harvesting on floods in snow environments. *Water Resources Research* 48: W10503.
- Hartman, G.F. & Scrivener, J.C. (1990). Impacts of forestry practices on a coastal stream ecosystem, Carnation Creek, British Columbia. *Canadian Bulletin of Fisheries and Aquatic Sciences* 223.
- Leopold, L.B. & Maddock, T. (1953). The hydraulic geometry of stream channels and some physiographic implications. *USGS Professional Paper* 252.
- Meidinger, D. & Pojar, J. (1991). *Ecosystems of British Columbia.* BC Ministry of Forests Special Report Series 6.
- Pike, R.G. et al. (2010). *Compendium of Forest Hydrology and Geomorphology in British Columbia.* BC MoFR Land Management Handbook 66.
- Pojar, J., Klinka, K., & Meidinger, D.V. (1991). Biogeoclimatic ecosystem classification in British Columbia. *Forest Ecology and Management* 36: 119-217.
- Schnorbus, M. et al. (2012). Impacts of climate change on water supply and demand in the Okanagan Basin. *Pacific Climate Impacts Consortium.*
- Wani, M.C., Taylor, H.L., Wall, M.E., Coggon, P., & McPhail, A.T. (1971). Plant antitumor agents. VI. The isolation and structure of taxol, a novel antileukemic and antitumor agent from *Taxus brevifolia*. *Journal of the American Chemical Society* 93(9): 2325-2327.

---

## Figure Captions

**Figure 1.** Estimated historical versus current remaining Pacific yew habitat by major BEC zone, showing percentage decline annotations.

**Figure 2.** Stacked bar chart of destroyed (red) versus remaining (green) yew habitat in each CWH subzone, ordered by total loss.

**Figure 3.** Stacked bar chart of destroyed versus remaining yew habitat in each ICH subzone.

**Figure 4.** Coastal Douglas-fir (CDF) zone land cover breakdown (left) and yew habitat status showing 99.1% decline (right).

**Figure 5.** Horizontal bar chart of estimated percentage yew habitat decline across all major BEC zones, ordered by severity.

**Figure 6.** (a) Overall yew habitat status showing remaining versus destroyed proportions; (b) distribution of remaining habitat by BEC zone.

**Figure 7.** Three-panel land cover comparison of CWH, ICH, and CDF zones showing the contrasting old-growth retention and logging intensity.

**Figure 8.** Scatter plot of logging intensity (% of subzone logged) versus estimated yew decline (%) across all BEC subzones. Bubble size is proportional to original habitat area.

**Figure 9.** Mean yew probability in old-growth forest across all BEC subzones with >100 ha of old-growth, colour-coded by zone.

**Figure 10.** Historical wildfire impact on yew habitat by decade (1910s–2020s), showing total burned area and estimated yew suppression.

**Figure 11.** Example study tiles showing raw yew probability (top) and VRI logging classification (bottom) for four representative locations.

**Figure 12.** Post-classification suppression waterfall: raw model prediction through logging, fire, and elevation filters to final current habitat estimate.

**Figure 13.** Logging age class distribution (<20 yr, 20–40 yr, 40–80 yr) across CWH, ICH, and CDF zones.

**Figure 14.** Classifier performance comparison on satellite embedding features (validation set).

**Figure 15.** Summary of threats to Pacific yew populations, showing quantified modelled impacts alongside estimated secondary threat magnitudes.

**Figure 16.** Heatmap of key metrics (old-growth yew rate, original habitat, current habitat, destroyed, fire loss) across the top 25 most-impacted BEC subzones.

**Figure 17.** Old-growth versus logged forest areas in CWH, ICH, and CDF, showing the old-growth:logged ratio.

**Figure 18.** Location map of 99 study tiles across British Columbia (circles = coastal, triangles = ICH interior).

**Figure 19.** Decline pathway waterfall charts for CWH, ICH, and CDF zones showing estimated original habitat through logging, fire, and elevation losses to final remaining habitat.
