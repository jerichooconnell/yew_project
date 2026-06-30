# Satellite Embedding–Based Mapping of Pacific Yew (*Taxus brevifolia*) Habitat Decline Across British Columbia, and an IUCN Red List Assessment of the Canadian Population

**Authors:** Peter Cresey, Karo Castro-Wunsch, Jericho O'Connell

**Correspondence:** `jericho.oconnell@gmail.com`

**Target journal:** *Biological Conservation*

**Keywords:** Pacific yew, *Taxus brevifolia*, habitat modelling, satellite embeddings, foundation models, old-growth forest, IUCN Red List, British Columbia, biogeoclimatic zones, paclitaxel

---

## Abstract

Pacific yew (*Taxus brevifolia*) is a slow-growing, shade-tolerant understory conifer that is both ecologically and culturally significant across its Pacific Northwest range. In British Columbia (BC), the species faces severe population decline driven primarily by industrial clear-cut logging, compounded by wildfire, stream erosion, and the legacy of paclitaxel (Taxol) bark extraction. Despite these pressures, no spatially explicit, province-scale assessment of yew habitat extent or decline has previously been undertaken.

We present a machine-learning framework using 64-band satellite spectral embeddings from the Google AlphaEarth Foundation model, classified by an XGBoost ensemble (AUC-ROC 0.996), to map Pacific yew habitat probability at 10 m resolution across 9,800 km² of BC spanning three major biogeoclimatic zones: Coastal Western Hemlock (CWH), Interior Cedar–Hemlock (ICH), and Coastal Douglas-fir (CDF). Post-classification suppression using BC Vegetation Resources Inventory stand-age records, historical fire perimeters, and digital elevation data quantifies both current remaining and historically destroyed habitat.

We estimate that **154,483 ha** of yew habitat existed historically across 98 study tiles, of which **47,534 ha (30.8%)** remains today — a **69.2% decline**. Calibrating from 461 deduplicated FAIB permanent sample plot records (conditional mature-stem density 10 stems/ha, median), this habitat supports an estimated **375,000–475,000 mature individuals**, far exceeding IUCN Criterion C thresholds but carrying a large-tree size structure consistent with historical over-harvest. The CDF zone has suffered near-total loss (99.1% decline), followed by ICH (74.7%) and CWH (69.1%). This estimated loss of 106,949 ha is overwhelmingly attributable to industrial logging, and only 5.6% of mapped yew habitat falls inside provincial parks — rising to 11.0% when conservancies and national parks are included. Field-measured population size structure (*n* = 120 trees) reveals a significant deficit of large-diameter adults (DBH > 30 cm: 7 observed vs. ~32 expected under a de Liocourt stable-population model; binomial *p* ≈ 4 × 10⁻⁹), independently corroborated by the FAIB inventory (*n* = 461 deduplicated stems; *p* ≈ 1.2 × 10⁻⁴³), consistent with selective removal of large individuals during the 1989–1993 taxol bark harvest.

Applying IUCN Red List criteria to the Canadian BC population, we find that the documented habitat decline supports a classification consistent with **Endangered (EN A2c)** — a significant upgrading from the current global Near Threatened (2013) listing. Because this assessment is inferred from modelled habitat decline rather than direct population census, we present it as evidence for reassessment rather than a formal listing. On the same basis, the CDF and ICH subpopulations individually reach Critically Endangered thresholds. We recommend a formal COSEWIC and IUCN reassessment and urge integration of Pacific yew habitat into forest stewardship planning across the CWH and ICH zones.

---

## 1. Introduction

Pacific yew (*Taxus brevifolia*) is a gymnosperm tree that occurs in the understorey of coniferous and deciduous forests throughout the Pacific Coast, from the southern tip of Alaska south through British Columbia into northern California. In BC, its range extends into the Interior Cedar–Hemlock (ICH) zone east of the Coast Mountains (Woods 2012; Pojar & MacKinnon 1994). It is a long-lived, dioecious conifer that occurs primarily as a scattered understorey tree in moist, old-growth forests of the Coastal Western Hemlock (CWH), ICH, and Coastal Douglas-fir (CDF) biogeoclimatic zones (Pojar et al. 1991; Meidinger & Pojar 1991).

Pacific yew plays a crucial role in coastal and riparian ecosystems, providing habitat and supporting biodiversity (Bolsinger & Jaramillo 1990). It is also culturally significant to Indigenous communities across its range, particularly in medicine, tool-making, and spiritual practices (Turner 1998, 2021; Turner & Hebda 2012). Despite its ecological and cultural importance, the species has faced multiple compounding threats throughout the twentieth century.

The species gained worldwide attention when the National Cancer Institute discovered in the 1960s that its bark contained paclitaxel, a potent mitotic inhibitor effective against ovarian, breast, and lung cancers (Wani et al. 1971). Because paclitaxel yield is extraordinarily low — approximately 1 kg per 9,080 kg of bark — and extraction required killing the tree, a "yew rush" in the early 1990s resulted in the destruction of hundreds of thousands of mature Pacific yew trees across the Pacific Northwest (Hartzell 1990). The crisis was largely averted by 1993 when semi-synthetic production from European yew (*Taxus baccata*) needles eliminated the need for wild bark harvest.

*Taxus brevifolia* is currently listed on the IUCN Red List as Near Threatened (Thomas 2013), with population decline attributed to historical bark harvesting and continuing industrial logging. This has resulted in a near 30% reduction in global population that is still considered to be declining. However, the primary ongoing threat to Pacific yew populations is industrial clear-cut logging. As a shade-obligate species requiring old-growth canopy conditions, Pacific yew is eliminated entirely from clearcut blocks and requires over a century to recolonise (Busing et al. 1995). It is particularly sensitive to forest fragmentation, making it highly dependent on old-growth ecosystems (Arsenault & Bradfield 1995; Reynolds 2022). Studies confirm that Pacific yews rarely re-establish in disturbed environments, and their scarcity of fruiting and low seedling survival rates further hinder recovery (Hartzell 1990; Council of the Haida Nation 2016). The species "exhibits slow recovery from major disturbances, potentially requiring centuries to return to pre-harvest abundance" (Busing et al. 1995). Pacific yew's very thin bark (approximately 0.5 cm) makes the tree highly sensitive to fire, hindering re-growth and re-establishment (Reynolds 2022; Busing et al. 1995). The increasing occurrence of high-intensity forest fires associated with climate change represents an additional negative population factor.

The remaining old-growth forests across BC are increasingly at risk from logging and climate change (Bergeron & Fenton 2012; Lindenmayer et al. 2012; Woods 2012). As the life processes of Pacific yew operate on a much longer time scale than modern forestry practices, which favour short harvest rotations and extensive clear-cutting, the yew struggles to re-establish itself once old-growth forest is cleared (Arsenault & Bradfield 1995; Council of the Haida Nation 2016).

Within this context of negative pressures, there is also a recognised lack of inventory data to understand the Pacific yew's current population status. There are acknowledged knowledge gaps and a lack of hard data on the species' current viability and distribution in BC (Reynolds 2022). This is a barrier to understanding the risks to the species and hinders informed decision-making by the provincial government and other stakeholders. An understanding of the species' population levels would also partly address First Nations concerns over the historical overharvesting of Pacific yew (Council of the Haida Nation 2016; Turner & Hebda 2012).

No comprehensive spatial assessment of Pacific yew habitat extent or decline has been attempted at the provincial scale. Existing range data consist of point observations (e.g., iNaturalist, herbarium records) supplemented by incidental mentions in forest inventory plots. This study addresses that gap using satellite-based spectral embeddings from a recently released deep learning foundation model and machine learning. The objective is to model the spatial distribution and quantify the decline of Pacific yew habitat across its primary BC range — and to apply the resulting habitat decline estimates to a formal IUCN Red List assessment of the Canadian BC population.

---

## 2. Methods

### Study Rationale

While population estimates have been produced by the US Forest Service from forest inventory data in the 1990s (TODO cite), the BC Forest Analysis and Inventory Branch (FAIB) permanent sample plot (PSP) network records 461 unique Pacific yew (*Taxus brevifolia*, species code TW) stems across 122 plots province-wide. These plots are not stratified by BEC subzone in proportion to yew abundance, and the majority fall in CWH stands dominated by other conifers; the number of yew records per subzone is insufficient to separately parameterise population dynamics for CWH, ICH, and CDF. The FAIB data were therefore used here to calibrate conditional stem density (where yew is present; §4.6) rather than as a direct population model. The aim of this study was to combine spatially explicit habitat mapping with FAIB-calibrated density to produce a contemporary, province-scale population estimate using the most current remote sensing tools.

Thus, this study was based on 10 m resolution spectral features extracted from the Google AlphaEarth Foundation model satellite embeddings (Brown et al. 2025) combined with iNaturalist crowd-sourced occurrence data, which was used to build an XGBoost (TODO cite xgboost) classifier to give a probability of finding a yew tree in a given 10 m × 10 m pixel. This probability map was then adjusted based on known logging, fire, and riparian erosion criteria to give a final yew probability map, from which total habitat area is estimated as the probability-weighted sum of pixel areas (probability mass; §2.5.4). Multiplying this habitat area by the FAIB-calibrated conditional mature-stem density provides a population estimate of the order of 375,000–475,000 mature individuals (§4.6).

Computational constraints limited the scope of this survey to 98 tiles selected to represent the diversity of BC yew habitat, rather than exhaustively covering the entire provincial range. Tiles, rather than individual pixels, were used as the unit of analysis to display the geographic predictions of the model, better enabling assessment of whether the model agreed with known habitat associations of the yew, such as higher densities near rivers and riparian zones and low densities in disturbed areas.

### 2.1 Study Area
The 98 study tiles (each ~10 × 10 km, ~100 km²) were distributed across British Columbia (Figure 5), covering a total of approximately 9,800 km². Of these, 85 tiles were located in coastal BC within the CWH and CDF zone, and 14 tiles were placed within the Interior Cedar–Hemlock (ICH) zone in the BC interior. Although the center of a tile would be in either the ICH, CDF, or CWH zone tile areas would sometimes sample adjecent zones. A full list of zones and areas sampled can be found in Table 1.

#### 2.1.1 Coastal Tile Selection

The 85 coastal tile centre pixels were selected by random stratified sampling within the CWH and CDF biogeoclimatic zones. The 14 ICH tiles were generated by random stratified sampling within the ICH biogeoclimatic zone. Boundary polygons for the zones were from the BC BEC map, v12. Sampling used a fixed random seed (42) projected in BC Albers and converted to WGS84.

#### Table 1.

Across all 98 tiles, the analysis intersects 12 major BEC zones and 69 subzones:

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

### 2.3 Training Data

#### 2.3.1 Positive Samples

Yew presence records comprised 1,043 research-grade *Taxus brevifolia* observations from iNaturalist (BC and Washington/Oregon), partitioned into 834 training and 209 validation records. iNaturalist records were retained only where the coordinate showed no sign of development or logging: although some yew trees persist after logging, these represent a small minority of logged area in the province and their mortality in plantations is high, and urban locations no longer reflect habitat in which yew grows naturally, both classes were therefore excluded from training to avoid confusing the model. Centre-pixel 64-dimensional embeddings were extracted at each observation location, yielding per-observation feature vectors of length 64 representing the spectral embedding at the pixel containing the GPS coordinate. An additional 64 field-verified positive observations from southern Vancouver Island were included in the positive training set, giving 1,107 combined positive records.

#### 2.3.2 Negative Samples

Negative (yew-absent) training data comprised 11,452 samples drawn from:
- BC Forest Analysis and Inventory Branch (FAIB) forest plots in non-yew species assemblages
- Alpine and subalpine locations above the elevational range of yew
- CWH-zone locations with confirmed non-yew canopy dominance (western hemlock, Sitka spruce stands)

The combined training set therefore comprised 12,495 unique records (1,043 positives and 11,452 negatives), with the GEE-derived negatives duplicated 2× at training time for an effective negative weight of 20,446.

### 2.4 Classifier Architecture

#### 2.4.1 Production XGBoost Model

The production classifier is an XGBoost gradient-boosted tree ensemble trained on the 64 raw embedding dimensions with the following hyperparameters:

| Parameter | Value |
|-----------|-------|
| Objective | `binary:logistic` |
| Eval metric | AUC |
| Max depth | 6 |
| Learning rate | 0.05 |
| Subsample | 0.8 |
| Column sample by tree | 0.8 |
| Number of boosting rounds | 500 (early stopping at 50) |

The model was trained with spatial cross-validation using K-means clustering (10 clusters) for train/validation/test splits (70/10/20%) to mitigate spatial autocorrelation.

#### 2.4.2 Comparison Models

Six additional classifiers were evaluated on identical features and splits (Figure 6):

| Model | AUC-ROC | Accuracy | F1 Score |
|-------|---------|----------|----------|
| **XGBoost (production)** | **0.996** | **0.989** | **0.947** |
| MLP + StandardScaler | 0.9961 | 0.986 | 0.977 |
| MLP raw embeddings | 0.9962 | 0.976 | 0.960 |
| Random Forest | 0.9896 | 0.984 | 0.974 |
| kNN (k=3) | 0.9909 | 0.911 | 0.833 |
| Logistic Regression | 0.9165 | 0.813 | 0.562 |

The production XGBoost AUC-ROC of 0.996 is statistically indistinguishable from the best-performing neural models. XGBoost was selected as the production model for its comparable performance, interpretability, and computational efficiency for per-pixel inference across ~100 million pixels. Although the MLP achieved comparable held-out performance, its propensity for overfitting, evident as noise in the spatial distribution of its classifications, discouraged its use on generalizability grounds.

#### 2.4.3 Alternative Feature Sets

A separate XGBoost model trained on 35 engineered features from the BC Vegetation Resources Inventory achieved AUC 0.82 with only 26% recall at operational thresholds, and a multi-modal model combining spectral embeddings with inventory features yielded only 5% recall due to scale mismatch. These results confirmed that satellite spectral embeddings combined with iNaturalist data, alone provided sufficient classification performance.

### 2.5 Post-Classification Suppression

Raw XGBoost probabilities represent spectral habitat suitability but do not account for disturbance history. We applied a sequential suppression pipeline to convert raw probabilities into ecologically realistic current habitat estimates.

#### 2.5.1 Logging Suppression

Stand-age information was extracted from the BC Vegetation Resources Inventory (VRI 2024, 6.87 million polygons). Each 10 m pixel was assigned to one of seven land-use categories:

| Category | Description | Suppression Factor |
|----------|-------------|-------------------:|
| 1 | Water / non-forest | ×0.00 |
| 2 | Logged <20 yr | ×0.00 |
| 3 | Logged 20–40 yr | ×0.00 |
| 4 | Logged 40–80 yr | ×0.00 |
| 5 | Forest 80–150 yr | ×0.00 |
| 6 | Alpine / barren | ×0.00 |
| 7 | Old-growth (>150 yr) | ×1.00 |

For habitat loss estimation, all non-old-growth forested land (categories 2–5) is treated as "logged" to estimate what yew habitat would have existed under pre-logging old-growth conditions. Although forest in the 80–150 yr class retains some potential to regenerate yew habitat and old-growth characteristics over time, by the IUCN Red List's reproductive-age criterion such stands would contain no reproductively mature adults, supporting their treatment as currently unsuitable.

#### 2.5.2 Fire Suppression

Historical fire perimeters (1900–2024) from the BC historical fire polygon dataset were filtered to fires ≥100 ha. A time-dependent fire modifier was applied:

$$\text{fire\_modifier} = \frac{2024 - \text{fire\_year}}{124}, \quad \text{clamped to } [0, 1]$$

This assumes linear recovery over 124 years (the span of the fire record), with recent fires causing near-complete suppression. The linear form is a simplifying assumption: post-fire recovery of shade-dependent understorey conifers is in reality non-linear and strongly site-dependent (varying with fire severity, residual canopy, and proximity to seed sources), and given yew's very thin bark and slow recolonisation it may be optimistic for high-severity burns. We did not perform a sensitivity analysis on the 124-year recovery period; because fire accounts for <1% of total estimated loss (§3.6), the overall decline estimate is insensitive to this choice. A partial-suppression modifier (rather than complete removal) was judged appropriate because fires, unlike clear-cut logging, generally do not consume the full area of the perimeter polygon, so mature individuals typically persist within burned polygons. For overlapping polygons, the most recent fire year takes precedence.

#### 2.5.3 Elevation Suppression

A Copernicus GLO-30 Digital Elevation Model was used to suppress yew predictions at very low coastal elevations:

$$\text{elev\_factor} = \text{clip}\left(\frac{\text{elevation}}{30}, 0, 1\right)$$

This linear ramp from 0 at sea level to 1 at 30 m removed 18,434 ha of false-positive predictions from intertidal and low-coastal zones. The 30 m threshold is applied as a conservative heuristic rather than a hard ecological boundary: Pacific yew is not salt-tolerant and does not establish in the intertidal, foreshore, and saline-influenced floodplain settings that dominate below this elevation (Busing et al. 1995), where the embedding classifier nonetheless returns occasional high probabilities driven by spectrally similar moist, closed-canopy vegetation. The linear ramp avoids a sharp cut-off and down-weights, rather than eliminates, predictions in the transitional 0–30 m band.

#### 2.5.4 Habitat Loss Estimation

For each BEC subzone, yew prevalence rate (*r*) was computed as the mean raw XGBoost probability across all old-growth pixels. This yew prevalence rate was used as an estimator of the original density of yews in the subzone. Estimated original habitat was *r* × (old-growth + logged pixels) × 0.01 ha/pixel. The factor 0.01 ha/pixel converts pixel counts to area: at the 10 m native resolution of the embeddings, each pixel covers 10 m × 10 m = 100 m² = 0.01 ha. Here and throughout, "probability mass" denotes the sum of per-pixel yew probabilities (each in [0, 1]) multiplied by this per-pixel area — equivalently, the area-weighted expected habitat extent. This approach uses continuous probability mass rather than binary thresholding (counting only pixels above a fixed probability cut-off), providing a more robust cumulative habitat-area estimate that is insensitive to the choice of threshold.

### 2.6 Secondary Threat Assessment

In addition to the quantitative logging and fire analysis, we reviewed the scientific literature to characterise secondary threats not directly modellable from remote sensing.

#### 2.6.1 Stream Erosion and Riparian Habitat Loss

Pacific yew preferentially occupies moist riparian zones (Busing et al. 1995). Logging-driven hydrological changes increase peak flows by 20–50% (Hartman & Scrivener 1990), causing channel widening proportional to $W \propto Q^{0.5}$ (Leopold & Maddock 1953). A water buffer sensitivity analysis was run across all 41 tiles with available grid data: applying binary morphological dilation (3 pixels = 30 m) to all water category pixels, then summing yew probability mass in the buffered old-growth pixels. This yielded **1,717 ha** of yew probability mass at risk from riparian erosion (5.9% of the 29,028 ha of remaining yew probability mass in those 42 tiles — a subset of the full 47,534 ha, restricted to the tiles for which the VRI water-category grid was available).

#### 2.6.2 Sea-Level Rise and Saltwater Intrusion

Pacific yew being sensitive to salt; saline groundwater intrusion kills root systems. Habitat below 1.0–1.5 m elevation in 50-year projections faces 100% loss, with an additional 15–20 m inland buffer for saltwater intrusion effects. Our elevation suppression (§2.5.3) captures current low-elevation effects but does not project future sea-level scenarios.

#### 2.6.3 Yew Big Bud Mite 

The Yew Big Bud Mite (*Cecidophyopsis psilaspis*) causes bud galls on *Taxus* species, with terminal bud mortality averaging over 20% in infested coastal BC populations, reducing growth rates by ~20% and seed (aril) production by ~25%  (TODO find citation).

#### 2.6.4 Ungulate Browsing

Wild ungulates preferentially browse Pacific yew foliage in winter. Browsing pressure causes 60–80% seedling/sapling mortality in areas with high ungulate density (&gt;10 deer/km²), creating a "browsing ceiling" that prevents recruitment to the established understorey stage (Busing et al. 1995; Council of the Haida Nation 2016).

#### 2.6.5 Wildfire Frequency Increase

Climate change is projected to reduce fire return intervals in drier maritime subzones to 80–120 years, potentially preventing populations from reaching reproductive maturity (~80–100 years) between successive fires.

#### 2.6.6 Historical Taxol Bark Harvest

Between 1989 and 1993, at peak harvest (1991), an estimated 360,000 mature yew trees per year were felled across the Pacific Northwest for paclitaxel extraction. Treating a single cancer patient required the bark of six 100+-year-old trees. This harvest was not spatially documented in GIS and cannot be directly modelled, but its legacy is embedded in the depleted large-tree cohort documented in §3.10.

### 2.7 Field Sampling of Population Size Structure

To ground-truth population age structure independently of remote sensing, diameter at 30 cm above ground was recorded for *n* = 120 Pacific yew individuals along East Muir Creek in January 2026. All yew stems ≥ 1 cm CBH within 30 m of the stream bank were measured along a 3 km portion of the river.

Size-class structure was compared against the de Liocourt (1898) reverse-J model, which describes the expected stem-frequency distribution in a balanced, self-sustaining uneven-aged stand as a constant ratio *q* of stems between successive (ascending) diameter classes. We use this model as a descriptive reference for a stable understory population rather than as a mechanistic demographic model: because Pacific yew is a slow, plastic, suppression-tolerant grower, diameter is an imperfect proxy for age, and we therefore frame the comparison in terms of size structure, not age structure. For Pacific yew in intact old-growth CWH and ICH stands, Bolsinger & Jaramillo (1990) and Graham (1994) document *q* ≈ 1.4–1.6 per 10-cm class; we adopt *q* = 1.5 as the central reference and report all comparisons across the full 1.4–1.6 range so that conclusions do not depend on a single assumed value. Expected stem counts per 10-cm class were obtained by scaling the geometric series *q*<sup>−i</sup> to the observed sample size.

Three complementary tests were applied. First, because the *a priori* prediction of selective large-tree removal concerns the large-diameter tail, we tested the count of stems > 30 cm DBH against the proportion expected under each reference *q* using an exact binomial test (one-sided, deficit). Second, a Pearson χ² goodness-of-fit test was applied to the binned counts (the three classes below 30 cm plus a pooled ≥ 30 cm class, so that all expected counts exceeded five). Third, to summarise the realised decline rate we fitted a single reverse-J slope to the whole observed distribution by ordinary least squares on the log-counts of all occupied 10-cm classes, yielding an empirical population *q* with a 95% confidence interval from a non-parametric bootstrap (2,000 resamples of the *n* = 120 stems); this whole-population *q* — unlike a fit restricted to the sparse adult tail — is well constrained and its direction relative to *q* = 1.5 is interpretable. Two caveats bound all of these tests: (i) understory stems < 10 cm DBH are readily overlooked, so the smallest class may be under-counted; and (ii) multi-stemmed (layering) individuals were recorded as separate stems (§2.7), which can inflate the small-stem classes relative to the genet-based assumptions of the de Liocourt model.

### 2.8 Interactive Web Map

Results are presented via an interactive web map hosted on GitHub Pages ([jerichooconnell.github.io/yew_project](https://jerichooconnell.github.io/yew_project/)). The map displays yew probability overlays for all 98 tiles, VRI-derived logging age class rasters, 5,700 historical fire polygons ≥100 ha (1900–2024) colour-coded by age, 1,201 protected areas (provincial parks, conservancies, national parks, ecological reserves, and protected areas, from the BC Data Catalogue), and a crowd-sourced field observation reporting interface.

---

## 3. Results

### 3.1 How much yew habitat remains, and where?

Across all 98 study tiles (~9,800 km², 69 BEC subzones), we estimate that 154,483 ha of yew habitat existed historically under pre-logging old-growth conditions, of which 47,534 ha (30.8%) remains today (Table 1, Figure 1). This represents an overall decline of 69.2%, with per-zone severity ranked in Figure 4. Unless otherwise stated, all headline habitat areas are *continuous probability-mass* estimates (the sum of per-pixel yew probability × 0.01 ha; §2.5.4) rather than counts of pixels above a fixed threshold; threshold-based figures used in specific secondary analyses (37,858 ha at P ≥ 0.5 and 29,028 ha in the 41-tile erosion subset) are defined where they appear.

The remaining 47,534 ha is not evenly distributed. The Coastal Western Hemlock zone holds 72% of it (34,386 ha), concentrated in remote hypermaritime terrain where logging pressure has been lower. The Interior Cedar–Hemlock zone holds a further 13% (6,385 ha), mostly in isolated valley-bottom fragments. At the opposite extreme, the Coastal Douglas-fir zone — which historically supported some of the densest yew populations on Vancouver Island and the Gulf Islands — retains only 36 ha. The geographic pattern of habitat, logging pressure, old-growth refugia, and protection across the study tiles is summarised in Figure 10: logging concentrates in the southern coast and interior, the largest old-growth refugia persist on the central and north coast, and protected areas are sparse and poorly aligned with the richest remaining habitat. The underlying per-tile model output is shown directly in Figures 11 and 12, which place the predicted-probability rasters at their true locations across Vancouver Island and the Central and North Coast respectively.

The model pipeline that converts raw per-pixel spectral probabilities to this final estimate applies three sequential corrections (Figure 2): logging suppression (removing predictions from stands <150 yr old, −146,506 ha; the correspondence between predicted probability and logging status is shown tile-by-tile in Figure 13), fire suppression (a time-decaying modifier on historical burn perimeters, −692 ha), and elevation correction (removing false positives below 30 m elevation, −18,434 ha). The elevation correction is a model accuracy adjustment, not a quantity of ecological loss, and is excluded from all habitat-decline accounting. Mean yew probability in old-growth pixels is highest in warm, moist subzones at low to mid elevation (ICHxw: 0.489; CWHvh1: 0.411; ICHdw1: 0.410; CDFmm: 0.381; Figure S6), reflecting the species' preference for frost-free, moisture-retentive sites.

**Table 1. Summary statistics by major BEC zone (zones with >10 ha estimated original yew habitat).**

| Zone | Est. Original (ha) | Current Remaining (ha) | Decline (%) | Logged Area (ha) | Fire Loss (ha) |
|------|--------------------:|-----------------------:|------------:|-----------------:|---------------:|
| CWH  | 111,407 | 34,386 | 69.1 | 312,128 | 460 |
| ICH  | 25,257 | 6,385 | 74.7 | 97,071 | 152 |
| CDF  | 3,889 | 36 | 99.1 | 9,986 | 0 |
| ESSF | 3,957 | 1,742 | 56.0 | 29,392 | 64 |
| MH   | 7,078 | 4,161 | 41.2 | 21,709 | 7 |
| IDF  | 1,697 | 463 | 72.7 | 5,768 | 0 |
| MS   | 552 | 114 | 79.4 | 6,511 | 6 |
| CMA  | 609 | 225 | 63.0 | 10,474 | 1 |
| **Total** | **154,483** | **47,534** | **69.2** | not summed† | **692** |

†The "Logged Area" column reports total logged *forest* area within each zone (all non-old-growth forested land), a distinct and larger quantity than the area of logged *yew habitat*. It is not summed because zone totals reflect overall forest disturbance, not yew-specific loss; the estimated logged yew-habitat area (original minus remaining) is 106,949 ha.

### 3.2 What drove the decline?

#### 3.2.1 Logging as the primary driver

Industrial clear-cut logging accounts for an estimated **106,949 ha** of destroyed yew habitat — approximately 99.4% of all ecological loss documented in this study. This dominance is reflected in the suppression pipeline: the logging filter removes 146,506 ha (68.7%) of raw model predictions, far exceeding the contributions of fire (692 ha) and all secondary threats combined. The three focal zones exhibit qualitatively different logging-driven decline pathways (Figure S13):

**Coastal Western Hemlock (CWH):** The CWH zone was logged extensively from the 1940s through the 1980s, with the 40–80 yr stand-age class now dominant (125,262 ha). Despite retaining 37.3% old-growth by area, 35.2% of the landscape is logged, and decline rates exceed 83% in the drier maritime subzones most intensively harvested (CWHxm2: 89.5%; CWHmm2: 90.7%; CWHdm: 83.9%). The CWHvh2 subzone retains the most yew habitat in absolute terms (10,355 ha) because its remote hypermaritime terrain (121,647 ha old-growth) has been less accessible to industrial forestry (Figure 3).

**Interior Cedar–Hemlock (ICH):** The ICH zone shows the second-highest overall decline (74.7%), exceeding the CWH despite a smaller absolute area, because 68.0% of the ICH landscape has been logged versus 35.2% of the CWH. Yew habitat in the ICH concentrates in accessible valley bottoms — precisely the terrain most intensively harvested. The ICHxw subzone has the highest raw yew probability in old-growth of any sampled subzone (0.489), yet has lost 83.8% of its estimated original habitat to logging. Recent logging (<20 yr stands: 22,579 ha, 23% of total logged ICH area) indicates that harvest pressure remains active (Figure S1).

**Coastal Douglas-fir (CDF):** The CDF zone represents a qualitatively different, more extreme trajectory: near-complete regional extirpation. Of 36,055 ha analysed, only 214 ha of old-growth remains (0.6%), and modelled current habitat is just 36 ha — a 99.1% decline. The CDFmm subzone historically had the highest yew prevalence of any sampled subzone (0.381), confirming that the CDF was prime yew habitat. Its near-total loss reflects not only commercial logging but decades of urban expansion and agricultural conversion across southern Vancouver Island and the Gulf Islands (Figure S2).

#### 3.2.2 Fire as a secondary driver

Across all tiles, 5,700 historical fire perimeters (1900–2024) covering approximately 96,543 ha intersected the study area, yielding an estimated **692 ha** of yew habitat suppression in remaining old-growth stands — small relative to logging but non-negligible in drier interior subzones (ICH: 152 ha; CWH: 460 ha). Fire suppression is estimated by a time-decaying modifier that assumes linear recovery over 124 years; under this model, older fires retain partial yew habitat because fires rarely consume the full polygon and mature individuals persist within burn perimeters. The 2020s burned area (8,146 ha) exceeds the 2010s (4,859 ha), consistent with an accelerating fire season driven by climate change (Figure S7), suggesting that fire's contribution to future habitat loss will grow.

Because fire is applied to the logging-suppressed grid in the pipeline, fire's contribution to habitat loss in areas that were *also* logged is not captured in the 692 ha figure. To estimate fire's independent contribution, we applied the fire modifier to the raw (unlogged) probability grid for the 41 tiles with cached pixel data, then scaled to all 98 tiles. This yields an estimated **3,460 ha** of yew habitat that fire perimeters would have suppressed independent of logging — approximately 2.2% of the original 154,483 ha. Of this, ~3,031 ha (88%) overlaps with areas that were subsequently logged, meaning the 692 ha figure captures fire's current net contribution while the remaining ~2,749 ha represents *dual-threatened* pixels where both logging and historical fire have affected the same yew habitat. The dominance of logging in the combined loss (>97% under either accounting) is unchanged by this decomposition, but it suggests that a subset of the logged landscape faces compounded recovery barriers beyond logging alone.

#### 3.2.3 Cumulative secondary threats

Beyond logging and fire, a suite of secondary threats compounds the ecological impact (Table 2). Of the quantified secondary threats, stream erosion is the most spatially significant: applying a 30 m buffer (3-pixel morphological dilation) to all VRI water-category pixels across 41 tiles with available grid data and summing yew probability mass in the buffered old-growth zone yields 1,717 ha at risk from riparian erosion — 5.9% of the 29,028 ha remaining yew probability mass in those tiles. This estimate draws on the Carnation Creek study showing 20–50% peak flow increases post-logging (Hartman & Scrivener 1990) and hydraulic geometry relationships predicting ~14% channel widening for a 30% flow increase ($W \propto Q^{0.5}$; Leopold & Maddock 1953).

**Table 2. Estimated impacts of threats to Pacific yew in BC.**

| Threat | Estimated Impact | Confidence | Type |
|--------|------------------|------------|------|
| Clear-cut logging | 106,949 ha total habitat loss | High (modelled) | Direct habitat destruction |
| Wildfire (historical) | 692 ha suppressed | High (modelled) | Direct mortality |
| Stream erosion buffer (30 m) | ~1,717 ha (5.9% of remaining in sampled tiles) | Moderate | Riparian habitat loss |
| Sea-level rise (future) | ~240 ha (<0.5% of remaining) | Low (projected) | Coastal inundation |
| Yew big bud mite | 20–25% bud mortality | Moderate (literature) | Growth/fecundity reduction |
| Ungulate browsing | 60–80% seedling mortality | Moderate (literature) | Regeneration failure |
| Historical Taxol harvest | Unknown (hundreds of thousands of trees) | Low (no spatial data) | Historical direct mortality |

### 3.3 Are remaining populations viable? Field evidence from population size structure

Independent of the remote sensing analysis, field-measured population size structure provides a second line of evidence on species status. Measurement of *n* = 120 Pacific yew individuals yielded CBH values of 1–218 cm, corresponding to converted DBH values of 0.3–69.4 cm (mean DBH = 15.0 cm, median = 12.7 cm; Figure 7). Size-class structure differed markedly from the de Liocourt reference distribution (*q* = 1.5), with a pronounced deficit of large-diameter stems and a surplus of mid-sized stems (Table 3).

**Table 3. Observed versus expected (de Liocourt *q* = 1.5) stem counts by 10-cm DBH class.**

| DBH Class (cm) | Observed | Expected (*q* = 1.5) | Surplus / Deficit |
|:---|---:|---:|---:|
| 0–10 | 36 | 41.6 | −5.6 |
| 10–20 | **56** | 27.7 | **+28.3 surplus** |
| 20–30 | 21 | 18.5 | +2.5 surplus |
| 30–40 | 2 | 12.3 | −10.3 |
| 40–50 | 4 | 8.2 | −4.2 |
| 50–60 | 0 | 5.5 | −5.5 |
| 60–70 | 1 | 3.7 | −2.7 |
| 70–80 | 0 | 2.4 | −2.4 |

†The large-tree deficit is insensitive to the assumed *q*: the expected count > 30 cm DBH is 38.2 (*q* = 1.4), 32.1 (*q* = 1.5), and 27.1 (*q* = 1.6), against 7 observed in every case.

Only **7** stems exceed 30 cm DBH, against 32.1 expected under *q* = 1.5 (and 27–38 across *q* = 1.4–1.6). An exact binomial test rejects the stable-population expectation for this tail (7 of 120 vs. expected proportion 0.27; *p* ≈ 4 × 10⁻⁹ at *q* = 1.5, and *p* < 10⁻⁶ across *q* = 1.4–1.6), as does a χ² goodness-of-fit test on binned counts (χ² = 49.5, df = 3, *p* ≈ 1 × 10⁻¹⁰). Fitting a single reverse-J slope to the whole observed distribution gives an empirical population *q* ≈ 2.0 (95% bootstrap CI 1.6–2.7), significantly steeper than the literature-stable *q* ≈ 1.5 — the realised distribution declines faster toward the large-diameter classes than a self-sustaining population, consistent with depletion of large adults. The observed distribution is not strictly monotonic (it peaks in the 10–20 cm class); this modal bulge most plausibly reflects under-detection of the smallest understorey stems and/or the recording of layering ramets as separate stems (§2.7), and does not affect the large-tree result.

The combination of abundant mid-sized stems and depleted large adults is consistent with selective harvest of large, bark-rich individuals for paclitaxel extraction during the 1989–1993 taxol rush: a recruitment bottleneck would instead deplete the smallest classes, which is not observed. We caution that this size signature is not uniquely diagnostic of harvest — an even-aged cohort that has grown into the 10–20 cm class would produce a broadly similar structure — and that the inference depends on the field sample being a representative stand census (§2.7). With that caveat, the depletion is ecologically significant because Pacific yew reaches bark-harvest size (~30 cm DBH) only after 150–250 years of growth (Graham 1994), making any lost large-tree cohort non-renewable on forestry time scales.

**FAIB PSP comparison.** The field-sample finding is independently corroborated by the provincial FAIB permanent sample plot (PSP) inventory. Among 461 *T. brevifolia* stems recorded across 122 BC plots, the large-tree deficit is still more pronounced: only **10 stems** exceed 30 cm DBH against 115.4 expected under a stable *q* = 1.5 distribution (binomial *p* ≈ 1.2 × 10⁻⁴³; χ² = 537, df = 3, *p* ≈ 5 × 10⁻¹¹⁶). The whole-population fitted *q* = **2.73** (95% CI 2.13–3.70), steeper than the field-sample value of *q* = 2.00 but in the same direction and with a substantially stronger statistical signal owing to the larger sample (Figure S14). The FAIB plots are drawn from operational forest inventory, not targeted yew surveys, making their large-tree deficit unlikely to reflect collection bias toward small trees. Together, the two independent datasets agree that the province-wide large-tree cohort is severely depleted relative to stable-population expectations.

The two datasets differ in their small-diameter composition: the FAIB sample is heavily weighted toward the 0–10 cm class (median DBH 5.5 cm), while the field sample peaks in the 10–20 cm class (median 12.7 cm), suggesting proportionally fewer seedlings and saplings in the field sample sites. Two mechanisms could explain this discrepancy. First, wild ungulates (black-tailed deer, Roosevelt elk, and moose) preferentially browse Pacific yew foliage and suppress seedling and sapling recruitment in areas of high ungulate density; if the field sample sites experience greater ungulate pressure than the average FAIB plot, small stems would be systematically removed from the field-site understorey (Busing et al. 1995; Council of the Haida Nation 2016). Second, the eriophyid mite *Cecidophyopsis psilaspis* suppresses new bud growth in coastal yew populations (Reynolds 2022); repeated mite damage can prevent small individuals from growing into the 10–20 cm class, effectively suppressing the 0–10 cm cohort over time. The FAIB plots, drawn from operational forest inventory, are more broadly distributed across the province and may include sites where these regeneration stressors are less prevalent. Regardless of mechanism, the modal shift does not alter the core finding: both datasets show the same large-tree depletion relative to stable-population expectations.

---

## 4. Discussion

### 4.1 A decline that likely understates true losses

The 69.2% overall decline reported here is a lower-bound estimate conditional on the sampled tiles, not a validated province-wide figure. Four sources of conservatism are worth noting. First, the analysis does not model future logging under existing tenure commitments, which will continue to reduce old-growth area. The true province-wide decline across the full ~3.8 million hectares of CWH+ICH+CDF zone would require systematic coverage but would almost certainly not be lower than the 69.2% reported here. Second, the logging-loss estimate does not incorporate indirect recruitment suppression by ungulates and bud mites. Both Vancouver Island and Haida Gwaii host overabundant deer populations following the extirpation of wolves and cougars from many coastal islands (TODO cite), and ungulate browse pressure on yew seedlings and saplings (0–10 cm DBH) has been documented to cause 60–80% seedling mortality in high-density areas, creating a regeneration debt that does not appear in canopy-cover mapping. The comparison between the field sample (fewer small-diameter stems, median 12.7 cm) and the FAIB inventory (more abundant small stems, median 5.5 cm; §3.3) is consistent with ungulate pressure or *Cecidophyopsis psilaspis* mite suppression being spatially heterogeneous — more severe in the coastal field sites than in the broader FAIB plot network. Third, our fire accounting (692 ha in the sequential pipeline) underestimates fire's historical footprint: when applied to the full original habitat independent of logging, fire perimeters would have suppressed an estimated ~3,460 ha (2.2% of original), with ~2,748 ha of this in areas also affected by logging (§3.2.2). Fourth, the study tiles were not randomly selected and may undersample the most heavily logged or most remote parts of the province.

### 4.2 Two independent lines of evidence converge

The remotely sensed habitat decline, the field-measured population size structure, and the provincial FAIB PSP inventory are methodologically independent — derived from different data sources, collection protocols, and spatial scales — yet all three converge on the same diagnosis: a population that has lost a large fraction of its historical extent and whose remaining individuals show a structural deficit in the size classes responsible for reproduction and bark-yield. The depleted large-tree cohort (>30 cm DBH; §3.3) corresponds to trees of 150–700+ years of age — exactly the cohort eliminated by commercial logging of old-growth stands and targeted taxol bark extraction. A viable self-sustaining population would show a reverse-J size distribution; instead, the field sample gives *q* ≈ 2.0 (95% CI 1.6–2.7) and the larger FAIB dataset gives *q* ≈ 2.7 (95% CI 2.1–3.7), both substantially steeper than the stable-population reference *q* = 1.5. This three-way convergence strengthens the evidentiary basis for the conservation concern raised by the remote sensing results alone, and underpins the IUCN assessment in §4.6.

### 4.3 Protected areas offer insufficient refuge

Only **5.6% of mapped yew habitat falls inside provincial parks** (2,121 ha of 37,858 ha at P ≥ 0.5), rising to **11.0%** (4,180 ha) when conservancies (1,602 ha, 4.2%) and national parks (457 ha, 1.2%) are included (Figure 9). Nearly half of all mapped habitat (17,785 ha, 47%) lies in tiles with no protected area at all, including the largest individual yew concentrations: Alberni Valley (3,632 ha), Port Hardy Forest (1,173 ha), Blunden Harbour (1,085 ha), and Jervis Inlet Slopes (984 ha); the geographic concentration of this unprotected habitat on the outer and central coast is shown in Figure 8. This protected fraction falls far short of the 30% target of the Kunming-Montreal Global Biodiversity Framework (2022), and is particularly concerning given that the largest unprotected concentrations lie in active timber supply areas with no current regulatory constraint on old-growth harvesting.

Priority conservation actions include: (1) incorporation of Pacific yew habitat into forest stewardship plans for CWH and ICH operating areas; (2) riparian buffer widening to protect streamside populations from logging-driven erosion; (3) ungulate management in high-value yew stands to permit natural regeneration; (4) monitoring for *Cecidophyopsis psilaspis* spread in coastal populations; and (5) long-term fire management planning that recognises yew's zero fire tolerance.

### 4.4 Compounding threats amplify logging impacts

The threats to Pacific yew are not independent, logging creates the conditions for other threats to intensify. Canopy removal increases peak flows, eroding the riparian zones where yew is most abundant (§3.2.3; Hartman & Scrivener 1990). Clearcuts attract ungulates to forest edges, increasing browse pressure on any yew recruits attempting to establish. Forest fragmentation creates drier microclimates that raise fire risk. Stressed, isolated populations may also show reduced resistance to *Cecidophyopsis psilaspis*. The result is that the logging loss estimates in §3.2 understate the full cascade of harm: even old-growth patches that remain unlogged lose ecological value when surrounded by clearcut matrix that promotes edge effects, elevates deer density, and reduces moisture retention.

### 4.5 Limitations

1. **Spectral similarity**: The classifier cannot distinguish Pacific yew understorey from spectrally similar moist-forest conditions; probabilities represent habitat suitability rather than confirmed presence. A probability surface was used in preference to a binary map because it is threshold-free and propagates uncertainty throughout the analysis (§2.5.4).
2. **VRI accuracy**: Stand-age assignments have known errors, particularly for multi-cohort stands and post-fire regeneration, which could lead to misclassification of some second-growth as old-growth or vice versa.
3. **Spatial coverage**: 98 tiles represent a random sample; extrapolation to province-wide totals requires caution.
4. **Static analysis**: The study represents a snapshot as of 2024; ongoing logging and climate change will alter the estimates.
5. **No systematic field validation**: Model predictions have not been validated against independent field surveys across the full study extent. The crowd-sourced field reporting interface on the web map (§2.8) could enable community-based validation over time.

### 4.6 IUCN Red List Assessment for the Canadian (British Columbia) Population

The current global IUCN Red List status of *Taxus brevifolia* is Near Threatened (NT; Thomas 2013), an assessment that pre-dates any spatially explicit, province-scale analysis of BC habitat extent or decline. Using the quantitative results of the present study, we apply IUCN Red List criteria (IUCN 2012) to the Canadian BC population, which represents the northern core of the species' global range and the largest temperate rainforest population.

**Criterion A — Population size reduction inferred from habitat decline**

Criterion A2c evaluates population size reductions estimated or inferred where the reduction or its causes may not have ceased, based on observed decline in habitat area, extent, or quality. Pacific yew reaches reproductive maturity at approximately 80–100 years (Graham 1994); three generations therefore span ~240–300 years, well encompassing the industrial logging era beginning in the 1920s. Our analysis documents a 69.2% decline in modelled yew habitat across 9,800 km² of sampled BC range. The primary cause, industrial clear-cut logging, has not ceased; it continues under existing tenure commitments.

| IUCN Criterion | Threshold | This Study | Assessment |
|:---|:---|:---|:---|
| CR A2c | ≥ 80% reduction | 99.1% (CDF); 74.7% (ICH) | **CR** at subpopulation level |
| EN A2c | ≥ 50% reduction | 69.2% (BC overall) | **EN** at population level |
| VU A2c | ≥ 30% reduction | Exceeded | — |

A 69.2% estimated habitat decline where the cause is ongoing and not reversible meets the Endangered (EN) threshold (≥50%). The 99.1% CDF decline and 74.7% ICH decline each individually meet the Critically Endangered threshold (≥80%) at the subpopulation level.

**Criterion A2d — Levels of exploitation**

The taxol bark harvest of 1989–1993 constituted direct exploitation at a scale unparalleled in the species' recorded history, with an estimated 360,000 mature trees per year felled across the Pacific Northwest at peak harvest (Hartzell 1990). Although bark harvest has ceased, its legacy is embedded in the depleted large-tree cohort documented in §3.3, providing independent field-based corroboration of the A2c finding. This reinforces the EN designation under criterion A2d.

**Criterion B — Geographic range**

Criterion B1 (EOO) does not apply; the EOO substantially exceeds the 5,000 km² EN threshold. The remaining modelled yew habitat totals ~475 km² across the 98 tiles, approaching the EN B2 AOO threshold of <500 km². However, the tiles cover only ~0.26% of the CWH+ICH+CDF zone extent; the 99-tile AOO cannot be extrapolated to a true provincial AOO without systematic province-wide coverage, and Pacific yew does not formally qualify for increased protection under this criterion.

**Criteria C, D, and E**

Analysis of the 461 deduplicated yew stems from 122 FAIB PSP plots gives a conditional mature-stem density (DBH ≥ 10 cm) of **10 stems/ha** (median; IQR 4.9–24.7 stems/ha) in plots where *T. brevifolia* was recorded. Because the 47,534 ha probability-mass habitat estimate is a probability-weighted area (Σ*p* × 0.01 ha/pixel; §2.5.4) it already accounts for partial occupancy, so multiplying by the conditional density gives an order-of-magnitude population estimate of approximately **475,000 mature individuals** (range 230,000–930,000 across the IQR). This substantially exceeds the <2,500 EN threshold under criterion C. Earlier literature estimates of 1–5 individuals/ha (Bolsinger & Jaramillo 1990; Busing et al. 1995) are landscape-average densities that conflate occupied and unoccupied stands; conditional density within occupied FAIB plots is an order of magnitude higher. Criteria D and E are not met.

**Summary assessment**

Under the best available evidence, the Canadian BC population of *Taxus brevifolia* supports a classification consistent with Endangered (EN A2c) under IUCN Red List criteria, recognising that this inference rests on modelled habitat decline rather than direct population census. This conclusion is independently corroborated by the field-measured population size structure (§3.3; §4.2). The CDF and ICH subpopulations individually reach Critically Endangered thresholds. The current global designation of Near Threatened (Thomas 2013) is not consistent with the magnitude of BC habitat decline documented here, nor with the ongoing nature of the primary threat. We recommend that COSEWIC and the IUCN undertake a formal reassessment of *Taxus brevifolia* using province-scale spatial data.

---

## 5. Conclusions

This study presents the first spatially explicit, province-scale assessment of Pacific yew (*Taxus brevifolia*) habitat extent and decline in British Columbia, using 64-dimensional satellite spectral embeddings from the Google AlphaEarth Foundation model classified by an XGBoost ensemble (AUC-ROC 0.996). Across 9,800 km² of sampled CWH, ICH, and CDF forest, we estimate a 69.2% decline from approximately 154,483 ha of original habitat to 47,534 ha remaining — driven overwhelmingly by industrial clear-cut logging, with secondary contributions from wildfire, riparian erosion, and the legacy of taxol bark extraction. Calibrated against 461 deduplicated FAIB permanent sample plot records (conditional mature-stem density: median 10 stems/ha, IQR 4.9–24.7), the remaining 47,534 ha supports an estimated **375,000–475,000 mature individuals**. Both the field-measured sample (*n* = 120; binomial *p* ≈ 4 × 10⁻⁹) and the FAIB inventory (*n* = 461 deduplicated stems; *p* ≈ 1.2 × 10⁻⁴³) independently confirm severe depletion of the large-adult cohort (DBH > 30 cm) relative to the de Liocourt stable-population model, consistent with selective harvest of large individuals during the 1989–1993 taxol rush.

The finding that only 11.0% of mapped yew habitat falls within protected-area designations underscores the species' extreme exposure to continued forestry operations. Application of IUCN Red List criteria to the Canadian BC population supports an Endangered (EN A2c) designation, a significant upgrading from the current global Near Threatened listing, with the CDF and ICH subpopulations individually meeting Critically Endangered thresholds. We urge a formal COSEWIC and IUCN reassessment of the species drawing on provincial-scale spatial data, and recommend that Pacific yew habitat be explicitly incorporated into forest stewardship plans across the Coastal Western Hemlock and Interior Cedar–Hemlock zones.

---

## Acknowledgements

`TODO: Acknowledgements territorial aknowledments`

---

## Data Availability

The interactive web map is available at [jerichooconnell.github.io/yew_project](https://jerichooconnell.github.io/yew_project/). Source code, model weights, and analysis scripts are hosted on GitHub. Satellite embeddings were accessed via Google Earth Engine; the VRI 2024 dataset was obtained from the BC Data Catalogue.

---

## References

- Arsenault, A., & Bradfield, G.E. (1995). Structural–compositional variation in three age-classes of temperate rainforests in southern coastal British Columbia. *Canadian Journal of Botany* 73(1): 54–64.
- Bergeron, Y., & Fenton, N.J. (2012). Boreal forests of eastern Canada revisited: Old growth, nonfire disturbances, forest succession, and biodiversity. *Botany* 90(6): 509–523.
- Bolsinger, C.L., & Jaramillo, A.E. (1990). *Taxus brevifolia* Nutt. Pacific yew. In: Burns, R.M. & Honkala, B.H. (eds.) *Silvics of North America, Vol. 1: Conifers.* USDA Forest Service Agriculture Handbook 654, pp. 573–579.
- Brown, C.F., Kazmierski, M.R., Pasquarella, V.J., Rucklidge, W.J., Samsikova, M., Zhang, C., Shelhamer, E., et al. (2025). AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data. *arXiv*:2507.22291. https://doi.org/10.48550/arXiv.2507.22291
- Busing, R.T., Halpern, C.B., & Spies, T.A. (1995). Ecology of Pacific yew (*Taxus brevifolia*) in western Oregon and Washington. *Conservation Biology* 9(5): 1199–1207.
- Church, M. (2006). Bed material transport and the morphology of alluvial river channels. *Annual Review of Earth and Planetary Sciences* 34: 325–354.
- COSEWIC (2024). *COSEWIC Assessment and Status Report on Pacific Yew* Taxus brevifolia *in Canada.* Committee on the Status of Endangered Wildlife in Canada, Ottawa.
- Council of the Haida Nation (2016). *hlgid — Western Yew* Taxus brevifolia *Effectiveness Monitoring Report.* Council of the Haida Nation, Haida Gwaii.
- de Liocourt, F. (1898). De l'aménagement des sapinières. *Bulletin de la Société Forestière de Franche-Comté et Belfort* (July): 396–409.
- Graham, R.T. (1994). *Taxus brevifolia* Nutt. Pacific yew. In: Burns, R.M. & Honkala, B.H. (eds.) *Silvics of North America, Vol. 1: Conifers.* USDA Forest Service Agriculture Handbook 654, pp. 573–579.
- Green, K.C., & Alila, Y. (2012). A paradigm shift in understanding and quantifying the effects of forest harvesting on floods in snow environments. *Water Resources Research* 48: W10503.
- Guisan, A., & Zimmermann, N.E. (2000). Predictive habitat distribution models in ecology. *Ecological Modelling* 135: 147–186.
- Hartman, G.F., & Scrivener, J.C. (1990). Impacts of forestry practices on a coastal stream ecosystem, Carnation Creek, British Columbia. *Canadian Bulletin of Fisheries and Aquatic Sciences* 223.
- Hartzell, H. (1990). *The Yew Tree: A Thousand Whispers.* Hulogosi Communications, Eugene, OR.
- IUCN (2012). *IUCN Red List Categories and Criteria: Version 3.1*, 2nd edition. IUCN, Gland, Switzerland.
- Leopold, L.B., & Maddock, T. (1953). The hydraulic geometry of stream channels and some physiographic implications. *USGS Professional Paper* 252.
- Lindenmayer, D.B., et al. (2012). Global perspectives on reference conditions: A synthesis of factors controlling deviations from historical norms. *Ecological Indicators* 20: 47–57.
- Meidinger, D., & Pojar, J. (1991). *Ecosystems of British Columbia.* BC Ministry of Forests Special Report Series 6, Victoria, BC.
- Meyer, H.A. (1952). Structure, growth, and drain in balanced uneven-aged forests. *Journal of Forestry* 50(2): 85–92.
- Pike, R.G., et al. (2010). *Compendium of Forest Hydrology and Geomorphology in British Columbia.* BC MoFR Land Management Handbook 66.
- Pojar, J., & MacKinnon, A. (1994). *Plants of Coastal British Columbia.* Lone Pine Publishing, Vancouver, BC.
- Pojar, J., Klinka, K., & Meidinger, D.V. (1991). Biogeoclimatic ecosystem classification in British Columbia. *Forest Ecology and Management* 36: 119–217.
- Reynolds, G.E.M. (2022). *Ecological Understandings of Indigenous Management Shape the Study of Pacific Yew.* MSc thesis, University of Victoria, BC.
- Schnorbus, M., et al. (2012). *Impacts of Climate Change in Three Hydrologic Regimes in British Columbia, Canada.* Pacific Climate Impacts Consortium, University of Victoria.
- Thomas, P. (2013). *Taxus brevifolia.* IUCN Red List of Threatened Species 2013. https://dx.doi.org/10.2305/IUCN.UK.2013-1.RLTS.T42546A2985765.en
- Turner, N.J. (1998). *Plant Technology of First Peoples in British Columbia*, 2nd edition. UBC Press, Vancouver.
- Turner, N.J. (2021). *Plants of Haida Gwaii.* Sono Nis Press, Winlaw, BC.
- Turner, N.J., & Hebda, R.J. (2012). *Saanich Ethnobotany: Culturally Important Plants of the WSANEC People.* Royal BC Museum, Victoria.
- Wani, M.C., Taylor, H.L., Wall, M.E., Coggon, P., & McPhail, A.T. (1971). Plant antitumor agents VI: The isolation and structure of taxol, a novel antileukemic and antitumor agent from *Taxus brevifolia*. *Journal of the American Chemical Society* 93(9): 2325–2327.
- Woods, S. (2012). *Western Yew Management in British Columbian Forests.* BC Ministry of Forests, Lands and Natural Resource Operations.

---

## Figure Captions

*Main-text figures (7 recommended for Biological Conservation):*

**Figure 1.** Estimated historical versus current remaining Pacific yew habitat by major BEC zone, with percentage decline annotations. Error bars represent uncertainty in yew prevalence rate estimation.

**Figure 2.** Post-classification suppression waterfall: raw model prediction through logging, fire, and elevation filters to final current habitat estimate across all 98 study tiles.

**Figure 3.** Stacked bar chart showing destroyed (red) versus remaining (green) yew habitat in each CWH subzone, ordered by total estimated original habitat.

**Figure 4.** Horizontal bar chart of estimated percentage yew habitat decline across all major BEC zones, ordered by severity of decline.

**Figure 5.** Location map of 98 study tiles across British Columbia (coastal CWH/CDF tiles = circles; ICH interior tiles = triangles), overlaid on biogeoclimatic zone boundaries.

**Figure 6.** Classifier performance comparison for six models trained on 64-dimensional AlphaEarth satellite embeddings (held-out validation set).

**Figure 7.** Pacific yew population size-class distribution (DBH converted from field-measured CBH, *n* = 120). Left panel: observed stem counts per 10-cm DBH class (green bars) versus the de Liocourt stable-population reference (*q* = 1.5; red dashed) and the reverse-J slope fitted to the whole observed distribution (*q* ≈ 2.0, 95% CI 1.6–2.7; blue dash-dot). Shaded area: the large-tree deficit (> 30 cm DBH, 7 observed vs. ~32 expected). Right panel: observed versus expected (*q* = 1.5) cumulative distribution functions, with the observed median (12.7 cm) marked.

**Figure 8.** Geographic distribution of mapped Pacific yew habitat across the 98 study tiles, each symbol sized in proportion to its mapped habitat area (P ≥ 0.5) and coloured by the percentage of that habitat falling inside protected areas (red = exposed, green = protected). Most high-habitat tiles, particularly on the outer and central coast, are largely unprotected.

**Figure 9.** Where the 37,858 ha of mapped yew habitat sits. Left: composition by protection status — provincial parks (5.6%), conservancies (4.2%), and national parks (1.2%) together protect only 11.0%, leaving 89% unprotected. Right: 17,785 ha (47%) of mapped habitat lies in study tiles with no protected area at all.

**Figure 10.** Geography of the four key tile-level variables across the 98 study tiles, plotted on a common base map with marker area proportional to mapped habitat: (a) mapped habitat (P ≥ 0.5); (b) percentage of forest area logged; (c) percentage old-growth remaining; (d) percentage of habitat inside protected areas. Logging pressure concentrates in the southern coast and interior, old-growth refugia persist on the central and north coast, and protection is sparse and spatially mismatched with the largest habitat concentrations.

**Figure 11.** Predicted Pacific yew probability shown as the actual per-tile model-output rasters placed at their true geographic locations on Vancouver Island (inset: location within BC). Each labelled patch is one ~10 × 10 km study tile, coloured by suppressed yew probability (green ≈ low → red/purple ≈ high); high-probability habitat is visible in old-growth-rich tiles such as Carmanah-Walbran, Port Renfrew, and Sooke Hills, while heavily logged tiles show sparse, fragmented predictions.

**Figure 12.** As Figure 11, for the Central and North Coast — the region holding the largest remaining yew habitat. Tile imagery shows substantial predicted habitat in less-accessible outer-coast and fjord tiles (e.g., Rivers Inlet, Smith Sound, Bella Bella), consistent with lower historical logging pressure in these remote settings.

**Figure 13.** The decline driver shown directly: the same Vancouver Island tiles with predicted yew probability (left) beside their VRI logging / forest-age classification (right). Tiles dominated by old-growth (dark green, right) carry the highest predicted yew probability (left), whereas tiles dominated by recent logging (red/orange/yellow, right) are largely zeroed by the suppression pipeline.

*Supplementary figures:*

**Figure S1.** Stacked bar chart of destroyed versus remaining yew habitat in each ICH subzone.

**Figure S2.** Coastal Douglas-fir (CDF) zone land cover breakdown and yew habitat status showing 99.1% decline.

**Figure S3.** (a) Overall yew habitat status pie chart; (b) distribution of remaining habitat by BEC zone.

**Figure S4.** Three-panel land cover comparison of CWH, ICH, and CDF zones.

**Figure S5.** Scatter plot of logging intensity versus estimated yew decline across all BEC subzones. Bubble size proportional to original habitat area.

**Figure S6.** Mean yew probability in old-growth forest across all BEC subzones with >100 ha of old-growth, colour-coded by zone.

**Figure S7.** Historical wildfire impact on yew habitat by decade (1910s–2020s).

**Figure S8.** Example study tiles showing raw yew probability (top) and VRI logging classification (bottom) for four representative locations.

**Figure S9.** Logging age class distribution (<20 yr, 20–40 yr, 40–80 yr) across CWH, ICH, and CDF zones.

**Figure S10.** Summary of threats to Pacific yew showing quantified modelled impacts alongside estimated secondary threat magnitudes.

**Figure S11.** Heatmap of key metrics across the top 25 most-impacted BEC subzones.

**Figure S12.** Old-growth versus logged forest areas in CWH, ICH, and CDF, showing the old-growth:logged ratio.

**Figure S13.** Decline pathway waterfall charts for CWH, ICH, and CDF zones separately.

**Figure S14.** De Liocourt size-structure comparison: FAIB PSP inventory (*n* = 431 live stems, deduplicated) versus field sample (*n* = 120 stems). Left panel: FAIB observed stem counts per 10-cm DBH class (teal bars) versus the de Liocourt stable-population reference (*q* = 1.5; red dashed) and the whole-population fitted reverse-J slope (*q* = 2.73, 95% CI 2.13–3.70; blue dash-dot). Centre panel: equivalent analysis for the field sample (*q* = 2.00, CI 1.63–2.69). Right panel: distribution of conditional TW stem density (PHF_TREE expansion factor) across FAIB plots with *T. brevifolia* present; median 10.0 mature stems/ha (DBH ≥ 10 cm) is the basis for the population estimate in §4.6. Both datasets show significant large-tree depletion (>30 cm DBH), with the FAIB signal considerably stronger owing to the larger sample size.
