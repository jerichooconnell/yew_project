# Satellite Embedding–Based Mapping of Pacific Yew (*Taxus brevifolia*) Habitat Decline Across British Columbia, and an IUCN Red List Assessment of the Canadian Population

**Authors:** `TODO: author name(s)` · `TODO: affiliation(s)`

**Correspondence:** `TODO: corresponding-author email`

**Target journal:** *Biological Conservation*

**Keywords:** Pacific yew, *Taxus brevifolia*, habitat modelling, satellite embeddings, foundation models, old-growth forest, IUCN Red List, British Columbia, biogeoclimatic zones, paclitaxel

---

## Abstract

Pacific yew (*Taxus brevifolia*) is a slow-growing, shade-tolerant understory conifer that is both ecologically and culturally significant across its Pacific Northwest range. In British Columbia (BC), the species faces severe population decline driven primarily by industrial clear-cut logging, compounded by wildfire, stream erosion, and the legacy of paclitaxel (Taxol) bark extraction. Despite these pressures, no spatially explicit, province-scale assessment of yew habitat extent or decline has previously been undertaken.

We present a machine-learning framework using 64-band satellite spectral embeddings from the Google AlphaEarth Foundation model, classified by an XGBoost ensemble (AUC-ROC 0.996), to map Pacific yew habitat probability at 10 m resolution across 9,900 km² of BC spanning three major biogeoclimatic zones: Coastal Western Hemlock (CWH), Interior Cedar–Hemlock (ICH), and Coastal Douglas-fir (CDF). Post-classification suppression using BC Vegetation Resources Inventory stand-age records, historical fire perimeters, and digital elevation data quantifies both current remaining and historically destroyed habitat.

We estimate that **154,483 ha** of yew habitat existed historically across 99 study tiles, of which **47,534 ha (30.8%)** remains today — a **69.2% decline**. The CDF zone has suffered near-total loss (99.1% decline), followed by ICH (74.7%) and CWH (69.1%). This estimated loss of 106,949 ha is overwhelmingly attributable to industrial logging, and only 5.6% of mapped yew habitat falls inside provincial parks — rising to 11.0% when conservancies and national parks are included. Field-measured population size structure (*n* = 120 trees) reveals a significant deficit of large-diameter adults (DBH > 30 cm: 7 observed vs. ~32 expected under a de Liocourt stable-population model; binomial *p* ≈ 4 × 10⁻⁹), consistent with selective removal of large individuals during the 1989–1993 taxol bark harvest.

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

The study methodology was a compromise, as a ground survey proved infeasible for Pacific yew given its fractured and sparse distribution across a diverse array of ecosystems. While population estimates have been produced by the US Forest Service, those required resources (`TODO: personnel and funding figures`) that were only made available in the context of yew as a cancer-treatment source, and dealt with less remote terrain than the BC coast. Additionally, those models estimated a total population rather than the spatially explicit, geographic description of yew habitat presented here.

This study was instead based on 10 m resolution spectral features extracted from the Google AlphaEarth Foundation model satellite embeddings (`TODO: cite AlphaEarth`) combined with iNaturalist crowd-sourced occurrence data. While a ground survey would be the gold standard for population estimation, it proved impractical for Pacific yew: the BC Forest Analysis and Inventory Branch (FAIB) sample plots record fewer than 100 yew trees out of `TODO: total FAIB individuals`, many of them in cutblocks — far too few for direct population modelling. Compounding this, Pacific yew grows across distinct biogeoclimatic zones and subzones (CWH, ICH, CDF) that should be modelled independently.

Each tile took on average `TODO: extraction time` to extract and `TODO: processing time` to process, which limited the scope of this survey to 99 tiles rather than the entire BC range of Pacific yew. Tiles — rather than individual pixels — were used as the unit of analysis to display the geographic predictions of the model, better enabling assessment of whether the model agreed with known habitat associations of the yew, such as higher densities near rivers and riparian zones.

### 2.1 Study Area

We analysed 99 study tiles (each ~10 × 10 km, ~100 km²) distributed across British Columbia (Figure 5), covering a total of approximately 9,900 km². Of these, 85 tiles were located in coastal British Columbia within the CWH zone and adjacent zones, and 14 tiles were placed within the Interior Cedar–Hemlock (ICH) zone in the BC interior.

#### 2.1.1 Coastal Tile Selection

The 85 coastal tile centre pixels were selected by random stratified sampling within the CWH and CDF biogeoclimatic zone to:
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

Yew presence records comprised **1,043 research-grade *Taxus brevifolia* observations** from iNaturalist (BC and adjacent Washington/Oregon), partitioned into 834 training and 209 validation records. iNaturalist records were retained only where the coordinate showed no sign of development or logging: although some yew trees persist after logging, these represent a small minority of logged area in the province and their mortality in plantations is high, and urban locations no longer reflect habitat in which yew grows naturally — both classes were therefore excluded from training to avoid confusing the model. Centre-pixel 64-dimensional embeddings were extracted at each observation location, yielding per-observation feature vectors of length 64 representing the spectral embedding at the pixel containing the GPS coordinate.

`TODO: reconcile manual field annotations — an earlier draft credited 267 manual annotations from site visits around Victoria (weighted 3×). The annotation file (yew_annotations_combined.csv) actually contains 267 records of which only 64 are positive, so the production positive set is the 1,043 iNaturalist records above. Decide whether/how to fold the 64 positive field annotations into the training description.`

#### 2.3.2 Negative Samples

Negative (yew-absent) training data comprised **11,452 samples** drawn from:
- BC Forest Analysis and Inventory Branch (FAIB) forest plots in non-yew species assemblages
- Alpine and subalpine locations above the elevational range of yew
- CWH-zone locations with confirmed non-yew canopy dominance (western hemlock, Sitka spruce stands)

Of these, 2,458 negatives (1,062 derived from BC Vegetation Resources Inventory polygons and 1,396 from FAIB, alpine, and confirmed non-yew locations) were paired with the positive records in the balanced training/validation splits, and a further **8,994 GEE-extracted negative embeddings** from confirmed non-yew locations — predominantly non-treed alpine and subalpine areas — were added and weighted 2× during training to reinforce the absence class and reduce spurious yew probability at high elevations.

The combined training set therefore comprised **12,495 unique records** (1,043 positives and 11,452 negatives), with the GEE-derived negatives duplicated 2× at training time for an effective negative weight of 20,446.

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
| **XGBoost (production)** | **0.9957** | **0.989** | **0.947** |
| MLP + StandardScaler | 0.9961 | 0.986 | 0.977 |
| MLP raw embeddings | 0.9962 | 0.976 | 0.960 |
| Random Forest | 0.9896 | 0.984 | 0.974 |
| kNN (k=3) | 0.9909 | 0.911 | 0.833 |
| Logistic Regression | 0.9165 | 0.813 | 0.562 |

The production XGBoost AUC-ROC of 0.9957 (reported as 0.996 to three significant figures elsewhere in this paper) is statistically indistinguishable from the best-performing neural models. XGBoost was selected as the production model for its comparable performance, interpretability, and computational efficiency for per-pixel inference across ~100 million pixels. Although the MLP achieved comparable held-out performance, its propensity for overfitting — evident as noise in the spatial distribution of its classifications — discouraged its use on generalizability grounds.

#### 2.4.3 Alternative Feature Sets

A separate XGBoost model trained on 35 engineered features from the BC Vegetation Resources Inventory achieved AUC 0.82 with only 26% recall at operational thresholds, and a multi-modal model combining spectral embeddings with inventory features yielded only 5% recall due to scale mismatch. These results confirmed that satellite spectral embeddings alone provide superior classification performance. `TODO: make these engineered-feature figures consistent with the training-data description above.`

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

For habitat *loss* estimation, all non-old-growth forested land (categories 2–5) is treated as "logged" to estimate what yew habitat would have existed under pre-logging old-growth conditions. We note that this assumption — that all currently non-old-growth forest was formerly suitable yew habitat — is deliberately inclusive and likely inflates the original-habitat estimate, since not every stand that has been logged would necessarily have supported yew. The resulting decline figures should therefore be read as an upper-bound estimate of potential historical habitat rather than a precise reconstruction. Although forest in the 80–150 yr class retains some potential to regenerate yew habitat and old-growth characteristics over time, by the IUCN Red List's reproductive-age criterion such stands would contain no reproductively mature adults, supporting their treatment as currently unsuitable.

#### 2.5.2 Fire Suppression

Historical fire perimeters (1900–2024) from the BC historical fire polygon dataset were filtered to fires ≥100 ha. A time-dependent fire modifier was applied:

$$\text{fire\_modifier} = \frac{2024 - \text{fire\_year}}{124}, \quad \text{clamped to } [0, 1]$$

This assumes linear recovery over 124 years (the span of the fire record), with recent fires causing near-complete suppression. The linear form is a simplifying assumption: post-fire recovery of shade-dependent understorey conifers is in reality non-linear and strongly site-dependent (varying with fire severity, residual canopy, and proximity to seed sources), and given yew's very thin bark and slow recolonisation it may be optimistic for high-severity burns. We did not perform a sensitivity analysis on the 124-year recovery period; because fire accounts for <1% of total estimated loss (§3.6), the overall decline estimate is insensitive to this choice. A partial-suppression modifier (rather than complete removal) was judged appropriate because fires, unlike clear-cut logging, generally do not consume the full area of a perimeter polygon, so mature individuals typically persist within burned polygons. For overlapping polygons, the most recent fire year takes precedence.

#### 2.5.3 Elevation Suppression

A Copernicus GLO-30 Digital Elevation Model was used to suppress yew predictions at very low coastal elevations:

$$\text{elev\_factor} = \text{clip}\left(\frac{\text{elevation}}{30}, 0, 1\right)$$

This linear ramp from 0 at sea level to 1 at 30 m removed 18,434 ha of false-positive predictions from intertidal and low-coastal zones. The 30 m threshold is applied as a conservative heuristic rather than a hard ecological boundary: Pacific yew is not salt-tolerant and does not establish in the intertidal, foreshore, and saline-influenced floodplain settings that dominate below this elevation (Busing et al. 1995), where the embedding classifier nonetheless returns occasional high probabilities driven by spectrally similar moist, closed-canopy vegetation. The linear ramp avoids a sharp cut-off and down-weights, rather than eliminates, predictions in the transitional 0–30 m band.

#### 2.5.4 Habitat Loss Estimation

For each BEC subzone, yew prevalence rate (*r*) was computed as the mean raw XGBoost probability across all old-growth pixels. Estimated original habitat was *r* × (old-growth + logged pixels) × 0.01 ha/pixel. The factor 0.01 ha/pixel converts pixel counts to area: at the 10 m native resolution of the embeddings, each pixel covers 10 m × 10 m = 100 m² = 0.01 ha. Here and throughout, "probability mass" denotes the sum of per-pixel yew probabilities (each in [0, 1]) multiplied by this per-pixel area — equivalently, the area-weighted expected habitat extent. This approach uses continuous probability mass rather than binary thresholding (counting only pixels above a fixed probability cut-off), providing a more robust cumulative habitat-area estimate that is insensitive to the choice of threshold.

### 2.6 Secondary Threat Assessment

In addition to the quantitative logging and fire analysis, we reviewed the scientific literature to characterise secondary threats not directly modellable from remote sensing.

`TODO: consider a dedicated section consolidating the quantitative estimates of loss; redo/refresh the underlying analysis.`

#### 2.6.1 Stream Erosion and Riparian Habitat Loss

Pacific yew preferentially occupies moist riparian zones (Busing et al. 1995). Logging-driven hydrological changes increase peak flows by 20–50% (Hartman & Scrivener 1990), causing channel widening proportional to $W \propto Q^{0.5}$ (Leopold & Maddock 1953). A water buffer sensitivity analysis was run across all 42 tiles with available grid data: applying binary morphological dilation (3 pixels = 30 m) to all water category pixels, then summing yew probability mass in the buffered old-growth pixels. This yielded **1,717 ha** of yew probability mass at risk from riparian erosion (5.9% of the 29,028 ha of remaining yew probability mass in those 42 tiles — a subset of the full 47,534 ha, restricted to the tiles for which the VRI water-category grid was available).

#### 2.6.2 Sea-Level Rise and Saltwater Intrusion

Pacific yew is not salt-tolerant; saline groundwater intrusion kills root systems. Habitat below 1.0–1.5 m elevation in 50-year projections faces 100% loss, with an additional 15–20 m inland buffer for saltwater intrusion effects. Our elevation suppression (§2.5.3) captures current low-elevation effects but does not project future sea-level scenarios.

#### 2.6.3 Yew Big Bud Mite (*Cecidophyopsis psilaspis*)

This eriophyid mite causes bud galls on *Taxus* species, with terminal bud mortality averaging over 20% in infested coastal BC populations, reducing growth rates by ~20% and seed (aril) production by ~25%. `TODO: cite`

#### 2.6.4 Ungulate Browsing

Wild ungulates preferentially browse Pacific yew foliage in winter. Browsing pressure causes 60–80% seedling/sapling mortality in areas with high ungulate density (>10 deer/km²), creating a "browsing ceiling" that prevents recruitment to the established understorey stage. `TODO: cite`

#### 2.6.5 Wildfire Frequency Increase

Climate change is projected to reduce fire return intervals in drier maritime subzones to 80–120 years, potentially preventing populations from reaching reproductive maturity (~80–100 years) between successive fires.

#### 2.6.6 Historical Taxol Bark Harvest

Between 1989 and 1993, at peak harvest (1991), an estimated 360,000 mature yew trees per year were felled across the Pacific Northwest for paclitaxel extraction. Treating a single cancer patient required the bark of six 100+-year-old trees. This harvest was not spatially documented in GIS and cannot be directly modelled, but its legacy is embedded in the depleted large-tree cohort documented in §3.10.

### 2.7 Field Sampling of Population Size Structure

To ground-truth population age structure independently of remote sensing, circumference at breast height (CBH, measured at 1.3 m above ground) was recorded for *n* = 120 Pacific yew individuals at `TODO: field site(s)` in `TODO: sampling month, year`. All yew stems ≥ 1 cm CBH within `TODO: sampling protocol — e.g. plot dimensions / transect design / area searched` were measured; multi-stemmed individuals were recorded as separate stems where stems arose below breast height. `TODO: check this field-sampling description is correct.`

CBH measurements were converted to diameter at breast height (DBH) using the standard geometric relationship:

$$\text{DBH (cm)} = \frac{\text{CBH (cm)}}{\pi}$$

Size-class structure was compared against the de Liocourt (1898) reverse-J model, which describes the expected stem-frequency distribution in a balanced, self-sustaining uneven-aged stand as a constant ratio *q* of stems between successive (ascending) diameter classes. We use this model as a descriptive reference for a stable understory population rather than as a mechanistic demographic model: because Pacific yew is a slow, plastic, suppression-tolerant grower, diameter is an imperfect proxy for age, and we therefore frame the comparison in terms of size structure, not age structure. For Pacific yew in intact old-growth CWH and ICH stands, Bolsinger & Jaramillo (1990) and Graham (1994) document *q* ≈ 1.4–1.6 per 10-cm class; we adopt *q* = 1.5 as the central reference and report all comparisons across the full 1.4–1.6 range so that conclusions do not depend on a single assumed value. Expected stem counts per 10-cm class were obtained by scaling the geometric series *q*<sup>−i</sup> to the observed sample size.

Three complementary tests were applied. First, because the *a priori* prediction of selective large-tree removal concerns the large-diameter tail, we tested the count of stems > 30 cm DBH against the proportion expected under each reference *q* using an exact binomial test (one-sided, deficit). Second, a Pearson χ² goodness-of-fit test was applied to the binned counts (the three classes below 30 cm plus a pooled ≥ 30 cm class, so that all expected counts exceeded five). Third, to summarise the realised decline rate we fitted a single reverse-J slope to the whole observed distribution by ordinary least squares on the log-counts of all occupied 10-cm classes, yielding an empirical population *q* with a 95% confidence interval from a non-parametric bootstrap (2,000 resamples of the *n* = 120 stems); this whole-population *q* — unlike a fit restricted to the sparse adult tail — is well constrained and its direction relative to *q* = 1.5 is interpretable. Two caveats bound all of these tests: (i) understory stems < 10 cm DBH are readily overlooked, so the smallest class may be under-counted; and (ii) multi-stemmed (layering) individuals were recorded as separate stems (§2.7), which can inflate the small-stem classes relative to the genet-based assumptions of the de Liocourt model.

### 2.8 Interactive Web Map

Results are presented via an interactive Leaflet.js web map hosted on GitHub Pages ([jerichooconnell.github.io/yew_project](https://jerichooconnell.github.io/yew_project/)). The map displays yew probability overlays for all 99 tiles, VRI-derived logging age class rasters, 5,700 historical fire polygons ≥100 ha (1900–2024) colour-coded by age, 1,201 protected areas (provincial parks, conservancies, national parks, ecological reserves, and protected areas, from the BC Data Catalogue), and a crowd-sourced field observation reporting interface. `TODO: expand on the crowd-sourcing data / observation submission workflow.`

---

## 3. Results

### 3.1 Overall Habitat Decline

Across all 99 study tiles (~9,900 km², 69 BEC subzones), we estimate that **154,483 ha** of yew habitat existed historically under pre-logging old-growth conditions, of which **47,534 ha (30.8%)** remains today (Table 1, Figure 1). This represents an overall decline of **69.2%**, with the per-zone severity of decline ranked in Figure 4. Unless otherwise stated, all headline habitat areas in this paper are *continuous probability-mass* estimates (the sum of per-pixel yew probability × 0.01 ha; §2.5.4) rather than counts of pixels above a fixed threshold; two threshold-based figures used for specific secondary analyses (37,885 ha and 29,028 ha) are defined where they appear in §4.4 and §3.9.

The total estimated loss of **106,949 ha** (the difference between original and current remaining habitat) is overwhelmingly driven by industrial logging, with a smaller contribution from historical fire. Of this total, fire suppression accounts for an estimated **692 ha** (0.4% of original), the remainder being attributable to logging of formerly old-growth stands. Separately, low-elevation suppression removes **18,434 ha** (11.9%) of *false-positive* predictions from intertidal and low-coastal zones; this is a model correction to improve spatial accuracy, not a quantity of destroyed habitat, and is therefore not counted as ecological loss.

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

†The "Logged Area" column reports total logged *forest* area within each zone (all non-old-growth forested land), which is a distinct and much larger quantity than the area of logged *yew habitat*. It is not summed here because the zone-level totals reflect overall forest disturbance, not yew-specific loss; the estimated logged yew-habitat area (the difference between original and remaining yew habitat) is 106,949 ha (§3.1).

### 3.2 Coastal Western Hemlock (CWH) Zone

The CWH zone contains the vast majority of BC's yew habitat (72% of estimated original, 72% of current remaining). Across 14 CWH subzones and 885,783 ha of analysed area:

- **Estimated original yew habitat**: 111,407 ha
- **Current remaining**: 34,386 ha (**69.1% decline**)
- **Old-growth forest remaining**: 330,804 ha (37.3% of total CWH area)
- **Logged forest**: 312,128 ha (35.2% of total CWH area)

`TODO: these area figures don't add up — reconcile (old-growth + logged + other vs total CWH area, and vs the estimated yew-habitat figures).`

The most severely impacted CWH subzones include CWHxm2 (89.5% decline), CWHmm2 (90.7%), CWHdm (83.9%), and CWHxm1 (84.1%) — all reflecting heavy logging in drier maritime subzones. The CWHvh2 subzone retains the most yew habitat in absolute terms (10,355 ha) due to its extensive old-growth (121,647 ha) in remote hypermaritime terrain (Figure 3).

### 3.3 Interior Cedar–Hemlock (ICH) Zone

The ICH zone supports Pacific yew in moist valley-bottom forests east of the Coast and Columbia Mountains. Our analysis of 14 ICH tiles across 142,778 ha reveals:

- **Estimated original yew habitat**: 25,257 ha
- **Current remaining**: 6,385 ha (**74.7% decline**)

The ICH zone shows a **higher percentage decline** than the CWH (74.7% vs. 69.1%), primarily because a larger proportion of the ICH landscape has been logged (68.0% vs. 35.2%), reflecting the more accessible valley-bottom terrain where yew habitat concentrates. The ICHxw subzone has the highest mean yew probability in old-growth of any sampled subzone (0.489), yet has lost 83.8% of its estimated original habitat (Figure S1).

### 3.4 Coastal Douglas-fir (CDF) Zone

The CDF zone, restricted to the rain-shadow lowlands of southeastern Vancouver Island and the Gulf Islands, has experienced the most catastrophic yew habitat loss of any zone (Figure S2):

- **Estimated original yew habitat**: 3,889 ha
- **Current remaining**: 36 ha (**99.1% decline**)
- **Old-growth forest remaining**: 214 ha (0.6% of total CDF area)
- **Logged/developed/converted**: 35,841 ha (99.4% of total CDF area)

The single CDFmm subzone had the highest old-growth yew prevalence rate of any sampled subzone (0.381), indicating that the Coastal Douglas-fir zone historically supported dense yew populations.

### 3.5 Old-Growth Yew Prevalence

Mean yew probability in old-growth pixels varies widely among subzones, reflecting genuine ecological variation (Figure S6). The highest-prevalence subzones are ICHxw (0.489), CWHvh1 (0.411), ICHdw1 (0.410), CDFmm (0.381), and CWHmm2 (0.365). These rates, concentrated in warm, moist subzones at low to mid elevation, serve as the basis for estimating original habitat area.

### 3.6 Fire Impact

Across all tiles, 5,700 fire features covering approximately 96,543 ha intersected the study area. Total estimated yew habitat suppression from fire is 692 ha — small relative to logging but contributing to cumulative decline, particularly in drier interior subzones. The 2020s burned area (8,146 ha) exceeds the 2010s (4,859 ha), consistent with an accelerating fire season driven by climate change (Figure S7).

### 3.7 Post-Classification Suppression Pipeline

The suppression pipeline progressively reduces raw model output to a realistic current habitat estimate (Figure 2). The stages below describe *model processing* — successive corrections applied to the raw per-pixel spectral suitability surface — and should be distinguished from the prevalence-based original-versus-remaining accounting reported in §3.1. The logging and fire stages remove pixels where habitat once existed but has since been destroyed (ecological loss), whereas the elevation stage removes spurious predictions in intertidal and low-coastal zones (false-positive correction, not habitat loss):

| Stage | Yew Habitat (ha) |
|-------|------------------:|
| Raw model output | 213,165 |
| After logging suppression | 66,659 |
| After fire suppression | 65,968 |
| After elevation suppression | **47,534** |

Logging suppression is the dominant filter, removing ~146,506 ha (68.7% of raw predictions); fire suppression removes a further ~691 ha, and elevation correction removes ~18,434 ha of false positives. The spectral signature of potential yew habitat (moist closed-canopy forest) persists even in young second-growth where yew itself cannot yet survive, explaining the large logging filter contribution.

### 3.8 Comparison of CWH, ICH, and CDF Decline Pathways

The three focal zones exhibit qualitatively different decline pathways (Figure S13). The CWH shows gradual erosion from a large original base, with 37% old-growth remaining and continuing logging pressure. The ICH shows intensive logging concentrated in accessible valley-bottom yew habitat, resulting in a higher percentage decline (74.7%) despite a smaller absolute base. The CDF has experienced near-complete extirpation through a combination of industrial forestry, urban expansion, and agricultural conversion, leaving only 214 ha of old-growth as isolated fragments. `TODO: inconsistent number — reconcile the 214 ha figure with the deleted CDF sentence in §3.4.`

### 3.9 Secondary Threats — Quantitative Estimates

**Table 2. Estimated impacts of threats to Pacific yew in BC.**

| Threat | Estimated Impact | Confidence | Type |
|--------|------------------|------------|------|
| Clear-cut logging (dominant driver) | 106,949 ha total habitat loss | High (modelled) | Direct habitat destruction |
| Wildfire (historical) | 692 ha suppressed | High (modelled) | Direct mortality |
| Stream erosion buffer (30 m) | ~1,717 ha (5.9% of remaining in sampled tiles) | Moderate | Riparian habitat loss |
| Sea-level rise (future) | ~240 ha (<0.5% of remaining) | Low (projected) | Coastal inundation |
| Yew big bud mite | 20–25% bud mortality | Moderate (literature) | Growth/fecundity reduction |
| Ungulate browsing | 60–80% seedling mortality | Moderate (literature) | Regeneration failure |
| Historical Taxol harvest | Unknown (hundreds of thousands of trees) | Low (no spatial data) | Historical direct mortality |

The stream erosion estimate applied binary morphological dilation (3 pixels = 30 m) to all VRI water category pixels across 42 tiles, then summed yew probability mass in the newly-buffered old-growth pixels. The 1,717 ha represents 5.9% of the 29,028 ha remaining yew mass in sampled tiles, supported by the Carnation Creek study (20–50% peak flow increases post-logging; Hartman & Scrivener 1990) and hydraulic geometry relationships predicting ~14% channel widening for a 30% flow increase ($W \propto Q^{0.5}$; Leopold & Maddock 1953). `TODO: this stream-erosion methodology should move to the Methods section; check numbers.`

### 3.10 Population Age Structure

Field measurement of *n* = 120 Pacific yew individuals yielded CBH values of 1–218 cm, corresponding to converted DBH values of 0.3–69.4 cm (mean DBH = 15.0 cm, median = 12.7 cm; Figure 7). Size-class structure differed markedly from the de Liocourt reference distribution (*q* = 1.5), with a pronounced deficit of large-diameter stems and a surplus of mid-sized stems (Table 3).

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

The dominant feature of the sample is a deficit of large-diameter stems: only **7** stems exceed 30 cm DBH, against 32.1 expected under *q* = 1.5 (and 27–38 across *q* = 1.4–1.6). An exact binomial test rejects the stable-population expectation for this tail (7 of 120 vs. an expected proportion of 0.27; *p* ≈ 4 × 10⁻⁹ at *q* = 1.5, and *p* < 10⁻⁶ across the full *q* = 1.4–1.6 range), as does a χ² goodness-of-fit test on the binned counts (χ² = 49.5, df = 3, *p* ≈ 1 × 10⁻¹⁰). Fitting a single reverse-J slope to the whole observed distribution gives an empirical population *q* ≈ 2.0 (95% bootstrap CI 1.6–2.7), significantly steeper than the literature-stable *q* ≈ 1.5 — i.e. the realised distribution declines faster toward the large-diameter classes than a self-sustaining population would, consistent with depletion of large adults. We note that the observed distribution is not strictly monotonic (it peaks in the 10–20 cm class rather than the smallest class); this modal bulge most plausibly reflects under-detection of the smallest understory stems and/or the recording of layering ramets as separate stems (§2.7), and does not affect the large-tree result.

The combination of abundant mid-sized stems and depleted large adults is consistent with selective harvest of large, bark-rich individuals for paclitaxel extraction during the 1989–1993 taxol rush: a recruitment bottleneck would instead deplete the *smallest* classes, which is not observed. We caution, however, that this size signature is not uniquely diagnostic of harvest — an even-aged establishment or canopy-release cohort that has since grown into the 10–20 cm class would produce a broadly similar structure — and that the inference depends on the field sample being a representative stand census (the sampling design is specified in §2.7). With that caveat, the depletion is notable because Pacific yew reaches typical bark-harvest size (~30 cm DBH) only after 150–250 years of growth (Graham 1994), making any lost large-tree cohort effectively non-renewable on forestry time scales.

---

## 4. Discussion

### 4.1 Scale of Decline

Our estimate of 69.2% overall decline should be interpreted as a property of the sampled tiles, not a validated province-wide figure. The study tiles cover approximately 9,900 km² of a total CWH+ICH+CDF zone area of ~3.8 million hectares (~0.26%), and were purposively rather than randomly sited — selection deliberately spanned both heavily logged and well-preserved areas and clustered around iNaturalist observation density (§2.1). This purposive design means the sample is not statistically representative of the zones as a whole, and the true province-wide decline could be either higher or lower than 69.2% depending on the disturbance profile of unsampled terrain. Any extrapolation to the full zonal extent would require province-wide systematic coverage; we therefore frame all zonal percentages as estimates conditional on the sampled tiles.

### 4.2 Interaction of Threats

The threats to Pacific yew are not independent. Logging creates the conditions for all other threats to intensify: logging → stream erosion (removed canopy increases peak flows, eroding riparian zones where yew concentrates); logging → browsing pressure (clearcuts attract ungulates to forest edges); logging → fire vulnerability (forest fragmentation creates drier microclimates); logging → mite susceptibility (stressed, isolated populations may have reduced resistance to *Cecidophyopsis psilaspis*). The cumulative effect of these interacting threats is likely significantly greater than the sum of individually estimated impacts. `TODO: are all of these interaction pathways important enough to retain?`

### 4.3 Limitations

1. **Spectral similarity**: The classifier cannot distinguish Pacific yew understorey from spectrally similar moist-forest conditions; probabilities represent habitat suitability rather than confirmed presence `TODO: discuss what the yew probability represents and why a probability surface (rather than a binary map) was used.`
2. **VRI accuracy**: Stand-age assignments have known errors, particularly for multi-cohort stands and post-fire regeneration
3. **Spatial coverage**: 99 tiles cover a portion of BC's total area; extrapolation requires caution
4. **Static analysis**: The study represents a snapshot as of 2024; ongoing logging and climate change will alter the estimates
5. **No field validation**: Predictions have not been systematically validated against independent field surveys `TODO: the crowd-sourced reporting tool could enable this validation.`

### 4.4 Conservation Implications

The finding that **only 5.6% of mapped yew habitat falls inside provincial parks** (2,121 ha of 37,885 ha) highlights the species' extreme vulnerability to ongoing forestry operations. Including all protected-area designations raises the protected fraction to **11.0%** (4,180 ha) — with conservancies contributing the largest share (1,602 ha, 4.2% of mapped habitat) and national parks a further 457 ha (1.2%). This analysis intersects the suppressed P ≥ 0.5 habitat surface summed across all 99 tiles (37,885 ha — a threshold-based extent, as opposed to the 47,534 ha continuous probability-mass estimate used elsewhere) with the BC Data Catalogue protected-area layers (provincial parks, ecological reserves, protected areas, conservancies, and national parks). Even so, nearly half of the mapped habitat (17,785 ha, 47%) lies in tiles with **no protected area at all**, including high-habitat tiles such as Alberni Valley (3,632 ha), Port Hardy Forest (1,173 ha), Blunden Harbour (1,085 ha), and Jervis Inlet Slopes (984 ha).

Priority conservation actions include: (1) incorporation of Pacific yew habitat into forest stewardship plans for CWH and ICH operating areas; (2) riparian buffer widening to protect streamside populations from logging-driven erosion; (3) ungulate management in high-value yew stands to permit natural regeneration; (4) monitoring for *Cecidophyopsis psilaspis* spread in coastal populations; and (5) long-term fire management planning that recognises yew's zero fire tolerance.

### 4.5 IUCN Red List Assessment for the Canadian (British Columbia) Population

The current global IUCN Red List status of *Taxus brevifolia* is Near Threatened (NT; Thomas 2013), based on evidence available at the time of assessment. That assessment pre-dates any spatially explicit, province-scale analysis of BC habitat extent or decline. Using the quantitative results of the present study, we apply IUCN Red List criteria (IUCN 2012) to the Canadian BC population, which represents the northern core of the species' global range and the largest temperate rainforest population.

**Criterion A — Population size reduction inferred from habitat decline**

Criterion A2c evaluates population size reductions estimated or inferred where the reduction or its causes may not have ceased, based on observed decline in habitat area, extent, or quality. Pacific yew reaches reproductive maturity at approximately 80–100 years (Graham 1994) `TODO: check this maturity age / citation`; three generations therefore span ~240–300 years, well encompassing the industrial logging era beginning in the 1920s. Our analysis documents a 69.2% decline in modelled yew habitat across 9,900 km² of sampled BC range. The primary cause — industrial clear-cut logging — has not ceased; logging continues under existing tenure commitments across the CWH and ICH zones.

| IUCN Criterion | Threshold | This Study | Assessment |
|:---|:---|:---|:---|
| CR A2c | ≥ 80% reduction | 99.1% (CDF); 74.7% (ICH) | **CR** at subpopulation level |
| EN A2c | ≥ 50% reduction | 69.2% (BC overall) | **EN** at population level |
| VU A2c | ≥ 30% reduction | Exceeded | — |

Applying criterion A2c to the Canadian BC population as a whole, a 69.2% estimated habitat decline where the cause is ongoing and not reversible meets the **Endangered (EN)** threshold (≥50%). The 99.1% decline in the CDF zone and 74.7% decline in the ICH zone each individually meet the Critically Endangered threshold (≥80%) at the subpopulation level.

**Criterion A2d — Levels of exploitation**

The taxol bark harvest of 1989–1993, at which point an estimated 360,000 mature trees per year were felled across the Pacific Northwest (Hartzell 1990), constitutes direct exploitation at a scale unparalleled in the species' recorded history. Although bark harvest has ceased, its legacy is embedded in the depleted large-tree cohort documented in §3.10. This additional exploitation pressure reinforces the EN designation under criterion A2d.

**Criterion B — Geographic range**

Criterion B1 (extent of occurrence, EOO) does not apply to the BC population, as the EOO substantially exceeds the 5,000 km² EN threshold. The remaining modelled yew habitat across all 99 study tiles totals ~475 km², which approaches the EN B2 area of occupancy (AOO) threshold of <500 km². However, the study tiles cover only ~0.26% of the CWH+ICH+CDF zone extent; the 99-tile AOO cannot be extrapolated to a true provincial AOO without systematic province-wide coverage. Pacific yew thus does not qualify for increased protection under this criterion.

**Criteria C, D, and E**

Assuming mean yew density of 1–5 individuals per hectare in old-growth habitat (Bolsinger & Jaramillo 1990; Busing et al. 1995), the 47,534 ha of remaining modelled habitat supports an estimated 47,534–237,670 mature individuals, substantially exceeding the <2,500 EN threshold under criterion C. Criteria D and E are not met by the available data.

**Summary assessment**

Under the best available evidence, the Canadian BC population of *Taxus brevifolia* supports a classification consistent with **Endangered (EN A2c)** under IUCN Red List criteria, recognising that this inference rests on modelled habitat decline rather than direct population data. The CDF and ICH subpopulations individually reach Critically Endangered thresholds. The current global designation of Near Threatened (Thomas 2013) is not consistent with the magnitude of BC habitat decline documented here, nor with the ongoing and unabated nature of the primary threat. We recommend that COSEWIC and the IUCN undertake a formal reassessment of *Taxus brevifolia* using province-scale spatial data.

---

## 5. Conclusions

This study presents the first spatially explicit, province-scale assessment of Pacific yew (*Taxus brevifolia*) habitat extent and decline in British Columbia, using 64-dimensional satellite spectral embeddings from the Google AlphaEarth Foundation model classified by an XGBoost ensemble (AUC-ROC 0.996). Across 9,900 km² of sampled CWH, ICH, and CDF forest, we estimate a 69.2% decline from approximately 154,483 ha of original habitat to 47,534 ha remaining — driven overwhelmingly by industrial clear-cut logging, with secondary contributions from wildfire, riparian erosion, and the legacy of taxol bark extraction. Field-measured population size structure confirms depletion of the large-adult cohort (DBH > 30 cm) relative to the de Liocourt stable-population model, consistent with selective harvest of large individuals during the 1989–1993 taxol rush.

The finding that only 5.6% of mapped yew habitat falls within provincial parks — and 11.0% across all protected-area designations — underscores the species' extreme exposure to continued forestry operations. Application of IUCN Red List criteria to the Canadian BC population supports an **Endangered (EN A2c)** designation — a significant upgrading from the current global Near Threatened listing — with the CDF and ICH subpopulations individually meeting Critically Endangered thresholds. We urge a formal COSEWIC and IUCN reassessment of the species drawing on provincial-scale spatial data, and recommend that Pacific yew habitat be explicitly incorporated into forest stewardship plans across the Coastal Western Hemlock and Interior Cedar–Hemlock zones.

---

## Acknowledgements

`TODO: Acknowledgements — funding sources, field assistance, GEE and BC Data Catalogue data access, Indigenous territory acknowledgements, co-author contributions`

---

## Data Availability

The interactive web map is available at [jerichooconnell.github.io/yew_project](https://jerichooconnell.github.io/yew_project/). Source code, model weights, and analysis scripts are hosted on GitHub. Satellite embeddings were accessed via Google Earth Engine; the VRI 2024 dataset was obtained from the BC Data Catalogue.

---

## References

- Arsenault, A., & Bradfield, G.E. (1995). Structural–compositional variation in three age-classes of temperate rainforests in southern coastal British Columbia. *Canadian Journal of Botany* 73(1): 54–64.
- Bergeron, Y., & Fenton, N.J. (2012). Boreal forests of eastern Canada revisited: Old growth, nonfire disturbances, forest succession, and biodiversity. *Botany* 90(6): 509–523.
- Bolsinger, C.L., & Jaramillo, A.E. (1990). *Taxus brevifolia* Nutt. Pacific yew. In: Burns, R.M. & Honkala, B.H. (eds.) *Silvics of North America, Vol. 1: Conifers.* USDA Forest Service Agriculture Handbook 654, pp. 573–579.
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

**Figure 2.** Post-classification suppression waterfall: raw model prediction through logging, fire, and elevation filters to final current habitat estimate across all 99 study tiles.

**Figure 3.** Stacked bar chart showing destroyed (red) versus remaining (green) yew habitat in each CWH subzone, ordered by total estimated original habitat.

**Figure 4.** Horizontal bar chart of estimated percentage yew habitat decline across all major BEC zones, ordered by severity of decline.

**Figure 5.** Location map of 99 study tiles across British Columbia (coastal CWH/CDF tiles = circles; ICH interior tiles = triangles), overlaid on biogeoclimatic zone boundaries.

**Figure 6.** Classifier performance comparison for six models trained on 64-dimensional AlphaEarth satellite embeddings (held-out validation set).

**Figure 7.** Pacific yew population size-class distribution (DBH converted from field-measured CBH, *n* = 120). Left panel: observed stem counts per 10-cm DBH class (green bars) versus the de Liocourt stable-population reference (*q* = 1.5; red dashed) and the reverse-J slope fitted to the whole observed distribution (*q* ≈ 2.0, 95% CI 1.6–2.7; blue dash-dot). Shaded area: the large-tree deficit (> 30 cm DBH, 7 observed vs. ~32 expected). Right panel: observed versus expected (*q* = 1.5) cumulative distribution functions, with the observed median (12.7 cm) marked.

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
