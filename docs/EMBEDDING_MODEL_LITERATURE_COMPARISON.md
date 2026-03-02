# YewMLP vs Literature: Modelling with AlphaEarth Satellite Embeddings

## Executive Summary

Our YewMLP approach (4-layer MLP with BatchNorm on 64D satellite embeddings) is
**non-standard relative to published methods** for these embeddings. The AlphaEarth
Foundations paper (Brown et al., 2025) and Google's official documentation
exclusively evaluate downstream tasks using **kNN and linear probes** — deliberately
minimal-parameter transfer methods.

**Empirical validation (Section 11)** shows that our MLP is **justified**: linear
probes perform poorly (AUC=0.916) because the yew detection boundary is genuinely
nonlinear. MLP (AUC=0.996) and Random Forest (AUC=0.990) both substantially
outperform the paper's recommended kNN/linear methods. StandardScaler on unit-length
embeddings is theoretically suboptimal but empirically harmless — MLP+scaler
achieves perfect recall.

**For production mapping, Random Forest on raw embeddings is recommended** as
the best balance of accuracy (F1=0.974), interpretability, and alignment with the
literature's preference for operating on unscaled embeddings. All 35 CWH spot tile
maps have been regenerated with RF.

---

## 1. Our Current Approach

### Architecture
- **YewMLP**: 64 → 128 → 64 → 32 → 1 (sigmoid output)
- BatchNorm + ReLU + Dropout(0.2) per hidden layer
- BCEWithLogitsLoss, Adam (lr=0.001, weight_decay=1e-4)
- CosineAnnealingLR, batch size 512, 100 epochs, early stopping (patience=6)

### Preprocessing
- `sklearn.StandardScaler` (zero mean, unit variance) on all 64 embedding bands
- `np.nan_to_num()` to replace NaN/Inf values

### Training Data
- ~1,043 positive samples (iNaturalist observations × 1 + manual annotations × 3 weight)
- ~4,458 negative samples (iNaturalist background + VRI logged areas × 2 weight)
- 4 iterative training rounds with progressively refined negatives

### Performance
- Accuracy: 98.85%, F1: 0.9471, AUC: 0.9980
- Applied at P ≥ 0.95 threshold for population mapping

---

## 2. AlphaEarth Foundations Paper (Brown et al., 2025)

### Embedding Design
The 64D embeddings are **unit-length vectors on the 63-sphere (S⁶³)**, produced by
a ~480M parameter model trained on 10.1M video sequences from 9 gridded data
sources (Sentinel-1 SAR, Sentinel-2, Landsat 8/9, GEDI canopy height, GLO-30 DEM,
ERA5-Land climate, GRACE gravity, NLCD land cover) plus geotagged text from
Wikipedia and **GBIF species occurrence records** (including Plantae, Animalia,
Fungi). The embeddings are quantized from 32-bit to 8-bit with negligible
performance loss.

**Crucially, GBIF species occurrence data — including iNaturalist records — was
used as a training target for the embedding model itself.** This means species
distribution information is already encoded in the embedding space, which partly
explains why simple classifiers work well for species-level tasks.

### Transfer Methods Evaluated
The paper tests **only two** downstream classifier types:

| Method | Description | Parameters |
|--------|-------------|------------|
| **kNN** (k=1, k=3) | k-nearest neighbours using **L2 distance** | 0 (non-parametric) |
| **Linear probe** | RidgeClassifier (one-vs-rest) or least-squares regression, λ=0 | 64 × n_classes |

These were chosen deliberately as "minimal parameter" methods to avoid unduly
penalizing any representation due to non-optimal hyperparameters.

**No neural networks, MLPs, Random Forests, SVMs, or gradient-boosted trees were
tested.** The entire point of the evaluation was to show the embeddings are so
information-rich that even the simplest classifiers achieve strong results.

### Most Relevant Evaluation: US Trees (Genus-Level)
This evaluation is the closest analogue to our yew detection task:

- **Labels:** 39 tree genera from GBIF/iNaturalist research-grade observations
- **Training:** 300 samples per class (11,700 total)
- **Test:** 33,682 samples
- **Valid period:** Single-date (instantaneous, based on observation date)
- **Location:** United States
- **Result:** AEF outperformed all baselines; SatCLIP was next-best
- **Best transfer method:** Linear probe (most evaluations) and kNN k=1 (some)

### Key Performance Finding
AEF reduced error magnitudes by ~23.9% on average vs the next-best approach across
all 15 evaluations. For tree species classification specifically, the paper notes
that AEF "shows previously unachievable performance in low-shot regimes with
simple classifiers, unlocking use cases that may have previously been untenable
given sparse observational records and/or highly detailed taxonomies."

---

## 3. Comparison: What Others Do vs What We Do

### 3.1 Classifier Choice

| Aspect | Published Methods | Our Approach |
|--------|-------------------|--------------|
| **Classifier** | kNN (k=1,3), linear probe | 4-layer MLP (128→64→32→1) |
| **Parameters** | 0 (kNN) or 64×C (linear) | ~13,000+ learnable params |
| **Regularization** | None needed (kNN), λ=0 (linear) | Dropout 0.2, BatchNorm, weight decay |
| **Training** | Fit once (no epochs) | 100 epochs with early stopping |
| **Distance metric** | L2 Euclidean (kNN) | Learned nonlinear mapping |

**Assessment:** Our MLP is vastly overparameterized relative to what published
methods require. The paper demonstrates that the embeddings are designed to be
linearly separable for downstream tasks — additional nonlinear capacity may help
marginally at best, and risks overfitting at worst.

### 3.2 Preprocessing

| Aspect | Published Methods | Our Approach |
|--------|-------------------|--------------|
| **Scaling** | None (raw embeddings) | StandardScaler (zero mean, unit variance) |
| **NaN handling** | Not discussed (clean data assumed) | `nan_to_num()` |

**Assessment:** This is a **significant divergence**. The embeddings are unit-length
vectors on S⁶³ — their geometric structure (angular distances, dot-product
similarities) is meaningful. `StandardScaler`:
1. Shifts the centroid off the sphere
2. Distorts relative distances between dimensions
3. Breaks the unit-length constraint
4. May reduce the effectiveness of angular/cosine relationships

The kNN in the paper uses L2 distance on the **raw** embeddings. Since the
embeddings are unit-length, L2 distance is monotonically related to cosine
distance: ‖a-b‖² = 2(1 - a·b). Scaling destroys this relationship.

For the MLP, the BatchNorm layers may partially compensate for the StandardScaler
distortion (BatchNorm effectively re-normalizes each layer's inputs), but the
initial distortion is still propagated through the first linear layer.

### 3.3 Sample Sizes

| Aspect | Published Methods | Our Approach |
|--------|-------------------|--------------|
| **Training samples** | 49–300 per class (max trial) | ~1,043 positives, ~4,458 negatives |
| **Task complexity** | Up to 40 classes | Binary (yew / not-yew) |
| **Low-shot regime** | 1 and 10 samples per class tested | N/A (substantial data) |

**Assessment:** We have **more** training data than most of the paper's evaluations.
The US Trees evaluation uses 300/class for 39 genera. For a binary task with
~1,043 positive and ~4,458 negative samples, we are well above the data volumes
the embeddings were designed for. The paper claims 150 samples per class suffice
for 87-class crop mapping. Our data abundance means even a suboptimal classifier
will likely work well.

### 3.4 Task Framing

| Aspect | Published Methods | Our Approach |
|--------|-------------------|--------------|
| **Task type** | Multi-class classification (genus) | Binary classification (species presence) |
| **Spatial unit** | Point samples (1 pixel) | Every pixel in 10 km × 10 km tiles |
| **Temporal** | Single-date or annual period | Annual embeddings (2020/2022) |
| **Application** | Genus distribution mapping | Population density estimation |

**Assessment:** Our binary classification task is simpler than 39-genus classification,
which favours our high performance. However, our spatial application (every pixel
in a tile) introduces spatial autocorrelation that point-sample evaluations avoid.
The paper enforces 1.28 km minimum spacing between samples specifically to reduce
spatial autocorrelation. Our training data from iNaturalist likely has clustered
observations, which could inflate apparent accuracy.

---

## 4. Baseline Comparisons in the Literature

The paper compared AEF against these approaches (all using kNN/linear transfer):

| Approach | Type | Dims | Inputs |
|----------|------|------|--------|
| XY | Control | 4 | Lat/lon |
| XYZ | Control | 5 | Lat/lon + elevation |
| ViT (ImageNet) | Control | 1024 | Sentinel-2 RGB |
| Composites | Designed | 16 | S1, S2, Landsat |
| CCDC | Designed | 54 | Landsat harmonics |
| MOSAIKS | Designed | 1024 | S1, S2, Landsat |
| SatCLIP | Learned | 256 | Location encoder |
| Prithvi | Learned | 768–2304 | HLS L30 |
| Clay | Learned | 768 | S1, S2, Landsat |
| **AEF** | **Learned** | **64** | **Multi-source** |

Key findings:
- **AEF outperformed all** in 15/15 evaluations at max-trial
- **SatCLIP** (location-only encoder) was next-best for US Trees — suggesting
  geographic location is highly informative for tree species
- **Prithvi** (NASA's geospatial foundation model) performed poorly, "no better
  than ViT control" — it wasn't designed as a feature space
- **MOSAIKS** was next-best for land cover tasks
- **CCDC** harmonics were next-best for crop mapping

---

## 5. The GBIF / Species Signal

A critical detail: AEF was **trained with GBIF species occurrence records** as a
text-contrastive target. The training pipeline:

1. GBIF records for Plantae, Animalia, Fungi (2017–2023) were filtered to
   research-grade observations with ≤240m spatial uncertainty
2. Species names matched to Wikipedia articles via GBIF taxon IDs
3. Text embeddings from species descriptions were used as contrastive alignment
   targets during model training

This means the embedding space already encodes species distribution patterns.
When we use iNaturalist yew observations (which are a subset of the GBIF data
used in training), we are essentially asking the embeddings to reconstruct
information they were explicitly trained to represent.

**Implication:** Simple classifiers (kNN, linear) may work especially well for
species detection because species-level patterns are already linearised in the
embedding space. An MLP adds unnecessary complexity for a task the embeddings
were optimised to support.

---

## 6. Geometric Concerns with StandardScaler

### Why It Matters
The AEF embeddings live on the unit hypersphere S⁶³. Two key properties:

1. **Dot product = cosine similarity** (since ‖e‖ = 1 for all embeddings)
2. **L2 distance ∝ angular distance**: ‖a - b‖² = 2(1 - cos θ)

These properties mean:
- Similar land cover → small angular distance → high cosine similarity
- The paper's kNN explicitly leverages this via L2 distance
- Unsupervised change detection uses normalised dot products directly

### What StandardScaler Does
Given training embeddings E ∈ ℝ^(N×64):
- Subtracts per-band mean: e'_i = e_i - μ_i
- Divides by per-band std: e''_i = e'_i / σ_i

After scaling:
- Vectors are **no longer unit-length**
- Pairwise distances are **distorted** (high-variance dimensions shrink, low-variance expand)
- The **angular geometry** the embeddings were designed to preserve is broken
- Cluster structure may be disrupted

### Mitigation
- Our BatchNorm layers partially compensate (they re-normalise each hidden layer)
- The MLP can learn to undo some distortion via its first linear layer
- But the initial information loss from dimensional re-weighting cannot be fully recovered

### Recommendation
For any non-MLP classifier (kNN, RF, linear), use **raw embeddings** without
StandardScaler. For the MLP, consider replacing StandardScaler with L2
normalisation (which preserves the spherical geometry) or removing it entirely.

---

## 7. Strengths of Our Approach

Despite the non-standard choices, several factors support the validity of our results:

1. **High performance (AUC=0.998)** — regardless of whether the classifier is
   optimal, it clearly works. The embeddings contain strong signal for yew detection.

2. **Iterative negative refinement** — our 4-round training with progressively
   better negatives is a form of hard negative mining that the paper's one-shot
   kNN/linear evaluations don't employ. This likely improves boundary quality.

3. **Generous sample sizes** — with ~1,043 positives, we exceed the paper's
   max-trial sizes. The MLP has enough data to learn without severe overfitting.

4. **Application-appropriate thresholding** — using P ≥ 0.95 instead of argmax
   is a conservative choice that reduces false positives at the cost of recall.

5. **Binary task simplicity** — distinguishing one species from background is far
   easier than 39-genus classification. High performance is expected.

6. **Post-hoc validation** — the forestry mask overlay and tile-matched methodology
   provide independent validation of spatial patterns.

---

## 8. Potential Weaknesses

1. **StandardScaler on unit-length embeddings** — geometrically inappropriate
   (see Section 6). The most concerning divergence from published practice.

2. **Overparameterised classifier** — ~13,000 parameters for a task published
   methods solve with 0 (kNN) or 64 (linear). Raises overfitting risk, especially
   with weighted/augmented training data.

3. **No comparison against simple baselines** — we never tested kNN or linear
   probe on these embeddings. Without this comparison, we cannot claim our MLP
   adds value over the paper's recommended approaches.

4. **Spatial autocorrelation** — iNaturalist observations cluster geographically.
   The paper enforces 1.28 km minimum spacing; we don't, which may inflate
   train/test performance metrics.

5. **No cross-validation** — with iterative retraining and a fixed validation
   split, there's risk of information leakage across rounds.

6. **Class imbalance handling** — sample weighting (annotations × 3, logged × 2)
   is ad hoc. The paper uses balanced per-class sampling.

---

## 9. Literature on Species Distribution Modelling with Foundation Models

### Traditional SDM Methods
- **MaxEnt** (Phillips et al., 2006) — presence-only modelling using environmental
  features. The standard in ecology but requires handcrafted features.
- **BRT/GBM** (Elith et al., 2008) — boosted regression trees on bioclimatic variables
- **Random Forest** — the workhorse of remote sensing classification

### Emerging Foundation Model Approaches
- **SatCLIP** (Klemmer et al., 2025) — location encoder trained contrastively with
  satellite images. Works well for species distribution (next-best after AEF for
  US Trees) but is location-only (no temporal dynamics).
- **MOSAIKS** (Rolf et al., 2021) — designed random-feature kernel on satellite
  imagery. Good for land cover, less for species.
- **Clay Foundation Model** (2024) — ViT-based, multi-source. Reasonable but
  outperformed by AEF on all tasks.

### How AEF Changes the Landscape
The AEF paper represents a paradigm shift: instead of engineering features or
fine-tuning foundation models, practitioners can use pre-computed 64D embeddings
with trivial classifiers. For species distribution modelling specifically:

- The embeddings encode **biophysical variables** (canopy height, climate,
  topography) plus **species occurrence patterns** (via GBIF text alignment)
- This is equivalent to an implicit "environmental niche model" — the embedding
  already captures the multivariate environmental envelope
- Simple kNN in embedding space effectively performs **similarity-based habitat
  matching** without explicit niche modelling

---

## 10. Recommendations

### Short-term (Validation)
1. **Test kNN (k=1, k=3) on raw embeddings** using the same train/test split.
   Report balanced accuracy and AUC. This is the primary recommended baseline
   from the literature.
2. **Test a linear probe** (logistic regression, λ=0) on raw embeddings.
3. **Test Random Forest** on raw embeddings — while not in the paper, RF is the
   standard in remote sensing and respects the native embedding geometry.
4. **Compare MLP with and without StandardScaler** — test the MLP on raw embeddings
   to isolate the scaling effect.

### Medium-term (If Results Warrant)
5. **Ensemble approach** — if kNN and MLP both perform well but disagree on
   borderline pixels, their consensus could be more robust.
6. **Cosine-distance kNN** — since embeddings are on S⁶³, cosine similarity may
   outperform L2 for kNN (though they're monotonically related for unit vectors).
7. **Spatial cross-validation** — implement block cross-validation to account for
   spatial autocorrelation in iNaturalist data.

### For Publication / Defence
8. The **defensible framing** is: "We used a standard MLP classifier on AlphaEarth
   Foundations embeddings. While the published literature demonstrates that even
   kNN achieves strong results on these embeddings, our MLP with iterative
   negative mining was designed to maximise boundary quality for population-level
   mapping. We validated the MLP against [kNN/linear/RF] baselines and found
   [comparable/superior] performance."
9. **Remove or justify StandardScaler** — either switch to raw embeddings (preferred)
   or provide explicit justification with ablation results.

---

## 11. Empirical Validation Results

We implemented all recommended methods and ran a head-to-head comparison on the
same training/validation split (21,489 train, 700 val). **Embeddings confirmed
unit-length** (mean L2 norm = 1.0001, σ = 0.002).

| Classifier | AUC | F1 | Balanced Acc | Precision | Recall | Scaling | Time |
|---|---|---|---|---|---|---|---|
| **MLP raw** | **0.9962** | 0.9596 | 0.9744 | 0.9484 | 0.9712 | none | 16.3s |
| **MLP + StandardScaler** | 0.9961 | **0.9765** | **0.9898** | 0.9541 | **1.0000** | SS | 13.3s |
| kNN (k=3) raw | 0.9909 | 0.8333 | 0.8634 | 0.9451 | 0.7452 | none | 0.0s |
| **Random Forest raw** | 0.9896 | 0.9742 | 0.9888 | 0.9498 | 1.0000 | none | 3.0s |
| Random Forest scaled | 0.9896 | 0.9742 | 0.9888 | 0.9498 | 1.0000 | SS | 1.8s |
| kNN (k=5) raw | 0.9847 | 0.7899 | 0.8308 | 0.9463 | 0.6779 | none | 0.0s |
| kNN (k=1) raw | 0.9730 | 0.9594 | 0.9730 | 0.9526 | 0.9663 | none | 0.1s |
| Logistic Regression raw | 0.9165 | 0.5619 | 0.6948 | 0.9231 | 0.4038 | none | 4.0s |
| Ridge Classifier raw | 0.9060 | 0.3643 | 0.6099 | 0.9400 | 0.2260 | none | 0.1s |

### Key Findings

1. **MLP+StandardScaler** and **MLP raw** are effectively tied for best AUC
   (0.9961 vs 0.9962). StandardScaler does **not** hurt MLP performance — in fact
   it achieves higher F1 (0.9765 vs 0.9596) and perfect recall.

2. **Random Forest raw** is the strongest non-neural method (AUC=0.9896, F1=0.9742),
   nearly matching MLP. RF is invariant to monotonic scaling, so raw vs scaled
   results are identical (as expected).

3. **Linear probes perform poorly** (AUC=0.916, F1=0.562), contradicting the
   expectation that the yew-detection boundary is linearly separable. This suggests
   the species-level signal in these embeddings requires nonlinear decision boundaries,
   possibly because yew co-occurs with diverse forest types.

4. **kNN (k=1) works well** (F1=0.9594) but **kNN (k=3,5) degrades sharply**
   in F1/recall, likely due to the class imbalance (20:1 neg:pos ratio diluting
   positive neighborhoods at higher k).

5. The **MLP is justified** for this task despite the literature recommending simpler
   methods. The yew detection boundary is genuinely nonlinear.

### Updated Assessment

Our original concern that "StandardScaler destroys the unit-length geometry" was
theoretically valid but **empirically harmless**. The MLP learns an arbitrary
nonlinear mapping and adapts to whatever input distribution it receives.
The scaler's role is to improve gradient flow during training (helps BatchNorm
converge faster), which is orthogonal to the angular geometry concern.

**For production mapping, RF raw is recommended** because:
- Nearly identical accuracy to MLP (ΔAUC=0.007, ΔF1=0.002)
- No GPU required for inference
- No scaler needed (operates on raw unit-length embeddings)
- More interpretable (feature importances available)
- Faster inference on CPU (~10s per 10km tile vs MLP on GPU)

All 35 CWH spot tile maps were regenerated with RF raw for comparison.

---

## 12. Summary Comparison Table

| Dimension | AlphaEarth Paper | Google Blog / Tutorials | Our YewMLP | **RF raw (new)** |
|-----------|-----------------|------------------------|------------|------------------|
| **Classifier** | kNN, linear probe | kNN, RF, ee.Classifier | 4-layer MLP | Random Forest 500 trees |
| **Preprocessing** | None (raw 8-bit quantised) | None | StandardScaler | None (raw) |
| **Training samples** | 49–300/class | ~150/class (crop demo) | ~1,043 pos / ~4,458 neg | same |
| **Task** | Multi-class (up to 40) | Various | Binary | Binary |
| **Validation** | 1.28km-spaced, balanced | N/A | iNat + VRI (clustered) | same |
| **Embedding norm** | Unit-length preserved | Unit-length preserved | Destroyed by scaling | **Preserved** |
| **Species evaluation** | 39 genera, US Trees | Mangrove mapping | 1 species (Taxus brevifolia) | same |
| **AUC** | N/A | N/A | 0.9961 | **0.9896** |
| **F1** | N/A | N/A | 0.9765 | **0.9742** |
| **Published?** | Nature-tier arXiv, 2025 | Google AI Blog | Project-internal | Project-internal |

---

## 12. References

- Brown, C.F., Kazmierski, M.R., Pasquarella, V.J. et al. (2025). AlphaEarth
  Foundations: An embedding field model for accurate and efficient global mapping
  from sparse label data. arXiv:2507.22291v2.
- Google Earth Engine (2025). Satellite Embedding V1 Annual dataset catalog.
  developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL
- Klemmer, K. et al. (2025). SatCLIP: Global, general-purpose location
  embeddings with satellite imagery. ICLR 2024.
- Rolf, E. et al. (2021). A generalizable and accessible approach to machine
  learning with global satellite imagery. Nature Communications 12:4392.
- Clay Foundation (2024). Clay Foundation Model. github.com/Clay-foundation/model

---

*Document generated from literature review of AlphaEarth Foundations paper and
Google Earth Engine documentation. Last updated: 2025.*
