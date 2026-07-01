# Draft reference audit & critical review

*Paper:* "Satellite Embedding–Based Mapping of Pacific Yew (*Taxus brevifolia*) Habitat Decline Across British Columbia, and an IUCN Red List Assessment of the Canadian Population."

This addresses two requests: (1) which recent works in the literature corpus are **missing** from the draft's reference list, and (2) **where the study may be lacking**. The review is based on the draft's full Methods, Results, Discussion, and Limitations sections.

## 1. Reference-list profile

The draft cites **31 references**, with a **median publication year of 2000**. The distribution:

| Period | Count | Character |
|---|---|---|
| pre-2000 | 15 | *Taxus* ecology classics (Busing & Spies 1995), BC silvics/ethnobotany, de Liocourt 1898, Wani 1971 (taxol isolation) |
| 2000–2012 | 10 | BC forest-hydrology handbooks, Guisan & Zimmermann 2000 (the *only* SDM-methods citation) |
| 2013–2019 | 2 | Thomas 2013 (IUCN), Reynolds 2022 thesis |
| 2020+ | 4 | Brown et al. 2025 (AlphaEarth), COSEWIC 2024, Reynolds 2022, Turner 2021 |

**The methodological backbone is essentially uncited from the recent literature.** Satellite embeddings are supported only by the AlphaEarth preprint itself; XGBoost, the SDM framework, presence-only/pseudo-absence sampling, spatial cross-validation, and RS-based IUCN assessment have *no* current citations (several appear as `TODO cite` in the text). None of the 70 works in the literature-map corpus appears in the reference list — the draft and the active methods literature do not currently intersect.

## 2. Recommended additions (mapped to the draft)

These are the highest-value missing works, grouped by the part of the paper they support. Full records are in `yew_missing_references.csv`.

### 2.1 Satellite embeddings — the method needs external validation beyond its own preprint

- Xiao Xiang Zhu et al. (2026). *On the foundations of Earth foundation models.* Communications Earth & Environment. [link](https://doi.org/10.1038/s43247-025-03127-x)
- Chao Jin et al. (2026). *Assessing the Utility of Satellite Embedding Features for Biomass Prediction in Subtropical Forests with Machine Learning.* Remote Sensing. [link](https://doi.org/10.3390/rs18030436)
- Yusheng Zheng et al. (2026). *Towards Trustworthy Urban Land Use Classification: A Synergistic Fusion of Deep Learning and Explainable Machine Learning with a Nanning Case Study.* Land. [link](https://doi.org/10.3390/land15010158)
- Yuchi Ma et al. (2026). *Harvesting AlphaEarth: Benchmarking the geospatial foundation model for agricultural downstream tasks.* International Journal of Applied Earth Observation and Geoinformation. [link](https://doi.org/10.1016/j.jag.2026.105258)

### 2.2 SDM reporting, pseudo-absence design & spatial validation — directly bears on §2.3–2.4

- Robert A. Barber et al. (2021). *Target‐group backgrounds prove effective at correcting sampling bias in Maxent models.* Diversity and Distributions. [link](https://doi.org/10.1111/ddi.13442)
- Elliott L. Hazen et al. (2021). *Where did they not go? Considerations for generating pseudo-absences for telemetry-based habitat models.* Movement Ecology. [link](https://doi.org/10.1186/s40462-021-00240-2)
- Damaris Zurell et al. (2020). *A standard protocol for reporting species distribution models.* Ecography. [link](https://doi.org/10.1111/ecog.04960)
- Pierre Ploton et al. (2020). *Spatial validation reveals poor predictive performance of large-scale ecological mapping models.* Nature Communications. [link](https://doi.org/10.1038/s41467-020-18321-y)

### 2.3 Area of occupancy & RS-based IUCN assessment — supports §4.6

- Victor Cazalis et al. (2024). *Accelerating and standardising IUCN Red List assessments with sRedList.* Biological Conservation. [link](https://doi.org/10.1016/j.biocon.2024.110761)
- Robert P. Anderson (2022). *Integrating habitat‐masked range maps with quantifications of prevalence to estimate area of occupancy in IUCN assessments.* Conservation Biology. [link](https://doi.org/10.1111/cobi.14019)
- Jamie M. Kass et al. (2020). *Improving area of occupancy estimates for parapatric species using distribution models and support vector machines.* Ecological Applications. [link](https://doi.org/10.1002/eap.2228)

### 2.4 RS-driven forest/tree demography & habitat change — supports the decline mapping

- Emily R. Lines et al. (2022). *The shape of trees: Reimagining forest ecology in three dimensions with remote sensing.* Journal of Ecology. [link](https://doi.org/10.1111/1365-2745.13944)
- Jessica Needham et al. (2022). *Tree crown damage and its effects on forest carbon cycling in a tropical forest.* Global Change Biology. [link](https://doi.org/10.1111/gcb.16318)
- Nate G. McDowell et al. (2020). *Pervasive shifts in forest dynamics in a changing world.* Science. [link](https://doi.org/10.1126/science.aaz9463)
- Abreham Berta Aneseyee et al. (2020). *The InVEST Habitat Quality Model Associated with Land Use/Cover Changes: A Qualitative Case Study of the Winike Watershed in the Omo-Gibe Basin, Southwest Ethiopia.* Remote Sensing. [link](https://doi.org/10.3390/rs12071103)

## 3. Areas where the work may be lacking


Ordered roughly by how much each affects the headline claims. These are framed as a friendly internal review, not a verdict — most are addressable.


### 3.1 The classifier's AUC almost certainly overstates yew-detection ability
The production XGBoost reports AUC-ROC 0.996 and 98.9% accuracy. For a cryptic understorey species mapped from 10 m satellite embeddings, that is implausibly high as a measure of *yew detection*, and the paper's own caveat (§4.5, "probabilities represent habitat suitability rather than confirmed presence") concedes the underlying issue without following it through to the metric. The cause is the **negative set**: the 11,452 negatives are alpine/subalpine, water, and non-yew canopy (hemlock/Sitka spruce) — spectrally *very* different from moist coastal old-growth. The model is largely learning to separate "moist low-elevation old-growth forest" from "everything obviously not that," which is an easy task, rather than yew from co-occurring suitable habitat where it is absent. A near-perfect AUC against easy negatives is consistent with a model that cannot actually localise yew within suitable forest. **Suggested fixes:** report performance against *hard* negatives (yew-absent moist old-growth at the same elevation/BEC subzone); report precision/recall at the operational threshold on a spatially held-out region, not just AUC; and treat the 0.996 as a habitat-discrimination figure, relabelling it as such throughout.

### 3.2 No independent field validation of the habitat map — and the field data that exist aren't used to test it
The paper acknowledges this (§4.5, "No systematic field validation"), but it is the single biggest evidentiary gap, and there is low-hanging fruit. The East Muir Creek transect (n = 120) and the 461 FAIB stems across 122 plots are spatial point datasets that could serve as independent presence checks against the predicted probability surface — yet they are used only for size-structure analysis, never to validate the map's spatial predictions. At minimum, the FAIB yew-present plots should fall in high-probability pixels and the non-yew plots in low-probability pixels; quantifying that (e.g., a confusion matrix at plot locations the model never saw, or a Boyce index for presence-only validation) would materially strengthen the map. As written, the central data product — the decline figure — rests on an unvalidated surface.

### 3.3 The headline 69.2% decline is largely an *old-growth loss* estimate relabelled as yew-habitat loss
The logging-suppression rule sets habitat to ×0 for *all* non-old-growth forest (categories 2–5, including 80–150 yr stands) and ×1 only for old-growth >150 yr. This binary, all-or-nothing mapping means the decline number is driven almost entirely by the equation **yew habitat = old-growth area**. That is a defensible conservation position (the paper argues it via the IUCN reproductive-age criterion), but it has two consequences that should be stated more plainly: (i) the result is closer to "old-growth has declined 69%" than to a yew-specific demographic loss, since yew demonstrably persists in younger and managed stands (the paper itself notes yew surviving post-logging); and (ii) the "original habitat" baseline (154,483 ha) is **modelled, not observed** — it assumes all currently-logged forest was once yew habitat at old-growth density. The decline is therefore the ratio of two model constructs. A sensitivity analysis with graded (non-binary) suppression factors, and an explicit statement that the baseline is counterfactual, would make the number more defensible.

### 3.4 No uncertainty on the decline figure, and purposive tiles undercut extrapolation
The 69.2% decline is a single point estimate with no confidence interval, despite being assembled from a probability surface, VRI polygons with known age errors (acknowledged §4.5), and fire/erosion modifiers — each a source of propagatable uncertainty. The probability-mass approach (Σp × area) yields a number but discards its variance. Separately, the 98 tiles are a *purposive* (non-random) sample covering ~0.26% of the zone extent; the paper is commendably candid that the figure is "conditional on the sampled tiles," but then the Abstract/Conclusions still lead with 69.2% as though province-wide. Either propagate uncertainty to a CI on the decline, or foreground the conditional framing consistently (including in the abstract).

### 3.5 The stated research program is a population model, but the paper has no demographic projection
Relative to a Pacific-yew *population model*, the demographic content here is thin. The Busing & Spies (1995) stage matrix (λ = 1.02) is borrowed only to drive a fire-recovery curve; there is no forward population projection, no PVA, no quantification of how the documented large-tree depletion translates into future trajectory or extinction probability. The size-structure analysis (de Liocourt reverse-J, q ≈ 2.0 vs 1.5) is a static, descriptive comparison — appropriately caveated as not mechanistic — not a demographic model. If the broader goal is a population model, the natural extension is an integral projection model (IPM) or matrix model parameterised from the size-structure data and FAIB densities, projecting λ and quasi-extinction under continued logging. The RS-demography corpus (e.g., IPMs from repeat LiDAR) is directly relevant template here.

### 3.6 IUCN Criterion A: temporal-window mismatch between what was measured and what A2c requires
Criterion A2c requires population reduction over the longer of 10 years or **3 generations** (~240–300 yr here, by the paper's own generation length). But the decline is measured as a 2024 *snapshot* — current old-growth versus modelled pre-logging baseline — not as a reduction tied to that moving window. Logging "beginning in the 1920s" is roughly within three generations, so the inference is plausible, but the mapping from a static habitat ratio to a 3-generation population reduction is asserted rather than derived. The assessment would be more robust if it stated the assumed decline trajectory over the generation window explicitly (and ideally bracketed Criterion A2c's "reduction may not have ceased" with the ongoing-logging projection it currently omits).

### 3.7 Presence data inherit iNaturalist sampling bias, only partially addressed
Positives are 1,043 iNaturalist records plus 64 field points. iNaturalist data carry strong spatial bias toward roads, trails, and population centres; this biases the embedding signature of "yew habitat" toward accessible forest and can inflate apparent association with particular conditions. The paper filters out developed/logged coordinates (good) but does not correct for observer-effort bias in the background/pseudo-absence design — the recommended target-group-background and ODMAP-reporting literature (§2.2 above) addresses exactly this. A target-group background (using other iNaturalist plant records to define the sampling surface) would be a relatively cheap, defensible upgrade.

### 3.8 Several secondary-threat parameters are unsourced or single-sourced
The mite bud-mortality figures (§2.6.3, ">20%", "~25% aril reduction") carry an explicit `TODO find citation`; ungulate browse mortality (60–80%) leans on a 1995 source and a monitoring report; the deer-overabundance claim is `TODO cite`. These feed the "compounding threats" narrative (§4.4) and the argument that the decline understates true loss. They should be sourced or flagged as assumptions, since reviewers will probe the quantitative threat multipliers.

### 3.9 Field size-structure inference rests on one 3-km creek transect
The n = 120 sample is from a single watercourse (East Muir Creek), generalised toward a province-scale claim. The FAIB corroboration (n = 461) genuinely strengthens this and is the paper's best move, but the field transect alone is a single site and the riparian-only sampling may not represent upslope stands. The large-tree-deficit conclusion is well supported by FAIB; the framing should lean on FAIB as the primary structural evidence and treat the transect as illustrative, to pre-empt a pseudoreplication critique.

## 4. Summary — priorities

**Reference list:** the scholarship is sound on yew natural history but the *methods* are uncited from the modern literature. The single most useful addition is a small set of method-validation citations (ODMAP reporting, target-group backgrounds, spatial-validation, AOO estimation, and an RS-IUCN precedent such as sRedList) — these both ground the methods and pre-empt reviewer objections. Resolve the in-text `TODO cite` markers (XGBoost, US Forest Service 1990s estimate, deer overabundance, mite damage).

**Biggest scientific exposures, in order:**
1. The 0.996 AUC overstates yew detection — re-evaluate against hard, same-habitat negatives and report precision/recall (§3.1).
2. The habitat map is unvalidated, yet FAIB/field points could validate it directly (§3.2).
3. The 69.2% decline is effectively an old-growth-loss metric against a counterfactual baseline, with no uncertainty interval — add graded suppression sensitivity and a CI, and keep the "conditional on sampled tiles" framing in the abstract (§3.3–3.4).
4. For a stated *population model*, add a forward demographic projection (IPM/matrix PVA) rather than only static size-structure (§3.5).

**What is already strong:** the two-independent-lines-of-evidence design (RS decline + size structure + FAIB), the candid Limitations section, the spatial cross-validation, the threshold-free probability-mass approach, and the careful q-sensitivity treatment of the size-structure test. The critique above is about tightening inference and citation, not about the core finding, which is plausibly correct.

---
*Method: reference list extracted from the draft (pp. 17–18) and compared against the 70-work literature-map corpus plus targeted OpenAlex searches for SDM-methods, AOO/IUCN, and validation literature. Recommended works are in `yew_missing_references.csv`. Critique is grounded in the draft's Methods/Results/Discussion (pp. 3–16).*
