# Implementation Plan: Pacific Yew Paper Revisions

**Handoff document for implementation.** This translates the reference audit and demographic
analysis into concrete, sequenced engineering tasks against the project's actual data holdings
(VRI polygons, FAIB plot data, AlphaEarth embeddings, field transect, iNaturalist occurrences).
Each workstream below is self-contained with inputs, method, deliverable, and acceptance
criteria, ordered by leverage on the paper's headline claims.

---

## Priority 0 — Do first: these two make every downstream number defensible

### P0.1 — Re-evaluate the classifier against hard, same-BEC-subzone negatives

**Problem:** Reported AUC = 0.996 is likely inflated. The current negative set (11,452
points) is drawn from alpine, water, and hemlock/spruce stands — spectrally trivial to
separate from moist coastal old-growth. The model may just be learning "old-growth vs.
obviously-not," not "yew-suitable vs. yew-absent-but-similar."

**Data needed:**
- VRI polygons with BEC (Biogeoclimatic Ecosystem Classification) subzone/variant attribute
- Existing AlphaEarth embedding tiles + current negative/positive labels
- FAIB plot locations (461 stems / 122 plots) and field transect points (n=120) — currently
  used only for size structure, needed here as an independent validation set (see P0.2)

**Method:**
1. For every current positive tile, identify its BEC subzone/variant.
2. Re-draw negatives restricted to the **same BEC subzone(s)** as positives — i.e., other
   conifer/old-growth stands within CWHvm1/CWHvm2 (or whichever subzones the yew tiles fall in)
   that are NOT yew-occupied, rather than alpine/water/hemlock-spruce from anywhere in the
   province.
3. Retrain (or re-evaluate the existing model, if retraining is out of scope) on this harder
   negative set.
4. Report **precision/recall/F1 at the operational classification threshold actually used to
   build the habitat map** — not just AUC, which is threshold-independent and can mask a
   collapse in usable precision.
5. Report both the original (easy-negative) and new (hard-negative) metrics side by side, with
   the difference discussed explicitly rather than only the better number retained.

**Deliverable:** A table (easy-negative vs. hard-negative: AUC, precision, recall, F1 at
operational threshold) plus 1 paragraph in Methods/Results explaining the distinction and
what it means for interpreting the habitat map's accuracy claims.

**Acceptance criteria:** Numbers are computed from an actual re-run, not estimated; the
BEC-subzone restriction logic is documented (which subzones, why); if hard-negative AUC drops
materially (e.g., >0.03–0.05), the abstract/conclusion language softens accordingly.

---

### P0.2 — Validate the habitat map against FAIB/field presence points

**Problem:** The central data product (the habitat-quality map) is never checked against the
presence data the project already has. FAIB plots and the field transect are used only for
size-structure analysis (de Liocourt), never to ask "do known yew-present locations actually
fall in high-probability pixels?"

**Data needed:**
- FAIB plot coordinates + species presence (461 stems, 122 plots)
- The habitat-probability raster/tiles output by the XGBoost classifier

**Method:**
1. Extract predicted habitat probability at each FAIB plot that has not been logged and is not in water. you may need to get the data from earth engine
   (point-in-raster/tile lookup).
2. Compute a **confusion matrix** at the operational threshold: known-present points classified
   as suitable vs. not.
3. Compute the **Boyce index** (continuous Boyce, not just AUC) across the full probability
   gradient — this is the standard presence-only validation metric in the SDM literature and
   directly answers "does higher predicted suitability actually track more presences" without
   needing true absences.
4. If FAIB/transect points were used anywhere in training, explicitly hold out a subset for
   this validation (spatial blocking, not random split, given tile-level autocorrelation — see
   P0.1's BEC-subzone logic and Ploton et al. 2020 in the reference guide).

**Deliverable:** Confusion matrix + Boyce index value + a short paragraph/figure inserted into
Results validating (or explicitly caveating) the habitat map against independent field data.

**Acceptance criteria:** Uses coordinates actually present in the FAIB/transect datasets (no
synthetic data); validation points are spatially distinct from any points used in classifier
training/tuning; Boyce index and confusion matrix are reported together (neither alone is
sufficient for presence-only validation).

---
## Priority 2 — Demographic projection (already scaffolded — needs real data substituted)

### P2.1 — Replace placeholder vital rates and threat parameters with fitted values from FAIB data

**Status:** A working stage-structured (Lefkovitch) matrix model with explicit, tunable mite-
damage and fire-multiplier parameters has already been built (see `yew_demographic_model.py`,
`yew_demographic_projection_report.md` in project artifacts) and calibrated to reproduce
Busing & Spies' (1995) λ=1.02 and the draft's q≈1.5 de Liocourt target. **This is a
reconstruction fit to match reported summary statistics, not a re-derivation from real stem
data — that substitution is the remaining work.**

**Data needed:**
- FAIB plot stem-level data (461 stems, DBH + status, ideally with plot-level density) to fit
  actual stage-specific survival/growth transition probabilities (not just the stable-stage
  ratio) — e.g., via a static size-distribution → transition-probability inference (Usher
  matrix approach) if repeat-census data isn't available, or a proper IPM if growth-increment
  data exists.
- Any available repeat-measurement or growth-increment data (even sparse) to replace the
  currently assumed 0–3 cm DBH/decade growth-probability schedule with a fitted one.
- A sourced mite dose-response relationship if one exists in the applied/forestry entomology
  literature (searched during the reference audit; none specific to *T. brevifolia* vital
  rates was found — worth one more targeted search pass over grey literature / provincial
  forest health reports before accepting the current 20%/25% placeholder).
- BC Wildfire Service / VRI fire-history layer, to replace the placeholder baseline annual
  burn probability (p₀ = 0.002) with an actual fitted value for the CWH coastal-yew zone.

**Method:**
1. Fit stage-transition probabilities directly from the FAIB DBH-class stem counts (via
   quotient/Usher-matrix estimation from the observed size distribution) instead of the current
   constraint-satisfying reconstruction.
2. Re-run the existing sensitivity grid (`yew_lambda_sensitivity_grid.csv` logic) with the
   refitted matrix.
3. Refit p₀ from the BC fire-history layer; recompute the φ-thresholds table
   (`yew_phi_required_for_iucn_thresholds.csv`) with the real baseline.
4. Re-generate the two-panel figure and all three CSVs with real inputs; diff against the
   current placeholder-based versions to characterize how much the qualitative conclusion
   (mite alone insufficient; fire multiplier ~12–17× needed to hit IUCN thresholds) moves.

**Deliverable:** Updated `yew_demographic_model.py` with fitted (not reconstructed) vital
rates, updated CSVs/figure, and a short paragraph noting what changed vs. the placeholder
version and why.

**Acceptance criteria:** Every vital rate traces to an actual data source (FAIB counts, fire
layer, or an explicitly cited literature value) — no remaining "chosen to satisfy λ=1.02"
justification for a rate that could instead be fitted from data.

---

## Priority 3 — Remaining items (lower leverage, still needed for completeness)

### P3.2 — IUCN A2c temporal-window alignment (§3.6)
The 2024 snapshot vs. 3-generation (~240–300 yr under the 80–100 yr generation-length bound)
window mismatch. Once P2.1's refitted matrix exists, project backward/forward using the
matrix to state a proper 3-generation percent reduction (as already computed in
`yew_scenario_3gen_reduction.csv`, to be refreshed under P2.1) rather than reporting the raw
2024-vs-baseline snapshot as if it were the 3-generation figure. Deliverable: replace the raw
snapshot percentage in the A2c discussion with the matrix-projected 3-generation figure, or
present both with a clear label distinguishing "observed snapshot" from "3-generation
projection."

### P3.3 — Source or explicitly flag secondary-threat parameters (§3.8)
Covered by P2.1 (mite/fire factors now explicit and separable) — remaining task is a final
literature pass (BC Ministry of Forests health reports, provincial pest surveys) specifically
for a *T. brevifolia*-or-*Taxus*-genus mite dose-response curve, since none surfaced in the
academic literature search. If none is found, state explicitly in the paper that the 20%/25%
multipliers are illustrative sensitivity bounds, not fitted values, with the placeholder
CAVEATS section from `yew_demographic_projection_report.md` §5 reused verbatim as the
paper's own limitation statement.

---

## Suggested Sequencing & Dependencies

```
P0.1 (hard negatives) ──┐
                        ├──> both feed the honest-accuracy narrative used throughout
P0.2 (map validation) ──┘         Results/Discussion revisions

P2.1 (refit demographic model) ── depends on FAIB stem data + fire-history layer being
                                    accessible; independent of P0/P1
      │
      ├──> P3.2 (A2c window) — depends on P2.1's refitted matrix
      └──> P3.3 (threat params) — depends on P2.1's parameter structure (already built)

```

**Recommended order for a single implementer:** P0.1 → P0.2 → P1.1 → P2.1 → P3.2/P3.3 → P3.1 →
P3.4. P0.1/P0.2 share the BEC-subzone and point-extraction infrastructure and should be built
together; P1.1 is independent and can be parallelized; P2.1 is the largest single piece of new
work and depends only on data access, not on P0/P1 completing first.

## Existing Artifacts to Reuse (not rebuild from scratch)

- `yew_demographic_model.py` — model skeleton (matrix construction, projection function) ready
  for P2.1's real-data refit; only the vital-rate-fitting function needs replacing.
- `yew_reference_integration_guide.txt` — formatted citations + exact insertion paragraphs for
  Ploton et al., Zurell et al., Cazalis et al., Anderson, Kass et al., Barber et al., Hazen et
  al. — use directly for P0.1/P0.2 (Ploton, Barber, Hazen), P1.1/P3.2 (Cazalis, Anderson, Kass),
  and P3.1 (Barber, Hazen).
- `yew_reference_audit_review.md` — full nine-point critique with the original reasoning behind
  each item, useful context if a question arises about why a task is scoped the way it is.
