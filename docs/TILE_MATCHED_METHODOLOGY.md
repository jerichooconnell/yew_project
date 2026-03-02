# Tile-Matched Controls — In-Depth Methodology

## 1. Overview

The **tile-matched controls** method (Option 5) estimates historical yew
population by using each 10×10 km tile as its own local control. The key insight:
mature forest surviving within a tile reveals the yew density that *was likely
present* in the now-logged portions of that same tile. This avoids the dilution
problem of a single province-wide average and naturally captures the strong
north–south gradient in yew density (e.g., Carmanah at ~31% vs Kitimat at ~0%).

**Script:** `scripts/analysis/yew_decline_tile_matched.py`
**Classifier:** Random Forest on raw embeddings (`rf_raw`, 500 trees)
**Threshold:** P ≥ 0.50 (RF natural decision boundary)
**Output:** `results/analysis/yew_decline_tile_matched.json`

---

## 2. Input Data

| Source | Description |
|--------|-------------|
| `tile_cache/{slug}_grid.npy`    | Random Forest (rf_raw) probability grid (0–1 per 10 m pixel) |
| `tile_cache/{slug}_logging.npy` | BC VRI rasterised land-use categories (10 m pixels) |
| `spot_stats.json`               | Tile metadata: name, lat/lon, dimensions |

### VRI Land-Use Categories

| Code | Category            | Treatment in analysis |
|------|---------------------|-----------------------|
| 0    | No data             | Excluded              |
| 1    | Water / non-forest  | Excluded (water mask) |
| 2    | Logged < 20 yr      | **Logged** — yew set to 0 |
| 3    | Logged 20–40 yr     | **Logged** — yew set to 0 |
| 4    | Logged 40–80 yr     | **Logged** — yew set to 0 |
| 5    | Mature forest > 80 yr | **Forest** — model predictions used |
| 6    | Alpine / barren     | Excluded              |

**Threshold:** P ≥ 0.50 throughout. A pixel is classified as "yew-present" if
the Random Forest classifier outputs probability ≥ 0.50 (i.e., a majority of
the 500 trees vote yew). This is the natural decision boundary for RF
`predict_proba`. The previous MLP used P ≥ 0.95 because its sigmoid output
concentrated near 1.0; RF tree-vote fractions rarely exceed 0.85.

---

## 3. Per-Tile Calculation (Step by Step)

For each of the 35 tiles:

### Step 3.1 — Extract masks

```python
forest_mask = (log == 5)       # mature forest (>80 yr)
logged_mask = (log == 2) | (log == 3) | (log == 4)  # all logged categories
```

### Step 3.2 — Count pixels

| Variable         | Formula                          | Meaning |
|------------------|----------------------------------|---------|
| `n_forest`       | `forest_mask.sum()`              | Number of mature forest pixels |
| `n_logged`       | `logged_mask.sum()`              | Number of logged pixels |
| `forested_px`    | `n_forest + n_logged`            | Total land that *is or was* productive forest |

### Step 3.3 — Current yew in mature forest

```
p50_forest = count of pixels where (grid[forest_mask] >= 0.50)
frac_yew   = p50_forest / n_forest
```

- `p50_forest` — number of 10 m pixels in this tile's mature forest where the
  RF classifier predicts P ≥ 0.50 for yew
- `frac_yew` — the **local yew rate**: fraction of mature forest pixels that
  are classified as yew. This is the tile's "control" rate.

### Step 3.4 — Historical yew in logged area

```
hist_in_logged = n_logged × frac_yew
```

**Assumption:** Before logging, the logged area within this tile had the same
yew density as the mature forest that survives within this tile. This is
reasonable because:
- At the 10 km scale, elevation, aspect, precipitation, and soil conditions are
  broadly similar across the tile
- Logging was not targeted specifically at yew-rich or yew-poor stands
- Mature forest acts as a local "reference condition"

### Step 3.5 — Historical total and decline

```
hist_total = p50_forest + hist_in_logged    (historical yew in this tile)
curr_total = p50_forest                      (logged areas = 0)
decline    = (1 − curr_total / hist_total) × 100
```

### Step 3.6 — Normalized density

```
curr_density = curr_total / forested_px     (fraction of forested land currently yew)
hist_density = hist_total / forested_px     (fraction of forested land historically yew)
loss_ratio   = hist_in_logged / curr_total  (lost / remaining)
```

**What density means:** If `curr_density = 0.05`, then 5 out of every 100 ha of
forested land currently has yew at P ≥ 0.95. In the output, this is multiplied
by 100 and reported as "5.000% = 5.000 ha yew per 100 ha forested."

---

## 4. Province-Wide Aggregation

### 4.1 — Summing across tiles

The tile-level totals are summed:

```
curr_total_px = Σ (p95_forest)        over all 35 tiles
hist_total_px = Σ (hist_total)        over all 35 tiles
lost_px       = hist_total_px − curr_total_px
decline_pct   = (1 − curr_total_px / hist_total_px) × 100
```

### 4.2 — Province-wide normalized density

```
tot_forested = Σ n_forest + Σ n_logged   (all forested land across 35 tiles)

curr_density_pct = (curr_total_px / tot_forested) × 100
hist_density_pct = (hist_total_px / tot_forested) × 100
```

### 4.3 — Loss:Remaining Ratio (L:R)

```
loss_ratio_agg = lost_px / curr_total_px
```

**How to interpret:**
- L:R = 1.0 → the amount of yew destroyed equals the amount that survives
- L:R > 1.0 → more yew was destroyed than currently remains
- L:R < 1.0 → most yew still survives

**Province-wide example (P ≥ 0.50, RF classifier):**
```
curr_total_px = 502,702 pixels  (= 5,027 ha of current yew)
hist_total_px = 1,391,732 pixels (= 13,917 ha of historical yew)
lost_px       = 889,030 pixels  (= 8,890 ha)

loss_ratio_agg = 889,030 / 502,702 = 1.77

→ For every 1 ha of yew that remains, 1.77 ha were lost to logging
```

**This is the number displayed as "Province-wide L:R = 1.77:1".**

---

## 5. How L:R Differs from Decline Percentage

| Metric           | Formula                                     | Value | Interpretation |
|------------------|---------------------------------------------|-------|----------------|
| Decline %        | `(1 − curr / hist) × 100`                  | 63.9% | "63.9% of historical yew is gone" |
| L:R ratio        | `lost / curr`                               | 1.77  | "1.77× as much was Lost as Remains" |

They convey the same information from different angles:
- **Decline %** answers: "What fraction of the original population is gone?"
- **L:R** answers: "How does the amount lost compare to what survives?"

The mathematical relationship:
```
L:R = decline / (1 − decline)

If decline = 63.9% = 0.639:
L:R = 0.639 / 0.361 = 1.77
```

---

## 6. Latitude Zone Aggregation

Tiles are grouped by latitude into three zones:

| Zone    | Latitude range | Description |
|---------|---------------|-------------|
| South   | 48–50°N       | Vancouver Island + Sunshine Coast |
| Central | 50–52°N       | Northern VI + mainland fjords |
| North   | 52–56°N       | Central + north coast |

For each zone, the computation is identical to the province-wide aggregation
but using only the tiles within that latitude band:

```python
z_curr = Σ t['curr_total']    for tiles where lat_lo ≤ lat < lat_hi
z_hist = Σ t['hist_total']    for tiles where lat_lo ≤ lat < lat_hi
z_for  = Σ t['forest_px']     for tiles in zone
z_log  = Σ t['logged_px']     for tiles in zone

z_forested     = z_for + z_log
z_decline      = (1 − z_curr / z_hist) × 100
z_curr_density = (z_curr / z_forested) × 100
z_hist_density = (z_hist / z_forested) × 100
z_loss_ratio   = (z_hist − z_curr) / z_curr
```

**Results (P ≥ 0.50, RF classifier):**

| Zone    | Sites | Historical density | Current density | L:R   | Decline |
|---------|-------|-------------------|-----------------|-------|---------|
| South   | 14    | 8.774%            | 3.038%          | 1.89  | 65.4%   |
| Central | 12    | 0.954%            | 0.551%          | 0.73  | 42.3%   |
| North   | 9     | 0.008%            | 0.006%          | 0.25  | 20.0%   |

---

## 7. CWH Zone Extrapolation

The 35 tiles (555,545 ha) represent ~1.5% of the CWH biogeoclimatic zone
(3,595,194 ha). To extrapolate:

```
sample_ha = total pixels across all tiles / 100  (100 px/ha at 10 m)
expansion = CWH_AREA_HA / sample_ha

cwh_current_ha    = curr_total_px / 100 × expansion
cwh_historical_ha = hist_total_px / 100 × expansion
cwh_lost_ha       = lost_px / 100 × expansion
```

This assumes the 35 tiles are broadly representative of the CWH zone. The tiles
were selected to cover the geographic range of the zone from south (Port
Renfrew, 48.5°N) to north (Bears, 56.1°N), with varying logging intensity.

---

## 8. Worked Example — Single Tile (Nanaimo Lakes)

```
Nanaimo Lakes — 49.0°N, -124.2°W   (RF classifier, P ≥ 0.50)

Land use breakdown:
  Mature forest:   2,296 ha      (22,960 px)   — from VRI category 5
  Logged:         12,612 ha     (126,120 px)   ← heavily logged
  Total forested: 14,908 ha

Current yew in mature forest:
  p50_forest = pixels where RF P ≥ 0.50 in forest = 317 (frac_yew = 1.38%)

Historical yew in logged area:
  hist_in_logged = 126,120 × 0.0138 = 1,740 pixels

Totals:
  hist_total = 317 + 1,740 = 2,063 pixels (206 ha)
  curr_total = 317 pixels (23 ha)
  lost       = 1,740 pixels (184 ha)

Metrics:
  decline = (1 − 23 / 206) × 100 = 89.0%
  L:R     = 184 / 23 = 8.05

  curr_density = 23 / 14,908 × 100 = 0.153%
  hist_density = 206 / 14,908 × 100 = 1.385%

  → Most heavily impacted tile: 89% decline, L:R of 8.05:1
  → Note: decline % and L:R are identical to MLP results because
    both depend only on forest:logged area ratio and local yew rate;
    only the absolute hectare figures change with the classifier.
```

---

## 9. Key Assumptions and Limitations

1. **Local control assumption:** Logged areas within a tile had similar yew
   density to the surviving mature forest in the same tile. This is broadly valid
   at the 10 km scale but breaks down if logging was systematically targeted at
   high-density yew stands.

2. **Logged = 0 yew:** Yew does not survive in logged forests. Per the user:
   "Yew doesn't grow in cut forests." VRI categories 2, 3, and 4 (logged <80 yr)
   are set to zero yew probability.

3. **P ≥ 0.50 threshold:** The natural RF decision boundary (majority vote
   among 500 trees). The previous MLP used P ≥ 0.95 because its sigmoid
   output concentrated near 1.0. The RF's tree-vote fractions rarely exceed
   0.85, so a high threshold would discard nearly all signal. The `--threshold`
   flag allows sensitivity analysis at other values.

4. **Model training bias:** The RF classifier was trained primarily on southern
   Vancouver Island data. It may underestimate yew presence in central/northern
   BC where different subspecies or growth forms may occur.

5. **Sample representativeness:** 35 tiles cover ~1.5% of the CWH zone. While
   they span the full latitude range, they may not perfectly represent the
   spatial heterogeneity of logging and yew density.

6. **VRI vintage:** The VRI categories are from the 2024 Vegetation Resources
   Inventory. Logging history is binned into age classes (<20 yr, 20–40 yr,
   40–80 yr) rather than exact years.

7. **Water exclusion:** VRI category 1 (water/non-forest) is excluded from all
   calculations. This means neither the "forested" nor "yew" counts include
   water pixels. Stream riparian areas that might support yew but are classified
   as water are not counted.

---

## 10. Density vs. Area Metrics

| Metric | What it measures | Use case |
|--------|-----------------|----------|
| **Yew area (ha)** | Total hectares of P ≥ 0.95 yew | Absolute scale of impact |
| **Decline %** | Fraction of historical area lost | Headline severity metric |
| **Density (ha/100 ha)** | Yew per unit of forested land | Normalizes for tile size and land composition |
| **L:R ratio** | Lost / remaining | Severity relative to what survives |

Density is the most appropriate metric for comparing across zones or tiles of
different sizes, because it controls for the amount of forested land available.
A tile with 1,000 ha of forest and 50 ha of yew (5.0% density) is directly
comparable to a tile with 10,000 ha of forest and 500 ha of yew (also 5.0%).
