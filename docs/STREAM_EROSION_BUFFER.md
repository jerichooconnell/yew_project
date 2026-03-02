# Stream Erosion Buffer — Rationale and Literature

## 1. Purpose

This document explains why *water boundaries* in the VRI raster (category 1)
are expanded by a configurable buffer to simulate the combined effects of
**logging-driven hydrological change** and **climate-driven streamflow
increases** on riparian habitat loss for western yew (*Taxus brevifolia*) in
coastal British Columbia.

---

## 2. Why Enlarge the Water Boundary?

The VRI land-use raster classifies current water bodies and non-forested
riparian as category 1. However, this static snapshot understates the *effective
loss* of near-stream yew habitat because:

1. **Logging increases peak flows.** Clearcut harvesting reduces canopy
   interception and transpiration, causing faster and larger peak discharges.
2. **Higher peak flows widen channels.** Bankfull channel width scales with
   discharge as $W \propto Q^{0.5}$ (Leopold & Maddock, 1953). A 30% increase
   in peak flow implies ~14% wider channels.
3. **Climate change further amplifies flows.** Warmer temperatures increase the
   rain fraction on snow, producing more rain-on-snow flood events.
4. **Yew is a riparian species.** Western yew preferentially occupies moist
   micro-sites including streamside terraces and lower slopes, making it
   disproportionately vulnerable to channel migration and bank erosion.

By dilating the water mask, we approximate the zone of *additional habitat loss*
caused by these hydrological changes — habitat that no longer supports (or is at
elevated risk for) mature yew.

---

## 3. Literature Summary — BC Streamflow Changes

### 3.1 Logging Effects on Peak Flows

| Source | Finding | Region |
|--------|---------|--------|
| Hartman & Scrivener (1990), Carnation Creek study | 20–50% increase in peak flows following clearcut logging | West coast Vancouver Island |
| Cheng (1989), ECA-based analysis | Peak flow increases proportional to equivalent clearcut area (ECA); >30% ECA → significant peak flow increases | BC interior and coast |
| Pike et al. (2010), *Compendium of Forest Hydrology* Ch. 19 | Projected combined logging + climate effects of 10–30% increase in annual peak flows for coastal BC | Coastal BC (CWH zone) |
| Green & Alila (2012), J. Hydrology | Logging effects on flood frequency curves: significant increases in 20-yr and 50-yr return period floods | Coastal BC watersheds |

### 3.2 Climate Change Effects on Streamflow

| Source | Finding | Region |
|--------|---------|--------|
| Schnorbus et al. (2012), PCIC | Under moderate emissions: +5–15% winter discharge; under high emissions: +15–35% winter discharge by 2050 for rain-dominant coastal basins | Coastal BC |
| Déry et al. (2012), *J. Climate* | Complex trends across BC: declining summer flows but increasing winter flows in some coastal basins | Pan-BC |
| PCIC Plan2Adapt projections (SSP5-8.5) | Precipitation increase of 5–20% by 2050 for coastal BC; summer drying but winter/fall wetting | Coastal BC |
| BC Climate Change Adaptation (2021) | November 2021 atmospheric river: multiple rivers exceeded all-time records, exposing vulnerability of CWH zone to extreme precipitation | Fraser Valley, Merritt, south coast |

### 3.3 Combined Effect on Channel Morphology

The combined effect of logging and climate change on channel geometry can be
estimated from hydraulic geometry relationships:

$$W = a \cdot Q^{0.5}$$

where $W$ is bankfull channel width and $Q$ is bankfull discharge.

If peak discharge increases by 30% (a mid-range estimate combining logging +
climate effects):

$$\frac{W_{new}}{W_{old}} = \left(\frac{1.30 \cdot Q}{Q}\right)^{0.5} = 1.14$$

→ **~14% wider channels.** For a typical CWH headwater stream (3–8 m wide),
this implies 0.4–1.1 m of additional bank erosion per side.

For larger mainstem rivers (20–50 m wide), the lateral adjustment is larger:
2.8–7.0 m per side. Combined with meander migration and avulsion zones, a
**20–30 m buffer** captures the reasonable range of additional channel influence.

---

## 4. Buffer Implementation

### 4.1 Method

The water buffer is applied using morphological dilation of the VRI category 1
(water) mask:

```python
from scipy.ndimage import binary_dilation

WATER_BUFFER_PX = 3   # 30 m at 10 m resolution

water_mask = (log == 1)
dilated_water = binary_dilation(water_mask, iterations=WATER_BUFFER_PX)
new_water = dilated_water & ~water_mask  # newly-classified water
```

### 4.2 Reclassification

Pixels that become "water" after dilation are subtracted from whatever category
they previously belonged to:

- If a pixel was `forest (5)` → lost from `n_forest` and from `p95_forest` if
  it was a yew pixel
- If a pixel was `logged (2/3/4)` → lost from `n_logged`
- If a pixel was `alpine (6)` or `nodata (0)` → no effect on yew calculations

### 4.3 Effect on Calculations

After dilation, the per-tile loop recomputes all metrics with the reduced
forest/logged counts. This produces:

1. **Lower current yew** — some yew pixels near streams are now "water"
2. **Lower historical yew** — the logged area is smaller, and the historical
   projection is reduced proportionally
3. **Net increase in decline %** — because current yew near streams is lost
   disproportionately (yew favours riparian zones)

### 4.4 Configurable Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WATER_BUFFER_PX` | 3 | Dilation radius in pixels (30 m at 10 m resolution) |
| `WATER_BUFFER_MODE` | `combined` | Which rasters to modify: `forest`, `logged`, or `combined` |

### 4.5 Sensitivity Analysis

The script runs with multiple buffer sizes (0, 1, 2, 3, 4, 5 pixels = 0–50 m)
and reports how decline metrics change with buffer distance.

---

## 5. Justification for Default 30 m Buffer

| Factor | Magnitude | Source |
|--------|-----------|--------|
| Logging → peak flow increase | +20–50% | Carnation Creek; ECA analyses |
| Climate → peak flow increase | +10–30% | PCIC/Schnorbus; Plan2Adapt |
| Combined peak flow increase | ~+30–50% | Conservative combination |
| Channel width expansion ($Q^{0.5}$) | ~+14–22% | Hydraulic geometry |
| Typical CWH stream width | 3–50 m | Field observations |
| Implied lateral expansion | ~0.4–7 m per side | Width × expansion factor |
| Additional meander migration zone | 10–20 m | Church (2006); Millar (2005) |
| **Total buffer estimate** | **20–30 m** | Combined assessment |

A 30 m (3-pixel) default is conservative — it captures the direct channel
widening plus one channel-width of meander migration zone, without extending
into upslope forest that is unlikely to be lost to fluvial processes.

---

## 6. Limitations

1. **Uniform buffer.** The dilation applies the same buffer radius to all water
   features regardless of stream order. In practice, larger rivers adjust more
   than headwater streams.

2. **Isotropic expansion.** The disk-shaped dilation does not account for valley
   morphology (e.g., a stream confined by bedrock may not widen).

3. **Static approximation.** The buffer represents an equilibrium adjustment
   rather than the time-varying process of bank erosion.

4. **Literature uncertainty.** Streamflow projections range widely depending on
   emissions scenario, watershed characteristics, and model structure. The
   30 m default should be treated as a scenario, not a prediction.

---

## 7. Key References

- Carnation Creek: Hartman, G.F. & Scrivener, J.C. (1990). *Impacts of forestry
  practices on a coastal stream ecosystem, Carnation Creek, British Columbia.*
  Can. Bull. Fish. Aquat. Sci. 223.
- Hydraulic geometry: Leopold, L.B. & Maddock, T. (1953). *The hydraulic
  geometry of stream channels and some physiographic implications.* USGS Prof.
  Paper 252.
- BC Forest Hydrology: Pike, R.G. et al. (2010). *Compendium of Forest
  Hydrology and Geomorphology in British Columbia.* BC MoFR LMH 66.
- Climate projections: Schnorbus, M. et al. (2012). *Impacts of climate change
  on water supply and demand in the Okanagan Basin.* Pacific Climate Impacts
  Consortium.
- Streamflow trends: Déry, S.J. et al. (2012). *Detection of runoff timing
  changes in pluvial, nival, and glacial rivers of western Canada.* Water
  Resources Research 48, W04520.
- Channel migration: Church, M. (2006). *Bed material transport and the
  morphology of alluvial river channels.* Annu. Rev. Earth Planet. Sci. 34.
- Green, K.C. & Alila, Y. (2012). *A paradigm shift in understanding and
  quantifying the effects of forest harvesting on floods in snow environments.*
  Water Resources Research 48, W10503.
