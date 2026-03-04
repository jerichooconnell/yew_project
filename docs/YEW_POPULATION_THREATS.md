# Secondary Threats to Pacific Yew (*Taxus brevifolia*) Populations

While clear-cut logging and historical overharvesting (for Taxol production) represent the most massive drivers of Pacific yew population decline, the species faces several critical compounding threats. As a slow-growing, highly shade-tolerant, late-successional understory tree, the Pacific yew is particularly vulnerable to localized environmental disruptions.

Below is a summary of the four secondary threats impacting modern Pacific yew populations:

---

## 1. Stream Erosion and Riparian Habitat Loss
**Mechanism:** Pacific yew heavily favors moist habitats; it is frequently found growing scattered along stream banks, river margins, canyon bottoms, and wet shaded ravines. This makes its physical root systems and microclimates highly dependent on stable riparian zones.
**Impact:** 
- **Hydrological Alteration:** Upstream clear-cutting and climate-change-driven extreme precipitation events lead to higher peak flows ("flashier" streams). This increases sheer stress on stream banks, physically washing away the soil supporting mature yews.
- **Microclimate Shift:** Because yews are highly sensitive to sudden sun exposure and drying, the erosion of surrounding stream-side vegetation destroys the moist, shaded microclimate they require to survive, subjecting them to lethal desiccation and blowdown. 

**Quantitative Modeling Parameters:**
*   **Spatial Buffer:** Apply a 15–30 meter "riparian risk buffer" to all stream vectors adjacent to upstream clear-cuts.
*   **Mortality Rate:** Assign a **20% to 40% immediate mortality penalty** to mature yew trees falling within this buffer due to windthrow (blowdown) and desiccation within the first 5 years post-logging.
*   **Habitat Proportion:** Assume roughly 10%–15% of all coastal yew populations exist strictly within these highly vulnerable riparian ribbons.

## 2. Rising Sea Levels and Saltwater Intrusion
**Mechanism:** Many key sub-populations of Pacific yew—particularly in coastal British Columbia, Vancouver Island, Puget Sound, and southeastern Alaska—grow at very low elevations along coastal fjords and inlets where moist marine air stabilizes their environment.
**Impact:**
- **Inundation & Salinity:** Sea-level rise threatens these near-shore habitats directly through coastal erosion and saltwater intrusion into the groundwater table. 
- **Root Toxicity:** Pacific yews are not salt-tolerant. As the tidal extent pushes further up coastal river mouths and into low-lying coastal contiguous forests, saline groundwater can kill the roots of old-growth yews, functionally shrinking the viable western edge of the species' range.

**Quantitative Modeling Parameters:**
*   **Elevation Threshold:** Apply **100% habitat loss/population mortality** to any yew raster grids located at `< 1.0 to 1.5 meters` above current mean sea level moving into the 50-year future projection.
*   **Salinity Buffer:** Expand the coastal inundation zone inland by **15–20 meters** past the high-tide line to account for lethal saltwater intrusion into the water table killing long-established root networks.

## 3. The Yew Big Bud Mite (*Cecidophyopsis psilaspis*)
**Mechanism:** The yew big bud mite is a microscopic Eriophyid mite (often considered an invasive pest) that causes the formation of bud galls on *Taxus* species. In places like coastal British Columbia, it has become ubiquitous and highly damaging.
**Impact:**
- **Bud Destruction:** The mites migrate into the buds to feed and reproduce, causing terminal and lateral buds to "blast" or gall rather than open. 
- **Reproductive & Structural Stunting:** This pest heavily suppresses the tree's already slow growth rate. Studies show that terminal bud mortality can average over 20% in infested areas. Because the mites colonize both vegetative and reproductive buds, severe infestations stop saplings from growing and mature trees from producing the fleshy arils (seeds) required for population regeneration.

**Quantitative Modeling Parameters:**
*   **Terminal Bud Mortality:** Parameterize a hard **20%–25% bud mortality rate** in modeled coastal sub-populations.
*   **Growth Penalty:** Decrease the modeled annual growth vector (height and diameter) by **20%** to reflect the structural loss of terminal leaders.
*   **Fecundity Drag:** Reduce the seed recruitment rate (aril generation) by **25%** to accurately model the impact of reproductive bud blasting on the next generation.

## 4. Ungulate Browsing (Moose, Elk, and Deer)
**Mechanism:** Pacific yew contains toxic alkaloids (taxines) that are dangerous to humans and livestock (like horses and cattle). However, wild ungulates—specifically moose, elk, white-tailed deer, and black-tailed deer—have adapted to digest it and preferentially browse the foliage.
**Impact:**
- **Winter Forage:** During harsh winters when standard browse is covered by snow or unavailable, ungulates rely heavily on the evergreen foliage of the Pacific shrub layer. 
- **Regeneration Failure:** Browsing pressure is considered a primary barrier to Pacific yew natural regeneration. Deer will strip seeds, seedlings, and low branches. Moose have been observed not only eating the foliage but heavily damaging the bark and breaking main stems. Where ungulate populations are unnaturally high (due to predator suppression or nearby logging clearings), yew saplings are relentlessly browsed down, preventing any new generations from reaching the mid-canopy.

**Quantitative Modeling Parameters:**
*   **Seedling Mortality / Stunting:** Apply a **60%–80% mortality or severe stunting rate** to the seedling/sapling class (< 2 meters tall) in areas modeled with high ungulate densities (e.g., >10 deer per square kilometer).
*   **Regeneration Bottleneck:** In matrix population models, heavily throttle the transition probability from "seedling" to "established understory tree" by **at least 70%** mimicking the browsing ceiling. This realistically forces the model to age the existing mature population without allowing replacement.

## 5. Increased Wildfire Frequency (Coastal Western Hemlock Zone)
**Mechanism:** Historically, the Coastal Western Hemlock (CWH) biogeoclimatic zone experienced an incredibly long fire return interval—averaging **250 to 350 years** in drier maritime subzones, and up to **1,000+ years** in hyper-maritime coastal rainforests. This stable, wet, old-growth environment is what allowed the extremely slow-growing Pacific yew to survive in the understory. 
**Impact:** 
- **Acute Vulnerability:** Pacific yew has exceptionally thin, flaky bark and shallow roots. It has absolutely no fire tolerance. Botanical studies show that even low-intensity ground fires result in immediate heat damage and cambium death, essentially eliminating the species from burned stands entirely.
- **Climate Shift:** Climate change—bringing hotter, drier summers and diminished snowpack to the Pacific Northwest—is drastically shrinking this fire cycle. As droughts deepen in the CWH zone, fires are becoming more frequent and severe. Because Pacific yews can take over a century to reach a reproductive, mature size, a shortened fire return interval prevents the population from ever fully regenerating before the next stand-replacing fire sweeps through.

**Quantitative Modeling Parameters:**
*   **Fire Mortality:** Apply a strict **100% mortality rate** to all yew age classes (seedling to mature) within the footprint of any simulated or historical fire polygon. They do not survive ground fires.
*   **Return Interval Shift:** If dynamically modeling future habitat viability in the CWH zone under standard climate change projections (e.g., RCP 4.5 or 8.5), reduce the historical fire return interval parameter from ~250 years down to **80–120 years** in the drier maritime coastal subzones. 
*   **Reproductive Viability:** Hardcode a logic rule: if a raster cell experiences fire, the required time for the yew population inside that cell to return to a fully mature, seed-casting state is a minimum of **80 to 100 years** post-fire.