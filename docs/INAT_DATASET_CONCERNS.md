# Potential Issues with iNaturalist-Based Yew Detection

## 1. Location Precision Issues

### GPS Accuracy Problems
- **Urban canyon effect**: Buildings/trees can degrade GPS accuracy
- **Device limitations**: Older phones may have 10-50m errors even when reported as "accurate"
- **User error**: Some users manually place pins on maps rather than using GPS
- **Positional accuracy underestimation**: The reported accuracy may not reflect true error

### Evidence to Check:
- Observations near buildings, valleys, or dense canopy
- Observations with suspiciously round coordinates (manually placed)
- Multiple observations at identical coordinates (popular trails)

## 2. Scale Mismatch

### Individual Trees vs Satellite Pixels
- **Yew trees are small**: Often <10m crown diameter
- **Sentinel-2 pixels**: 10m resolution
- **A single yew**: May only affect 1-4 pixels in the entire 64×64 image
- **Problem**: The CNN sees mostly the surrounding forest, not the yew itself

### Comparison:
- **iNaturalist observations**: Point locations of individual trees
- **Forestry inventory**: Sample plots (often 50-400m² circles) more likely to show forest-level patterns
- **Satellite signal**: Integrates entire pixel area (100m²)

## 3. Habitat vs Individual Detection

### What the Model Might Learn:
- **Yew habitat characteristics** (old growth forests, moist sites)
- **NOT individual yew trees** (too small to see)
- **Risk**: Model learns "places where yew could be" not "yew presence"

### Test:
If the model works, it's detecting habitat suitability, not actual yew trees

## 4. Data Quality Issues

### iNaturalist Observation Quality
- **Obscured coordinates**: Some observations have deliberately fuzzy locations (endangered species)
- **Cultivated specimens**: Parks, gardens, planted trees (not natural populations)
- **Misidentifications**: Even "research grade" can have errors
- **Temporal mismatch**: Observations from 1980-2025, but satellite images from 2020-2024

### Forestry Inventory Issues
- **Non-yew sites may have yew**: The inventory might have missed small/understory yew
- **Coarse species classification**: "No yew recorded" ≠ "definitely no yew present"
- **Different detection methods**: Professional inventory vs citizen science

## 5. Class Imbalance Still Present

### Current Dataset
- **200 yew** : **100 non-yew** = 2:1 ratio
- Better than 61:5698 (0.01:1) from forestry data
- But still fewer non-yew examples than might be ideal
- Model could still overpredict yew

## 6. Geographic Clustering

### Observation Bias
- **iNaturalist hotspots**: Popular trails, urban parks, accessible areas
- **Forestry plots**: Systematic sampling, includes remote areas
- **Risk**: Model learns "popular hiking areas" instead of yew habitat

### Check:
- Are iNat observations clustered near roads/trails?
- Do forestry non-yew plots sample different geographic areas?

## 7. Image Quality Issues

### Satellite Data Problems
- **Cloud cover**: Even with 30% threshold, some haze/shadows
- **Seasonal variation**: 2020-2024 composite may mix seasons
- **Terrain shadows**: Mountains cast shadows, affect spectral values
- **No spatial variation**: Some pixels still returning constant values

## 8. Fundamental Detectability

### The Core Challenge
Pacific Yew characteristics that make satellite detection difficult:
- **Understory species**: Hidden under canopy (30-80% of observations)
- **Small size**: 5-15m height, <10m crown diameter
- **Scattered distribution**: Rarely forms pure stands
- **No unique spectral signature**: Looks like other conifers/understory
- **Low density**: 1-10 trees per hectare when present

### Reality Check:
If professional foresters struggle to find yew on the ground, can satellites see it from space?

## 9. Model Learning Wrong Patterns

### What the CNN Might Actually Learn:
1. **Elevation patterns**: iNat observations at lower/accessible elevations
2. **Road proximity**: Citizen scientists near roads
3. **Forest maturity**: Old growth more photogenic → more observations
4. **Regional biases**: More observations near population centers
5. **Terrain accessibility**: Flat areas over steep slopes

### None of these are "yew presence"

## 10. Validation Challenges

### How to Know if it Works?
- **No ground truth**: Can't verify model predictions without field visits
- **Circular logic**: If model "finds" yew, how do you confirm it's real?
- **Precision vs Recall tradeoff**: High recall (finding many sites) = low precision (many false positives)

## Recommendations for Your Review

### During Manual Review, Flag:
1. ✗ **Urban/suburban locations** (cultivated yew likely)
2. ✗ **Parking lots/trailheads in image** (GPS error)
3. ✗ **Obvious cloud/shadow issues** (poor image quality)
4. ✗ **All-water or all-field images** (location error)
5. ✗ **Very recent observations** (2024-2025) with 2020-2024 imagery (tree may not be there)
6. ✗ **Suspicious clustering** (multiple obs at exact same spot)
7. ✓ **Clear forest images** that match iNat photo/description
8. ✓ **Reasonable habitat** (old growth, moist forest)

### Questions to Ask:
- Does the satellite image show the habitat described in iNaturalist?
- If the observation says "old growth forest," do you see that?
- If it says "trailside," can you see a trail?
- Does the forest in the image look like yew habitat?

## Best Case Scenario

Even if everything works perfectly, the model is detecting:
- **Yew-suitable habitat** visible from satellites
- NOT individual yew trees

This could still be useful for:
- Prioritizing field survey areas
- Identifying high-probability zones
- Reducing search area for ground crews

## Worst Case Scenario

The model learns spurious correlations:
- "Easy hiking access" → labeled as yew
- "Mature forest near cities" → labeled as yew
- Actually predicting observation bias, not yew presence

Your manual review will help identify these issues early!
