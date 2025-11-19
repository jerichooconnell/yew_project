# Misclassification Analysis & City Filtering

## Overview

This document describes two new tools for improving model quality:

1. **Misclassification Visualization** - Identify images the model struggles with most
2. **City Filtering** - Pre-filter observations near urban areas

## 1. Misclassification Visualization

### Purpose

The visualization tool helps you understand model weaknesses by showing:
- **False Positives**: Non-yew sites the model confidently predicted as yew
- **False Negatives**: Yew sites the model confidently missed

This is crucial for:
- Identifying problematic training samples
- Understanding what visual features confuse the model
- Guiding data collection priorities
- Improving model performance through targeted fixes

### Usage

```bash
# Analyze validation set with filtered data
python scripts/visualization/visualize_misclassifications.py \
    --model models/checkpoints/resnet18_20251114_125300_best.pth \
    --data-split val \
    --use-filtered \
    --top-k 15

# Analyze training set
python scripts/visualization/visualize_misclassifications.py \
    --model models/checkpoints/best_model.pth \
    --data-split train \
    --top-k 20
```

### Arguments

- `--model`: Path to trained model checkpoint (.pth)
- `--data-split`: Which split to analyze (train/val)
- `--use-filtered`: Use filtered dataset splits (if available)
- `--top-k`: Number of top misclassifications to show (default: 20)
- `--batch-size`: Batch size for inference (default: 16)
- `--output-dir`: Output directory (default: results/misclassifications)

### Output Files

1. **false_positives.png**
   - Grid of non-yew images the model predicted as yew
   - Sorted by confidence (most confident mistakes first)
   - Shows RGB composite and location coordinates
   - Helps identify visual patterns that confuse the model

2. **false_negatives.png**
   - Grid of yew images the model missed
   - Sorted by confidence
   - Includes iNaturalist observation IDs for verification
   - Helps identify poor quality samples or edge cases

3. **misclassification_summary.txt**
   - Detailed text report with:
     - Confidence scores
     - Probability distributions
     - File paths
     - Geographic coordinates
     - iNaturalist URLs (for yew samples)

4. **misclassifications.json**
   - Machine-readable JSON format
   - For programmatic analysis and filtering

### Current Results (Filtered Model)

**Model**: ResNet18, trained on filtered dataset (106 yew + 100 non-yew)
**Validation Accuracy**: 75.6% (31/41 correct)

**False Positives**: 5 samples
- Most confident: 95.5% (no_yew/6006713.npy)
- These non-yew sites have visual features similar to yew habitat

**False Negatives**: 5 samples
- Most confident: 97.3% (inat_yew/inat_3831463.npy)
- Check these observations - might be poor quality or mislabeled

### How to Use Results

1. **Review False Positives**
   - Open false_positives.png
   - Look for common patterns (e.g., forest type, elevation, terrain)
   - Consider if these sites should be excluded from training
   - May indicate need for more diverse non-yew samples

2. **Review False Negatives**
   - Open false_negatives.png
   - Check iNaturalist URLs to verify observations
   - Look for image quality issues (clouds, shadows, position errors)
   - Mark these samples as "bad" in review tool if problematic

3. **Update Training Data**
   - Exclude problematic samples identified in analysis
   - Re-train model with cleaner dataset
   - Run visualization again to verify improvement

## 2. City Filtering

### Purpose

Urban and suburban areas can confound the model because:
- **Different vegetation patterns** - Parks, gardens, ornamental plantings
- **Human infrastructure** - Buildings, roads, parking lots visible in imagery
- **Geographic bias** - Cities not representative of natural yew habitat
- **Data quality** - GPS errors more common in urban areas

City filtering automatically excludes observations within a defined radius of major Northwest cities.

### Included Cities

**Washington State:**
- Seattle (30km radius), Tacoma (20km), Spokane (20km), Bellingham (15km)
- Olympia (15km), Vancouver (15km), Everett (15km), and others

**Oregon:**
- Portland (30km), Eugene (20km), Salem (15km), Medford (15km)
- Bend (15km), Corvallis (10km), Gresham (10km)

**British Columbia:**
- Vancouver (30km), Victoria (20km), Surrey (15km), Burnaby (15km)
- Richmond (15km), Nanaimo (15km), Kelowna (15km), Kamloops (15km)
- Prince George (20km), and others

**Total**: 28 major cities with city-specific exclusion radii

### Usage

#### Standalone Testing

Test city filtering on existing data:

```bash
# Test on yew observations
python scripts/preprocessing/city_filter.py \
    --csv data/ee_imagery/image_patches_64x64/inat_yew_image_metadata.csv \
    --lat-col lat \
    --lon-col lon

# Test with custom minimum distance (override city-specific radii)
python scripts/preprocessing/city_filter.py \
    --csv data.csv \
    --lat-col latitude \
    --lon-col longitude \
    --min-distance 25 \
    --output filtered_data.csv
```

#### Integrated with Extraction

Apply city filtering during image extraction:

```bash
# Extract yew images with city filter
python scripts/preprocessing/extract_more_yew_bc_wa.py \
    --limit 200 \
    --max-accuracy 50 \
    --workers 6 \
    --filter-cities

# Extract with custom minimum distance
python scripts/preprocessing/extract_more_yew_bc_wa.py \
    --limit 200 \
    --filter-cities \
    --min-city-distance 20

# Extract non-yew images with city filter
python scripts/preprocessing/extract_no_yew_images.py \
    --limit 150 \
    --workers 6 \
    --filter-cities
```

### Current Dataset Impact

Applied to 323 BC/WA yew observations:
- **Excluded**: 126 observations (39%)
- **Retained**: 197 observations (61%)

Top exclusions:
- Victoria, BC: 46 samples
- Seattle, WA: 26 samples
- Portland, OR: 14 samples
- Vancouver, BC: 7 samples

### City-Specific Radii

City radii were chosen based on:
- Metropolitan area size
- Urban sprawl patterns
- Natural habitat boundaries

Large cities (30km): Seattle, Portland, Vancouver (BC)
Medium cities (15-20km): Victoria, Eugene, Spokane, Tacoma
Smaller cities (10-15km): Bellingham, Olympia, Bend, etc.

You can override with `--min-city-distance` for uniform filtering.

### Technical Implementation

**Distance Calculation**: Haversine formula for great circle distance
- Accounts for Earth's curvature
- Accurate for regional scales (<1000km)
- Returns distance in kilometers

**Filtering Process**:
1. Calculate distance from observation to all cities
2. Check if within any city's exclusion radius
3. Record nearest city and distance for statistics
4. Exclude if within radius, retain otherwise

**Performance**: Very fast, processes 1000s of observations per second

## Complete Workflow

### Step 1: Train Initial Model

```bash
python scripts/training/train_cnn.py \
    --architecture resnet18 \
    --epochs 50 \
    --batch-size 16
```

### Step 2: Analyze Misclassifications

```bash
python scripts/visualization/visualize_misclassifications.py \
    --model models/checkpoints/resnet18_best.pth \
    --data-split val \
    --top-k 20
```

### Step 3: Review Problematic Samples

1. Open `results/misclassifications/false_positives.png`
2. Open `results/misclassifications/false_negatives.png`
3. Check iNaturalist URLs for false negatives
4. Identify samples to exclude

### Step 4: Extract More Data with City Filter

```bash
# Get more rural/natural yew samples
python scripts/preprocessing/extract_more_yew_bc_wa.py \
    --limit 200 \
    --max-accuracy 50 \
    --workers 6 \
    --filter-cities

# Get more rural/natural non-yew samples
python scripts/preprocessing/extract_no_yew_images.py \
    --limit 200 \
    --workers 6 \
    --filter-cities
```

### Step 5: Review All Samples

```bash
# Review all samples (yew and non-yew)
python scripts/visualization/review_all_images.py --dataset all
```

### Step 6: Create Filtered Splits

```bash
python scripts/preprocessing/create_filtered_splits.py
```

### Step 7: Retrain with Better Data

```bash
python scripts/training/train_cnn.py \
    --use-filtered \
    --architecture resnet18 \
    --epochs 50 \
    --batch-size 16
```

### Step 8: Re-analyze Improvements

```bash
python scripts/visualization/visualize_misclassifications.py \
    --model models/checkpoints/resnet18_new_best.pth \
    --data-split val \
    --use-filtered \
    --top-k 20
```

## Expected Improvements

### From Misclassification Analysis
- **Identify weak samples**: 10-20% of data may be low quality
- **Understand failure modes**: Visual patterns that confuse the model
- **Targeted data collection**: Focus on underrepresented scenarios
- **Validation**: Verify iNaturalist observations are correct

### From City Filtering
- **Reduce geographic bias**: Focus on natural habitat, not urban parks
- **Improve generalization**: Model learns natural patterns, not human artifacts
- **Cleaner visual features**: Fewer buildings/roads in imagery
- **Better GPS accuracy**: Rural observations typically more accurate

### Combined Impact
- **Estimated accuracy gain**: 5-10 percentage points
- **More robust model**: Better performance on new locations
- **Clearer decision boundaries**: Less confounding from urban features
- **Higher confidence**: Model more certain on true positives

## Files Reference

### Scripts
- `scripts/visualization/visualize_misclassifications.py` - Main visualization tool
- `scripts/preprocessing/city_filter.py` - City filtering module
- `scripts/preprocessing/extract_more_yew_bc_wa.py` - Updated with `--filter-cities`
- `scripts/preprocessing/extract_no_yew_images.py` - Updated with `--filter-cities`

### Output Directories
- `results/misclassifications/` - Visualization results
- `results/figures/` - General training visualizations
- `data/processed/` - Filtered splits

## Troubleshooting

### Misclassification Tool

**Issue**: "Split file not found"
- **Solution**: Run training first to generate splits, or check `--use-filtered` flag

**Issue**: "Model checkpoint not found"
- **Solution**: Verify model path, check `models/checkpoints/` directory

**Issue**: Too few misclassifications
- **Solution**: Model is performing well! Consider increasing `--top-k` or analyzing training set

### City Filter

**Issue**: "KeyError: 'latitude'"
- **Solution**: Check column names in CSV, use `--lat-col` and `--lon-col` arguments

**Issue**: Too many observations excluded
- **Solution**: Increase `--min-city-distance` to override city-specific radii

**Issue**: Not enough observations retained
- **Solution**: Decrease `--min-city-distance` or don't use `--filter-cities`

## Next Steps

1. **Analyze current model** - Run misclassification visualization on latest model
2. **Review problematic samples** - Check false positives/negatives visually
3. **Apply city filtering** - Re-extract data with `--filter-cities` flag
4. **Manual review** - Use review tool on all samples (yew and non-yew)
5. **Retrain** - Train new model with cleaner, larger, city-filtered dataset
6. **Compare performance** - Document improvements in accuracy/confidence
7. **Iterate** - Repeat cycle to continuously improve model

## References

- Haversine formula: https://en.wikipedia.org/wiki/Haversine_formula
- False positive/negative analysis: Standard ML evaluation technique
- Urban habitat bias: Well-documented issue in species distribution modeling
