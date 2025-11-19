# Data Filtering and Enhancement Summary

**Date**: November 14, 2025  
**Objective**: Improve model training data quality by filtering confounding samples and balancing the dataset

## Changes Made

### 1. Geographic Filtering - Remove California Samples

**Rationale**: California yew observations may confound the model due to:
- Different climate conditions (Mediterranean vs Pacific Northwest)
- Different forest ecosystem composition
- Potential subspecies differences (T. brevifolia vs californica intergrades)

**Implementation**:
- Filtered iNaturalist yew observations to latitude >= 42°N
- Removes samples south of the California-Oregon border
- Focuses model on BC and Washington ecosystems

**Results**:
- Original dataset: 200 iNaturalist yew observations
- After California filter: 173 observations (27 removed)
- California samples latitude range: 37.77°N - 41.99°N

### 2. Manual Review Quality Filtering

**Rationale**: Use expert manual review to ensure high-quality training data

**Implementation**:
- Applied review results from `image_review_results.json`
- Only included samples marked as "good" (high quality, clear forest, good GPS accuracy)
- Excluded samples marked as "bad" (urban areas, poor imagery, location errors)

**Results**:
- Reviewed: 200 samples
- Marked as "good": 130 samples
- After both CA filter + quality filter: **106 high-quality yew samples**

**Final Yew Dataset**:
- 106 samples from BC and Washington
- Latitude range: 42.04°N - 53.35°N
- Longitude range: -132.28°E - -113.81°E
- Median GPS accuracy: 10.0m
- Observation years: 2012-2019

### 3. Dataset Balancing - Additional Non-Yew Samples

**Rationale**: Balance class distribution to prevent model bias

**Original State**:
- Yew: 106 samples (after filtering)
- Non-yew: 100 samples
- Imbalance: ~1:1 ratio (acceptable but could be better)

**Target**:
- Extract 300 additional non-yew samples
- New total: 400 non-yew samples
- New ratio: 106 yew : 400 non-yew (~1:3.8)

**Status**: Extraction in progress (~6% complete, ~10 minutes remaining)

**Source**: BC Forest Inventory data
- CWH and ICH biogeoclimatic zones
- Sites confirmed without yew presence
- Random sampling with different seed to avoid duplication

## Data Quality Improvements

### Quality Assurance Applied

1. **Geographic Consistency**
   - All yew samples from Pacific Northwest (BC/WA)
   - Consistent climate and ecosystem conditions
   - Removes Mediterranean climate confounding

2. **Manual Expert Review**
   - Visual inspection of satellite imagery
   - Verification against iNaturalist observations
   - Quality checks:
     * ✓ Forest habitat visible
     * ✓ Good spatial variation in imagery
     * ✓ GPS accuracy < 50m
     * ✗ Urban areas excluded
     * ✗ Clouds/shadows excluded
     * ✗ Water/fields excluded

3. **Class Balance**
   - Increased non-yew samples for better model generalization
   - Prevents overfitting to rare positive class
   - Better representation of non-yew forest types

## Training Dataset Specification

### Final Composition (when extraction completes)

**Yew (Positive Class)**:
- 106 samples
- Source: iNaturalist citizen science observations
- Geographic scope: British Columbia and Washington State
- Quality: Manually reviewed and verified "good" samples
- Biogeoclimatic zones: CWH, ICH (mixed)
- GPS accuracy: 10m median

**Non-Yew (Negative Class)**:
- 400 samples (100 existing + 300 new)
- Source: BC Forest Inventory systematic surveys
- Geographic scope: British Columbia
- Biogeoclimatic zones: CWH, ICH only
- Verified absence of yew in species composition

**Total Dataset**: 506 samples (106 yew + 400 non-yew)

### Train/Validation Split

80/20 stratified split:
- **Training**: ~405 samples (85 yew + 320 non-yew)
- **Validation**: ~101 samples (21 yew + 80 non-yew)

## File Locations

### Filtered Datasets
- Yew (filtered): `data/processed/inat_yew_filtered_good.csv`
- Non-yew (all): `data/ee_imagery/image_patches_64x64/no_yew_image_metadata.csv`

### Training Splits
- Training: `data/processed/train_split_filtered.csv`
- Validation: `data/processed/val_split_filtered.csv`

### Review Data
- Manual reviews: `data/ee_imagery/image_review_results.json`
- Original yew metadata: `data/ee_imagery/image_patches_64x64/inat_yew_image_metadata.csv`

## Scripts Created

1. **`scripts/preprocessing/filter_and_extract_dataset.py`**
   - Complete pipeline: filter CA, extract samples, create splits
   - Handles Earth Engine authentication
   - Parallel extraction with progress tracking

2. **`scripts/preprocessing/create_filtered_splits.py`**
   - Standalone script to create train/val splits
   - Uses existing filtered data
   - Quick execution (no Earth Engine needed)

3. **Updated: `scripts/preprocessing/extract_no_yew_images.py`**
   - Added `--limit` and `--workers` arguments
   - Skip already extracted samples
   - Better error handling

4. **Updated: `scripts/training/train_cnn.py`**
   - Added `--use-filtered` flag
   - Automatically uses filtered datasets when flag is set
   - Backward compatible with original data

## Next Steps

### 1. Wait for Extraction to Complete
Monitor progress:
```bash
# Check extraction progress
ls data/ee_imagery/image_patches_64x64/no_yew/*.npy | wc -l

# Should reach 400 when complete
```

### 2. Create Training Splits
Once extraction completes:
```bash
python scripts/preprocessing/create_filtered_splits.py
```

### 3. Train Model with Filtered Data
```bash
python scripts/training/train_cnn.py \
    --use-filtered \
    --architecture resnet18 \
    --epochs 50 \
    --batch-size 16 \
    --lr 0.001
```

### 4. Compare Model Performance
- Train on filtered data (current)
- Compare to original baseline model
- Expected improvements:
  * Better geographic generalization
  * Reduced false positives from California ecosystems
  * More robust to class imbalance
  * Better precision/recall balance

## Expected Impact on Model

### Improvements
1. **Geographic Consistency**: Model trained on uniform ecosystem (Pacific Northwest)
2. **Data Quality**: Only verified high-quality samples used
3. **Class Balance**: Better representation of non-yew forests (3.8:1 ratio)
4. **Generalization**: Larger, more diverse negative class

### Potential Trade-offs
- Smaller positive class (106 vs 200 samples)
- Less geographic diversity in yew samples (no California)
- Model may not generalize to California yew habitats

### Mitigation
- Data augmentation applied during training (flips, rotations)
- Pretrained ResNet backbone provides strong feature extraction
- High-quality samples more valuable than quantity
- Focused geographic scope improves regional model performance

## Validation Strategy

After training with filtered data:

1. **Internal Validation**
   - 20% holdout validation set
   - Confusion matrix, ROC-AUC, precision/recall

2. **Geographic Validation**
   - Test on held-out BC locations
   - Test on Washington locations separately
   - Verify no California leakage

3. **Comparison to Baseline**
   - Original model (200 yew, 100 non-yew)
   - Filtered model (106 yew, 400 non-yew)
   - Metrics: accuracy, AUC, F1-score

4. **Error Analysis**
   - Examine false positives/negatives
   - Identify problematic ecosystem types
   - Refine future data collection

## Timeline

- ✅ California filtering: Complete (106 samples)
- ✅ Manual review application: Complete
- ⏳ Non-yew extraction: In progress (~6% complete, ~10 min)
- ⏹️ Create training splits: Pending extraction
- ⏹️ Model training: Pending splits (~20-30 minutes)
- ⏹️ Evaluation: Pending training (~5 minutes)

**Total estimated time**: ~45 minutes from now
