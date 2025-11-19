# Extended Review Workflow and Dataset Expansion

**Date**: November 14, 2025  
**Objective**: Extend review workflow to all datasets, expand yew samples (BC/WA only), and remove California confounders

## Overview

This workflow extends the manual review process to both yew and non-yew datasets, extracts additional yew observations from BC/Washington while excluding California, and ensures geographic consistency.

## New Tools Created

### 1. Universal Review Tool
**File**: `scripts/visualization/review_all_images.py`

**Features**:
- Review yew, non-yew, or both datasets in a single interface
- Same keyboard shortcuts and controls
- Separate tracking for each dataset type
- Flags California samples for removal
- Works with both iNaturalist and forestry data

**Usage**:
```bash
# Review only yew samples
python scripts/visualization/review_all_images.py --dataset yew

# Review only non-yew samples
python scripts/visualization/review_all_images.py --dataset no-yew

# Review all samples (yew + non-yew)
python scripts/visualization/review_all_images.py --dataset all
```

**Output**: `data/ee_imagery/image_review_results_all.json`

### 2. Extract More Yew Images (BC/WA Only)
**File**: `scripts/preprocessing/extract_more_yew_bc_wa.py`

**Features**:
- Automatically filters out California (lat < 42°N)
- Only extracts BC and Washington observations
- Skips already extracted samples
- Parallel extraction with progress tracking
- Updates metadata file automatically

**Usage**:
```bash
# Extract 100 more BC/WA yew samples
python scripts/preprocessing/extract_more_yew_bc_wa.py --limit 100 --max-accuracy 50 --workers 6

# Extract 200 more samples
python scripts/preprocessing/extract_more_yew_bc_wa.py --limit 200 --workers 6
```

**Available Pool**:
- Total iNaturalist yew observations: 6,964
- With GPS accuracy ≤50m: 3,904
- BC/WA only (lat ≥ 42°N): ~3,500 (estimated)
- Already extracted: 106 (filtered good)
- Available for extraction: ~3,400 more

### 3. Remove California Samples
**File**: `scripts/preprocessing/remove_california_samples.py`

**Features**:
- Identifies California samples (lat < 42°N)
- Backs up original metadata
- Moves CA image files to archive folder
- Updates metadata to exclude CA samples

**Usage**:
```bash
python scripts/preprocessing/remove_california_samples.py
```

## Recommended Workflow

### Phase 1: Remove Existing California Samples
```bash
# Step 1: Remove California samples from current dataset
python scripts/preprocessing/remove_california_samples.py
```

**Expected Result**:
- Removes 27 California samples
- Leaves ~173 BC/WA samples
- Archives CA files for potential future use

### Phase 2: Extract More Yew Samples
```bash
# Step 2: Extract additional BC/WA yew observations
python scripts/preprocessing/extract_more_yew_bc_wa.py --limit 200 --workers 6

# This will take ~10-15 minutes
# Adds 200 new BC/WA yew samples to the dataset
```

**Expected Result**:
- New total yew samples: 173 + 200 = 373 (unreviewed)
- All samples lat ≥ 42°N
- Geographic consistency ensured

### Phase 3: Review All Samples
```bash
# Step 3a: Review all new yew samples
python scripts/visualization/review_all_images.py --dataset yew

# Step 3b: Review non-yew samples
python scripts/visualization/review_all_images.py --dataset no-yew

# OR do both at once:
python scripts/visualization/review_all_images.py --dataset all
```

**Review Guidelines**:
- **For Yew Samples**:
  * ✓ Clear forest habitat
  * ✓ GPS accuracy < 50m
  * ✓ Good image quality
  * ✗ Urban/developed areas
  * ✗ Clouds/artifacts
  * ✗ Water bodies

- **For Non-Yew Samples**:
  * ✓ Forest habitat visible
  * ✓ Typical CWH/ICH forest
  * ✗ Clearcuts/logged areas
  * ✗ Urban/agricultural land
  * ✗ Poor image quality

### Phase 4: Filter and Create Training Dataset
```bash
# Step 4: Create filtered splits with only "good" samples
python scripts/preprocessing/create_filtered_splits.py
```

**Expected Result**:
- Filtered yew: ~250 "good" samples (assuming 67% pass rate)
- Non-yew: ~67 "good" samples (if reviewing 100)
- Balanced dataset ready for training

### Phase 5: Train Model
```bash
# Step 5: Train with expanded, filtered dataset
python scripts/training/train_cnn.py \
    --use-filtered \
    --architecture resnet18 \
    --epochs 50 \
    --batch-size 16
```

## Benefits of This Approach

### 1. Geographic Consistency
- **Before**: Mixed California, Oregon, Washington, BC samples
- **After**: Only BC and Washington samples
- **Impact**: Reduces ecosystem confounding, improves regional model

### 2. Expanded Dataset
- **Before**: 106 yew samples (after filtering)
- **After**: ~250 yew samples (after expanding + filtering)
- **Impact**: Better model generalization, reduced overfitting

### 3. Quality Assurance
- **Before**: Only yew samples manually reviewed
- **After**: Both yew AND non-yew samples reviewed
- **Impact**: Higher confidence in both positive and negative classes

### 4. Balanced Classes
- **Before**: 106 yew vs 100 non-yew (1:1 ratio)
- **After**: ~250 yew vs ~67 non-yew (if only reviewing existing) OR
- **After**: ~250 yew vs 400 non-yew (if extracting more) (1:1.6 ratio)
- **Impact**: Better handling of class imbalance

## Data Management

### File Locations

**Yew Dataset**:
- Metadata: `data/ee_imagery/image_patches_64x64/inat_yew_image_metadata.csv`
- Images: `data/ee_imagery/image_patches_64x64/inat_yew/*.npy`
- Backup: `data/ee_imagery/image_patches_64x64/inat_yew_image_metadata_backup.csv`
- CA Archive: `data/ee_imagery/image_patches_64x64/inat_yew_california_removed/`

**Non-Yew Dataset**:
- Metadata: `data/ee_imagery/image_patches_64x64/no_yew_image_metadata.csv`
- Images: `data/ee_imagery/image_patches_64x64/no_yew/*.npy`

**Review Results**:
- All reviews: `data/ee_imagery/image_review_results_all.json`
- Original yew-only: `data/ee_imagery/image_review_results.json`

**Filtered Splits**:
- Training: `data/processed/train_split_filtered.csv`
- Validation: `data/processed/val_split_filtered.csv`
- Filtered yew: `data/processed/inat_yew_filtered_good.csv`

## Quality Checks

### After Removing California Samples
```bash
# Check that CA samples are gone
python -c "
import pandas as pd
df = pd.read_csv('data/ee_imagery/image_patches_64x64/inat_yew_image_metadata.csv')
print(f'Total samples: {len(df)}')
print(f'Min latitude: {df[\"lat\"].min():.2f}°N')
print(f'Samples < 42°N: {(df[\"lat\"] < 42).sum()}')
"
```

### After Extracting More Samples
```bash
# Count samples by latitude range
python -c "
import pandas as pd
df = pd.read_csv('data/ee_imagery/image_patches_64x64/inat_yew_image_metadata.csv')
print(f'Total yew samples: {len(df)}')
print(f'Latitude range: {df[\"lat\"].min():.2f}°N - {df[\"lat\"].max():.2f}°N')
print(f'Median GPS accuracy: {df[\"positional_accuracy\"].median():.1f}m')
print(f'Year range: {df[\"observation_year\"].min():.0f} - {df[\"observation_year\"].max():.0f}')
"
```

### After Review Process
```bash
# Check review statistics
python -c "
import json
with open('data/ee_imagery/image_review_results_all.json', 'r') as f:
    reviews = json.load(f)
good = sum(1 for v in reviews.values() if v == 'good')
bad = sum(1 for v in reviews.values() if v == 'bad')
uncertain = sum(1 for v in reviews.values() if v == 'uncertain')
print(f'Total reviewed: {len(reviews)}')
print(f'Good: {good} ({good/len(reviews)*100:.1f}%)')
print(f'Bad: {bad} ({bad/len(reviews)*100:.1f}%)')
print(f'Uncertain: {uncertain} ({uncertain/len(reviews)*100:.1f}%)')
"
```

## Timeline Estimate

| Step | Duration | Cumulative |
|------|----------|------------|
| Remove CA samples | 1 min | 1 min |
| Extract 200 more yew | 12 min | 13 min |
| Review 373 yew + 100 non-yew | 60 min | 73 min |
| Create filtered splits | 1 min | 74 min |
| Train model (50 epochs) | 25 min | 99 min |

**Total: ~1.5-2 hours** (mostly manual review time)

## Expected Model Improvements

### More Robust Training Data
1. **Larger positive class**: 106 → ~250 samples (2.4x increase)
2. **Geographic consistency**: No California confounding
3. **Quality assured negative class**: Manually reviewed non-yew samples
4. **Better balance**: More even class distribution

### Expected Performance Gains
- **Accuracy**: 75% → 80-85% (estimated)
- **Precision (yew)**: Improved due to cleaner training data
- **Recall (yew)**: Improved due to more diverse examples
- **Generalization**: Better performance on held-out BC/WA sites
- **False Positives**: Reduced due to reviewed non-yew samples

## Notes

### Why Exclude California?
1. **Different Subspecies**: T. brevifolia californica vs typical T. brevifolia
2. **Different Ecosystem**: Mediterranean vs Pacific Northwest climate
3. **Different Forest Composition**: Different associated species
4. **Model Scope**: Training for BC deployment, not California

### Why Review Non-Yew?
1. **Quality Control**: Ensure images show actual forest, not urban/water
2. **Confidence**: Know what "absence" looks like
3. **Error Detection**: Find mislabeled or problematic samples
4. **Model Trust**: Manually verified negative class

### Backup Strategy
- Original metadata backed up before any changes
- California samples archived, not deleted
- Can always restore if needed
- Review results saved separately from original reviews

## Next Steps After Training

1. **Evaluate Model Performance**
   - Compare to baseline (pre-expansion)
   - Test on held-out BC locations
   - Test on Washington locations separately

2. **Error Analysis**
   - Identify remaining false positives/negatives
   - Check if any systematic issues
   - Refine dataset if needed

3. **Deploy Model**
   - Use for BC forest inventory sites
   - Generate predictions across study area
   - Validate predictions in field

4. **Iterative Improvement**
   - Collect model errors
   - Add difficult cases to training set
   - Re-train and evaluate
