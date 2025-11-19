# Image Review Workflow

## Purpose
Manually review satellite images to ensure they align with iNaturalist observations and identify quality issues.

## Tools Created

### 1. Review Tool (`scripts/visualization/review_inat_images.py`)
Interactive viewer to review each satellite image alongside iNaturalist data.

**Features:**
- Side-by-side RGB and false color satellite images
- iNaturalist observation details (date, location, user, description)
- Direct link to open observation on iNaturalist website
- Quality checks (GPS accuracy, spatial variation)
- Keyboard shortcuts for fast review

**Usage:**
```bash
python scripts/visualization/review_inat_images.py
```

**Controls:**
- **Arrow Keys** or **Buttons**: Navigate between images
- **Y**: Mark as Good ✓
- **N**: Mark as Bad ✗
- **U**: Mark as Uncertain ?
- **S**: Skip (move to next without marking)
- **Open iNat Button**: Opens observation in web browser
- **Q** or **Save & Quit**: Save progress and exit

### 2. Analysis Tool (`scripts/analysis/analyze_review_results.py`)
Analyzes your reviews and creates filtered datasets.

**Usage:**
```bash
python scripts/analysis/analyze_review_results.py
```

**Output:**
- Summary statistics
- Three filtered CSV files:
  1. `inat_yew_filtered_good.csv` - Only "good" images
  2. `inat_yew_filtered_acceptable.csv` - Good + Uncertain
  3. `inat_yew_filtered_not_bad.csv` - Everything except bad

## Review Checklist

### Mark as GOOD ✓
- [ ] Satellite image shows forest matching iNat description
- [ ] GPS accuracy < 50m (ideally < 20m)
- [ ] Good spatial variation in image (not all one color)
- [ ] No obvious clouds, shadows, or artifacts
- [ ] Image date range (2020-2024) overlaps with observation
- [ ] Habitat looks suitable for yew (old growth, moist forest)

### Mark as BAD ✗
- [ ] Urban/suburban area (likely cultivated yew)
- [ ] Parking lot, building, or road visible at center
- [ ] Mostly water, field, or non-forest
- [ ] Heavy cloud cover or shadows
- [ ] GPS accuracy > 100m
- [ ] All pixels nearly identical (extraction error)

### Mark as UNCERTAIN ?
- [ ] Image quality marginal but usable
- [ ] GPS accuracy 50-100m (borderline)
- [ ] Habitat looks plausible but not ideal
- [ ] Minor cloud/shadow issues
- [ ] Not sure if location matches iNat description

## Review Progress

Reviews are saved to: `data/ee_imagery/image_review_results.json`

You can stop and resume at any time. Progress is automatically saved on quit.

## Workflow

### Step 1: Initial Quick Review (30-60 minutes)
Review first 50-100 images to identify common issues:

```bash
python scripts/visualization/review_inat_images.py
```

Focus on obvious problems:
- Urban locations
- Water/field images
- Severe GPS errors
- Poor image quality

### Step 2: Analyze Initial Results

```bash
python scripts/analysis/analyze_review_results.py
```

Check:
- What percentage are bad?
- Are GPS accuracy issues common?
- Do you see patterns in bad images?

### Step 3: Adjust Extraction Parameters (if needed)

If >30% are bad, consider stricter filtering:

```bash
# More selective extraction
python scripts/preprocessing/extract_inat_yew_images.py \
    --max-accuracy 20 \    # Stricter GPS requirement
    --limit 200 \
    --workers 4
```

### Step 4: Complete Review

Review remaining images at your own pace. Aim for:
- At least 150 "good" or "acceptable" images for training
- Document any patterns you notice

### Step 5: Use Filtered Dataset

After review, use the filtered datasets for training:

```python
# In your training script
metadata = pd.read_csv('data/ee_imagery/image_patches_64x64/inat_yew_filtered_good.csv')
# Use only high-quality observations
```

## Common Issues to Watch For

### Location Errors
- **Symptom**: Image center is in water, field, or parking lot
- **Cause**: GPS error or manually-placed pin
- **Action**: Mark as bad

### Urban/Cultivated Yew
- **Symptom**: Buildings, manicured landscapes, park settings
- **Cause**: Planted ornamental yews, not natural populations
- **Action**: Mark as bad (unless studying cultivated yews)

### Temporal Mismatch
- **Symptom**: Very old observation (1980s-1990s) or very recent (2024-2025)
- **Cause**: Satellite imagery from 2020-2024 may not match
- **Action**: Mark as uncertain or bad

### GPS Clustering
- **Symptom**: Multiple observations at exact same coordinates
- **Cause**: Popular trail, default pin location
- **Action**: Check if habitat looks good; mark uncertain if unclear

### Poor Image Quality
- **Symptom**: Clouds, shadows, low variation
- **Cause**: Satellite data issues
- **Action**: Mark as bad

## Tips for Efficient Review

1. **Use keyboard shortcuts** (Y/N/U) instead of clicking buttons
2. **Open iNat page** for uncertain cases to see photos/comments
3. **Review in batches** of 20-30 images, take breaks
4. **Trust your judgment** - if it looks wrong, mark it bad
5. **Be consistent** - develop your own criteria and stick to it
6. **Document patterns** - note common issues in comments

## Expected Results

Based on typical iNaturalist data quality:

- **Good**: 60-70% (120-140 images)
- **Uncertain**: 15-20% (30-40 images)
- **Bad**: 10-25% (20-50 images)

If your results differ significantly:
- >40% bad → Consider stricter extraction filters
- <10% bad → You might be too lenient; look more carefully
- <50% good → May need more observations or different approach

## After Review

### Use the Results

1. **Training**: Use `inat_yew_filtered_good.csv` for CNN training
2. **Validation**: Keep some "good" images separate for testing
3. **Documentation**: Note issues found in project docs
4. **Iteration**: If too few good images, extract more with better filters

### Report Summary

After completing review, document:
- Total images reviewed
- Percentage good/bad/uncertain
- Common issues identified
- Recommended accuracy threshold
- Suggested improvements for next extraction

Save this in your project documentation!
