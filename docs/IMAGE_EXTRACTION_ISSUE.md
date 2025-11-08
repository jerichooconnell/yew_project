# Image Extraction Issue - Resolved

## Problem

The extracted 64x64 pixel images from Earth Engine had **no spatial variation** - all pixels in each band had the same value. This made the images appear as single-colored squares instead of actual satellite imagery.

### Root Cause

The `sampleRectangle()` method with a buffered point region was returning aggregated (mean) values rather than individual pixel arrays. This happened because:

1. Using `point.buffer().bounds()` created an imprecise region
2. The sampling didn't properly preserve pixel-level detail
3. No validation was done to check for spatial variation

### Evidence

```python
# Image 1: 1231676
Blue band:  Min: 1842.00, Max: 1842.00, Std: 0.00, Unique values: 1
Green band: Min: 1536.00, Max: 1536.00, Std: 0.00, Unique values: 1
Red band:   Min: 1389.00, Max: 1389.00, Std: 0.00, Unique values: 1
NIR band:   Min: 2333.50, Max: 2333.50, Std: 0.00, Unique values: 1
```

All 4,096 pixels (64x64) per band had identical values!

## Solution

Updated `extract_sentinel2_patch()` function in `scripts/preprocessing/extract_ee_image_patches.py`:

### Changes Made:

1. **Precise Region Definition**: Use `ee.Geometry.Rectangle()` with exact coordinates instead of buffered point
   ```python
   region = ee.Geometry.Rectangle([lon - half_size_deg, lat - half_size_deg,
                                   lon + half_size_deg, lat + half_size_deg])
   ```

2. **Validation Check**: Reject images with no spatial variation
   ```python
   if (blue.std() == 0 and green.std() == 0 and 
       red.std() == 0 and nir.std() == 0):
       return None  # Constant values - extraction failed
   ```

3. **Improved Resizing**: Handle non-square arrays properly
   ```python
   zoom_factor_y = patch_size / blue.shape[0]
   zoom_factor_x = patch_size / blue.shape[1]
   blue = zoom(blue, (zoom_factor_y, zoom_factor_x), order=1)
   ```

## Next Steps

### To Fix Existing Images:

1. **Delete** the current extracted images (they're unusable):
   ```bash
   rm -rf data/ee_imagery/image_patches_64x64/yew/*
   rm -rf data/ee_imagery/image_patches_64x64/no_yew/*
   rm data/ee_imagery/image_patches_64x64/image_metadata.csv
   rm data/ee_imagery/image_patches_64x64/extraction_progress.json
   ```

2. **Re-run extraction** with the fixed script:
   ```bash
   python scripts/preprocessing/extract_ee_image_patches.py
   ```

3. **Verify** the new images have spatial variation:
   ```python
   import numpy as np
   img = np.load('data/ee_imagery/image_patches_64x64/no_yew/SITE_ID.npy')
   print(f"Blue std: {img[0].std():.2f}")  # Should be > 0
   print(f"Unique values: {len(np.unique(img[0]))}")  # Should be > 1
   ```

### Expected Results:

- **Before**: All pixels same value, std = 0, unique_values = 1
- **After**: Spatial variation, std > 0, unique_values > 100

The new extraction will:
- Automatically skip sites with no spatial variation
- Produce valid 64x64 pixel patches with real satellite data
- Be suitable for CNN training

## Alternative Solution (if issues persist)

If `sampleRectangle()` continues to fail, we can switch to using Earth Engine's thumbnail API:

```python
url = composite.getThumbURL({
    'region': region,
    'dimensions': [patch_size, patch_size],
    'format': 'npy'
})
# Download from URL and save
```

This is slower but more reliable for getting actual pixel arrays.
