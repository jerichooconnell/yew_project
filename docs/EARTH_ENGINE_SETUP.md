# Earth Engine Setup and Usage Guide

## Overview

The `extract_ee_imagery.py` script extracts satellite imagery and environmental data from Google Earth Engine for forest plot locations.

## What It Extracts

### 1. Sentinel-2 Imagery
- **RGB bands** (Blue, Green, Red) at 10m resolution
- **NIR band** (Near-Infrared) at 10m resolution  
- Growing season composite (June-August)
- Cloud-masked median values

### 2. Vegetation Indices
- **NDVI** (Normalized Difference Vegetation Index)
- **EVI** (Enhanced Vegetation Index)

### 3. Terrain Data
- **Elevation** from SRTM (30m resolution)
- **Slope** (degrees)
- **Aspect** (direction)

## Installation

### 1. Install Earth Engine API

```bash
conda activate yew_pytorch
pip install earthengine-api
```

### 2. Authenticate (First Time Only)

```bash
earthengine authenticate
```

This will:
1. Open a browser window
2. Ask you to sign in with your Google account
3. Give you an authorization code
4. Paste the code back in the terminal

### 3. Optional: Install geemap for visualization

```bash
pip install geemap
```

## Usage

### Quick Test (10 Sample Plots)

```bash
cd /home/jericho/yew_project
conda activate yew_pytorch
python scripts/preprocessing/extract_ee_imagery.py
# Choose option 1 when prompted
```

### Extract Point Data (All Plots)

```bash
python scripts/preprocessing/extract_ee_imagery.py
# Choose option 2 when prompted
```

This extracts mean values within 250m buffer around each plot.

### Extract Image Patches (For CNN)

```bash
python scripts/preprocessing/extract_ee_imagery.py
# Choose option 3 when prompted
```

This extracts 64x64 pixel image patches (640m x 640m at 10m resolution).

## Output Files

Saved to `data/ee_imagery/`:

- `sentinel2_data_YYYYMMDD_HHMMSS.csv` - Extracted data in CSV format
- `sentinel2_data_YYYYMMDD_HHMMSS.pkl` - Same data as pickle (preserves dtypes)
- `sentinel2_data_YYYYMMDD_HHMMSS_metadata.json` - Extraction metadata
- `sample_locations_map.html` - Interactive map (if geemap installed)

## Data Format

### Point Data Output

| Column | Description | Unit |
|--------|-------------|------|
| plot_id | Site identifier | - |
| x, y | BC Albers coordinates | m |
| lon, lat | WGS84 coordinates | degrees |
| year | Measurement year | - |
| blue | Sentinel-2 B2 | reflectance (0-1) |
| green | Sentinel-2 B3 | reflectance (0-1) |
| red | Sentinel-2 B4 | reflectance (0-1) |
| nir | Sentinel-2 B8 | reflectance (0-1) |
| ndvi | Normalized Difference Vegetation Index | -1 to 1 |
| evi | Enhanced Vegetation Index | -1 to 1 |
| elevation | SRTM elevation | meters |
| slope | Terrain slope | degrees |
| aspect | Terrain aspect | degrees |
| success | Extraction successful | True/False |

## Integration with Model

### Option 1: Using Point Data as Features

Add extracted features to tabular encoder:

```python
# In yew_density_model.py, update numerical features:
numerical_cols = [
    'BA_HA_LS', 'BA_HA_DS', 'STEMS_HA_LS', 'STEMS_HA_DS',
    'VHA_WSV_LS', 'VHA_NTWB_LS', 'SI_M_TLSO', 'HT_TLSO',
    'AGEB_TLSO', 'BC_ALBERS_X', 'BC_ALBERS_Y', 'MEAS_YR',
    # Add Earth Engine features:
    'ndvi', 'evi', 'elevation', 'slope', 'aspect'
]
```

### Option 2: Using Image Patches for CNN

Replace placeholder images with real Sentinel-2 patches:

```python
# Modify YewDensityDataset._generate_placeholder_image()
def _load_sentinel2_image(self, idx):
    # Load actual 4-channel image (RGB + NIR)
    image_path = self.imagery_paths[idx]
    image = np.load(image_path)  # Shape: (4, 64, 64)
    return torch.FloatTensor(image)
```

## Rate Limits and Quotas

**Earth Engine Quotas (Free Tier):**
- 5,000 requests per day
- ~32,000 plots = multiple days of extraction

**Strategies:**
1. Run in batches (script auto-saves progress every 50 plots)
2. Use `save_interval` parameter to checkpoint frequently
3. If quota exceeded, resume next day from temp files

**Resume from checkpoint:**
```bash
# Find last temp file
ls -lt data/ee_imagery/temp_extraction_*.csv | head -1

# Skip already processed plots in your code
```

## Troubleshooting

### Authentication Issues

```bash
# Re-authenticate
earthengine authenticate

# Check credentials
python -c "import ee; ee.Initialize(); print('Success!')"
```

### Quota Exceeded

Error: `"Quota exceeded"`

**Solution:**
- Wait 24 hours for quota reset
- Or upgrade to Earth Engine paid tier
- Script saves progress automatically - just restart

### Missing Data

Some plots may fail due to:
- No Sentinel-2 coverage in time window
- Excessive cloud cover
- Edge of satellite swath

**Check success rate:**
```python
import pandas as pd
df = pd.read_csv('data/ee_imagery/sentinel2_data_*.csv')
print(f"Success rate: {df['success'].mean()*100:.1f}%")
```

### Coordinate Conversion Issues

If getting errors with coordinates:
- Ensure BC Albers coordinates are valid
- Check for NaN values in BC_ALBERS_X/Y
- Verify coordinates are within BC bounds

## Advanced Usage

### Custom Date Range

Modify the extraction dates for specific seasons:

```python
# In extract_point_data(), change:
start_date = f'{year}-06-01'  # Growing season start
end_date = f'{year}-08-31'    # Growing season end

# Example: Full year
start_date = f'{year}-01-01'
end_date = f'{year}-12-31'
```

### Additional Bands

Add more Sentinel-2 bands:

```python
# In __init__:
self.sentinel_bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']  # Add SWIR
self.output_bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
```

### Different Satellite

Use Landsat instead of Sentinel-2:

```python
# Replace in get_sentinel2_composite():
s2 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
    .filterBounds(point) \
    .filterDate(start_date, end_date)
```

## Performance Tips

1. **Test first** - Always run with option 1 (10 plots) to verify setup
2. **Batch processing** - Let script handle batching automatically
3. **Monitor progress** - Check temp files in `data/ee_imagery/`
4. **Parallel processing** - Can run multiple instances for different subsets
5. **Network** - Stable internet required for Earth Engine API

## Next Steps After Extraction

1. **Merge with forest data:**
   ```python
   ee_data = pd.read_csv('data/ee_imagery/sentinel2_data_*.csv')
   forest_data = pd.read_csv('data/raw/bc_sample_data-2025-10-09/bc_sample_data.csv')
   merged = forest_data.merge(ee_data, left_on='SITE_IDENTIFIER', right_on='plot_id')
   ```

2. **Update training script** - Modify `yew_density_model.py` to use real imagery

3. **Retrain model** - Run training with actual satellite data

4. **Analyze feature importance** - Which bands/indices are most predictive?

## Resources

- [Earth Engine Python API](https://developers.google.com/earth-engine/guides/python_install)
- [Sentinel-2 Documentation](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)
- [SRTM Elevation](https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003)
- [Earth Engine Code Editor](https://code.earthengine.google.com/) - Test queries interactively
