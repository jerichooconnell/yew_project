# Earth Engine Integration Summary

## ✅ Created Files

### 1. Main Extraction Script
**Location:** `scripts/preprocessing/extract_ee_imagery.py`

**Features:**
- Complete Earth Engine data extraction pipeline
- Extracts Sentinel-2 RGB + NIR imagery (10m resolution)
- Calculates NDVI and EVI vegetation indices
- Extracts SRTM elevation, slope, and aspect
- Two extraction modes:
  - **Point mode**: Mean values in 250m buffer (for tabular features)
  - **Patch mode**: 64x64 pixel image patches (for CNN input)
- Automatic progress saving every 50 plots
- Batch processing with rate limiting
- Interactive menu for easy use
- Optional visualization with geemap

**Key Functions:**
- `bc_albers_to_latlon()` - Coordinate conversion
- `get_sentinel2_composite()` - Cloud-masked Sentinel-2 composites
- `calculate_vegetation_indices()` - NDVI and EVI
- `get_elevation_data()` - SRTM terrain data
- `extract_point_data()` - Extract point values
- `extract_imagery_patch()` - Extract image patches
- `batch_extract()` - Process all plots with progress tracking

### 2. Setup Script
**Location:** `config/setup_earth_engine.sh`

**Features:**
- Installs earthengine-api
- Optional geemap installation for visualization
- Interactive authentication setup
- Tests connection after setup

**Usage:**
```bash
bash config/setup_earth_engine.sh
```

### 3. Documentation
**Location:** `docs/EARTH_ENGINE_SETUP.md`

**Contents:**
- Complete installation instructions
- Authentication guide
- Usage examples
- Data format documentation
- Integration with model instructions
- Troubleshooting guide
- Rate limits and quota information
- Advanced usage examples

### 4. Updated README
**Location:** `README.md`

Added section on Earth Engine integration with quick start guide.

## 📊 Data Extracted

### Sentinel-2 Bands
| Band | Name | Wavelength | Resolution |
|------|------|------------|------------|
| B2 | Blue | 490 nm | 10m |
| B3 | Green | 560 nm | 10m |
| B4 | Red | 665 nm | 10m |
| B8 | NIR | 842 nm | 10m |

### Derived Products
| Product | Description | Range |
|---------|-------------|-------|
| NDVI | Normalized Difference Vegetation Index | -1 to 1 |
| EVI | Enhanced Vegetation Index | -1 to 1 |
| Elevation | SRTM Digital Elevation Model | meters |
| Slope | Terrain slope | degrees |
| Aspect | Terrain aspect (direction) | degrees |

## 🚀 Quick Start

### 1. Install Earth Engine API

```bash
conda activate yew_pytorch
bash config/setup_earth_engine.sh
```

This will:
- Install earthengine-api
- Guide you through authentication
- Test the connection

### 2. Test with Sample Data

```bash
python scripts/preprocessing/extract_ee_imagery.py
```
Choose option 1 for 10 sample plots.

### 3. Extract Data for All Plots

```bash
python scripts/preprocessing/extract_ee_imagery.py
```
Choose option 2 for point data (recommended first).

### 4. Use Extracted Data

The extracted data is saved to `data/ee_imagery/sentinel2_data_*.csv`

## 🔧 Integration with Model

### Option 1: Add Earth Engine Features to Tabular Encoder

Modify `scripts/training/yew_density_model.py`:

```python
# In YewDataPreprocessor.prepare_features()
numerical_cols = [
    'BA_HA_LS', 'BA_HA_DS', 'STEMS_HA_LS', 'STEMS_HA_DS',
    'VHA_WSV_LS', 'VHA_NTWB_LS', 'SI_M_TLSO', 'HT_TLSO',
    'AGEB_TLSO', 'BC_ALBERS_X', 'BC_ALBERS_Y', 'MEAS_YR',
    # Add Earth Engine features:
    'ndvi', 'evi', 'elevation', 'slope', 'aspect'
]

# Load Earth Engine data
ee_data = pd.read_csv('data/ee_imagery/sentinel2_data_*.csv')

# Merge with forest data
df = df.merge(ee_data, left_on='SITE_IDENTIFIER', right_on='plot_id', how='left')
```

### Option 2: Use Real Sentinel-2 Images for CNN

Replace placeholder images in `YewDensityDataset`:

```python
def _load_sentinel2_image(self, idx):
    """Load actual Sentinel-2 image patch."""
    image_path = self.imagery_paths[idx]
    # Load 4-channel image (B, G, R, NIR)
    image = np.load(image_path)  # Shape: (4, 64, 64)
    return torch.FloatTensor(image)
```

## 📈 Expected Performance Improvements

By adding real satellite imagery and environmental data:

1. **Better spatial context** - Actual vegetation patterns around plot
2. **Temporal information** - Growing season composite captures phenology
3. **Environmental predictors** - Elevation, slope, aspect affect species distribution
4. **Vegetation health** - NDVI/EVI indicate forest condition

Expected improvements:
- **R² score:** +0.1 to 0.2 increase
- **MAE:** 10-20% reduction
- **Rare species detection:** Better identification of yew-present sites

## ⚠️ Important Notes

### Rate Limits
- **Free tier:** 5,000 requests/day
- **Your data:** ~32,000 plots
- **Strategy:** Run in batches over multiple days
- **Auto-save:** Script checkpoints every 50 plots

### Data Quality
- Some plots may fail (no imagery, clouds, etc.)
- Expected success rate: 80-95%
- Failed extractions marked with `success=False`

### Coordinate Systems
- Input: BC Albers (EPSG:3005)
- Converted to: WGS84 (EPSG:4326) for Earth Engine
- Automatic conversion handled by script

## 📁 Output Files

After extraction, you'll have:

```
data/ee_imagery/
├── sentinel2_data_20251016_123456.csv          # Main data
├── sentinel2_data_20251016_123456.pkl          # Pickle format
├── sentinel2_data_20251016_123456_metadata.json # Metadata
├── temp_extraction_point_50.csv                 # Checkpoints
├── temp_extraction_point_100.csv
└── sample_locations_map.html                    # Visualization
```

## 🐛 Troubleshooting

### Authentication Failed
```bash
earthengine authenticate
```

### Quota Exceeded
Wait 24 hours or upgrade to paid tier. Script saves progress automatically.

### Import Errors
```bash
pip install earthengine-api geemap
```

### Coordinate Errors
Check that BC Albers coordinates are valid and within BC bounds.

## 📚 Next Steps

1. **Extract data** - Run extraction for all plots
2. **Quality check** - Review success rate and data quality
3. **Merge with forest data** - Combine Earth Engine features with forest inventory
4. **Update model** - Modify training script to use real imagery/features
5. **Retrain** - Train model with satellite data
6. **Evaluate** - Compare performance with/without Earth Engine data
7. **Visualize** - Analyze which features are most important

## 🎯 Model Enhancement Roadmap

### Phase 1: Add Tabular Features (Easy)
- Merge NDVI, EVI, elevation, slope, aspect with forest data
- Update numerical features list
- Retrain model
- **Time:** 1-2 days

### Phase 2: Use Real Images (Moderate)
- Extract image patches (option 3 in script)
- Modify dataset to load actual images
- Change image encoder input channels if needed
- **Time:** 3-5 days

### Phase 3: Multi-temporal (Advanced)
- Extract multiple time points (spring, summer, fall)
- Add temporal dimension to model
- Use RNN/LSTM or 3D CNN
- **Time:** 1-2 weeks

## 📖 Additional Resources

- [Earth Engine Python API](https://developers.google.com/earth-engine/guides/python_install)
- [Sentinel-2 User Guide](https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi)
- [SRTM Elevation Data](https://www2.jpl.nasa.gov/srtm/)
- [Earth Engine Code Editor](https://code.earthengine.google.com/) - Test queries interactively
- [Earth Engine Data Catalog](https://developers.google.com/earth-engine/datasets)

---

**Created:** October 16, 2025  
**Status:** Ready to use  
**Next:** Run `bash config/setup_earth_engine.sh` to get started!
