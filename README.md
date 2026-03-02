# Pacific Yew Density Prediction Model

https://jerichooconnell.github.io/yew_project/

Multi-modal deep learning model for predicting Pacific Yew (*Taxus brevifolia*) density using satellite imagery and forest inventory data.

## 🌲 Project Overview

This project develops a hybrid CNN + tabular neural network to predict Pacific Yew density (stems/hectare) at forest sites across British Columbia. The model addresses extreme class imbalance (only 0.31% of sites contain yew) using focal loss and weighted sampling.

**Key Features:**
- Multi-modal architecture (ResNet18 + Entity Embeddings)
- Spatial cross-validation to prevent data leakage
- 11.5M parameters, trained on 32,125 forest sites
- Handles extreme class imbalance with advanced techniques

## 📁 Project Structure

```
yew_project/
├── data/
│   ├── raw/                          # Original datasets
│   │   ├── bc_sample_data-2025-10-09/
│   │   └── faib_*.csv
│   ├── processed/                    # Analysis results
│   │   └── pacific_yew_*.csv
│   └── lookup_tables/                # Reference data
│       ├── biogeoclimactic_zone_keys.txt
│       └── tree_name_keys.txt
│
├── models/
│   ├── checkpoints/                  # Trained model weights
│   │   ├── best_yew_density_model.pth
│   │   └── yew_density_model_final.pth
│   └── artifacts/                    # Preprocessors & metadata
│       ├── yew_preprocessor.pkl
│       └── yew_feature_info.pkl
│
├── scripts/
│   ├── analysis/                     # Data analysis scripts
│   │   ├── analyze_pacific_yew.py
│   │   ├── analyze_pacific_yew_bc_sample.py
│   │   └── analyze_yew_correlations.py
│   ├── visualization/                # Plotting scripts
│   │   └── detailed_yew_histograms.py
│   └── training/                     # Model training
│       └── yew_density_model.py
│
├── results/
│   ├── figures/                      # Generated plots
│   │   ├── yew_training_history.png
│   │   ├── pacific_yew_correlations.png
│   │   └── *.png
│   ├── tables/                       # Statistical outputs
│   └── reports/                      # Analysis summaries
│       └── yew_correlations_summary.md
│
├── config/                           # Environment & configuration
│   ├── setup_yew_env.sh
│   ├── setup_yew_env_minimal.sh
│   ├── cleanup_conda.sh
│   ├── yew_pytorch_env.yml
│   └── test_environment.py
│
└── docs/                             # Documentation
    └── ENVIRONMENT_SETUP.md
```

## 🚀 Quick Start

### 1. Environment Setup

Choose between minimal (3GB) or full (10GB) installation:

```bash
# Minimal installation (recommended for limited disk space)
cd config/
bash setup_yew_env_minimal.sh

# OR full installation
bash setup_yew_env.sh
```

### 2. Activate Environment

```bash
conda activate yew_pytorch
```

### 3. Test Installation

```bash
python config/test_environment.py
```

### 4. Extract Earth Engine Data (Optional but Recommended)

**First time setup:**
```bash
bash config/setup_earth_engine.sh
```

**Extract satellite imagery for all plots:**
```bash
python scripts/preprocessing/extract_ee_imagery.py
```

See [docs/EARTH_ENGINE_SETUP.md](docs/EARTH_ENGINE_SETUP.md) for detailed instructions.

### 5. Run Training

```bash
python scripts/training/yew_density_model.py
```

## �️ Earth Engine Data Extraction

Extract Sentinel-2 imagery and environmental data for your forest plots:

**Features Extracted:**
- Sentinel-2 RGB + NIR bands (10m resolution)
- NDVI and EVI vegetation indices
- SRTM elevation, slope, aspect (30m resolution)
- Growing season composites (June-August)
- Cloud-masked median values

**Quick start:**
```bash
# Install Earth Engine API
bash config/setup_earth_engine.sh

# Test with 10 sample plots
python scripts/preprocessing/extract_ee_imagery.py
# Choose option 1

# Process all plots
python scripts/preprocessing/extract_ee_imagery.py
# Choose option 2
```

**Output:** Saved to `data/ee_imagery/sentinel2_data_*.csv`

See [docs/EARTH_ENGINE_SETUP.md](docs/EARTH_ENGINE_SETUP.md) for complete documentation.

## �📊 Model Architecture

**Hybrid Multi-Modal Deep Learning Model:**

1. **Image Encoder** (ResNet18)
   - Pretrained on ImageNet
   - Processes Sentinel-2 imagery (RGB + NIR)
   - 512 features → 256-dim embedding
   - For satellite imagery processing

2. **Tabular Encoder**
   - Entity embeddings for 4 categorical features (BEC zones, species, etc.)
   - 12+ numerical features (forest metrics + Earth Engine data)
   - 12 numerical features (basal area, height, location, etc.)
   - Dense network: 140 → 128 → 64

3. **Fusion Network**
   - Combines image + tabular embeddings (320-dim)
   - 3-layer network: 256 → 128 → 64

4. **Output**
   - Single neuron with ReLU (non-negative density prediction)

**Training Specs:**
- Focal Loss (α=0.25, γ=2.0) for imbalance
- Weighted sampling (10x for yew-present sites)
- Spatial block cross-validation
- Early stopping (patience=15)
- Learning rate scheduling

## 📈 Results

**Training Performance:**
- Training set: 21,956 samples (448 spatial blocks)
- Validation set: 3,457 samples (65 spatial blocks)
- Test set: 6,712 samples (129 spatial blocks)

**Key Findings:**
- Only 99 sites (0.31%) contain Pacific Yew
- Density range: 0-1,113 stems/ha
- Mean density: 0.22 stems/ha

## 📝 Analysis Scripts

### Data Analysis
```bash
# Basic Pacific Yew analysis
python scripts/analysis/analyze_pacific_yew.py

# BC sample data analysis
python scripts/analysis/analyze_pacific_yew_bc_sample.py

# Correlation analysis
python scripts/analysis/analyze_yew_correlations.py
```

### Visualization
```bash
# Generate detailed histograms
python scripts/visualization/detailed_yew_histograms.py
```

## 🔧 Configuration

Environment configurations are in `config/`:

- `yew_pytorch_env.yml` - Conda environment specification
- `setup_yew_env.sh` - Full installation script
- `setup_yew_env_minimal.sh` - Minimal installation (3GB)
- `cleanup_conda.sh` - Free disk space script
- `test_environment.py` - Verify installation

## 📚 Data Sources

**BC Forest Inventory Data:**
- 32,125 forest plot measurements
- Species composition strings
- Biogeoclimatic zones (BEC)
- Forest structure metrics

**Features Used:**
- **Numerical (12):** Basal area, stem density, volume, site index, height, age, location, year
- **Categorical (4):** BEC zone, TSA district, establishment type, dominant species

## 🎯 Next Steps

1. **Earth Engine Integration**
   - Extract Sentinel-2/Landsat imagery for each site
   - Replace placeholder images with real satellite data

2. **Model Improvements**
   - Multi-temporal imagery (seasonal variation)
   - Attention mechanisms for feature importance
   - Ensemble methods

3. **Deployment**
   - Create prediction API
   - Generate probability maps for BC
   - Conservation planning tools

## 🐛 Troubleshooting

**Disk Space Issues:**
```bash
cd config/
bash cleanup_conda.sh
```

**CUDA/GPU Issues:**
See `docs/ENVIRONMENT_SETUP.md` for detailed troubleshooting.

**Import Errors:**
```bash
conda activate yew_pytorch
pip install <missing-package>
```

## 📄 License

Research project for Pacific Yew conservation.

## 🤝 Contributing

This is a research project. For questions or collaborations, please open an issue.

## 📖 References

**Data Dictionary:**
- `data/raw/bc_sample_data-2025-10-09/data_dictionary.csv`

**BEC Zone Information:**
- `data/lookup_tables/biogeoclimactic_zone_keys.txt`
- `data/lookup_tables/tree_name_keys.txt`

**Analysis Reports:**
- `results/reports/yew_correlations_summary.md`

---

**Last Updated:** October 16, 2025  
**Model Version:** 1.0  
**PyTorch Version:** 2.1.0 with CUDA 11.8
