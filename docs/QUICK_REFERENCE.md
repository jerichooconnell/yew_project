# Quick Reference Guide

## Common Commands

### Environment Management

```bash
# Activate environment
conda activate yew_pytorch

# Deactivate environment
conda deactivate

# Test environment
python config/test_environment.py

# Clean conda cache (if disk space issues)
bash config/cleanup_conda.sh
```

### Running Scripts

```bash
# Training the model
python scripts/training/yew_density_model.py

# Analysis
python scripts/analysis/analyze_yew_correlations.py
python scripts/analysis/analyze_pacific_yew_bc_sample.py

# Visualization
python scripts/visualization/detailed_yew_histograms.py
```

### Working with Results

```bash
# View training plots
eog results/figures/yew_training_history.png

# Check correlation reports
cat results/reports/yew_correlations_summary.md

# List all figures
ls -lh results/figures/
```

### Model Artifacts

```bash
# Model checkpoints
ls -lh models/checkpoints/

# View model info
python -c "import pickle; info = pickle.load(open('models/artifacts/yew_feature_info.pkl', 'rb')); print(info)"
```

## File Locations Cheat Sheet

| Content | Location |
|---------|----------|
| Raw data | `data/raw/` |
| Processed results | `data/processed/` |
| Lookup tables | `data/lookup_tables/` |
| Model weights | `models/checkpoints/` |
| Preprocessors | `models/artifacts/` |
| Analysis scripts | `scripts/analysis/` |
| Visualization | `scripts/visualization/` |
| Training script | `scripts/training/` |
| Figures/plots | `results/figures/` |
| Reports | `results/reports/` |
| Environment config | `config/` |
| Documentation | `docs/` |

## Data Paths for Scripts

Update these paths in your scripts if needed:

```python
# Raw data
BC_SAMPLE_DATA = "data/raw/bc_sample_data-2025-10-09/bc_sample_data.csv"
DATA_DICT = "data/raw/bc_sample_data-2025-10-09/data_dictionary.csv"

# Lookup tables
BEC_ZONES = "data/lookup_tables/biogeoclimactic_zone_keys.txt"
TREE_NAMES = "data/lookup_tables/tree_name_keys.txt"

# Model artifacts
MODEL_PATH = "models/checkpoints/best_yew_density_model.pth"
PREPROCESSOR = "models/artifacts/yew_preprocessor.pkl"
FEATURE_INFO = "models/artifacts/yew_feature_info.pkl"

# Output
FIGURES_DIR = "results/figures/"
TABLES_DIR = "results/tables/"
REPORTS_DIR = "results/reports/"
```

## GPU Usage

```bash
# Check GPU
nvidia-smi

# Monitor GPU during training
watch -n 1 nvidia-smi
```

## Disk Space Management

```bash
# Check space
df -h .

# Clean conda cache
bash config/cleanup_conda.sh

# Remove old checkpoints (if needed)
rm models/checkpoints/old_*.pth
```

## Updating Scripts

When moving scripts, remember to update paths:

**Before:**
```python
df = pd.read_csv('bc_sample_data-2025-10-09/bc_sample_data.csv')
```

**After:**
```python
df = pd.read_csv('../data/raw/bc_sample_data-2025-10-09/bc_sample_data.csv')
# OR use absolute paths
import os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'data/raw/bc_sample_data-2025-10-09/bc_sample_data.csv'))
```

## Troubleshooting

### Import errors after reorganizing
```bash
# Add project root to Python path
export PYTHONPATH="/home/jericho/yew_project:$PYTHONPATH"
```

### File not found errors
- Check if paths in scripts are relative to script location
- Use absolute paths or path joining with `os.path.join()`

### Model checkpoint issues
- Models are in `models/checkpoints/`
- Artifacts are in `models/artifacts/`
