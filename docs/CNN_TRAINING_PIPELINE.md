# CNN Training Pipeline for Yew Detection

## Overview

This pipeline extracts 64x64 pixel Sentinel-2 image patches and trains a CNN to detect Pacific Yew presence in forest sites.

## Data Filtering

- **Zones**: CWH and ICH biogeoclimatic zones only
- **Sites**: Unique SITE_IDENTIFIERs (deduplicated) only
- **Total sites**: ~5,759 (61 with yew, 5,698 without)
- **Class imbalance**: ~1.1% yew prevalence

## Pipeline Steps

### 1. Extract Image Patches from Earth Engine

```bash
python scripts/preprocessing/extract_ee_image_patches.py
```

**What it does:**
- Loads deduplicated inventory data
- Filters to CWH/ICH zones only
- Converts coordinates to lat/lon
- Extracts 64x64 pixel patches from Sentinel-2
- Saves as numpy arrays (.npy files)
- Creates metadata CSV

**Output:**
```
data/ee_imagery/image_patches_64x64/
├── yew/                          # Images with yew present
│   ├── SITE_ID_1.npy
│   ├── SITE_ID_2.npy
│   └── ...
├── no_yew/                       # Images without yew
│   ├── SITE_ID_1.npy
│   └── ...
├── image_metadata.csv            # Full metadata
├── extraction_progress.json      # Progress tracking
└── (train/val/test splits created during training)
```

**Image format:**
- Shape: `(4, 64, 64)` - [Blue, Green, Red, NIR]
- Type: float32
- Values: Raw Sentinel-2 reflectance (0-10000)

**Features:**
- Automatic resume on interruption
- Progress tracking every 100 sites
- Rate limiting to avoid Earth Engine quotas
- Handles missing data gracefully

**Estimated time:**
- ~0.2-0.5 seconds per site
- ~5,759 sites = 20-50 minutes total

### 2. Train CNN

```bash
python scripts/training/train_yew_cnn.py
```

**What it does:**
- Loads image patches and metadata
- Creates stratified train/val/test splits (70/15/15)
- Applies data augmentation to training set
- Trains CNN with weighted sampling for class imbalance
- Monitors validation loss and saves best model
- Evaluates on test set
- Generates training plots and metrics

**Model Architecture:**
- 4 convolutional blocks with batch normalization
- Progressive channel expansion: 32 → 64 → 128 → 256
- Dropout for regularization
- Global average pooling
- Binary classification head

**Output:**
```
models/
├── checkpoints/
│   └── yew_cnn_best.pth         # Best validation model
└── artifacts/
    └── yew_cnn_final.pth        # Final model

results/
├── figures/
│   ├── yew_cnn_training_history.png
│   └── yew_cnn_confusion_matrix.png
└── reports/
    └── yew_cnn_training_summary.json
```

## Data Augmentation

Training images undergo:
- Horizontal/vertical flips
- Random 90° rotations
- Shift, scale, and rotate
- Brightness/contrast adjustments
- Gaussian noise

## Class Imbalance Handling

Two strategies:
1. **Weighted sampling**: Yew samples are oversampled 10x during training
2. **Data augmentation**: Increases effective training set size

## Dependencies

```bash
# Core
pip install torch torchvision
pip install numpy pandas scikit-learn

# Earth Engine
pip install earthengine-api
pip install pyproj

# Data augmentation
pip install albumentations

# Visualization
pip install matplotlib seaborn
```

## Monitoring Training

The script prints:
- Epoch-by-epoch loss and accuracy
- Learning rate updates
- Best model saves
- Final test metrics including:
  - Accuracy
  - Precision, Recall, F1 per class
  - ROC AUC
  - Confusion matrix

## Customization

Edit configuration in `train_yew_cnn.py`:
```python
config = {
    'batch_size': 32,           # Batch size
    'num_epochs': 50,           # Training epochs
    'learning_rate': 0.001,     # Initial learning rate
    'weight_decay': 1e-4,       # L2 regularization
    'yew_weight': 10.0,         # Oversampling weight for yew class
    'dropout_rate': 0.5         # Dropout probability
}
```

## Notes

- **Image extraction** only needs to be run once
- Extraction supports **resume** - can be interrupted and restarted
- **GPU recommended** for training (but not required)
- Training takes ~10-30 minutes depending on hardware
- Model checkpoints saved automatically

## Comparison with Previous Approach

| Aspect | Previous (Tabular) | Current (CNN) |
|--------|-------------------|---------------|
| Features | Mean spectral values (6) + terrain (3) | 64×64 pixel patches (4 bands) |
| Spatial info | None | Full spatial patterns |
| Model | Hybrid (tabular + placeholder images) | Pure CNN |
| Parameters | ~500K | ~2M |
| Training data | All zones, duplicates | CWH/ICH only, unique sites |
