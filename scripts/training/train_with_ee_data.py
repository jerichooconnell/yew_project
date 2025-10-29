#!/usr/bin/env python3
"""
Train Pacific Yew Model with Earth Engine Data
==============================================

Trains the hybrid model using extracted Earth Engine satellite features.

Author: Analysis Tool
Date: October 2025
"""

from yew_density_model import (
    HybridYewDensityModel,
    YewDensityDataset,
    YewDensityTrainer,
    spatial_train_test_split
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import re

# Import the model classes from existing script
import sys
sys.path.append('scripts/training')


def parse_species_composition(composition_string):
    """Parse species composition string to extract Pacific Yew (TW) percentage."""
    if not composition_string or pd.isna(composition_string):
        return 0

    pattern = r'TW(\d{2,3})'  # TW = Taxus brevifolia (Pacific Yew)
    match = re.search(pattern, str(composition_string))

    if match:
        return int(match.group(1))
    return 0


def load_and_merge_data():
    """Load and merge Earth Engine data with forest inventory."""
    print("Loading Earth Engine data...")
    ee_df = pd.read_csv('data/ee_imagery/ee_extraction_progress.csv')
    print(f"  Earth Engine data: {len(ee_df)} plots")

    print("Loading forest inventory data...")
    inv_df = pd.read_csv(
        'data/raw/bc_sample_data-2025-10-09/bc_sample_data.csv', low_memory=False)
    print(f"  Forest inventory: {len(inv_df)} plots")

    # Merge on SITE_IDENTIFIER
    ee_df['SITE_IDENTIFIER'] = ee_df['plot_id'].astype(str)
    inv_df['SITE_IDENTIFIER'] = inv_df['SITE_IDENTIFIER'].astype(str)

    merged_df = inv_df.merge(ee_df, on='SITE_IDENTIFIER',
                             how='inner', suffixes=('', '_ee'))
    print(f"  Merged dataset: {len(merged_df)} rows")
    print(f"  Unique sites: {merged_df['SITE_IDENTIFIER'].nunique()}")

    return merged_df


def prepare_training_data(merged_df):
    """Prepare features and targets for training."""
    print("\nPreparing training data...")

    # Calculate yew density from species composition
    merged_df['YEW_PERCENTAGE'] = merged_df['SPB_CPCT_LS'].apply(
        parse_species_composition)
    merged_df['YEW_DENSITY_HA'] = (
        merged_df['YEW_PERCENTAGE'] / 100.0) * merged_df['STEMS_HA_LS'].fillna(0)

    print(f"\nYew statistics:")
    print(
        f"  Plots with yew (density > 0): {(merged_df['YEW_DENSITY_HA'] > 0).sum()}")
    print(f"  Max density: {merged_df['YEW_DENSITY_HA'].max():.1f} stems/ha")
    yew_present = merged_df[merged_df['YEW_DENSITY_HA'] > 0]
    if len(yew_present) > 0:
        print(
            f"  Mean density (where > 0): {yew_present['YEW_DENSITY_HA'].mean():.1f} stems/ha")
        print(
            f"  Median density (where > 0): {yew_present['YEW_DENSITY_HA'].median():.1f} stems/ha")

    # Define numerical features (Earth Engine + forest metrics)
    numerical_cols = [
        # Earth Engine satellite features
        'blue', 'green', 'red', 'nir', 'ndvi', 'evi',
        'elevation', 'slope', 'aspect',
        # Forest inventory features
        'BA_HA_LS', 'STEMS_HA_LS', 'VHA_WSV_LS',
        'SI_M_TLSO', 'HT_TLSO', 'AGEB_TLSO'
    ]

    # Define categorical features
    categorical_cols = [
        'BEC_ZONE',
        'TSA_DESC',
        'SPC_LIVE_1'
    ]

    # Filter to rows with all required features
    required_cols = numerical_cols + categorical_cols + \
        ['YEW_DENSITY_HA', 'BC_ALBERS_X', 'BC_ALBERS_Y']
    training_df = merged_df[required_cols].copy()

    # Drop rows with missing values
    initial_len = len(training_df)
    training_df = training_df.dropna()
    print(
        f"\nRows after removing missing values: {len(training_df)} (dropped {initial_len - len(training_df)})")

    if len(training_df) == 0:
        print("\nERROR: No valid training data after filtering!")
        print("Checking data availability:")
        for col in numerical_cols + categorical_cols:
            if col in merged_df.columns:
                missing = merged_df[col].isna().sum()
                print(
                    f"  {col}: {missing}/{len(merged_df)} missing ({100*missing/len(merged_df):.1f}%)")
        return None, None, None, None

    # Prepare numerical features
    scaler = StandardScaler()
    numerical_features = scaler.fit_transform(
        training_df[numerical_cols].values)

    # Prepare categorical features
    categorical_features = {}
    categorical_dims = {}
    label_encoders = {}

    for col in categorical_cols:
        training_df[col] = training_df[col].fillna('UNKNOWN')
        le = LabelEncoder()
        encoded = le.fit_transform(training_df[col].astype(str))
        categorical_features[col] = encoded
        categorical_dims[col] = len(le.classes_)
        label_encoders[col] = le
        print(f"  {col}: {len(le.classes_)} unique values")

    # Get targets
    targets = training_df['YEW_DENSITY_HA'].values
    coordinates = training_df[['BC_ALBERS_X', 'BC_ALBERS_Y']].values

    feature_info = {
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'categorical_dims': categorical_dims,
        'num_numerical': len(numerical_cols),
        'coordinates': coordinates,
        'scaler': scaler,
        'label_encoders': label_encoders
    }

    print(f"\nFeature summary:")
    print(f"  Numerical features: {len(numerical_cols)}")
    print(f"  Categorical features: {len(categorical_cols)}")
    print(f"  Total samples: {len(targets)}")
    print(f"  Samples with yew (target > 0): {(targets > 0).sum()}")
    print(
        f"  Target range: {targets.min():.2f} - {targets.max():.2f} stems/ha")

    return numerical_features, categorical_features, targets, feature_info


def main():
    print("="*70)
    print("Pacific Yew Density Prediction with Earth Engine Data")
    print("="*70)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Load and merge data
    print("\n" + "="*70)
    print("STEP 1: Loading and Merging Data")
    print("="*70)

    merged_df = load_and_merge_data()

    # Prepare training data
    print("\n" + "="*70)
    print("STEP 2: Feature Engineering")
    print("="*70)

    result = prepare_training_data(merged_df)
    if result[0] is None:
        print("\nTraining aborted due to data issues.")
        return

    numerical_features, categorical_features, targets, feature_info = result

    # Create spatial splits
    print("\n" + "="*70)
    print("STEP 3: Creating Spatial Train/Val/Test Splits")
    print("="*70)

    train_idx, val_idx, test_idx = spatial_train_test_split(
        feature_info['coordinates'],
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )

    # Create placeholder imagery paths
    imagery_paths = [f"placeholder_{i}" for i in range(len(targets))]

    # Helper function
    def subset_categorical(cat_features, indices):
        return {name: values[indices] for name, values in cat_features.items()}

    # Create datasets
    train_dataset = YewDensityDataset(
        [imagery_paths[i] for i in train_idx],
        numerical_features[train_idx],
        subset_categorical(categorical_features, train_idx),
        targets[train_idx]
    )

    val_dataset = YewDensityDataset(
        [imagery_paths[i] for i in val_idx],
        numerical_features[val_idx],
        subset_categorical(categorical_features, val_idx),
        targets[val_idx]
    )

    test_dataset = YewDensityDataset(
        [imagery_paths[i] for i in test_idx],
        numerical_features[test_idx],
        subset_categorical(categorical_features, test_idx),
        targets[test_idx]
    )

    # Create data loaders
    train_sampler = YewDensityTrainer.create_weighted_sampler(
        targets[train_idx], yew_weight=10.0)

    train_loader = DataLoader(
        train_dataset, batch_size=32, sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32,
                             shuffle=False, num_workers=0)

    # Create model
    print("\n" + "="*70)
    print("STEP 4: Building Model")
    print("="*70)

    model = HybridYewDensityModel(
        image_channels=3,
        num_numerical_features=feature_info['num_numerical'],
        categorical_dims=feature_info['categorical_dims'],
        image_embedding_dim=256,
        tabular_embedding_dim=32,
        fusion_hidden_dims=[256, 128, 64]
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Train model
    print("\n" + "="*70)
    print("STEP 5: Training Model")
    print("="*70)

    trainer = YewDensityTrainer(model, device=device)
    trainer.train(train_loader, val_loader,
                  num_epochs=100, learning_rate=0.001)

    # Evaluate on test set
    print("\n" + "="*70)
    print("STEP 6: Final Evaluation")
    print("="*70)

    criterion = nn.MSELoss()
    test_loss, test_mae, test_rmse, test_r2, test_preds, test_targets = trainer.validate(
        test_loader, criterion)

    print(f"\nTest Set Results:")
    print(f"  MAE: {test_mae:.4f} stems/ha")
    print(f"  RMSE: {test_rmse:.4f} stems/ha")
    print(f"  R² Score: {test_r2:.4f}")

    # Yew-present sites only
    yew_present_mask = test_targets > 0
    if yew_present_mask.sum() > 0:
        yew_mae = mean_absolute_error(
            test_targets[yew_present_mask], test_preds[yew_present_mask])
        yew_rmse = np.sqrt(mean_squared_error(
            test_targets[yew_present_mask], test_preds[yew_present_mask]))
        print(
            f"\nPerformance on yew-present sites ({yew_present_mask.sum()} plots):")
        print(f"  MAE: {yew_mae:.4f} stems/ha")
        print(f"  RMSE: {yew_rmse:.4f} stems/ha")

    # Save everything
    print("\n" + "="*70)
    print("STEP 7: Saving Results")
    print("="*70)

    output_dir = Path('models/artifacts')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_dir / 'yew_model_with_ee_data.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")

    # Save preprocessor
    preprocessor_path = output_dir / 'yew_preprocessor_ee.pkl'
    with open(preprocessor_path, 'wb') as f:
        pickle.dump({
            'scaler': feature_info['scaler'],
            'label_encoders': feature_info['label_encoders'],
            'feature_info': feature_info
        }, f)
    print(f"Preprocessor saved: {preprocessor_path}")

    # Save training history
    history_path = Path('results/figures/yew_training_history_ee.png')
    history_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.plot_training_history(str(history_path))

    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
