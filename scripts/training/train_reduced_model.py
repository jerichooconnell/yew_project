#!/usr/bin/env python3
"""
Train Reduced Model & Compare Performance
=========================================

Trains model with reduced feature set (9 features) and compares
with original model (15 features).

Author: Analysis Tool
Date: October 2025
"""

from train_with_ee_data import load_and_merge_data, parse_species_composition
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
import time

import sys
sys.path.append('scripts/training')


def prepare_training_data_with_selected_features(merged_df, selected_features):
    """Prepare training data using only selected features."""
    print("\nPreparing training data with selected features...")

    # Calculate yew density
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

    # Use only selected numerical features
    numerical_cols = selected_features

    # Categorical features
    categorical_cols = ['BEC_ZONE', 'TSA_DESC', 'SPC_LIVE_1']

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
        print("\nERROR: No valid training data!")
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

    return numerical_features, categorical_features, targets, feature_info


def train_model(numerical_features, categorical_features, targets, feature_info,
                model_name, device, num_epochs=100):
    """Train a model and return metrics."""
    print("\n" + "="*70)
    print(f"TRAINING: {model_name}")
    print("="*70)

    # Create spatial splits
    train_idx, val_idx, test_idx = spatial_train_test_split(
        feature_info['coordinates'],
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )

    # Create placeholder imagery paths
    imagery_paths = [f"placeholder_{i}" for i in range(len(targets))]

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
    model = HybridYewDensityModel(
        image_channels=3,
        num_numerical_features=feature_info['num_numerical'],
        categorical_dims=feature_info['categorical_dims'],
        image_embedding_dim=256,
        tabular_embedding_dim=32,
        fusion_hidden_dims=[256, 128, 64]
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Train
    start_time = time.time()
    trainer = YewDensityTrainer(model, device=device)
    trainer.train(train_loader, val_loader,
                  num_epochs=num_epochs, learning_rate=0.001)
    training_time = time.time() - start_time

    # Evaluate on test set
    criterion = nn.MSELoss()
    test_loss, test_mae, test_rmse, test_r2, test_preds, test_targets = trainer.validate(
        test_loader, criterion)

    # Calculate metrics for yew-present samples
    yew_present_mask = test_targets > 0
    yew_mae = yew_rmse = 0
    if yew_present_mask.sum() > 0:
        yew_mae = mean_absolute_error(
            test_targets[yew_present_mask], test_preds[yew_present_mask])
        yew_rmse = np.sqrt(mean_squared_error(
            test_targets[yew_present_mask], test_preds[yew_present_mask]))

    metrics = {
        'model_name': model_name,
        'n_features': feature_info['num_numerical'],
        'n_params': total_params,
        'training_time': training_time,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'yew_mae': yew_mae,
        'yew_rmse': yew_rmse,
        'n_yew_test': yew_present_mask.sum(),
        'history': trainer.history
    }

    return model, metrics, test_preds, test_targets


def compare_models(metrics_original, metrics_reduced):
    """Compare original and reduced models."""
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)

    comparison = pd.DataFrame([
        {
            'Model': 'Original (15 features)',
            'Features': metrics_original['n_features'],
            'Parameters': f"{metrics_original['n_params']:,}",
            'Training Time': f"{metrics_original['training_time']/60:.1f} min",
            'Test MAE': f"{metrics_original['test_mae']:.4f}",
            'Test RMSE': f"{metrics_original['test_rmse']:.4f}",
            'Test R²': f"{metrics_original['test_r2']:.4f}",
            'Yew MAE': f"{metrics_original['yew_mae']:.4f}",
            'Yew RMSE': f"{metrics_original['yew_rmse']:.4f}"
        },
        {
            'Model': 'Reduced (9 features)',
            'Features': metrics_reduced['n_features'],
            'Parameters': f"{metrics_reduced['n_params']:,}",
            'Training Time': f"{metrics_reduced['training_time']/60:.1f} min",
            'Test MAE': f"{metrics_reduced['test_mae']:.4f}",
            'Test RMSE': f"{metrics_reduced['test_rmse']:.4f}",
            'Test R²': f"{metrics_reduced['test_r2']:.4f}",
            'Yew MAE': f"{metrics_reduced['yew_mae']:.4f}",
            'Yew RMSE': f"{metrics_reduced['yew_rmse']:.4f}"
        }
    ])

    print("\n" + comparison.to_string(index=False))

    # Calculate improvements
    print("\n" + "="*70)
    print("IMPROVEMENTS")
    print("="*70)

    mae_change = (
        (metrics_reduced['test_mae'] - metrics_original['test_mae']) / metrics_original['test_mae']) * 100
    rmse_change = (
        (metrics_reduced['test_rmse'] - metrics_original['test_rmse']) / metrics_original['test_rmse']) * 100
    r2_change = metrics_reduced['test_r2'] - metrics_original['test_r2']
    yew_mae_change = (
        (metrics_reduced['yew_mae'] - metrics_original['yew_mae']) / metrics_original['yew_mae']) * 100
    time_change = ((metrics_reduced['training_time'] -
                   metrics_original['training_time']) / metrics_original['training_time']) * 100

    print(f"\nFeature reduction: 15 → 9 ({40:.0f}% fewer features)")
    print(f"Training time change: {time_change:+.1f}%")
    print(f"Test MAE change: {mae_change:+.2f}%")
    print(f"Test RMSE change: {rmse_change:+.2f}%")
    print(f"Test R² change: {r2_change:+.4f}")
    print(f"Yew-present MAE change: {yew_mae_change:+.2f}%")

    if mae_change < 0:
        print(
            f"\n✓ REDUCED MODEL IS BETTER (lower MAE by {abs(mae_change):.1f}%)")
    elif abs(mae_change) < 5:
        print(f"\n≈ MODELS ARE SIMILAR (within 5% MAE difference)")
    else:
        print(
            f"\n✗ Original model is better (lower MAE by {abs(mae_change):.1f}%)")

    return comparison


def plot_comparison(metrics_original, metrics_reduced, output_dir):
    """Create comparison visualizations."""
    print("\n" + "="*70)
    print("CREATING COMPARISON VISUALIZATIONS")
    print("="*70)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Training curves - Loss
    ax = axes[0, 0]
    ax.plot(metrics_original['history']['val_loss'],
            label='Original (15 features)', linewidth=2)
    ax.plot(metrics_reduced['history']['val_loss'],
            label='Reduced (9 features)', linewidth=2, linestyle='--')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: Training curves - MAE
    ax = axes[0, 1]
    ax.plot(metrics_original['history']['val_mae'],
            label='Original (15 features)', linewidth=2)
    ax.plot(metrics_reduced['history']['val_mae'],
            label='Reduced (9 features)', linewidth=2, linestyle='--')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation MAE', fontsize=12)
    ax.set_title('Validation MAE Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 3: Bar chart of metrics
    ax = axes[1, 0]
    metrics_names = ['Test MAE', 'Test RMSE', 'Yew MAE']
    original_vals = [metrics_original['test_mae'],
                     metrics_original['test_rmse'], metrics_original['yew_mae']]
    reduced_vals = [metrics_reduced['test_mae'],
                    metrics_reduced['test_rmse'], metrics_reduced['yew_mae']]

    x = np.arange(len(metrics_names))
    width = 0.35

    ax.bar(x - width/2, original_vals, width,
           label='Original (15 features)', alpha=0.8, edgecolor='black')
    ax.bar(x + width/2, reduced_vals, width,
           label='Reduced (9 features)', alpha=0.8, edgecolor='black')
    ax.set_ylabel('Error (stems/ha)', fontsize=12)
    ax.set_title('Test Set Performance Metrics',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Feature count and training time
    ax = axes[1, 1]
    categories = ['Features', 'Training Time\n(minutes)']
    original_vals = [metrics_original['n_features'],
                     metrics_original['training_time']/60]
    reduced_vals = [metrics_reduced['n_features'],
                    metrics_reduced['training_time']/60]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, original_vals, width,
                   label='Original', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, reduced_vals, width,
                   label='Reduced', alpha=0.8, edgecolor='black')

    ax.set_ylabel('Count / Minutes', fontsize=12)
    ax.set_title('Model Complexity Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_path = output_dir / 'model_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")


def main():
    print("="*70)
    print("TRAIN REDUCED MODEL & COMPARE PERFORMANCE")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Load selected features
    with open('models/artifacts/selected_features.pkl', 'rb') as f:
        feature_config = pickle.load(f)

    selected_features = feature_config['selected_features']
    print(f"\nSelected features ({len(selected_features)}):")
    for feat in selected_features:
        print(f"  - {feat}")

    # Load data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    merged_df = load_and_merge_data()

    # Train ORIGINAL model (15 features)
    print("\n" + "="*70)
    print("STEP 1: TRAINING ORIGINAL MODEL (15 features)")
    print("="*70)

    all_features = [
        'blue', 'green', 'red', 'nir', 'ndvi', 'evi',
        'elevation', 'slope', 'aspect',
        'BA_HA_LS', 'STEMS_HA_LS', 'VHA_WSV_LS',
        'SI_M_TLSO', 'HT_TLSO', 'AGEB_TLSO'
    ]

    num_feat_orig, cat_feat_orig, targets_orig, info_orig = prepare_training_data_with_selected_features(
        merged_df, all_features
    )

    model_orig, metrics_orig, preds_orig, targets_test_orig = train_model(
        num_feat_orig, cat_feat_orig, targets_orig, info_orig,
        "Original Model (15 features)", device, num_epochs=100
    )

    # Train REDUCED model (9 features)
    print("\n" + "="*70)
    print("STEP 2: TRAINING REDUCED MODEL (9 features)")
    print("="*70)

    num_feat_red, cat_feat_red, targets_red, info_red = prepare_training_data_with_selected_features(
        merged_df, selected_features
    )

    model_red, metrics_red, preds_red, targets_test_red = train_model(
        num_feat_red, cat_feat_red, targets_red, info_red,
        "Reduced Model (9 features)", device, num_epochs=100
    )

    # Compare models
    print("\n" + "="*70)
    print("STEP 3: COMPARING MODELS")
    print("="*70)

    comparison_df = compare_models(metrics_orig, metrics_red)

    # Save comparison
    output_dir = Path('results/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    tables_dir = Path('results/tables')
    tables_dir.mkdir(parents=True, exist_ok=True)

    comparison_df.to_csv(tables_dir / 'model_comparison.csv', index=False)
    print(
        f"\nComparison table saved to: {tables_dir / 'model_comparison.csv'}")

    # Visualize
    plot_comparison(metrics_orig, metrics_red, output_dir)

    # Save reduced model
    model_path = Path('models/artifacts/yew_model_reduced_9_features.pth')
    torch.save(model_red.state_dict(), model_path)
    print(f"\nReduced model saved to: {model_path}")

    # Save preprocessor for reduced model
    preprocessor_path = Path('models/artifacts/yew_preprocessor_reduced.pkl')
    with open(preprocessor_path, 'wb') as f:
        pickle.dump({
            'scaler': info_red['scaler'],
            'label_encoders': info_red['label_encoders'],
            'feature_info': info_red,
            'selected_features': selected_features
        }, f)
    print(f"Preprocessor saved to: {preprocessor_path}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
