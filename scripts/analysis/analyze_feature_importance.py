#!/usr/bin/env python3
"""
Feature Importance Analysis for Pacific Yew Model
=================================================

Analyzes which features are most important for predicting yew density.

Methods:
1. Permutation importance
2. Gradient-based attribution
3. Correlation analysis

Author: Analysis Tool
Date: October 2025
"""

from train_with_ee_data import load_and_merge_data, prepare_training_data
from yew_density_model import HybridYewDensityModel, YewDensityDataset
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import sys

sys.path.append('scripts/training')


def load_trained_model(model_path, feature_info):
    """Load the trained model."""
    model = HybridYewDensityModel(
        image_channels=3,
        num_numerical_features=feature_info['num_numerical'],
        categorical_dims=feature_info['categorical_dims'],
        image_embedding_dim=256,
        tabular_embedding_dim=32,
        fusion_hidden_dims=[256, 128, 64]
    )

    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model


def permutation_importance(model, dataset, feature_names, device='cuda', n_repeats=5):
    """
    Calculate permutation importance for each feature.

    Measures how much performance drops when a feature is randomly shuffled.
    """
    print("\nCalculating permutation importance...")
    print("This may take a few minutes...")

    model.eval()

    # Get baseline predictions
    baseline_preds = []
    targets_all = []

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Baseline predictions"):
            image, numerical, categorical, target = dataset[i]

            image = image.unsqueeze(0).to(device)
            numerical = numerical.unsqueeze(0).to(device)
            categorical_batch = {name: values.unsqueeze(0).to(device)
                                 for name, values in categorical.items()}

            pred = model(image, numerical, categorical_batch).cpu().item()
            baseline_preds.append(pred)
            targets_all.append(target.item())

    baseline_preds = np.array(baseline_preds)
    targets_all = np.array(targets_all)
    baseline_mae = mean_absolute_error(targets_all, baseline_preds)

    print(f"Baseline MAE: {baseline_mae:.4f}")

    # Calculate importance for each numerical feature
    importances = {}

    for feat_idx, feat_name in enumerate(tqdm(feature_names, desc="Features")):
        mae_scores = []

        for repeat in range(n_repeats):
            permuted_preds = []

            with torch.no_grad():
                for i in range(len(dataset)):
                    image, numerical, categorical, target = dataset[i]

                    # Permute this feature
                    permuted_numerical = numerical.clone()
                    random_idx = np.random.randint(0, len(dataset))
                    _, random_numerical, _, _ = dataset[random_idx]
                    permuted_numerical[feat_idx] = random_numerical[feat_idx]

                    image = image.unsqueeze(0).to(device)
                    permuted_numerical = permuted_numerical.unsqueeze(
                        0).to(device)
                    categorical_batch = {name: values.unsqueeze(0).to(device)
                                         for name, values in categorical.items()}

                    pred = model(image, permuted_numerical,
                                 categorical_batch).cpu().item()
                    permuted_preds.append(pred)

            permuted_preds = np.array(permuted_preds)
            permuted_mae = mean_absolute_error(targets_all, permuted_preds)
            mae_scores.append(permuted_mae - baseline_mae)

        importances[feat_name] = {
            'mean': np.mean(mae_scores),
            'std': np.std(mae_scores)
        }

    return importances, baseline_mae


def gradient_based_importance(model, dataset, feature_names, device='cuda', n_samples=1000):
    """
    Calculate gradient-based feature importance.

    Measures how sensitive predictions are to each feature.
    """
    print("\nCalculating gradient-based importance...")

    model.eval()

    feature_gradients = {name: [] for name in feature_names}

    for i in tqdm(range(min(n_samples, len(dataset))), desc="Samples"):
        image, numerical, categorical, target = dataset[i]

        image = image.unsqueeze(0).to(device).requires_grad_(True)
        numerical = numerical.unsqueeze(0).to(device).requires_grad_(True)
        categorical_batch = {name: values.unsqueeze(0).to(device)
                             for name, values in categorical.items()}

        # Forward pass
        output = model(image, numerical, categorical_batch)

        # Backward pass
        output.backward()

        # Store gradients for numerical features
        if numerical.grad is not None:
            grads = numerical.grad.abs().cpu().numpy().flatten()
            for feat_idx, feat_name in enumerate(feature_names):
                feature_gradients[feat_name].append(grads[feat_idx])

    # Aggregate gradients
    importance_scores = {}
    for feat_name in feature_names:
        grads = np.array(feature_gradients[feat_name])
        importance_scores[feat_name] = {
            'mean': np.mean(grads),
            'std': np.std(grads)
        }

    return importance_scores


def correlation_analysis(merged_df, feature_cols, target_col='YEW_DENSITY_HA'):
    """
    Analyze correlation between features and target.
    """
    print("\nCalculating feature correlations...")

    # Filter to complete cases
    analysis_df = merged_df[feature_cols + [target_col]].dropna()

    correlations = {}
    for col in feature_cols:
        if analysis_df[col].dtype in [np.float64, np.int64]:
            corr = analysis_df[col].corr(analysis_df[target_col])
            correlations[col] = abs(corr)  # Absolute correlation

    return correlations


def plot_feature_importance(importances_dict, title, output_path):
    """Plot feature importance results."""
    # Sort by importance
    sorted_features = sorted(importances_dict.items(),
                             key=lambda x: x[1]['mean'],
                             reverse=True)

    features = [f[0] for f in sorted_features]
    means = [f[1]['mean'] for f in sorted_features]
    stds = [f[1].get('std', 0) for f in sorted_features]

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))

    y_pos = np.arange(len(features))
    ax.barh(y_pos, means, xerr=stds, capsize=3, alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")

    return fig


def plot_correlation_heatmap(merged_df, feature_cols, target_col, output_path):
    """Plot correlation heatmap."""
    analysis_df = merged_df[feature_cols + [target_col]].dropna()

    # Select only numeric columns
    numeric_cols = [col for col in feature_cols if analysis_df[col].dtype in [
        np.float64, np.int64]]
    numeric_cols.append(target_col)

    corr_matrix = analysis_df[numeric_cols].corr()

    # Plot
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8}, ax=ax)

    ax.set_title('Feature Correlation Matrix\n(Includes Yew Density)',
                 fontsize=14, fontweight='bold', pad=20)

    # Highlight target row/column
    target_idx = numeric_cols.index(target_col)
    ax.add_patch(plt.Rectangle((0, target_idx), len(numeric_cols), 1,
                               fill=False, edgecolor='red', lw=3))
    ax.add_patch(plt.Rectangle((target_idx, 0), 1, len(numeric_cols),
                               fill=False, edgecolor='red', lw=3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")

    return fig


def main():
    print("="*70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("\nLoading data...")
    merged_df = load_and_merge_data()
    numerical_features, categorical_features, targets, feature_info = prepare_training_data(
        merged_df)

    if numerical_features is None:
        print("Error loading data!")
        return

    feature_names = feature_info['numerical_cols']
    print(f"Analyzing {len(feature_names)} numerical features")

    # Load model
    print("\nLoading trained model...")
    model_path = 'models/artifacts/yew_model_with_ee_data.pth'
    model = load_trained_model(model_path, feature_info)
    model = model.to(device)
    print("Model loaded successfully")

    # Create dataset (use subset for speed)
    n_samples = min(2000, len(targets))
    indices = np.random.choice(len(targets), n_samples, replace=False)

    def subset_categorical(cat_features, indices):
        return {name: values[indices] for name, values in cat_features.items()}

    imagery_paths = [f"placeholder_{i}" for i in indices]
    dataset = YewDensityDataset(
        imagery_paths,
        numerical_features[indices],
        subset_categorical(categorical_features, indices),
        targets[indices]
    )

    print(f"\nUsing {len(dataset)} samples for analysis")

    # Output directory
    output_dir = Path('results/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Method 1: Permutation Importance
    print("\n" + "="*70)
    print("METHOD 1: Permutation Importance")
    print("="*70)
    perm_importance, baseline_mae = permutation_importance(
        model, dataset, feature_names, device=device, n_repeats=3
    )

    plot_feature_importance(
        perm_importance,
        'Feature Importance (Permutation Method)\nIncrease in MAE when feature is shuffled',
        output_dir / 'feature_importance_permutation.png'
    )

    # Method 2: Gradient-based Importance
    print("\n" + "="*70)
    print("METHOD 2: Gradient-Based Importance")
    print("="*70)
    grad_importance = gradient_based_importance(
        model, dataset, feature_names, device=device, n_samples=1000
    )

    plot_feature_importance(
        grad_importance,
        'Feature Importance (Gradient-Based)\nAverage absolute gradient magnitude',
        output_dir / 'feature_importance_gradients.png'
    )

    # Method 3: Correlation Analysis
    print("\n" + "="*70)
    print("METHOD 3: Correlation Analysis")
    print("="*70)
    correlations = correlation_analysis(merged_df, feature_names)

    corr_dict = {k: {'mean': v, 'std': 0} for k, v in correlations.items()}
    plot_feature_importance(
        corr_dict,
        'Feature Importance (Correlation)\nAbsolute correlation with yew density',
        output_dir / 'feature_importance_correlation.png'
    )

    # Correlation heatmap
    plot_correlation_heatmap(
        merged_df, feature_names, 'YEW_DENSITY_HA',
        output_dir / 'feature_correlation_heatmap.png'
    )

    # Summary table
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE SUMMARY")
    print("="*70)

    summary_data = []
    for feat in feature_names:
        row = {
            'Feature': feat,
            'Permutation': perm_importance.get(feat, {}).get('mean', 0),
            'Gradient': grad_importance.get(feat, {}).get('mean', 0),
            'Correlation': correlations.get(feat, 0)
        }
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)

    # Normalize scores to 0-1 range for comparison
    for col in ['Permutation', 'Gradient', 'Correlation']:
        max_val = summary_df[col].abs().max()
        if max_val > 0:
            summary_df[f'{col}_norm'] = summary_df[col].abs() / max_val

    # Average rank
    summary_df['Average_Importance'] = (
        summary_df['Permutation_norm'] +
        summary_df['Gradient_norm'] +
        summary_df['Correlation_norm']
    ) / 3

    # Sort by average importance
    summary_df = summary_df.sort_values('Average_Importance', ascending=False)

    print("\nTop 10 Most Important Features:")
    print(summary_df[['Feature', 'Permutation', 'Gradient', 'Correlation',
          'Average_Importance']].head(10).to_string(index=False))

    # Save full summary
    summary_path = 'results/tables/feature_importance_summary.csv'
    Path('results/tables').mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"\nFull summary saved to: {summary_path}")

    # Create combined visualization
    print("\nCreating combined visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    top_n = 10
    for idx, (method, ax) in enumerate(zip(['Permutation', 'Gradient', 'Correlation'], axes)):
        top_features = summary_df.nlargest(top_n, f'{method}_norm')

        y_pos = np.arange(len(top_features))
        values = top_features[method].values

        ax.barh(y_pos, values, alpha=0.7, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['Feature'].values)
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score')
        ax.set_title(
            f'{method} Importance\n(Top {top_n} Features)', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance_combined.png',
                dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'feature_importance_combined.png'}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
