#!/usr/bin/env python3
"""
Feature Importance Analysis for ResNet Transfer Learning Model
==============================================================

Analyzes which features are most important for Pacific Yew prediction
in the ResNet-based transfer learning model.

Methods:
1. Permutation importance - shuffle each feature and measure performance drop
2. Gradient-based importance - analyze gradients with respect to inputs
3. Correlation analysis - examine feature correlations with target

Author: Analysis Tool
Date: October 28, 2025
"""

import sys
sys.path.append('scripts/training')

from train_yew_model_with_ee import (
    load_and_merge_data, create_target_variable, prepare_features,
    create_simple_tabular_model, SimpleYewDataset
)
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from tqdm import tqdm


def load_trained_model(model_path, feature_info):
    """Load the trained ResNet model."""
    model = create_simple_tabular_model(
        feature_info['num_numerical'],
        feature_info['categorical_dims'],
        hidden_dims=[256, 128, 64]
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model


def permutation_importance(model, test_loader, device, feature_names, n_repeats=5):
    """Calculate permutation importance for each feature."""
    print("\nCalculating permutation importance...")
    model = model.to(device)
    model.eval()
    
    # Get baseline performance
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for numerical, categorical, targets in test_loader:
            numerical = numerical.to(device)
            categorical = {k: v.to(device) for k, v in categorical.items()}
            
            logits = model(numerical, categorical)
            probs = torch.sigmoid(logits)
            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(targets.numpy())
    
    baseline_preds = np.array(all_preds)
    targets_np = np.array(all_targets)
    baseline_f1 = f1_score(targets_np, (baseline_preds > 0.5).astype(int), zero_division=0)
    
    print(f"Baseline F1: {baseline_f1:.4f}")
    
    # Calculate importance for each feature
    importances = {}
    
    for feat_idx, feat_name in enumerate(tqdm(feature_names, desc="Features")):
        importance_scores = []
        
        for _ in range(n_repeats):
            permuted_preds = []
            
            for numerical, categorical, targets in test_loader:
                # Permute the feature
                numerical_permuted = numerical.clone()
                numerical_permuted[:, feat_idx] = numerical_permuted[
                    torch.randperm(numerical_permuted.size(0)), feat_idx
                ]
                
                numerical_permuted = numerical_permuted.to(device)
                categorical = {k: v.to(device) for k, v in categorical.items()}
                
                with torch.no_grad():
                    logits = model(numerical_permuted, categorical)
                    probs = torch.sigmoid(logits)
                    permuted_preds.extend(probs.cpu().numpy())
            
            permuted_preds = np.array(permuted_preds)
            permuted_f1 = f1_score(targets_np, (permuted_preds > 0.5).astype(int), zero_division=0)
            importance_scores.append(baseline_f1 - permuted_f1)
        
        importances[feat_name] = {
            'mean': np.mean(importance_scores),
            'std': np.std(importance_scores)
        }
    
    return importances


def gradient_based_importance(model, test_loader, device, feature_names):
    """Calculate gradient-based feature importance."""
    print("\nCalculating gradient-based importance...")
    model = model.to(device)
    model.eval()
    
    gradients = {name: [] for name in feature_names}
    
    for numerical, categorical, targets in tqdm(test_loader, desc="Batches"):
        numerical = numerical.to(device)
        numerical.requires_grad = True
        categorical = {k: v.to(device) for k, v in categorical.items()}
        targets = targets.to(device)
        
        # Forward pass
        logits = model(numerical, categorical)
        loss = nn.BCEWithLogitsLoss()(logits.squeeze(), targets.float())
        
        # Backward pass
        loss.backward()
        
        # Store gradients
        if numerical.grad is not None:
            grads = numerical.grad.abs().cpu().numpy()
            for i, name in enumerate(feature_names):
                gradients[name].extend(grads[:, i])
        
        numerical.grad = None
    
    # Calculate mean absolute gradient for each feature
    importance = {}
    for name in feature_names:
        importance[name] = np.mean(gradients[name])
    
    return importance


def correlation_analysis(merged_df, feature_names):
    """Analyze correlation between features and target."""
    print("\nCalculating correlations with target...")
    
    correlations = {}
    for feat in feature_names:
        if feat in merged_df.columns:
            corr = merged_df[feat].corr(merged_df['has_yew'])
            correlations[feat] = abs(corr)
    
    return correlations


def plot_importance_comparison(perm_importance, grad_importance, correlations, output_path):
    """Create comprehensive importance visualization."""
    # Prepare data
    features = list(perm_importance.keys())
    perm_scores = [perm_importance[f]['mean'] for f in features]
    perm_stds = [perm_importance[f]['std'] for f in features]
    grad_scores = [grad_importance.get(f, 0) for f in features]
    corr_scores = [correlations.get(f, 0) for f in features]
    
    # Normalize scores
    perm_norm = np.array(perm_scores)
    if perm_norm.max() > 0:
        perm_norm = perm_norm / perm_norm.max()
    
    grad_norm = np.array(grad_scores)
    if grad_norm.max() > 0:
        grad_norm = grad_norm / grad_norm.max()
    
    corr_norm = np.array(corr_scores)
    if corr_norm.max() > 0:
        corr_norm = corr_norm / corr_norm.max()
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Permutation Importance
    ax = axes[0, 0]
    sorted_idx = np.argsort(perm_scores)
    y_pos = np.arange(len(features))
    ax.barh(y_pos, np.array(perm_scores)[sorted_idx], 
            xerr=np.array(perm_stds)[sorted_idx],
            color='steelblue', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([features[i] for i in sorted_idx])
    ax.set_xlabel('Importance (F1 drop)')
    ax.set_title('Permutation Importance')
    ax.grid(axis='x', alpha=0.3)
    
    # 2. Gradient-based Importance
    ax = axes[0, 1]
    sorted_idx = np.argsort(grad_scores)
    ax.barh(y_pos, np.array(grad_scores)[sorted_idx], 
            color='coral', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([features[i] for i in sorted_idx])
    ax.set_xlabel('Mean Absolute Gradient')
    ax.set_title('Gradient-based Importance')
    ax.grid(axis='x', alpha=0.3)
    
    # 3. Correlation with Target
    ax = axes[1, 0]
    sorted_idx = np.argsort(corr_scores)
    ax.barh(y_pos, np.array(corr_scores)[sorted_idx], 
            color='seagreen', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([features[i] for i in sorted_idx])
    ax.set_xlabel('Absolute Correlation')
    ax.set_title('Correlation with Yew Presence')
    ax.grid(axis='x', alpha=0.3)
    
    # 4. Combined normalized scores
    ax = axes[1, 1]
    df_combined = pd.DataFrame({
        'Feature': features,
        'Permutation': perm_norm,
        'Gradient': grad_norm,
        'Correlation': corr_norm
    })
    df_combined['Average'] = df_combined[['Permutation', 'Gradient', 'Correlation']].mean(axis=1)
    df_combined = df_combined.sort_values('Average')
    
    x = np.arange(len(features))
    width = 0.25
    ax.barh(x - width, df_combined['Permutation'], width, label='Permutation', alpha=0.7)
    ax.barh(x, df_combined['Gradient'], width, label='Gradient', alpha=0.7)
    ax.barh(x + width, df_combined['Correlation'], width, label='Correlation', alpha=0.7)
    ax.set_yticks(x)
    ax.set_yticklabels(df_combined['Feature'])
    ax.set_xlabel('Normalized Importance')
    ax.set_title('Combined Feature Importance (Normalized)')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nImportance plot saved to {output_path}")
    
    return df_combined


def analyze_satellite_vs_inventory(perm_importance, feature_names):
    """Compare importance of satellite vs inventory features."""
    satellite_features = ['blue', 'green', 'red', 'nir', 'ndvi', 'evi', 
                          'elevation', 'slope', 'aspect']
    
    satellite_importance = []
    inventory_importance = []
    
    for feat in feature_names:
        importance = perm_importance[feat]['mean']
        if feat in satellite_features:
            satellite_importance.append(importance)
        else:
            inventory_importance.append(importance)
    
    print("\n" + "="*70)
    print("Satellite vs Inventory Features")
    print("="*70)
    print(f"Satellite features (n={len(satellite_importance)}):")
    print(f"  Mean importance: {np.mean(satellite_importance):.6f}")
    print(f"  Total importance: {np.sum(satellite_importance):.6f}")
    print(f"\nInventory features (n={len(inventory_importance)}):")
    print(f"  Mean importance: {np.mean(inventory_importance):.6f}")
    print(f"  Total importance: {np.sum(inventory_importance):.6f}")


def main():
    """Main analysis pipeline."""
    print("="*70)
    print("ResNet Transfer Learning - Feature Importance Analysis")
    print("="*70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    merged_df = load_and_merge_data(
        'data/ee_imagery/ee_extraction_progress.csv',
        'data/raw/bc_sample_data-2025-10-09/bc_sample_data.csv'
    )
    
    merged_df = create_target_variable(merged_df)
    numerical_features, categorical_features, targets, feature_info = prepare_features(merged_df)
    
    # Load model
    print("\nLoading trained model...")
    model_path = 'models/checkpoints/best_yew_model_ee.pth'
    model = load_trained_model(model_path, feature_info)
    print("Model loaded successfully")
    
    # Create test dataset
    from train_yew_model_with_ee import spatial_train_test_split
    
    train_idx, val_idx, test_idx = spatial_train_test_split(
        feature_info['coordinates'],
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    def subset_cat(cat_features, indices):
        return {name: values[indices] for name, values in cat_features.items()}
    
    test_dataset = SimpleYewDataset(
        numerical_features[test_idx],
        subset_cat(categorical_features, test_idx),
        targets[test_idx]
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=2
    )
    
    # Get feature names
    ee_features = ['blue', 'green', 'red', 'nir', 'ndvi', 'evi',
                   'elevation', 'slope', 'aspect']
    inventory_features = ['BA_HA_LS', 'STEMS_HA_LS', 'VHA_WSV_LS',
                          'SI_M_TLSO', 'HT_TLSO', 'AGEB_TLSO']
    feature_names = ee_features + inventory_features
    
    # Calculate importance scores
    perm_importance = permutation_importance(
        model, test_loader, device, feature_names, n_repeats=3
    )
    
    grad_importance = gradient_based_importance(
        model, test_loader, device, feature_names
    )
    
    correlations = correlation_analysis(merged_df, feature_names)
    
    # Print results
    print("\n" + "="*70)
    print("Feature Importance Summary")
    print("="*70)
    print(f"{'Feature':<20} {'Permutation':<15} {'Gradient':<15} {'Correlation':<15}")
    print("-"*70)
    
    for feat in feature_names:
        perm = perm_importance[feat]['mean']
        grad = grad_importance.get(feat, 0)
        corr = correlations.get(feat, 0)
        print(f"{feat:<20} {perm:<15.6f} {grad:<15.6f} {corr:<15.6f}")
    
    # Analyze satellite vs inventory
    analyze_satellite_vs_inventory(perm_importance, feature_names)
    
    # Create visualizations
    Path('results/figures').mkdir(parents=True, exist_ok=True)
    df_combined = plot_importance_comparison(
        perm_importance, grad_importance, correlations,
        'results/figures/resnet_feature_importance.png'
    )
    
    # Save results to CSV
    Path('results/tables').mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame({
        'feature': feature_names,
        'permutation_importance': [perm_importance[f]['mean'] for f in feature_names],
        'permutation_std': [perm_importance[f]['std'] for f in feature_names],
        'gradient_importance': [grad_importance.get(f, 0) for f in feature_names],
        'correlation': [correlations.get(f, 0) for f in feature_names]
    })
    results_df = results_df.sort_values('permutation_importance', ascending=False)
    results_df.to_csv('results/tables/resnet_feature_importance.csv', index=False)
    print("\nResults saved to results/tables/resnet_feature_importance.csv")
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
