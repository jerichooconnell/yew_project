#!/usr/bin/env python3
"""
Variance Reduction & Feature Selection
======================================

Identifies and removes low-importance features to simplify the model
and reduce overfitting.

Methods:
1. Low variance features (nearly constant)
2. Low importance features (permutation + gradient)
3. Highly correlated features (redundancy removal)
4. Recursive feature elimination

Author: Analysis Tool
Date: October 2025
"""

from train_with_ee_data import load_and_merge_data, prepare_training_data
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
import pickle

import sys
sys.path.append('scripts/training')


def analyze_feature_variance(numerical_features, feature_names, threshold=0.01):
    """
    Identify features with low variance.
    Low variance = feature is nearly constant, not useful for prediction.
    """
    print("\n" + "="*70)
    print("1. LOW VARIANCE FEATURE ANALYSIS")
    print("="*70)

    variances = np.var(numerical_features, axis=0)

    variance_df = pd.DataFrame({
        'Feature': feature_names,
        'Variance': variances,
        'Keep': variances > threshold
    }).sort_values('Variance')

    print(f"\nVariance threshold: {threshold}")
    print(f"\nFeatures sorted by variance:")
    print(variance_df.to_string(index=False))

    low_var_features = variance_df[variance_df['Variance']
                                   <= threshold]['Feature'].tolist()
    print(f"\nLow variance features to remove ({len(low_var_features)}):")
    for feat in low_var_features:
        var_val = variance_df[variance_df['Feature']
                              == feat]['Variance'].values[0]
        print(f"  - {feat}: variance = {var_val:.6f}")

    return variance_df, low_var_features


def analyze_feature_correlations(numerical_features, feature_names, threshold=0.9):
    """
    Identify highly correlated feature pairs.
    High correlation = redundancy, can remove one.
    """
    print("\n" + "="*70)
    print("2. FEATURE CORRELATION ANALYSIS")
    print("="*70)

    # Create correlation matrix
    corr_matrix = np.corrcoef(numerical_features.T)
    corr_df = pd.DataFrame(
        corr_matrix, columns=feature_names, index=feature_names)

    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            if abs(corr_matrix[i, j]) > threshold:
                high_corr_pairs.append({
                    'Feature1': feature_names[i],
                    'Feature2': feature_names[j],
                    'Correlation': corr_matrix[i, j]
                })

    if high_corr_pairs:
        print(f"\nHighly correlated pairs (|r| > {threshold}):")
        for pair in high_corr_pairs:
            print(
                f"  {pair['Feature1']} <-> {pair['Feature2']}: r = {pair['Correlation']:.3f}")
    else:
        print(f"\nNo highly correlated pairs found (|r| > {threshold})")

    return corr_df, high_corr_pairs


def load_importance_scores():
    """Load previously computed importance scores."""
    print("\n" + "="*70)
    print("3. LOADING FEATURE IMPORTANCE SCORES")
    print("="*70)

    importance_path = 'results/tables/feature_importance_summary.csv'
    if not Path(importance_path).exists():
        print(f"Warning: {importance_path} not found!")
        print("Run analyze_feature_importance.py first.")
        return None

    importance_df = pd.read_csv(importance_path)
    print(f"\nLoaded importance scores for {len(importance_df)} features")
    return importance_df


def select_features_by_importance(importance_df, percentile=25):
    """
    Select features based on importance scores.
    Remove bottom X% of features.
    """
    print("\n" + "="*70)
    print("4. IMPORTANCE-BASED FEATURE SELECTION")
    print("="*70)

    threshold = np.percentile(importance_df['Average_Importance'], percentile)

    print(f"\nRemoving features below {percentile}th percentile")
    print(f"Importance threshold: {threshold:.4f}")

    keep_features = importance_df[importance_df['Average_Importance']
                                  >= threshold]['Feature'].tolist()
    remove_features = importance_df[importance_df['Average_Importance']
                                    < threshold]['Feature'].tolist()

    print(f"\nFeatures to keep ({len(keep_features)}):")
    for feat in keep_features:
        imp = importance_df[importance_df['Feature']
                            == feat]['Average_Importance'].values[0]
        print(f"  ✓ {feat}: importance = {imp:.4f}")

    print(f"\nFeatures to remove ({len(remove_features)}):")
    for feat in remove_features:
        imp = importance_df[importance_df['Feature']
                            == feat]['Average_Importance'].values[0]
        print(f"  ✗ {feat}: importance = {imp:.4f}")

    return keep_features, remove_features


def remove_redundant_features(high_corr_pairs, importance_df):
    """
    For highly correlated pairs, keep the more important one.
    """
    print("\n" + "="*70)
    print("5. REMOVING REDUNDANT FEATURES")
    print("="*70)

    if not high_corr_pairs:
        print("No redundant features to remove")
        return []

    redundant_features = set()

    for pair in high_corr_pairs:
        feat1 = pair['Feature1']
        feat2 = pair['Feature2']

        # Get importance scores
        imp1 = importance_df[importance_df['Feature']
                             == feat1]['Average_Importance'].values
        imp2 = importance_df[importance_df['Feature']
                             == feat2]['Average_Importance'].values

        if len(imp1) == 0 or len(imp2) == 0:
            continue

        # Keep the more important feature
        if imp1[0] >= imp2[0]:
            redundant_features.add(feat2)
            print(f"  Removing {feat2} (keeping {feat1})")
            print(f"    Correlation: {pair['Correlation']:.3f}")
            print(
                f"    Importance: {feat1}={imp1[0]:.4f}, {feat2}={imp2[0]:.4f}")
        else:
            redundant_features.add(feat1)
            print(f"  Removing {feat1} (keeping {feat2})")
            print(f"    Correlation: {pair['Correlation']:.3f}")
            print(
                f"    Importance: {feat1}={imp1[0]:.4f}, {feat2}={imp2[0]:.4f}")

    return list(redundant_features)


def create_final_feature_set(all_features, variance_df, remove_by_importance,
                             redundant_features, importance_df):
    """
    Create final recommended feature set.
    """
    print("\n" + "="*70)
    print("6. FINAL FEATURE SET RECOMMENDATION")
    print("="*70)

    # Start with all features
    keep_features = set(all_features)
    removed_reasons = {}

    # Remove low variance
    low_var = variance_df[variance_df['Variance'] <= 0.01]['Feature'].tolist()
    for feat in low_var:
        if feat in keep_features:
            keep_features.remove(feat)
            removed_reasons[feat] = 'Low variance'

    # Remove low importance
    for feat in remove_by_importance:
        if feat in keep_features:
            keep_features.remove(feat)
            removed_reasons[feat] = 'Low importance'

    # Remove redundant
    for feat in redundant_features:
        if feat in keep_features:
            keep_features.remove(feat)
            removed_reasons[feat] = 'Redundant (highly correlated)'

    keep_features = sorted(list(keep_features))

    print(f"\nOriginal features: {len(all_features)}")
    print(f"Recommended features: {len(keep_features)}")
    print(f"Features removed: {len(all_features) - len(keep_features)}")
    print(f"Reduction: {(1 - len(keep_features)/len(all_features))*100:.1f}%")

    print(f"\nFINAL FEATURE SET ({len(keep_features)} features):")
    for feat in keep_features:
        imp = importance_df[importance_df['Feature']
                            == feat]['Average_Importance'].values
        imp_str = f"{imp[0]:.4f}" if len(imp) > 0 else "N/A"
        print(f"  ✓ {feat} (importance: {imp_str})")

    print(f"\nREMOVED FEATURES ({len(removed_reasons)}):")
    for feat, reason in sorted(removed_reasons.items()):
        print(f"  ✗ {feat}: {reason}")

    return keep_features, removed_reasons


def visualize_feature_selection(all_features, keep_features, importance_df,
                                removed_reasons, output_path):
    """
    Create visualization of feature selection results.
    """
    print("\n" + "="*70)
    print("7. CREATING VISUALIZATIONS")
    print("="*70)

    # Create summary dataframe
    summary = []
    for feat in all_features:
        imp = importance_df[importance_df['Feature']
                            == feat]['Average_Importance'].values
        imp_val = imp[0] if len(imp) > 0 else 0

        status = 'Keep' if feat in keep_features else 'Remove'
        reason = removed_reasons.get(feat, 'Selected')

        summary.append({
            'Feature': feat,
            'Importance': imp_val,
            'Status': status,
            'Reason': reason
        })

    summary_df = pd.DataFrame(summary).sort_values(
        'Importance', ascending=False)

    # Plot 1: Feature importance with keep/remove colors
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    # Top plot: All features colored by status
    colors = ['green' if s == 'Keep' else 'red' for s in summary_df['Status']]
    y_pos = np.arange(len(summary_df))

    axes[0].barh(y_pos, summary_df['Importance'],
                 color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(summary_df['Feature'])
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Importance Score', fontsize=12)
    axes[0].set_title('Feature Selection Results\nGreen = Keep, Red = Remove',
                      fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7,
              label=f'Keep ({len(keep_features)} features)'),
        Patch(facecolor='red', alpha=0.7,
              label=f'Remove ({len(removed_reasons)} features)')
    ]
    axes[0].legend(handles=legend_elements, loc='lower right')

    # Bottom plot: Removal reasons
    removed_df = summary_df[summary_df['Status'] == 'Remove']
    if len(removed_df) > 0:
        reason_colors = {
            'Low variance': 'orange',
            'Low importance': 'red',
            'Redundant (highly correlated)': 'purple'
        }
        colors = [reason_colors.get(r, 'gray') for r in removed_df['Reason']]
        y_pos = np.arange(len(removed_df))

        axes[1].barh(y_pos, removed_df['Importance'],
                     color=colors, alpha=0.7, edgecolor='black')
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(removed_df['Feature'])
        axes[1].invert_yaxis()
        axes[1].set_xlabel('Importance Score', fontsize=12)
        axes[1].set_title('Removed Features (Colored by Reason)',
                          fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')

        # Legend
        legend_elements = [
            Patch(facecolor=color, alpha=0.7, label=reason)
            for reason, color in reason_colors.items()
            if reason in removed_df['Reason'].values
        ]
        axes[1].legend(handles=legend_elements, loc='lower right')
    else:
        axes[1].text(0.5, 0.5, 'No features removed',
                     ha='center', va='center', fontsize=16)
        axes[1].set_xticks([])
        axes[1].set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")

    return summary_df


def save_feature_config(keep_features, removed_reasons, output_path):
    """Save the feature selection configuration."""
    config = {
        'selected_features': keep_features,
        'removed_features': removed_reasons,
        'n_selected': len(keep_features),
        'n_removed': len(removed_reasons),
        'reduction_pct': (1 - len(keep_features)/(len(keep_features) + len(removed_reasons))) * 100
    }

    with open(output_path, 'wb') as f:
        pickle.dump(config, f)

    print(f"\nFeature configuration saved to: {output_path}")


def main():
    print("="*70)
    print("VARIANCE REDUCTION & FEATURE SELECTION")
    print("="*70)

    # Load data
    print("\nLoading data...")
    merged_df = load_and_merge_data()
    numerical_features, categorical_features, targets, feature_info = prepare_training_data(
        merged_df)

    if numerical_features is None:
        print("Error loading data!")
        return

    feature_names = feature_info['numerical_cols']
    print(f"\nStarting with {len(feature_names)} numerical features")

    # 1. Analyze variance
    variance_df, low_var_features = analyze_feature_variance(
        numerical_features, feature_names, threshold=0.01
    )

    # 2. Analyze correlations
    corr_df, high_corr_pairs = analyze_feature_correlations(
        numerical_features, feature_names, threshold=0.85
    )

    # 3. Load importance scores
    importance_df = load_importance_scores()
    if importance_df is None:
        print("\nCannot proceed without importance scores. Exiting.")
        return

    # 4. Select by importance (remove bottom 20%)
    keep_by_importance, remove_by_importance = select_features_by_importance(
        importance_df, percentile=20
    )

    # 5. Remove redundant features
    redundant_features = remove_redundant_features(
        high_corr_pairs, importance_df)

    # 6. Create final feature set
    keep_features, removed_reasons = create_final_feature_set(
        feature_names, variance_df, remove_by_importance,
        redundant_features, importance_df
    )

    # 7. Visualize
    output_dir = Path('results/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = visualize_feature_selection(
        feature_names, keep_features, importance_df, removed_reasons,
        output_dir / 'feature_selection_results.png'
    )

    # Save results
    tables_dir = Path('results/tables')
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(
        tables_dir / 'feature_selection_summary.csv', index=False)
    print(f"Saved: {tables_dir / 'feature_selection_summary.csv'}")

    save_feature_config(keep_features, removed_reasons,
                        'models/artifacts/selected_features.pkl')

    # Print final summary
    print("\n" + "="*70)
    print("FEATURE REDUCTION SUMMARY")
    print("="*70)
    print(f"\nOriginal features: {len(feature_names)}")
    print(f"Selected features: {len(keep_features)}")
    print(f"Removed features: {len(removed_reasons)}")
    print(
        f"Variance reduction: {(1 - len(keep_features)/len(feature_names))*100:.1f}%")

    print("\n" + "="*70)
    print("RECOMMENDED NEXT STEPS")
    print("="*70)
    print("""
1. Retrain the model using only the selected features:
   - Update train_with_ee_data.py to use selected_features.pkl
   - This should improve model performance and reduce overfitting
   
2. Compare model performance:
   - Original model (15 features) vs. Reduced model (X features)
   - Check if validation metrics improve
   
3. The selected features are saved in:
   models/artifacts/selected_features.pkl
   
4. Use this configuration for all future predictions
    """)

    print("="*70)


if __name__ == "__main__":
    main()
