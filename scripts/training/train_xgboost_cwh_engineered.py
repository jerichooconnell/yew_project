#!/usr/bin/env python3
"""
Pacific Yew XGBoost - CWH Zone-Specific Model with Engineered Features
======================================================================

Trains an XGBoost model specifically for the CWH (Coastal Western Hemlock) zone
with engineered features based on forestry domain knowledge.

CWH zone contains 182 of 234 yew plots (77.8% of all yew).

Author: Analysis Tool
Date: October 28, 2025
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
from pathlib import Path
import warnings
import re
warnings.filterwarnings('ignore')


def parse_species_composition(composition_string):
    """Parse species composition string to extract Pacific Yew (TW) percentage."""
    if not composition_string or pd.isna(composition_string):
        return 0

    pattern = r'TW(\d{2,3})'  # TW = Taxus brevifolia (Pacific Yew)
    match = re.search(pattern, str(composition_string))

    if match:
        return int(match.group(1))
    return 0


def load_and_merge_data(ee_data_path, inventory_path):
    """Load and merge Earth Engine data with forest inventory."""
    print("Loading Earth Engine and inventory data...")

    # Load EE data
    ee_df = pd.read_csv(ee_data_path)
    print(f"  Earth Engine data: {len(ee_df)} records")

    # Load inventory
    inv_df = pd.read_csv(inventory_path, low_memory=False)
    print(f"  Forest inventory: {len(inv_df)} records")

    # Merge on SITE_IDENTIFIER
    ee_df['SITE_IDENTIFIER'] = ee_df['plot_id'].astype(str)
    inv_df['SITE_IDENTIFIER'] = inv_df['SITE_IDENTIFIER'].astype(str)

    merged_df = inv_df.merge(ee_df, on='SITE_IDENTIFIER',
                             how='inner', suffixes=('', '_ee'))

    print(f"  Merged dataset: {len(merged_df)} records")

    # Calculate yew percentage from species composition
    merged_df['YEW_PERCENTAGE'] = merged_df['SPB_CPCT_LS'].apply(
        parse_species_composition)

    # Get coordinates
    if 'x' in merged_df.columns and 'y' in merged_df.columns:
        merged_df['BC_ALBERS_X'] = merged_df['x']
        merged_df['BC_ALBERS_Y'] = merged_df['y']

    return merged_df


def engineer_features(df):
    """
    Create engineered features based on forestry domain knowledge.

    Returns dataframe with additional features:
    - Structural diversity metrics (dead/live ratios)
    - Stand productivity indicators
    - Site-relative metrics
    - Composite complexity index
    - Categorical binning
    - Log transforms
    """
    df = df.copy()

    print("\nEngineering features...")

    # 1. STRUCTURAL DIVERSITY METRICS (Dead wood / Live wood ratios)
    # Dead wood is important for yew habitat
    df['BA_RATIO'] = df['BA_HA_DS'].fillna(0) / (df['BA_HA_LS'].fillna(0) + 1)
    df['STEMS_RATIO'] = df['STEMS_HA_DS'].fillna(
        0) / (df['STEMS_HA_LS'].fillna(0) + 1)
    df['VOLUME_RATIO'] = df['VHA_WSV_DS'].fillna(
        0) / (df['VHA_WSV_LS'].fillna(0) + 1)

    # 2. STAND PRODUCTIVITY INDICATORS
    # Volume per stem indicates tree size
    df['VOLUME_PER_STEM'] = df['VHA_WSV_LS'].fillna(
        0) / (df['STEMS_HA_LS'].fillna(0) + 1)
    df['BA_PER_STEM'] = df['BA_HA_LS'].fillna(
        0) / (df['STEMS_HA_LS'].fillna(0) + 1)

    # Age to height ratio indicates growth rate
    df['AGE_HEIGHT_RATIO'] = df['AGEB_TLSO'].fillna(
        0) / (df['HT_TLSO'].fillna(0) + 1)

    # Height to site index ratio
    df['HEIGHT_SI_RATIO'] = df['HT_TLSO'].fillna(
        0) / (df['SI_M_TLSO'].fillna(0) + 1)

    # 3. SITE-RELATIVE METRICS
    # Basal area per unit of site productivity
    df['BA_PER_SI'] = df['BA_HA_LS'].fillna(
        0) / (df['SI_M_TLSO'].fillna(0) + 1)
    df['STEMS_PER_SI'] = df['STEMS_HA_LS'].fillna(
        0) / (df['SI_M_TLSO'].fillna(0) + 1)

    # 4. COMPOSITE COMPLEXITY INDEX
    # Combines multiple indicators of old-growth/complex structure
    # Higher values = more complex structure (good for yew)
    df['STRUCTURE_INDEX'] = (
        (df['BA_RATIO'] + 1) *  # More dead wood
        (df['VOLUME_PER_STEM'] / 100) *  # Larger trees
        (df['AGE_HEIGHT_RATIO'] / 10)  # Older relative to height
    )

    # 5. CATEGORICAL BINNING
    # Age classes
    df['AGE_CLASS'] = pd.cut(
        df['AGEB_TLSO'].fillna(0),
        bins=[0, 40, 80, 120, 250, 999],
        labels=['young', 'mature', 'old', 'very_old', 'ancient']
    )

    # Height classes
    df['HEIGHT_CLASS'] = pd.cut(
        df['HT_TLSO'].fillna(0),
        bins=[0, 10, 20, 30, 100],
        labels=['short', 'medium', 'tall', 'very_tall']
    )

    # Site index classes
    df['SI_CLASS'] = pd.cut(
        df['SI_M_TLSO'].fillna(0),
        bins=[0, 15, 25, 35, 100],
        labels=['poor', 'medium', 'good', 'excellent']
    )

    # 6. LOG TRANSFORMS (for skewed distributions)
    df['LOG_BA'] = np.log1p(df['BA_HA_LS'].fillna(0))
    df['LOG_STEMS'] = np.log1p(df['STEMS_HA_LS'].fillna(0))
    df['LOG_VOLUME'] = np.log1p(df['VHA_WSV_LS'].fillna(0))

    # 7. TOTAL METRICS (Live + Dead)
    df['TOTAL_BA'] = df['BA_HA_LS'].fillna(0) + df['BA_HA_DS'].fillna(0)
    df['TOTAL_STEMS'] = df['STEMS_HA_LS'].fillna(
        0) + df['STEMS_HA_DS'].fillna(0)
    df['TOTAL_VOLUME'] = df['VHA_WSV_LS'].fillna(
        0) + df['VHA_WSV_DS'].fillna(0)

    num_engineered = 19
    print(f"  Created {num_engineered} engineered features")
    print("    - Structural diversity: BA_RATIO, STEMS_RATIO, VOLUME_RATIO")
    print("    - Productivity: VOLUME_PER_STEM, BA_PER_STEM, AGE_HEIGHT_RATIO, HEIGHT_SI_RATIO")
    print("    - Site-relative: BA_PER_SI, STEMS_PER_SI")
    print("    - Complexity: STRUCTURE_INDEX")
    print("    - Categorical: AGE_CLASS, HEIGHT_CLASS, SI_CLASS")
    print("    - Log transforms: LOG_BA, LOG_STEMS, LOG_VOLUME")
    print("    - Totals: TOTAL_BA, TOTAL_STEMS, TOTAL_VOLUME")

    return df


def prepare_features(df, use_engineered=True):
    """Prepare features for training."""
    print("\nPreparing features...")

    # Base numerical features (from original best model)
    base_numerical = [
        'BA_HA_LS', 'STEMS_HA_LS', 'VHA_WSV_LS',
        'SI_M_TLSO', 'HT_TLSO', 'AGEB_TLSO'
    ]

    # Engineered numerical features
    engineered_numerical = [
        'BA_RATIO', 'STEMS_RATIO', 'VOLUME_RATIO',
        'VOLUME_PER_STEM', 'BA_PER_STEM', 'AGE_HEIGHT_RATIO', 'HEIGHT_SI_RATIO',
        'BA_PER_SI', 'STEMS_PER_SI', 'STRUCTURE_INDEX',
        'LOG_BA', 'LOG_STEMS', 'LOG_VOLUME',
        'TOTAL_BA', 'TOTAL_STEMS', 'TOTAL_VOLUME'
    ]

    # Engineered categorical features
    engineered_categorical = ['AGE_CLASS', 'HEIGHT_CLASS', 'SI_CLASS']

    if use_engineered:
        numerical_cols = base_numerical + engineered_numerical
        categorical_cols = engineered_categorical
        print(f"  Using ENGINEERED features:")
        print(f"    {len(numerical_cols)} numerical features")
        print(f"    {len(categorical_cols)} categorical features")
    else:
        numerical_cols = base_numerical
        categorical_cols = []
        print(f"  Using BASELINE features:")
        print(f"    {len(numerical_cols)} numerical features")
        print(f"    0 categorical features")

    # Fill missing values
    for col in numerical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Extract features
    X_numerical = df[numerical_cols].values

    # Handle categorical features
    if categorical_cols:
        # One-hot encode categorical features
        X_categorical = pd.get_dummies(
            df[categorical_cols],
            prefix=categorical_cols,
            drop_first=False  # Keep all categories for XGBoost
        )

        # Combine numerical and categorical
        X = np.concatenate([X_numerical, X_categorical.values], axis=1)

        feature_names = numerical_cols + list(X_categorical.columns)
    else:
        X = X_numerical
        feature_names = numerical_cols

    print(f"  Total features: {X.shape[1]}")

    return X, feature_names


def spatial_train_val_test_split(coordinates, test_size=0.2, val_size=0.1, random_state=42):
    """Split data spatially using K-means clustering."""
    n_samples = len(coordinates)
    n_test = int(n_samples * test_size)
    n_val = int(n_samples * val_size)
    n_train = n_samples - n_test - n_val

    # Use K-means to create spatial clusters
    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters,
                    random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(coordinates)

    # Assign clusters to train/val/test
    np.random.seed(random_state)
    clusters = np.random.permutation(n_clusters)

    n_test_clusters = max(1, int(n_clusters * test_size))
    n_val_clusters = max(1, int(n_clusters * val_size))

    test_clusters = clusters[:n_test_clusters]
    val_clusters = clusters[n_test_clusters:n_test_clusters + n_val_clusters]
    train_clusters = clusters[n_test_clusters + n_val_clusters:]

    train_idx = np.where(np.isin(cluster_labels, train_clusters))[0]
    val_idx = np.where(np.isin(cluster_labels, val_clusters))[0]
    test_idx = np.where(np.isin(cluster_labels, test_clusters))[0]

    return train_idx, val_idx, test_idx


def train_and_evaluate(X, y, train_idx, val_idx, test_idx, feature_names, zone_name):
    """Train XGBoost and evaluate at multiple thresholds."""

    # Split data
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"\n{zone_name} - Dataset sizes:")
    print(
        f"  Train: {len(y_train)} samples, {y_train.sum()} yew ({100*y_train.mean():.2f}%)")
    print(
        f"  Val:   {len(y_val)} samples, {y_val.sum()} yew ({100*y_val.mean():.2f}%)")
    print(
        f"  Test:  {len(y_test)} samples, {y_test.sum()} yew ({100*y_test.mean():.2f}%)")

    # Calculate class imbalance
    num_no_yew = (y_train == 0).sum()
    num_yew = (y_train == 1).sum()
    if num_yew > 0:
        scale_pos_weight = num_no_yew / num_yew
    else:
        scale_pos_weight = 1.0

    print(f"  Class imbalance: {scale_pos_weight:.1f}:1")

    # Train XGBoost
    print(f"\n{zone_name} - Training XGBoost...")

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'tree_method': 'hist'
    }

    evals = [(dtrain, 'train'), (dval, 'val')]
    evals_result = {}

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=False,
        evals_result=evals_result
    )

    best_iteration = model.best_iteration
    print(f"  Best iteration: {best_iteration}")

    # Predict on test set
    y_pred_proba = model.predict(dtest)

    # Evaluate at multiple thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6]

    print(f"\n{zone_name} - Test Set Performance:")
    print(f"  ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(
        f"  Average Precision: {average_precision_score(y_test, y_pred_proba):.4f}")
    print(f"\n  Performance at different thresholds:")
    print(f"  {'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print(f"  {'-'*48}")

    results = []
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"  {threshold:<12.1f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")

        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

    # Feature importance
    importance = model.get_score(importance_type='gain')
    importance_df = pd.DataFrame([
        {'feature': k, 'importance': v}
        for k, v in importance.items()
    ]).sort_values('importance', ascending=False)

    # Normalize importance to percentages
    total_importance = importance_df['importance'].sum()
    importance_df['importance_pct'] = 100 * \
        importance_df['importance'] / total_importance

    print(f"\n{zone_name} - Top 10 Most Important Features:")
    for idx, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']:<25} {row['importance_pct']:>6.2f}%")

    return {
        'model': model,
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'avg_precision': average_precision_score(y_test, y_pred_proba),
        'results': results,
        'importance': importance_df,
        'best_iteration': best_iteration
    }


def print_comparison(baseline_results, engineered_results):
    """Print side-by-side comparison of baseline vs engineered."""
    print("\n" + "="*80)
    print("BASELINE vs ENGINEERED FEATURES COMPARISON - CWH ZONE")
    print("="*80)

    print("\nBASELINE MODEL (6 features):")
    print(f"  ROC AUC: {baseline_results['roc_auc']:.4f}")
    print(f"  Average Precision: {baseline_results['avg_precision']:.4f}")
    print(f"\n  Threshold Performance:")
    print(f"  {'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print(f"  {'-'*48}")
    for r in baseline_results['results']:
        print(
            f"  {r['threshold']:<12.1f} {r['precision']:<12.4f} {r['recall']:<12.4f} {r['f1']:<12.4f}")

    print("\n" + "-"*80)

    print("\nENGINEERED FEATURES MODEL (22 features):")
    print(f"  ROC AUC: {engineered_results['roc_auc']:.4f}")
    print(f"  Average Precision: {engineered_results['avg_precision']:.4f}")
    print(f"\n  Threshold Performance:")
    print(f"  {'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print(f"  {'-'*48}")
    for r in engineered_results['results']:
        print(
            f"  {r['threshold']:<12.1f} {r['precision']:<12.4f} {r['recall']:<12.4f} {r['f1']:<12.4f}")

    print("\n" + "-"*80)

    # Calculate improvements
    roc_improvement = 100 * \
        (engineered_results['roc_auc'] -
         baseline_results['roc_auc']) / baseline_results['roc_auc']
    ap_improvement = 100 * (engineered_results['avg_precision'] -
                            baseline_results['avg_precision']) / baseline_results['avg_precision']

    print("\nIMPROVEMENTS:")
    print(
        f"  ROC AUC: {baseline_results['roc_auc']:.4f} → {engineered_results['roc_auc']:.4f} ({roc_improvement:+.1f}%)")
    print(
        f"  Average Precision: {baseline_results['avg_precision']:.4f} → {engineered_results['avg_precision']:.4f} ({ap_improvement:+.1f}%)")

    print("\n  At threshold 0.5:")
    baseline_05 = [r for r in baseline_results['results']
                   if r['threshold'] == 0.5][0]
    engineered_05 = [r for r in engineered_results['results']
                     if r['threshold'] == 0.5][0]

    prec_improvement = 100 * (engineered_05['precision'] - baseline_05['precision']) / \
        baseline_05['precision'] if baseline_05['precision'] > 0 else 0
    recall_improvement = 100 * (engineered_05['recall'] - baseline_05['recall']) / \
        baseline_05['recall'] if baseline_05['recall'] > 0 else 0
    f1_improvement = 100 * (engineered_05['f1'] - baseline_05['f1']) / \
        baseline_05['f1'] if baseline_05['f1'] > 0 else 0

    print(
        f"    Precision: {baseline_05['precision']:.4f} → {engineered_05['precision']:.4f} ({prec_improvement:+.1f}%)")
    print(
        f"    Recall: {baseline_05['recall']:.4f} → {engineered_05['recall']:.4f} ({recall_improvement:+.1f}%)")
    print(
        f"    F1: {baseline_05['f1']:.4f} → {engineered_05['f1']:.4f} ({f1_improvement:+.1f}%)")

    print("\n" + "="*80)


def main():
    """Main training pipeline."""
    print("="*80)
    print("Pacific Yew XGBoost - CWH Zone-Specific with Engineered Features")
    print("="*80)

    # Load and merge data
    df = load_and_merge_data(
        'data/ee_imagery/ee_extraction_progress.csv',
        'data/raw/bc_sample_data-2025-10-09/bc_sample_data.csv'
    )

    # Filter to CWH zone only
    print("\nFiltering to CWH zone...")
    df_cwh = df[df['BEC_ZONE'] == 'CWH'].copy()
    print(f"  CWH records: {len(df_cwh)}")

    # Create target
    df_cwh['has_yew'] = (df_cwh['YEW_PERCENTAGE'] > 0).astype(int)
    num_yew = df_cwh['has_yew'].sum()
    print(
        f"  Yew plots in CWH: {num_yew} ({100*df_cwh['has_yew'].mean():.2f}%)")

    if num_yew < 10:
        print(
            f"\nWARNING: Only {num_yew} yew plots in CWH zone - may not be enough for reliable training!")

    # Engineer features
    df_cwh = engineer_features(df_cwh)

    # Get coordinates for spatial splitting
    if 'BC_ALBERS_X' in df_cwh.columns:
        coordinates = df_cwh[['BC_ALBERS_X', 'BC_ALBERS_Y']].values
    else:
        coordinates = df_cwh[['x', 'y']].values

    # Create spatial splits (same for both models)
    print("\nCreating spatial train/val/test splits...")
    train_idx, val_idx, test_idx = spatial_train_val_test_split(
        coordinates,
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )

    # Target variable
    y = df_cwh['has_yew'].values

    print("\n" + "="*80)
    print("TRAINING BASELINE MODEL (6 features)")
    print("="*80)

    # Prepare baseline features
    X_baseline, feature_names_baseline = prepare_features(
        df_cwh, use_engineered=False)

    # Train baseline
    baseline_results = train_and_evaluate(
        X_baseline, y, train_idx, val_idx, test_idx,
        feature_names_baseline, "CWH Baseline"
    )

    print("\n" + "="*80)
    print("TRAINING ENGINEERED FEATURES MODEL (22 features)")
    print("="*80)

    # Prepare engineered features
    X_engineered, feature_names_engineered = prepare_features(
        df_cwh, use_engineered=True)

    # Train engineered
    engineered_results = train_and_evaluate(
        X_engineered, y, train_idx, val_idx, test_idx,
        feature_names_engineered, "CWH Engineered"
    )

    # Print comparison
    print_comparison(baseline_results, engineered_results)

    # Save results
    print("\nSaving results...")

    # Save comparison table
    comparison_data = []
    for threshold in [0.3, 0.4, 0.5, 0.6]:
        baseline_r = [r for r in baseline_results['results']
                      if r['threshold'] == threshold][0]
        engineered_r = [r for r in engineered_results['results']
                        if r['threshold'] == threshold][0]

        comparison_data.append({
            'zone': 'CWH',
            'model': 'baseline',
            'threshold': threshold,
            'precision': baseline_r['precision'],
            'recall': baseline_r['recall'],
            'f1': baseline_r['f1'],
            'roc_auc': baseline_results['roc_auc'],
            'avg_precision': baseline_results['avg_precision']
        })

        comparison_data.append({
            'zone': 'CWH',
            'model': 'engineered',
            'threshold': threshold,
            'precision': engineered_r['precision'],
            'recall': engineered_r['recall'],
            'f1': engineered_r['f1'],
            'roc_auc': engineered_results['roc_auc'],
            'avg_precision': engineered_results['avg_precision']
        })

    comparison_df = pd.DataFrame(comparison_data)
    Path('results/tables').mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(
        'results/tables/cwh_feature_engineering_comparison.csv', index=False)
    print("  Saved: results/tables/cwh_feature_engineering_comparison.csv")

    # Save feature importance
    engineered_results['importance'].to_csv(
        'results/tables/cwh_engineered_feature_importance.csv',
        index=False
    )
    print("  Saved: results/tables/cwh_engineered_feature_importance.csv")

    # Save model
    Path('models/checkpoints').mkdir(parents=True, exist_ok=True)
    engineered_results['model'].save_model(
        'models/checkpoints/xgboost_cwh_engineered.json')
    print("  Saved: models/checkpoints/xgboost_cwh_engineered.json")

    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)


if __name__ == "__main__":
    main()
