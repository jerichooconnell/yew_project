#!/usr/bin/env python3
"""
XGBoost with Engineered Features
=================================

Tests impact of domain-informed feature engineering:
1. Structural diversity metrics (dead/live ratios)
2. Stand productivity indicators (volume per stem, age/height)
3. Site quality combinations
4. Interaction terms

Author: Analysis Tool
Date: October 28, 2025
"""

from collections import defaultdict
import pickle
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from train_yew_model_with_ee import (
    load_and_merge_data, create_target_variable
)
import sys
sys.path.append('scripts/training')


def engineer_features(df):
    """Create derived features based on forestry domain knowledge."""
    print("\nEngineering features...")

    df_eng = df.copy()

    # 1. STRUCTURAL DIVERSITY METRICS
    # Dead to live ratios indicate stand disturbance and structural complexity
    # Yew often grows in structurally complex stands

    df_eng['BA_RATIO'] = np.where(
        df_eng['BA_HA_LS'] > 0,
        df_eng['BA_HA_DS'] / (df_eng['BA_HA_LS'] + 1),
        0
    )

    df_eng['STEMS_RATIO'] = np.where(
        df_eng['STEMS_HA_LS'] > 0,
        df_eng['STEMS_HA_DS'] / (df_eng['STEMS_HA_LS'] + 1),
        0
    )

    df_eng['VOLUME_RATIO'] = np.where(
        df_eng['VHA_WSV_LS'] > 0,
        df_eng['VHA_WSV_DS'] / (df_eng['VHA_WSV_LS'] + 1),
        0
    )

    # Total stand metrics (live + dead)
    df_eng['TOTAL_BA'] = df_eng['BA_HA_LS'] + df_eng['BA_HA_DS']
    df_eng['TOTAL_STEMS'] = df_eng['STEMS_HA_LS'] + df_eng['STEMS_HA_DS']
    df_eng['TOTAL_VOLUME'] = df_eng['VHA_WSV_LS'] + df_eng['VHA_WSV_DS']

    # 2. STAND PRODUCTIVITY INDICATORS
    # Volume per stem indicates tree size distribution
    # Age/height ratio indicates site quality and stand history

    df_eng['VOLUME_PER_STEM'] = np.where(
        df_eng['STEMS_HA_LS'] > 0,
        df_eng['VHA_WSV_LS'] / df_eng['STEMS_HA_LS'],
        0
    )

    df_eng['BA_PER_STEM'] = np.where(
        df_eng['STEMS_HA_LS'] > 0,
        df_eng['BA_HA_LS'] / df_eng['STEMS_HA_LS'],
        0
    )

    # Age/height ratio (higher = slower growing, potentially older stands)
    df_eng['AGE_HEIGHT_RATIO'] = np.where(
        df_eng['HT_TLSO'] > 0,
        df_eng['AGEB_TLSO'] / df_eng['HT_TLSO'],
        0
    )

    # Height/site index ratio (growth performance relative to potential)
    df_eng['HEIGHT_SI_RATIO'] = np.where(
        df_eng['SI_M_TLSO'] > 0,
        df_eng['HT_TLSO'] / df_eng['SI_M_TLSO'],
        0
    )

    # 3. STAND DENSITY METRICS
    # Relative density compared to site potential

    df_eng['BA_PER_SI'] = np.where(
        df_eng['SI_M_TLSO'] > 0,
        df_eng['BA_HA_LS'] / df_eng['SI_M_TLSO'],
        0
    )

    df_eng['STEMS_PER_SI'] = np.where(
        df_eng['SI_M_TLSO'] > 0,
        df_eng['STEMS_HA_LS'] / df_eng['SI_M_TLSO'],
        0
    )

    # 4. STAND COMPLEXITY INDEX
    # Combination of structural features

    df_eng['STRUCTURE_INDEX'] = (
        (df_eng['BA_RATIO'] + 1) *  # Dead wood presence
        (df_eng['VOLUME_PER_STEM'] / 100) *  # Large trees
        (df_eng['AGE_HEIGHT_RATIO'] / 10)  # Old growth indicator
    )

    # 5. BINNED FEATURES (for non-linear relationships)
    # Age classes
    df_eng['AGE_CLASS'] = pd.cut(
        df_eng['AGEB_TLSO'],
        bins=[0, 40, 80, 120, 200, 1000],
        labels=['young', 'mature', 'old', 'very_old', 'ancient']
    ).astype(str)

    # Height classes
    df_eng['HEIGHT_CLASS'] = pd.cut(
        df_eng['HT_TLSO'],
        bins=[0, 15, 25, 35, 100],
        labels=['short', 'medium', 'tall', 'very_tall']
    ).astype(str)

    # Site index classes
    df_eng['SI_CLASS'] = pd.cut(
        df_eng['SI_M_TLSO'],
        bins=[0, 20, 30, 40, 100],
        labels=['poor', 'medium', 'good', 'excellent']
    ).astype(str)

    # Log transforms for skewed distributions
    df_eng['LOG_BA'] = np.log1p(df_eng['BA_HA_LS'])
    df_eng['LOG_STEMS'] = np.log1p(df_eng['STEMS_HA_LS'])
    df_eng['LOG_VOLUME'] = np.log1p(df_eng['VHA_WSV_LS'])

    print(
        f"  Created {len([c for c in df_eng.columns if c not in df.columns])} new features")

    return df_eng


def prepare_features(df, use_engineered=True, zone_filter=None):
    """Prepare features with optional engineered features."""

    if zone_filter:
        df = df[df['BEC_ZONE'] == zone_filter].copy()
        print(f"\nFiltering to {zone_filter} zone: {len(df)} plots")

    if use_engineered:
        df = engineer_features(df)

        # Original 6 numerical features
        base_numerical = [
            'BA_HA_LS', 'STEMS_HA_LS', 'VHA_WSV_LS',
            'SI_M_TLSO', 'HT_TLSO', 'AGEB_TLSO'
        ]

        # Engineered numerical features
        engineered_numerical = [
            'BA_RATIO', 'STEMS_RATIO', 'VOLUME_RATIO',
            'TOTAL_BA', 'TOTAL_STEMS', 'TOTAL_VOLUME',
            'VOLUME_PER_STEM', 'BA_PER_STEM',
            'AGE_HEIGHT_RATIO', 'HEIGHT_SI_RATIO',
            'BA_PER_SI', 'STEMS_PER_SI',
            'STRUCTURE_INDEX',
            'LOG_BA', 'LOG_STEMS', 'LOG_VOLUME'
        ]

        numerical_cols = base_numerical + engineered_numerical

        # Base categorical
        categorical_cols = ['BEC_ZONE']

        # Engineered categorical
        engineered_categorical = ['AGE_CLASS', 'HEIGHT_CLASS', 'SI_CLASS']
        categorical_cols.extend(engineered_categorical)

        feature_type = "WITH ENGINEERED FEATURES"
    else:
        # Original features only
        numerical_cols = [
            'BA_HA_LS', 'STEMS_HA_LS', 'VHA_WSV_LS',
            'SI_M_TLSO', 'HT_TLSO', 'AGEB_TLSO'
        ]
        categorical_cols = ['BEC_ZONE']
        feature_type = "BASELINE (NO ENGINEERING)"

    # Filter zone-specific
    if zone_filter is not None:
        categorical_cols = [c for c in categorical_cols if c != 'BEC_ZONE']

    # Filter to valid columns
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    categorical_cols = [col for col in categorical_cols if col in df.columns]

    print(f"\nPreparing features - {feature_type}")
    print(f"  Numerical features: {len(numerical_cols)}")
    if len(numerical_cols) <= 10:
        print(f"    {numerical_cols}")
    else:
        print(f"    Base: {numerical_cols[:6]}")
        print(f"    Engineered: {numerical_cols[6:]}")
    print(f"  Categorical features: {len(categorical_cols)}")
    print(f"    {categorical_cols}")

    # Handle missing values
    X_num = df[numerical_cols].copy()
    for col in numerical_cols:
        median_val = X_num[col].median()
        if pd.isna(median_val):
            median_val = 0
        X_num[col] = X_num[col].fillna(median_val)

    # Replace inf with large values
    X_num = X_num.replace([np.inf, -np.inf], 0)

    # Encode categoricals
    label_encoders = {}
    if categorical_cols:
        for col in categorical_cols:
            le = LabelEncoder()
            X_cat = pd.DataFrame()
            X_cat[col] = le.fit_transform(
                df[col].fillna('UNKNOWN').astype(str))
            label_encoders[col] = le
            X_num = pd.concat([X_num, X_cat], axis=1)
            print(f"    {col}: {len(le.classes_)} categories")

    # Target
    y = df['has_yew'].values

    # Coordinates
    coordinates = df[['x', 'y']].values

    feature_names = list(X_num.columns)

    print(f"  Total features: {len(feature_names)}")
    print(f"  Total samples: {len(X_num)}")
    print(f"  Samples with yew: {y.sum()} ({100*y.mean():.2f}%)")

    return X_num, y, coordinates, label_encoders, feature_names


def spatial_split(coordinates, test_size=0.15, val_size=0.15, random_state=42):
    """Spatial train/val/test split using KMeans."""
    from sklearn.cluster import KMeans

    n_samples = len(coordinates)
    n_test_clusters = max(1, int(n_samples * test_size / 50))
    n_val_clusters = max(1, int(n_samples * val_size / 50))
    n_total_clusters = max(3, n_test_clusters + n_val_clusters +
                           (n_samples - n_test_clusters - n_val_clusters) // 50)

    kmeans = KMeans(n_clusters=n_total_clusters,
                    random_state=random_state, n_init=10)
    cluster_ids = kmeans.fit_predict(coordinates)

    unique_clusters = np.unique(cluster_ids)
    np.random.seed(random_state)
    np.random.shuffle(unique_clusters)

    test_clusters = unique_clusters[:n_test_clusters]
    val_clusters = unique_clusters[n_test_clusters:n_test_clusters+n_val_clusters]
    train_clusters = unique_clusters[n_test_clusters+n_val_clusters:]

    train_idx = np.where(np.isin(cluster_ids, train_clusters))[0]
    val_idx = np.where(np.isin(cluster_ids, val_clusters))[0]
    test_idx = np.where(np.isin(cluster_ids, test_clusters))[0]

    return train_idx, val_idx, test_idx


def train_and_evaluate(X, y, coords, model_name="Model", use_engineered=True):
    """Train and evaluate XGBoost model."""

    train_idx, val_idx, test_idx = spatial_split(coords)

    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]

    print(f"\nSplit sizes:")
    print(
        f"  Train: {len(train_idx)} ({y_train.sum()} yew, {100*y_train.mean():.2f}%)")
    print(
        f"  Val:   {len(val_idx)} ({y_val.sum()} yew, {100*y_val.mean():.2f}%)")
    print(
        f"  Test:  {len(test_idx)} ({y_test.sum()} yew, {100*y_test.mean():.2f}%)")

    # Class weighting
    scale_pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    print(f"\n  Class imbalance: {scale_pos_weight:.1f}:1")

    # Train model
    print(f"\n  Training {model_name}...")
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective='binary:logistic',
        eval_metric=['auc', 'aucpr'],
        random_state=42,
        early_stopping_rounds=50
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=0
    )

    print(f"  Best iteration: {model.best_iteration}")

    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    # Evaluate on test set
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Test at different thresholds
    results = {}
    for threshold in [0.3, 0.4, 0.5, 0.6]:
        y_pred = (y_pred_proba >= threshold).astype(int)
        results[threshold] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }

    results['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
    results['avg_precision'] = average_precision_score(y_test, y_pred_proba)

    return model, results, feature_importance


def print_comparison(baseline_results, engineered_results):
    """Print comparison table."""
    print("\n" + "="*80)
    print("BASELINE vs ENGINEERED FEATURES COMPARISON")
    print("="*80)

    print(f"\n{'Metric':<20} {'Baseline':<15} {'Engineered':<15} {'Change':<15}")
    print("-"*80)

    # ROC AUC
    baseline_auc = baseline_results['roc_auc']
    eng_auc = engineered_results['roc_auc']
    change = ((eng_auc - baseline_auc) / baseline_auc *
              100) if baseline_auc > 0 else 0
    print(f"{'ROC AUC':<20} {baseline_auc:<15.4f} {eng_auc:<15.4f} {change:+.1f}%")

    # Avg Precision
    baseline_ap = baseline_results['avg_precision']
    eng_ap = engineered_results['avg_precision']
    change = ((eng_ap - baseline_ap) / baseline_ap *
              100) if baseline_ap > 0 else 0
    print(f"{'Avg Precision':<20} {baseline_ap:<15.4f} {eng_ap:<15.4f} {change:+.1f}%")

    print("\nAt threshold = 0.5:")
    for metric in ['precision', 'recall', 'f1']:
        baseline_val = baseline_results[0.5][metric]
        eng_val = engineered_results[0.5][metric]
        if baseline_val > 0:
            change = ((eng_val - baseline_val) / baseline_val * 100)
            print(
                f"{metric.capitalize():<20} {baseline_val:<15.4f} {eng_val:<15.4f} {change:+.1f}%")
        else:
            print(
                f"{metric.capitalize():<20} {baseline_val:<15.4f} {eng_val:<15.4f} {'N/A':<15}")

    print("\nAt threshold = 0.4:")
    for metric in ['precision', 'recall', 'f1']:
        baseline_val = baseline_results[0.4][metric]
        eng_val = engineered_results[0.4][metric]
        if baseline_val > 0:
            change = ((eng_val - baseline_val) / baseline_val * 100)
            print(
                f"{metric.capitalize():<20} {baseline_val:<15.4f} {eng_val:<15.4f} {change:+.1f}%")
        else:
            print(
                f"{metric.capitalize():<20} {baseline_val:<15.4f} {eng_val:<15.4f} {'N/A':<15}")


def main():
    """Main pipeline."""

    print("="*80)
    print("XGBOOST WITH ENGINEERED FEATURES")
    print("="*80)

    # Create output directories
    Path('models/checkpoints').mkdir(parents=True, exist_ok=True)
    Path('results/tables').mkdir(parents=True, exist_ok=True)
    Path('results/figures').mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    merged_df = load_and_merge_data(
        'data/ee_imagery/ee_extraction_progress.csv',
        'data/raw/bc_sample_data-2025-10-09/bc_sample_data.csv'
    )
    df = create_target_variable(merged_df)

    # =========================================================================
    # 1. BASELINE MODEL (original 6 features + BEC_ZONE)
    # =========================================================================
    print("\n" + "="*80)
    print("1. BASELINE MODEL (Original Features Only)")
    print("="*80)

    X_base, y_base, coords_base, _, feat_names_base = prepare_features(
        df, use_engineered=False, zone_filter=None
    )

    baseline_model, baseline_results, baseline_importance = train_and_evaluate(
        X_base, y_base, coords_base,
        model_name="Baseline",
        use_engineered=False
    )

    print("\nTop 10 features (baseline):")
    print(baseline_importance.head(10).to_string(index=False))

    # =========================================================================
    # 2. ENGINEERED FEATURES MODEL
    # =========================================================================
    print("\n" + "="*80)
    print("2. ENGINEERED FEATURES MODEL")
    print("="*80)

    X_eng, y_eng, coords_eng, _, feat_names_eng = prepare_features(
        df, use_engineered=True, zone_filter=None
    )

    engineered_model, engineered_results, engineered_importance = train_and_evaluate(
        X_eng, y_eng, coords_eng,
        model_name="Engineered Features",
        use_engineered=True
    )

    print("\nTop 20 features (engineered):")
    print(engineered_importance.head(20).to_string(index=False))

    # =========================================================================
    # 3. COMPARISON
    # =========================================================================
    print_comparison(baseline_results, engineered_results)

    # Save results
    comparison_df = pd.DataFrame({
        'Model': ['Baseline', 'Engineered'],
        'ROC_AUC': [baseline_results['roc_auc'], engineered_results['roc_auc']],
        'Avg_Precision': [baseline_results['avg_precision'], engineered_results['avg_precision']],
        'Precision_0.5': [baseline_results[0.5]['precision'], engineered_results[0.5]['precision']],
        'Recall_0.5': [baseline_results[0.5]['recall'], engineered_results[0.5]['recall']],
        'F1_0.5': [baseline_results[0.5]['f1'], engineered_results[0.5]['f1']],
        'Precision_0.4': [baseline_results[0.4]['precision'], engineered_results[0.4]['precision']],
        'Recall_0.4': [baseline_results[0.4]['recall'], engineered_results[0.4]['recall']],
        'F1_0.4': [baseline_results[0.4]['f1'], engineered_results[0.4]['f1']]
    })

    comparison_df.to_csv(
        'results/tables/feature_engineering_comparison.csv', index=False)

    # Save feature importance
    engineered_importance.to_csv(
        'results/tables/engineered_feature_importance.csv', index=False)

    # Save model
    engineered_model.save_model('models/checkpoints/xgboost_engineered.json')

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nSaved:")
    print("  - results/tables/feature_engineering_comparison.csv")
    print("  - results/tables/engineered_feature_importance.csv")
    print("  - models/checkpoints/xgboost_engineered.json")


if __name__ == '__main__':
    main()
