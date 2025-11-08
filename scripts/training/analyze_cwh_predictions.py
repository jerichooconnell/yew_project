#!/usr/bin/env python3
"""
Analyze CWH model predictions to understand threshold behavior.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
import re
warnings.filterwarnings('ignore')


def parse_species_composition(composition_string):
    """Parse species composition string to extract Pacific Yew (TW) percentage."""
    if not composition_string or pd.isna(composition_string):
        return 0

    pattern = r'TW(\d{2,3})'
    match = re.search(pattern, str(composition_string))
    if match:
        return int(match.group(1))
    return 0


def load_and_merge_data(ee_data_path, inventory_path):
    """Load and merge Earth Engine data with forest inventory."""
    ee_df = pd.read_csv(ee_data_path)
    inv_df = pd.read_csv(inventory_path, low_memory=False)

    ee_df['SITE_IDENTIFIER'] = ee_df['plot_id'].astype(str)
    inv_df['SITE_IDENTIFIER'] = inv_df['SITE_IDENTIFIER'].astype(str)

    merged_df = inv_df.merge(ee_df, on='SITE_IDENTIFIER',
                             how='inner', suffixes=('', '_ee'))
    merged_df['YEW_PERCENTAGE'] = merged_df['SPB_CPCT_LS'].apply(
        parse_species_composition)

    if 'x' in merged_df.columns and 'y' in merged_df.columns:
        merged_df['BC_ALBERS_X'] = merged_df['x']
        merged_df['BC_ALBERS_Y'] = merged_df['y']

    return merged_df


def engineer_features(df):
    """Create engineered features."""
    df = df.copy()

    df['BA_RATIO'] = df['BA_HA_DS'].fillna(0) / (df['BA_HA_LS'].fillna(0) + 1)
    df['STEMS_RATIO'] = df['STEMS_HA_DS'].fillna(
        0) / (df['STEMS_HA_LS'].fillna(0) + 1)
    df['VOLUME_RATIO'] = df['VHA_WSV_DS'].fillna(
        0) / (df['VHA_WSV_LS'].fillna(0) + 1)
    df['VOLUME_PER_STEM'] = df['VHA_WSV_LS'].fillna(
        0) / (df['STEMS_HA_LS'].fillna(0) + 1)
    df['BA_PER_STEM'] = df['BA_HA_LS'].fillna(
        0) / (df['STEMS_HA_LS'].fillna(0) + 1)
    df['AGE_HEIGHT_RATIO'] = df['AGEB_TLSO'].fillna(
        0) / (df['HT_TLSO'].fillna(0) + 1)
    df['HEIGHT_SI_RATIO'] = df['HT_TLSO'].fillna(
        0) / (df['SI_M_TLSO'].fillna(0) + 1)
    df['BA_PER_SI'] = df['BA_HA_LS'].fillna(
        0) / (df['SI_M_TLSO'].fillna(0) + 1)
    df['STEMS_PER_SI'] = df['STEMS_HA_LS'].fillna(
        0) / (df['SI_M_TLSO'].fillna(0) + 1)
    df['STRUCTURE_INDEX'] = (
        (df['BA_RATIO'] + 1) *
        (df['VOLUME_PER_STEM'] / 100) *
        (df['AGE_HEIGHT_RATIO'] / 10)
    )
    df['AGE_CLASS'] = pd.cut(
        df['AGEB_TLSO'].fillna(0),
        bins=[0, 40, 80, 120, 250, 999],
        labels=['young', 'mature', 'old', 'very_old', 'ancient']
    )
    df['HEIGHT_CLASS'] = pd.cut(
        df['HT_TLSO'].fillna(0),
        bins=[0, 10, 20, 30, 100],
        labels=['short', 'medium', 'tall', 'very_tall']
    )
    df['SI_CLASS'] = pd.cut(
        df['SI_M_TLSO'].fillna(0),
        bins=[0, 15, 25, 35, 100],
        labels=['poor', 'medium', 'good', 'excellent']
    )
    df['LOG_BA'] = np.log1p(df['BA_HA_LS'].fillna(0))
    df['LOG_STEMS'] = np.log1p(df['STEMS_HA_LS'].fillna(0))
    df['LOG_VOLUME'] = np.log1p(df['VHA_WSV_LS'].fillna(0))
    df['TOTAL_BA'] = df['BA_HA_LS'].fillna(0) + df['BA_HA_DS'].fillna(0)
    df['TOTAL_STEMS'] = df['STEMS_HA_LS'].fillna(
        0) + df['STEMS_HA_DS'].fillna(0)
    df['TOTAL_VOLUME'] = df['VHA_WSV_LS'].fillna(
        0) + df['VHA_WSV_DS'].fillna(0)

    return df


def prepare_features(df):
    """Prepare engineered features for training."""
    engineered_numerical = [
        'BA_HA_LS', 'STEMS_HA_LS', 'VHA_WSV_LS', 'SI_M_TLSO', 'HT_TLSO', 'AGEB_TLSO',
        'BA_RATIO', 'STEMS_RATIO', 'VOLUME_RATIO',
        'VOLUME_PER_STEM', 'BA_PER_STEM', 'AGE_HEIGHT_RATIO', 'HEIGHT_SI_RATIO',
        'BA_PER_SI', 'STEMS_PER_SI', 'STRUCTURE_INDEX',
        'LOG_BA', 'LOG_STEMS', 'LOG_VOLUME',
        'TOTAL_BA', 'TOTAL_STEMS', 'TOTAL_VOLUME'
    ]
    engineered_categorical = ['AGE_CLASS', 'HEIGHT_CLASS', 'SI_CLASS']

    for col in engineered_numerical:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    X_numerical = df[engineered_numerical].values
    X_categorical = pd.get_dummies(
        df[engineered_categorical], prefix=engineered_categorical, drop_first=False)
    X = np.concatenate([X_numerical, X_categorical.values], axis=1)
    feature_names = engineered_numerical + list(X_categorical.columns)

    return X, feature_names


def spatial_split(coordinates, test_size=0.2, val_size=0.1, random_state=42):
    """Spatial train/val/test split."""
    n_samples = len(coordinates)
    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters,
                    random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(coordinates)

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


def main():
    """Analyze CWH model predictions at different thresholds."""
    print("="*80)
    print("Analyzing CWH Model Predictions")
    print("="*80)

    # Load data
    df = load_and_merge_data(
        'data/ee_imagery/ee_extraction_progress.csv',
        'data/raw/bc_sample_data-2025-10-09/bc_sample_data.csv'
    )

    # Filter to CWH
    df_cwh = df[df['BEC_ZONE'] == 'CWH'].copy()
    df_cwh['has_yew'] = (df_cwh['YEW_PERCENTAGE'] > 0).astype(int)
    print(f"\nCWH records: {len(df_cwh)}, Yew: {df_cwh['has_yew'].sum()}")

    # Engineer features
    df_cwh = engineer_features(df_cwh)
    X, feature_names = prepare_features(df_cwh)
    y = df_cwh['has_yew'].values
    coordinates = df_cwh[['BC_ALBERS_X', 'BC_ALBERS_Y']].values

    # Spatial split
    train_idx, val_idx, test_idx = spatial_split(coordinates)
    X_test, y_test = X[test_idx], y[test_idx]

    # Load model
    model = xgb.Booster()
    model.load_model('models/checkpoints/xgboost_cwh_engineered.json')

    # Predict
    dtest = xgb.DMatrix(X_test, feature_names=feature_names)
    y_pred_proba = model.predict(dtest)

    print(f"\nPrediction statistics:")
    print(f"  Min probability: {y_pred_proba.min():.6f}")
    print(f"  Max probability: {y_pred_proba.max():.6f}")
    print(f"  Mean probability: {y_pred_proba.mean():.6f}")
    print(f"  Median probability: {np.median(y_pred_proba):.6f}")
    print(f"  Std probability: {y_pred_proba.std():.6f}")

    print(f"\nPredictions above different thresholds:")
    for threshold in [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
        n_above = (y_pred_proba >= threshold).sum()
        pct = 100 * n_above / len(y_pred_proba)
        print(f"  {threshold:.3f}: {n_above:>5} predictions ({pct:>5.2f}%)")

    print(f"\nPerformance at fine-grained thresholds:")
    print(
        f"  {'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'# Preds':<10}")
    print(f"  {'-'*58}")

    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        n_preds = y_pred.sum()

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(
            f"  {threshold:<12.3f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {n_preds:<10}")

    # Find optimal threshold
    best_f1 = 0
    best_threshold = 0
    for threshold in np.arange(0.001, 0.5, 0.001):
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"\nOptimal threshold (maximizes F1): {best_threshold:.4f}")
    y_pred_optimal = (y_pred_proba >= best_threshold).astype(int)
    print(
        f"  Precision: {precision_score(y_test, y_pred_optimal, zero_division=0):.4f}")
    print(
        f"  Recall: {recall_score(y_test, y_pred_optimal, zero_division=0):.4f}")
    print(f"  F1: {f1_score(y_test, y_pred_optimal, zero_division=0):.4f}")
    print(f"  Predictions: {y_pred_optimal.sum()}")


if __name__ == "__main__":
    main()
