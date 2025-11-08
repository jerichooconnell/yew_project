#!/usr/bin/env python3
"""
XGBoost Enhanced Training - Zone-Specific Models on Original Data
==================================================================

Strategy: Train separate models for CWH and ICH zones using the full dataset
- Keep ALL data (no aggressive cleaning that removes 31% of yew plots)
- Train zone-specific models with zone-specific thresholds
- Use better regularization instead of outlier removal

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


def prepare_features(df, zone_filter=None):
    """Prepare features for training."""

    if zone_filter:
        df = df[df['BEC_ZONE'] == zone_filter].copy()
        print(f"\nFiltering to {zone_filter} zone only: {len(df)} plots")

    # 6 numerical features
    numerical_cols = [
        'BA_HA_LS', 'STEMS_HA_LS', 'VHA_WSV_LS',
        'SI_M_TLSO', 'HT_TLSO', 'AGEB_TLSO'
    ]

    # BEC_ZONE categorical (only if not zone-specific)
    if zone_filter is None:
        categorical_cols = ['BEC_ZONE']
    else:
        categorical_cols = []

    # Filter to valid columns
    numerical_cols = [col for col in numerical_cols if col in df.columns]

    print(f"\n  Numerical features: {len(numerical_cols)}")
    print(f"    {numerical_cols}")
    if categorical_cols:
        print(f"  Categorical features: {len(categorical_cols)}")
        print(f"    {categorical_cols}")

    # Handle missing values
    X_num = df[numerical_cols].copy()
    for col in numerical_cols:
        X_num[col] = X_num[col].fillna(X_num[col].median())

    # Encode categoricals
    label_encoders = {}
    if categorical_cols:
        for col in categorical_cols:
            le = LabelEncoder()
            X_cat = pd.DataFrame()
            X_cat[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            X_num = pd.concat([X_num, X_cat], axis=1)

    # Target
    y = df['has_yew'].values

    # Coordinates for spatial split
    coordinates = df[['x', 'y']].values

    feature_names = list(X_num.columns)

    print(f"  Total features: {len(feature_names)}")
    print(f"  Total samples: {len(X_num)}")
    print(f"  Samples with yew: {y.sum()} ({100*y.mean():.2f}%)")

    return X_num, y, coordinates, label_encoders, feature_names


def spatial_split(coordinates, test_size=0.15, val_size=0.15, random_state=42):
    """Create spatial train/val/test split using KMeans."""
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


def optimize_threshold(y_true, y_pred_proba):
    """Find optimal decision threshold based on F1 score."""
    thresholds = np.arange(0.1, 0.96, 0.05)
    best_f1 = 0
    best_threshold = 0.5
    results = []

    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        results.append({
            'threshold': thresh,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    return best_threshold, results


def train_model(X_train, y_train, X_val, y_val, model_name="XGBoost"):
    """Train XGBoost model with early stopping."""

    scale_pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    print(f"\n  Class imbalance: {scale_pos_weight:.1f}:1")
    print(f"  Using scale_pos_weight={scale_pos_weight:.1f}")

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

    print(f"\n  Training {model_name}...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=0
    )

    print(f"  Best iteration: {model.best_iteration}")

    return model


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Evaluate model performance."""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'avg_precision': average_precision_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

    return metrics, y_pred_proba


def print_metrics(metrics, label="Model"):
    """Print evaluation metrics."""
    print(f"\n{label} Performance:")
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  Precision:    {metrics['precision']:.4f}")
    print(f"  Recall:       {metrics['recall']:.4f}")
    print(f"  F1 Score:     {metrics['f1']:.4f}")
    print(f"  ROC AUC:      {metrics['roc_auc']:.4f}")
    print(f"  Avg Precision:{metrics['avg_precision']:.4f}")

    cm = metrics['confusion_matrix']
    print(f"\n  Confusion Matrix:")
    print(f"    TN: {cm[0,0]:4d}  FP: {cm[0,1]:4d}")
    print(f"    FN: {cm[1,0]:4d}  TP: {cm[1,1]:4d}")


def main():
    """Main training pipeline."""

    print("="*70)
    print("XGBOOST ENHANCED TRAINING")
    print("Cleaned Data + Zone-Specific Models")
    print("="*70)

    # Create output directories
    Path('models/checkpoints').mkdir(parents=True, exist_ok=True)
    Path('models/artifacts').mkdir(parents=True, exist_ok=True)
    Path('results/tables').mkdir(parents=True, exist_ok=True)
    Path('results/figures').mkdir(parents=True, exist_ok=True)

    # Load original merged data
    print("Loading full dataset...")
    merged_df = load_and_merge_data(
        'data/ee_imagery/ee_extraction_progress.csv',
        'data/raw/bc_sample_data-2025-10-09/bc_sample_data.csv'
    )
    df = create_target_variable(merged_df)

    # =========================================================================
    # 1. GLOBAL MODEL (both zones combined with BEC_ZONE as feature)
    # =========================================================================
    print("\n" + "="*70)
    print("1. GLOBAL MODEL (CWH + ICH with BEC_ZONE feature)")
    print("="*70)

    X, y, coords, encoders, feat_names = prepare_features(df, zone_filter=None)
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

    global_model = train_model(X_train, y_train, X_val, y_val, "Global Model")

    # Optimize threshold on validation set
    y_val_proba = global_model.predict_proba(X_val)[:, 1]
    best_thresh_global, _ = optimize_threshold(y_val, y_val_proba)
    print(f"\n  Optimized threshold: {best_thresh_global:.2f}")

    # Evaluate on test set
    metrics_global_default, _ = evaluate_model(
        global_model, X_test, y_test, threshold=0.5)
    metrics_global_opt, _ = evaluate_model(
        global_model, X_test, y_test, threshold=best_thresh_global)

    print_metrics(metrics_global_default, "Global Model (threshold=0.5)")
    print_metrics(metrics_global_opt,
                  f"Global Model (threshold={best_thresh_global:.2f})")

    # Save global model
    global_model.save_model('models/checkpoints/xgboost_global_cleaned.json')

    # =========================================================================
    # 2. CWH-SPECIFIC MODEL
    # =========================================================================
    print("\n" + "="*70)
    print("2. CWH-SPECIFIC MODEL")
    print("="*70)

    X_cwh, y_cwh, coords_cwh, _, feat_names_cwh = prepare_features(
        df, zone_filter='CWH')
    train_idx_cwh, val_idx_cwh, test_idx_cwh = spatial_split(coords_cwh)

    X_train_cwh, y_train_cwh = X_cwh.iloc[train_idx_cwh], y_cwh[train_idx_cwh]
    X_val_cwh, y_val_cwh = X_cwh.iloc[val_idx_cwh], y_cwh[val_idx_cwh]
    X_test_cwh, y_test_cwh = X_cwh.iloc[test_idx_cwh], y_cwh[test_idx_cwh]

    print(f"\nSplit sizes:")
    print(
        f"  Train: {len(train_idx_cwh)} ({y_train_cwh.sum()} yew, {100*y_train_cwh.mean():.2f}%)")
    print(
        f"  Val:   {len(val_idx_cwh)} ({y_val_cwh.sum()} yew, {100*y_val_cwh.mean():.2f}%)")
    print(
        f"  Test:  {len(test_idx_cwh)} ({y_test_cwh.sum()} yew, {100*y_test_cwh.mean():.2f}%)")

    cwh_model = train_model(X_train_cwh, y_train_cwh,
                            X_val_cwh, y_val_cwh, "CWH Model")

    # Optimize threshold
    y_val_proba_cwh = cwh_model.predict_proba(X_val_cwh)[:, 1]
    best_thresh_cwh, _ = optimize_threshold(y_val_cwh, y_val_proba_cwh)
    print(f"\n  Optimized threshold: {best_thresh_cwh:.2f}")

    # Evaluate
    metrics_cwh_default, _ = evaluate_model(
        cwh_model, X_test_cwh, y_test_cwh, threshold=0.5)
    metrics_cwh_opt, _ = evaluate_model(
        cwh_model, X_test_cwh, y_test_cwh, threshold=best_thresh_cwh)

    print_metrics(metrics_cwh_default, "CWH Model (threshold=0.5)")
    print_metrics(metrics_cwh_opt,
                  f"CWH Model (threshold={best_thresh_cwh:.2f})")

    # Save CWH model
    cwh_model.save_model('models/checkpoints/xgboost_cwh_cleaned.json')

    # =========================================================================
    # 3. ICH-SPECIFIC MODEL
    # =========================================================================
    print("\n" + "="*70)
    print("3. ICH-SPECIFIC MODEL")
    print("="*70)

    X_ich, y_ich, coords_ich, _, feat_names_ich = prepare_features(
        df, zone_filter='ICH')
    train_idx_ich, val_idx_ich, test_idx_ich = spatial_split(coords_ich)

    X_train_ich, y_train_ich = X_ich.iloc[train_idx_ich], y_ich[train_idx_ich]
    X_val_ich, y_val_ich = X_ich.iloc[val_idx_ich], y_ich[val_idx_ich]
    X_test_ich, y_test_ich = X_ich.iloc[test_idx_ich], y_ich[test_idx_ich]

    print(f"\nSplit sizes:")
    print(
        f"  Train: {len(train_idx_ich)} ({y_train_ich.sum()} yew, {100*y_train_ich.mean():.2f}%)")
    print(
        f"  Val:   {len(val_idx_ich)} ({y_val_ich.sum()} yew, {100*y_val_ich.mean():.2f}%)")
    print(
        f"  Test:  {len(test_idx_ich)} ({y_test_ich.sum()} yew, {100*y_test_ich.mean():.2f}%)")

    ich_model = train_model(X_train_ich, y_train_ich,
                            X_val_ich, y_val_ich, "ICH Model")

    # Optimize threshold
    y_val_proba_ich = ich_model.predict_proba(X_val_ich)[:, 1]
    best_thresh_ich, _ = optimize_threshold(y_val_ich, y_val_proba_ich)
    print(f"\n  Optimized threshold: {best_thresh_ich:.2f}")

    # Evaluate
    metrics_ich_default, _ = evaluate_model(
        ich_model, X_test_ich, y_test_ich, threshold=0.5)
    metrics_ich_opt, _ = evaluate_model(
        ich_model, X_test_ich, y_test_ich, threshold=best_thresh_ich)

    print_metrics(metrics_ich_default, "ICH Model (threshold=0.5)")
    print_metrics(metrics_ich_opt,
                  f"ICH Model (threshold={best_thresh_ich:.2f})")

    # Save ICH model
    ich_model.save_model('models/checkpoints/xgboost_ich_cleaned.json')

    # =========================================================================
    # 4. SUMMARY COMPARISON
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)

    summary_data = []
    summary_data.append(
        ['Model', 'Threshold', 'Precision', 'Recall', 'F1', 'ROC AUC'])
    summary_data.append(['Global', '0.50',
                        f"{metrics_global_default['precision']:.4f}",
                         f"{metrics_global_default['recall']:.4f}",
                         f"{metrics_global_default['f1']:.4f}",
                         f"{metrics_global_default['roc_auc']:.4f}"])
    summary_data.append(['Global', f'{best_thresh_global:.2f}',
                        f"{metrics_global_opt['precision']:.4f}",
                         f"{metrics_global_opt['recall']:.4f}",
                         f"{metrics_global_opt['f1']:.4f}",
                         f"{metrics_global_opt['roc_auc']:.4f}"])
    summary_data.append(['CWH', '0.50',
                        f"{metrics_cwh_default['precision']:.4f}",
                         f"{metrics_cwh_default['recall']:.4f}",
                         f"{metrics_cwh_default['f1']:.4f}",
                         f"{metrics_cwh_default['roc_auc']:.4f}"])
    summary_data.append(['CWH', f'{best_thresh_cwh:.2f}',
                        f"{metrics_cwh_opt['precision']:.4f}",
                         f"{metrics_cwh_opt['recall']:.4f}",
                         f"{metrics_cwh_opt['f1']:.4f}",
                         f"{metrics_cwh_opt['roc_auc']:.4f}"])
    summary_data.append(['ICH', '0.50',
                        f"{metrics_ich_default['precision']:.4f}",
                         f"{metrics_ich_default['recall']:.4f}",
                         f"{metrics_ich_default['f1']:.4f}",
                         f"{metrics_ich_default['roc_auc']:.4f}"])
    summary_data.append(['ICH', f'{best_thresh_ich:.2f}',
                        f"{metrics_ich_opt['precision']:.4f}",
                         f"{metrics_ich_opt['recall']:.4f}",
                         f"{metrics_ich_opt['f1']:.4f}",
                         f"{metrics_ich_opt['roc_auc']:.4f}"])

    print("\nTest Set Performance:")
    for row in summary_data:
        print(
            f"  {row[0]:8s} {row[1]:6s}  P:{row[2]:7s}  R:{row[3]:7s}  F1:{row[4]:7s}  AUC:{row[5]:7s}")

    # Save summary
    summary_df = pd.DataFrame(summary_data[1:], columns=summary_data[0])
    summary_df.to_csv(
        'results/tables/zone_specific_comparison.csv', index=False)

    # Save thresholds
    thresholds = {
        'global': best_thresh_global,
        'cwh': best_thresh_cwh,
        'ich': best_thresh_ich
    }
    with open('models/artifacts/optimized_thresholds.pkl', 'wb') as f:
        pickle.dump(thresholds, f)

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\nSaved models:")
    print("  - models/checkpoints/xgboost_global_cleaned.json")
    print("  - models/checkpoints/xgboost_cwh_cleaned.json")
    print("  - models/checkpoints/xgboost_ich_cleaned.json")
    print("\nSaved artifacts:")
    print("  - models/artifacts/optimized_thresholds.pkl")
    print("  - results/tables/zone_specific_comparison.csv")


if __name__ == '__main__':
    main()
