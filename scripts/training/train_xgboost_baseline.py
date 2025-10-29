#!/usr/bin/env python3
"""
XGBoost Baseline Model - No Satellite Data
===========================================

Trains an XGBoost model using ONLY forest inventory features (no satellite data).
This serves as a baseline to evaluate whether satellite imagery improves predictions.

Features used:
- Forest inventory: BA_HA_LS, STEMS_HA_LS, VHA_WSV_LS, SI_M_TLSO, HT_TLSO, AGEB_TLSO
- Categorical: BEC_ZONE, TSA_DESC, SAMPLE_ESTABLISHMENT_TYPE

Features NOT used:
- Satellite: blue, green, red, nir, ndvi, evi, elevation, slope, aspect

Author: Analysis Tool
Date: October 28, 2025
"""

import sys
sys.path.append('scripts/training')

from train_yew_model_with_ee import (
    load_and_merge_data, create_target_variable, parse_species_composition
)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from collections import defaultdict


def spatial_train_test_split_simple(coordinates, test_size=0.2, val_size=0.1, random_state=42):
    """Simple spatial split using KMeans clustering."""
    from sklearn.cluster import KMeans
    
    n_samples = len(coordinates)
    n_test_clusters = int(n_samples * test_size / 50)
    n_val_clusters = int(n_samples * val_size / 50)
    n_clusters = n_test_clusters + n_val_clusters + (n_samples - n_test_clusters - n_val_clusters) // 50
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
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
    
    print(f"\nSpatial split created:")
    print(f"  Train: {len(train_idx)} samples ({len(train_clusters)} spatial blocks)")
    print(f"  Val: {len(val_idx)} samples ({len(val_clusters)} spatial blocks)")
    print(f"  Test: {len(test_idx)} samples ({len(test_clusters)} spatial blocks)")
    
    return train_idx, val_idx, test_idx


def prepare_inventory_only_features(df):
    """Prepare features using ALL available forest inventory data (no satellite)."""
    print("\nPreparing inventory-only features (ALL AVAILABLE)...")
    
    # ALL forest inventory numerical features (excluding IDs, coordinates, and metadata)
    # Live stand metrics
    inventory_numerical = [
        'BA_HA_LS',      # Basal area per hectare - live stand
        'STEMS_HA_LS',   # Stems per hectare - live stand
        'VHA_WSV_LS',    # Volume per hectare, whole stem volume - live stand
        'VHA_NTWB_LS',   # Volume per hectare, net timber with bark - live stand
        'SI_M_TLSO',     # Site index in meters - top live standing only
        'HT_TLSO',       # Height - top live standing only
        'AGEB_TLSO',     # Age breast height - top live standing only
        'AGET_TLSO',     # Age total - top live standing only
        # Dead stand metrics
        'BA_HA_DS',      # Basal area per hectare - dead stand
        'STEMS_HA_DS',   # Stems per hectare - dead stand
        'VHA_WSV_DS',    # Volume per hectare, whole stem volume - dead stand
        'VHA_NTWB_DS',   # Volume per hectare, net timber with bark - dead stand
        # Plot/measurement info
        'MEAS_YR',       # Measurement year
        'VISIT_NUMBER',  # Visit number
        'NO_PLOTS',      # Number of plots
        'UTIL'           # Utilization level
        # Removed spatial coordinates: IP_EAST, IP_NRTH, IP_UTM (cause memorization/overfitting)
    ]
    
    # Categorical features (keeping only low-cardinality ones to reduce overfitting)
    categorical_cols = [
        'BEC_ZONE',                  # Biogeoclimatic zone (4 categories)
        'TSA_DESC',                  # Timber supply area (31 categories)
        'SAMPLE_ESTABLISHMENT_TYPE', # Sample type (6 categories)
        'YSM_MAIN_FM',              # Years since main fire/management (2 categories)
        'MAT_MAIN_FM'               # Management activity type (2 categories)
        # Removed high-cardinality categoricals that cause overfitting:
        # 'BECLABEL' (65 categories), 'SPC_LIVE_1' (22 categories),
        # 'TFL' (26 categories), 'OWN_SCHED_DESCRIP' (24 categories),
        # 'GRID_SIZE', 'GRID_BASE', 'UTM_SOURCE', 'PSP_STATUS', 'SAMP_TYP' (metadata)
    ]
    
    # Filter to valid columns that exist in dataframe
    numerical_cols = [col for col in inventory_numerical if col in df.columns]
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    
    # Verify numerical columns are actually numeric
    numerical_cols_clean = []
    for col in numerical_cols:
        try:
            df[col].median()  # Test if it's numeric
            numerical_cols_clean.append(col)
        except:
            print(f"    Warning: {col} appears non-numeric, treating as categorical")
            categorical_cols.append(col)
    numerical_cols = numerical_cols_clean
    
    print(f"  Numerical features: {len(numerical_cols)}")
    print(f"    {numerical_cols}")
    print(f"  Categorical features: {len(categorical_cols)}")
    print(f"    {categorical_cols}")
    
    # Handle missing values
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())
    
    for col in categorical_cols:
        df[col] = df[col].fillna('UNKNOWN')
    
    # Create feature matrix
    X = df[numerical_cols].copy()
    
    # Add encoded categorical features
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"    {col}: {len(le.classes_)} categories")
    
    # Get targets
    y = df['has_yew'].values
    
    # Get coordinates for spatial splitting
    coordinates = df[['BC_ALBERS_X', 'BC_ALBERS_Y']].values
    
    print(f"\n  Total samples: {len(y)}")
    print(f"  Samples with yew: {y.sum()} ({100*y.mean():.2f}%)")
    print(f"  Total features: {len(numerical_cols) + len(categorical_cols)}")
    
    return X, y, coordinates, label_encoders, numerical_cols + categorical_cols


def optimize_threshold(y_true, y_pred_proba, thresholds=None):
    """Find optimal threshold to maximize F1 score."""
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)
    
    best_f1 = 0
    best_threshold = 0.5
    results = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1, results


def plot_results(train_history, threshold_results, feature_importance, output_dir):
    """Create comprehensive result visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Training history
    ax = axes[0, 0]
    if 'train_auc' in train_history and len(train_history['train_auc']) > 0:
        ax.plot(train_history['train_auc'], label='Train AUC', marker='o')
        ax.plot(train_history['val_auc'], label='Val AUC', marker='s')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('AUC')
        ax.set_title('Training History')
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No training history available', 
                ha='center', va='center', transform=ax.transAxes)
    
    # 2. Threshold analysis
    ax = axes[0, 1]
    thresholds = [r['threshold'] for r in threshold_results]
    precisions = [r['precision'] for r in threshold_results]
    recalls = [r['recall'] for r in threshold_results]
    f1s = [r['f1'] for r in threshold_results]
    
    ax.plot(thresholds, precisions, label='Precision', marker='o')
    ax.plot(thresholds, recalls, label='Recall', marker='s')
    ax.plot(thresholds, f1s, label='F1 Score', marker='^')
    ax.set_xlabel('Decision Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Threshold Optimization')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Feature importance
    ax = axes[1, 0]
    importance_df = pd.DataFrame({
        'feature': feature_importance['features'],
        'importance': feature_importance['importance']
    }).sort_values('importance', ascending=True)
    
    y_pos = np.arange(len(importance_df))
    ax.barh(y_pos, importance_df['importance'], color='steelblue', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(importance_df['feature'])
    ax.set_xlabel('Importance (Gain)')
    ax.set_title('XGBoost Feature Importance')
    ax.grid(axis='x', alpha=0.3)
    
    # 4. Performance comparison table
    ax = axes[1, 1]
    ax.axis('off')
    
    table_data = [
        ['Metric', 'Threshold 0.5', 'Optimized'],
        ['Threshold', '0.50', f"{threshold_results[-1]['threshold']:.2f}"],
        ['Precision', f"{threshold_results[8]['precision']:.4f}", 
         f"{max(threshold_results, key=lambda x: x['f1'])['precision']:.4f}"],
        ['Recall', f"{threshold_results[8]['recall']:.4f}", 
         f"{max(threshold_results, key=lambda x: x['f1'])['recall']:.4f}"],
        ['F1 Score', f"{threshold_results[8]['f1']:.4f}", 
         f"{max(threshold_results, key=lambda x: x['f1'])['f1']:.4f}"]
    ]
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Performance Comparison', pad=20, fontsize=12, weight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'xgboost_baseline_results.png', dpi=300, bbox_inches='tight')
    print(f"\nResults plot saved to {output_dir / 'xgboost_baseline_results.png'}")


def main():
    """Main training pipeline."""
    print("="*70)
    print("XGBoost Baseline - Inventory Features Only (No Satellite)")
    print("="*70)
    
    # Set seeds
    np.random.seed(42)
    
    # Load data
    merged_df = load_and_merge_data(
        'data/ee_imagery/ee_extraction_progress.csv',
        'data/raw/bc_sample_data-2025-10-09/bc_sample_data.csv'
    )
    
    merged_df = create_target_variable(merged_df)
    
    # Prepare features (inventory only)
    X, y, coordinates, label_encoders, feature_names = prepare_inventory_only_features(merged_df)
    
    # Spatial split
    print("\nCreating spatial train/val/test splits...")
    train_idx, val_idx, test_idx = spatial_train_test_split_simple(
        coordinates, test_size=0.2, val_size=0.1, random_state=42
    )
    
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]
    
    print(f"\nClass distribution:")
    print(f"  Train: {y_train.sum()} yew / {len(y_train)-y_train.sum()} no-yew ({100*y_train.mean():.2f}%)")
    print(f"  Val: {y_val.sum()} yew / {len(y_val)-y_val.sum()} no-yew ({100*y_val.mean():.2f}%)")
    print(f"  Test: {y_test.sum()} yew / {len(y_test)-y_test.sum()} no-yew ({100*y_test.mean():.2f}%)")
    
    # Calculate scale_pos_weight for class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"\nClass imbalance ratio: {scale_pos_weight:.1f}:1")
    print(f"Using scale_pos_weight={scale_pos_weight:.1f} in XGBoost")
    
    # Train XGBoost
    print("\nTraining XGBoost model...")
    
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
    
    # Train with validation monitoring
    train_history = defaultdict(list)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=10
    )
    
    # Get training history
    results = model.evals_result()
    if results:
        train_history['train_auc'] = results['validation_0'].get('auc', [])
        train_history['val_auc'] = results['validation_1'].get('auc', [])
    
    print(f"\nBest iteration: {model.best_iteration}")
    print(f"Best score: {model.best_score:.4f}")
    
    # Predictions
    print("\nGenerating predictions...")
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Optimize threshold on validation set
    print("\nOptimizing decision threshold on validation set...")
    best_threshold, best_f1, threshold_results = optimize_threshold(y_val, y_val_pred_proba)
    print(f"Best threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")
    
    # Evaluate on test set with both thresholds
    print("\n" + "="*70)
    print("Test Set Evaluation")
    print("="*70)
    
    # Default threshold (0.5)
    y_test_pred_default = (y_test_pred_proba > 0.5).astype(int)
    acc_default = accuracy_score(y_test, y_test_pred_default)
    prec_default = precision_score(y_test, y_test_pred_default, zero_division=0)
    rec_default = recall_score(y_test, y_test_pred_default, zero_division=0)
    f1_default = f1_score(y_test, y_test_pred_default, zero_division=0)
    
    # Optimized threshold
    y_test_pred_opt = (y_test_pred_proba > best_threshold).astype(int)
    acc_opt = accuracy_score(y_test, y_test_pred_opt)
    prec_opt = precision_score(y_test, y_test_pred_opt, zero_division=0)
    rec_opt = recall_score(y_test, y_test_pred_opt, zero_division=0)
    f1_opt = f1_score(y_test, y_test_pred_opt, zero_division=0)
    
    # AUC scores
    try:
        auc_score = roc_auc_score(y_test, y_test_pred_proba)
        ap_score = average_precision_score(y_test, y_test_pred_proba)
    except:
        auc_score = 0.0
        ap_score = 0.0
    
    print(f"\nTest Set Results (threshold=0.5):")
    print(f"  Accuracy: {acc_default:.4f}")
    print(f"  Precision: {prec_default:.4f}")
    print(f"  Recall: {rec_default:.4f}")
    print(f"  F1 Score: {f1_default:.4f}")
    
    print(f"\nTest Set Results (optimized threshold={best_threshold:.2f}):")
    print(f"  Accuracy: {acc_opt:.4f}")
    print(f"  Precision: {prec_opt:.4f}")
    print(f"  Recall: {rec_opt:.4f}")
    print(f"  F1 Score: {f1_opt:.4f}")
    
    print(f"\nProbability-based Metrics:")
    print(f"  ROC AUC: {auc_score:.4f}")
    print(f"  Average Precision: {ap_score:.4f}")
    
    print(f"\nTest set composition:")
    print(f"  Yew present: {y_test.sum()} ({100*y_test.mean():.2f}%)")
    print(f"  Yew absent: {(y_test == 0).sum()}")
    
    # Confusion matrix
    print("\nConfusion Matrix (optimized threshold):")
    cm = confusion_matrix(y_test, y_test_pred_opt)
    print(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")
    
    # Feature importance
    print("\n" + "="*70)
    print("Feature Importance")
    print("="*70)
    
    importance = model.feature_importances_
    feature_importance = {
        'features': feature_names,
        'importance': importance
    }
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))
    
    # Save results
    print("\n" + "="*70)
    print("Saving Results")
    print("="*70)
    
    # Save model
    Path('models/checkpoints').mkdir(parents=True, exist_ok=True)
    model.save_model('models/checkpoints/xgboost_baseline.json')
    print("  Model saved to models/checkpoints/xgboost_baseline.json")
    
    # Save preprocessors
    Path('models/artifacts').mkdir(parents=True, exist_ok=True)
    with open('models/artifacts/xgboost_baseline_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    print("  Encoders saved to models/artifacts/xgboost_baseline_encoders.pkl")
    
    # Save metrics
    metrics = {
        'test_accuracy_default': acc_default,
        'test_precision_default': prec_default,
        'test_recall_default': rec_default,
        'test_f1_default': f1_default,
        'test_accuracy_optimized': acc_opt,
        'test_precision_optimized': prec_opt,
        'test_recall_optimized': rec_opt,
        'test_f1_optimized': f1_opt,
        'test_auc': auc_score,
        'test_average_precision': ap_score,
        'best_threshold': best_threshold,
        'scale_pos_weight': scale_pos_weight
    }
    
    with open('models/artifacts/xgboost_baseline_metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    print("  Metrics saved to models/artifacts/xgboost_baseline_metrics.pkl")
    
    # Save feature importance
    Path('results/tables').mkdir(parents=True, exist_ok=True)
    importance_df.to_csv('results/tables/xgboost_feature_importance.csv', index=False)
    print("  Feature importance saved to results/tables/xgboost_feature_importance.csv")
    
    # Create visualizations
    Path('results/figures').mkdir(parents=True, exist_ok=True)
    plot_results(train_history, threshold_results, feature_importance, Path('results/figures'))
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    
    return model, metrics, importance_df


if __name__ == "__main__":
    main()
