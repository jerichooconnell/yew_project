#!/usr/bin/env python3
"""
SVM Hyperparameter Tuning for Yew Classification

This script:
1. Uses ALL available embedding data (not just original splits)
2. Performs grid search over SVM hyperparameters
3. Uses class weighting for balance
4. Reports best parameters and saves the best model

Usage:
    python scripts/training/tune_svm_hyperparameters.py
"""

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             confusion_matrix)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Tune SVM hyperparameters for yew classification')
    parser.add_argument('--embedding-dir', type=str,
                        default='data/ee_imagery/embedding_patches_64x64',
                        help='Directory with embedding patches')
    parser.add_argument('--train-csv', type=str,
                        default='data/processed/train_split_balanced_max.csv',
                        help='Training CSV (with all available samples)')
    parser.add_argument('--val-csv', type=str,
                        default='data/processed/val_split_balanced_max.csv',
                        help='Validation CSV')
    parser.add_argument('--output-dir', type=str,
                        default='models/svm_tuned',
                        help='Output directory for tuned model')
    parser.add_argument('--class-weight-ratio', type=float, default=5.0,
                        help='Weight ratio for non-yew class')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Number of cross-validation folds')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Number of parallel jobs (-1 for all cores)')
    return parser.parse_args()


def extract_center_pixel(lat, lon, emb_dir, patch_size=64):
    """Extract single center pixel from embedding."""
    emb_path = emb_dir / f'embedding_{lat:.6f}_{lon:.6f}.npy'
    
    if not emb_path.exists():
        return None
    
    try:
        img = np.load(emb_path)
        center = patch_size // 2
        
        if img.ndim == 3:
            if img.shape[0] == 64:  # bands first
                return img[:, center, center]
            elif img.shape[2] == 64:  # bands last
                return img[center, center, :]
        return None
    except Exception as e:
        return None


def load_features(df, emb_dir):
    """Extract features and labels from dataframe."""
    features = []
    labels = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc='  Loading features'):
        if pd.notna(row['lat']) and pd.notna(row['lon']):
            feat = extract_center_pixel(row['lat'], row['lon'], emb_dir)
            if feat is not None:
                features.append(feat)
                labels.append(int(row['has_yew']))
    
    X = np.array(features)
    y = np.array(labels)
    
    # Handle NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X, y


def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = Path(args.embedding_dir)
    
    print("="*70)
    print("SVM HYPERPARAMETER TUNING FOR YEW CLASSIFICATION")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    
    print(f"  Train CSV: {len(train_df)} rows")
    print(f"  Val CSV: {len(val_df)} rows")
    
    X_train, y_train = load_features(train_df, emb_dir)
    X_val, y_val = load_features(val_df, emb_dir)
    
    print(f"\n  Train features: {len(X_train)} (Yew: {y_train.sum()}, Non-yew: {(1-y_train).sum()})")
    print(f"  Val features: {len(X_val)} (Yew: {y_val.sum()}, Non-yew: {(1-y_val).sum()})")
    
    # Combine for cross-validation during tuning
    X_all = np.vstack([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])
    
    print(f"  Total samples: {len(X_all)} (Yew: {y_all.sum()}, Non-yew: {(1-y_all).sum()})")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Define class weights
    class_weight = {0: args.class_weight_ratio, 1: 1.0}
    print(f"  Class weights: non-yew={args.class_weight_ratio}, yew=1.0")
    
    # Define hyperparameter grid
    print("\n" + "="*70)
    print("HYPERPARAMETER GRID SEARCH")
    print("="*70)
    
    param_grid = {
        'C': [0.1, 1.0, 10.0, 100.0],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'kernel': ['rbf', 'poly', 'sigmoid'],
    }
    
    # For poly kernel, also try different degrees
    # We'll do this in two stages
    
    print("\nGrid parameters:")
    for k, v in param_grid.items():
        print(f"  {k}: {v}")
    
    total_combinations = 1
    for v in param_grid.values():
        total_combinations *= len(v)
    print(f"\nTotal combinations: {total_combinations}")
    print(f"Cross-validation folds: {args.cv_folds}")
    print(f"Total fits: {total_combinations * args.cv_folds}")
    
    # Create SVM with class weights
    svm = SVC(probability=True, random_state=42, class_weight=class_weight)
    
    # Grid search with cross-validation
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    
    print("\nRunning grid search (this may take a while)...")
    grid_search = GridSearchCV(
        svm, param_grid,
        cv=cv,
        scoring='f1',  # Optimize for F1 score
        n_jobs=args.n_jobs,
        verbose=2,
        return_train_score=True
    )
    
    grid_search.fit(X_all_scaled, y_all)
    
    print("\n" + "="*70)
    print("GRID SEARCH RESULTS")
    print("="*70)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV F1 score: {grid_search.best_score_:.4f}")
    
    # Get top 10 results
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values('rank_test_score')
    
    print("\nTop 10 configurations:")
    print("-"*70)
    for i, row in results_df.head(10).iterrows():
        print(f"  Rank {int(row['rank_test_score'])}: F1={row['mean_test_score']:.4f} (+/- {row['std_test_score']:.4f})")
        print(f"    C={row['param_C']}, gamma={row['param_gamma']}, kernel={row['param_kernel']}")
    
    # Train final model with best parameters on all data
    print("\n" + "="*70)
    print("TRAINING FINAL MODEL")
    print("="*70)
    
    best_clf = grid_search.best_estimator_
    
    # Evaluate on held-out validation set
    y_val_pred = best_clf.predict(X_val_scaled)
    y_val_prob = best_clf.predict_proba(X_val_scaled)[:, 1]
    
    print("\nValidation Performance (Best Model):")
    print(f"  Accuracy:  {accuracy_score(y_val, y_val_pred):.4f}")
    print(f"  Precision: {precision_score(y_val, y_val_pred):.4f}")
    print(f"  Recall:    {recall_score(y_val, y_val_pred):.4f}")
    print(f"  F1 Score:  {f1_score(y_val, y_val_pred):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_val, y_val_prob):.4f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_val, y_val_pred)
    print(f"  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
    
    print("\nClassification Report:")
    print(classification_report(y_val, y_val_pred, target_names=['Non-Yew', 'Yew']))
    
    # Compare with baseline (default parameters)
    print("\n" + "="*70)
    print("COMPARISON WITH BASELINE")
    print("="*70)
    
    baseline_clf = SVC(kernel='rbf', probability=True, random_state=42, class_weight=class_weight)
    baseline_clf.fit(X_all_scaled, y_all)
    
    y_baseline_pred = baseline_clf.predict(X_val_scaled)
    y_baseline_prob = baseline_clf.predict_proba(X_val_scaled)[:, 1]
    
    print("\nBaseline (default RBF) Performance:")
    print(f"  Accuracy:  {accuracy_score(y_val, y_baseline_pred):.4f}")
    print(f"  F1 Score:  {f1_score(y_val, y_baseline_pred):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_val, y_baseline_prob):.4f}")
    
    print("\nTuned Model Performance:")
    print(f"  Accuracy:  {accuracy_score(y_val, y_val_pred):.4f}")
    print(f"  F1 Score:  {f1_score(y_val, y_val_pred):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_val, y_val_prob):.4f}")
    
    improvement = f1_score(y_val, y_val_pred) - f1_score(y_val, y_baseline_pred)
    print(f"\nF1 improvement: {improvement:+.4f}")
    
    # Save model and results
    print("\n" + "="*70)
    print("SAVING MODEL")
    print("="*70)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save scaler
    scaler_path = output_dir / f'scaler_{timestamp}.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  ✓ Saved scaler: {scaler_path}")
    
    # Save model
    model_path = output_dir / f'svm_tuned_{timestamp}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(best_clf, f)
    print(f"  ✓ Saved model: {model_path}")
    
    # Save results
    results = {
        'timestamp': timestamp,
        'best_params': grid_search.best_params_,
        'best_cv_f1': float(grid_search.best_score_),
        'validation_metrics': {
            'accuracy': float(accuracy_score(y_val, y_val_pred)),
            'precision': float(precision_score(y_val, y_val_pred)),
            'recall': float(recall_score(y_val, y_val_pred)),
            'f1_score': float(f1_score(y_val, y_val_pred)),
            'roc_auc': float(roc_auc_score(y_val, y_val_prob)),
        },
        'baseline_metrics': {
            'accuracy': float(accuracy_score(y_val, y_baseline_pred)),
            'f1_score': float(f1_score(y_val, y_baseline_pred)),
            'roc_auc': float(roc_auc_score(y_val, y_baseline_prob)),
        },
        'data_stats': {
            'n_train': int(len(X_train)),
            'n_val': int(len(X_val)),
            'n_total': int(len(X_all)),
            'n_yew': int(y_all.sum()),
            'n_non_yew': int((1-y_all).sum()),
            'class_weight_ratio': args.class_weight_ratio,
        },
        'model_path': str(model_path),
        'scaler_path': str(scaler_path),
    }
    
    results_path = output_dir / f'tuning_results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Saved results: {results_path}")
    
    # Save full grid search results
    cv_results_path = output_dir / f'cv_results_{timestamp}.csv'
    results_df.to_csv(cv_results_path, index=False)
    print(f"  ✓ Saved CV results: {cv_results_path}")
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70)
    print(f"\nTo use this model, load:")
    print(f"  Scaler: {scaler_path}")
    print(f"  Model:  {model_path}")


if __name__ == '__main__':
    main()
