#!/usr/bin/env python3
"""
Evaluate the center pixel classifier used in predict_center_pixel_map.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import json


def extract_center_pixel(lat, lon, emb_dir, patch_size=64):
    """Extract center pixel from embedding."""
    emb_path = emb_dir / f'embedding_{lat:.6f}_{lon:.6f}.npy'
    if not emb_path.exists():
        return None
    try:
        img = np.load(emb_path)
        center = patch_size // 2
        return img[:, center, center]
    except Exception:
        return None


def extract_features_from_split(df, emb_dir):
    """Extract center pixel features and labels."""
    features = []
    labels = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Extracting features'):
        if pd.notna(row['lat']) and pd.notna(row['lon']):
            center_data = extract_center_pixel(row['lat'], row['lon'], emb_dir)
            if center_data is not None:
                features.append(center_data)
                labels.append(int(row['has_yew']))
    return np.array(features), np.array(labels)


def evaluate_model(clf, X, y, set_name='Test'):
    """Compute classification metrics."""
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)[:, 1]
    
    metrics = {
        'accuracy': float(accuracy_score(y, y_pred)),
        'precision': float(precision_score(y, y_pred, zero_division=0)),
        'recall': float(recall_score(y, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, y_prob)),
        'n_samples': int(len(y)),
        'n_positive': int(y.sum()),
        'n_negative': int(len(y) - y.sum())
    }
    
    cm = confusion_matrix(y, y_pred)
    metrics['confusion_matrix'] = {
        'true_negative': int(cm[0, 0]),
        'false_positive': int(cm[0, 1]),
        'false_negative': int(cm[1, 0]),
        'true_positive': int(cm[1, 1])
    }
    
    print(f"\n{set_name} Set Results:")
    print(f"  Samples: {metrics['n_samples']} (Yew: {metrics['n_positive']}, Non-yew: {metrics['n_negative']})")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN: {cm[0, 0]:4d}  FP: {cm[0, 1]:4d}")
    print(f"    FN: {cm[1, 0]:4d}  TP: {cm[1, 1]:4d}")
    
    return metrics


def main():
    print("Evaluating Center Pixel Classifier (with StandardScaler)")
    print("="*70)
    
    embedding_dir = Path('data/ee_imagery/embedding_patches_64x64')
    train_path = 'data/processed/train_split_filtered.csv'
    val_path = 'data/processed/val_split_filtered.csv'
    
    # Load data
    print("\nLoading training data...")
    train_df = pd.read_csv(train_path)
    X_train, y_train = extract_features_from_split(train_df, embedding_dir)
    
    print("\nLoading validation data...")
    val_df = pd.read_csv(val_path)
    X_val, y_val = extract_features_from_split(val_df, embedding_dir)
    
    print(f"\nTotal training samples: {len(X_train)}")
    print(f"Total validation samples: {len(X_val)}")
    
    # Handle inf/nan
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Train with StandardScaler (same as predict_center_pixel_map.py)
    print("\nTraining SVM with StandardScaler (RBF kernel)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    clf = SVC(kernel='rbf', probability=True, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    # Evaluate
    results = {}
    results['train'] = evaluate_model(clf, X_train_scaled, y_train, 'Training')
    results['val'] = evaluate_model(clf, X_val_scaled, y_val, 'Validation')
    
    # Save
    output_path = Path('results/predictions/center_pixel_method/evaluation.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved results to {output_path}")
    
    # Print comparison note
    print("\n" + "="*70)
    print("NOTE: This uses StandardScaler (like predict_center_pixel_map.py)")
    print("Compare with classify_every_pixel_in_bbox.py which uses NO scaling")
    print("="*70)


if __name__ == '__main__':
    main()
