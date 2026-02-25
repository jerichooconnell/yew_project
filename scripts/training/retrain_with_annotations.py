#!/usr/bin/env python3
"""
Retrain SVM using human annotations from the annotation tool.

This script takes the exported CSV from the annotation tool and extracts
embedding features directly from the cached embedding_image.npy (the same
full-area embedding used for classification). It then combines these new
annotations with the existing training data to retrain the SVM.

Usage:
    python scripts/training/retrain_with_annotations.py \
        --annotations results/predictions/jordan_river_tuned/yew_annotations.csv \
        --embedding-image results/predictions/jordan_river_tuned/embedding_image.npy \
        --metadata results/predictions/jordan_river_tuned/metadata.json \
        --output-dir models/svm_annotated
"""

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm


def extract_features_from_annotations(annotations_csv, embedding_image, metadata):
    """
    Extract 64-band embedding features from the annotation pixel locations.
    
    The annotation CSV has px_row, px_col which map directly into the
    embedding_image array.
    """
    df = pd.read_csv(annotations_csv)
    print(f"  Loaded {len(df)} annotations")
    print(f"    Yew: {(df['has_yew'] == 1).sum()}")
    print(f"    Not Yew: {(df['has_yew'] == 0).sum()}")
    
    h, w, n_bands = embedding_image.shape
    features = []
    labels = []
    skipped = 0
    
    for _, row in df.iterrows():
        pr = int(row['px_row'])
        pc = int(row['px_col'])
        
        # Bounds check
        if 0 <= pr < h and 0 <= pc < w:
            feat = embedding_image[pr, pc, :]  # 64-band feature vector
            if not np.all(feat == 0) and not np.any(np.isnan(feat)):
                features.append(feat)
                labels.append(int(row['has_yew']))
            else:
                skipped += 1
        else:
            skipped += 1
    
    if skipped > 0:
        print(f"    Skipped {skipped} annotations (out of bounds or zero embeddings)")
    
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32)


def extract_features_from_split(df, emb_dir, patch_size=64):
    """Extract center pixel features from individual embedding files."""
    features = []
    labels = []
    center = patch_size // 2
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc='  Loading embeddings'):
        if pd.notna(row['lat']) and pd.notna(row['lon']):
            emb_path = emb_dir / f"embedding_{row['lat']:.6f}_{row['lon']:.6f}.npy"
            if emb_path.exists():
                try:
                    img = np.load(emb_path)
                    if img.ndim == 3:
                        if img.shape[0] == 64:
                            feat = img[:, center, center]
                        elif img.shape[2] == 64:
                            feat = img[center, center, :]
                        else:
                            continue
                        features.append(feat)
                        labels.append(int(row['has_yew']))
                except Exception:
                    continue
    
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser(
        description='Retrain SVM with human annotations',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--annotations', type=str, required=True,
                        help='CSV file from annotation tool')
    parser.add_argument('--embedding-image', type=str, required=True,
                        help='Path to embedding_image.npy from classification')
    parser.add_argument('--metadata', type=str, required=True,
                        help='Path to metadata.json from classification')
    parser.add_argument('--existing-train-csv', type=str,
                        default='data/processed/train_split_balanced_max.csv',
                        help='Existing training CSV')
    parser.add_argument('--existing-val-csv', type=str,
                        default='data/processed/val_split_balanced_max.csv',
                        help='Existing validation CSV')
    parser.add_argument('--existing-emb-dir', type=str,
                        default='data/processed/embeddings',
                        help='Directory with existing embedding .npy files')
    parser.add_argument('--output-dir', type=str, default='models/svm_annotated',
                        help='Output directory for retrained model')
    parser.add_argument('--class-weight-ratio', type=float, default=5.0,
                        help='Class weight ratio for non-yew (default: 5.0)')
    parser.add_argument('--C', type=float, default=10.0,
                        help='SVM C parameter (default: 10.0)')
    parser.add_argument('--gamma', type=str, default='scale',
                        help='SVM gamma parameter (default: scale)')
    parser.add_argument('--annotation-weight', type=float, default=3.0,
                        help='How much to weight annotation samples relative to originals (default: 3.0)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Only use annotations for training (skip existing training data)')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("SVM RETRAINING WITH HUMAN ANNOTATIONS")
    print("=" * 60)
    
    # Step 1: Load annotation features
    print("\n1. Extracting features from annotations...")
    embedding_image = np.load(args.embedding_image)
    with open(args.metadata, 'r') as f:
        metadata = json.load(f)
    
    X_ann, y_ann = extract_features_from_annotations(
        args.annotations, embedding_image, metadata
    )
    print(f"  Annotation features: {len(X_ann)} (Yew: {y_ann.sum()}, Not Yew: {(1-y_ann).sum()})")
    
    if len(X_ann) == 0:
        print("ERROR: No valid annotation features extracted!")
        return
    
    # Step 2: Load existing training data
    if not args.skip_existing:
        print("\n2. Loading existing training data...")
        emb_dir = Path(args.existing_emb_dir)
        
        train_df = pd.read_csv(args.existing_train_csv)
        val_df = pd.read_csv(args.existing_val_csv)
        print(f"  Existing train CSV: {len(train_df)} rows")
        print(f"  Existing val CSV: {len(val_df)} rows")
        
        X_train_existing, y_train_existing = extract_features_from_split(train_df, emb_dir)
        X_val_existing, y_val_existing = extract_features_from_split(val_df, emb_dir)
        print(f"  Existing train features: {len(X_train_existing)} (Yew: {y_train_existing.sum()})")
        print(f"  Existing val features: {len(X_val_existing)} (Yew: {y_val_existing.sum()})")
    else:
        X_train_existing = np.zeros((0, X_ann.shape[1]), dtype=np.float32)
        y_train_existing = np.zeros(0, dtype=np.int32)
        X_val_existing = np.zeros((0, X_ann.shape[1]), dtype=np.float32)
        y_val_existing = np.zeros(0, dtype=np.int32)
    
    # Step 3: Combine data with annotation weighting
    # Repeat annotation data to give it more influence
    print(f"\n3. Combining datasets (annotation weight: {args.annotation_weight}x)...")
    
    n_repeats = max(1, int(args.annotation_weight))
    X_ann_weighted = np.tile(X_ann, (n_repeats, 1))
    y_ann_weighted = np.tile(y_ann, n_repeats)
    
    X_all = np.vstack([X_train_existing, X_val_existing, X_ann_weighted])
    y_all = np.concatenate([y_train_existing, y_val_existing, y_ann_weighted])
    
    # Clean
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    
    yew_count = int(y_all.sum())
    notyew_count = int((1 - y_all).sum())
    print(f"  Combined: {len(X_all)} samples (Yew: {yew_count}, Not Yew: {notyew_count})")
    print(f"    From existing: {len(X_train_existing) + len(X_val_existing)}")
    print(f"    From annotations: {len(X_ann)} × {n_repeats} = {len(X_ann_weighted)}")
    
    # Step 4: Train
    print(f"\n4. Training SVM (C={args.C}, gamma={args.gamma})...")
    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)
    
    class_weight = {0: args.class_weight_ratio, 1: 1.0}
    print(f"  Class weights: non-yew={args.class_weight_ratio}, yew=1.0")
    
    gamma = args.gamma if args.gamma in ('scale', 'auto') else float(args.gamma)
    clf = SVC(kernel='rbf', C=args.C, gamma=gamma, 
              probability=True, random_state=42, class_weight=class_weight)
    clf.fit(X_all_scaled, y_all)
    print(f"  ✓ SVM trained on {len(X_all)} samples")
    
    # Step 5: Evaluate on annotations (in-sample check)
    print(f"\n5. Evaluation...")
    X_ann_scaled = scaler.transform(X_ann)
    y_ann_pred = clf.predict(X_ann_scaled)
    y_ann_prob = clf.predict_proba(X_ann_scaled)[:, 1]
    
    print(f"\n  Annotation accuracy (in-sample):")
    print(f"    Accuracy:  {accuracy_score(y_ann, y_ann_pred):.4f}")
    print(f"    F1 Score:  {f1_score(y_ann, y_ann_pred):.4f}")
    if len(np.unique(y_ann)) > 1:
        print(f"    ROC-AUC:   {roc_auc_score(y_ann, y_ann_prob):.4f}")
    
    print(f"\n  Confusion Matrix (annotations only):")
    cm = confusion_matrix(y_ann, y_ann_pred)
    print(f"    TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"    FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
    
    print(f"\n  Classification Report:")
    print(classification_report(y_ann, y_ann_pred, target_names=['Not Yew', 'Yew']))
    
    # Evaluate on existing validation if available
    if len(X_val_existing) > 0:
        X_val_scaled = scaler.transform(X_val_existing)
        y_val_pred = clf.predict(X_val_scaled)
        y_val_prob = clf.predict_proba(X_val_scaled)[:, 1]
        
        print(f"\n  Existing validation set performance:")
        print(f"    Accuracy:  {accuracy_score(y_val_existing, y_val_pred):.4f}")
        print(f"    F1 Score:  {f1_score(y_val_existing, y_val_pred):.4f}")
        if len(np.unique(y_val_existing)) > 1:
            print(f"    ROC-AUC:   {roc_auc_score(y_val_existing, y_val_prob):.4f}")
    
    # Step 6: Save
    print(f"\n6. Saving model...")
    model_path = output_dir / f'svm_annotated_{timestamp}.pkl'
    scaler_path = output_dir / f'scaler_annotated_{timestamp}.pkl'
    results_path = output_dir / f'retrain_results_{timestamp}.json'
    
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"  ✓ Model: {model_path}")
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  ✓ Scaler: {scaler_path}")
    
    results = {
        'timestamp': timestamp,
        'annotations_file': str(args.annotations),
        'n_annotations': len(X_ann),
        'n_annotations_yew': int(y_ann.sum()),
        'n_annotations_notyew': int((1 - y_ann).sum()),
        'n_existing': len(X_train_existing) + len(X_val_existing),
        'n_total_training': len(X_all),
        'annotation_weight': args.annotation_weight,
        'class_weight_ratio': args.class_weight_ratio,
        'svm_C': args.C,
        'svm_gamma': args.gamma,
        'annotation_accuracy': float(accuracy_score(y_ann, y_ann_pred)),
        'annotation_f1': float(f1_score(y_ann, y_ann_pred)),
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Results: {results_path}")
    
    print(f"\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"\nTo classify with this model:")
    print(f"  python scripts/prediction/classify_large_area_export.py \\")
    print(f"    --bbox <lat_min> <lat_max> <lon_min> <lon_max> \\")
    print(f"    --model-path {model_path} \\")
    print(f"    --scaler-path {scaler_path} \\")
    print(f"    --output-dir results/predictions/<area_name>_annotated")


if __name__ == '__main__':
    main()
