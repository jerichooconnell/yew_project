#!/usr/bin/env python3
"""
Compare classifier approaches on AlphaEarth satellite embeddings.

Tests the methods recommended by the AlphaEarth Foundations paper
(Brown et al., 2025) — kNN and linear probes — against our current
MLP, plus Random Forest, on both raw and StandardScaler'd embeddings.

Outputs:
  - Console comparison table
  - results/training/classifier_comparison.json
  - Saves best model artifacts to results/predictions/south_vi_large/

Usage:
    python scripts/training/compare_classifiers.py
"""

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, roc_auc_score, precision_score,
                             recall_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# ─── YewMLP (same as production) ────────────────────────────────────────────
class YewMLP(nn.Module):
    def __init__(self, input_dim=64, hidden_dims=(128, 64, 32)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ─── Data loading ────────────────────────────────────────────────────────────
def extract_center_pixel(lat, lon, emb_dir, patch_size=64):
    emb_path = emb_dir / f'embedding_{lat:.6f}_{lon:.6f}.npy'
    if not emb_path.exists():
        return None
    try:
        img = np.load(emb_path)
        center = patch_size // 2
        if img.ndim == 3:
            if img.shape[0] == 64:
                return img[:, center, center]
            elif img.shape[2] == 64:
                return img[center, center, :]
        return None
    except Exception:
        return None


def load_training_data(train_csv, val_csv, emb_dir):
    emb_dir = Path(emb_dir)

    def extract(df, label):
        features, labels = [], []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f'  {label}'):
            if pd.notna(row['lat']) and pd.notna(row['lon']):
                feat = extract_center_pixel(row['lat'], row['lon'], emb_dir)
                if feat is not None:
                    features.append(feat)
                    labels.append(int(row['has_yew']))
        return np.array(features, np.float32), np.array(labels, np.int32)

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    print(f"  CSVs: {len(train_df)} train, {len(val_df)} val")

    X_train, y_train = extract(train_df, 'Train')
    X_val, y_val = extract(val_df, 'Val')

    X_all = np.vstack([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  Total: {len(X_all)} samples ({y_all.sum()} yew, "
          f"{len(y_all)-y_all.sum()} non-yew)")
    print(f"  Val:   {len(X_val)} ({y_val.sum()} yew, {(y_val==0).sum()} non-yew)")
    return X_all, y_all, X_val, y_val


def load_gee_negatives(csv_path, weight=1, val_fraction=0.0, seed=42):
    df = pd.read_csv(csv_path)
    emb_cols = [c for c in df.columns if c.startswith('emb_')]
    X = df[emb_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    valid = np.any(X != 0, axis=1)
    X = X[valid]
    print(f"  GEE negatives: {valid.sum()} valid from {csv_path}")

    if val_fraction > 0:
        rng = np.random.RandomState(seed)
        n = len(X)
        n_val = int(n * val_fraction)
        idx = rng.permutation(n)
        X_train, X_val = X[idx[n_val:]], X[idx[:n_val]]
        y_train = np.zeros(len(X_train), dtype=np.int32)
        y_val = np.zeros(len(X_val), dtype=np.int32)
        if weight > 1:
            X_train = np.tile(X_train, (weight, 1))
            y_train = np.tile(y_train, weight)
        return X_train, y_train, X_val, y_val
    else:
        y = np.zeros(len(X), dtype=np.int32)
        if weight > 1:
            X = np.tile(X, (weight, 1))
            y = np.tile(y, weight)
        return X, y


# ─── MLP training ────────────────────────────────────────────────────────────
def train_mlp(X_train, y_train, X_val, y_val, device, use_scaler=True,
              epochs=100, lr=0.001, batch_size=512):
    """Train MLP. Returns (model, scaler_or_None, metrics, probs_val)."""
    scaler = None
    if use_scaler:
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train).astype(np.float32)
        X_v_s = scaler.transform(X_val).astype(np.float32)
    else:
        X_tr_s = X_train.astype(np.float32)
        X_v_s = X_val.astype(np.float32)

    X_t = torch.from_numpy(X_tr_s).to(device)
    y_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
    X_val_t = torch.from_numpy(X_v_s).to(device)

    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
    model = YewMLP(input_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_f1 = 0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                probs = torch.sigmoid(model(X_val_t)).cpu().numpy()
                f1 = f1_score(y_val, (probs >= 0.5).astype(int))
            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 6:
                    break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        probs = torch.sigmoid(model(X_val_t)).cpu().numpy()

    return model, scaler, probs


# ─── Evaluation ──────────────────────────────────────────────────────────────
def evaluate(y_true, y_prob, name):
    """Compute metrics dict from true labels and predicted probabilities."""
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        'name': name,
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_true, y_prob)),
    }


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Compare classifiers on AEF embeddings')
    parser.add_argument('--train-csv', default='data/processed/train_split_balanced_max.csv')
    parser.add_argument('--val-csv', default='data/processed/val_split_balanced_max.csv')
    parser.add_argument('--emb-dir', default='data/ee_imagery/embedding_patches_64x64')
    parser.add_argument('--gee-negatives', default='data/processed/combined_negative_embeddings.csv')
    parser.add_argument('--gee-negatives-weight', type=int, default=2)
    parser.add_argument('--output-dir', default='results/training')
    parser.add_argument('--model-dir', default='results/predictions/south_vi_large')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print("=" * 70)
    print("CLASSIFIER COMPARISON: AlphaEarth Satellite Embeddings")
    print("=" * 70)
    print(f"Device: {device}")
    print()

    # ── Load data ─────────────────────────────────────────────────────────
    print("Loading training data...")
    X_all, y_all, X_val, y_val = load_training_data(
        args.train_csv, args.val_csv, args.emb_dir
    )

    # Add GEE negatives
    print(f"\nLoading GEE negatives...")
    has_val_neg = (y_val == 0).sum() > 0
    val_frac = 0.2 if not has_val_neg else 0.0

    result = load_gee_negatives(
        args.gee_negatives, weight=args.gee_negatives_weight,
        val_fraction=val_frac
    )
    if val_frac > 0:
        X_neg_tr, y_neg_tr, X_neg_val, y_neg_val = result
        X_all = np.vstack([X_all, X_neg_tr])
        y_all = np.concatenate([y_all, y_neg_tr])
        X_val = np.vstack([X_val, X_neg_val])
        y_val = np.concatenate([y_val, y_neg_val])
    else:
        X_neg, y_neg = result
        X_all = np.vstack([X_all, X_neg])
        y_all = np.concatenate([y_all, y_neg])

    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\nFinal dataset:")
    print(f"  Train: {len(X_all)} ({y_all.sum()} yew, {(y_all==0).sum()} non-yew)")
    print(f"  Val:   {len(X_val)} ({y_val.sum()} yew, {(y_val==0).sum()} non-yew)")

    # Check embedding norms to confirm unit-length property
    norms = np.linalg.norm(X_all, axis=1)
    valid_norms = norms[norms > 0.01]
    print(f"\n  Embedding L2 norms: mean={valid_norms.mean():.4f}, "
          f"std={valid_norms.std():.4f}, min={valid_norms.min():.4f}, "
          f"max={valid_norms.max():.4f}")
    if abs(valid_norms.mean() - 1.0) < 0.15:
        print("  → Embeddings are approximately unit-length (confirming AEF design)")
    else:
        print(f"  → Embeddings are NOT unit-length (mean norm={valid_norms.mean():.3f})")

    # ── Train all classifiers ─────────────────────────────────────────────
    results = []

    # --- 1. kNN (k=1) on RAW embeddings (AEF paper method) ---
    print(f"\n{'='*70}")
    print("1. kNN (k=1) — raw embeddings [AEF paper method]")
    print(f"{'='*70}")
    t0 = time.time()
    knn1 = KNeighborsClassifier(n_neighbors=1, metric='euclidean', n_jobs=-1)
    knn1.fit(X_all, y_all)
    # kNN doesn't give probabilities directly; use distance-based weighting
    # For kNN k=1, we use predict_proba via internal distance logic
    knn1_probs = knn1.predict_proba(X_val)[:, 1]
    elapsed = time.time() - t0
    m = evaluate(y_val, knn1_probs, 'kNN (k=1) raw')
    m['time_s'] = round(elapsed, 2)
    m['scaling'] = 'none'
    results.append(m)
    print(f"  {elapsed:.1f}s — Acc={m['accuracy']:.4f} F1={m['f1']:.4f} "
          f"AUC={m['roc_auc']:.4f} BA={m['balanced_accuracy']:.4f}")

    # --- 2. kNN (k=3) on RAW embeddings (AEF paper method) ---
    print(f"\n{'='*70}")
    print("2. kNN (k=3) — raw embeddings [AEF paper method]")
    print(f"{'='*70}")
    t0 = time.time()
    knn3 = KNeighborsClassifier(n_neighbors=3, metric='euclidean', n_jobs=-1)
    knn3.fit(X_all, y_all)
    knn3_probs = knn3.predict_proba(X_val)[:, 1]
    elapsed = time.time() - t0
    m = evaluate(y_val, knn3_probs, 'kNN (k=3) raw')
    m['time_s'] = round(elapsed, 2)
    m['scaling'] = 'none'
    results.append(m)
    print(f"  {elapsed:.1f}s — Acc={m['accuracy']:.4f} F1={m['f1']:.4f} "
          f"AUC={m['roc_auc']:.4f} BA={m['balanced_accuracy']:.4f}")

    # --- 3. kNN (k=5) on RAW embeddings ---
    print(f"\n{'='*70}")
    print("3. kNN (k=5) — raw embeddings")
    print(f"{'='*70}")
    t0 = time.time()
    knn5 = KNeighborsClassifier(n_neighbors=5, metric='euclidean', n_jobs=-1)
    knn5.fit(X_all, y_all)
    knn5_probs = knn5.predict_proba(X_val)[:, 1]
    elapsed = time.time() - t0
    m = evaluate(y_val, knn5_probs, 'kNN (k=5) raw')
    m['time_s'] = round(elapsed, 2)
    m['scaling'] = 'none'
    results.append(m)
    print(f"  {elapsed:.1f}s — Acc={m['accuracy']:.4f} F1={m['f1']:.4f} "
          f"AUC={m['roc_auc']:.4f} BA={m['balanced_accuracy']:.4f}")

    # --- 4. Linear probe / logistic regression on RAW (AEF paper method) ---
    print(f"\n{'='*70}")
    print("4. Logistic Regression — raw embeddings [AEF paper: linear probe]")
    print(f"{'='*70}")
    t0 = time.time()
    lr_raw = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', n_jobs=-1)
    lr_raw.fit(X_all, y_all)
    lr_raw_probs = lr_raw.predict_proba(X_val)[:, 1]
    elapsed = time.time() - t0
    m = evaluate(y_val, lr_raw_probs, 'Logistic Regression raw')
    m['time_s'] = round(elapsed, 2)
    m['scaling'] = 'none'
    results.append(m)
    print(f"  {elapsed:.1f}s — Acc={m['accuracy']:.4f} F1={m['f1']:.4f} "
          f"AUC={m['roc_auc']:.4f} BA={m['balanced_accuracy']:.4f}")

    # --- 5. Ridge Classifier on RAW (exact AEF paper method) ---
    print(f"\n{'='*70}")
    print("5. Ridge Classifier — raw embeddings [exact AEF paper method]")
    print(f"{'='*70}")
    t0 = time.time()
    ridge = RidgeClassifier(alpha=0.0)
    ridge.fit(X_all, y_all)
    # RidgeClassifier doesn't have predict_proba; use decision_function
    ridge_scores = ridge.decision_function(X_val)
    # Convert to probabilities via sigmoid
    ridge_probs = 1.0 / (1.0 + np.exp(-ridge_scores))
    elapsed = time.time() - t0
    m = evaluate(y_val, ridge_probs, 'Ridge Classifier raw')
    m['time_s'] = round(elapsed, 2)
    m['scaling'] = 'none'
    results.append(m)
    print(f"  {elapsed:.1f}s — Acc={m['accuracy']:.4f} F1={m['f1']:.4f} "
          f"AUC={m['roc_auc']:.4f} BA={m['balanced_accuracy']:.4f}")

    # --- 6. Random Forest on RAW ---
    print(f"\n{'='*70}")
    print("6. Random Forest — raw embeddings")
    print(f"{'='*70}")
    t0 = time.time()
    rf_raw = RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=5,
        class_weight='balanced', n_jobs=-1, random_state=42
    )
    rf_raw.fit(X_all, y_all)
    rf_raw_probs = rf_raw.predict_proba(X_val)[:, 1]
    elapsed = time.time() - t0
    m = evaluate(y_val, rf_raw_probs, 'Random Forest raw')
    m['time_s'] = round(elapsed, 2)
    m['scaling'] = 'none'
    results.append(m)
    print(f"  {elapsed:.1f}s — Acc={m['accuracy']:.4f} F1={m['f1']:.4f} "
          f"AUC={m['roc_auc']:.4f} BA={m['balanced_accuracy']:.4f}")

    # --- 7. Random Forest on StandardScaler'd embeddings ---
    print(f"\n{'='*70}")
    print("7. Random Forest — StandardScaler embeddings")
    print(f"{'='*70}")
    scaler_rf = StandardScaler()
    X_all_s = scaler_rf.fit_transform(X_all)
    X_val_s = scaler_rf.transform(X_val)
    t0 = time.time()
    rf_scaled = RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=5,
        class_weight='balanced', n_jobs=-1, random_state=42
    )
    rf_scaled.fit(X_all_s, y_all)
    rf_scaled_probs = rf_scaled.predict_proba(X_val_s)[:, 1]
    elapsed = time.time() - t0
    m = evaluate(y_val, rf_scaled_probs, 'Random Forest scaled')
    m['time_s'] = round(elapsed, 2)
    m['scaling'] = 'StandardScaler'
    results.append(m)
    print(f"  {elapsed:.1f}s — Acc={m['accuracy']:.4f} F1={m['f1']:.4f} "
          f"AUC={m['roc_auc']:.4f} BA={m['balanced_accuracy']:.4f}")

    # --- 8. MLP with StandardScaler (CURRENT production method) ---
    print(f"\n{'='*70}")
    print("8. MLP — StandardScaler embeddings [CURRENT production method]")
    print(f"{'='*70}")
    t0 = time.time()
    mlp_model_s, mlp_scaler_s, mlp_probs_s = train_mlp(
        X_all, y_all, X_val, y_val, device, use_scaler=True
    )
    elapsed = time.time() - t0
    m = evaluate(y_val, mlp_probs_s, 'MLP + StandardScaler')
    m['time_s'] = round(elapsed, 2)
    m['scaling'] = 'StandardScaler'
    results.append(m)
    print(f"  {elapsed:.1f}s — Acc={m['accuracy']:.4f} F1={m['f1']:.4f} "
          f"AUC={m['roc_auc']:.4f} BA={m['balanced_accuracy']:.4f}")

    # --- 9. MLP on RAW embeddings (no scaler) ---
    print(f"\n{'='*70}")
    print("9. MLP — raw embeddings (no scaler)")
    print(f"{'='*70}")
    t0 = time.time()
    mlp_model_r, _, mlp_probs_r = train_mlp(
        X_all, y_all, X_val, y_val, device, use_scaler=False
    )
    elapsed = time.time() - t0
    m = evaluate(y_val, mlp_probs_r, 'MLP raw')
    m['time_s'] = round(elapsed, 2)
    m['scaling'] = 'none'
    results.append(m)
    print(f"  {elapsed:.1f}s — Acc={m['accuracy']:.4f} F1={m['f1']:.4f} "
          f"AUC={m['roc_auc']:.4f} BA={m['balanced_accuracy']:.4f}")

    # ── Results table ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("COMPARISON RESULTS")
    print(f"{'='*70}")
    print(f"\n{'Name':<32} {'Acc':>6} {'BA':>6} {'F1':>6} {'Prec':>6} "
          f"{'Rec':>6} {'AUC':>6} {'Scale':>12} {'Time':>6}")
    print("-" * 96)

    # Sort by AUC descending
    results.sort(key=lambda x: x['roc_auc'], reverse=True)
    for r in results:
        print(f"{r['name']:<32} {r['accuracy']:>6.4f} {r['balanced_accuracy']:>6.4f} "
              f"{r['f1']:>6.4f} {r['precision']:>6.4f} {r['recall']:>6.4f} "
              f"{r['roc_auc']:>6.4f} {r['scaling']:>12} {r['time_s']:>5.1f}s")

    # ── Save results ──────────────────────────────────────────────────────
    json_path = out_dir / 'classifier_comparison.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {json_path}")

    # ── Identify best model and save artifacts ────────────────────────────
    best = results[0]  # highest AUC
    print(f"\n{'='*70}")
    print(f"BEST MODEL: {best['name']} (AUC={best['roc_auc']:.4f}, F1={best['f1']:.4f})")
    print(f"{'='*70}")

    # Save the best classifier for use by classify_cwh_spots.py
    # We save all viable classifiers so the spot classifier can pick
    artifacts_dir = model_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Always save: RF (raw), kNN (k=3, raw), Logistic (raw), MLP (scaled), MLP (raw)
    print("\nSaving classifier artifacts...")

    # Random Forest (raw)
    with open(artifacts_dir / 'rf_raw_model.pkl', 'wb') as f:
        pickle.dump(rf_raw, f)
    print(f"  ✓ rf_raw_model.pkl")

    # kNN (k=3, raw)
    with open(artifacts_dir / 'knn3_raw_model.pkl', 'wb') as f:
        pickle.dump(knn3, f)
    print(f"  ✓ knn3_raw_model.pkl")

    # Logistic regression (raw)
    with open(artifacts_dir / 'logistic_raw_model.pkl', 'wb') as f:
        pickle.dump(lr_raw, f)
    print(f"  ✓ logistic_raw_model.pkl")

    # MLP + StandardScaler (current production)
    torch.save(mlp_model_s.state_dict(), artifacts_dir / 'mlp_scaled_model.pth')
    with open(artifacts_dir / 'mlp_scaler.pkl', 'wb') as f:
        pickle.dump(mlp_scaler_s, f)
    print(f"  ✓ mlp_scaled_model.pth + mlp_scaler.pkl")

    # MLP raw (no scaler)
    torch.save(mlp_model_r.state_dict(), artifacts_dir / 'mlp_raw_model.pth')
    print(f"  ✓ mlp_raw_model.pth")

    print(f"\n✓ All artifacts saved to {artifacts_dir}")
    print(f"\nTo regenerate maps with a specific classifier:")
    print(f"  python scripts/prediction/classify_cwh_spots.py "
          f"--classifier rf_raw --force-reclassify")


if __name__ == '__main__':
    main()
