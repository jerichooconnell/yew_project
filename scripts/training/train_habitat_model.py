#!/usr/bin/env python3
"""
Train a YewMLP "habitat suitability" model WITHOUT forestry negatives.

Purpose:
  The production model uses 6,194 FAIB forestry negatives (forest inventory
  plots with no yew recorded — many of which are logged/managed sites where
  yew was removed by disturbance). This model EXCLUDES those, training only
  with:
    • Positives: iNaturalist yew observations + manual annotations
    • Negatives: Alpine/barren hard-negatives only (non-habitat by terrain)

  By comparing this habitat-suitability model's predictions on logged areas
  with the production model, we can estimate what yew density WOULD exist
  without forestry disturbance — enabling a historical population estimate.

Usage:
    python scripts/training/train_habitat_model.py

Outputs:
    results/predictions/south_vi_large/habitat_model.pth
    results/predictions/south_vi_large/habitat_scaler.pkl
    results/predictions/south_vi_large/habitat_metrics.json
"""

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ── Import shared components from classify_tiled_gpu ──────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'prediction'))
from classify_tiled_gpu import (
    YewMLP,
    extract_center_pixel,
    load_annotation_features,
)


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent.parent          # yew_project/
MODEL_DIR = BASE / 'results' / 'predictions' / 'south_vi_large'
TILE_CACHE = MODEL_DIR / 'tiles'

# Training data
TRAIN_CSV   = BASE / 'data' / 'processed' / 'inat_yew_positives_train.csv'
VAL_CSV     = BASE / 'data' / 'processed' / 'inat_yew_positives_val.csv'
ANNOT_CSV   = BASE / 'data' / 'raw' / 'yew_annotations_combined.csv'
EMB_DIR     = BASE / 'data' / 'ee_imagery' / 'embedding_patches_64x64'

# Negatives — ONLY alpine, no FAIB forestry
ALPINE_NEG  = BASE / 'data' / 'processed' / 'alpine_negatives' / 'alpine_negative_embeddings.csv'

# Outputs
OUT_MODEL   = MODEL_DIR / 'habitat_model.pth'
OUT_SCALER  = MODEL_DIR / 'habitat_scaler.pkl'
OUT_METRICS = MODEL_DIR / 'habitat_metrics.json'


def load_inat_data(train_csv, val_csv, emb_dir):
    """Load iNaturalist training data (positives + some non-yew)."""
    emb_dir = Path(emb_dir)

    def extract(df, label):
        features, labels = [], []
        for _, row in df.iterrows():
            if pd.notna(row['lat']) and pd.notna(row['lon']):
                feat = extract_center_pixel(row['lat'], row['lon'], emb_dir)
                if feat is not None:
                    features.append(feat)
                    labels.append(int(row['has_yew']))
        if features:
            print(f"  {label}: {len(features)} samples "
                  f"({sum(labels)} yew, {len(labels)-sum(labels)} non-yew)")
        return (np.array(features, np.float32), np.array(labels, np.int32)) if features \
            else (np.empty((0, 64), np.float32), np.empty(0, np.int32))

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    print(f"\nLoading iNaturalist data...")
    print(f"  CSVs: {len(train_df)} train, {len(val_df)} val")

    X_train, y_train = extract(train_df, 'Train')
    X_val, y_val = extract(val_df, 'Val')

    X_all = np.vstack([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

    return X_all, y_all, X_val, y_val


def load_alpine_negatives(csv_path, val_fraction=0.2, seed=42):
    """Load alpine/barren negatives (no forestry influence)."""
    df = pd.read_csv(csv_path)
    emb_cols = [c for c in df.columns if c.startswith('emb_')]
    X = df[emb_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    valid = np.any(X != 0, axis=1)
    X = X[valid]
    print(f"\nLoading alpine negatives: {len(X)} valid from {csv_path.name}")

    rng = np.random.RandomState(seed)
    n_val = int(len(X) * val_fraction)
    idx = rng.permutation(len(X))
    X_val, X_train = X[idx[:n_val]], X[idx[n_val:]]
    y_train = np.zeros(len(X_train), np.int32)
    y_val = np.zeros(len(X_val), np.int32)
    print(f"  Split: {len(X_train)} train, {len(X_val)} val")
    return X_train, y_train, X_val, y_val


def train_mlp(X_all, y_all, X_val, y_val, scaler, device,
              epochs=100, lr=0.001, batch_size=512):
    """Train YewMLP — identical to classify_tiled_gpu.train_mlp."""
    X_all_s = scaler.transform(X_all).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    X_t = torch.from_numpy(X_all_s).to(device)
    y_t = torch.from_numpy(y_all.astype(np.float32)).to(device)
    X_val_t = torch.from_numpy(X_val_s).to(device)
    y_val_t = torch.from_numpy(y_val.astype(np.float32)).to(device)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = YewMLP(input_dim=X_all.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_f1 = 0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t)
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
                val_preds = (val_probs >= 0.5).astype(int)
                val_labels = y_val

                acc = accuracy_score(val_labels, val_preds)
                f1 = f1_score(val_labels, val_preds)
                auc = roc_auc_score(val_labels, val_probs)

            avg_loss = total_loss / len(X_all)
            print(f"  Epoch {epoch+1:3d}/{epochs}: loss={avg_loss:.4f}, "
                  f"val_acc={acc:.4f}, val_F1={f1:.4f}, val_AUC={auc:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 6:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        val_logits = model(X_val_t)
        val_probs = torch.sigmoid(val_logits).cpu().numpy()
        val_preds = (val_probs >= 0.5).astype(int)
        metrics = {
            'accuracy': float(accuracy_score(y_val, val_preds)),
            'f1_score': float(f1_score(y_val, val_preds)),
            'roc_auc': float(roc_auc_score(y_val, val_probs)),
            'n_train': len(X_all),
            'epochs': epochs,
            'best_f1': float(best_f1),
        }
    print(f"\n  Best: acc={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}, "
          f"AUC={metrics['roc_auc']:.4f}")
    return model, metrics


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("HABITAT SUITABILITY MODEL (no forestry negatives)")
    print("=" * 70)
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── 1. Load positive data (iNat + annotations) ───────────────────────────
    X_all, y_all, X_val, y_val = load_inat_data(TRAIN_CSV, VAL_CSV, EMB_DIR)

    # Annotations (3× weight)
    with open(MODEL_DIR / 'tile_info.json') as f:
        tile_info = json.load(f)
    print(f"\nLoading annotations from {ANNOT_CSV.name}")
    X_ann, y_ann = load_annotation_features(
        str(ANNOT_CSV), TILE_CACHE, tile_info, weight=3
    )
    if len(X_ann) > 0:
        X_all = np.vstack([X_all, X_ann])
        y_all = np.concatenate([y_all, y_ann])
        X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\n  After iNat + annotations: {len(X_all)} samples "
          f"({y_all.sum()} yew, {len(y_all)-y_all.sum()} non-yew)")

    # ── 2. Load ONLY alpine negatives (no FAIB forestry) ─────────────────────
    X_neg_tr, y_neg_tr, X_neg_val, y_neg_val = load_alpine_negatives(ALPINE_NEG)

    X_all = np.vstack([X_all, X_neg_tr])
    y_all = np.concatenate([y_all, y_neg_tr])
    X_val = np.vstack([X_val, X_neg_val])
    y_val = np.concatenate([y_val, y_neg_val])
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\n  FINAL TRAINING DATA (habitat suitability — no forestry negatives):")
    print(f"    Train: {len(X_all)} ({y_all.sum()} yew, {len(y_all)-y_all.sum()} non-yew)")
    print(f"    Val:   {len(X_val)} ({y_val.sum()} yew, {len(y_val)-y_val.sum()} non-yew)")
    print(f"    Negatives: alpine/barren ONLY — no FAIB forest inventory plots")

    # ── 3. Train model ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("TRAINING")
    print(f"{'='*70}")

    scaler = StandardScaler()
    scaler.fit(X_all)

    model, metrics = train_mlp(X_all, y_all, X_val, y_val, scaler, device, epochs=100)

    # ── 4. Save ──────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), OUT_MODEL)
    with open(OUT_SCALER, 'wb') as f:
        pickle.dump(scaler, f)

    metrics['description'] = 'Habitat suitability model — trained without FAIB forestry negatives'
    metrics['negatives'] = 'alpine/barren only (2,800 samples)'
    metrics['positives'] = 'iNaturalist + manual annotations'
    with open(OUT_METRICS, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  ✓ Model:   {OUT_MODEL}")
    print(f"  ✓ Scaler:  {OUT_SCALER}")
    print(f"  ✓ Metrics: {OUT_METRICS}")

    # ── 5. Quick comparison on a single tile ─────────────────────────────────
    print(f"\n{'='*70}")
    print("QUICK COMPARISON: habitat model vs production model")
    print(f"{'='*70}")

    # Load production model
    prod_model = YewMLP(input_dim=64).to(device)
    prod_model.load_state_dict(torch.load(MODEL_DIR / 'mlp_model.pth',
                                           map_location=device, weights_only=True))
    prod_model.eval()
    with open(MODEL_DIR / 'mlp_scaler.pkl', 'rb') as f:
        prod_scaler = pickle.load(f)

    # Classify the first available tile with both models
    sample_tile = sorted(TILE_CACHE.glob('emb_*.npy'))[0]
    emb = np.load(sample_tile)
    H, W, C = emb.shape
    flat = emb.reshape(-1, C).astype(np.float32)
    flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)

    with torch.no_grad():
        # Habitat model
        hab_scaled = scaler.transform(flat).astype(np.float32)
        hab_probs = torch.sigmoid(
            model(torch.from_numpy(hab_scaled).to(device))
        ).cpu().numpy()

        # Production model
        prod_scaled = prod_scaler.transform(flat).astype(np.float32)
        prod_probs = torch.sigmoid(
            prod_model(torch.from_numpy(prod_scaled).to(device))
        ).cpu().numpy()

    print(f"\n  Tile: {sample_tile.name} ({H}×{W} = {H*W:,} px)")
    for name, p in [('Production', prod_probs), ('Habitat', hab_probs)]:
        print(f"  {name:12s}: mean={p.mean():.4f}  "
              f"P≥0.5={100*(p>=0.5).mean():.2f}%  "
              f"P≥0.95={100*(p>=0.95).mean():.2f}%")

    print(f"\nDone. Use scripts/analysis/compare_habitat_models.py to run full analysis.")


if __name__ == '__main__':
    main()
