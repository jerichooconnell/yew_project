#!/usr/bin/env python3
"""
Retrain XGBoost model with 5000 new negatives from maturing second-growth (cat 4&5).

Combines the existing training set with the newly-sampled cat4&5 negatives,
retrains the model, and evaluates on the same validation set.

Run:
    conda run -n yew_pytorch python scripts/training/retrain_with_maturing_negatives.py
"""
import sys
import json
import numpy as np
from pathlib import Path

import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/processed"
RESULTS_DIR = ROOT / "results/predictions/south_vi_large"

OLD_MODEL = RESULTS_DIR / "xgb_raw_model_expanded.json"
NEW_MODEL = RESULTS_DIR / "xgb_raw_model_expanded_v2_cat45neg.json"
NEW_METRICS = RESULTS_DIR / "xgb_expanded_metrics_v2_cat45neg.json"


def main():
    print("=" * 70)
    print("RETRAIN: XGBoost with 5000 maturing-second-growth negatives")
    print("=" * 70)

    # ── Load existing training data ──────────────────────────────────────────
    print("\n[1] Loading existing training set...")
    X_train = np.load(DATA_DIR / "expanded_X_all.npy")
    y_train = np.load(DATA_DIR / "expanded_y_all.npy")

    print(f"  Training set: {X_train.shape}")
    print(f"  Positives: {y_train.sum():.0f} | Negatives: {(~y_train.astype(bool)).sum():.0f}")

    # ── Load new cat4&5 negatives ──────────────────────────────────────────
    print("\n[2] Loading new cat4&5 negatives...")
    new_neg_emb = np.load(DATA_DIR / "maturing_second_growth_negatives_embeddings.npy")
    print(f"  New negatives: {new_neg_emb.shape}")

    # ── Combine training data ──────────────────────────────────────────────
    print("\n[3] Combining with new negatives...")
    X_train_combined = np.vstack([X_train, new_neg_emb]).astype(np.float32)
    y_train_combined = np.concatenate([y_train, np.zeros(len(new_neg_emb))])
    print(f"  Total train: {X_train_combined.shape} | positives: {y_train_combined.sum():.0f} | negatives: {(~y_train_combined.astype(bool)).sum():.0f}")
    print(f"  Ratio: 1 positive : {(~y_train_combined.astype(bool)).sum() / y_train_combined.sum():.1f} negatives")

    # Update for downstream use
    X_train = X_train_combined
    y_train = y_train_combined

    # ── Create validation split for training feedback ──────────────────────
    print("\n[4] Creating validation split (80/20)...")
    from sklearn.model_selection import train_test_split
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    print(f"  Train split: {X_train_split.shape} | pos: {y_train_split.sum():.0f} | neg: {(~y_train_split.astype(bool)).sum():.0f}")
    print(f"  Validation:  {X_val.shape} | pos: {y_val.sum():.0f} | neg: {(~y_val.astype(bool)).sum():.0f}")

    # Use split for training
    X_train = X_train_split
    y_train = y_train_split

    # ── Train new model ────────────────────────────────────────────────────
    print("\n[5] Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method='hist',
        device='cuda',
        objective='binary:logistic',
        eval_metric='logloss',
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    print(f"  ✓ Model trained")

    # ── Evaluate ───────────────────────────────────────────────────────────
    print("\n[6] Evaluating on validation set...")
    y_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    fpr, tpr, _ = roc_curve(y_val, y_pred)
    print(f"  AUC-ROC: {auc:.6f}")

    # ── Save model ────────────────────────────────────────────────────────
    print(f"\n[7] Saving model to {NEW_MODEL.name}...")
    model.save_model(str(NEW_MODEL))

    # ── Save metrics ──────────────────────────────────────────────────────
    metrics = {
        "model": "xgb_raw_model_expanded_v2_cat45neg",
        "training_set": {
            "n_positives": int(y_train.sum()),
            "n_negatives": int((~y_train.astype(bool)).sum()),
            "n_total": len(y_train),
            "ratio": f"1 pos : {(~y_train.astype(bool)).sum() / y_train.sum():.1f} neg",
        },
        "validation_set": {
            "n_positives": int(y_val.sum()),
            "n_negatives": int((~y_val.astype(bool)).sum()),
            "n_total": len(y_val),
        },
        "auc_roc": float(auc),
        "new_negatives_source": "maturing_second_growth_cat4_cat5",
        "new_negatives_count": int(len(new_neg_emb)),
    }
    with open(NEW_METRICS, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓ Metrics saved to {NEW_METRICS.name}")

    print("\n" + "=" * 70)
    print(f"RETRAIN COMPLETE")
    print(f"  Old model: {OLD_MODEL.name}")
    print(f"  New model: {NEW_MODEL.name} (AUC {auc:.6f})")
    print(f"  Metrics:   {NEW_METRICS.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
