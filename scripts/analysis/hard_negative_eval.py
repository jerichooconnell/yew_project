#!/usr/bin/env python3
"""
P0.1 — Hard-negative classifier evaluation.

Evaluates the production XGBoost yew-habitat classifier against two negative sets:
  - "Easy" negatives: sub-boreal, dry-interior, boreal zones (SBS, IDF, BWBS, SBPS, PP)
    spectrally distinct from old-growth coastal/interior wet forest.
  - "Hard" negatives: same BEC zones as yew (CWH, ICH, CDF) — old-growth-capable
    stands spectrally similar to yew habitat that simply lack recorded yew.

A high AUC on easy negatives inflates apparent accuracy if the model is learning
"coastal moist forest vs. everything else" rather than "yew habitat vs. similar-looking
non-yew habitat."

Run:
    /home/jericho/anaconda3/envs/yew_pytorch/bin/python scripts/analysis/hard_negative_eval.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
import xgboost as xgb

ROOT  = Path("/home/jericho/yew_project")
MODEL = ROOT / "results/predictions/south_vi_large/xgb_raw_model_expanded.json"

# ── Negative BEC zone classification ────────────────────────────────────────
EASY_ZONES = {"SBS", "IDF", "BWBS", "SBPS", "PP", "MS", "ESSF"}  # non-coastal / dry interior
HARD_ZONES = {"CWH", "ICH", "CDF", "MH", "CMA"}                  # coastal / wet interior (yew zones)


def load_data():
    model = xgb.XGBClassifier()
    model.load_model(MODEL)

    # Positives: 6,171 iNaturalist T. brevifolia embeddings
    X_pos = np.load(ROOT / "data/processed/all_positive_embeddings.npy")
    pos_meta = pd.read_csv(ROOT / "data/processed/all_positive_metadata.csv")
    # Keep only BC records for the BC-focused model
    bc_mask = pos_meta["state"].str.upper().isin({"BC", "BRITISH COLUMBIA"}) | pos_meta["state"].isna()
    # Positives include all locations; model was trained on all iNat positives, so this is
    # in-distribution. We flag this and use it only for the hard-vs-easy comparison, not
    # as a standalone accuracy claim.
    X_pos = X_pos[:len(pos_meta)]  # align lengths

    # Negatives: combined_negatives.csv has embeddings + BEC zone
    neg = pd.read_csv(ROOT / "data/processed/combined_negatives.csv", low_memory=False)
    emb_cols = [f"emb_{i}" for i in range(64)]
    X_neg = neg[emb_cols].values

    bec = neg["bec_zone"].fillna("UNKNOWN")
    easy_mask = bec.isin(EASY_ZONES)
    hard_mask = bec.isin(HARD_ZONES)

    return model, X_pos, X_neg, easy_mask, hard_mask, bec


def eval_set(model, X_pos, X_neg_sub, label):
    X = np.vstack([X_pos, X_neg_sub])
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg_sub))])
    probs = model.predict_proba(X)[:, 1]
    # Operational threshold: 0.5 (default XGBoost)
    preds = (probs >= 0.5).astype(int)
    auc   = roc_auc_score(y, probs)
    ap    = average_precision_score(y, probs)
    f1    = f1_score(y, preds, zero_division=0)
    prec  = precision_score(y, preds, zero_division=0)
    rec   = recall_score(y, preds, zero_division=0)
    tn = ((preds == 0) & (y == 0)).sum()
    fp = ((preds == 1) & (y == 0)).sum()
    fn = ((preds == 0) & (y == 1)).sum()
    tp = ((preds == 1) & (y == 1)).sum()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    print(f"\n── {label} (n_pos={len(X_pos)}, n_neg={len(X_neg_sub)}) ──")
    print(f"  AUC-ROC : {auc:.4f}")
    print(f"  Avg Prec: {ap:.4f}")
    print(f"  F1@0.5  : {f1:.4f}   Prec={prec:.4f}   Recall={rec:.4f}   Specificity={spec:.4f}")
    print(f"  Conf mat: TP={tp} FP={fp} TN={tn} FN={fn}")
    return {"label": label, "n_pos": int(len(X_pos)), "n_neg": int(len(X_neg_sub)),
            "auc": round(auc, 4), "avg_precision": round(ap, 4),
            "f1": round(f1, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "specificity": round(spec, 4),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def main():
    print("Loading model and data...")
    model, X_pos, X_neg, easy_mask, hard_mask, bec = load_data()
    print(f"  Positives: {len(X_pos)}")
    print(f"  All negatives: {len(X_neg)}")
    print(f"  Easy negatives (non-coastal): {easy_mask.sum()}")
    print(f"  Hard negatives (CWH/ICH/CDF): {hard_mask.sum()}")
    print()
    print("BEC zone breakdown of negatives:")
    print(bec.value_counts().to_string())

    results = []
    results.append(eval_set(model, X_pos, X_neg,                 "All negatives (mixed BEC)"))
    results.append(eval_set(model, X_pos, X_neg[easy_mask],      "Easy negatives (non-coastal BEC)"))
    results.append(eval_set(model, X_pos, X_neg[hard_mask],      "Hard negatives (CWH / ICH / CDF)"))
    results.append(eval_set(model, X_pos, X_neg[~easy_mask & ~hard_mask], "Other zones"))

    print("\n── Summary table ──")
    print(f"{'Negative set':<40} {'AUC':>6} {'AvgP':>6} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Spec':>6}")
    for r in results:
        print(f"{r['label']:<40} {r['auc']:>6.4f} {r['avg_precision']:>6.4f} {r['f1']:>6.4f} {r['precision']:>6.4f} {r['recall']:>6.4f} {r['specificity']:>6.4f}")

    out_path = ROOT / "results/analysis/hard_negative_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")

    print("""
NOTE on interpretation
──────────────────────
These metrics are computed on the FULL combined_negatives.csv, which overlaps with
the training set (the training negative split was drawn from this pool). This means
they are in-sample and will overestimate real-world performance.

Their value here is COMPARATIVE, not absolute:
  - If AUC(Hard) << AUC(Easy), the model is partially learning zone identity (easy vs.
    coastal), not just yew-suitability within a zone.
  - If AUC(Hard) ≈ AUC(Easy), the model discriminates well even within the same BEC zones.
A proper held-out evaluation would require new embeddings from GEE, which is out of scope;
this in-sample comparison provides the directional signal requested by the plan.
""")


if __name__ == "__main__":
    main()
