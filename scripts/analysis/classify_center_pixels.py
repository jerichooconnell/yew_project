#!/usr/bin/env python3
"""
Classify Yew Using Center Pixel(s) from 64-Channel Embeddings

This script attempts to separate yew from non-yew samples using only:
1. The central 1 pixel (64 features)
2. The central 9 pixels (3x3 = 576 features)

from the 64-channel embeddings.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import argparse

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report
)

plt.rcParams['figure.dpi'] = 100
sns.set_style('whitegrid')


def extract_center_pixel(lat, lon, emb_dir, patch_size=64):
    """
    Extract single center pixel from embedding.
    Returns shape: (64,) for 64 channels
    """
    emb_path = emb_dir / f'embedding_{lat:.6f}_{lon:.6f}.npy'
    
    if not emb_path.exists():
        return None
    
    img = np.load(emb_path)  # Shape: (64, 64, 64)
    center = patch_size // 2
    
    # Extract single center pixel
    center_pixel = img[:, center, center]
    return center_pixel


def extract_center_9_pixels(lat, lon, emb_dir, patch_size=64):
    """
    Extract 3x3 center pixels from embedding.
    Returns shape: (64, 3, 3) for 64 channels x 3x3 pixels, flattened to (576,)
    """
    emb_path = emb_dir / f'embedding_{lat:.6f}_{lon:.6f}.npy'
    
    if not emb_path.exists():
        return None
    
    img = np.load(emb_path)  # Shape: (64, 64, 64)
    center = patch_size // 2
    
    # Extract 3x3 center pixels
    center_pixels = img[:, center-1:center+2, center-1:center+2]
    return center_pixels.flatten()  # 64 * 9 = 576 features


def extract_features_from_split(df, emb_dir, use_9_pixels=False):
    """
    Extract features and labels from a data split.
    """
    features = []
    labels = []
    
    extract_func = extract_center_9_pixels if use_9_pixels else extract_center_pixel
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Extracting features'):
        if pd.notna(row['lat']) and pd.notna(row['lon']):
            try:
                center_data = extract_func(row['lat'], row['lon'], emb_dir)
                
                if center_data is not None:
                    features.append(center_data)
                    labels.append(int(row['has_yew']))
            except:
                pass
    
    return np.array(features), np.array(labels)


def train_classifiers(X_train, y_train, X_val, y_val, X_test, y_test, experiment_name):
    """
    Train multiple classifiers and return results.
    """
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42)
    }
    
    results = {}
    
    print(f'\nTraining classifiers for {experiment_name}...')
    print('='*80)
    
    for name, clf in classifiers.items():
        print(f'\nTraining {name}...')
        clf.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = clf.predict(X_train)
        y_pred_val = clf.predict(X_val)
        y_pred_test = clf.predict(X_test)
        
        # Probabilities for ROC AUC
        if hasattr(clf, 'predict_proba'):
            y_prob_val = clf.predict_proba(X_val)[:, 1]
            y_prob_test = clf.predict_proba(X_test)[:, 1]
        else:
            y_prob_val = clf.decision_function(X_val)
            y_prob_test = clf.decision_function(X_test)
        
        # Calculate metrics
        results[name] = {
            'train_acc': accuracy_score(y_train, y_pred_train),
            'val_acc': accuracy_score(y_val, y_pred_val),
            'test_acc': accuracy_score(y_test, y_pred_test),
            'val_precision': precision_score(y_val, y_pred_val, zero_division=0),
            'val_recall': recall_score(y_val, y_pred_val, zero_division=0),
            'val_f1': f1_score(y_val, y_pred_val, zero_division=0),
            'val_auc': roc_auc_score(y_val, y_prob_val),
            'test_precision': precision_score(y_test, y_pred_test, zero_division=0),
            'test_recall': recall_score(y_test, y_pred_test, zero_division=0),
            'test_f1': f1_score(y_test, y_pred_test, zero_division=0),
            'test_auc': roc_auc_score(y_test, y_prob_test),
            'y_prob_test': y_prob_test,
            'y_pred_test': y_pred_test,
            'model': clf
        }
        
        print(f'  Val Accuracy: {results[name]["val_acc"]:.4f}')
        print(f'  Test Accuracy: {results[name]["test_acc"]:.4f}')
        print(f'  Test F1: {results[name]["test_f1"]:.4f}')
        print(f'  Test AUC: {results[name]["test_auc"]:.4f}')
    
    return results


def plot_comparison(results_1px, results_9px, y_test_1px, y_test_9px, output_dir):
    """
    Create comparison visualizations.
    """
    # Comparison table
    comparison_df = pd.DataFrame({
        'Model': list(results_1px.keys()),
        '1 Pixel - Test Acc': [r['test_acc'] for r in results_1px.values()],
        '1 Pixel - Test F1': [r['test_f1'] for r in results_1px.values()],
        '1 Pixel - Test AUC': [r['test_auc'] for r in results_1px.values()],
        '9 Pixels - Test Acc': [r['test_acc'] for r in results_9px.values()],
        '9 Pixels - Test F1': [r['test_f1'] for r in results_9px.values()],
        '9 Pixels - Test AUC': [r['test_auc'] for r in results_9px.values()]
    })
    
    # Visualize comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = ['Test Acc', 'Test F1', 'Test AUC']
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        x = np.arange(len(comparison_df['Model']))
        width = 0.35
        
        ax.bar(x - width/2, comparison_df[f'1 Pixel - {metric}'], width, label='1 Pixel (64 feat)', alpha=0.8)
        ax.bar(x + width/2, comparison_df[f'9 Pixels - {metric}'], width, label='9 Pixels (576 feat)', alpha=0.8)
        
        ax.set_xlabel('Model', fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.0])
    
    plt.suptitle('Classification Performance: Center Pixels Only', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_metrics.png', dpi=150, bbox_inches='tight')
    print(f'Saved: {output_dir / "comparison_metrics.png"}')
    plt.close()
    
    # ROC curves
    best_model_1px = max(results_1px.items(), key=lambda x: x[1]['test_auc'])[0]
    best_model_9px = max(results_9px.items(), key=lambda x: x[1]['test_auc'])[0]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1 pixel ROC
    ax = axes[0]
    fpr, tpr, _ = roc_curve(y_test_1px, results_1px[best_model_1px]['y_prob_test'])
    auc = results_1px[best_model_1px]['test_auc']
    ax.plot(fpr, tpr, linewidth=2, label=f'{best_model_1px} (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5)')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve - Single Center Pixel', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 9 pixels ROC
    ax = axes[1]
    fpr, tpr, _ = roc_curve(y_test_9px, results_9px[best_model_9px]['y_prob_test'])
    auc = results_9px[best_model_9px]['test_auc']
    ax.plot(fpr, tpr, linewidth=2, label=f'{best_model_9px} (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5)')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve - Center 3x3 Pixels', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves.png', dpi=150, bbox_inches='tight')
    print(f'Saved: {output_dir / "roc_curves.png"}')
    plt.close()
    
    # Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1 pixel confusion matrix
    ax = axes[0]
    cm = confusion_matrix(y_test_1px, results_1px[best_model_1px]['y_pred_test'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_title(f'Confusion Matrix - 1 Pixel\n{best_model_1px}', fontsize=12, fontweight='bold')
    ax.set_xticklabels(['Non-Yew', 'Yew'])
    ax.set_yticklabels(['Non-Yew', 'Yew'])
    
    # 9 pixels confusion matrix
    ax = axes[1]
    cm = confusion_matrix(y_test_9px, results_9px[best_model_9px]['y_pred_test'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_title(f'Confusion Matrix - 9 Pixels\n{best_model_9px}', fontsize=12, fontweight='bold')
    ax.set_xticklabels(['Non-Yew', 'Yew'])
    ax.set_yticklabels(['Non-Yew', 'Yew'])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
    print(f'Saved: {output_dir / "confusion_matrices.png"}')
    plt.close()
    
    return comparison_df


def print_summary(results_1px, results_9px, y_test_1px):
    """
    Print summary statistics.
    """
    print('\n' + '='*80)
    print('SUMMARY: Yew Classification Using Center Pixel(s) from 64-Channel Embeddings')
    print('='*80)
    print()
    
    print('BEST PERFORMANCE (1 Center Pixel - 64 features):')
    print('-' * 40)
    best_1px_name = max(results_1px.items(), key=lambda x: x[1]['test_f1'])[0]
    best_1px = results_1px[best_1px_name]
    print(f'  Model: {best_1px_name}')
    print(f'  Test Accuracy: {best_1px["test_acc"]:.4f}')
    print(f'  Test Precision: {best_1px["test_precision"]:.4f}')
    print(f'  Test Recall: {best_1px["test_recall"]:.4f}')
    print(f'  Test F1: {best_1px["test_f1"]:.4f}')
    print(f'  Test AUC: {best_1px["test_auc"]:.4f}')
    print()
    
    print('BEST PERFORMANCE (Center 3x3 Pixels - 576 features):')
    print('-' * 40)
    best_9px_name = max(results_9px.items(), key=lambda x: x[1]['test_f1'])[0]
    best_9px = results_9px[best_9px_name]
    print(f'  Model: {best_9px_name}')
    print(f'  Test Accuracy: {best_9px["test_acc"]:.4f}')
    print(f'  Test Precision: {best_9px["test_precision"]:.4f}')
    print(f'  Test Recall: {best_9px["test_recall"]:.4f}')
    print(f'  Test F1: {best_9px["test_f1"]:.4f}')
    print(f'  Test AUC: {best_9px["test_auc"]:.4f}')
    print()
    
    print('KEY FINDINGS:')
    print('-' * 40)
    
    if best_9px['test_f1'] > best_1px['test_f1']:
        improvement = (best_9px['test_f1'] - best_1px['test_f1']) / best_1px['test_f1'] * 100
        print(f'  ✓ 9 pixels outperforms 1 pixel by {improvement:.1f}% in F1 score')
    else:
        improvement = (best_1px['test_f1'] - best_9px['test_f1']) / best_9px['test_f1'] * 100
        print(f'  ✓ 1 pixel outperforms 9 pixels by {improvement:.1f}% in F1 score')
    
    # Calculate average improvement across all models
    avg_f1_1px = np.mean([r['test_f1'] for r in results_1px.values()])
    avg_f1_9px = np.mean([r['test_f1'] for r in results_9px.values()])
    print(f'  • Average F1 (1 pixel): {avg_f1_1px:.4f}')
    print(f'  • Average F1 (9 pixels): {avg_f1_9px:.4f}')
    
    # Baseline comparison
    majority_baseline = max(y_test_1px.sum(), len(y_test_1px) - y_test_1px.sum()) / len(y_test_1px)
    print(f'  • Majority class baseline: {majority_baseline:.4f}')
    print(f'  • Best model improvement over baseline: {(best_9px["test_acc"] - majority_baseline):.4f}')
    
    print()
    print('='*80)


def main(args):
    """
    Main function to run the classification experiment.
    """
    # Load train/val splits (no test split exists)
    print('Loading data splits...')
    train_df = pd.read_csv(args.train_path)
    val_df = pd.read_csv(args.val_path)
    
    # Use validation set as test set for final evaluation
    test_df = val_df.copy()
    
    print(f'Train samples: {len(train_df)} (Yew: {train_df["has_yew"].sum()}, Non-yew: {len(train_df) - train_df["has_yew"].sum()})')
    print(f'Val/Test samples: {len(val_df)} (Yew: {val_df["has_yew"].sum()}, Non-yew: {len(val_df) - val_df["has_yew"].sum()})')
    print(f'\nNote: Using validation set as test set since no separate test split exists.')
    
    embedding_dir = Path(args.embedding_dir)
    
    # ============================================================
    # Experiment 1: Single Center Pixel (64 features)
    # ============================================================
    print('\n' + '='*80)
    print('EXPERIMENT 1: Single Center Pixel (64 features)')
    print('='*80)
    
    print('\nExtracting single center pixel features...')
    X_train_1px, y_train_1px = extract_features_from_split(train_df, embedding_dir, use_9_pixels=False)
    X_val_1px, y_val_1px = extract_features_from_split(val_df, embedding_dir, use_9_pixels=False)
    X_test_1px, y_test_1px = extract_features_from_split(test_df, embedding_dir, use_9_pixels=False)
    
    print(f'\nSingle Pixel Features Extracted:')
    print(f'  Train: {X_train_1px.shape}, Yew: {y_train_1px.sum()}/{len(y_train_1px)}')
    print(f'  Val: {X_val_1px.shape}, Yew: {y_val_1px.sum()}/{len(y_val_1px)}')
    print(f'  Test: {X_test_1px.shape}, Yew: {y_test_1px.sum()}/{len(y_test_1px)}')
    
    # Check for problematic values
    print('\nData quality check (1 pixel):')
    print(f'  NaN count: {np.isnan(X_train_1px).sum()}')
    print(f'  Inf count: {np.isinf(X_train_1px).sum()}')
    print(f'  Range: [{np.nanmin(X_train_1px):.4f}, {np.nanmax(X_train_1px):.4f}]')
    
    # Replace inf with nan, then handle nans
    X_train_1px = np.nan_to_num(X_train_1px, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_1px = np.nan_to_num(X_val_1px, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_1px = np.nan_to_num(X_test_1px, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Standardize features
    scaler_1px = StandardScaler()
    X_train_1px_scaled = scaler_1px.fit_transform(X_train_1px)
    X_val_1px_scaled = scaler_1px.transform(X_val_1px)
    X_test_1px_scaled = scaler_1px.transform(X_test_1px)
    print('Features standardized.')
    
    # Train classifiers
    results_1px = train_classifiers(
        X_train_1px_scaled, y_train_1px,
        X_val_1px_scaled, y_val_1px,
        X_test_1px_scaled, y_test_1px,
        'Single Center Pixel'
    )
    
    # Results summary table
    results_1px_df = pd.DataFrame({
        'Model': list(results_1px.keys()),
        'Train Acc': [r['train_acc'] for r in results_1px.values()],
        'Val Acc': [r['val_acc'] for r in results_1px.values()],
        'Test Acc': [r['test_acc'] for r in results_1px.values()],
        'Test Precision': [r['test_precision'] for r in results_1px.values()],
        'Test Recall': [r['test_recall'] for r in results_1px.values()],
        'Test F1': [r['test_f1'] for r in results_1px.values()],
        'Test AUC': [r['test_auc'] for r in results_1px.values()]
    })
    
    print('\n' + '='*80)
    print('RESULTS: Single Center Pixel (64 features)')
    print('='*80)
    print(results_1px_df.to_string(index=False))
    
    # ============================================================
    # Experiment 2: Center 9 Pixels (3x3 = 576 features)
    # ============================================================
    print('\n' + '='*80)
    print('EXPERIMENT 2: Center 9 Pixels (3x3 = 576 features)')
    print('='*80)
    
    print('\nExtracting 3x3 center pixel features...')
    X_train_9px, y_train_9px = extract_features_from_split(train_df, embedding_dir, use_9_pixels=True)
    X_val_9px, y_val_9px = extract_features_from_split(val_df, embedding_dir, use_9_pixels=True)
    X_test_9px, y_test_9px = extract_features_from_split(test_df, embedding_dir, use_9_pixels=True)
    
    print(f'\n3x3 Pixel Features Extracted:')
    print(f'  Train: {X_train_9px.shape}, Yew: {y_train_9px.sum()}/{len(y_train_9px)}')
    print(f'  Val: {X_val_9px.shape}, Yew: {y_val_9px.sum()}/{len(y_val_9px)}')
    print(f'  Test: {X_test_9px.shape}, Yew: {y_test_9px.sum()}/{len(y_test_9px)}')
    
    # Check for problematic values
    print('\nData quality check (9 pixels):')
    print(f'  NaN count: {np.isnan(X_train_9px).sum()}')
    print(f'  Inf count: {np.isinf(X_train_9px).sum()}')
    print(f'  Range: [{np.nanmin(X_train_9px):.4f}, {np.nanmax(X_train_9px):.4f}]')
    
    # Replace inf with nan, then handle nans
    X_train_9px = np.nan_to_num(X_train_9px, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_9px = np.nan_to_num(X_val_9px, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_9px = np.nan_to_num(X_test_9px, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Standardize features
    scaler_9px = StandardScaler()
    X_train_9px_scaled = scaler_9px.fit_transform(X_train_9px)
    X_val_9px_scaled = scaler_9px.transform(X_val_9px)
    X_test_9px_scaled = scaler_9px.transform(X_test_9px)
    print('Features standardized.')
    
    # Train classifiers
    results_9px = train_classifiers(
        X_train_9px_scaled, y_train_9px,
        X_val_9px_scaled, y_val_9px,
        X_test_9px_scaled, y_test_9px,
        'Center 3x3 Pixels'
    )
    
    # Results summary table
    results_9px_df = pd.DataFrame({
        'Model': list(results_9px.keys()),
        'Train Acc': [r['train_acc'] for r in results_9px.values()],
        'Val Acc': [r['val_acc'] for r in results_9px.values()],
        'Test Acc': [r['test_acc'] for r in results_9px.values()],
        'Test Precision': [r['test_precision'] for r in results_9px.values()],
        'Test Recall': [r['test_recall'] for r in results_9px.values()],
        'Test F1': [r['test_f1'] for r in results_9px.values()],
        'Test AUC': [r['test_auc'] for r in results_9px.values()]
    })
    
    print('\n' + '='*80)
    print('RESULTS: Center 3x3 Pixels (576 features)')
    print('='*80)
    print(results_9px_df.to_string(index=False))
    
    # ============================================================
    # Comparison & Visualization
    # ============================================================
    print('\n' + '='*80)
    print('COMPARISON & VISUALIZATION')
    print('='*80)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_df = plot_comparison(results_1px, results_9px, y_test_1px, y_test_9px, output_dir)
    
    print('\n' + '='*80)
    print('COMPARISON: 1 Pixel vs 9 Pixels')
    print('='*80)
    print(comparison_df.to_string(index=False))
    
    # Print summary
    print_summary(results_1px, results_9px, y_test_1px)
    
    # Save results
    results_1px_df.to_csv(output_dir / 'classification_1_pixel_results.csv', index=False)
    results_9px_df.to_csv(output_dir / 'classification_9_pixels_results.csv', index=False)
    comparison_df.to_csv(output_dir / 'classification_comparison.csv', index=False)
    
    print('\nResults saved to results/analysis/')
    print(f'  - {output_dir / "classification_1_pixel_results.csv"}')
    print(f'  - {output_dir / "classification_9_pixels_results.csv"}')
    print(f'  - {output_dir / "classification_comparison.csv"}')
    print(f'  - {output_dir / "comparison_metrics.png"}')
    print(f'  - {output_dir / "roc_curves.png"}')
    print(f'  - {output_dir / "confusion_matrices.png"}')
    
    print('\n' + '='*80)
    print('CLASSIFICATION EXPERIMENT COMPLETE')
    print('='*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Classify yew using center pixels from 64-channel embeddings'
    )
    parser.add_argument(
        '--train-path',
        type=str,
        default='data/processed/train_split_filtered.csv',
        help='Path to training data CSV'
    )
    parser.add_argument(
        '--val-path',
        type=str,
        default='data/processed/val_split_filtered.csv',
        help='Path to validation data CSV'
    )
    parser.add_argument(
        '--embedding-dir',
        type=str,
        default='data/ee_imagery/embedding_patches_64x64',
        help='Directory containing embedding patches'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/analysis',
        help='Output directory for results and plots'
    )
    
    args = parser.parse_args()
    main(args)
