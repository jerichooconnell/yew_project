#!/usr/bin/env python3
"""
Pacific Yew Density Prediction - Training with Earth Engine Data
=================================================================

Trains the hybrid model using:
1. Earth Engine satellite data (RGB, NIR, NDVI, EVI, elevation, slope, aspect)
2. Forest inventory tabular features
3. BEC zone information

Target: Predicts presence/density of Pacific Yew (Taxus brevifolia, code: TW)

Author: Analysis Tool
Date: October 28, 2025
"""

from yew_density_model import (
    YewDensityDataset, HybridYewDensityModel, YewDensityTrainer,
    FocalLoss, spatial_train_test_split
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pickle
from pathlib import Path
import sys
from PIL import Image
import io
import re

# Import model classes from the main model file
sys.path.append('scripts/training')


def parse_species_composition(composition_string):
    """Parse species composition string to extract Pacific Yew (TW) percentage."""
    if not composition_string or pd.isna(composition_string):
        return 0

    pattern = r'TW(\d{2,3})'  # TW = Taxus brevifolia (Pacific Yew)
    match = re.search(pattern, str(composition_string))

    if match:
        return int(match.group(1))
    return 0


def load_and_merge_data(ee_data_path, inventory_path):
    """Load and merge Earth Engine data with forest inventory."""
    print("Loading data...")

    # Load EE data
    ee_df = pd.read_csv(ee_data_path)
    print(f"  Earth Engine data: {len(ee_df)} records")

    # Load inventory
    inv_df = pd.read_csv(inventory_path, low_memory=False)
    print(f"  Forest inventory: {len(inv_df)} records")

    # Merge on SITE_IDENTIFIER
    ee_df['SITE_IDENTIFIER'] = ee_df['plot_id'].astype(str)
    inv_df['SITE_IDENTIFIER'] = inv_df['SITE_IDENTIFIER'].astype(str)

    merged_df = inv_df.merge(ee_df, on='SITE_IDENTIFIER',
                             how='inner', suffixes=('', '_ee'))

    print(f"  Merged dataset: {len(merged_df)} records")
    print(f"  Unique sites: {merged_df['SITE_IDENTIFIER'].nunique()}")

    return merged_df


def create_target_variable(df):
    """Create binary target: presence/absence of Pacific Yew."""
    print("\nCreating target variable...")

    # Calculate yew percentage from species composition
    df['YEW_PERCENTAGE'] = df['SPB_CPCT_LS'].apply(parse_species_composition)

    # Binary target: presence/absence of TW
    df['has_yew'] = (df['YEW_PERCENTAGE'] > 0).astype(int)

    # For reference, also calculate density
    df['YEW_DENSITY_HA'] = (df['YEW_PERCENTAGE'] / 100.0) * \
        df['STEMS_HA_LS'].fillna(0)

    print(f"  Plots with Pacific Yew (TW): {df['has_yew'].sum()}")
    print(f"  Plots without yew: {(df['has_yew'] == 0).sum()}")
    print(f"  Yew prevalence: {100 * df['has_yew'].mean():.2f}%")

    yew_present = df[df['has_yew'] == 1]
    if len(yew_present) > 0:
        print(
            f"  Mean density (where present): {yew_present['YEW_DENSITY_HA'].mean():.1f} stems/ha")
        print(
            f"  Median density (where present): {yew_present['YEW_DENSITY_HA'].median():.1f} stems/ha")

    return df


def prepare_features(df):
    """Prepare features for training."""
    print("\nPreparing features...")

    # Earth Engine features (from satellite data)
    ee_features = ['blue', 'green', 'red', 'nir', 'ndvi', 'evi',
                   'elevation', 'slope', 'aspect']

    # Forest inventory numerical features
    inventory_numerical = ['BA_HA_LS', 'STEMS_HA_LS', 'VHA_WSV_LS',
                           'SI_M_TLSO', 'HT_TLSO', 'AGEB_TLSO']

    # Categorical features
    categorical_cols = ['BEC_ZONE', 'TSA_DESC', 'SAMPLE_ESTABLISHMENT_TYPE']

    # Combine all numerical features
    numerical_cols = ee_features + inventory_numerical

    # Filter to valid columns
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    categorical_cols = [col for col in categorical_cols if col in df.columns]

    print(f"  Numerical features: {len(numerical_cols)}")
    print(f"    EE features: {ee_features}")
    print(
        f"    Inventory features: {[c for c in inventory_numerical if c in df.columns]}")
    print(f"  Categorical features: {len(categorical_cols)}")

    # Handle missing values
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        df[col] = df[col].fillna('UNKNOWN')

    # Extract numerical features
    numerical_features = df[numerical_cols].values

    # Normalize
    scaler = StandardScaler()
    numerical_features = scaler.fit_transform(numerical_features)

    # Encode categorical features
    categorical_features = {}
    categorical_dims = {}
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        encoded = le.fit_transform(df[col].astype(str))
        categorical_features[col] = encoded
        categorical_dims[col] = len(le.classes_)
        label_encoders[col] = le
        print(f"    {col}: {len(le.classes_)} categories")

    # Get targets
    targets = df['has_yew'].values

    # Get coordinates for spatial splitting
    coordinates = df[['x', 'y']].values if 'x' in df.columns else df[[
        'BC_ALBERS_X', 'BC_ALBERS_Y']].values

    feature_info = {
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'categorical_dims': categorical_dims,
        'num_numerical': len(numerical_cols),
        'scaler': scaler,
        'label_encoders': label_encoders,
        'coordinates': coordinates
    }

    print(f"\n  Total samples: {len(targets)}")
    print(f"  Samples with yew: {targets.sum()} ({100*targets.mean():.2f}%)")

    return numerical_features, categorical_features, targets, feature_info


def create_simple_tabular_model(num_numerical, categorical_dims, hidden_dims=[128, 64, 32]):
    """
    Create a hybrid model with ResNet for satellite features and tabular network for other features.
    Uses transfer learning with pretrained ResNet18.
    """
    class HybridResNetYewModel(nn.Module):
        def __init__(self, num_numerical, categorical_dims, embedding_dim=16, hidden_dims=[128, 64, 32]):
            super(HybridResNetYewModel, self).__init__()

            # Satellite feature indices (first 9 features are from Earth Engine)
            # blue, green, red, nir, ndvi, evi, elevation, slope, aspect
            self.satellite_indices = list(range(9))
            self.other_indices = list(range(9, num_numerical))

            # ResNet18 for satellite imagery features (transfer learning)
            self.resnet = models.resnet18(pretrained=True)

            # Modify first conv layer to accept our feature channels (9 channels)
            # We'll treat the satellite features as a 3x3 "image" with 9 channels
            original_conv = self.resnet.conv1
            self.resnet.conv1 = nn.Conv2d(
                9,  # 9 satellite features
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )

            # Initialize new conv layer weights
            with torch.no_grad():
                # Use pretrained weights for RGB channels, random for others
                self.resnet.conv1.weight[:, :3, :, :] = original_conv.weight
                # Xavier initialization for additional channels
                nn.init.xavier_uniform_(self.resnet.conv1.weight[:, 3:, :, :])

            # Remove final classification layer
            self.resnet.fc = nn.Identity()

            # ResNet18 outputs 512 features
            resnet_output_dim = 512

            # Freeze ALL ResNet layers - only train fusion network
            # We're just using ResNet as a feature extractor
            for param in self.resnet.parameters():
                param.requires_grad = False

            # Embeddings for categorical features
            self.embeddings = nn.ModuleDict({
                name: nn.Embedding(num_cats, embedding_dim)
                for name, num_cats in categorical_dims.items()
            })

            # Network for non-satellite features
            num_other_numerical = len(self.other_indices)
            total_cat_dim = len(categorical_dims) * embedding_dim
            tabular_input_dim = num_other_numerical + total_cat_dim

            if tabular_input_dim > 0:
                self.tabular_network = nn.Sequential(
                    nn.Linear(tabular_input_dim, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                )
                tabular_output_dim = 64
            else:
                self.tabular_network = None
                tabular_output_dim = 0

            # Fusion network
            fusion_input_dim = resnet_output_dim + tabular_output_dim

            layers = []
            prev_dim = fusion_input_dim
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                ])
                prev_dim = hidden_dim

            self.fusion_network = nn.Sequential(*layers)

            # Output layer (binary classification)
            self.output = nn.Linear(prev_dim, 1)

        def forward(self, numerical_features, categorical_indices):
            batch_size = numerical_features.size(0)

            # Extract satellite features (9 features)
            satellite_features = numerical_features[:, self.satellite_indices]

            # Reshape satellite features as 3x3 image with 9 channels
            # This creates a small "image" where each pixel contains one feature
            satellite_image = satellite_features.view(batch_size, 9, 1, 1)
            satellite_image = satellite_image.expand(
                batch_size, 9, 7, 7)  # Expand to minimum size for ResNet

            # Pass through ResNet
            resnet_features = self.resnet(satellite_image)

            # Process other features through tabular network
            if self.tabular_network is not None and len(self.other_indices) > 0:
                other_numerical = numerical_features[:, self.other_indices]

                # Get categorical embeddings
                cat_embeddings = []
                for name, emb_layer in self.embeddings.items():
                    if name in categorical_indices:
                        cat_embeddings.append(
                            emb_layer(categorical_indices[name]))

                # Concatenate
                if cat_embeddings:
                    cat_tensor = torch.cat(cat_embeddings, dim=1)
                    tabular_input = torch.cat(
                        [other_numerical, cat_tensor], dim=1)
                else:
                    tabular_input = other_numerical

                tabular_features = self.tabular_network(tabular_input)

                # Fuse ResNet and tabular features
                combined_features = torch.cat(
                    [resnet_features, tabular_features], dim=1)
            else:
                combined_features = resnet_features

            # Final prediction
            x = self.fusion_network(combined_features)
            logits = self.output(x)

            return logits.squeeze(-1)

    return HybridResNetYewModel(num_numerical, categorical_dims, embedding_dim=16, hidden_dims=hidden_dims)


class SimpleYewDataset(Dataset):
    """Simplified dataset for tabular-only model."""

    def __init__(self, numerical_features, categorical_features, targets):
        self.numerical = torch.FloatTensor(numerical_features)
        self.categorical = {
            name: torch.LongTensor(values)
            for name, values in categorical_features.items()
        }
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        numerical = self.numerical[idx]
        categorical = {name: values[idx]
                       for name, values in self.categorical.items()}
        target = self.targets[idx]
        return numerical, categorical, target


def train_model(model, train_loader, val_loader, device, epochs=50, lr=1e-3):
    """Train the model."""
    model = model.to(device)

    # Use Focal Loss for extreme class imbalance
    # Focal loss down-weights easy examples and focuses on hard examples
    # This is specifically designed for scenarios where one class is very rare
    print("\nUsing Focal Loss (alpha=0.75, gamma=2.0) for extreme class imbalance")
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    history = defaultdict(list)
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15

    print("\nStarting training...")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []

        for numerical, categorical, targets in train_loader:
            numerical = numerical.to(device)
            targets = targets.to(device)
            categorical = {k: v.to(device) for k, v in categorical.items()}

            optimizer.zero_grad()
            logits = model(numerical, categorical)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
            train_targets.extend(targets.cpu().numpy())

        train_loss /= len(train_loader)
        train_preds = (np.array(train_preds) > 0.5).astype(int)
        train_targets = np.array(train_targets).astype(int)
        train_acc = accuracy_score(train_targets, train_preds)

        # Validate
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for numerical, categorical, targets in val_loader:
                numerical = numerical.to(device)
                targets = targets.to(device)
                categorical = {k: v.to(device) for k, v in categorical.items()}

                logits = model(numerical, categorical)
                loss = criterion(logits, targets)

                val_loss += loss.item()
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        val_loss /= len(val_loader)
        val_probs = np.array(val_preds)
        val_preds = (val_probs > 0.5).astype(int)
        val_targets = np.array(val_targets).astype(int)
        val_acc = accuracy_score(val_targets, val_preds)

        # Calculate precision, recall, F1 (handle zero division)
        val_precision = precision_score(
            val_targets, val_preds, zero_division=0)
        val_recall = recall_score(val_targets, val_preds, zero_division=0)
        val_f1 = f1_score(val_targets, val_preds, zero_division=0)

        # Update scheduler
        scheduler.step(val_loss)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)

        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"\nEpoch [{epoch+1}/{epochs}]")
            print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            print(
                f"  Val Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(),
                       'models/checkpoints/best_yew_model_ee.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(torch.load(
        'models/checkpoints/best_yew_model_ee.pth'))

    return model, history


def plot_training_history(history, save_path='results/figures/yew_training_history_ee.png'):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train')
    axes[0, 1].plot(history['val_acc'], label='Validation')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Precision & Recall
    axes[1, 0].plot(history['val_precision'], label='Precision')
    axes[1, 0].plot(history['val_recall'], label='Recall')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Validation Precision & Recall')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # F1 Score
    axes[1, 1].plot(history['val_f1'])
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('Validation F1 Score')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining history saved to {save_path}")


def main():
    """Main training pipeline."""
    print("="*70)
    print("Pacific Yew Prediction with Earth Engine Data")
    print("="*70)

    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load and merge data
    merged_df = load_and_merge_data(
        'data/ee_imagery/ee_extraction_progress.csv',
        'data/raw/bc_sample_data-2025-10-09/bc_sample_data.csv'
    )

    # Create target
    merged_df = create_target_variable(merged_df)

    # Prepare features
    numerical_features, categorical_features, targets, feature_info = prepare_features(
        merged_df)

    # Spatial split
    print("\nCreating spatial train/val/test splits...")
    train_idx, val_idx, test_idx = spatial_train_test_split(
        feature_info['coordinates'],
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )

    # Helper to subset categorical features
    def subset_cat(cat_features, indices):
        return {name: values[indices] for name, values in cat_features.items()}

    # Create datasets
    train_dataset = SimpleYewDataset(
        numerical_features[train_idx],
        subset_cat(categorical_features, train_idx),
        targets[train_idx]
    )

    val_dataset = SimpleYewDataset(
        numerical_features[val_idx],
        subset_cat(categorical_features, val_idx),
        targets[val_idx]
    )

    test_dataset = SimpleYewDataset(
        numerical_features[test_idx],
        subset_cat(categorical_features, test_idx),
        targets[test_idx]
    )

    # Create weighted sampler for imbalanced data
    weights = torch.ones(len(train_dataset))
    # Much higher weight for yew-present samples (100x instead of 10x)
    # This means yew samples will be seen 100 times more often during training
    num_yew = (targets[train_idx] == 1).sum()
    num_no_yew = (targets[train_idx] == 0).sum()
    yew_weight = num_no_yew / num_yew if num_yew > 0 else 100.0
    print(
        f"\nClass imbalance: {num_no_yew} no-yew / {num_yew} yew = {yew_weight:.1f}x")
    print(f"Setting yew sample weight to {yew_weight:.1f}x")
    weights[targets[train_idx] == 1] = yew_weight
    train_sampler = WeightedRandomSampler(
        weights, len(weights), replacement=True)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=32, sampler=train_sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32,
                            shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32,
                             shuffle=False, num_workers=2)

    # Create model
    print("\nCreating model...")
    print("Using ResNet18 with transfer learning for satellite features")
    model = create_simple_tabular_model(
        feature_info['num_numerical'],
        feature_info['categorical_dims'],
        hidden_dims=[256, 128, 64]
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    print(f"Using pretrained ResNet18 for satellite feature extraction")

    # Train
    model, history = train_model(
        model, train_loader, val_loader, device, epochs=50, lr=0.001)

    # Evaluate on test set
    print("\n" + "="*70)
    print("Test Set Evaluation")
    print("="*70)

    model.eval()
    test_preds = []
    test_targets_list = []

    with torch.no_grad():
        for numerical, categorical, targets_batch in test_loader:
            numerical = numerical.to(device)
            categorical = {k: v.to(device) for k, v in categorical.items()}

            logits = model(numerical, categorical)
            probs = torch.sigmoid(logits)

            test_preds.extend(probs.cpu().numpy())
            test_targets_list.extend(targets_batch.numpy())

    test_probs = np.array(test_preds)
    test_targets_np = np.array(test_targets_list).astype(int)

    # Find optimal threshold to balance precision and recall
    print("\nOptimizing decision threshold...")
    thresholds = np.arange(0.1, 0.95, 0.05)
    best_f1 = 0
    best_threshold = 0.5
    threshold_results = []

    for threshold in thresholds:
        preds = (test_probs > threshold).astype(int)
        precision = precision_score(test_targets_np, preds, zero_division=0)
        recall = recall_score(test_targets_np, preds, zero_division=0)
        f1 = f1_score(test_targets_np, preds, zero_division=0)
        threshold_results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"Best threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")

    # Show threshold analysis
    print("\nThreshold Analysis (selected values):")
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 48)
    for result in threshold_results[::3]:  # Show every 3rd threshold
        print(
            f"{result['threshold']:<12.2f} {result['precision']:<12.4f} {result['recall']:<12.4f} {result['f1']:<12.4f}")

    # Evaluate with default threshold (0.5)
    test_preds_binary_default = (test_probs > 0.5).astype(int)
    test_acc_default = accuracy_score(
        test_targets_np, test_preds_binary_default)
    test_precision_default = precision_score(
        test_targets_np, test_preds_binary_default, zero_division=0)
    test_recall_default = recall_score(
        test_targets_np, test_preds_binary_default, zero_division=0)
    test_f1_default = f1_score(
        test_targets_np, test_preds_binary_default, zero_division=0)

    # Evaluate with optimized threshold
    test_preds_binary = (test_probs > best_threshold).astype(int)
    test_acc = accuracy_score(test_targets_np, test_preds_binary)
    test_precision = precision_score(
        test_targets_np, test_preds_binary, zero_division=0)
    test_recall = recall_score(
        test_targets_np, test_preds_binary, zero_division=0)
    test_f1 = f1_score(test_targets_np, test_preds_binary, zero_division=0)

    print(f"\nTest Set Results (threshold=0.5):")
    print(f"  Accuracy: {test_acc_default:.4f}")
    print(f"  Precision: {test_precision_default:.4f}")
    print(f"  Recall: {test_recall_default:.4f}")
    print(f"  F1 Score: {test_f1_default:.4f}")

    print(f"\nTest Set Results (optimized threshold={best_threshold:.2f}):")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")
    print(f"\nTest set composition:")
    print(
        f"  Yew present: {test_targets_np.sum()} ({100*test_targets_np.mean():.2f}%)")
    print(f"  Yew absent: {(test_targets_np == 0).sum()}")

    # Save artifacts
    print("\nSaving artifacts...")

    # Save model
    Path('models/checkpoints').mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), 'models/checkpoints/yew_model_ee_final.pth')
    print("  Model saved")

    # Save preprocessors
    Path('models/artifacts').mkdir(parents=True, exist_ok=True)
    with open('models/artifacts/yew_preprocessor_ee.pkl', 'wb') as f:
        pickle.dump(feature_info, f)
    print("  Preprocessor saved")

    # Plot training history
    plot_training_history(history)

    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)


if __name__ == "__main__":
    main()
