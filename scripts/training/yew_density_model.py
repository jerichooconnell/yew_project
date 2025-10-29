#!/usr/bin/env python3
"""
Pacific Yew Density Prediction Model - PyTorch Implementation
=============================================================

Multi-modal deep learning model that combines:
1. Satellite imagery (Earth Engine data)
2. Tabular features (BEC zone, forest metrics, etc.)

Predicts Pacific Yew density (stems/hectare) at forest sites.

Author: Analysis Tool
Date: October 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.optim as optim
from torchvision import models, transforms

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pickle
import os
from typing import Tuple, Dict, List, Optional


class YewDensityDataset(Dataset):
    """
    PyTorch Dataset for Pacific Yew density prediction combining imagery and tabular data.
    """

    def __init__(self,
                 imagery_paths: List[str],
                 numerical_features: np.ndarray,
                 categorical_features: Dict[str, np.ndarray],
                 targets: np.ndarray,
                 transform=None):
        """
        Args:
            imagery_paths: List of paths to satellite images (or preloaded arrays)
            numerical_features: Numerical features only
            categorical_features: Dict of categorical feature arrays
            targets: Yew density values (stems/ha)
            transform: Optional image transformations
        """
        self.imagery_paths = imagery_paths
        self.numerical_features = torch.FloatTensor(numerical_features)
        self.categorical_features = {
            name: torch.LongTensor(values) for name, values in categorical_features.items()
        }
        self.targets = torch.FloatTensor(targets)
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # Load or get image (placeholder for now - will be populated by Earth Engine script)
        # For now, we'll generate synthetic imagery based on features
        # In production, this would load actual satellite imagery
        image = self._generate_placeholder_image(idx)

        if self.transform:
            image = self.transform(image)

        numerical = self.numerical_features[idx]
        categorical = {name: values[idx] for name,
                       values in self.categorical_features.items()}
        target = self.targets[idx]

        return image, numerical, categorical, target

    def _generate_placeholder_image(self, idx):
        """
        Generate placeholder image based on tabular features.
        In production, replace this with actual satellite imagery loading.
        """
        # Create a simple 3-channel placeholder (64x64 RGB)
        # This simulates vegetation patterns based on features
        np.random.seed(idx)  # Deterministic for reproducibility

        # Use some features to influence the image
        feature_vec = self.numerical_features[idx].numpy()

        # Create synthetic image with patterns
        base_color = feature_vec[:3] if len(
            feature_vec) >= 3 else np.array([0.3, 0.5, 0.2])
        base_color = np.clip(base_color, 0, 1)

        image = np.random.rand(3, 64, 64) * 0.3 + \
            base_color[:, None, None] * 0.7
        image = image.astype(np.float32)

        return torch.FloatTensor(image)


class ImageEncoder(nn.Module):
    """
    CNN encoder for satellite imagery using pretrained ResNet backbone.
    """

    def __init__(self, num_channels=3, pretrained=True, embedding_dim=256):
        super(ImageEncoder, self).__init__()

        # Use ResNet18 as backbone (can swap for ResNet34, ResNet50, EfficientNet, etc.)
        self.backbone = models.resnet18(pretrained=pretrained)

        # Modify first layer if using different number of channels (e.g., multispectral)
        if num_channels != 3:
            self.backbone.conv1 = nn.Conv2d(
                num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Remove the final classification layer
        self.backbone.fc = nn.Identity()

        # Add custom embedding layers
        self.fc = nn.Sequential(
            nn.Linear(512, embedding_dim),  # ResNet18 outputs 512 features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU()
        )

    def forward(self, x):
        features = self.backbone(x)
        embedding = self.fc(features)
        return embedding


class TabularEncoder(nn.Module):
    """
    Neural network encoder for tabular features with entity embeddings for categoricals.
    """

    def __init__(self,
                 num_numerical: int,
                 categorical_dims: Dict[str, int],
                 embedding_dim=32,
                 hidden_dims=[128, 64]):
        super(TabularEncoder, self).__init__()

        self.num_numerical = num_numerical
        self.categorical_dims = categorical_dims

        # Create embeddings for each categorical variable
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(num_categories, embedding_dim)
            for name, num_categories in categorical_dims.items()
        })

        # Calculate total input dimension
        total_embedding_dim = len(categorical_dims) * embedding_dim
        input_dim = num_numerical + total_embedding_dim

        # Build dense layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim

        self.network = nn.Sequential(*layers)
        self.output_dim = prev_dim

    def forward(self, numerical_features, categorical_indices):
        """
        Args:
            numerical_features: Tensor of shape (batch_size, num_numerical)
            categorical_indices: Dict of tensors {feature_name: (batch_size,)}
        """
        # Get embeddings for categorical features
        embedded_categoricals = []
        for name, embedding_layer in self.embeddings.items():
            if name in categorical_indices:
                embedded = embedding_layer(categorical_indices[name])
                embedded_categoricals.append(embedded)

        # Concatenate numerical and embedded categorical features
        if embedded_categoricals:
            categorical_tensor = torch.cat(embedded_categoricals, dim=1)
            x = torch.cat([numerical_features, categorical_tensor], dim=1)
        else:
            x = numerical_features

        return self.network(x)


class HybridYewDensityModel(nn.Module):
    """
    Hybrid model combining image and tabular encoders for yew density prediction.
    """

    def __init__(self,
                 image_channels=3,
                 num_numerical_features=10,
                 categorical_dims=None,
                 image_embedding_dim=256,
                 tabular_embedding_dim=32,
                 fusion_hidden_dims=[256, 128, 64]):
        super(HybridYewDensityModel, self).__init__()

        categorical_dims = categorical_dims or {}

        # Image encoder
        self.image_encoder = ImageEncoder(
            num_channels=image_channels,
            pretrained=True,
            embedding_dim=image_embedding_dim
        )

        # Tabular encoder
        self.tabular_encoder = TabularEncoder(
            num_numerical=num_numerical_features,
            categorical_dims=categorical_dims,
            embedding_dim=tabular_embedding_dim,
            hidden_dims=[128, 64]
        )

        # Fusion network
        fusion_input_dim = image_embedding_dim + self.tabular_encoder.output_dim

        fusion_layers = []
        prev_dim = fusion_input_dim
        for hidden_dim in fusion_hidden_dims:
            fusion_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim

        self.fusion_network = nn.Sequential(*fusion_layers)

        # Output layer (predicting density)
        self.output = nn.Sequential(
            nn.Linear(prev_dim, 1),
            nn.ReLU()  # Density must be non-negative
        )

    def forward(self, image, numerical_features, categorical_indices):
        # Encode image
        image_embedding = self.image_encoder(image)

        # Encode tabular data
        tabular_embedding = self.tabular_encoder(
            numerical_features, categorical_indices)

        # Fuse embeddings
        combined = torch.cat([image_embedding, tabular_embedding], dim=1)
        fused = self.fusion_network(combined)

        # Predict density
        density = self.output(fused)

        # Only squeeze last dimension to preserve batch dimension
        return density.squeeze(-1)


class YewDataPreprocessor:
    """
    Preprocesses data for the yew density prediction model.
    """

    def __init__(self, bc_sample_path='bc_sample_data-2025-10-09/bc_sample_data.csv'):
        self.bc_sample_path = bc_sample_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.categorical_mappings = {}

    def parse_species_composition(self, composition_string):
        """Parse species composition string to extract Pacific Yew (TW) percentage."""
        if not composition_string or pd.isna(composition_string):
            return 0

        import re
        pattern = r'TW(\d{2,3})'  # TW = Taxus brevifolia (Pacific Yew)
        match = re.search(pattern, str(composition_string))

        if match:
            return int(match.group(1))
        return 0

    def load_and_prepare_data(self):
        """Load BC sample data and prepare features and targets."""
        print("Loading BC sample data...")
        df = pd.read_csv(self.bc_sample_path)

        print(f"Loaded {len(df)} records")

        # Calculate yew density
        df['YEW_PERCENTAGE'] = df['SPB_CPCT_LS'].apply(
            self.parse_species_composition)
        df['YEW_DENSITY_HA'] = (df['YEW_PERCENTAGE'] /
                                100.0) * df['STEMS_HA_LS'].fillna(0)

        # Filter for valid records with location and forest data
        df = df[
            (df['BC_ALBERS_X'].notna()) &
            (df['BC_ALBERS_Y'].notna()) &
            (df['STEMS_HA_LS'].notna())
        ].copy()

        print(f"After filtering: {len(df)} valid records")
        print(
            f"Records with yew: {(df['YEW_DENSITY_HA'] > 0).sum()} ({(df['YEW_DENSITY_HA'] > 0).sum() / len(df) * 100:.2f}%)")

        return df

    def prepare_features(self, df):
        """
        Prepare numerical and categorical features.

        Returns:
            numerical_features: np.ndarray
            categorical_features: dict of np.ndarray
            targets: np.ndarray
            feature_info: dict with metadata
        """
        print("\nPreparing features...")

        # Define numerical features
        numerical_cols = [
            'BA_HA_LS',      # Basal area
            'BA_HA_DS',      # Dead standing basal area
            'STEMS_HA_LS',   # Stem density
            'STEMS_HA_DS',   # Dead stem density
            'VHA_WSV_LS',    # Volume
            'VHA_NTWB_LS',   # Net merchantable volume
            'SI_M_TLSO',     # Site index
            'HT_TLSO',       # Height
            'AGEB_TLSO',     # Age
            'BC_ALBERS_X',   # Location X
            'BC_ALBERS_Y',   # Location Y
            'MEAS_YR'        # Measurement year
        ]

        # Define categorical features
        categorical_cols = [
            'BEC_ZONE',
            'TSA_DESC',
            'SAMPLE_ESTABLISHMENT_TYPE',
            'SPC_LIVE_1'
        ]

        # Fill missing numerical values
        for col in numerical_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        # Extract numerical features
        numerical_features = df[numerical_cols].values

        # Normalize numerical features
        numerical_features = self.scaler.fit_transform(numerical_features)

        # Encode categorical features
        categorical_features = {}
        categorical_dims = {}

        for col in categorical_cols:
            if col in df.columns:
                # Fill missing categoricals
                df[col] = df[col].fillna('UNKNOWN')

                # Encode
                le = LabelEncoder()
                encoded = le.fit_transform(df[col].astype(str))

                categorical_features[col] = encoded
                self.label_encoders[col] = le
                categorical_dims[col] = len(le.classes_)

                print(f"  {col}: {len(le.classes_)} unique values")

        # Get targets
        targets = df['YEW_DENSITY_HA'].values

        # Get site identifiers for spatial splitting
        site_ids = df['SITE_IDENTIFIER'].values
        coordinates = df[['BC_ALBERS_X', 'BC_ALBERS_Y']].values

        feature_info = {
            'numerical_cols': numerical_cols,
            'categorical_cols': categorical_cols,
            'categorical_dims': categorical_dims,
            'num_numerical': len(numerical_cols),
            'site_ids': site_ids,
            'coordinates': coordinates
        }

        print(f"\nFeature summary:")
        print(f"  Numerical features: {len(numerical_cols)}")
        print(f"  Categorical features: {len(categorical_cols)}")
        print(f"  Total samples: {len(targets)}")
        print(f"  Samples with yew (target > 0): {(targets > 0).sum()}")
        print(
            f"  Target range: {targets.min():.2f} - {targets.max():.2f} stems/ha")
        print(f"  Target mean: {targets.mean():.2f} stems/ha")
        print(f"  Target median: {np.median(targets):.2f} stems/ha")

        return numerical_features, categorical_features, targets, feature_info

    def save_preprocessor(self, path='yew_preprocessor.pkl'):
        """Save preprocessor for later use."""
        with open(path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'label_encoders': self.label_encoders
            }, f)
        print(f"Preprocessor saved to {path}")


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in regression.
    Focuses training on hard examples.
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions, targets):
        mse = (predictions - targets) ** 2
        focal_weight = (1 + mse) ** self.gamma
        loss = focal_weight * mse
        return loss.mean()


class YewDensityTrainer:
    """
    Training pipeline for the yew density prediction model.
    """

    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device) if model is not None else None
        self.device = device
        self.history = defaultdict(list)

    @staticmethod
    def create_weighted_sampler(targets, yew_weight=10.0):
        """
        Create weighted sampler to handle class imbalance.
        Give higher weight to samples with yew present.
        """
        weights = torch.ones(len(targets))
        # Higher weight for yew-present samples
        weights[targets > 0] = yew_weight

        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )

        return sampler

    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_targets = []

        for batch_idx, (images, numerical, categorical, targets) in enumerate(train_loader):
            images = images.to(self.device)
            numerical = numerical.to(self.device)
            targets = targets.to(self.device)

            # Move categorical features to device
            categorical_indices = {
                name: values.to(self.device) for name, values in categorical.items()
            }

            optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images, numerical, categorical_indices)

            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_predictions.extend(outputs.detach().cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

        avg_loss = total_loss / len(train_loader)
        predictions = np.array(all_predictions)
        targets_np = np.array(all_targets)

        # Calculate metrics
        mae = mean_absolute_error(targets_np, predictions)
        rmse = np.sqrt(mean_squared_error(targets_np, predictions))

        return avg_loss, mae, rmse

    def validate(self, val_loader, criterion):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for images, numerical, categorical, targets in val_loader:
                images = images.to(self.device)
                numerical = numerical.to(self.device)
                targets = targets.to(self.device)

                # Move categorical features to device
                categorical_indices = {
                    name: values.to(self.device) for name, values in categorical.items()
                }

                outputs = self.model(images, numerical, categorical_indices)
                loss = criterion(outputs, targets)

                total_loss += loss.item()
                all_predictions.extend(outputs.cpu().numpy().tolist())
                all_targets.extend(targets.cpu().numpy().tolist())

        avg_loss = total_loss / len(val_loader)
        predictions = np.array(all_predictions)
        targets_np = np.array(all_targets)

        # Calculate metrics
        mae = mean_absolute_error(targets_np, predictions)
        rmse = np.sqrt(mean_squared_error(targets_np, predictions))
        r2 = r2_score(targets_np, predictions)

        return avg_loss, mae, rmse, r2, predictions, targets_np

    def train(self, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
        """Full training loop."""
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        optimizer = optim.Adam(self.model.parameters(),
                               lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15

        print("\nStarting training...")
        print(f"Device: {self.device}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Initial learning rate: {learning_rate}")

        for epoch in range(num_epochs):
            # Train
            train_loss, train_mae, train_rmse = self.train_epoch(
                train_loader, optimizer, criterion
            )

            # Validate
            val_loss, val_mae, val_rmse, val_r2, val_preds, val_targets = self.validate(
                val_loader, criterion
            )

            # Update learning rate
            scheduler.step(val_loss)

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_mae'].append(train_mae)
            self.history['train_rmse'].append(train_rmse)
            self.history['val_loss'].append(val_loss)
            self.history['val_mae'].append(val_mae)
            self.history['val_rmse'].append(val_rmse)
            self.history['val_r2'].append(val_r2)

            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f"\nEpoch [{epoch+1}/{num_epochs}]")
                print(
                    f"  Train Loss: {train_loss:.4f}, MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}")
                print(
                    f"  Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(),
                           'best_yew_density_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break

        print("\nTraining complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")

        # Load best model
        self.model.load_state_dict(torch.load('best_yew_density_model.pth'))

    def plot_training_history(self, save_path='training_history.png'):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Validation')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # MAE
        axes[0, 1].plot(self.history['train_mae'], label='Train')
        axes[0, 1].plot(self.history['val_mae'], label='Validation')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # RMSE
        axes[1, 0].plot(self.history['train_rmse'], label='Train')
        axes[1, 0].plot(self.history['val_rmse'], label='Validation')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].set_title('Root Mean Squared Error')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # R²
        axes[1, 1].plot(self.history['val_r2'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('R² Score')
        axes[1, 1].set_title('Validation R² Score')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")


def spatial_train_test_split(coordinates, test_size=0.2, val_size=0.1, random_state=42):
    """
    Create train/val/test splits that respect spatial autocorrelation.
    Ensures nearby sites don't end up in different splits.
    """
    from sklearn.cluster import KMeans

    # Use K-means clustering to create spatial blocks
    n_clusters = max(10, int(len(coordinates) / 50))  # At least 10 clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    spatial_blocks = kmeans.fit_predict(coordinates)

    unique_blocks = np.unique(spatial_blocks)

    # Split blocks instead of individual samples
    train_val_blocks, test_blocks = train_test_split(
        unique_blocks, test_size=test_size, random_state=random_state
    )

    train_blocks, val_blocks = train_test_split(
        train_val_blocks, test_size=val_size/(1-test_size), random_state=random_state
    )

    # Get indices for each split
    train_idx = np.where(np.isin(spatial_blocks, train_blocks))[0]
    val_idx = np.where(np.isin(spatial_blocks, val_blocks))[0]
    test_idx = np.where(np.isin(spatial_blocks, test_blocks))[0]

    print(f"\nSpatial split created:")
    print(
        f"  Train: {len(train_idx)} samples ({len(train_blocks)} spatial blocks)")
    print(f"  Val: {len(val_idx)} samples ({len(val_blocks)} spatial blocks)")
    print(
        f"  Test: {len(test_idx)} samples ({len(test_blocks)} spatial blocks)")

    return train_idx, val_idx, test_idx


def main():
    """Main training pipeline."""
    print("="*70)
    print("Pacific Yew Density Prediction - Multi-Modal Deep Learning")
    print("PyTorch Implementation")
    print("="*70)

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Step 1: Load and preprocess data
    print("\n" + "="*70)
    print("STEP 1: Data Loading and Preprocessing")
    print("="*70)

    preprocessor = YewDataPreprocessor()
    df = preprocessor.load_and_prepare_data()
    numerical_features, categorical_features, targets, feature_info = preprocessor.prepare_features(
        df)

    # Step 2: Create spatial train/val/test splits
    print("\n" + "="*70)
    print("STEP 2: Creating Spatial Train/Val/Test Splits")
    print("="*70)

    train_idx, val_idx, test_idx = spatial_train_test_split(
        feature_info['coordinates'],
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )

    # Create placeholder imagery paths (in production, these would be real image files)
    imagery_paths = [f"placeholder_{i}" for i in range(len(targets))]

    # Helper function to subset categorical features
    def subset_categorical_features(categorical_features, indices):
        return {name: values[indices] for name, values in categorical_features.items()}

    # Create datasets
    train_dataset = YewDensityDataset(
        [imagery_paths[i] for i in train_idx],
        numerical_features[train_idx],
        subset_categorical_features(categorical_features, train_idx),
        targets[train_idx]
    )

    val_dataset = YewDensityDataset(
        [imagery_paths[i] for i in val_idx],
        numerical_features[val_idx],
        subset_categorical_features(categorical_features, val_idx),
        targets[val_idx]
    )

    test_dataset = YewDensityDataset(
        [imagery_paths[i] for i in test_idx],
        numerical_features[test_idx],
        subset_categorical_features(categorical_features, test_idx),
        targets[test_idx]
    )

    # Create data loaders with weighted sampling for training
    train_sampler = YewDensityTrainer.create_weighted_sampler(
        targets[train_idx], yew_weight=10.0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=train_sampler,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )

    # Step 3: Create model
    print("\n" + "="*70)
    print("STEP 3: Model Architecture")
    print("="*70)

    model = HybridYewDensityModel(
        image_channels=3,  # RGB (or more for multispectral)
        num_numerical_features=feature_info['num_numerical'],
        categorical_dims=feature_info['categorical_dims'],
        image_embedding_dim=256,
        tabular_embedding_dim=32,
        fusion_hidden_dims=[256, 128, 64]
    )

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

    print(f"\nModel created successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"\nModel architecture:")
    print(model)

    # Step 4: Train model
    print("\n" + "="*70)
    print("STEP 4: Training")
    print("="*70)

    trainer = YewDensityTrainer(model, device=device)
    trainer.train(
        train_loader,
        val_loader,
        num_epochs=50,
        learning_rate=0.001
    )

    # Step 5: Evaluate on test set
    print("\n" + "="*70)
    print("STEP 5: Final Evaluation on Test Set")
    print("="*70)

    criterion = nn.MSELoss()
    test_loss, test_mae, test_rmse, test_r2, test_preds, test_targets = trainer.validate(
        test_loader, criterion
    )

    print(f"\nTest Set Results:")
    print(f"  MAE: {test_mae:.4f} stems/ha")
    print(f"  RMSE: {test_rmse:.4f} stems/ha")
    print(f"  R² Score: {test_r2:.4f}")

    # Calculate metrics for yew-present samples only
    yew_present_mask = test_targets > 0
    if yew_present_mask.sum() > 0:
        yew_mae = mean_absolute_error(
            test_targets[yew_present_mask],
            test_preds[yew_present_mask]
        )
        yew_rmse = np.sqrt(mean_squared_error(
            test_targets[yew_present_mask],
            test_preds[yew_present_mask]
        ))
        print(f"\nPerformance on yew-present sites only:")
        print(f"  MAE: {yew_mae:.4f} stems/ha")
        print(f"  RMSE: {yew_rmse:.4f} stems/ha")
        print(f"  Number of yew-present sites: {yew_present_mask.sum()}")

    # Step 6: Save artifacts
    print("\n" + "="*70)
    print("STEP 6: Saving Artifacts")
    print("="*70)

    # Save model
    torch.save(model.state_dict(), 'yew_density_model_final.pth')
    print("Model saved to 'yew_density_model_final.pth'")

    # Save preprocessor
    preprocessor.save_preprocessor('yew_preprocessor.pkl')

    # Save training history plot
    trainer.plot_training_history('yew_training_history.png')

    # Save feature info
    with open('yew_feature_info.pkl', 'wb') as f:
        pickle.dump(feature_info, f)
    print("Feature info saved to 'yew_feature_info.pkl'")

    print("\n" + "="*70)
    print("Training pipeline completed successfully!")
    print("="*70)
    print("\nNext steps:")
    print("1. Extract satellite imagery using Earth Engine (see extract_ee_imagery.py)")
    print("2. Re-train model with actual satellite data")
    print("3. Use trained model for predictions on new sites")
    print("="*70)


if __name__ == "__main__":
    main()
