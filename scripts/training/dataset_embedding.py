#!/usr/bin/env python3
"""
PyTorch Dataset for Google Satellite Embedding data.

Loads 64-channel embedding patches and applies augmentations for training.

Author: GitHub Copilot
Date: November 20, 2025
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class EmbeddingDataset(Dataset):
    """
    Dataset for Google Satellite Embedding patches.

    Each sample is a 64-channel embedding vector (64, H, W).
    Unlike spectral data, embeddings are already normalized feature vectors
    in a learned embedding space, so we apply minimal normalization.
    """

    def __init__(self, metadata_csv, image_dir, transform=None, normalize_embeddings=True):
        """
        Initialize dataset.

        Args:
            metadata_csv: Path to CSV with columns: lat/latitude, lon/longitude, label/has_yew
            image_dir: Directory containing embedding .npy files
            transform: Albumentations transform pipeline
            normalize_embeddings: Whether to normalize embeddings (recommended: True)
        """
        self.df = pd.read_csv(metadata_csv)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.normalize_embeddings = normalize_embeddings

        # Standardize column names
        if 'lat' in self.df.columns and 'latitude' not in self.df.columns:
            self.df['latitude'] = self.df['lat']
        if 'lon' in self.df.columns and 'longitude' not in self.df.columns:
            self.df['longitude'] = self.df['lon']

        # Map text labels to integers
        if 'label' in self.df.columns:
            self.df['label_int'] = (self.df['label'] == 'yew').astype(int)
        elif 'has_yew' in self.df.columns:
            self.df['label_int'] = self.df['has_yew'].astype(int)
        else:
            self.df['label_int'] = 0  # Default for prediction

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        lat = row['latitude']
        lon = row['longitude']
        label = row['label_int']

        # Load embedding patch
        filename = f'embedding_{lat:.6f}_{lon:.6f}.npy'
        img_path = self.image_dir / filename

        if not img_path.exists():
            # Return dummy data if file doesn't exist
            img = np.zeros((64, 64, 64), dtype=np.float32)
        else:
            img = np.load(img_path)  # Shape: (64, 64, 64)

            # Convert to (H, W, C) for albumentations
            img = np.transpose(img, (1, 2, 0))  # (64, 64, 64)

        # Normalize embeddings if requested
        # Embeddings are typically in range [-1, 1] but can vary
        if self.normalize_embeddings:
            # Apply per-sample standardization
            mean = img.mean(axis=(0, 1), keepdims=True)
            std = img.std(axis=(0, 1), keepdims=True)
            # Avoid division by zero or very small std (use epsilon for numerical stability)
            std = np.maximum(std, 1e-6)
            img = (img - mean) / std
            # Clip extreme values to prevent numerical issues
            img = np.clip(img, -10, 10)

        # Apply transformations
        if self.transform:
            transformed = self.transform(image=img)
            img = transformed['image']
        else:
            # Convert to tensor manually
            img = torch.from_numpy(img).permute(2, 0, 1)  # (C, H, W)

        return img.float(), torch.tensor(label, dtype=torch.long)


def get_transforms(train=True, img_size=64):
    """
    Get augmentation transforms for training or validation.

    Args:
        train: Whether to apply training augmentations
        img_size: Image size (default: 64)

    Returns:
        Albumentations transform pipeline
    """
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=45,
                p=0.5
            ),
            # Note: We don't apply color/brightness augmentations to embeddings
            # since they're not in spectral space
            ToTensorV2()
        ])
    else:
        return A.Compose([
            ToTensorV2()
        ])


def get_dataloaders(train_csv, val_csv, image_dir, batch_size=16, num_workers=4):
    """
    Create train and validation dataloaders.

    Args:
        train_csv: Path to training metadata CSV
        val_csv: Path to validation metadata CSV
        image_dir: Directory containing embedding patches
        batch_size: Batch size
        num_workers: Number of data loading workers

    Returns:
        train_loader, val_loader, train_dataset, val_dataset
    """
    train_dataset = EmbeddingDataset(
        train_csv,
        image_dir,
        # Disable augmentation for now to test
        transform=get_transforms(train=False),
        normalize_embeddings=False  # Embeddings are already normalized by Google
    )

    val_dataset = EmbeddingDataset(
        val_csv,
        image_dir,
        transform=get_transforms(train=False),
        normalize_embeddings=False  # Embeddings are already normalized by Google
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, train_dataset, val_dataset


if __name__ == '__main__':
    # Test dataset creation
    print("Testing EmbeddingDataset...")

    # Create dummy data
    dummy_csv = '/tmp/test_embedding.csv'
    dummy_dir = Path('/tmp/test_embeddings')
    dummy_dir.mkdir(exist_ok=True)

    # Create dummy CSV
    pd.DataFrame({
        'latitude': [48.5, 48.6],
        'longitude': [-123.5, -123.6],
        'label': ['yew', 'no_yew']
    }).to_csv(dummy_csv, index=False)

    # Create dummy embedding files
    for lat, lon in [(48.5, -123.5), (48.6, -123.6)]:
        filename = f'embedding_{lat:.6f}_{lon:.6f}.npy'
        dummy_embedding = np.random.randn(64, 64, 64).astype(np.float32)
        np.save(dummy_dir / filename, dummy_embedding)

    # Test dataset
    dataset = EmbeddingDataset(
        dummy_csv,
        dummy_dir,
        transform=get_transforms(train=True),
        normalize_embeddings=True
    )

    img, label = dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Label: {label}")
    print(f"Image dtype: {img.dtype}")
    print(f"Image range: [{img.min():.3f}, {img.max():.3f}]")
    print("✓ Dataset test passed!")
