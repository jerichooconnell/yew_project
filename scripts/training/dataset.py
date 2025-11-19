#!/usr/bin/env python3
"""
Custom Dataset for Satellite Imagery
=====================================

PyTorch Dataset for loading 4-channel (RGB+NIR) satellite imagery.

Author: GitHub Copilot
Date: November 14, 2025
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SatelliteImageDataset(Dataset):
    """Dataset for loading satellite images with 4 channels (B, G, R, NIR)."""

    def __init__(self, metadata_files, image_base_dir, transform=None, augment=False):
        """
        Args:
            metadata_files: List of metadata CSV paths or single path
            image_base_dir: Base directory containing image subdirectories
            transform: Albumentations transform pipeline
            augment: Whether to apply data augmentation
        """
        self.image_base_dir = Path(image_base_dir)

        # Load metadata
        if isinstance(metadata_files, (list, tuple)):
            dfs = [pd.read_csv(f) for f in metadata_files]
            self.metadata = pd.concat(dfs, ignore_index=True)
        else:
            self.metadata = pd.read_csv(metadata_files)

        # Setup transforms
        if transform is None:
            if augment:
                self.transform = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.Transpose(p=0.5),
                    # Normalize each channel independently
                    A.Normalize(mean=[0.0, 0.0, 0.0, 0.0],
                                std=[1.0, 1.0, 1.0, 1.0],
                                max_pixel_value=1.0),
                    ToTensorV2()
                ])
            else:
                self.transform = A.Compose([
                    A.Normalize(mean=[0.0, 0.0, 0.0, 0.0],
                                std=[1.0, 1.0, 1.0, 1.0],
                                max_pixel_value=1.0),
                    ToTensorV2()
                ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """Get a single image and label."""
        row = self.metadata.iloc[idx]

        # Load image
        img_path = self.image_base_dir / row['image_path']
        img = np.load(img_path)  # Shape: (4, 64, 64)

        # Convert to (H, W, C) for albumentations
        img = np.transpose(img, (1, 2, 0))  # (64, 64, 4)

        # Normalize to [0, 1] using percentile clipping
        img_normalized = np.zeros_like(img, dtype=np.float32)
        for i in range(4):
            band = img[:, :, i]
            p_low = np.percentile(band, 2)
            p_high = np.percentile(band, 98)
            band_clipped = np.clip(band, p_low, p_high)
            band_min = band_clipped.min()
            band_max = band_clipped.max()
            if band_max > band_min:
                img_normalized[:, :, i] = (
                    band_clipped - band_min) / (band_max - band_min)
            else:
                img_normalized[:, :, i] = 0.0

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=img_normalized)
            img_tensor = transformed['image']
        else:
            img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1)

        # Get label
        label = 1 if row['has_yew'] else 0

        # Create metadata dictionary with all relevant info
        metadata = {
            'image_path': row['image_path'],
            'latitude': row.get('latitude', None),
            'longitude': row.get('longitude', None),
            'observation_id': row.get('observation_id', None),
            'source_dataset': row.get('source_dataset', None),
            'index': idx
        }

        return img_tensor, label, metadata


def custom_collate_fn(batch):
    """
    Custom collate function to handle metadata dictionaries.

    Args:
        batch: List of (image, label, metadata) tuples

    Returns:
        Tuple of (images, labels, metadata_list)
    """
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    metadata = [item[2] for item in batch]  # Keep as list of dicts
    return images, labels, metadata


def get_dataloaders(yew_metadata, no_yew_metadata, image_base_dir,
                    batch_size=16, val_split=0.2, num_workers=4, random_seed=42):
    """
    Create train and validation dataloaders with balanced classes.

    Args:
        yew_metadata: Path to yew metadata CSV
        no_yew_metadata: Path to no-yew metadata CSV
        image_base_dir: Base directory for images
        batch_size: Batch size for training
        val_split: Fraction of data for validation
        num_workers: Number of data loading workers
        random_seed: Random seed for reproducibility

    Returns:
        train_loader, val_loader, dataset_info dict
    """
    # Load metadata
    yew_df = pd.read_csv(yew_metadata)
    no_yew_df = pd.read_csv(no_yew_metadata)

    # Add labels
    yew_df['has_yew'] = True
    no_yew_df['has_yew'] = False

    # Shuffle and split each class
    np.random.seed(random_seed)

    yew_indices = np.random.permutation(len(yew_df))
    no_yew_indices = np.random.permutation(len(no_yew_df))

    yew_val_size = int(len(yew_df) * val_split)
    no_yew_val_size = int(len(no_yew_df) * val_split)

    # Split indices
    yew_train_idx = yew_indices[yew_val_size:]
    yew_val_idx = yew_indices[:yew_val_size]
    no_yew_train_idx = no_yew_indices[no_yew_val_size:]
    no_yew_val_idx = no_yew_indices[:no_yew_val_size]

    # Create train/val dataframes
    train_df = pd.concat([
        yew_df.iloc[yew_train_idx],
        no_yew_df.iloc[no_yew_train_idx]
    ], ignore_index=True)

    val_df = pd.concat([
        yew_df.iloc[yew_val_idx],
        no_yew_df.iloc[no_yew_val_idx]
    ], ignore_index=True)

    # Save splits for reproducibility
    train_df.to_csv('data/processed/train_split.csv', index=False)
    val_df.to_csv('data/processed/val_split.csv', index=False)

    # Create datasets
    train_dataset = SatelliteImageDataset(
        'data/processed/train_split.csv',
        image_base_dir,
        augment=True
    )

    val_dataset = SatelliteImageDataset(
        'data/processed/val_split.csv',
        image_base_dir,
        augment=False
    )

    # Create dataloaders with custom collate function
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    # Dataset info
    info = {
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'train_yew': len(yew_train_idx),
        'train_no_yew': len(no_yew_train_idx),
        'val_yew': len(yew_val_idx),
        'val_no_yew': len(no_yew_val_idx),
        'batch_size': batch_size,
        'num_workers': num_workers
    }

    return train_loader, val_loader, info


if __name__ == '__main__':
    # Test dataset loading
    print("Testing dataset loading...")

    yew_meta = 'data/ee_imagery/image_patches_64x64/inat_yew_image_metadata.csv'
    no_yew_meta = 'data/ee_imagery/image_patches_64x64/no_yew_image_metadata.csv'
    base_dir = 'data/ee_imagery/image_patches_64x64'

    train_loader, val_loader, info = get_dataloaders(
        yew_meta, no_yew_meta, base_dir,
        batch_size=8, val_split=0.2
    )

    print(f"\nDataset Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Test loading a batch
    print("\nLoading test batch...")
    images, labels, obs_ids = next(iter(train_loader))
    print(f"  Batch shape: {images.shape}")
    print(f"  Labels: {labels}")
    print(
        f"  Label distribution: {labels.sum().item()} yew, {(labels == 0).sum().item()} no-yew")
    print("\n✓ Dataset loading successful!")
