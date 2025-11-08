#!/usr/bin/env python3
"""
CNN Data Loader for 64x64 Image Patches
=======================================

PyTorch Dataset and DataLoader for training CNNs on Sentinel-2 image patches.

Author: GitHub Copilot
Date: November 7, 2025
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


class YewImageDataset(Dataset):
    """
    PyTorch Dataset for 64x64 Sentinel-2 image patches.

    Images are stored as numpy arrays with shape (4, 64, 64) representing
    [Blue, Green, Red, NIR] bands.
    """

    def __init__(
        self,
        image_dir: str,
        metadata_csv: str,
        transform=None,
        normalize=True,
        augment=False
    ):
        """
        Args:
            image_dir: Directory containing yew/ and no_yew/ subdirectories
            metadata_csv: Path to image_metadata.csv
            transform: Custom albumentations transform (overrides augment)
            normalize: If True, normalize bands to [0, 1]
            augment: If True, apply data augmentation
        """
        self.image_dir = Path(image_dir)
        self.normalize = normalize

        # Load metadata
        self.metadata = pd.read_csv(metadata_csv)
        print(f"Loaded {len(self.metadata)} image records")

        # Verify images exist
        valid_records = []
        for idx, row in self.metadata.iterrows():
            image_path = self.image_dir / row['image_path']
            if image_path.exists():
                valid_records.append(row)

        self.metadata = pd.DataFrame(valid_records).reset_index(drop=True)
        print(f"Found {len(self.metadata)} valid images")
        print(f"  Yew: {self.metadata['has_yew'].sum()}")
        print(f"  No-yew: {(~self.metadata['has_yew']).sum()}")

        # Setup transforms
        if transform is not None:
            self.transform = transform
        elif augment:
            self.transform = self._get_augmentation_pipeline()
        else:
            self.transform = self._get_basic_pipeline()

    def _get_basic_pipeline(self):
        """Basic transform: just convert to tensor."""
        return A.Compose([
            ToTensorV2()
        ])

    def _get_augmentation_pipeline(self):
        """Augmentation pipeline for training."""
        return A.Compose([
            # Spatial transforms
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=45,
                p=0.5
            ),

            # Pixel-level transforms (careful with multi-spectral data)
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),

            ToTensorV2()
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor of shape (4, 64, 64) - [Blue, Green, Red, NIR]
            label: 1 if yew present, 0 otherwise
            metadata: Dict with site_identifier, bec_zone, etc.
        """
        row = self.metadata.iloc[idx]

        # Load image
        image_path = self.image_dir / row['image_path']
        image = np.load(image_path)  # Shape: (4, 64, 64)

        # Normalize if requested
        if self.normalize:
            # Sentinel-2 values are typically 0-10000
            # Normalize to [0, 1]
            image = image / 10000.0
            image = np.clip(image, 0, 1)

        # Convert to channels-last for albumentations (64, 64, 4)
        image = np.transpose(image, (1, 2, 0))

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1)

        # Get label
        label = 1 if row['has_yew'] else 0

        # Metadata
        metadata = {
            'site_identifier': row['site_identifier'],
            'bec_zone': row['bec_zone'],
            'measurement_year': row['measurement_year']
        }

        return image.float(), torch.tensor(label, dtype=torch.long), metadata


def create_weighted_sampler(dataset: YewImageDataset, yew_weight: float = 10.0):
    """
    Create a weighted sampler to handle class imbalance.

    Args:
        dataset: YewImageDataset instance
        yew_weight: Weight for yew class (default 10.0 for ~1% yew prevalence)

    Returns:
        WeightedRandomSampler
    """
    labels = dataset.metadata['has_yew'].values

    # Calculate weights
    weights = np.ones(len(labels))
    weights[labels == True] = yew_weight

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )

    print(f"Created weighted sampler:")
    print(f"  Yew weight: {yew_weight}x")
    print(
        f"  Expected yew samples per epoch: ~{(labels * yew_weight).sum() / weights.sum() * len(weights):.0f}")

    return sampler


def create_data_loaders(
    image_dir: str,
    metadata_csv: str,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    batch_size: int = 32,
    num_workers: int = 4,
    yew_weight: float = 10.0,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.

    Args:
        image_dir: Directory with image patches
        metadata_csv: Path to metadata CSV
        train_frac: Fraction for training (default 0.7)
        val_frac: Fraction for validation (default 0.15)
        batch_size: Batch size
        num_workers: Number of data loading workers
        yew_weight: Weight for yew class in sampler
        random_state: Random seed

    Returns:
        train_loader, val_loader, test_loader
    """
    # Load full metadata to split
    full_metadata = pd.read_csv(metadata_csv)

    # Stratified split to maintain yew/no-yew ratio
    from sklearn.model_selection import train_test_split

    # First split: train vs (val + test)
    train_meta, temp_meta = train_test_split(
        full_metadata,
        train_size=train_frac,
        stratify=full_metadata['has_yew'],
        random_state=random_state
    )

    # Second split: val vs test
    val_size = val_frac / (1 - train_frac)
    val_meta, test_meta = train_test_split(
        temp_meta,
        train_size=val_size,
        stratify=temp_meta['has_yew'],
        random_state=random_state
    )

    print(f"\nDataset splits:")
    print(
        f"  Train: {len(train_meta)} images ({train_meta['has_yew'].sum()} yew)")
    print(f"  Val:   {len(val_meta)} images ({val_meta['has_yew'].sum()} yew)")
    print(
        f"  Test:  {len(test_meta)} images ({test_meta['has_yew'].sum()} yew)")

    # Save split metadata
    output_dir = Path(image_dir)
    train_meta.to_csv(output_dir / 'train_metadata.csv', index=False)
    val_meta.to_csv(output_dir / 'val_metadata.csv', index=False)
    test_meta.to_csv(output_dir / 'test_metadata.csv', index=False)
    print(f"  Split metadata saved to {output_dir}")

    # Create datasets
    train_dataset = YewImageDataset(
        image_dir,
        output_dir / 'train_metadata.csv',
        augment=True
    )

    val_dataset = YewImageDataset(
        image_dir,
        output_dir / 'val_metadata.csv',
        augment=False
    )

    test_dataset = YewImageDataset(
        image_dir,
        output_dir / 'test_metadata.csv',
        augment=False
    )

    # Create weighted sampler for training
    train_sampler = create_weighted_sampler(
        train_dataset, yew_weight=yew_weight)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


# Example usage
if __name__ == '__main__':
    print("="*80)
    print("YEW IMAGE DATASET TEST")
    print("="*80)

    image_dir = 'data/ee_imagery/image_patches_64x64'
    metadata_csv = f'{image_dir}/image_metadata.csv'

    # Check if data exists
    if not Path(metadata_csv).exists():
        print(f"\n✗ Metadata not found: {metadata_csv}")
        print("  Run extract_ee_image_patches.py first to download images")
        exit(1)

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        image_dir=image_dir,
        metadata_csv=metadata_csv,
        batch_size=16,
        num_workers=2,
        yew_weight=10.0
    )

    # Test loading a batch
    print("\n" + "="*80)
    print("TESTING DATA LOADING")
    print("="*80)

    for images, labels, metadata in train_loader:
        print(f"\nBatch loaded successfully!")
        # Should be (batch_size, 4, 64, 64)
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Value ranges:")
        print(f"    Images: [{images.min():.3f}, {images.max():.3f}]")
        print(f"    Labels: {labels.unique().tolist()}")
        print(f"  Yew in batch: {labels.sum().item()} / {len(labels)}")
        break

    print("\n✓ Data loading successful!")
