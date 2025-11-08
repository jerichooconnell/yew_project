#!/usr/bin/env python3
"""
Display 64x64 Sentinel-2 Image Patches
======================================

Visualizes extracted image patches with different band combinations.

Author: GitHub Copilot
Date: November 7, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns


def normalize_for_display(band_array, percentile_clip=2):
    """Normalize band values for display (0-255)."""
    # Clip extreme values
    p_low = np.percentile(band_array, percentile_clip)
    p_high = np.percentile(band_array, 100 - percentile_clip)

    # Normalize to 0-255
    band_clipped = np.clip(band_array, p_low, p_high)
    if p_high > p_low:
        band_norm = ((band_clipped - p_low) /
                     (p_high - p_low) * 255).astype(np.uint8)
    else:
        band_norm = np.zeros_like(band_clipped, dtype=np.uint8)

    return band_norm


def load_and_display_patches(image_dir, n_yew=5, n_no_yew=10):
    """Load and display sample image patches."""

    image_dir = Path(image_dir)

    # Load metadata - try both filenames
    metadata_files = [
        image_dir / 'yew_image_metadata.csv',  # Yew-only extraction
        image_dir / 'image_metadata.csv'        # Full extraction
    ]

    metadata = None
    for metadata_file in metadata_files:
        if metadata_file.exists():
            metadata = pd.read_csv(metadata_file)
            print(f"Loaded metadata from: {metadata_file.name}")
            break

    if metadata is None:
        print(f"\n✗ No metadata file found in {image_dir}")
        print("  Looking for: 'yew_image_metadata.csv' or 'image_metadata.csv'")
        print("  Run extraction script first:")
        print("    python scripts/preprocessing/extract_yew_images_only.py")
        return

    print(f"Total images available: {len(metadata)}")
    print(f"  Yew: {metadata['has_yew'].sum()}")
    print(f"  No-yew: {(~metadata['has_yew']).sum()}")

    # Get sample paths
    yew_meta = metadata[metadata['has_yew'] == True].head(n_yew)
    no_yew_meta = metadata[metadata['has_yew'] == False].head(n_no_yew)

    # Adjust n_no_yew if we have fewer than requested
    if len(no_yew_meta) == 0:
        print("\nNote: No non-yew images available (yew-only dataset)")
        n_no_yew = 0

    print(
        f"\nDisplaying {len(yew_meta)} yew and {len(no_yew_meta)} non-yew images")

    # Load images
    yew_images = []
    yew_info = []
    for _, row in yew_meta.iterrows():
        img_path = image_dir / row['image_path']
        if img_path.exists():
            # Shape: (4, 64, 64) [Blue, Green, Red, NIR]
            img = np.load(img_path)
            yew_images.append(img)
            yew_info.append({
                'site_id': row['site_identifier'],
                'zone': row['bec_zone'],
                'year': row['measurement_year']
            })

    no_yew_images = []
    no_yew_info = []
    for _, row in no_yew_meta.iterrows():
        img_path = image_dir / row['image_path']
        if img_path.exists():
            img = np.load(img_path)
            no_yew_images.append(img)
            no_yew_info.append({
                'site_id': row['site_identifier'],
                'zone': row['bec_zone'],
                'year': row['measurement_year']
            })

    print(
        f"Loaded {len(yew_images)} yew and {len(no_yew_images)} non-yew images")

    # Create visualizations
    create_comparison_figure(yew_images, no_yew_images, yew_info, no_yew_info)
    create_band_display(yew_images, no_yew_images, yew_info, no_yew_info)
    create_spectral_analysis(yew_images, no_yew_images)


def create_comparison_figure(yew_images, no_yew_images, yew_info, no_yew_info):
    """Create true color and false color comparison."""

    n_yew = min(len(yew_images), 5)
    n_no_yew = min(len(no_yew_images), 5)
    n_total = max(n_yew, n_no_yew)

    if n_total == 0:
        print("No images to display")
        return

    # Create figure with 4 rows: yew true color, yew false color, no-yew true color, no-yew false color
    fig, axes = plt.subplots(4, n_total, figsize=(3*n_total, 13))
    if n_total == 1:
        axes = axes.reshape(-1, 1)

    # Yew - True Color (RGB)
    for i in range(n_total):
        ax = axes[0, i]
        if i < len(yew_images):
            img = yew_images[i]
            # Extract RGB (bands 2, 1, 0 = Red, Green, Blue)
            red = normalize_for_display(img[2])
            green = normalize_for_display(img[1])
            blue = normalize_for_display(img[0])
            rgb = np.dstack([red, green, blue])

            ax.imshow(rgb)
            ax.set_title(f"YEW - True Color\n{yew_info[i]['site_id']}\n{yew_info[i]['zone']} ({yew_info[i]['year']})",
                         fontsize=9, color='green', fontweight='bold')
        ax.axis('off')

    # Yew - False Color (NIR-R-G)
    for i in range(n_total):
        ax = axes[1, i]
        if i < len(yew_images):
            img = yew_images[i]
            nir = normalize_for_display(img[3])
            red = normalize_for_display(img[2])
            green = normalize_for_display(img[1])
            false_color = np.dstack([nir, red, green])

            ax.imshow(false_color)
            ax.set_title(f"YEW - False Color (NIR-R-G)\n{yew_info[i]['site_id']}",
                         fontsize=9, color='green', fontweight='bold')
        ax.axis('off')

    # Non-Yew - True Color
    for i in range(n_total):
        ax = axes[2, i]
        if i < len(no_yew_images):
            img = no_yew_images[i]
            red = normalize_for_display(img[2])
            green = normalize_for_display(img[1])
            blue = normalize_for_display(img[0])
            rgb = np.dstack([red, green, blue])

            ax.imshow(rgb)
            ax.set_title(f"NON-YEW - True Color\n{no_yew_info[i]['site_id']}\n{no_yew_info[i]['zone']} ({no_yew_info[i]['year']})",
                         fontsize=9)
        ax.axis('off')

    # Non-Yew - False Color
    for i in range(n_total):
        ax = axes[3, i]
        if i < len(no_yew_images):
            img = no_yew_images[i]
            nir = normalize_for_display(img[3])
            red = normalize_for_display(img[2])
            green = normalize_for_display(img[1])
            false_color = np.dstack([nir, red, green])

            ax.imshow(false_color)
            ax.set_title(f"NON-YEW - False Color (NIR-R-G)\n{no_yew_info[i]['site_id']}",
                         fontsize=9)
        ax.axis('off')

    plt.suptitle('64x64 Sentinel-2 Image Patches: Yew vs Non-Yew Sites',
                 fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()

    save_path = Path('results/figures/ee_patches_comparison.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.show()


def create_band_display(yew_images, no_yew_images, yew_info, no_yew_info):
    """Display individual spectral bands."""

    n_samples = min(3, len(yew_images), len(no_yew_images))
    if n_samples == 0:
        return

    band_names = ['Blue', 'Green', 'Red', 'NIR']

    fig, axes = plt.subplots(n_samples*2, 4, figsize=(16, 4*n_samples*2))

    for sample_idx in range(n_samples):
        # Yew image bands
        if sample_idx < len(yew_images):
            img = yew_images[sample_idx]
            for band_idx, band_name in enumerate(band_names):
                ax = axes[sample_idx*2, band_idx]
                band_data = normalize_for_display(img[band_idx])
                im = ax.imshow(band_data, cmap='viridis')
                ax.set_title(f"YEW - {band_name}\n{yew_info[sample_idx]['site_id']}",
                             fontsize=10, color='green', fontweight='bold')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046)

        # Non-yew image bands
        if sample_idx < len(no_yew_images):
            img = no_yew_images[sample_idx]
            for band_idx, band_name in enumerate(band_names):
                ax = axes[sample_idx*2 + 1, band_idx]
                band_data = normalize_for_display(img[band_idx])
                im = ax.imshow(band_data, cmap='viridis')
                ax.set_title(f"NON-YEW - {band_name}\n{no_yew_info[sample_idx]['site_id']}",
                             fontsize=10)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle('Individual Spectral Bands', fontsize=16, fontweight='bold')
    plt.tight_layout()

    save_path = Path('results/figures/ee_patches_individual_bands.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.show()


def create_spectral_analysis(yew_images, no_yew_images):
    """Analyze spectral signatures."""

    if len(yew_images) == 0 or len(no_yew_images) == 0:
        print("Not enough images for spectral analysis")
        return

    band_names = ['Blue', 'Green', 'Red', 'NIR']

    # Compute mean values per band
    yew_means = []
    yew_stds = []
    for band_idx in range(4):
        band_values = [img[band_idx].mean() for img in yew_images]
        yew_means.append(np.mean(band_values))
        yew_stds.append(np.std(band_values))

    no_yew_means = []
    no_yew_stds = []
    for band_idx in range(4):
        band_values = [img[band_idx].mean() for img in no_yew_images]
        no_yew_means.append(np.mean(band_values))
        no_yew_stds.append(np.std(band_values))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot
    x = np.arange(len(band_names))
    width = 0.35

    axes[0].bar(x - width/2, yew_means, width, yerr=yew_stds,
                label='Yew Sites', color='green', alpha=0.7, capsize=5)
    axes[0].bar(x + width/2, no_yew_means, width, yerr=no_yew_stds,
                label='Non-Yew Sites', color='gray', alpha=0.7, capsize=5)

    axes[0].set_xlabel('Spectral Band', fontsize=12)
    axes[0].set_ylabel('Mean Reflectance Value', fontsize=12)
    axes[0].set_title('Average Spectral Signatures',
                      fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(band_names)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Compute NDVI for all images
    yew_ndvi = []
    for img in yew_images:
        nir = img[3].mean()
        red = img[2].mean()
        if (nir + red) > 0:
            ndvi = (nir - red) / (nir + red)
            yew_ndvi.append(ndvi)

    no_yew_ndvi = []
    for img in no_yew_images:
        nir = img[3].mean()
        red = img[2].mean()
        if (nir + red) > 0:
            ndvi = (nir - red) / (nir + red)
            no_yew_ndvi.append(ndvi)

    # NDVI distribution
    axes[1].hist(yew_ndvi, bins=20, alpha=0.7, label=f'Yew (n={len(yew_ndvi)})',
                 color='green', edgecolor='black')
    axes[1].hist(no_yew_ndvi, bins=20, alpha=0.7, label=f'Non-Yew (n={len(no_yew_ndvi)})',
                 color='gray', edgecolor='black')
    axes[1].set_xlabel('NDVI', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('NDVI Distribution', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Add statistics
    if len(yew_ndvi) > 0 and len(no_yew_ndvi) > 0:
        axes[1].axvline(np.mean(yew_ndvi), color='green', linestyle='--',
                        linewidth=2, label=f'Yew mean: {np.mean(yew_ndvi):.3f}')
        axes[1].axvline(np.mean(no_yew_ndvi), color='gray', linestyle='--',
                        linewidth=2, label=f'Non-Yew mean: {np.mean(no_yew_ndvi):.3f}')

    plt.tight_layout()

    save_path = Path('results/figures/ee_patches_spectral_analysis.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.show()


def main():
    print("="*80)
    print("DISPLAYING EARTH ENGINE IMAGE PATCHES")
    print("="*80)

    image_dir = 'data/ee_imagery/image_patches_64x64'

    if not Path(image_dir).exists():
        print(f"\n✗ Image directory not found: {image_dir}")
        print("  Run extract_ee_image_patches.py first")
        return

    load_and_display_patches(image_dir, n_yew=5, n_no_yew=10)

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
