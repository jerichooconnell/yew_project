#!/usr/bin/env python3
"""
Convert .npy image files to PNG format
======================================

Converts 4-band (B,G,R,NIR) Sentinel-2 .npy files to PNG images.
Creates both RGB (true color) and false color (NIR,R,G) versions.

Author: GitHub Copilot
Date: November 7, 2025
"""

import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def normalize_band(band, percentile_clip=2):
    """
    Normalize a band to 0-255 range with percentile clipping.

    Args:
        band: 2D numpy array
        percentile_clip: Percentile for clipping outliers

    Returns:
        Normalized band as uint8
    """
    # Clip outliers
    p_low = np.percentile(band, percentile_clip)
    p_high = np.percentile(band, 100 - percentile_clip)

    band_clipped = np.clip(band, p_low, p_high)

    # Normalize to 0-255
    band_min = band_clipped.min()
    band_max = band_clipped.max()

    if band_max > band_min:
        normalized = ((band_clipped - band_min) / (band_max - band_min) * 255)
    else:
        normalized = np.zeros_like(band_clipped)

    return normalized.astype(np.uint8)


def convert_npy_to_png(npy_path, output_dir, create_false_color=True):
    """
    Convert a single .npy file to PNG.

    Args:
        npy_path: Path to .npy file
        output_dir: Directory to save PNG files
        create_false_color: If True, also create false color (NIR) version

    Returns:
        Tuple of (rgb_path, false_color_path) or None if failed
    """
    try:
        # Load image: shape (4, 64, 64) = (Blue, Green, Red, NIR)
        img = np.load(npy_path)

        if img.shape[0] != 4 or img.ndim != 3:
            print(f"Skipping {npy_path.name}: unexpected shape {img.shape}")
            return None

        blue = img[0]
        green = img[1]
        red = img[2]
        nir = img[3]

        # Normalize each band
        blue_norm = normalize_band(blue)
        green_norm = normalize_band(green)
        red_norm = normalize_band(red)
        nir_norm = normalize_band(nir)

        # Create RGB image (true color)
        rgb_array = np.stack([red_norm, green_norm, blue_norm], axis=2)
        rgb_img = Image.fromarray(rgb_array, mode='RGB')

        # Save RGB
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = npy_path.stem
        rgb_path = output_dir / f"{stem}_rgb.png"
        rgb_img.save(rgb_path)

        false_color_path = None
        if create_false_color:
            # Create false color (NIR, Red, Green) - highlights vegetation
            false_color_array = np.stack(
                [nir_norm, red_norm, green_norm], axis=2)
            false_color_img = Image.fromarray(false_color_array, mode='RGB')

            false_color_path = output_dir / f"{stem}_false_color.png"
            false_color_img.save(false_color_path)

        return rgb_path, false_color_path

    except Exception as e:
        print(f"Error converting {npy_path.name}: {e}")
        return None


def convert_all_npy_files(input_dir, output_dir=None, create_false_color=True):
    """
    Convert all .npy files in a directory to PNG.

    Args:
        input_dir: Directory containing .npy files
        output_dir: Directory to save PNG files (default: input_dir/png)
        create_false_color: If True, create false color versions
    """
    input_dir = Path(input_dir)

    if output_dir is None:
        output_dir = input_dir / 'png'
    else:
        output_dir = Path(output_dir)

    # Find all .npy files
    npy_files = list(input_dir.glob('*.npy'))

    if len(npy_files) == 0:
        print(f"No .npy files found in {input_dir}")
        return

    print(f"Found {len(npy_files)} .npy files")
    print(f"Output directory: {output_dir}")
    print(
        f"Creating {'RGB + False Color' if create_false_color else 'RGB only'} images")
    print()

    success_count = 0

    for npy_file in tqdm(npy_files, desc="Converting"):
        result = convert_npy_to_png(npy_file, output_dir, create_false_color)
        if result is not None:
            success_count += 1

    print()
    print(f"✓ Converted {success_count}/{len(npy_files)} files")
    print(f"✓ Saved to: {output_dir}")


def main():
    """Main entry point."""
    print("="*80)
    print("CONVERTING NPY FILES TO PNG")
    print("="*80)
    print()

    # Convert yew images
    yew_dir = Path('data/ee_imagery/image_patches_64x64/yew')

    if yew_dir.exists():
        print("Converting yew images...")
        convert_all_npy_files(
            yew_dir,
            output_dir=yew_dir / 'png',
            create_false_color=True
        )
    else:
        print(f"✗ Directory not found: {yew_dir}")

    print()
    print("✓ Done!")


if __name__ == '__main__':
    main()
