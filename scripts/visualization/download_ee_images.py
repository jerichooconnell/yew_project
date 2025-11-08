#!/usr/bin/env python3
"""
Download and Display Earth Engine Thumbnail Images
==================================================

Downloads 64x64 pixel thumbnail images from Earth Engine URLs stored in CSV files
and creates visualizations comparing yew vs non-yew sites.

Author: GitHub Copilot
Date: November 7, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
from pathlib import Path
import time
from tqdm import tqdm
import ee
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials


def initialize_earth_engine():
    """Initialize Earth Engine with authentication."""
    try:
        ee.Initialize(project='carbon-storm-206002')
        print("✓ Earth Engine initialized")
        return True
    except Exception as e:
        print(f"✗ Earth Engine initialization failed: {e}")
        print("  Please run: earthengine authenticate")
        return False


def get_authenticated_session():
    """Create a requests session with Earth Engine authentication."""
    try:
        # Get credentials from Earth Engine
        credentials = ee.data.get_persistent_credentials()
        if credentials is None:
            print("✗ No Earth Engine credentials found")
            return None

        # Create a session with the credentials
        session = requests.Session()

        # For Earth Engine API, we need to use the access token
        # Try to get the token from the credentials
        if hasattr(credentials, 'token'):
            session.headers.update({
                'Authorization': f'Bearer {credentials.token}'
            })
        elif hasattr(credentials, 'access_token'):
            session.headers.update({
                'Authorization': f'Bearer {credentials.access_token}'
            })
        else:
            print("  Warning: Could not extract access token from credentials")
            print("  Attempting to use unauthenticated requests (may fail)")

        return session
    except Exception as e:
        print(f"  Warning: Could not create authenticated session: {e}")
        return requests.Session()


def download_image_from_url(url, session=None, max_retries=3, timeout=10):
    """
    Download image from Earth Engine URL.

    Args:
        url: Earth Engine thumbnail URL
        session: Requests session (with auth if available)
        max_retries: Number of retry attempts
        timeout: Request timeout in seconds

    Returns:
        PIL Image object or None if failed
    """
    if session is None:
        session = requests.Session()

    for attempt in range(max_retries):
        try:
            response = session.get(url, timeout=timeout)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                return img
            else:
                if attempt == 0:  # Only print once
                    print(
                        f"  Status {response.status_code} for URL: {url[:80]}...")
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait before retry
            else:
                print(f"  Failed after {max_retries} attempts: {str(e)[:50]}")
    return None


def load_ee_data_with_urls():
    """Load Earth Engine data that contains image URLs."""
    print("Loading Earth Engine data with image URLs...")

    # Try different CSV files that might have image URLs
    csv_files = [
        'data/ee_imagery/temp_extraction_patch_100.csv',
        'data/ee_imagery/temp_extraction_patch_50.csv',
        'data/ee_imagery/ee_extraction_progress.csv'
    ]

    for csv_file in csv_files:
        csv_path = Path(csv_file)
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, low_memory=False)
                if 'image_url' in df.columns:
                    print(f"✓ Loaded {csv_file}")
                    print(f"  Found {len(df)} records with image URLs")
                    return df, csv_file
            except Exception as e:
                print(f"  Error loading {csv_file}: {e}")

    print("✗ No CSV files with 'image_url' column found!")
    return None, None


def get_yew_labels(df):
    """
    Determine which plots have yew based on available data.

    Args:
        df: DataFrame with Earth Engine data

    Returns:
        Series with boolean yew labels
    """
    # Load inventory data to get yew labels
    inv_path = Path('data/processed/bc_sample_data_deduplicated.csv')
    if inv_path.exists():
        print("Loading yew labels from inventory data...")
        inv_df = pd.read_csv(inv_path, low_memory=False)

        # Parse yew presence
        import re

        def parse_yew(composition_string):
            if not composition_string or pd.isna(composition_string):
                return False
            pattern = r'TW(\d{2,3})'
            match = re.search(pattern, str(composition_string))
            return match is not None

        inv_df['has_yew'] = inv_df['SPB_CPCT_LS'].apply(parse_yew)
        inv_df['SITE_IDENTIFIER'] = inv_df['SITE_IDENTIFIER'].astype(str)

        # Merge with EE data
        df['plot_id'] = df['plot_id'].astype(str)
        df = df.merge(inv_df[['SITE_IDENTIFIER', 'has_yew']],
                      left_on='plot_id', right_on='SITE_IDENTIFIER', how='left')

        print(f"  Labeled {df['has_yew'].sum()} yew plots")
        return df
    else:
        print("  Warning: Could not load inventory data, using is_active_site flag")
        df['has_yew'] = df.get('is_active_site', False)
        return df


def download_sample_images(df, session=None, n_yew=10, n_no_yew=10):
    """
    Download sample images for yew and non-yew sites.

    Args:
        df: DataFrame with image URLs and yew labels
        session: Authenticated requests session
        n_yew: Number of yew images to download
        n_no_yew: Number of non-yew images to download

    Returns:
        Tuple of (yew_images, no_yew_images, yew_metadata, no_yew_metadata)
    """
    # Filter valid URLs
    df_valid = df[df['image_url'].notna(
    ) & df['image_url'].str.startswith('http')].copy()

    if 'has_yew' not in df_valid.columns:
        df_valid = get_yew_labels(df_valid)

    # Sample yew and non-yew sites
    yew_sites = df_valid[df_valid['has_yew'] == True].head(n_yew)
    no_yew_sites = df_valid[df_valid['has_yew'] == False].head(n_no_yew)

    print(f"\nDownloading {len(yew_sites)} yew site images...")
    yew_images = []
    yew_metadata = []
    for idx, row in tqdm(yew_sites.iterrows(), total=len(yew_sites)):
        img = download_image_from_url(row['image_url'], session=session)
        if img is not None:
            yew_images.append(img)
            yew_metadata.append({
                'plot_id': row['plot_id'],
                'ndvi': row.get('ndvi', None),
                'elevation': row.get('elevation', None)
            })
        time.sleep(0.2)  # Be nice to the server

    print(f"\nDownloading {len(no_yew_sites)} non-yew site images...")
    no_yew_images = []
    no_yew_metadata = []
    for idx, row in tqdm(no_yew_sites.iterrows(), total=len(no_yew_sites)):
        img = download_image_from_url(row['image_url'], session=session)
        if img is not None:
            no_yew_images.append(img)
            no_yew_metadata.append({
                'plot_id': row['plot_id'],
                'ndvi': row.get('ndvi', None),
                'elevation': row.get('elevation', None)
            })
        time.sleep(0.2)

    print(
        f"\n✓ Downloaded {len(yew_images)} yew images and {len(no_yew_images)} non-yew images")
    return yew_images, no_yew_images, yew_metadata, no_yew_metadata


def display_image_grid(images, metadata, title, save_path, n_cols=5):
    """
    Display a grid of images with metadata.

    Args:
        images: List of PIL Image objects
        metadata: List of metadata dictionaries
        title: Plot title
        save_path: Path to save the figure
        n_cols: Number of columns in grid
    """
    if len(images) == 0:
        print(f"No images to display for {title}")
        return

    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx in range(n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        if idx < n_images:
            ax.imshow(images[idx])
            ax.axis('off')

            # Add metadata as title
            meta = metadata[idx]
            label = f"ID: {meta['plot_id']}"
            if meta.get('ndvi') is not None:
                label += f"\nNDVI: {meta['ndvi']:.3f}"
            if meta.get('elevation') is not None:
                label += f"\nElev: {meta['elevation']:.0f}m"

            ax.set_title(label, fontsize=9)
        else:
            ax.axis('off')

    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.show()


def create_comparison_figure(yew_images, no_yew_images, yew_meta, no_yew_meta, save_path):
    """Create side-by-side comparison of yew vs non-yew images."""
    n_samples = min(5, len(yew_images), len(no_yew_images))

    if n_samples == 0:
        print("Not enough images for comparison")
        return

    fig, axes = plt.subplots(2, n_samples, figsize=(3*n_samples, 7))

    # Yew images on top
    for i in range(n_samples):
        axes[0, i].imshow(yew_images[i])
        axes[0, i].axis('off')
        meta = yew_meta[i]
        label = f"Yew Site\nID: {meta['plot_id']}"
        if meta.get('ndvi') is not None:
            label += f"\nNDVI: {meta['ndvi']:.3f}"
        axes[0, i].set_title(label, fontsize=10,
                             color='green', fontweight='bold')

    # Non-yew images on bottom
    for i in range(n_samples):
        axes[1, i].imshow(no_yew_images[i])
        axes[1, i].axis('off')
        meta = no_yew_meta[i]
        label = f"Non-Yew Site\nID: {meta['plot_id']}"
        if meta.get('ndvi') is not None:
            label += f"\nNDVI: {meta['ndvi']:.3f}"
        axes[1, i].set_title(label, fontsize=10)

    plt.suptitle('Sentinel-2 Imagery: Yew vs Non-Yew Forest Sites',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.show()


def compute_average_image(images):
    """Compute average image from a list of PIL images."""
    if len(images) == 0:
        return None

    arrays = [np.array(img) for img in images]
    avg_array = np.mean(arrays, axis=0).astype(np.uint8)
    return Image.fromarray(avg_array)


def main():
    """Main execution function."""
    print("=" * 80)
    print("EARTH ENGINE IMAGE DOWNLOAD AND VISUALIZATION")
    print("=" * 80)

    # Initialize Earth Engine
    print("\nInitializing Earth Engine...")
    if not initialize_earth_engine():
        print("\n✗ Cannot proceed without Earth Engine authentication")
        print("  Run: earthengine authenticate")
        return

    # Get authenticated session
    print("Creating authenticated session...")
    session = get_authenticated_session()

    # Create output directory
    output_dir = Path('results/figures/ee_thumbnails')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df, csv_file = load_ee_data_with_urls()
    if df is None:
        print("\n✗ Cannot proceed without image URLs")
        return

    # Label yew sites
    df = get_yew_labels(df)

    # Download sample images
    yew_imgs, no_yew_imgs, yew_meta, no_yew_meta = download_sample_images(
        df, session=session, n_yew=15, n_no_yew=15
    )

    if len(yew_imgs) == 0 and len(no_yew_imgs) == 0:
        print("\n✗ No images downloaded successfully")
        print("  Note: Earth Engine thumbnail URLs may require additional authentication")
        print("  The URLs might have expired or need re-generation")
        return

    # Create visualizations
    print("\nCreating visualizations...")

    if len(yew_imgs) > 0:
        display_image_grid(
            yew_imgs, yew_meta,
            'Pacific Yew Sites - Sentinel-2 Imagery',
            output_dir / 'yew_sites_grid.png',
            n_cols=5
        )

    if len(no_yew_imgs) > 0:
        display_image_grid(
            no_yew_imgs, no_yew_meta,
            'Non-Yew Sites - Sentinel-2 Imagery',
            output_dir / 'no_yew_sites_grid.png',
            n_cols=5
        )

    # Comparison figure
    create_comparison_figure(
        yew_imgs, no_yew_imgs, yew_meta, no_yew_meta,
        output_dir / 'yew_vs_no_yew_comparison.png'
    )

    # Compute average images
    if len(yew_imgs) > 0:
        avg_yew = compute_average_image(yew_imgs)
        if avg_yew:
            avg_yew.save(output_dir / 'average_yew_site.png')
            print(f"✓ Saved: {output_dir / 'average_yew_site.png'}")

    if len(no_yew_imgs) > 0:
        avg_no_yew = compute_average_image(no_yew_imgs)
        if avg_no_yew:
            avg_no_yew.save(output_dir / 'average_no_yew_site.png')
            print(f"✓ Saved: {output_dir / 'average_no_yew_site.png'}")

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print(f"Images saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
