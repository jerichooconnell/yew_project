#!/usr/bin/env python3
"""
Generate Spatial Predictions Using Center Pixel Classification

This script uses the trained center pixel classifier to generate predictions
across the entire Jordan River area and creates visualization overlay maps.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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

    try:
        img = np.load(emb_path)  # Shape: (64, 64, 64)
        center = patch_size // 2

        # Extract single center pixel
        center_pixel = img[:, center, center]
        return center_pixel
    except Exception as e:
        print(f'Error loading {emb_path}: {e}')
        return None


def extract_features_from_split(df, emb_dir):
    """
    Extract center pixel features and labels from a data split.
    """
    features = []
    labels = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc='Extracting features'):
        if pd.notna(row['lat']) and pd.notna(row['lon']):
            center_data = extract_center_pixel(row['lat'], row['lon'], emb_dir)

            if center_data is not None:
                features.append(center_data)
                labels.append(int(row['has_yew']))

    return np.array(features), np.array(labels)


def get_all_embedding_coordinates(emb_dir):
    """
    Get all lat/lon coordinates from embedding files.
    """
    emb_dir = Path(emb_dir)
    coords = []

    for emb_file in emb_dir.glob('embedding_*.npy'):
        # Parse filename: embedding_lat_lon.npy
        parts = emb_file.stem.split('_')
        if len(parts) == 3:
            try:
                lat = float(parts[1])
                lon = float(parts[2])
                coords.append((lat, lon))
            except ValueError:
                continue

    return coords


def predict_all_locations(coords, emb_dir, model, scaler):
    """
    Generate predictions for all coordinates.
    """
    predictions = []

    for lat, lon in tqdm(coords, desc='Generating predictions'):
        center_pixel = extract_center_pixel(lat, lon, emb_dir)

        if center_pixel is not None:
            # Handle inf/nan values
            center_pixel = np.nan_to_num(
                center_pixel, nan=0.0, posinf=0.0, neginf=0.0)

            # Reshape and scale
            X = center_pixel.reshape(1, -1)
            X_scaled = scaler.transform(X)

            # Predict
            prob = model.predict_proba(X_scaled)[0, 1]

            predictions.append({
                'latitude': lat,
                'longitude': lon,
                'yew_probability': prob
            })

    return pd.DataFrame(predictions)


def create_overlay_visualization(pred_df, tile_cache_dir, output_dir):
    """
    Create overlay visualization using scatter plot approach.
    """
    print('\nCreating overlay visualization...')

    # Get bounds
    lat_min, lat_max = pred_df['latitude'].min(), pred_df['latitude'].max()
    lon_min, lon_max = pred_df['longitude'].min(), pred_df['longitude'].max()

    print(f'Region: {lat_min:.4f}°N to {lat_max:.4f}°N')
    print(f'        {lon_min:.4f}°W to {lon_max:.4f}°W')
    print(f'Total predictions: {len(pred_df)}')

    print(f'Region: {lat_min:.4f}°N to {lat_max:.4f}°N')
    print(f'        {lon_min:.4f}°W to {lon_max:.4f}°W')
    print(f'Total predictions: {len(pred_df)}')

    # Create visualization
    print('Creating scatter plot visualization...')

    # Custom colormap
    colormap_colors = [
        '#2166ac', '#4393c3', '#92c5de', '#d1e5f0',
        '#fddbc7', '#f4a582', '#d6604d', '#b2182b'
    ]
    cmap = LinearSegmentedColormap.from_list(
        'yew_prob', colormap_colors, N=256)

    # Create scatter plot with probabilities
    fig, ax = plt.subplots(figsize=(16, 12))

    # Scatter plot colored by probability
    scatter = ax.scatter(pred_df['longitude'], pred_df['latitude'],
                         c=pred_df['yew_probability'], cmap=cmap,
                         s=20, alpha=0.8, vmin=0, vmax=1,
                         edgecolors='none', marker='s')

    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Latitude', fontsize=14)
    ax.set_title('Yew Probability Map (Center Pixel Classification)\nJordan River Area, Southern Vancouver Island',
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, color='gray', linewidth=0.5)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Probability of Yew Presence', fontsize=12)

    # Statistics box
    lat_range_km = (lat_max - lat_min) * 111
    lon_range_km = (lon_max - lon_min) * 111 * \
        np.cos(np.radians((lat_max + lat_min) / 2))

    stats_text = (
        f'Region: {lat_min:.4f}°N - {lat_max:.4f}°N\n'
        f'        {lon_min:.4f}°W - {lon_max:.4f}°W\n'
        f'\n'
        f'Predictions: {len(pred_df)}\n'
        f'Area: ~{lat_range_km:.1f}km × {lon_range_km:.1f}km\n'
        f'\n'
        f'Mean prob: {pred_df["yew_probability"].mean():.3f}\n'
        f'Std dev: {pred_df["yew_probability"].std():.3f}\n'
        f'\n'
        f'High (>0.7): {(pred_df["yew_probability"] > 0.7).sum()} '
        f'({(pred_df["yew_probability"] > 0.7).sum()/len(pred_df)*100:.1f}%)\n'
        f'Med (0.3-0.7): {((pred_df["yew_probability"] >= 0.3) & (pred_df["yew_probability"] <= 0.7)).sum()} '
        f'({((pred_df["yew_probability"] >= 0.3) & (pred_df["yew_probability"] <= 0.7)).sum()/len(pred_df)*100:.1f}%)\n'
        f'Low (<0.3): {(pred_df["yew_probability"] < 0.3).sum()} '
        f'({(pred_df["yew_probability"] < 0.3).sum()/len(pred_df)*100:.1f}%)\n'
        f'\n'
        f'Method: Center Pixel (64-channel)\n'
        f'Classifier: SVM (RBF kernel)'
    )

    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
            fontsize=9,
            family='monospace')

    plt.tight_layout()

    # Save main visualization
    output_path = output_dir / 'center_pixel_map.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved map visualization to: {output_path}')
    plt.close()

    # High probability locations map
    high_prob_df = pred_df[pred_df['yew_probability'] > 0.7].copy()

    if len(high_prob_df) > 0:
        fig2, ax2 = plt.subplots(figsize=(16, 12))

        # All locations in light gray
        ax2.scatter(pred_df['longitude'], pred_df['latitude'],
                    c='lightgray', s=10, alpha=0.3, edgecolors='none', marker='s',
                    label='All locations')

        # High probability locations colored
        scatter2 = ax2.scatter(high_prob_df['longitude'], high_prob_df['latitude'],
                               c=high_prob_df['yew_probability'], cmap=cmap,
                               s=40, alpha=0.9, vmin=0.7, vmax=1.0,
                               edgecolors='black', linewidths=0.5, marker='o',
                               label=f'High probability (n={len(high_prob_df)})')

        ax2.set_xlabel('Longitude', fontsize=14)
        ax2.set_ylabel('Latitude', fontsize=14)
        ax2.set_title('High Probability Yew Locations (>0.7)\nJordan River Area',
                      fontsize=16, fontweight='bold')
        ax2.grid(True, alpha=0.3, color='gray', linewidth=0.5)
        ax2.set_xlim(lon_min, lon_max)
        ax2.set_ylim(lat_min, lat_max)
        ax2.legend(loc='upper right')

        cbar2 = plt.colorbar(scatter2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label('Yew Probability', fontsize=12)

        plt.tight_layout()

        output_path_high = output_dir / 'center_pixel_high_probability_map.png'
        fig2.savefig(output_path_high, dpi=300, bbox_inches='tight')
        print(f'Saved high probability map to: {output_path_high}')
        plt.close()

    # Distribution plots
    fig2, (ax_hist, ax_box) = plt.subplots(1, 2, figsize=(14, 5))

    ax_hist.hist(pred_df['yew_probability'], bins=50,
                 edgecolor='black', alpha=0.7)
    ax_hist.axvline(pred_df['yew_probability'].mean(), color='red',
                    linestyle='--', linewidth=2, label=f'Mean: {pred_df["yew_probability"].mean():.3f}')
    ax_hist.axvline(pred_df['yew_probability'].median(), color='orange',
                    linestyle='--', linewidth=2, label=f'Median: {pred_df["yew_probability"].median():.3f}')
    ax_hist.set_xlabel('Yew Probability', fontsize=12)
    ax_hist.set_ylabel('Frequency', fontsize=12)
    ax_hist.set_title(
        'Distribution of Predicted Probabilities (Center Pixel)', fontsize=14, fontweight='bold')
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)

    ax_box.boxplot([pred_df['yew_probability']], vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7))
    ax_box.set_ylabel('Yew Probability', fontsize=12)
    ax_box.set_title('Probability Distribution',
                     fontsize=14, fontweight='bold')
    ax_box.set_xticklabels(['All Predictions'])
    ax_box.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path2 = output_dir / 'center_pixel_probability_distribution.png'
    fig2.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f'Saved distribution plot to: {output_path2}')
    plt.close()


def main(args):
    """
    Main function to generate spatial predictions and visualizations.
    """
    print('='*80)
    print('CENTER PIXEL CLASSIFICATION - SPATIAL PREDICTION MAP')
    print('='*80)

    # Load training data to train the model
    print('\nLoading training data...')
    train_df = pd.read_csv(args.train_path)
    val_df = pd.read_csv(args.val_path)

    embedding_dir = Path(args.embedding_dir)

    # Extract features from training data
    print('\nExtracting training features...')
    X_train, y_train = extract_features_from_split(train_df, embedding_dir)
    X_val, y_val = extract_features_from_split(val_df, embedding_dir)

    # Combine train and validation for final model
    X_all = np.vstack([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])

    print(f'\nTotal training samples: {len(X_all)}')
    print(f'  Yew: {y_all.sum()}')
    print(f'  Non-yew: {len(y_all) - y_all.sum()}')

    # Handle inf/nan values
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    # Train final model on all available data
    print('\nTraining final SVM classifier on all data...')
    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)

    model = SVC(kernel='rbf', probability=True, random_state=42)
    model.fit(X_all_scaled, y_all)

    print('Model trained successfully!')

    # Get all embedding coordinates
    print('\nFinding all embedding coordinates...')
    all_coords = get_all_embedding_coordinates(embedding_dir)
    print(f'Found {len(all_coords)} embedding locations')

    # Generate predictions for all locations
    print('\nGenerating predictions for entire area...')
    pred_df = predict_all_locations(all_coords, embedding_dir, model, scaler)

    print(f'\nGenerated {len(pred_df)} predictions')
    print(f'Mean probability: {pred_df["yew_probability"].mean():.3f}')
    print(f'Std dev: {pred_df["yew_probability"].std():.3f}')
    print(
        f'Range: [{pred_df["yew_probability"].min():.3f}, {pred_df["yew_probability"].max():.3f}]')

    # Save predictions
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pred_file = output_dir / f'center_pixel_predictions_{timestamp}.csv'
    pred_df.to_csv(pred_file, index=False)
    print(f'\nSaved predictions to: {pred_file}')

    # Create visualizations
    tile_cache_dir = Path(args.tile_cache_dir) if args.tile_cache_dir else None
    create_overlay_visualization(pred_df, tile_cache_dir, output_dir)

    # High probability locations
    high_prob = pred_df[pred_df['yew_probability'] > 0.8].sort_values(
        'yew_probability', ascending=False)

    if len(high_prob) > 0:
        print(f'\n{len(high_prob)} high-probability locations (>0.8):')
        print(high_prob[['latitude', 'longitude', 'yew_probability']].head(
            10).to_string(index=False))

        high_prob_path = output_dir / 'high_probability_locations.csv'
        high_prob.to_csv(high_prob_path, index=False)
        print(f'\nSaved high-probability locations to: {high_prob_path}')

    print('\n' + '='*80)
    print('SPATIAL PREDICTION MAP GENERATION COMPLETE')
    print('='*80)
    print(f'\nOutput files:')
    print(f'  - {pred_file}')
    print(f'  - {output_dir / "center_pixel_map.png"}')
    print(f'  - {output_dir / "center_pixel_probability_distribution.png"}')
    if len(high_prob) > 0:
        print(f'  - {output_dir / "center_pixel_high_probability_map.png"}')
        print(f'  - {output_dir / "high_probability_locations.csv"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate spatial predictions using center pixel classification'
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
        '--tile-cache-dir',
        type=str,
        default='results/predictions/southern_vancouver_island/tile_cache',
        help='Directory containing cached satellite tiles for visualization'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/predictions/center_pixel_method',
        help='Output directory for predictions and visualizations'
    )

    args = parser.parse_args()
    main(args)
