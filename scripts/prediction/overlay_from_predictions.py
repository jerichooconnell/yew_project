#!/usr/bin/env python3
"""
Filter existing center-pixel predictions for a Vancouver Island bounding box
and create overlay visualizations (scatter + high-probability map).

Usage:
    python scripts/prediction/overlay_from_predictions.py --pred-csv <file> --output-dir <dir> \
        --lat-min 48.0 --lat-max 50.9 --lon-min -125.9 --lon-max -123.0

If --pred-csv is not provided, the script will pick the latest CSV in
`results/predictions/center_pixel_method`.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def find_latest_predictions(dirpath):
    p = Path(dirpath)
    files = sorted(p.glob('center_pixel_predictions_*.csv'))
    return files[-1] if files else None


def create_maps(pred_df, output_dir, bbox):
    lat_min, lat_max, lon_min, lon_max = bbox
    output_dir.mkdir(parents=True, exist_ok=True)

    # Colormap
    colormap_colors = [
        '#2166ac', '#4393c3', '#92c5de', '#d1e5f0',
        '#fddbc7', '#f4a582', '#d6604d', '#b2182b'
    ]
    cmap = LinearSegmentedColormap.from_list(
        'yew_prob', colormap_colors, N=256)

    # Scatter map
    fig, ax = plt.subplots(figsize=(12, 10))
    sc = ax.scatter(pred_df['longitude'], pred_df['latitude'], c=pred_df['yew_probability'],
                    cmap=cmap, s=30, marker='s', vmin=0, vmax=1, edgecolors='none')
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Center-pixel Yew Probabilities - Vancouver Island Subset')
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('P(Yew)')
    plt.tight_layout()
    out1 = output_dir / 'vancouver_island_center_pixel_map.png'
    fig.savefig(out1, dpi=200, bbox_inches='tight')
    plt.close(fig)

    # High-probability map (>0.7)
    hp = pred_df[pred_df['yew_probability'] > 0.7]
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    ax2.scatter(pred_df['longitude'], pred_df['latitude'],
                c='lightgray', s=10, alpha=0.4, marker='s')
    if len(hp) > 0:
        sc2 = ax2.scatter(hp['longitude'], hp['latitude'], c=hp['yew_probability'], cmap=cmap,
                          s=60, edgecolors='black', linewidths=0.3, vmin=0.7, vmax=1.0)
        cbar2 = plt.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label('P(Yew)')
    ax2.set_xlim(lon_min, lon_max)
    ax2.set_ylim(lat_min, lat_max)
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_title(f'High-probability Yew Locations (>0.7) - n={len(hp)}')
    plt.tight_layout()
    out2 = output_dir / 'vancouver_island_high_prob_map.png'
    fig2.savefig(out2, dpi=200, bbox_inches='tight')
    plt.close(fig2)

    # Distribution
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.hist(pred_df['yew_probability'], bins=40,
             color='tab:blue', edgecolor='k')
    ax3.set_xlabel('P(Yew)')
    ax3.set_ylabel('Count')
    ax3.set_title('Probability Distribution - Vancouver Island Subset')
    plt.tight_layout()
    out3 = output_dir / 'vancouver_island_probability_distribution.png'
    fig3.savefig(out3, dpi=150, bbox_inches='tight')
    plt.close(fig3)

    return out1, out2, out3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-csv', type=str, default=None)
    parser.add_argument('--pred-dir', type=str,
                        default='results/predictions/center_pixel_method')
    parser.add_argument('--output-dir', type=str,
                        default='results/predictions/vancouver_island')
    parser.add_argument('--lat-min', type=float, default=48.0)
    parser.add_argument('--lat-max', type=float, default=50.9)
    parser.add_argument('--lon-min', type=float, default=-125.9)
    parser.add_argument('--lon-max', type=float, default=-123.0)
    args = parser.parse_args()

    pred_file = Path(args.pred_csv) if args.pred_csv else find_latest_predictions(
        args.pred_dir)
    if pred_file is None or not pred_file.exists():
        print('No predictions CSV found. Run the center-pixel prediction script first.')
        return

    print(f'Loading predictions: {pred_file}')
    df = pd.read_csv(pred_file)

    bbox = (args.lat_min, args.lat_max, args.lon_min, args.lon_max)
    # Filter
    df_sub = df[(df['latitude'] >= bbox[0]) & (df['latitude'] <= bbox[1]) &
                (df['longitude'] >= bbox[2]) & (df['longitude'] <= bbox[3])].copy()

    print(f'Selected {len(df_sub)} predictions within bbox {bbox}')
    if len(df_sub) == 0:
        print('No points in bbox; try expanding the bbox or regenerate predictions for the area.')
        return

    out1, out2, out3 = create_maps(df_sub, Path(args.output_dir), bbox)
    print('Created maps:')
    print(' -', out1)
    print(' -', out2)
    print(' -', out3)


if __name__ == '__main__':
    main()
