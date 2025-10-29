#!/usr/bin/env python3
"""
Map Plot Locations in British Columbia
========================================

Creates visualizations showing where the forest plots are located
and their elevation characteristics.

Author: Analysis Tool
Date: October 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Try to import cartopy for BC map
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("Note: cartopy not installed, using simple matplotlib plots")


def plot_bc_locations_simple(df, output_path='results/figures/bc_plot_locations.png'):
    """
    Create a simple scatter plot of plot locations with elevation.
    """
    print("\nCreating BC plot location maps...")

    # Create figure with multiple panels
    fig = plt.figure(figsize=(18, 12))

    # Panel 1: All plots with elevation
    ax1 = plt.subplot(2, 3, 1)
    scatter = ax1.scatter(df['lon'], df['lat'],
                          c=df['elevation'],
                          cmap='terrain',
                          s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Longitude', fontsize=11)
    ax1.set_ylabel('Latitude', fontsize=11)
    ax1.set_title('Plot Locations Colored by Elevation',
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Elevation (m)', fontsize=10)

    # Add BC boundaries (approximate)
    ax1.set_xlim(-140, -114)
    ax1.set_ylim(48, 60)

    # Panel 2: Elevation histogram
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(df['elevation'], bins=30, color='brown',
             alpha=0.7, edgecolor='black')
    ax2.axvline(df['elevation'].mean(), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {df["elevation"].mean():.0f} m')
    ax2.axvline(df['elevation'].median(), color='blue', linestyle='--',
                linewidth=2, label=f'Median: {df["elevation"].median():.0f} m')
    ax2.set_xlabel('Elevation (m)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Elevation Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Panel 3: Elevation box plot by region
    ax3 = plt.subplot(2, 3, 3)

    # Define regions by latitude
    df_copy = df.copy()
    df_copy['region'] = pd.cut(df_copy['lat'],
                               bins=[0, 50, 53, 56, 100],
                               labels=['South\n(< 50°N)', 'Central\n(50-53°N)',
                                       'North\n(53-56°N)', 'Far North\n(> 56°N)'])

    regions = ['South\n(< 50°N)', 'Central\n(50-53°N)',
               'North\n(53-56°N)', 'Far North\n(> 56°N)']
    data_to_plot = [df_copy[df_copy['region'] == r]
                    ['elevation'].dropna() for r in regions]

    bp = ax3.boxplot(data_to_plot, labels=regions, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    ax3.set_ylabel('Elevation (m)', fontsize=11)
    ax3.set_title('Elevation by Region', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Panel 4: Latitude vs Elevation
    ax4 = plt.subplot(2, 3, 4)
    scatter = ax4.scatter(df['lat'], df['elevation'],
                          c=df['lon'], cmap='viridis',
                          s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax4.set_xlabel('Latitude (°N)', fontsize=11)
    ax4.set_ylabel('Elevation (m)', fontsize=11)
    ax4.set_title('Elevation vs Latitude', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Longitude', fontsize=10)

    # Add trend line
    z = np.polyfit(df['lat'], df['elevation'], 1)
    p = np.poly1d(z)
    ax4.plot(df['lat'], p(df['lat']), "r--",
             alpha=0.8, linewidth=2, label='Trend')
    ax4.legend()

    # Panel 5: Longitude vs Elevation
    ax5 = plt.subplot(2, 3, 5)
    scatter = ax5.scatter(df['lon'], df['elevation'],
                          c=df['lat'], cmap='plasma',
                          s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax5.set_xlabel('Longitude (°W)', fontsize=11)
    ax5.set_ylabel('Elevation (m)', fontsize=11)
    ax5.set_title('Elevation vs Longitude', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax5)
    cbar.set_label('Latitude', fontsize=10)

    # Panel 6: Regional statistics table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('tight')
    ax6.axis('off')

    # Create statistics table
    stats_data = []
    for region in regions:
        region_data = df_copy[df_copy['region'] == region]
        if len(region_data) > 0:
            stats_data.append([
                region.replace('\n', ' '),
                len(region_data),
                f"{region_data['elevation'].mean():.0f}",
                f"{region_data['elevation'].min():.0f}",
                f"{region_data['elevation'].max():.0f}"
            ])

    table = ax6.table(cellText=stats_data,
                      colLabels=[
                          'Region', 'Count', 'Mean\nElev (m)', 'Min\nElev (m)', 'Max\nElev (m)'],
                      cellLoc='center',
                      loc='center',
                      bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(stats_data) + 1):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E8F5E9')

    ax6.set_title('Regional Statistics', fontsize=12,
                  fontweight='bold', pad=20)

    # Overall title
    fig.suptitle(f'BC Forest Plot Locations - Elevation Analysis (n={len(df)} plots)',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")

    return df_copy


def create_elevation_summary(df):
    """
    Print detailed elevation statistics.
    """
    print("\n" + "="*70)
    print("ELEVATION STATISTICS")
    print("="*70)

    print(f"\nOverall Statistics (n={len(df)}):")
    print(f"  Mean elevation: {df['elevation'].mean():.1f} m")
    print(f"  Median elevation: {df['elevation'].median():.1f} m")
    print(f"  Std deviation: {df['elevation'].std():.1f} m")
    print(
        f"  Range: {df['elevation'].min():.0f} - {df['elevation'].max():.0f} m")

    print(f"\nPercentiles:")
    for p in [10, 25, 50, 75, 90]:
        print(f"  {p}th percentile: {df['elevation'].quantile(p/100):.0f} m")

    print(f"\nElevation Categories:")
    bins = [0, 500, 1000, 1500, 2000, 5000]
    labels = ['Low (< 500m)', 'Mid (500-1000m)', 'High (1000-1500m)',
              'Very High (1500-2000m)', 'Alpine (> 2000m)']
    df['elev_category'] = pd.cut(df['elevation'], bins=bins, labels=labels)
    print(df['elev_category'].value_counts().sort_index())

    print(f"\nGeographic Distribution:")
    # South BC (< 50°N)
    south = df[df['lat'] < 50]
    if len(south) > 0:
        print(
            f"  South BC (< 50°N): {len(south)} plots, elevation {south['elevation'].mean():.0f} ± {south['elevation'].std():.0f} m")

    # Central BC (50-53°N)
    central = df[(df['lat'] >= 50) & (df['lat'] < 53)]
    if len(central) > 0:
        print(
            f"  Central BC (50-53°N): {len(central)} plots, elevation {central['elevation'].mean():.0f} ± {central['elevation'].std():.0f} m")

    # North BC (53-56°N)
    north = df[(df['lat'] >= 53) & (df['lat'] < 56)]
    if len(north) > 0:
        print(
            f"  North BC (53-56°N): {len(north)} plots, elevation {north['elevation'].mean():.0f} ± {north['elevation'].std():.0f} m")

    # Far North BC (> 56°N)
    far_north = df[df['lat'] >= 56]
    if len(far_north) > 0:
        print(
            f"  Far North BC (> 56°N): {len(far_north)} plots, elevation {far_north['elevation'].mean():.0f} ± {far_north['elevation'].std():.0f} m")

    print("\n" + "="*70)


def main():
    """
    Main execution.
    """
    print("="*70)
    print("BC PLOT LOCATION MAPPER")
    print("="*70)

    # Find the most recent extraction file
    ee_dir = Path('data/ee_imagery')
    csv_files = sorted(ee_dir.glob('ee_batch_*.csv'),
                       key=lambda x: x.stat().st_mtime, reverse=True)

    if not csv_files:
        print("No extraction files found! Run extract_ee_imagery_fast.py first.")
        return

    # Load most recent
    latest_file = csv_files[0]
    print(f"\nLoading data from: {latest_file}")
    df = pd.read_csv(latest_file)

    print(f"Loaded {len(df)} plots")
    print(f"\nCoordinate ranges:")
    print(f"  Latitude: {df['lat'].min():.2f}° to {df['lat'].max():.2f}°")
    print(f"  Longitude: {df['lon'].min():.2f}° to {df['lon'].max():.2f}°")

    # Filter to valid elevation data
    df_valid = df[df['elevation'].notna()].copy()
    print(f"\nPlots with valid elevation: {len(df_valid)}")

    # Create elevation summary
    create_elevation_summary(df_valid)

    # Create visualizations
    df_with_regions = plot_bc_locations_simple(df_valid)

    # Save processed data with regions
    output_path = Path('data/processed/ee_data_with_regions.csv')
    df_with_regions.to_csv(output_path, index=False)
    print(f"\n✓ Saved data with regions to {output_path}")

    print("\n" + "="*70)
    print("MAPPING COMPLETE")
    print("="*70)
    print(f"\nVisualization saved to: results/figures/bc_plot_locations.png")
    print("Review the map to verify plot locations and elevation distributions.")


if __name__ == "__main__":
    main()
