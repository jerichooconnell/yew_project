#!/usr/bin/env python3
"""
Create an interactive HTML map viewer for yew probability predictions.
Uses folium for interactive visualization with zoom, pan, and hover info.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import folium
from folium.plugins import HeatMap, MarkerCluster
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import argparse


# Enhanced colormap matching the KMZ version
YEWCMAP_VISIBLE = LinearSegmentedColormap.from_list(
    'yew_visible',
    [
        (0.00, (0.90, 0.90, 0.90, 0.00)),   # P=0   → transparent gray
        (0.05, (0.60, 0.80, 0.60, 0.60)),   # P=0.05 → light green
        (0.10, (0.30, 0.70, 0.30, 0.75)),   # P=0.10 → green
        (0.30, (0.50, 0.85, 0.10, 0.85)),   # P=0.30 → yellow-green
        (0.50, (1.00, 0.90, 0.00, 0.90)),   # P=0.50 → yellow
        (0.70, (0.90, 0.40, 0.10, 0.95)),   # P=0.70 → orange
        (1.00, (0.70, 0.00, 0.50, 0.95)),   # P=1.0  → purple-red
    ],
    N=256,
)


def prob_to_color(prob):
    """Convert probability to hex color using colormap."""
    rgba = YEWCMAP_VISIBLE(prob)
    r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
    return f'#{r:02x}{g:02x}{b:02x}'


def prob_to_opacity(prob):
    """Convert probability to opacity."""
    rgba = YEWCMAP_VISIBLE(prob)
    return rgba[3]


def create_interactive_map(predictions_csv, output_html, sample_rate=1.0, threshold=0.01, 
                          show_training=True):
    """
    Create interactive HTML map from prediction CSV.
    
    Args:
        predictions_csv: Path to CSV with lat, lon, prob columns
        output_html: Path for output HTML file
        sample_rate: Fraction of points to include (1.0 = all)
        threshold: Minimum probability to display
        show_training: Whether to include training data on map
    """
    print(f'Loading predictions from {predictions_csv}...')
    df = pd.read_csv(predictions_csv)
    print(f'  Loaded {len(df):,} points')
    
    # Filter by threshold
    df = df[df['prob'] >= threshold].copy()
    print(f'  After P≥{threshold} filter: {len(df):,} points')
    
    # Sample if requested
    if sample_rate < 1.0:
        df = df.sample(frac=sample_rate, random_state=42)
        print(f'  After {sample_rate:.1%} sampling: {len(df):,} points')
    
    # Sort by probability (ascending) so high-prob points are rendered on top
    df = df.sort_values('prob')
    
    # Center map on data
    center_lat = df['lat'].mean()
    center_lon = df['lon'].mean()
    
    print(f'\nCreating interactive map...')
    print(f'  Center: ({center_lat:.3f}, {center_lon:.3f})')
    print(f'  Probability range: {df["prob"].min():.4f} - {df["prob"].max():.4f}')
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles='OpenStreetMap',
        control_scale=True,
    )
    
    # Add satellite imagery option
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Add terrain option
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Terrain',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Add points as circles with color based on probability
    print(f'  Adding {len(df):,} points...')
    
    # Create feature groups for different probability ranges
    fg_low = folium.FeatureGroup(name='Low (0.01-0.1)', show=True)
    fg_med = folium.FeatureGroup(name='Medium (0.1-0.3)', show=True)
    fg_high = folium.FeatureGroup(name='High (0.3-0.5)', show=True)
    fg_very_high = folium.FeatureGroup(name='Very High (≥0.5)', show=True)
    
    # Add points to appropriate groups
    for idx, row in df.iterrows():
        lat, lon, prob = row['lat'], row['lon'], row['prob']
        color = prob_to_color(prob)
        opacity = prob_to_opacity(prob)
        
        # Determine feature group
        if prob >= 0.5:
            fg = fg_very_high
            radius = 8
        elif prob >= 0.3:
            fg = fg_high
            radius = 6
        elif prob >= 0.1:
            fg = fg_med
            radius = 5
        else:
            fg = fg_low
            radius = 4
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=min(opacity * 1.2, 0.9),  # Boost opacity slightly for visibility
            opacity=min(opacity * 1.2, 0.9),
            popup=f'Probability: {prob:.3f}<br>Lat: {lat:.4f}<br>Lon: {lon:.4f}',
            tooltip=f'P={prob:.3f}',
        ).add_to(fg)
    
    # Add feature groups to map
    fg_low.add_to(m)
    fg_med.add_to(m)
    fg_high.add_to(m)
    fg_very_high.add_to(m)
    
    # Add training data if requested
    if show_training:
        print('\nAdding training data...')
        
        # Load positive training data (iNat yew observations)
        train_pos_path = Path('data/processed/inat_yew_positives_train.csv')
        val_pos_path = Path('data/processed/inat_yew_positives_val.csv')
        
        fg_train_pos = folium.FeatureGroup(name='Training: Yew (positive)', show=False)
        
        if train_pos_path.exists():
            df_train_pos = pd.read_csv(train_pos_path)
            print(f'  Training positives: {len(df_train_pos):,}')
            for _, row in df_train_pos.iterrows():
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=6,
                    color='#00ff00',
                    fill=True,
                    fillColor='#00ff00',
                    fillOpacity=0.7,
                    opacity=0.9,
                    popup=f'Training Positive (Yew)<br>Lat: {row["lat"]:.4f}<br>Lon: {row["lon"]:.4f}',
                    tooltip='Training: Yew',
                ).add_to(fg_train_pos)
        
        if val_pos_path.exists():
            df_val_pos = pd.read_csv(val_pos_path)
            print(f'  Validation positives: {len(df_val_pos):,}')
            for _, row in df_val_pos.iterrows():
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=6,
                    color='#66ff66',
                    fill=True,
                    fillColor='#66ff66',
                    fillOpacity=0.6,
                    opacity=0.8,
                    popup=f'Validation Positive (Yew)<br>Lat: {row["lat"]:.4f}<br>Lon: {row["lon"]:.4f}',
                    tooltip='Validation: Yew',
                ).add_to(fg_train_pos)
        
        fg_train_pos.add_to(m)
        
        # Load negative training data (FAIB non-yew sites)
        neg_path = Path('data/processed/faib_negatives/faib_negative_embeddings.csv')
        
        if neg_path.exists():
            df_neg = pd.read_csv(neg_path)
            print(f'  FAIB negatives: {len(df_neg):,}')
            
            # Sample negatives if there are too many
            if len(df_neg) > 500:
                df_neg_sample = df_neg.sample(n=500, random_state=42)
                print(f'  Displaying 500 sampled FAIB sites')
            else:
                df_neg_sample = df_neg
            
            fg_train_neg = folium.FeatureGroup(name='Training: Non-yew (FAIB)', show=False)
            
            for _, row in df_neg_sample.iterrows():
                bec_zone = row.get('bec_zone', 'Unknown')
                site_id = row.get('site_identifier', 'Unknown')
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=4,
                    color='#ff0000',
                    fill=True,
                    fillColor='#ff0000',
                    fillOpacity=0.4,
                    opacity=0.6,
                    popup=f'Training Negative (Non-yew)<br>BEC: {bec_zone}<br>Site: {site_id}<br>Lat: {row["lat"]:.4f}<br>Lon: {row["lon"]:.4f}',
                    tooltip=f'FAIB: {bec_zone}',
                ).add_to(fg_train_neg)
            
            fg_train_neg.add_to(m)
    
    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; right: 50px; width: 220px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p style="margin-bottom: 5px; font-weight: bold;">Yew Probability</p>
    <p style="margin: 2px;"><span style="background-color: #99cc99; padding: 2px 10px;">█</span> 0.05 - 0.10</p>
    <p style="margin: 2px;"><span style="background-color: #4db34d; padding: 2px 10px;">█</span> 0.10 - 0.30</p>
    <p style="margin: 2px;"><span style="background-color: #7fd919; padding: 2px 10px;">█</span> 0.30 - 0.50</p>
    <p style="margin: 2px;"><span style="background-color: #ffe600; padding: 2px 10px;">█</span> 0.50 - 0.70</p>
    <p style="margin: 2px;"><span style="background-color: #e66619; padding: 2px 10px;">█</span> 0.70 - 1.00</p>
    <p style="margin-bottom: 5px; margin-top: 10px; font-weight: bold;">Training Data</p>
    <p style="margin: 2px;"><span style="background-color: #00ff00; padding: 2px 10px;">●</span> Yew (positive)</p>
    <p style="margin: 2px;"><span style="background-color: #ff0000; padding: 2px 10px;">●</span> Non-yew (FAIB)</p>
    <p style="margin-top: 8px; font-size: 11px; color: #666;">
        FAIB negatives model<br>
        Based on 2,168 sample points
    </p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add title
    title_html = f'''
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                width: 500px; background-color: rgba(255,255,255,0.9); 
                border:2px solid grey; z-index:9999; font-size:16px; 
                padding: 10px; text-align: center;">
    <h3 style="margin: 0 0 5px 0;">Pacific Yew Distribution - CWH Zone</h3>
    <p style="margin: 0; font-size: 12px; color: #666;">
        Model trained with FAIB tree inventory negatives<br>
        {len(df):,} points displayed (P≥{threshold})
    </p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Save map
    print(f'\nSaving to {output_html}...')
    m.save(str(output_html))
    
    file_size_kb = output_html.stat().st_size / 1024
    print(f'✓ Done — {file_size_kb:.0f} KB')
    print(f'\nOpen in browser: file://{output_html.absolute()}')
    
    return m


def main():
    parser = argparse.ArgumentParser(description='Create interactive yew probability map')
    parser.add_argument('--input', type=str, 
                        default='results/analysis/cwh_yew_faib_negatives/sample_predictions.csv',
                        help='Path to predictions CSV')
    parser.add_argument('--output', type=str,
                        default='results/analysis/cwh_yew_faib_negatives/interactive_map.html',
                        help='Output HTML file')
    parser.add_argument('--sample-rate', type=float, default=1.0,
                        help='Fraction of points to display (0-1)')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='Minimum probability to display')
    parser.add_argument('--show-training', action='store_true', default=True,
                        help='Show training data on map')
    parser.add_argument('--no-training', dest='show_training', action='store_false',
                        help='Hide training data')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    create_interactive_map(
        predictions_csv=input_path,
        output_html=output_path,
        sample_rate=args.sample_rate,
        threshold=args.threshold,
        show_training=args.show_training,
    )


if __name__ == '__main__':
    main()
