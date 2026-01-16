#!/usr/bin/env python3
"""
Add No-Yew Sample Coordinates
==============================

Adds lat/lon coordinates to train/val splits by extracting them from the 
original BC forest inventory data via the no_yew metadata file.

Author: GitHub Copilot
Date: January 2025
"""

import pandas as pd
from pyproj import Transformer
from pathlib import Path


def load_no_yew_coordinates():
    """Load site identifiers and convert their BC Albers coords to lat/lon."""
    print("Loading no_yew site identifiers...")
    
    # Load metadata to get site identifiers
    meta_df = pd.read_csv('data/ee_imagery/image_patches_64x64/no_yew_image_metadata.csv')
    print(f"  Found {len(meta_df)} no_yew samples in metadata")
    
    # Load original inventory data
    inv_df = pd.read_csv('data/processed/bc_sample_data_deduplicated.csv', low_memory=False)
    print(f"  Loaded {len(inv_df)} inventory records")
    
    # Filter to just the sites we need
    site_ids = meta_df['site_identifier'].values
    inv_df = inv_df[inv_df['SITE_IDENTIFIER'].isin(site_ids)].copy()
    print(f"  Found {len(inv_df)} matching inventory records")
    
    # Check coordinate availability
    has_coords = (inv_df['BC_ALBERS_X'].notna() & inv_df['BC_ALBERS_Y'].notna()).sum()
    print(f"  Records with BC Albers coordinates: {has_coords}")
    
    if has_coords == 0:
        raise ValueError("No coordinates found in inventory data!")
    
    # Convert BC Albers to WGS84 lat/lon
    print("  Converting BC Albers -> WGS84 lat/lon...")
    transformer = Transformer.from_crs('EPSG:3005', 'EPSG:4326', always_xy=True)
    coords = transformer.transform(inv_df['BC_ALBERS_X'].values, 
                                   inv_df['BC_ALBERS_Y'].values)
    
    # Create lookup: site_identifier -> (lat, lon)
    coord_lookup = {}
    for site_id, lon, lat in zip(inv_df['SITE_IDENTIFIER'].values, coords[0], coords[1]):
        coord_lookup[site_id] = (lat, lon)
    
    print(f"  Created coordinate lookup for {len(coord_lookup)} sites")
    return coord_lookup


def add_coordinates_to_split(split_path, coord_lookup):
    """Add lat/lon coordinates to a train/val split CSV."""
    print(f"\nProcessing {split_path}...")
    
    # Load split
    df = pd.read_csv(split_path)
    print(f"  Original: {len(df)} samples")
    
    # Count by class (handle both 'label' and 'has_yew' columns)
    label_col = 'label' if 'label' in df.columns else 'has_yew'
    yew_count = (df[label_col] == (1 if label_col == 'label' else True)).sum()
    no_yew_count = (df[label_col] == (0 if label_col == 'label' else False)).sum()
    print(f"    Yew: {yew_count}, No_yew: {no_yew_count}")
    
    # Check existing coordinates
    has_lat = 'lat' in df.columns or 'latitude' in df.columns
    has_lon = 'lon' in df.columns or 'longitude' in df.columns
    
    if not has_lat:
        df['lat'] = None
    if not has_lon:
        df['lon'] = None
    
    # Standardize column names
    if 'latitude' in df.columns:
        df['lat'] = df['latitude']
        df = df.drop(columns=['latitude'])
    if 'longitude' in df.columns:
        df['lon'] = df['longitude']
        df = df.drop(columns=['longitude'])
    
    # Count existing coordinates
    existing_coords = (~df['lat'].isna()).sum()
    print(f"    Samples with existing coordinates: {existing_coords}")
    
    # Add coordinates for no_yew samples
    added_count = 0
    for idx, row in df.iterrows():
        # Skip if already has coordinates
        if pd.notna(row['lat']) and pd.notna(row['lon']):
            continue
        
        # Skip if not a no_yew sample
        label_col = 'label' if 'label' in df.columns else 'has_yew'
        is_no_yew = (row[label_col] == 0) if label_col == 'label' else (row[label_col] == False)
        if not is_no_yew:
            continue
        
        # Extract site_identifier from image_path: "no_yew/1231676.npy" -> 1231676
        image_path = row['image_path']
        if 'no_yew/' in image_path:
            site_id_str = image_path.split('/')[-1].replace('.npy', '')
            try:
                site_id = int(site_id_str)
                if site_id in coord_lookup:
                    lat, lon = coord_lookup[site_id]
                    df.at[idx, 'lat'] = lat
                    df.at[idx, 'lon'] = lon
                    added_count += 1
            except ValueError:
                print(f"    Warning: Could not parse site_id from {image_path}")
    
    print(f"    Added coordinates to {added_count} no_yew samples")
    
    # Final counts
    final_coords = (~df['lat'].isna()).sum()
    print(f"    Final: {final_coords} samples with coordinates")
    
    # Save updated split
    backup_path = split_path.replace('.csv', '_backup.csv')
    print(f"    Backing up original to {Path(backup_path).name}")
    df_original = pd.read_csv(split_path)
    df_original.to_csv(backup_path, index=False)
    
    print(f"    Saving updated split...")
    df.to_csv(split_path, index=False)
    print(f"    ✓ Saved {split_path}")
    
    return df


def main():
    """Main execution function."""
    print("="*70)
    print("Adding No-Yew Sample Coordinates")
    print("="*70)
    
    # Load coordinate lookup
    coord_lookup = load_no_yew_coordinates()
    
    # Process train split
    train_df = add_coordinates_to_split(
        'data/processed/train_split_filtered.csv',
        coord_lookup
    )
    
    # Process validation split
    val_df = add_coordinates_to_split(
        'data/processed/val_split_filtered.csv',
        coord_lookup
    )
    
    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    train_coords = (~train_df['lat'].isna()).sum()
    val_coords = (~val_df['lat'].isna()).sum()
    
    print(f"Training set: {train_coords}/{len(train_df)} with coordinates ({train_coords/len(train_df)*100:.1f}%)")
    print(f"Validation set: {val_coords}/{len(val_df)} with coordinates ({val_coords/len(val_df)*100:.1f}%)")
    
    print("\n✓ Coordinate addition complete!")
    print("You can now run the embedding extraction script.")


if __name__ == '__main__':
    main()
