#!/usr/bin/env python3
"""
Apply BC Forestry VRI (Vegetation Resource Inventory) data to yew probability maps.

Uses harvest history and land cover from the VEG_COMP_LYR_R1_POLY layer to:
1. Reduce probability in logged areas (yew doesn't regenerate after logging)
2. Zero out non-forested areas (water, rock, urban)
3. Optionally show harvest boundaries on the interactive map

Usage:
    python scripts/preprocessing/apply_forestry_mask.py \
        --input-dir results/predictions/jordan_river_tuned \
        --gdb /path/to/VEG_COMP_LYR_R1_POLY_2024.gdb \
        --output-dir results/predictions/jordan_river_forestry
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box


def load_forestry_data(gdb_path, lat_min, lat_max, lon_min, lon_max, layer='VEG_COMP_LYR_R1_POLY'):
    """Load VRI forestry polygons for the given bounding box."""
    print("Loading BC Forestry VRI data...")
    
    # Transform bbox to EPSG:3005 (BC Albers) which is the GDB CRS
    transformer = Transformer.from_crs('EPSG:4326', 'EPSG:3005', always_xy=True)
    x_min, y_min = transformer.transform(lon_min, lat_min)
    x_max, y_max = transformer.transform(lon_max, lat_max)
    print(f"  EPSG:3005 bbox: x=[{x_min:.0f}, {x_max:.0f}], y=[{y_min:.0f}, {y_max:.0f}]")
    
    # Read only polygons within bbox
    gdf = gpd.read_file(gdb_path, layer=layer, bbox=(x_min, y_min, x_max, y_max))
    print(f"  Found {len(gdf)} forestry polygons")
    
    # Reproject to WGS84
    gdf = gdf.to_crs('EPSG:4326')
    
    return gdf


def classify_polygons(gdf):
    """
    Classify each polygon into categories relevant for yew habitat.

    Uses multiple VRI fields to identify logged/disturbed areas:
      - HARVEST_DATE: explicit harvest record
      - OPENING_SOURCE: cutblock origin (3=RESULTS, 4=imagery, 7=harvest, 11=opening)
      - PROJ_AGE_1: explicit projected stand age
      - PROJ_AGE_CLASS_CD_1: age class (1=1-20yr ... 9=251+yr) — 92% coverage
        vs 58% for PROJ_AGE_1
      - OPENING_IND: polygon is an opening (cutblock)
    
    Returns a column 'yew_suitability' with values:
        0.0  = Definitely no yew (water, rock, recently logged)
        0.05 = Very recently logged (< 20 years, or cutblock age class 1)
        0.15 = Logged 20-40 years ago (or cutblock age class 2)
        0.3  = Unlikely yew (young plantation/cutblock, 40-60 years)
        0.5  = Possible yew (managed forest cutblock, 60-80 years)
        0.7  = Some recovery possible (logged > 80 years ago)
        1.0  = Good potential (old growth, unlogged, > 80 years)
    """
    print("\nClassifying polygons for yew suitability...")
    
    gdf = gdf.copy()
    gdf['yew_suitability'] = 1.0  # Default: assume suitable
    gdf['mask_reason'] = 'old_growth_or_unlogged'
    
    # =====================================================================
    # 1. Non-vegetated areas (water, rock, urban) → 0.0
    # =====================================================================
    non_veg = gdf['BCLCS_LEVEL_1'] == 'N'
    gdf.loc[non_veg, 'yew_suitability'] = 0.0
    gdf.loc[non_veg, 'mask_reason'] = 'non_vegetated'
    
    # Water bodies → 0.0
    water = gdf['BCLCS_LEVEL_2'] == 'W'
    gdf.loc[water, 'yew_suitability'] = 0.0
    gdf.loc[water, 'mask_reason'] = 'water'
    
    # Non-treed vegetated (shrub, herb) → 0.1
    non_treed = (gdf['BCLCS_LEVEL_1'] == 'V') & (gdf['BCLCS_LEVEL_2'] == 'N')
    gdf.loc[non_treed, 'yew_suitability'] = 0.1
    gdf.loc[non_treed, 'mask_reason'] = 'non_treed_vegetation'
    
    # =====================================================================
    # 2. Harvested areas with explicit HARVEST_DATE — classify by age
    # =====================================================================
    has_harvest = gdf['HARVEST_DATE'].notna()
    
    if has_harvest.any():
        now = datetime.now()
        harvest_dates = gdf.loc[has_harvest, 'HARVEST_DATE']
        
        # Handle timezone-aware dates
        years_since = harvest_dates.apply(
            lambda x: (now - x.replace(tzinfo=None)).days / 365.25 if x is not None else None
        )
        
        # Recently logged (< 20 years) → very low suitability
        recent_20 = has_harvest & (years_since < 20)
        gdf.loc[recent_20.reindex(gdf.index, fill_value=False), 'yew_suitability'] = 0.05
        gdf.loc[recent_20.reindex(gdf.index, fill_value=False), 'mask_reason'] = 'harvest_date_<20yr'
        
        # Logged 20-40 years ago
        medium_harvest = has_harvest & (years_since >= 20) & (years_since < 40)
        gdf.loc[medium_harvest.reindex(gdf.index, fill_value=False), 'yew_suitability'] = 0.15
        gdf.loc[medium_harvest.reindex(gdf.index, fill_value=False), 'mask_reason'] = 'harvest_date_20-40yr'
        
        # Logged 40-80 years ago
        old_harvest = has_harvest & (years_since >= 40) & (years_since < 80)
        gdf.loc[old_harvest.reindex(gdf.index, fill_value=False), 'yew_suitability'] = 0.4
        gdf.loc[old_harvest.reindex(gdf.index, fill_value=False), 'mask_reason'] = 'harvest_date_40-80yr'
        
        # Logged > 80 years ago → more recovery
        very_old_harvest = has_harvest & (years_since >= 80)
        gdf.loc[very_old_harvest.reindex(gdf.index, fill_value=False), 'yew_suitability'] = 0.7
        gdf.loc[very_old_harvest.reindex(gdf.index, fill_value=False), 'mask_reason'] = 'harvest_date_>80yr'
    
    # =====================================================================
    # 3. OPENING_SOURCE cutblocks WITHOUT harvest date
    #
    #    OPENING_SOURCE codes in BC VRI:
    #      3  = RESULTS (silviculture database)
    #      4  = Interpreted from aerial/satellite imagery (biggest gap!)
    #      7  = Logging/harvest submission
    #      11 = Opening Layer intersection
    #
    #    Source 4 alone covers ~130,000 ha in this area — most without a
    #    HARVEST_DATE.  These are cutblocks identified from imagery that
    #    were never given a formal harvest record.
    #
    #    We use PROJ_AGE_CLASS_CD_1 (92% coverage) as the primary age
    #    indicator, falling back to PROJ_AGE_1 when available.
    # =====================================================================
    is_opening_cutblock = (
        gdf['OPENING_SOURCE'].isin(['3', '4', '7', '11'])
        & ~has_harvest  # Don't override harvest-date classification
    )
    
    if is_opening_cutblock.any():
        # Use age class code (1=1-20yr, 2=21-40yr, ..., 9=251+yr)
        age_cls = gdf['PROJ_AGE_CLASS_CD_1'].fillna('')
        
        # Age class 1: 1-20 years → 0.05
        cls1 = is_opening_cutblock & (age_cls == '1')
        gdf.loc[cls1, 'yew_suitability'] = np.minimum(gdf.loc[cls1, 'yew_suitability'], 0.05)
        gdf.loc[cls1 & (gdf['yew_suitability'] <= 0.05), 'mask_reason'] = 'cutblock_age_1-20yr'
        
        # Age class 2: 21-40 years → 0.15
        cls2 = is_opening_cutblock & (age_cls == '2')
        gdf.loc[cls2, 'yew_suitability'] = np.minimum(gdf.loc[cls2, 'yew_suitability'], 0.15)
        gdf.loc[cls2 & (gdf['yew_suitability'] <= 0.15), 'mask_reason'] = 'cutblock_age_21-40yr'
        
        # Age class 3: 41-60 years → 0.3
        cls3 = is_opening_cutblock & (age_cls == '3')
        gdf.loc[cls3, 'yew_suitability'] = np.minimum(gdf.loc[cls3, 'yew_suitability'], 0.3)
        gdf.loc[cls3 & (gdf['yew_suitability'] <= 0.3), 'mask_reason'] = 'cutblock_age_41-60yr'
        
        # Age class 4: 61-80 years → 0.5
        cls4 = is_opening_cutblock & (age_cls == '4')
        gdf.loc[cls4, 'yew_suitability'] = np.minimum(gdf.loc[cls4, 'yew_suitability'], 0.5)
        gdf.loc[cls4 & (gdf['yew_suitability'] <= 0.5), 'mask_reason'] = 'cutblock_age_61-80yr'
        
        # Age class 5-6: 81-120 years → 0.7 (some recovery)
        cls56 = is_opening_cutblock & age_cls.isin(['5', '6'])
        gdf.loc[cls56, 'yew_suitability'] = np.minimum(gdf.loc[cls56, 'yew_suitability'], 0.7)
        gdf.loc[cls56 & (gdf['yew_suitability'] <= 0.7), 'mask_reason'] = 'cutblock_age_81-120yr'
        
        # Age class 7-9: > 120 years — leave at 1.0 (sufficient recovery time)
        
        # Cutblocks with NO age class at all — treat as moderately suspicious
        no_age_cls = is_opening_cutblock & (age_cls == '')
        gdf.loc[no_age_cls, 'yew_suitability'] = np.minimum(gdf.loc[no_age_cls, 'yew_suitability'], 0.4)
        gdf.loc[no_age_cls & (gdf['yew_suitability'] <= 0.4), 'mask_reason'] = 'cutblock_no_age_info'
        
        n_cutblock = is_opening_cutblock.sum()
        n_reduced = (gdf.loc[is_opening_cutblock, 'yew_suitability'] < 1.0).sum()
        print(f"  Cutblock openings (OPENING_SOURCE 3/4/7/11, no harvest date): "
              f"{n_cutblock} polygons, {n_reduced} with reduced suitability")
    
    # =====================================================================
    # 4. Young stands by PROJ_AGE_1 — catch remaining polygons not yet
    #    handled by harvest date or opening source
    # =====================================================================
    already_classified = has_harvest | is_opening_cutblock
    has_age = gdf['PROJ_AGE_1'].notna() & ~already_classified
    if has_age.any():
        young = has_age & (gdf['PROJ_AGE_1'] < 40)
        gdf.loc[young, 'yew_suitability'] = np.minimum(
            gdf.loc[young, 'yew_suitability'], 0.2
        )
        gdf.loc[young & (gdf['yew_suitability'] <= 0.2), 'mask_reason'] = 'young_stand_<40yr'
        
        medium_age = has_age & (gdf['PROJ_AGE_1'] >= 40) & (gdf['PROJ_AGE_1'] < 80)
        gdf.loc[medium_age, 'yew_suitability'] = np.minimum(
            gdf.loc[medium_age, 'yew_suitability'], 0.6
        )
        gdf.loc[medium_age & (gdf['yew_suitability'] <= 0.6), 'mask_reason'] = 'medium_stand_40-80yr'
    
    # =====================================================================
    # 5. Young stands by PROJ_AGE_CLASS_CD_1 only (no explicit age, no
    #    harvest date, not an opening source cutblock)
    # =====================================================================
    still_unclassified = ~already_classified & ~has_age
    has_age_class_only = still_unclassified & gdf['PROJ_AGE_CLASS_CD_1'].notna()
    if has_age_class_only.any():
        age_cls2 = gdf['PROJ_AGE_CLASS_CD_1']
        
        young_cls = has_age_class_only & age_cls2.isin(['1', '2'])
        gdf.loc[young_cls, 'yew_suitability'] = np.minimum(
            gdf.loc[young_cls, 'yew_suitability'], 0.2
        )
        gdf.loc[young_cls & (gdf['yew_suitability'] <= 0.2), 'mask_reason'] = 'age_class_1-2_no_source'
        
        medium_cls = has_age_class_only & age_cls2.isin(['3', '4'])
        gdf.loc[medium_cls, 'yew_suitability'] = np.minimum(
            gdf.loc[medium_cls, 'yew_suitability'], 0.6
        )
        gdf.loc[medium_cls & (gdf['yew_suitability'] <= 0.6), 'mask_reason'] = 'age_class_3-4_no_source'
    
    # =====================================================================
    # Print summary
    # =====================================================================
    print("\n  Suitability classification summary:")
    reasons = gdf.groupby('mask_reason').agg(
        count=('yew_suitability', 'size'),
        suitability=('yew_suitability', 'first')
    ).sort_values('suitability')
    for reason, row in reasons.iterrows():
        print(f"    {reason}: {row['count']} polygons (suitability={row['suitability']:.2f})")
    
    return gdf


def rasterize_suitability(gdf, prob_grid_shape, lat_min, lat_max, lon_min, lon_max):
    """
    Rasterize the yew suitability values onto the probability grid.
    Returns a grid of the same shape as prob_grid with suitability values 0-1.
    """
    print("\nRasterizing forestry data onto probability grid...")
    
    h, w = prob_grid_shape
    
    # Create rasterio transform
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, w, h)
    
    # Rasterize suitability values
    shapes = [(geom, val) for geom, val in zip(gdf.geometry, gdf['yew_suitability'])]
    
    if len(shapes) == 0:
        print("  Warning: No shapes to rasterize, returning all 1.0")
        return np.ones((h, w), dtype=np.float32)
    
    suit_grid = rasterize(
        shapes,
        out_shape=(h, w),
        transform=transform,
        fill=1.0,  # Default for areas outside any polygon
        dtype=np.float32,
        all_touched=True
    )
    
    print(f"  Rasterized grid: {suit_grid.shape}")
    print(f"  Suitability range: [{suit_grid.min():.3f}, {suit_grid.max():.3f}]")
    print(f"  Pixels with reduced suitability: {(suit_grid < 1.0).sum():,} ({100*(suit_grid < 1.0).sum()/(h*w):.1f}%)")
    
    return suit_grid


def rasterize_harvest_boundaries(gdf, prob_grid_shape, lat_min, lat_max, lon_min, lon_max):
    """
    Create a binary mask showing harvest area boundaries for visualization.
    Returns grid: 1 = harvested area, 0 = not harvested
    """
    h, w = prob_grid_shape
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, w, h)
    
    harvested = gdf[gdf['HARVEST_DATE'].notna()]
    
    if len(harvested) == 0:
        return np.zeros((h, w), dtype=np.uint8)
    
    shapes = [(geom, 1) for geom in harvested.geometry]
    
    harvest_grid = rasterize(
        shapes,
        out_shape=(h, w),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True
    )
    
    return harvest_grid


def apply_mask(prob_grid, suit_grid, method='multiply'):
    """
    Apply suitability mask to probability grid.
    
    Methods:
        'multiply': P_new = P_old * suitability
        'cap': P_new = min(P_old, suitability)
        'weighted': P_new = P_old * (0.3 + 0.7 * suitability)  # softer
    """
    if method == 'multiply':
        masked = prob_grid * suit_grid
    elif method == 'cap':
        masked = np.minimum(prob_grid, suit_grid)
    elif method == 'weighted':
        # Suitability of 0 still allows 30% of original probability
        masked = prob_grid * (0.3 + 0.7 * suit_grid)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return masked.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description='Apply BC Forestry data to yew probability maps',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Prediction directory with prob_grid.npy and metadata.json')
    parser.add_argument('--gdb', type=str, 
                        default='/home/jericho/Downloads/firefox-downloads/VEG_COMP_LYR_R1_POLY_2024.gdb',
                        help='Path to BC VRI geodatabase')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: input-dir with _forestry suffix)')
    parser.add_argument('--method', type=str, default='multiply',
                        choices=['multiply', 'cap', 'weighted'],
                        help='How to apply the mask (default: multiply)')
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path(str(input_dir) + '_forestry')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("APPLYING BC FORESTRY DATA TO YEW PROBABILITY MAP")
    print("=" * 60)
    
    # Load prediction data
    print("\nLoading prediction data...")
    prob_grid = np.load(input_dir / 'prob_grid.npy')
    rgb_image = np.load(input_dir / 'rgb_image.npy')
    
    with open(input_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    bbox = metadata['bbox']
    # Support both metadata formats (SVM uses lat_min/lon_min, GPU uses south/north/west/east)
    if 'lat_min' in bbox:
        lat_min, lat_max = bbox['lat_min'], bbox['lat_max']
        lon_min, lon_max = bbox['lon_min'], bbox['lon_max']
    else:
        lat_min, lat_max = bbox['south'], bbox['north']
        lon_min, lon_max = bbox['west'], bbox['east']
    
    print(f"  Prob grid: {prob_grid.shape}")
    print(f"  Bbox: {lat_min:.4f}°N to {lat_max:.4f}°N, {lon_min:.4f}°W to {lon_max:.4f}°W")
    
    # Load forestry data
    gdf = load_forestry_data(args.gdb, lat_min, lat_max, lon_min, lon_max)
    
    # Classify polygons
    gdf = classify_polygons(gdf)
    
    # Rasterize
    suit_grid = rasterize_suitability(gdf, prob_grid.shape, lat_min, lat_max, lon_min, lon_max)
    harvest_grid = rasterize_harvest_boundaries(gdf, prob_grid.shape, lat_min, lat_max, lon_min, lon_max)
    
    # Apply mask
    print(f"\nApplying forestry mask (method={args.method})...")
    print(f"  Original prob: mean={prob_grid.mean():.4f}, P≥0.5: {(prob_grid >= 0.5).sum():,}")
    
    masked_prob = apply_mask(prob_grid, suit_grid, method=args.method)
    
    print(f"  Masked prob:   mean={masked_prob.mean():.4f}, P≥0.5: {(masked_prob >= 0.5).sum():,}")
    print(f"  Pixels reduced: {(masked_prob < prob_grid).sum():,}")
    
    # Save outputs
    print("\nSaving results...")
    np.save(output_dir / 'prob_grid.npy', masked_prob)
    np.save(output_dir / 'prob_grid_original.npy', prob_grid)
    np.save(output_dir / 'suitability_grid.npy', suit_grid)
    np.save(output_dir / 'harvest_grid.npy', harvest_grid)
    np.save(output_dir / 'rgb_image.npy', rgb_image)
    
    # Copy embedding image if exists
    emb_path = input_dir / 'embedding_image.npy'
    if emb_path.exists():
        import shutil
        shutil.copy2(emb_path, output_dir / 'embedding_image.npy')
    
    # Update metadata
    metadata['forestry_mask'] = {
        'method': args.method,
        'gdb_path': args.gdb,
        'polygons_loaded': len(gdf),
        'harvested_polygons': int(gdf['HARVEST_DATE'].notna().sum()),
        'pixels_affected': int((masked_prob < prob_grid).sum()),
        'original_mean_prob': float(prob_grid.mean()),
        'masked_mean_prob': float(masked_prob.mean()),
        'suitability_classes': {
            reason: int((gdf['mask_reason'] == reason).sum())
            for reason in gdf['mask_reason'].unique()
        }
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✓ prob_grid.npy (forestry-masked)")
    print(f"  ✓ prob_grid_original.npy (unmasked)")
    print(f"  ✓ suitability_grid.npy")
    print(f"  ✓ harvest_grid.npy")
    print(f"  ✓ metadata.json")
    
    # Statistics
    print(f"\n{'='*60}")
    print("COMPARISON: ORIGINAL vs FORESTRY-MASKED")
    print(f"{'='*60}")
    scale = metadata.get('scale_m', 10)
    pixel_area_ha = (scale ** 2) / 10000
    
    for thresh in [0.3, 0.5, 0.7, 0.9]:
        orig_count = int((prob_grid >= thresh).sum())
        mask_count = int((masked_prob >= thresh).sum())
        diff = orig_count - mask_count
        print(f"  P≥{thresh}: {orig_count:>8,} → {mask_count:>8,} pixels "
              f"({orig_count*pixel_area_ha:>8.1f} → {mask_count*pixel_area_ha:>8.1f} ha, "
              f"-{diff:,} pixels removed)")
    
    print(f"\n✓ All results saved to: {output_dir}")
    print(f"\nNext: Generate interactive map:")
    print(f"  python scripts/visualization/create_interactive_map.py \\")
    print(f"    --input-dir {output_dir}")


if __name__ == '__main__':
    main()
