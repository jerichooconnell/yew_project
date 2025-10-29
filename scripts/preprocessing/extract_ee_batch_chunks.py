#!/usr/bin/env python3
"""
Chunked Earth Engine Extraction - Robust Version
=================================================

Processes in small chunks and saves progress incrementally.
Can be interrupted and resumed.

Usage:
    nohup python extract_ee_batch_chunks.py > extraction.log 2>&1 &
"""

import ee
import pandas as pd
import json
import time
import sys
from pathlib import Path
from datetime import datetime


# Configuration
PROJECT = 'carbon-storm-206002'
DATA_FILE = 'data/raw/bc_sample_data-2025-10-09/bc_sample_data.csv'
OUTPUT_DIR = Path('data/ee_imagery')
CHUNK_SIZE = 500  # Smaller chunks for stability
SELECTED_ZONES = ['CWH', 'ICH', 'IDF', 'CDF']

# Progress tracking
PROGRESS_FILE = OUTPUT_DIR / 'extraction_progress.json'
RESULTS_FILE = OUTPUT_DIR / 'extraction_results_ongoing.csv'


def init_ee():
    """Initialize Earth Engine."""
    try:
        ee.Initialize(project=PROJECT)
        print(f"✓ Earth Engine initialized with project: {PROJECT}")
        return True
    except Exception as e:
        print(f"✗ Error initializing Earth Engine: {e}")
        return False


def bc_albers_to_wgs84(x, y):
    """Convert BC Albers to WGS84."""
    point = ee.Geometry.Point([x, y], proj='EPSG:3005')
    coords = point.transform('EPSG:4326', 1).coordinates()
    return coords


def get_sentinel2_composite(start_date, end_date):
    """Get Sentinel-2 composite."""
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
        .select(['B2', 'B3', 'B4', 'B8'], ['blue', 'green', 'red', 'nir'])

    composite = s2.median()

    # Add NDVI and EVI
    ndvi = composite.normalizedDifference(['nir', 'red']).rename('ndvi')
    evi = composite.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
        {
            'NIR': composite.select('nir'),
            'RED': composite.select('red'),
            'BLUE': composite.select('blue')
        }
    ).rename('evi')

    return composite.addBands([ndvi, evi])


def add_terrain(image):
    """Add elevation and terrain."""
    dem = ee.Image('USGS/SRTMGL1_003')
    elevation = dem.select('elevation')
    terrain = ee.Algorithms.Terrain(dem)
    slope = terrain.select('slope')
    aspect = terrain.select('aspect')

    return image.addBands([elevation, slope, aspect])


def extract_chunk(chunk_df, start_date='2022-06-01', end_date='2024-08-31'):
    """Extract data for a chunk of plots."""
    results = []

    # Get imagery
    composite = get_sentinel2_composite(start_date, end_date)
    composite = add_terrain(composite)

    # Create FeatureCollection
    features = []
    for idx, row in chunk_df.iterrows():
        x = float(row['BC_ALBERS_X'])
        y = float(row['BC_ALBERS_Y'])
        coords = bc_albers_to_wgs84(x, y)
        point = ee.Geometry.Point(coords)

        properties = {
            'plot_id': str(row['SITE_IDENTIFIER']),
            'x': x,
            'y': y
        }
        features.append(ee.Feature(point, properties))

    fc = ee.FeatureCollection(features)

    # Extract values
    def extract_at_point(feature):
        point = feature.geometry()
        values = composite.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point.buffer(250),
            scale=10,
            maxPixels=1e9
        )
        return feature.set(values)

    result_fc = fc.map(extract_at_point)

    # Download with retry logic for rate limiting
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result_info = result_fc.getInfo()
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 30  # 30s, 60s, 90s
                print(
                    f"    Retry {attempt+1}/{max_retries} after {wait_time}s (rate limit or timeout)")
                time.sleep(wait_time)
            else:
                raise

    # Parse results
    for feature in result_info['features']:
        props = feature['properties']
        coords = feature['geometry']['coordinates']

        results.append({
            'plot_id': props.get('plot_id'),
            'x': props.get('x'),
            'y': props.get('y'),
            'lon': coords[0],
            'lat': coords[1],
            'blue': props.get('blue'),
            'green': props.get('green'),
            'red': props.get('red'),
            'nir': props.get('nir'),
            'ndvi': props.get('ndvi'),
            'evi': props.get('evi'),
            'elevation': props.get('elevation'),
            'slope': props.get('slope'),
            'aspect': props.get('aspect')
        })

    return results


def load_progress():
    """Load progress from file."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {'last_completed_chunk': -1, 'total_extracted': 0}


def save_progress(chunk_idx, total_extracted):
    """Save progress to file."""
    progress = {
        'last_completed_chunk': chunk_idx,
        'total_extracted': total_extracted,
        'timestamp': datetime.now().isoformat()
    }
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def main():
    print("="*70)
    print("CHUNKED EARTH ENGINE EXTRACTION")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Initialize EE
    if not init_ee():
        sys.exit(1)

    # Load data
    print("Loading plot data...")
    df = pd.read_csv(DATA_FILE, low_memory=False)
    df = df[(df['BC_ALBERS_X'].notna()) & (df['BC_ALBERS_Y'].notna())].copy()
    print(f"  Total plots with coordinates: {len(df)}")

    # Filter to active sites
    active_mask = (df['PSP_STATUS'] == 'A') | (df['LAST_MSMT'] == 'N')
    df = df[active_mask].copy()
    print(f"  Active sites: {len(df)}")

    # Filter to selected BEC zones
    bec_mask = df['BEC_ZONE'].isin(SELECTED_ZONES)
    df = df[bec_mask].copy()
    print(f"  Sites in {SELECTED_ZONES}: {len(df)}")
    print()

    # Check for existing progress
    progress = load_progress()
    start_chunk = progress['last_completed_chunk'] + 1

    if start_chunk > 0:
        print(f"Resuming from chunk {start_chunk}")
        print(f"Already extracted: {progress['total_extracted']} plots")
        print()

    # Load existing results if any
    if RESULTS_FILE.exists() and start_chunk > 0:
        existing_df = pd.read_csv(RESULTS_FILE)
        all_results = existing_df.to_dict('records')
        print(f"Loaded {len(all_results)} existing results")
    else:
        all_results = []

    # Process in chunks
    num_chunks = (len(df) + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"Processing {len(df)} plots in {num_chunks} chunks of {CHUNK_SIZE}")
    print("="*70)
    print()

    start_time = time.time()

    for chunk_idx in range(start_chunk, num_chunks):
        chunk_start_time = time.time()

        start_idx = chunk_idx * CHUNK_SIZE
        end_idx = min((chunk_idx + 1) * CHUNK_SIZE, len(df))
        chunk_df = df.iloc[start_idx:end_idx]

        print(
            f"Chunk {chunk_idx+1}/{num_chunks}: plots {start_idx+1}-{end_idx}")

        try:
            # Extract data
            chunk_results = extract_chunk(chunk_df)
            all_results.extend(chunk_results)

            # Save incremental results
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(RESULTS_FILE, index=False)

            # Save progress
            save_progress(chunk_idx, len(all_results))

            chunk_time = time.time() - chunk_start_time
            total_time = time.time() - start_time
            plots_per_sec = len(chunk_results) / \
                chunk_time if chunk_time > 0 else 0

            print(
                f"  ✓ Extracted {len(chunk_results)} plots in {chunk_time:.1f}s ({plots_per_sec:.1f} plots/s)")
            print(
                f"  Total: {len(all_results)}/{len(df)} ({100*len(all_results)/len(df):.1f}%)")
            print(f"  Elapsed: {total_time/60:.1f} min")

            # Estimate remaining time
            if len(all_results) > 0:
                rate = len(all_results) / total_time
                remaining = (len(df) - len(all_results)) / \
                    rate if rate > 0 else 0
                print(f"  Est. remaining: {remaining/60:.1f} min")

            # Small delay to avoid rate limiting (especially after every 5 chunks)
            if (chunk_idx + 1) % 5 == 0:
                print("  Pausing 10s to avoid rate limits...")
                time.sleep(10)
            print()

        except Exception as e:
            print(f"  ✗ Error: {e}")
            print(f"  Progress saved. Can resume from chunk {chunk_idx}")
            print()
            continue

    # Final save
    print("="*70)
    print("EXTRACTION COMPLETE")
    print("="*70)

    final_df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_path = OUTPUT_DIR / f'ee_batch_bec_filtered_{timestamp}.csv'
    final_df.to_csv(final_path, index=False)

    print(f"Total plots extracted: {len(final_df)}")
    print(f"Success rate: {100*len(final_df)/len(df):.1f}%")
    print(f"Results saved to: {final_path}")

    # Statistics
    if len(final_df) > 0:
        print()
        print("Data Quality:")
        print(
            f"  NDVI: {final_df['ndvi'].mean():.3f} ± {final_df['ndvi'].std():.3f}")
        print(
            f"  Elevation: {final_df['elevation'].mean():.0f} ± {final_df['elevation'].std():.0f} m")
        print(f"  Valid NDVI: {final_df['ndvi'].notna().sum()} plots")

    total_time = time.time() - start_time
    print()
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average speed: {len(final_df)/total_time*60:.1f} plots/minute")
    print()
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Clean up progress file
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress has been saved.")
        print(f"Resume by running this script again.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
