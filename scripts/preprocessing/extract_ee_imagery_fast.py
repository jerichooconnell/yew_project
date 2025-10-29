#!/usr/bin/env python3
"""
Fast Batch Earth Engine Data Extraction
========================================

Uses Earth Engine's server-side batch processing for much faster extraction.
Instead of looping through plots one-by-one, this creates a FeatureCollection
and processes everything server-side in parallel.

Speed improvement: ~100-1000x faster than sequential processing!

Author: Analysis Tool
Date: October 2025
"""

import ee
import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime


class FastEarthEngineExtractor:
    """
    Fast batch extraction using Earth Engine server-side operations.
    """

    def __init__(self, output_dir='data/ee_imagery', project='carbon-storm-206002'):
        """Initialize Earth Engine with project."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            ee.Initialize(project=project)
            print("✓ Earth Engine initialized successfully")
        except Exception as e:
            print(f"Error initializing Earth Engine: {e}")
            raise

    def bc_albers_to_wgs84(self, x, y):
        """Convert BC Albers (EPSG:3005) to WGS84 (EPSG:4326)."""
        point = ee.Geometry.Point([x, y], proj='EPSG:3005')
        coords = point.transform('EPSG:4326', 1).coordinates()
        return coords

    def create_feature_collection(self, df):
        """
        Create an Earth Engine FeatureCollection from plot locations.
        This is the key to fast processing - all plots processed together!
        """
        print("\nCreating FeatureCollection from plot locations...")

        features = []
        for idx, row in df.iterrows():
            x = float(row['BC_ALBERS_X'])
            y = float(row['BC_ALBERS_Y'])

            # Convert coordinates to WGS84
            coords = self.bc_albers_to_wgs84(x, y)
            point = ee.Geometry.Point(coords)

            # Create feature with metadata
            properties = {
                'plot_id': str(row.get('SITE_IDENTIFIER', f'plot_{idx}')),
                'x': x,
                'y': y,
                'measurement_year': int(row.get('MEAS_YR', 2020)) if pd.notna(row.get('MEAS_YR')) else 2020,
                'psp_status': str(row.get('PSP_STATUS', '')),
                'last_msmt': str(row.get('LAST_MSMT', 'Y'))
            }

            features.append(ee.Feature(point, properties))

        fc = ee.FeatureCollection(features)
        print(f"✓ Created FeatureCollection with {len(features)} plots")
        return fc

    def get_imagery_composite(self, start_date, end_date):
        """
        Get Sentinel-2 median composite for the time period.
        Returns an image with all bands ready for extraction.
        """
        # Sentinel-2 Surface Reflectance
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))

        # Cloud masking
        def mask_clouds(image):
            qa = image.select('QA60')
            cloud_mask = qa.bitwiseAnd(1 << 10).eq(0).And(
                qa.bitwiseAnd(1 << 11).eq(0))
            return image.updateMask(cloud_mask).divide(10000)

        # Get median composite
        composite = s2.map(mask_clouds).median()

        # Select bands and rename
        composite = composite.select(
            ['B2', 'B3', 'B4', 'B8'],
            ['blue', 'green', 'red', 'nir']
        )

        # Add NDVI
        ndvi = composite.normalizedDifference(['nir', 'red']).rename('ndvi')

        # Add EVI
        evi = composite.expression(
            '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
            {
                'NIR': composite.select('nir'),
                'RED': composite.select('red'),
                'BLUE': composite.select('blue')
            }
        ).rename('evi')

        return composite.addBands([ndvi, evi])

    def add_terrain(self, image):
        """Add elevation and terrain derivatives."""
        dem = ee.Image('USGS/SRTMGL1_003').select('elevation')
        terrain = ee.Algorithms.Terrain(dem)

        return image.addBands([
            dem,
            terrain.select('slope'),
            terrain.select('aspect')
        ])

    def extract_values_batch(self, feature_collection, imagery_period='2022-2024'):
        """
        Extract all satellite data for all plots in ONE operation!
        This is massively faster than looping.
        """
        print(
            f"\nExtracting satellite data for imagery period: {imagery_period}...")

        # Determine date range
        if imagery_period == '2022-2024':
            start_date = '2022-06-01'
            end_date = '2024-08-31'
        elif imagery_period == '2020-2022':
            start_date = '2020-06-01'
            end_date = '2022-08-31'
        elif imagery_period == '2015-2017':
            start_date = '2015-06-01'
            end_date = '2017-08-31'
        else:
            # Parse custom period
            years = imagery_period.split('-')
            start_date = f'{years[0]}-06-01'
            end_date = f'{years[1]}-08-31'

        # Get imagery composite
        composite = self.get_imagery_composite(start_date, end_date)
        composite = self.add_terrain(composite)

        # Define function to extract values at each point
        def extract_at_point(feature):
            # Get point geometry
            point = feature.geometry()

            # Extract values in 250m buffer (mean)
            values = composite.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point.buffer(250),
                scale=10,
                maxPixels=1e9
            )

            # Add extracted values to feature properties
            return feature.set(values).set('imagery_period', imagery_period)

        # Map extraction over all features (server-side parallel processing!)
        print("  Running server-side extraction (this may take 1-2 minutes)...")
        result_fc = feature_collection.map(extract_at_point)

        # Get results
        print("  Downloading results...")
        result_info = result_fc.getInfo()

        return result_info

    def process_all_plots_fast(self, df):
        """
        Fast processing of all plots using batch operations.
        Groups plots by imagery period for optimal performance.

        NOTE: Only processes ACTIVE sites - inactive sites are excluded
        since current satellite imagery won't correlate with historical
        field measurements.
        """
        print("\n" + "="*70)
        print("FAST BATCH PROCESSING - ACTIVE SITES ONLY")
        print("="*70)

        # Filter to ACTIVE sites only
        active_mask = (df['PSP_STATUS'] == 'A') | (df['LAST_MSMT'] == 'N')
        df_active = df[active_mask].copy()
        df_inactive = df[~active_mask].copy()

        print(f"\nFiltering to active sites:")
        print(f"  Total plots: {len(df)}")
        print(
            f"  Active sites (PSP_STATUS='A' or LAST_MSMT='N'): {len(df_active)}")
        print(f"  Inactive sites (excluded): {len(df_inactive)}")
        print(f"  Processing: {len(df_active)} active sites")

        if len(df_inactive) > 0:
            print(f"\n  Rationale: Excluding inactive sites because current satellite")
            print(f"             imagery (2022-2024) won't correlate with historical")
            print(f"             measurements (some from 1970s-1990s)")

        # Use only active sites
        df = df_active

        # Determine imagery period for each plot
        def get_imagery_period(row):
            # All active sites use latest imagery
            return '2022-2024'

        df['imagery_period'] = df.apply(get_imagery_period, axis=1)

        # Group by imagery period
        period_groups = df.groupby('imagery_period')
        print(f"\nFound {len(period_groups)} different imagery periods:")
        for period, group in period_groups:
            print(f"  {period}: {len(group)} plots")

        # Process each group in chunks to avoid timeouts
        all_results = []
        CHUNK_SIZE = 1000  # Process 1000 plots at a time

        for period, group_df in period_groups:
            print(f"\n{'='*70}")
            print(f"Processing {len(group_df)} plots for period {period}")
            print(f"{'='*70}")

            # Split into chunks
            num_chunks = (len(group_df) + CHUNK_SIZE - 1) // CHUNK_SIZE
            print(
                f"Processing in {num_chunks} chunks of {CHUNK_SIZE} plots each...")

            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * CHUNK_SIZE
                end_idx = min((chunk_idx + 1) * CHUNK_SIZE, len(group_df))
                chunk_df = group_df.iloc[start_idx:end_idx]

                print(
                    f"\n  Chunk {chunk_idx+1}/{num_chunks}: plots {start_idx+1}-{end_idx}...")

                # Create FeatureCollection for this chunk
                fc = self.create_feature_collection(chunk_df)

                # Extract data (server-side batch operation!)
                try:
                    result_info = self.extract_values_batch(fc, period)

                    # Parse results
                    features = result_info['features']
                    print(f"  ✓ Successfully extracted {len(features)} plots")

                    for feature in features:
                        props = feature['properties']
                        coords = feature['geometry']['coordinates']

                        result = {
                            'plot_id': props.get('plot_id'),
                            'x': props.get('x'),
                            'y': props.get('y'),
                            'lon': coords[0],
                            'lat': coords[1],
                            'measurement_year': props.get('measurement_year'),
                            'imagery_period': props.get('imagery_period'),
                            'blue': props.get('blue'),
                            'green': props.get('green'),
                            'red': props.get('red'),
                            'nir': props.get('nir'),
                            'ndvi': props.get('ndvi'),
                            'evi': props.get('evi'),
                            'elevation': props.get('elevation'),
                            'slope': props.get('slope'),
                            'aspect': props.get('aspect'),
                            'success': True
                        }
                        all_results.append(result)

                    # Save incremental progress every 2000 plots
                    if len(all_results) % 2000 == 0 or (chunk_idx + 1) == num_chunks:
                        temp_df = pd.DataFrame(all_results)
                        temp_path = self.output_dir / 'ee_extraction_progress.csv'
                        temp_df.to_csv(temp_path, index=False)
                        print(
                            f"  💾 Progress saved: {len(all_results)} plots extracted so far")

                except Exception as e:
                    print(f"  ✗ Error processing chunk: {e}")
                    continue

        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)

        print("\n" + "="*70)
        print("EXTRACTION COMPLETE")
        print("="*70)
        print(f"Total plots processed: {len(results_df)}")
        print(f"Success rate: {len(results_df)/len(df)*100:.1f}%")

        return results_df

    def save_results(self, df, prefix='ee_batch_data'):
        """Save extraction results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save CSV
        csv_path = self.output_dir / f'{prefix}_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Saved CSV to {csv_path}")

        # Save pickle
        pkl_path = self.output_dir / f'{prefix}_{timestamp}.pkl'
        df.to_pickle(pkl_path)
        print(f"✓ Saved pickle to {pkl_path}")

        # Save metadata
        metadata = {
            'extraction_date': timestamp,
            'num_plots': int(len(df)),
            'success_count': int(df['success'].sum()) if 'success' in df.columns else int(len(df)),
            'columns': list(df.columns)
        }

        meta_path = self.output_dir / f'{prefix}_{timestamp}_metadata.json'
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata to {meta_path}")

        return csv_path


def main():
    """Main execution."""
    print("="*70)
    print("FAST BATCH EARTH ENGINE EXTRACTION")
    print("="*70)

    # Initialize
    extractor = FastEarthEngineExtractor()

    # Load data
    print("\nLoading plot locations...")
    df = pd.read_csv(
        'data/raw/bc_sample_data-2025-10-09/bc_sample_data.csv', low_memory=False)
    df = df[(df['BC_ALBERS_X'].notna()) & (df['BC_ALBERS_Y'].notna())].copy()
    print(f"Loaded {len(df)} plots with valid coordinates")

    # Filter to active sites only
    print("\nFiltering to active sites...")
    active_mask = (df['PSP_STATUS'] == 'A') | (df['LAST_MSMT'] == 'N')
    df = df[active_mask].copy()
    print(f"Active sites: {len(df)}")

    # Filter to selected BEC zones (CWH, ICH, IDF, CDF)
    print("\nFiltering to selected BEC zones (CWH, ICH, IDF, CDF)...")
    selected_zones = ['CWH', 'ICH', 'IDF', 'CDF']
    bec_mask = df['BEC_ZONE'].isin(selected_zones)
    df = df[bec_mask].copy()
    print(f"Sites in selected BEC zones: {len(df)}")
    print("BEC zone breakdown:")
    print(df['BEC_ZONE'].value_counts().to_string())

    # Process all filtered plots
    df_sample = df
    prefix = 'ee_batch_bec_filtered'
    print(f"\nProcessing ALL {len(df_sample)} filtered plots...")

    # Process all filtered plots
    df_sample = df
    prefix = 'ee_batch_bec_filtered'
    print(f"\nProcessing ALL {len(df_sample)} filtered plots...")
    print("This will be MUCH faster than the loop-based approach!")

    start_time = time.time()

    # Process
    results_df = extractor.process_all_plots_fast(df_sample)

    elapsed = time.time() - start_time

    # Save
    csv_path = extractor.save_results(results_df, prefix=prefix)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Plots processed: {len(results_df)}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Speed: {len(results_df)/elapsed*60:.1f} plots/minute")
    print(f"Results saved to: {csv_path}")
    print("="*70)

    # Show statistics
    if len(results_df) > 0:
        print("\nData Quality:")
        print(
            f"  NDVI: {results_df['ndvi'].mean():.3f} ± {results_df['ndvi'].std():.3f}")
        print(
            f"  Elevation: {results_df['elevation'].mean():.0f} ± {results_df['elevation'].std():.0f} m")
        print(f"  Valid data: {results_df['ndvi'].notna().sum()} plots")


if __name__ == "__main__":
    main()
