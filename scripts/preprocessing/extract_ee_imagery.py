#!/usr/bin/env python3
"""
Earth Engine Data Extraction for Pacific Yew Sites
===================================================

Extracts Sentinel-2 imagery (RGB + NIR), elevation, and environmental data
for forest plot locations in British Columbia.

Features extracted:
- Sentinel-2 RGB + NIR bands (10m resolution)
- SRTM elevation (30m resolution)
- Terrain derivatives (slope, aspect)
- NDVI, EVI vegetation indices
- Seasonal composites (growing season focus)

Author: Analysis Tool
Date: October 2025
"""

import ee
import pandas as pd
import numpy as np
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple
import pickle
from datetime import datetime
import json

# Try to import geemap for visualization (optional)
try:
    import geemap
    HAS_GEEMAP = True
except ImportError:
    HAS_GEEMAP = False
    print("Note: geemap not installed. Visualization features disabled.")


class EarthEngineExtractor:
    """
    Extract satellite imagery and environmental data from Google Earth Engine.
    """

    def __init__(self, output_dir='data/ee_imagery'):
        """
        Initialize Earth Engine and set up output directory.

        Args:
            output_dir: Directory to save extracted imagery
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Earth Engine with project
        try:
            ee.Initialize(project='carbon-storm-206002')
            print("✓ Earth Engine initialized successfully")
        except Exception as e:
            print(f"Error initializing Earth Engine: {e}")
            print("Please run: earthengine authenticate")
            raise

        # Define band names
        self.sentinel_bands = ['B2', 'B3', 'B4', 'B8']  # Blue, Green, Red, NIR
        self.output_bands = ['blue', 'green', 'red', 'nir']

    def load_plot_locations(self, csv_path='data/raw/bc_sample_data-2025-10-09/bc_sample_data.csv'):
        """
        Load forest plot locations from BC sample data.

        Returns:
            DataFrame with plot coordinates and metadata
        """
        print(f"\nLoading plot locations from {csv_path}...")
        df = pd.read_csv(csv_path)

        # Filter for valid coordinates
        df = df[
            (df['BC_ALBERS_X'].notna()) &
            (df['BC_ALBERS_Y'].notna())
        ].copy()

        print(f"Loaded {len(df)} plots with valid coordinates")

        # Filter to ACTIVE sites only
        active_mask = (df['PSP_STATUS'] == 'A') | (df['LAST_MSMT'] == 'N')
        df_inactive = df[~active_mask]
        df = df[active_mask].copy()

        print(f"\nFiltering to active sites:")
        print(f"  Active sites: {len(df)}")
        print(f"  Inactive sites (excluded): {len(df_inactive)}")
        print(f"  Rationale: Current satellite imagery only meaningful for active sites")

        return df

    def bc_albers_to_latlon(self, x, y):
        """
        Convert BC Albers coordinates to lat/lon.
        BC Albers is EPSG:3005, WGS84 is EPSG:4326

        Args:
            x, y: BC Albers coordinates

        Returns:
            (lon, lat) tuple in WGS84
        """
        # Create a point in BC Albers projection
        point = ee.Geometry.Point([x, y], proj='EPSG:3005')
        # Transform to WGS84
        coords = point.transform('EPSG:4326', 1).coordinates().getInfo()
        return coords[0], coords[1]  # lon, lat

    def get_sentinel2_composite(self, point, start_date, end_date, buffer=500):
        """
        Get Sentinel-2 median composite for a location.

        Args:
            point: ee.Geometry.Point
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            buffer: Buffer radius in meters

        Returns:
            ee.Image with median composite or None if no imagery
        """
        # Load Sentinel-2 Surface Reflectance
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(point) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))  # Relaxed cloud threshold

        # Check if any images exist
        count = s2.size().getInfo()
        if count == 0:
            return None

        # Cloud masking function
        def mask_s2_clouds(image):
            qa = image.select('QA60')
            # Bits 10 and 11 are clouds and cirrus
            cloud_bit_mask = 1 << 10
            cirrus_bit_mask = 1 << 11
            mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
                qa.bitwiseAnd(cirrus_bit_mask).eq(0))
            return image.updateMask(mask).divide(10000)

        # Apply cloud mask and get median
        composite = s2.map(mask_s2_clouds).median()

        return composite

    def calculate_vegetation_indices(self, image):
        """
        Calculate NDVI and EVI from Sentinel-2 image.

        Args:
            image: ee.Image with B4 (Red) and B8 (NIR) bands

        Returns:
            ee.Image with added NDVI and EVI bands
        """
        # NDVI = (NIR - Red) / (NIR + Red)
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('ndvi')

        # EVI = 2.5 * ((NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1))
        evi = image.expression(
            '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
            {
                'NIR': image.select('B8'),
                'RED': image.select('B4'),
                'BLUE': image.select('B2')
            }
        ).rename('evi')

        return image.addBands([ndvi, evi])

    def get_elevation_data(self, point):
        """
        Get elevation and terrain data from SRTM.

        Args:
            point: ee.Geometry.Point

        Returns:
            ee.Image with elevation, slope, aspect
        """
        # SRTM Digital Elevation Model
        dem = ee.Image('USGS/SRTMGL1_003')
        elevation = dem.select('elevation')

        # Calculate terrain derivatives
        terrain = ee.Algorithms.Terrain(elevation)
        slope = terrain.select('slope')
        aspect = terrain.select('aspect')

        return elevation.addBands([slope, aspect])

    def extract_point_data(self, row, year=None):
        """
        Extract all data for a single plot location.

        Args:
            row: DataFrame row with BC_ALBERS_X, BC_ALBERS_Y, MEAS_YR
            year: Optional year override

        Returns:
            Dictionary with extracted data
        """
        x, y = row['BC_ALBERS_X'], row['BC_ALBERS_Y']
        plot_id = row.get('SITE_IDENTIFIER', f'plot_{int(x)}_{int(y)}')

        # Use measurement year if available (for record keeping)
        if year is None:
            year = row.get('MEAS_YR', 2020)
            if pd.isna(year):
                year = 2020
        year = int(year)

        # For active sites, always use latest imagery (2022-2024)
        # No need to check is_active since we filtered to active sites only
        start_date = '2022-06-01'
        end_date = '2024-08-31'
        imagery_year = '2022-2024'
        is_active = True  # All sites are active now

        try:
            # Convert coordinates to lat/lon
            lon, lat = self.bc_albers_to_latlon(x, y)
            point = ee.Geometry.Point([lon, lat])

            # Buffer around point (250m radius)
            buffer_dist = 250
            region = point.buffer(buffer_dist)

            # Get Sentinel-2 composite
            s2_composite = self.get_sentinel2_composite(
                point, start_date, end_date, buffer_dist)

            # Check if we got valid imagery
            if s2_composite is None:
                return {
                    'plot_id': plot_id,
                    'x': x,
                    'y': y,
                    'lon': lon,
                    'lat': lat,
                    'measurement_year': year,
                    'imagery_period': imagery_year,
                    'is_active_site': is_active,
                    'success': False,
                    'error': 'No Sentinel-2 imagery available for this date/location'
                }

            # Add vegetation indices
            s2_composite = self.calculate_vegetation_indices(s2_composite)

            # Get elevation data
            terrain = self.get_elevation_data(point)

            # Combine all bands
            combined = s2_composite.select(
                self.sentinel_bands + ['ndvi', 'evi']).addBands(terrain)

            # Extract values at point (mean of 250m buffer)
            values = combined.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=10,  # 10m resolution
                maxPixels=1e9
            ).getInfo()

            # Prepare output dictionary
            result = {
                'plot_id': plot_id,
                'x': x,
                'y': y,
                'lon': lon,
                'lat': lat,
                'measurement_year': year,
                'imagery_period': imagery_year,
                'is_active_site': is_active,
                'success': True
            }

            # Add band values
            for i, band in enumerate(self.sentinel_bands):
                result[self.output_bands[i]] = values.get(band, np.nan)

            # Add derived products
            result['ndvi'] = values.get('ndvi', np.nan)
            result['evi'] = values.get('evi', np.nan)
            result['elevation'] = values.get('elevation', np.nan)
            result['slope'] = values.get('slope', np.nan)
            result['aspect'] = values.get('aspect', np.nan)

            return result

        except Exception as e:
            print(f"Error processing plot {plot_id}: {e}")
            return {
                'plot_id': plot_id,
                'x': x,
                'y': y,
                'measurement_year': year,
                'is_active_site': is_active,
                'success': False,
                'error': str(e)
            }

    def extract_imagery_patch(self, row, year=None, patch_size=64):
        """
        Extract image patch (not just point value) for CNN input.

        Args:
            row: DataFrame row with coordinates
            year: Optional year override
            patch_size: Size of image patch in pixels

        Returns:
            Dictionary with image data and metadata
        """
        x, y = row['BC_ALBERS_X'], row['BC_ALBERS_Y']
        plot_id = row.get('SITE_IDENTIFIER', f'plot_{int(x)}_{int(y)}')

        if year is None:
            year = row.get('MEAS_YR', 2020)
            if pd.isna(year):
                year = 2020
        year = int(year)

        # For active sites only, use latest imagery (2022-2024)
        # All sites are active since we filtered earlier
        start_date = '2022-06-01'
        end_date = '2024-08-31'
        imagery_year = '2022-2024'
        is_active = True  # All sites are active now

        try:
            # Convert coordinates
            lon, lat = self.bc_albers_to_latlon(x, y)
            point = ee.Geometry.Point([lon, lat])

            # Calculate region (640m x 640m at 10m resolution = 64x64 pixels)
            buffer_dist = (patch_size * 10) / 2  # 10m resolution
            region = point.buffer(buffer_dist).bounds()

            # Get Sentinel-2 composite
            s2_composite = self.get_sentinel2_composite(
                point, start_date, end_date, buffer_dist)
            s2_composite = self.calculate_vegetation_indices(s2_composite)

            # Select bands for image (4 channels: RGB + NIR)
            image_data = s2_composite.select(self.sentinel_bands)

            # Get the image as array
            # Note: For production, you'd download the actual image patch
            # Here we'll get the URL for download
            url = image_data.getThumbURL({
                'region': region,
                'dimensions': f'{patch_size}x{patch_size}',
                'format': 'npy'
            })

            # Also get point values for tabular features
            point_values = s2_composite.select(['ndvi', 'evi']).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point.buffer(250),
                scale=10
            ).getInfo()

            terrain = self.get_elevation_data(point).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point.buffer(250),
                scale=30
            ).getInfo()

            return {
                'plot_id': plot_id,
                'x': x,
                'y': y,
                'lon': lon,
                'lat': lat,
                'measurement_year': year,
                'imagery_period': imagery_year,
                'is_active_site': is_active,
                'image_url': url,
                'ndvi': point_values.get('ndvi', np.nan),
                'evi': point_values.get('evi', np.nan),
                'elevation': terrain.get('elevation', np.nan),
                'slope': terrain.get('slope', np.nan),
                'aspect': terrain.get('aspect', np.nan),
                'success': True
            }

        except Exception as e:
            print(f"Error processing plot {plot_id}: {e}")
            return {
                'plot_id': plot_id,
                'x': x,
                'y': y,
                'measurement_year': year,
                'is_active_site': is_active,
                'success': False,
                'error': str(e)
            }

    def batch_extract(self, df, batch_size=100, mode='point', save_interval=50):
        """
        Extract data for all plots in batches.

        Args:
            df: DataFrame with plot locations
            batch_size: Number of plots to process in memory
            mode: 'point' for point values or 'patch' for image patches
            save_interval: Save progress every N plots

        Returns:
            DataFrame with extracted data
        """
        results = []
        total = len(df)

        print(f"\n{'='*70}")
        print(f"Starting batch extraction: {total} plots")
        print(f"Mode: {mode}")
        print(f"{'='*70}\n")

        extract_func = self.extract_point_data if mode == 'point' else self.extract_imagery_patch

        for i, (idx, row) in enumerate(df.iterrows()):
            if i % 10 == 0:
                print(
                    f"Processing plot {i+1}/{total} ({(i+1)/total*100:.1f}%)")

            result = extract_func(row)
            results.append(result)

            # Save progress periodically
            if (i + 1) % save_interval == 0:
                temp_df = pd.DataFrame(results)
                temp_path = self.output_dir / \
                    f'temp_extraction_{mode}_{i+1}.csv'
                temp_df.to_csv(temp_path, index=False)
                print(f"  → Progress saved to {temp_path}")

            # Rate limiting (Earth Engine quota)
            if (i + 1) % batch_size == 0:
                print(f"  → Pausing briefly (Earth Engine rate limit)...")
                time.sleep(2)

        # Create final DataFrame
        results_df = pd.DataFrame(results)

        # Report success rate
        success_count = results_df['success'].sum()
        print(f"\n{'='*70}")
        print(f"Extraction complete!")
        print(
            f"Success: {success_count}/{total} ({success_count/total*100:.1f}%)")
        print(f"{'='*70}\n")

        return results_df

    def save_extraction_results(self, df, prefix='ee_data'):
        """
        Save extraction results to CSV and pickle.

        Args:
            df: DataFrame with extraction results
            prefix: Filename prefix
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save CSV
        csv_path = self.output_dir / f'{prefix}_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved CSV to {csv_path}")

        # Save pickle (preserves dtypes)
        pkl_path = self.output_dir / f'{prefix}_{timestamp}.pkl'
        df.to_pickle(pkl_path)
        print(f"✓ Saved pickle to {pkl_path}")

        # Save metadata
        metadata = {
            'extraction_date': timestamp,
            'num_plots': len(df),
            'success_rate': df['success'].mean(),
            'columns': list(df.columns),
            'bands_extracted': self.output_bands
        }

        meta_path = self.output_dir / f'{prefix}_{timestamp}_metadata.json'
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata to {meta_path}")

        return csv_path, pkl_path

    def visualize_sample_locations(self, df, sample_size=100):
        """
        Create an interactive map of sample locations (requires geemap).

        Args:
            df: DataFrame with lon/lat columns
            sample_size: Number of points to display
        """
        if not HAS_GEEMAP:
            print("geemap not installed. Skipping visualization.")
            return

        # Sample if too many points
        if len(df) > sample_size:
            df_sample = df.sample(sample_size, random_state=42)
        else:
            df_sample = df

        # Create map centered on BC
        Map = geemap.Map(center=[54, -125], zoom=6)

        # Add points
        features = []
        for _, row in df_sample.iterrows():
            if 'lon' in row and 'lat' in row:
                point = ee.Geometry.Point([row['lon'], row['lat']])
                features.append(ee.Feature(
                    point, {'plot_id': str(row.get('plot_id', ''))}))

        fc = ee.FeatureCollection(features)
        Map.addLayer(fc, {'color': 'red'}, 'Sample Plots')

        # Add base layers
        Map.addLayer(ee.Image('USGS/SRTMGL1_003'),
                     {'min': 0, 'max': 3000, 'palette': [
                         'blue', 'green', 'red']},
                     'Elevation', False)

        # Save map
        map_path = self.output_dir / 'sample_locations_map.html'
        Map.save(str(map_path))
        print(f"✓ Map saved to {map_path}")


def main():
    """
    Main extraction pipeline.
    """
    print("="*70)
    print("Earth Engine Data Extraction for Pacific Yew Sites")
    print("="*70)

    # Initialize extractor
    extractor = EarthEngineExtractor(output_dir='data/ee_imagery')

    # Load plot locations
    df = extractor.load_plot_locations(
        'data/raw/bc_sample_data-2025-10-09/bc_sample_data.csv')

    # Option to test with small sample first
    print("\nOptions:")
    print("1. Test with 10 sample plots")
    print("2. Process all plots (point data)")
    print("3. Process all plots (image patches)")

    choice = input("\nEnter choice (1-3) [default: 1]: ").strip() or '1'

    if choice == '1':
        print("\n--- Testing with 10 sample plots ---")
        df_sample = df.sample(min(10, len(df)), random_state=42)
        results = extractor.batch_extract(df_sample, mode='point')

    elif choice == '2':
        print("\n--- Processing all plots (point data) ---")
        results = extractor.batch_extract(df, batch_size=100, mode='point')

    elif choice == '3':
        print("\n--- Processing all plots (image patches) ---")
        results = extractor.batch_extract(df, batch_size=50, mode='patch')

    else:
        print("Invalid choice. Exiting.")
        return

    # Save results
    csv_path, pkl_path = extractor.save_extraction_results(
        results, prefix='sentinel2_data')

    # Display summary statistics
    print("\n" + "="*70)
    print("Extraction Summary Statistics")
    print("="*70)

    successful = results[results['success'] == True]

    if len(successful) > 0:
        print(f"\nSuccessful extractions: {len(successful)}")
        print("\nSentinel-2 bands (median values):")
        for band in ['blue', 'green', 'red', 'nir']:
            if band in successful.columns:
                print(
                    f"  {band.upper()}: {successful[band].median():.4f} (median)")

        print("\nVegetation indices:")
        if 'ndvi' in successful.columns:
            print(f"  NDVI: {successful['ndvi'].median():.4f} (median)")
        if 'evi' in successful.columns:
            print(f"  EVI: {successful['evi'].median():.4f} (median)")

        print("\nTerrain:")
        if 'elevation' in successful.columns:
            print(
                f"  Elevation: {successful['elevation'].median():.1f} m (median)")
        if 'slope' in successful.columns:
            print(f"  Slope: {successful['slope'].median():.1f}° (median)")

    print("\n" + "="*70)
    print("Extraction complete!")
    print(f"Results saved to: {csv_path}")
    print("="*70)

    # Optional: Create visualization
    if HAS_GEEMAP and len(successful) > 0:
        try:
            print("\nCreating visualization map...")
            extractor.visualize_sample_locations(successful)
        except Exception as e:
            print(f"Note: Could not create visualization map: {e}")
            print("This is optional and doesn't affect the extracted data.")


if __name__ == "__main__":
    main()
