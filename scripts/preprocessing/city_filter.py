#!/usr/bin/env python3
"""
Filter out observations near major Northwest cities.

This module provides functionality to exclude observations that are too close
to urban areas, which can confound the model with non-forest features.

Major NW cities included:
    - Seattle, WA
    - Portland, OR
    - Vancouver, BC
    - Victoria, BC
    - Spokane, WA
    - Eugene, OR
    - Tacoma, WA
    - Bellingham, WA
    - Salem, OR
    - Olympia, WA
    - Nanaimo, BC
    - Kelowna, BC
    - Kamloops, BC
    - Prince George, BC

Author: GitHub Copilot
Date: 2024-11-14
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict


# Major Northwest cities (name, latitude, longitude, exclusion radius in km)
NW_CITIES = [
    # Washington State
    ("Seattle", 47.6062, -122.3321, 30),
    ("Tacoma", 47.2529, -122.4443, 20),
    ("Spokane", 47.6588, -117.4260, 20),
    ("Bellingham", 48.7519, -122.4787, 15),
    ("Olympia", 47.0379, -122.9007, 15),
    ("Vancouver (WA)", 45.6387, -122.6615, 15),
    ("Everett", 47.9790, -122.2021, 15),
    ("Kent", 47.3809, -122.2348, 10),
    ("Renton", 47.4829, -122.2171, 10),
    ("Bellevue", 47.6101, -122.2015, 10),
    ("Redmond", 47.6740, -122.1215, 10),

    # Oregon
    ("Portland", 45.5152, -122.6784, 30),
    ("Eugene", 44.0521, -123.0868, 20),
    ("Salem", 44.9429, -123.0351, 15),
    ("Gresham", 45.4982, -122.4302, 10),
    ("Bend", 44.0582, -121.3153, 15),
    ("Medford", 42.3265, -122.8756, 15),
    ("Corvallis", 44.5646, -123.2620, 10),

    # British Columbia
    ("Vancouver", 49.2827, -123.1207, 30),
    ("Victoria", 48.4284, -123.3656, 20),
    ("Surrey", 49.1913, -122.8490, 15),
    ("Burnaby", 49.2488, -122.9805, 15),
    ("Richmond", 49.1666, -123.1336, 15),
    ("Nanaimo", 49.1659, -123.9401, 15),
    ("Kelowna", 49.8880, -119.4960, 15),
    ("Kamloops", 50.6745, -120.3273, 15),
    ("Prince George", 53.9171, -122.7497, 20),
    ("Abbotsford", 49.0504, -122.3045, 15),
    ("Chilliwack", 49.1577, -121.9509, 10),
    ("Coquitlam", 49.2838, -122.7932, 10),
]


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great circle distance between two points in kilometers.

    Uses the Haversine formula for accurate distance calculation on a sphere.

    Args:
        lat1, lon1: Latitude and longitude of point 1 in degrees
        lat2, lon2: Latitude and longitude of point 2 in degrees

    Returns:
        Distance in kilometers
    """
    # Earth radius in kilometers
    R = 6371.0

    # Convert to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * \
        np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance


def find_nearest_city(lat: float, lon: float) -> Tuple[str, float]:
    """
    Find the nearest city and distance to it.

    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees

    Returns:
        Tuple of (city_name, distance_km)
    """
    min_distance = float('inf')
    nearest_city = None

    for city_name, city_lat, city_lon, _ in NW_CITIES:
        distance = haversine_distance(lat, lon, city_lat, city_lon)
        if distance < min_distance:
            min_distance = distance
            nearest_city = city_name

    return nearest_city, min_distance


def is_near_city(lat: float, lon: float, min_distance_km: float = None) -> Tuple[bool, str, float]:
    """
    Check if a location is near any major NW city.

    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        min_distance_km: Override exclusion radius (use city-specific if None)

    Returns:
        Tuple of (is_too_close, nearest_city, distance_km)
    """
    for city_name, city_lat, city_lon, exclusion_radius in NW_CITIES:
        distance = haversine_distance(lat, lon, city_lat, city_lon)

        # Use city-specific radius or override
        threshold = min_distance_km if min_distance_km is not None else exclusion_radius

        if distance < threshold:
            return True, city_name, distance

    # Find nearest city even if not excluded
    nearest_city, nearest_distance = find_nearest_city(lat, lon)
    return False, nearest_city, nearest_distance


def filter_dataframe(df: pd.DataFrame,
                     lat_col: str = 'latitude',
                     lon_col: str = 'longitude',
                     min_distance_km: float = None,
                     add_city_info: bool = True) -> pd.DataFrame:
    """
    Filter DataFrame to exclude observations near cities.

    Args:
        df: DataFrame with latitude and longitude columns
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        min_distance_km: Minimum distance from cities in km (use city-specific if None)
        add_city_info: Whether to add nearest_city and distance_to_city columns

    Returns:
        Filtered DataFrame
    """
    # Initialize columns
    df = df.copy()
    df['near_city'] = False
    df['nearest_city'] = ''
    df['distance_to_city'] = np.nan

    # Check each row
    for idx, row in df.iterrows():
        lat = row[lat_col]
        lon = row[lon_col]

        if pd.isna(lat) or pd.isna(lon):
            continue

        is_near, nearest, distance = is_near_city(lat, lon, min_distance_km)
        df.at[idx, 'near_city'] = is_near
        df.at[idx, 'nearest_city'] = nearest
        df.at[idx, 'distance_to_city'] = distance

    # Count exclusions
    n_excluded = df['near_city'].sum()
    print(f"Excluded {n_excluded} observations near cities")

    if n_excluded > 0:
        print("\nExclusions by city:")
        excluded = df[df['near_city']]
        for city in excluded['nearest_city'].value_counts().head(10).items():
            print(f"  {city[0]}: {city[1]} observations")

    # Filter out near-city observations
    filtered_df = df[~df['near_city']].copy()

    # Optionally keep city info columns
    if not add_city_info:
        filtered_df = filtered_df.drop(
            columns=['near_city', 'nearest_city', 'distance_to_city'])

    return filtered_df


def get_city_summary(df: pd.DataFrame,
                     lat_col: str = 'latitude',
                     lon_col: str = 'longitude') -> pd.DataFrame:
    """
    Generate summary of observations by nearest city.

    Args:
        df: DataFrame with latitude and longitude
        lat_col: Name of latitude column
        lon_col: Name of longitude column

    Returns:
        DataFrame with city summary statistics
    """
    # Find nearest city for each observation
    cities = []
    distances = []

    for _, row in df.iterrows():
        lat = row[lat_col]
        lon = row[lon_col]

        if pd.isna(lat) or pd.isna(lon):
            cities.append(None)
            distances.append(np.nan)
            continue

        nearest, distance = find_nearest_city(lat, lon)
        cities.append(nearest)
        distances.append(distance)

    # Create summary DataFrame
    summary_df = pd.DataFrame({
        'nearest_city': cities,
        'distance_km': distances
    })

    # Group by city
    city_summary = summary_df.groupby('nearest_city').agg({
        'distance_km': ['count', 'mean', 'min', 'max']
    }).round(2)

    city_summary.columns = ['count', 'mean_distance_km',
                            'min_distance_km', 'max_distance_km']
    city_summary = city_summary.sort_values('count', ascending=False)

    return city_summary


def main():
    """Test city filtering functionality."""
    import argparse

    parser = argparse.ArgumentParser(description='Test city filtering')
    parser.add_argument('--csv', type=str, required=True,
                        help='CSV file with lat/lon columns')
    parser.add_argument('--lat-col', type=str, default='latitude',
                        help='Latitude column name')
    parser.add_argument('--lon-col', type=str, default='longitude',
                        help='Longitude column name')
    parser.add_argument('--min-distance', type=float, default=None,
                        help='Minimum distance from cities in km')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path (optional)')

    args = parser.parse_args()

    # Load data
    print(f"Loading {args.csv}...")
    df = pd.read_csv(args.csv)
    print(f"  Total observations: {len(df)}")

    # Generate summary before filtering
    print("\nObservations by nearest city (before filtering):")
    summary = get_city_summary(df, args.lat_col, args.lon_col)
    print(summary.to_string())

    # Apply filter
    print("\nApplying city filter...")
    filtered_df = filter_dataframe(
        df,
        lat_col=args.lat_col,
        lon_col=args.lon_col,
        min_distance_km=args.min_distance,
        add_city_info=True
    )

    print(f"\nRemaining observations: {len(filtered_df)}")
    print(f"Retention rate: {len(filtered_df)/len(df)*100:.1f}%")

    # Save if requested
    if args.output:
        filtered_df.to_csv(args.output, index=False)
        print(f"\nSaved filtered data to: {args.output}")


if __name__ == '__main__':
    main()
