#!/usr/bin/env python3
"""
Pacific Yew (Taxus brevifolia) Analysis Script for BC Sample Data
================================================================

This script analyzes Pacific Yew distribution and ecological associations using
the BC sample data (bc_sample_data.csv) format, where species composition is 
encoded in the SPB_CPCT_LS column as concatenated strings like "CW044FD038HW012TW004EP002".

Author: Analysis Tool
Date: October 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter, defaultdict
from scipy import stats


def load_reference_data():
    """
    Load reference data for species codes and biogeoclimatic zones.
    """
    # Create dictionaries to hold the mappings
    species_map = {}
    bec_zone_map = {}
    bec_subzone_map = {}
    bec_variant_map = {}

    # Load species code mappings
    try:
        with open('tree_name_keys.txt', 'r') as f:
            next(f)  # Skip header
            for line in f:
                if line.strip():
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        code = parts[0].strip()
                        name = parts[1].strip()
                        species_map[code] = name
        print(f"Loaded {len(species_map)} species name mappings")
    except FileNotFoundError:
        print(
            "Warning: tree_name_keys.txt not found. Species codes will be displayed as-is.")

    # Load BEC zone mappings
    try:
        with open('biogeoclimactic_zone_keys.txt', 'r') as f:
            next(f)  # Skip header
            for line in f:
                if line.strip():
                    parts = line.strip().split('\t')
                    if len(parts) >= 4:
                        zone_code = parts[0].strip()
                        zone_name = parts[1].strip()
                        subzone_code = parts[2].strip()
                        subzone_name = parts[3].strip()

                        # Map zone codes to names
                        if zone_code and zone_name:
                            bec_zone_map[zone_code] = zone_name

                        # Map subzone codes to names
                        if subzone_code and subzone_name:
                            bec_subzone_map[subzone_code] = subzone_name

                        # For variants, we'll need to parse the BECLABEL column directly
                        # as variants are typically single characters after the subzone
        print(
            f"Loaded {len(bec_zone_map)} BEC zone mappings and {len(bec_subzone_map)} subzone mappings")
    except FileNotFoundError:
        print("Warning: biogeoclimactic_zone_keys.txt not found. BEC codes will be displayed as-is.")

    return species_map, bec_zone_map, bec_subzone_map, bec_variant_map


def parse_species_composition(composition_string):
    """
    Parse a species composition string like "CW044FD038HW012TW004EP002" into
    individual species and their percentages.

    Args:
        composition_string: String containing species codes and percentages

    Returns:
        Dictionary with species codes as keys and percentages as values
    """
    if not composition_string or composition_string == "" or pd.isna(composition_string):
        return {}

    # Use regex to find all species-percentage pairs
    # Pattern: 2-3 letters followed by 2-3 digits
    pattern = r'([A-Z]{2,3})(\d{2,3})'
    matches = re.findall(pattern, composition_string)

    species_dict = {}
    for species_code, percentage_str in matches:
        try:
            percentage = int(percentage_str)
            species_dict[species_code] = percentage
        except ValueError:
            continue

    return species_dict


def parse_bec_label(bec_label):
    """
    Parse a BEC label like "ICHmw3" into zone, subzone, and variant components.

    Args:
        bec_label: String containing the full BEC designation

    Returns:
        Tuple of (zone, subzone, variant)
    """
    if not bec_label or pd.isna(bec_label):
        return None, None, None

    # BEC labels typically follow the pattern: ZONE + subzone + variant
    # e.g., ICHmw3, BWBSmk, CWHvm1, etc.

    # Extract the zone (usually 2-4 uppercase letters at the start)
    zone_match = re.match(r'^([A-Z]{2,4})', bec_label)
    zone = zone_match.group(1) if zone_match else None

    if zone:
        remainder = bec_label[len(zone):]
        # Extract subzone (usually 1-2 lowercase letters)
        subzone_match = re.match(r'^([a-z]{1,2})', remainder)
        subzone = subzone_match.group(1) if subzone_match else None

        if subzone:
            # Extract variant (remaining characters, usually digits or letters)
            variant = remainder[len(subzone):] if len(
                remainder) > len(subzone) else None
        else:
            variant = remainder if remainder else None
    else:
        subzone = None
        variant = None

    return zone, subzone, variant


def load_data():
    """
    Load the BC sample data.
    """
    print("Loading BC sample data...")

    try:
        sample_data = pd.read_csv(
            'bc_sample_data-2025-10-09/bc_sample_data.csv')
        print(f"Loaded {len(sample_data)} sample records")
    except FileNotFoundError:
        print("Error: bc_sample_data.csv file not found!")
        return None

    return sample_data


def identify_yew_sites(sample_data):
    """
    Identify sites that contain Pacific Yew trees by parsing the SPB_CPCT_LS column.

    Args:
        sample_data: DataFrame containing sample data

    Returns:
        Dictionary with site IDs as keys and yew percentage as values
    """
    if sample_data is None:
        return {}

    print("\nIdentifying sites with Pacific Yew trees...")

    yew_sites = {}
    total_sites_checked = 0

    for idx, row in sample_data.iterrows():
        total_sites_checked += 1
        site_id = row['SITE_IDENTIFIER']
        composition = row['SPB_CPCT_LS']

        species_dict = parse_species_composition(composition)

        if 'TW' in species_dict:
            yew_percentage = species_dict['TW']
            yew_sites[site_id] = yew_percentage

    print(f"Checked {total_sites_checked} sample records")
    print(f"Found Pacific Yew at {len(yew_sites)} distinct sites")

    if yew_sites:
        total_yew_percentage = sum(yew_sites.values())
        avg_yew_percentage = total_yew_percentage / len(yew_sites)
        max_yew_percentage = max(yew_sites.values())
        print(
            f"Average Pacific Yew percentage at yew sites: {avg_yew_percentage:.1f}%")
        print(f"Maximum Pacific Yew percentage: {max_yew_percentage}%")

    return yew_sites


def find_associated_species(sample_data, yew_sites):
    """
    Identify tree species that commonly occur with Pacific Yew and analyze their relationships.

    Args:
        sample_data: DataFrame containing sample data
        yew_sites: Dictionary of sites with yew trees

    Returns:
        DataFrame of species associations with ecological metrics
    """
    if sample_data is None or not yew_sites:
        return pd.DataFrame()

    print("\nAnalyzing species associations with Pacific Yew...")

    # Get all samples from sites where yew is found
    yew_site_ids = list(yew_sites.keys())
    yew_site_samples = sample_data[sample_data['SITE_IDENTIFIER'].isin(
        yew_site_ids)]

    # Count species co-occurrences and collect metrics
    species_counts = defaultdict(int)
    species_percentages = defaultdict(list)
    species_sites = defaultdict(set)
    yew_percentages_by_site = defaultdict(list)

    for idx, row in yew_site_samples.iterrows():
        site_id = row['SITE_IDENTIFIER']
        composition = row['SPB_CPCT_LS']

        species_dict = parse_species_composition(composition)

        if 'TW' in species_dict:  # Confirm this sample has yew
            yew_pct = species_dict['TW']

            for species, percentage in species_dict.items():
                if species != 'TW':  # Don't count yew with itself
                    species_counts[species] += 1
                    species_percentages[species].append(percentage)
                    species_sites[species].add(site_id)
                    yew_percentages_by_site[species].append(yew_pct)

    # Calculate association metrics
    association_data = []
    total_yew_sites = len(yew_sites)

    for species, count in species_counts.items():
        if count > 0:
            # Calculate co-occurrence metrics
            co_occurrence_rate = (count / total_yew_sites) * 100
            avg_percentage = np.mean(species_percentages[species])
            max_percentage = np.max(species_percentages[species])
            min_percentage = np.min(species_percentages[species])

            # Calculate average yew percentage when this species is present
            avg_yew_when_present = np.mean(yew_percentages_by_site[species])

            association_data.append({
                'Species': species,
                'Co_occurrence_Count': count,
                'Co_occurrence_Rate_%': co_occurrence_rate,
                'Sites_with_Species': len(species_sites[species]),
                'Avg_Species_Percentage': avg_percentage,
                'Max_Species_Percentage': max_percentage,
                'Min_Species_Percentage': min_percentage,
                'Avg_Yew_Percentage_When_Present': avg_yew_when_present,
                # Composite metric
                'Association_Strength': co_occurrence_rate * (avg_percentage / 100)
            })

    # Convert to DataFrame and sort by co-occurrence count (descending)
    associations_df = pd.DataFrame(association_data)

    if len(associations_df) > 0:
        associations_df = associations_df.sort_values(
            'Co_occurrence_Count', ascending=False).reset_index(drop=True)
        print(
            f"Found {len(associations_df)} species co-occurring with Pacific Yew")

        # Display top associations
        print("\nTop 10 species associations:")
        for idx, row in associations_df.head(10).iterrows():
            print(f"  {idx+1:2d}. {row['Species']:3s}: {row['Co_occurrence_Count']:3d} sites "
                  f"({row['Co_occurrence_Rate_%']:4.1f}%), avg {row['Avg_Species_Percentage']:4.1f}% composition")

    return associations_df


def find_bec_zones(sample_data, yew_sites):
    """
    Identify biogeoclimatic zones, subzones, and variants where Pacific Yew is commonly found.
    Calculate density-normalized metrics using tree stems per hectare data.

    Args:
        sample_data: DataFrame containing sample data
        yew_sites: Dictionary of sites with yew trees

    Returns:
        tuple: (zone_df, subzone_df, variant_df) - DataFrames for zones, subzones and variants
    """
    if sample_data is None or not yew_sites:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    print("\nAnalyzing biogeoclimatic zones for Pacific Yew with density normalization...")

    # Filter sample data for sites with yew trees
    yew_site_ids = list(yew_sites.keys())
    yew_site_samples = sample_data[sample_data['SITE_IDENTIFIER'].isin(
        yew_site_ids)]

    if len(yew_site_samples) == 0:
        print("No sample data found for yew sites")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Parse BEC labels and collect zone information with density data
    zone_data = []

    for idx, row in yew_site_samples.iterrows():
        site_id = row['SITE_IDENTIFIER']
        bec_label = row['BECLABEL']
        bec_zone = row['BEC_ZONE']
        yew_percentage = yew_sites.get(site_id, 0)
        stems_ha = row['STEMS_HA_LS'] if pd.notna(row['STEMS_HA_LS']) else 0
        meas_date = row['MEAS_DT']
        meas_year = row['MEAS_YR'] if pd.notna(row['MEAS_YR']) else None

        # Calculate yew density (stems/ha) from percentage and total stems
        yew_density = (yew_percentage / 100.0) * \
            stems_ha if stems_ha > 0 else 0

        # Parse BEC label
        zone, subzone, variant = parse_bec_label(bec_label)

        # Use BEC_ZONE column as primary zone if parsing fails
        if not zone and bec_zone:
            zone = bec_zone

        zone_data.append({
            'SITE_IDENTIFIER': site_id,
            'BEC_Zone': zone,
            'BEC_Subzone': subzone,
            'BEC_Variant': variant,
            'BEC_Label': bec_label,
            'Yew_Percentage': yew_percentage,
            'Total_Stems_HA': stems_ha,
            'Yew_Density_HA': yew_density,
            'Measurement_Date': meas_date,
            'Measurement_Year': meas_year
        })

    zone_df_raw = pd.DataFrame(zone_data)

    # ===== ZONE LEVEL ANALYSIS WITH DENSITY =====
    zone_groups = zone_df_raw.groupby('BEC_Zone')
    zone_counts = zone_df_raw['BEC_Zone'].value_counts()
    total_sites = len(zone_df_raw)
    zone_percentages = (zone_counts / total_sites) * 100

    # Calculate comprehensive zone statistics
    zone_stats = zone_groups.agg({
        'Yew_Percentage': ['mean', 'max', 'std'],
        'Total_Stems_HA': ['mean', 'sum', 'std'],
        'Yew_Density_HA': ['mean', 'sum', 'max', 'std']
    }).reset_index()

    # Flatten column names
    zone_stats.columns = ['BEC_Zone', 'Avg_Yew_Percentage', 'Max_Yew_Percentage', 'Std_Yew_Percentage',
                          'Avg_Total_Stems_HA', 'Sum_Total_Stems_HA', 'Std_Total_Stems_HA',
                          'Avg_Yew_Density_HA', 'Sum_Yew_Density_HA', 'Max_Yew_Density_HA', 'Std_Yew_Density_HA']

    bec_df = pd.DataFrame({
        'BEC_Zone': zone_counts.index,
        'Count': zone_counts.values,
        'Percentage': zone_percentages.values
    })

    # Merge with density statistics
    bec_df = bec_df.merge(zone_stats, on='BEC_Zone', how='left')

    # Calculate normalized density metrics
    bec_df['Yew_Density_Per_Total_Density'] = bec_df['Avg_Yew_Density_HA'] / \
        bec_df['Avg_Total_Stems_HA']
    bec_df['Yew_Density_Per_Total_Density'] = bec_df['Yew_Density_Per_Total_Density'].fillna(
        0)

    bec_df = bec_df.sort_values(
        'Avg_Yew_Density_HA', ascending=False).reset_index(drop=True)

    # ===== SUBZONE LEVEL ANALYSIS WITH DENSITY =====
    zone_df_raw['Zone_Subzone'] = zone_df_raw['BEC_Zone'].astype(
        str) + zone_df_raw['BEC_Subzone'].astype(str)
    subzone_groups = zone_df_raw.groupby('Zone_Subzone')
    subzone_counts = zone_df_raw['Zone_Subzone'].value_counts()
    subzone_percentages = (subzone_counts / total_sites) * 100

    subzone_stats = subzone_groups.agg({
        'Yew_Percentage': ['mean', 'max'],
        'Total_Stems_HA': ['mean'],
        'Yew_Density_HA': ['mean', 'max']
    }).reset_index()

    subzone_stats.columns = ['Zone_Subzone', 'Avg_Yew_Percentage', 'Max_Yew_Percentage',
                             'Avg_Total_Stems_HA', 'Avg_Yew_Density_HA', 'Max_Yew_Density_HA']

    subzone_df = pd.DataFrame({
        'Zone_Subzone': subzone_counts.index,
        'Count': subzone_counts.values,
        'Percentage': subzone_percentages.values
    })

    subzone_df = subzone_df.merge(subzone_stats, on='Zone_Subzone', how='left')
    subzone_df['Yew_Density_Per_Total_Density'] = subzone_df['Avg_Yew_Density_HA'] / \
        subzone_df['Avg_Total_Stems_HA']
    subzone_df['Yew_Density_Per_Total_Density'] = subzone_df['Yew_Density_Per_Total_Density'].fillna(
        0)
    subzone_df = subzone_df.sort_values(
        'Avg_Yew_Density_HA', ascending=False).reset_index(drop=True)

    # ===== VARIANT LEVEL ANALYSIS WITH DENSITY =====
    variant_groups = zone_df_raw.groupby('BEC_Label')
    variant_counts = zone_df_raw['BEC_Label'].value_counts()
    variant_percentages = (variant_counts / total_sites) * 100

    variant_stats = variant_groups.agg({
        'Yew_Percentage': ['mean', 'max'],
        'Total_Stems_HA': ['mean'],
        'Yew_Density_HA': ['mean', 'max']
    }).reset_index()

    variant_stats.columns = ['BEC_Label', 'Avg_Yew_Percentage', 'Max_Yew_Percentage',
                             'Avg_Total_Stems_HA', 'Avg_Yew_Density_HA', 'Max_Yew_Density_HA']

    variant_df = pd.DataFrame({
        'BEC_Label': variant_counts.index,
        'Count': variant_counts.values,
        'Percentage': variant_percentages.values
    })

    variant_df = variant_df.merge(variant_stats, on='BEC_Label', how='left')
    variant_df['Yew_Density_Per_Total_Density'] = variant_df['Avg_Yew_Density_HA'] / \
        variant_df['Avg_Total_Stems_HA']
    variant_df['Yew_Density_Per_Total_Density'] = variant_df['Yew_Density_Per_Total_Density'].fillna(
        0)
    variant_df = variant_df.sort_values(
        'Avg_Yew_Density_HA', ascending=False).reset_index(drop=True)

    print(f"Found {len(bec_df)} BEC zones, {len(subzone_df)} subzones, and {len(variant_df)} full BEC designations")
    print("Analysis includes density normalization (yew stems/ha normalized by total stems/ha)")

    return bec_df, subzone_df, variant_df


def analyze_yew_characteristics(sample_data, yew_sites):
    """
    Analyze characteristics of Pacific Yew occurrence in the dataset.

    Args:
        sample_data: DataFrame containing sample data
        yew_sites: Dictionary of sites with yew trees

    Returns:
        Dictionary with yew statistics
    """
    if sample_data is None or not yew_sites:
        return {}

    print("\nAnalyzing Pacific Yew characteristics...")

    yew_percentages = list(yew_sites.values())

    # Calculate statistics
    stats = {
        'site_count': len(yew_sites),
        'avg_percentage': np.mean(yew_percentages),
        'median_percentage': np.median(yew_percentages),
        'max_percentage': np.max(yew_percentages),
        'min_percentage': np.min(yew_percentages),
        'std_percentage': np.std(yew_percentages),
        'total_samples': len(sample_data),
        'percentage_of_sites_with_yew': (len(yew_sites) / len(sample_data['SITE_IDENTIFIER'].unique())) * 100
    }

    return stats


def analyze_yew_temporal_distribution(sample_data, yew_sites):
    """
    Analyze the temporal distribution of Pacific Yew measurements.

    Args:
        sample_data: DataFrame containing sample data
        yew_sites: Dictionary of sites with yew trees

    Returns:
        DataFrame with temporal analysis data
    """
    if sample_data is None or not yew_sites:
        return pd.DataFrame()

    print("\nAnalyzing temporal distribution of Pacific Yew measurements...")

    # Filter for sites with yew trees
    yew_site_ids = list(yew_sites.keys())
    yew_site_samples = sample_data[sample_data['SITE_IDENTIFIER'].isin(
        yew_site_ids)]

    temporal_data = []

    for idx, row in yew_site_samples.iterrows():
        site_id = row['SITE_IDENTIFIER']
        meas_date = row['MEAS_DT']
        meas_year = row['MEAS_YR']
        yew_percentage = yew_sites.get(site_id, 0)
        bec_zone = row['BEC_ZONE']
        bec_label = row['BECLABEL']
        stems_ha = row['STEMS_HA_LS'] if pd.notna(row['STEMS_HA_LS']) else 0

        # Calculate yew density
        yew_density = (yew_percentage / 100.0) * \
            stems_ha if stems_ha > 0 else 0

        temporal_data.append({
            'SITE_IDENTIFIER': site_id,
            'Measurement_Date': meas_date,
            'Measurement_Year': meas_year,
            'Yew_Percentage': yew_percentage,
            'Yew_Density_HA': yew_density,
            'BEC_Zone': bec_zone,
            'BEC_Label': bec_label,
            'Total_Stems_HA': stems_ha
        })

    temporal_df = pd.DataFrame(temporal_data)

    # Convert measurement date to datetime
    temporal_df['Measurement_Date'] = pd.to_datetime(
        temporal_df['Measurement_Date'], errors='coerce')

    # Extract additional temporal features
    temporal_df['Year'] = temporal_df['Measurement_Date'].dt.year
    temporal_df['Month'] = temporal_df['Measurement_Date'].dt.month
    temporal_df['Season'] = temporal_df['Month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })

    print(f"Temporal analysis covers {len(temporal_df)} yew measurements")
    if len(temporal_df) > 0:
        valid_dates = temporal_df['Measurement_Date'].dropna()
        if len(valid_dates) > 0:
            print(
                f"Date range: {valid_dates.min().strftime('%Y-%m-%d')} to {valid_dates.max().strftime('%Y-%m-%d')}")

    return temporal_df


def visualize_results(associations_df, bec_data, yew_stats, sample_data=None, yew_sites=None, temporal_df=None):
    """
    Create visualizations of the results including density analysis and temporal distribution.

    Args:
        associations_df: DataFrame of species associations
        bec_data: Tuple of (bec_df, subzone_df, variant_df) DataFrames
        yew_stats: Dictionary with yew statistics
        sample_data: Original sample data for additional visualizations
        yew_sites: Dictionary of yew sites for additional visualizations
        temporal_df: DataFrame with temporal analysis data
    """
    # Unpack BEC zone data
    bec_df, subzone_df, variant_df = bec_data
    print("\nGenerating visualizations with density analysis...")

    # Set up the figure layout - now with 6 subplots
    fig = plt.figure(figsize=(20, 16))

    # 1. Top species associations
    if len(associations_df) > 0:
        plt.subplot(3, 2, 1)
        top_associations = associations_df.head(15)
        bars = plt.bar(range(len(top_associations)), top_associations['Co_occurrence_Count'],
                       color='darkgreen', alpha=0.7)
        plt.xlabel('Species Code')
        plt.ylabel('Number of Co-occurrences')
        plt.title('Top 15 Species Co-occurring with Pacific Yew')
        plt.xticks(range(len(top_associations)),
                   top_associations['Species'], rotation=45)

        # Add percentage labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            rate = top_associations.iloc[i]['Co_occurrence_Rate_%']
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                     f'{rate:.1f}%', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()

    # 2. BEC Zone distribution by density (NEW - density normalized)
    if len(bec_df) > 0:
        plt.subplot(3, 2, 2)
        top_zones = bec_df.head(10)

        # Create a dual-axis plot showing both count and normalized density
        x_pos = range(len(top_zones))

        # Primary axis - site count
        bars1 = plt.bar([x - 0.2 for x in x_pos], top_zones['Count'],
                        width=0.4, color='skyblue', alpha=0.7, label='Site Count')

        # Secondary axis - normalized density
        ax2 = plt.gca().twinx()
        bars2 = ax2.bar([x + 0.2 for x in x_pos], top_zones['Yew_Density_Per_Total_Density'] * 100,
                        width=0.4, color='darkred', alpha=0.7, label='Normalized Density (%)')

        plt.gca().set_xlabel('BEC Zone')
        plt.gca().set_ylabel('Number of Sites with Pacific Yew', color='skyblue')
        ax2.set_ylabel('Yew Density / Total Density (%)', color='darkred')
        plt.gca().set_title('BEC Zones: Site Count vs Normalized Yew Density')
        plt.gca().set_xticks(x_pos)
        plt.gca().set_xticklabels(top_zones['BEC_Zone'], rotation=45)

        # Add legends
        plt.gca().legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.tight_layout()

    # 3. Pacific Yew density distribution (NEW - shows actual density in stems/ha)
    if len(bec_df) > 0:
        plt.subplot(3, 2, 3)

        # Get all yew density values from temporal_df if available
        if temporal_df is not None and len(temporal_df) > 0:
            yew_densities = temporal_df['Yew_Density_HA'].values
            # Only positive densities
            yew_densities = yew_densities[yew_densities > 0]

            if len(yew_densities) > 0:
                plt.hist(yew_densities, bins=20, color='darkred',
                         alpha=0.7, edgecolor='black')
                plt.xlabel('Pacific Yew Density (stems/hectare)')
                plt.ylabel('Number of Sites')
                plt.title('Distribution of Pacific Yew Density')
                plt.axvline(np.mean(yew_densities), color='red', linestyle='--',
                            label=f'Mean: {np.mean(yew_densities):.1f} stems/ha')
                plt.legend()
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'No density data available',
                         ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Pacific Yew Density Distribution')
        else:
            plt.text(0.5, 0.5, 'No temporal data available',
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Pacific Yew Density Distribution')

    # 4. Temporal distribution (NEW - measurement dates)
    if temporal_df is not None and len(temporal_df) > 0:
        plt.subplot(3, 2, 4)

        # Filter for valid dates
        valid_temporal = temporal_df[temporal_df['Measurement_Date'].notna()]

        if len(valid_temporal) > 0:
            # Plot measurement dates over time
            dates = valid_temporal['Measurement_Date']
            yew_densities = valid_temporal['Yew_Density_HA']

            # Create scatter plot with date on x-axis
            scatter = plt.scatter(dates, yew_densities, alpha=0.6, s=50,
                                  c=valid_temporal['Yew_Percentage'], cmap='Reds')

            plt.xlabel('Measurement Date')
            plt.ylabel('Yew Density (stems/ha)')
            plt.title('Pacific Yew Measurements Over Time')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)

            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('Yew Percentage (%)')

            # Format x-axis
            import matplotlib.dates as mdates
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        else:
            plt.text(0.5, 0.5, 'No valid measurement dates',
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Pacific Yew Measurements Over Time')

    # 5. Seasonal distribution (NEW)
    if temporal_df is not None and len(temporal_df) > 0:
        plt.subplot(3, 2, 5)

        # Count measurements by season
        season_counts = temporal_df['Season'].value_counts()
        if len(season_counts) > 0:
            colors = {'Spring': 'lightgreen', 'Summer': 'gold',
                      'Fall': 'orange', 'Winter': 'lightblue'}
            season_colors = [colors.get(season, 'gray')
                             for season in season_counts.index]

            bars = plt.bar(season_counts.index, season_counts.values,
                           color=season_colors, alpha=0.7)
            plt.xlabel('Season')
            plt.ylabel('Number of Yew Measurements')
            plt.title('Seasonal Distribution of Pacific Yew Measurements')

            # Add count labels on bars
            for bar, count in zip(bars, season_counts.values):
                plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                         str(count), ha='center', va='bottom')
        else:
            plt.text(0.5, 0.5, 'No seasonal data available',
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Seasonal Distribution of Pacific Yew Measurements')

    # 6. Density vs Total Forest Density by BEC Zone (NEW - ecological insight)
    if len(bec_df) > 0:
        plt.subplot(3, 2, 6)

        # Filter out zones with very low sample sizes for clarity
        significant_zones = bec_df[bec_df['Count'] >= 2]

        if len(significant_zones) > 0:
            scatter = plt.scatter(significant_zones['Avg_Total_Stems_HA'],
                                  significant_zones['Avg_Yew_Density_HA'],
                                  # Size by sample count
                                  s=significant_zones['Count']*20,
                                  alpha=0.6, c=significant_zones['Yew_Density_Per_Total_Density']*100,
                                  cmap='Reds')

            # Add zone labels for significant points
            for idx, row in significant_zones.iterrows():
                plt.annotate(row['BEC_Zone'],
                             (row['Avg_Total_Stems_HA'],
                              row['Avg_Yew_Density_HA']),
                             xytext=(5, 5), textcoords='offset points', fontsize=9)

            plt.xlabel('Average Total Forest Density (stems/ha)')
            plt.ylabel('Average Yew Density (stems/ha)')
            plt.title(
                'Pacific Yew vs Total Forest Density by BEC Zone\n(Bubble size = sample count)')
            plt.grid(True, alpha=0.3)

            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('Normalized Yew Density (%)')
        else:
            plt.text(0.5, 0.5, 'Insufficient data for density comparison',
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Pacific Yew vs Total Forest Density by BEC Zone')

    # Add a main title to the figure
    plt.suptitle('Pacific Yew (Taxus brevifolia) Ecological Analysis - BC Sample Data\nDensity-Normalized Analysis with Temporal Distribution',
                 fontsize=16, fontweight='bold', y=0.98)

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('pacific_yew_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualizations saved to 'pacific_yew_analysis.png'")

    # Create a separate figure for detailed BEC zone analysis
    visualize_bec_details(bec_df, subzone_df, variant_df)


def visualize_bec_details(bec_df, subzone_df, variant_df):
    """
    Create detailed visualizations of BEC zones, subzones and variants with density metrics.

    Args:
        bec_df: DataFrame with BEC zone data
        subzone_df: DataFrame with subzone data
        variant_df: DataFrame with variant data
    """
    if len(subzone_df) == 0:
        print("No subzone data available for detailed BEC visualization")
        return

    print("Generating detailed BEC zone visualizations with density analysis...")

    # Create a new figure for BEC details
    fig = plt.figure(figsize=(16, 12))

    # 1. Subzone analysis with density
    plt.subplot(3, 1, 1)
    top_subzones = subzone_df.head(15)

    # Create dual bars showing count and normalized density
    x_pos = range(len(top_subzones))
    width = 0.35

    bars1 = plt.bar([x - width/2 for x in x_pos], top_subzones['Count'],
                    width, color='forestgreen', alpha=0.7, label='Site Count')

    # Secondary axis for normalized density
    ax2 = plt.gca().twinx()
    bars2 = ax2.bar([x + width/2 for x in x_pos], top_subzones['Yew_Density_Per_Total_Density'] * 100,
                    width, color='darkred', alpha=0.7, label='Normalized Density (%)')

    plt.gca().set_xlabel('BEC Zone + Subzone')
    plt.gca().set_ylabel('Number of Sites', color='forestgreen')
    ax2.set_ylabel('Yew Density / Total Density (%)', color='darkred')
    plt.gca().set_title('Top 15 BEC Subzones: Site Count vs Normalized Yew Density')
    plt.gca().set_xticks(x_pos)
    plt.gca().set_xticklabels(top_subzones['Zone_Subzone'], rotation=45)

    # Add legends
    plt.gca().legend(loc='upper left')
    ax2.legend(loc='upper right')

    # 2. Full BEC designations with density metrics
    plt.subplot(3, 1, 2)
    top_variants = variant_df.head(15)

    x_pos = range(len(top_variants))
    bars1 = plt.bar([x - width/2 for x in x_pos], top_variants['Count'],
                    width, color='darkorange', alpha=0.7, label='Site Count')

    # Secondary axis for normalized density
    ax2 = plt.gca().twinx()
    bars2 = ax2.bar([x + width/2 for x in x_pos], top_variants['Yew_Density_Per_Total_Density'] * 100,
                    width, color='darkblue', alpha=0.7, label='Normalized Density (%)')

    plt.gca().set_xlabel('Full BEC Designation')
    plt.gca().set_ylabel('Number of Sites', color='darkorange')
    ax2.set_ylabel('Yew Density / Total Density (%)', color='darkblue')
    plt.gca().set_title('Top 15 Full BEC Designations: Site Count vs Normalized Yew Density')
    plt.gca().set_xticks(x_pos)
    plt.gca().set_xticklabels(top_variants['BEC_Label'], rotation=45)

    # Add legends
    plt.gca().legend(loc='upper left')
    ax2.legend(loc='upper right')

    # 3. Density efficiency analysis (yew density vs total forest density)
    plt.subplot(3, 1, 3)
    if len(bec_df) > 0:
        # Filter for zones with sufficient data
        significant_zones = bec_df[bec_df['Count'] >= 2]

        if len(significant_zones) > 0:
            # Create bubble plot showing relationship between total forest density and yew efficiency
            scatter = plt.scatter(significant_zones['Avg_Total_Stems_HA'],
                                  significant_zones['Yew_Density_Per_Total_Density'] * 100,
                                  # Size by absolute yew density
                                  s=significant_zones['Avg_Yew_Density_HA']*100,
                                  alpha=0.6, c=significant_zones['Count'], cmap='viridis')

            # Add zone labels
            for idx, row in significant_zones.iterrows():
                plt.annotate(row['BEC_Zone'],
                             (row['Avg_Total_Stems_HA'],
                              row['Yew_Density_Per_Total_Density']*100),
                             xytext=(5, 5), textcoords='offset points', fontsize=9)

            plt.xlabel('Average Total Forest Density (stems/ha)')
            plt.ylabel('Normalized Yew Density (%)')
            plt.title(
                'BEC Zone Efficiency: Yew Density Relative to Total Forest Density\n(Bubble size = absolute yew density)')
            plt.grid(True, alpha=0.3)

            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('Number of Sites')

            # Add trend line if enough data points
            if len(significant_zones) >= 3:
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    significant_zones['Avg_Total_Stems_HA'],
                    significant_zones['Yew_Density_Per_Total_Density'] * 100
                )
                line = slope * \
                    significant_zones['Avg_Total_Stems_HA'] + intercept
                plt.plot(significant_zones['Avg_Total_Stems_HA'], line, 'r--', alpha=0.8,
                         label=f'Trend (R²={r_value**2:.3f})')
                plt.legend()
        else:
            plt.text(0.5, 0.5, 'Insufficient data for density analysis',
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('BEC Zone Density Efficiency Analysis')

    # Add a main title
    plt.suptitle('Detailed Biogeoclimatic Analysis of Pacific Yew Distribution\nDensity-Normalized Metrics',
                 fontsize=16, fontweight='bold', y=0.98)

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('pacific_yew_bec_details.png', dpi=300, bbox_inches='tight')
    print("Detailed BEC visualizations saved to 'pacific_yew_bec_details.png'")


def generate_report(associations_df, bec_data, yew_stats, mapping_data=None, temporal_df=None):
    """
    Generate a comprehensive textual report of the Pacific Yew analysis results including density metrics.

    Args:
        associations_df: DataFrame of species associations
        bec_data: Tuple of (bec_df, subzone_df, variant_df) DataFrames
        yew_stats: Dictionary with yew statistics
        mapping_data: Tuple containing reference mapping dictionaries
        temporal_df: DataFrame with temporal analysis data
    """
    # Unpack BEC data
    bec_df, subzone_df, variant_df = bec_data

    # Unpack mapping data if provided
    species_map = {}
    bec_zone_map = {}
    bec_subzone_map = {}
    bec_variant_map = {}

    if mapping_data:
        species_map, bec_zone_map, bec_subzone_map, bec_variant_map = mapping_data

    # Create a more visually distinct report header
    print("\n" + "="*70)
    print(" " * 15 + "PACIFIC YEW (Taxus brevifolia) ANALYSIS REPORT")
    print(" " * 8 + "BC Sample Data Analysis - Density-Normalized Site-Level Study")
    print("="*70)

    # Add timestamp
    from datetime import datetime
    print(f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Yew characteristics section
    if yew_stats:
        print("\n" + "-"*60)
        print("PACIFIC YEW OCCURRENCE CHARACTERISTICS:")
        print("-"*60)
        print(f"Sites with Pacific Yew: {yew_stats['site_count']:,}")
        print(
            f"Total sites analyzed: {yew_stats.get('total_samples', 'N/A'):,}")
        print(
            f"Percentage of sites with yew: {yew_stats.get('percentage_of_sites_with_yew', 0):.2f}%")
        print(
            f"Average yew percentage (at yew sites): {yew_stats['avg_percentage']:.2f}%")
        print(f"Median yew percentage: {yew_stats['median_percentage']:.2f}%")
        print(f"Maximum yew percentage: {yew_stats['max_percentage']:.1f}%")
        print(f"Minimum yew percentage: {yew_stats['min_percentage']:.1f}%")
        print(f"Standard deviation: {yew_stats['std_percentage']:.2f}%")

    # Temporal analysis section
    if temporal_df is not None and len(temporal_df) > 0:
        print("\n" + "-"*60)
        print("TEMPORAL DISTRIBUTION ANALYSIS:")
        print("-"*60)

        valid_dates = temporal_df['Measurement_Date'].dropna()
        if len(valid_dates) > 0:
            print(
                f"Measurement period: {valid_dates.min().strftime('%Y-%m-%d')} to {valid_dates.max().strftime('%Y-%m-%d')}")

            # Year distribution
            year_counts = temporal_df['Year'].value_counts().sort_index()
            print(
                f"Years with yew measurements: {', '.join(map(str, year_counts.index))}")
            print(
                f"Most measurements in: {year_counts.idxmax()} ({year_counts.max()} sites)")

            # Seasonal distribution
            season_counts = temporal_df['Season'].value_counts()
            if len(season_counts) > 0:
                print(f"Seasonal distribution:")
                for season, count in season_counts.sort_values(ascending=False).items():
                    percentage = (count / len(temporal_df)) * 100
                    print(
                        f"  {season}: {count} measurements ({percentage:.1f}%)")

            # Density statistics over time
            density_stats = temporal_df['Yew_Density_HA'].describe()
            print(f"Yew density statistics (stems/hectare):")
            print(f"  Mean: {density_stats['mean']:.2f}")
            print(f"  Median: {density_stats['50%']:.2f}")
            print(f"  Maximum: {density_stats['max']:.1f}")
            print(f"  Standard deviation: {density_stats['std']:.2f}")

    # Species associations section
    if len(associations_df) > 0:
        print("\n" + "-"*60)
        print("SPECIES ASSOCIATIONS:")
        print("-"*60)
        print(
            f"Total species found co-occurring with Pacific Yew: {len(associations_df)}")
        print("\nTop 10 most frequently associated species:")

        for idx, row in associations_df.head(10).iterrows():
            species_code = row['Species']
            species_name = species_map.get(species_code, species_code)
            print(f"  {idx+1:2d}. {species_code} ({species_name})")
            print(
                f"      Co-occurs at {row['Co_occurrence_Count']} sites ({row['Co_occurrence_Rate_%']:.1f}%)")
            print(
                f"      Average composition: {row['Avg_Species_Percentage']:.1f}% (range: {row['Min_Species_Percentage']:.0f}%-{row['Max_Species_Percentage']:.0f}%)")
            print(
                f"      Average yew % when present: {row['Avg_Yew_Percentage_When_Present']:.1f}%")

    # BEC zone distribution with density metrics
    if len(bec_df) > 0:
        print("\n" + "-"*60)
        print("BIOGEOCLIMATIC ZONE DISTRIBUTION (DENSITY-NORMALIZED):")
        print("-"*60)
        print(f"Total BEC zones with Pacific Yew: {len(bec_df)}")
        print("\nTop 10 BEC zones by normalized yew density:")

        for idx, row in bec_df.head(10).iterrows():
            zone_code = row['BEC_Zone']
            zone_name = bec_zone_map.get(zone_code, zone_code)
            print(f"  {idx+1:2d}. {zone_code} ({zone_name})")
            print(
                f"      {row['Count']} sites ({row['Percentage']:.1f}% of yew sites)")
            print(
                f"      Average yew density: {row['Avg_Yew_Density_HA']:.2f} stems/ha")
            print(
                f"      Average total forest density: {row['Avg_Total_Stems_HA']:.0f} stems/ha")
            print(
                f"      Normalized yew density: {row['Yew_Density_Per_Total_Density']*100:.3f}%")
            print(
                f"      Maximum yew density: {row['Max_Yew_Density_HA']:.2f} stems/ha")

    # Subzone analysis with density
    if len(subzone_df) > 0:
        print("\n" + "-"*50)
        print("DETAILED SUBZONE ANALYSIS (DENSITY-NORMALIZED):")
        print("-"*50)
        print(f"Total zone+subzone combinations: {len(subzone_df)}")
        print("\nTop 10 subzone combinations by normalized density:")

        for idx, row in subzone_df.head(10).iterrows():
            print(
                f"  {idx+1:2d}. {row['Zone_Subzone']}: {row['Count']} sites ({row['Percentage']:.1f}%)")
            print(
                f"      Yew density: {row['Avg_Yew_Density_HA']:.2f} stems/ha")
            print(
                f"      Normalized density: {row['Yew_Density_Per_Total_Density']*100:.3f}%")

    # Full BEC designation analysis
    if len(variant_df) > 0:
        print("\n" + "-"*50)
        print("FULL BEC DESIGNATION ANALYSIS (DENSITY-NORMALIZED):")
        print("-"*50)
        print(f"Total BEC designations: {len(variant_df)}")
        print("\nTop 10 full BEC designations by normalized density:")

        for idx, row in variant_df.head(10).iterrows():
            print(
                f"  {idx+1:2d}. {row['BEC_Label']}: {row['Count']} sites ({row['Percentage']:.1f}%)")
            print(
                f"      Yew density: {row['Avg_Yew_Density_HA']:.2f} stems/ha")
            print(
                f"      Normalized density: {row['Yew_Density_Per_Total_Density']*100:.3f}%")

    # Summary and ecological implications
    print("\n" + "-"*60)
    print("ECOLOGICAL SUMMARY (DENSITY-BASED INSIGHTS):")
    print("-"*60)

    # Generate insights based on the data
    if len(associations_df) > 0 and len(bec_df) > 0:
        top_associate = associations_df.iloc[0]['Species']
        top_associate_name = species_map.get(top_associate, top_associate)
        top_zone = bec_df.iloc[0]['BEC_Zone']
        top_zone_name = bec_zone_map.get(top_zone, top_zone)

        print(
            f"• Pacific Yew most commonly associates with {top_associate} ({top_associate_name})")
        print(
            f"• Highest density (normalized) in {top_zone} ({top_zone_name}) biogeoclimatic zones")

        # Density-based insights
        if len(bec_df) > 0:
            avg_normalized_density = bec_df['Yew_Density_Per_Total_Density'].mean(
            ) * 100
            max_normalized_density = bec_df['Yew_Density_Per_Total_Density'].max(
            ) * 100

            if avg_normalized_density < 0.1:
                print(
                    f"• Pacific Yew represents a very minor component of forest stands (avg {avg_normalized_density:.3f}%)")
            elif avg_normalized_density < 1.0:
                print(
                    f"• Pacific Yew is a minor but consistent component (avg {avg_normalized_density:.2f}%)")
            else:
                print(
                    f"• Pacific Yew can be a notable component in some stands (avg {avg_normalized_density:.2f}%)")

            print(
                f"• Maximum normalized density reached: {max_normalized_density:.2f}%")

        high_diversity_sites = len(
            [s for s in associations_df['Co_occurrence_Count'] if s >= 5])
        print(
            f"• Found in diverse forest communities ({high_diversity_sites} species co-occur frequently)")

        if len(bec_df) >= 5:
            print(
                "• Shows wide ecological amplitude across multiple biogeoclimatic zones")
        else:
            print("• Shows preference for specific biogeoclimatic conditions")

        # Temporal insights
        if temporal_df is not None and len(temporal_df) > 0:
            year_range = temporal_df['Year'].max() - temporal_df['Year'].min()
            if year_range > 10:
                print(
                    f"• Long-term monitoring data available ({year_range} years)")

            season_counts = temporal_df['Season'].value_counts()
            if len(season_counts) > 0:
                dominant_season = season_counts.idxmax()
                print(f"• Most measurements conducted in {dominant_season}")

    print("\n" + "="*70)
    print("Report generated successfully. Analysis data saved to CSV files.")
    print("Key Innovation: Density normalization provides ecological context")
    print("by accounting for overall forest density in each biogeoclimatic zone.")
    print("="*70)


def save_results_to_csv(associations_df, bec_data, yew_stats, yew_sites, temporal_df=None):
    """
    Save analysis results to CSV files for further analysis.

    Args:
        associations_df: DataFrame of species associations
        bec_data: Tuple of (bec_df, subzone_df, variant_df) DataFrames
        yew_stats: Dictionary with yew statistics
        yew_sites: Dictionary of yew sites
        temporal_df: DataFrame with temporal analysis data
    """
    bec_df, subzone_df, variant_df = bec_data

    print("\nSaving results to CSV files...")

    # Save species associations
    if len(associations_df) > 0:
        associations_df.to_csv('pacific_yew_associations.csv', index=False)
        print("• Species associations saved to 'pacific_yew_associations.csv'")

    # Save BEC zone data with density metrics
    if len(bec_df) > 0:
        bec_df.to_csv('pacific_yew_bec_zones.csv', index=False)
        print("• BEC zones (with density metrics) saved to 'pacific_yew_bec_zones.csv'")

    if len(subzone_df) > 0:
        subzone_df.to_csv('pacific_yew_bec_subzones.csv', index=False)
        print(
            "• BEC subzones (with density metrics) saved to 'pacific_yew_bec_subzones.csv'")

    if len(variant_df) > 0:
        variant_df.to_csv('pacific_yew_bec_variants.csv', index=False)
        print(
            "• BEC variants (with density metrics) saved to 'pacific_yew_bec_variants.csv'")

    # Save temporal analysis data
    if temporal_df is not None and len(temporal_df) > 0:
        temporal_df.to_csv('pacific_yew_temporal_analysis.csv', index=False)
        print("• Temporal analysis data saved to 'pacific_yew_temporal_analysis.csv'")


def main():
    """Main analysis function."""
    # Print a visually appealing banner
    print("\n" + "="*70)
    print(" "*12 + "PACIFIC YEW ANALYSIS TOOL - BC SAMPLE DATA")
    print(" "*5 + "Density-Normalized Ecological Association & Temporal Distribution Study")
    print("-"*70)
    print(" Pacific Yew (Taxus brevifolia) - Species Code: TW")
    print(" Analysis includes:")
    print("   • Species associations and biogeoclimatic zone distribution")
    print("   • Density normalization (yew stems/ha ÷ total stems/ha)")
    print("   • Temporal distribution of measurements")
    print(" Data source: BC Sample Data (bc_sample_data.csv)")
    print("="*70)

    # Load reference data for species and BEC zone translations
    species_map, bec_zone_map, bec_subzone_map, bec_variant_map = load_reference_data()
    mapping_data = (species_map, bec_zone_map,
                    bec_subzone_map, bec_variant_map)

    # Load data
    sample_data = load_data()
    if sample_data is None:
        print("Error: Could not load sample data. Exiting.")
        return

    # Identify sites with Pacific Yew
    yew_sites = identify_yew_sites(sample_data)
    if not yew_sites:
        print("No Pacific Yew found in the dataset. Exiting.")
        return

    # Analyze species associations
    associations_df = find_associated_species(sample_data, yew_sites)

    # Analyze BEC zones, subzones, and variants with density normalization
    bec_df, subzone_df, variant_df = find_bec_zones(sample_data, yew_sites)
    bec_data = (bec_df, subzone_df, variant_df)

    # Analyze Yew characteristics
    yew_stats = analyze_yew_characteristics(sample_data, yew_sites)

    # Analyze temporal distribution
    temporal_df = analyze_yew_temporal_distribution(sample_data, yew_sites)

    # Generate visualizations with density and temporal analysis
    visualize_results(associations_df, bec_data, yew_stats,
                      sample_data, yew_sites, temporal_df)

    # Generate comprehensive report
    generate_report(associations_df, bec_data,
                    yew_stats, mapping_data, temporal_df)

    # Save results to CSV files
    save_results_to_csv(associations_df, bec_data,
                        yew_stats, yew_sites, temporal_df)

    print(f"\nAnalysis complete! Found Pacific Yew at {len(yew_sites)} sites.")
    print("Key enhancements in this analysis:")
    print("• Density normalization accounts for total forest density")
    print("• Temporal analysis shows measurement dates and seasonal patterns")
    print("• Enhanced ecological insights through density-based metrics")
    print("Check the generated PNG files for visualizations and CSV files for detailed data.")


if __name__ == "__main__":
    main()
