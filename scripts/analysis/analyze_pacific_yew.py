import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter, defaultdict


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
        with open("tree_name_keys.txt", "r") as f:
            lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2:
                    # Look for species code at the end of line (typically 2 chars)
                    if re.match(r'^[A-Z][a-z]$', parts[-1]):
                        code = parts[-1]
                        # Extract common name by finding where it starts (after "Name")
                        line_text = line.strip()
                        if "Common Name" in line_text:
                            continue

                        name_parts = []
                        for i, part in enumerate(parts[:-1]):
                            if not part[0].isupper() or part.lower() == "x":
                                name_parts.append(part)
                            elif i > 0 and parts[i-1].lower() == "x":  # Handle hybrids
                                name_parts.append(part)

                        common_name = ' '.join(
                            name_parts) if name_parts else parts[0]
                        species_map[code] = common_name

        print(f"Loaded {len(species_map)} tree species mappings")
    except FileNotFoundError:
        print("Warning: Could not find tree_name_keys.txt file")

    # Load BEC zone mappings
    try:
        with open("biogeoclimactic_zone_keys.txt", "r") as f:
            lines = f.readlines()

            current_zone_name = ""
            for line in lines:
                parts = line.strip().split('\t')

                # Skip header lines or empty lines
                if not parts or parts[0] == "Zone" or not parts[0].strip():
                    continue

                # If this is a zone name line (no BEC code)
                if len(parts) == 1 and parts[0] not in ["", "–"]:
                    current_zone_name = parts[0].strip()
                elif len(parts) >= 2:
                    # Process lines with BEC unit codes
                    bec_code_parts = parts[0].strip().split()
                    if len(bec_code_parts) >= 1:
                        zone_code = bec_code_parts[0]  # e.g., CWH

                        # Store zone name
                        if zone_code and current_zone_name:
                            bec_zone_map[zone_code] = current_zone_name

                        # Process subzone and variant if available
                        if len(bec_code_parts) >= 2:
                            subzone_code = bec_code_parts[1]  # e.g., vm
                            subzone_name = parts[1].strip() if len(
                                parts) > 1 else ""

                            # Store subzone information
                            if subzone_code and subzone_name:
                                key = f"{zone_code}_{subzone_code}"
                                bec_subzone_map[key] = subzone_name

                            # Store variant information if available
                            if len(bec_code_parts) >= 3:
                                variant_code = bec_code_parts[2]  # e.g., 1
                                variant_name = parts[2].strip() if len(
                                    parts) > 2 else ""

                                if variant_code and variant_name:
                                    key = f"{zone_code}_{subzone_code}_{variant_code}"
                                    bec_variant_map[key] = variant_name

        print(f"Loaded {len(bec_zone_map)} BEC zone mappings, {len(bec_subzone_map)} subzone mappings, and {len(bec_variant_map)} variant mappings")
    except FileNotFoundError:
        print("Warning: Could not find biogeoclimactic_zone_keys.txt file")

    return species_map, bec_zone_map, bec_subzone_map, bec_variant_map


def get_species_name(code, species_map):
    """Convert species code to common name."""
    if not code or not species_map:
        return code

    if code in species_map:
        return f"{species_map[code]} ({code})"
    return code


def get_bec_zone_name(zone, subzone=None, variant=None, bec_zone_map=None, bec_subzone_map=None, bec_variant_map=None):
    """Convert BEC zone codes to full names."""
    if not zone or not bec_zone_map:
        return zone

    zone_name = bec_zone_map.get(zone, zone)

    if subzone and bec_subzone_map:
        subzone_key = f"{zone}_{subzone}"
        subzone_name = bec_subzone_map.get(subzone_key, subzone)
        zone_name = f"{zone_name} {subzone_name}"

    if variant and bec_variant_map:
        variant_key = f"{zone}_{subzone}_{variant}"
        variant_name = bec_variant_map.get(variant_key, variant)
        if variant_name != "–":  # Skip empty variants
            zone_name = f"{zone_name} {variant_name}"

    return zone_name


def load_data():
    """
    Load the tree data and site header data, then filter for latest visits and separate by plot status.
    """
    print("Loading data files...")

    try:
        # Try loading the main tree data file
        tree_data = pd.read_csv("faib_tree_detail.csv", dtype=str)
        print(f"Loaded tree data with {len(tree_data)} rows")
    except FileNotFoundError:
        print("Error: Could not find the tree data file 'faib_tree_detail.csv'")
        print("Please ensure the file exists in the current directory or provide the correct filename.")
        return None, None

    try:
        # Load the site header data that contains BEC zone information and site status
        header_data = pd.read_csv("faib_header.csv", dtype=str)
        print(f"Loaded site header data with {len(header_data)} rows")
    except FileNotFoundError:
        print("Error: Could not find the site header file 'faib_header.csv'")
        return tree_data, None

    # Filter for latest visits only
    if 'VISIT_NUMBER' in tree_data.columns:
        print("Filtering tree data to include only the latest visit for each site...")
        tree_data['VISIT_NUMBER'] = pd.to_numeric(
            tree_data['VISIT_NUMBER'], errors='coerce')

        # Find the maximum visit number for each site
        latest_visits = tree_data.groupby('SITE_IDENTIFIER')[
            'VISIT_NUMBER'].max().reset_index()
        latest_visits.columns = ['SITE_IDENTIFIER', 'MAX_VISIT']

        # Merge back to get only records from the latest visit
        tree_data = tree_data.merge(latest_visits, on='SITE_IDENTIFIER')
        tree_data = tree_data[tree_data['VISIT_NUMBER']
                              == tree_data['MAX_VISIT']]
        tree_data = tree_data.drop('MAX_VISIT', axis=1)

        print(
            f"After filtering for latest visits: {len(tree_data)} tree records")
    else:
        print("Warning: VISIT_NUMBER column not found in tree data")

    # Filter header data and tree data based on site status (Active vs Inactive)
    if 'SITE_STATUS_CODE' in header_data.columns:
        print("Separating sites by status (Active: A, Inactive: IA)...")

        active_sites = header_data[header_data['SITE_STATUS_CODE']
                                   == 'A']['SITE_IDENTIFIER'].tolist()
        inactive_sites = header_data[header_data['SITE_STATUS_CODE']
                                     == 'IA']['SITE_IDENTIFIER'].tolist()

        print(
            f"Found {len(active_sites)} active sites and {len(inactive_sites)} inactive sites")

        # Filter tree data to match available sites in header
        tree_data = tree_data[tree_data['SITE_IDENTIFIER'].isin(
            header_data['SITE_IDENTIFIER'])]
        print(
            f"Tree data after filtering for sites with header data: {len(tree_data)} records")

        # Add site status information to the return (we'll handle this in the analysis functions)
        header_data['SITE_STATUS'] = header_data['SITE_STATUS_CODE'].map(
            {'A': 'Active', 'IA': 'Inactive'})
    else:
        print("Warning: SITE_STATUS_CODE column not found in header data")
        active_sites = []
        inactive_sites = []

    # Try to load plot header data for area normalization
    try:
        # Try both relative and absolute paths
        try:
            plot_header_data = pd.read_csv("faib_plot_header.csv", dtype={
                                           'SITE_IDENTIFIER': str, 'PLOT_AREA_MAIN': float})
        except FileNotFoundError:
            # Try absolute path
            plot_header_data = pd.read_csv(r"C:\Users\jericho1\Downloads\faib_plot_header.csv", dtype={
                                           'SITE_IDENTIFIER': str, 'PLOT_AREA_MAIN': float})
        print(f"Loaded plot header data with {len(plot_header_data)} rows")

        # Verify PLOT_AREA_MAIN column exists
        if 'PLOT_AREA_MAIN' not in plot_header_data.columns:
            print("Warning: PLOT_AREA_MAIN column not found in plot header data")
        else:
            print(
                f"Plot area data available for normalization (total area: {plot_header_data['PLOT_AREA_MAIN'].sum():.2f} hectares)")
    except FileNotFoundError:
        print("Warning: Plot header file 'faib_plot_header.csv' not found - will use tree count normalization")
    except Exception as e:
        print(f"Warning: Error loading plot header data: {str(e)}")

    # Convert column types as needed
    numeric_cols = ['DBH', 'HEIGHT', 'TREE_WT',
                    'BA_TREE', 'VOL_WSV', 'VOL_MER', 'VOL_DWB']
    for col in numeric_cols:
        if col in tree_data.columns:
            tree_data[col] = pd.to_numeric(tree_data[col], errors='coerce')

    return tree_data, header_data


def identify_yew_sites(tree_data, header_data=None):
    """
    Identify sites that contain Pacific Yew trees, separated by site status (Active vs Inactive).

    Args:
        tree_data: DataFrame containing tree data
        header_data: DataFrame containing site header information with site status

    Returns:
        Tuple of (active_yew_sites, inactive_yew_sites, all_yew_sites) - dictionaries with site IDs as keys and yew tree counts as values
    """
    if tree_data is None:
        return {}, {}, {}

    print("\nIdentifying sites with Pacific Yew trees...")

    # Filter for Pacific Yew trees
    yew_trees = tree_data[tree_data['SPECIES'] == 'TW']

    if len(yew_trees) == 0:
        print("No Pacific Yew trees found in the dataset!")
        return {}, {}, {}

    print(f"Found {len(yew_trees)} Pacific Yew trees in the dataset")

    # Count the number of yew trees per site
    all_yew_sites = yew_trees['SITE_IDENTIFIER'].value_counts().to_dict()

    # Separate by site status if header data is available
    active_yew_sites = {}
    inactive_yew_sites = {}

    if header_data is not None and 'SITE_STATUS_CODE' in header_data.columns:
        # Get site status mapping
        site_status_map = header_data.set_index('SITE_IDENTIFIER')[
            'SITE_STATUS_CODE'].to_dict()

        for site_id, yew_count in all_yew_sites.items():
            site_status = site_status_map.get(site_id, 'Unknown')
            if site_status == 'A':
                active_yew_sites[site_id] = yew_count
            elif site_status == 'IA':
                inactive_yew_sites[site_id] = yew_count

        print(
            f"Found Pacific Yew trees at {len(all_yew_sites)} distinct sites:")
        print(f"  - Active sites: {len(active_yew_sites)}")
        print(f"  - Inactive sites: {len(inactive_yew_sites)}")
        print(f"  - Total active yew trees: {sum(active_yew_sites.values())}")
        print(
            f"  - Total inactive yew trees: {sum(inactive_yew_sites.values())}")
    else:
        print(
            f"Found Pacific Yew trees at {len(all_yew_sites)} distinct sites")
        print("Warning: Site status information not available")

    return active_yew_sites, inactive_yew_sites, all_yew_sites


def find_associated_species(tree_data, yew_sites, site_status="All"):
    """
    Identify tree species that commonly occur with Pacific Yew and analyze their relationships.

    Args:
        tree_data: DataFrame containing tree data
        yew_sites: Dictionary of sites with yew trees
        site_status: String indicating which sites to analyze ("Active", "Inactive", or "All")

    Returns:
        DataFrame of species associations with enhanced ecological metrics
    """
    if tree_data is None or not yew_sites:
        return pd.DataFrame()

    print(
        f"\nAnalyzing species associations with Pacific Yew ({site_status} sites)...")

    # Get all trees from sites where yew is found
    yew_site_ids = list(yew_sites.keys())
    trees_at_yew_sites = tree_data[tree_data['SITE_IDENTIFIER'].isin(
        yew_site_ids)]

    # Count species occurrences at yew sites
    species_counts = defaultdict(int)
    species_dbh = defaultdict(list)  # Track DBH for size comparison
    species_height = defaultdict(list)  # Track height for size comparison
    # Track which sites each species occurs in
    species_sites = defaultdict(set)
    # Track total number of trees of each species
    species_tree_counts = defaultdict(int)

    # Get yew tree DBH values for comparison
    yew_trees = trees_at_yew_sites[trees_at_yew_sites['SPECIES'] == 'TW']
    yew_dbh_values = yew_trees['DBH'].dropna(
    ).tolist() if 'DBH' in yew_trees.columns else []
    yew_avg_dbh = np.mean(yew_dbh_values) if yew_dbh_values else np.nan

    # Count co-occurrences by site
    for site_id, site_df in trees_at_yew_sites.groupby('SITE_IDENTIFIER'):
        site_species = set(site_df['SPECIES'].unique())

        # Count each non-yew species at this site
        for species in site_species:
            if species != 'TW':
                species_counts[species] += 1
                species_sites[species].add(site_id)

                # Track metrics for this species at this site
                species_trees = site_df[site_df['SPECIES'] == species]
                species_tree_counts[species] += len(species_trees)

                # Add DBH values for this species at this site
                if 'DBH' in species_trees.columns:
                    species_dbh_values = species_trees['DBH'].dropna()
                    species_dbh[species].extend(species_dbh_values.tolist())

                # Add height values for this species at this site
                if 'HEIGHT' in species_trees.columns:
                    species_height_values = species_trees['HEIGHT'].dropna()
                    species_height[species].extend(
                        species_height_values.tolist())

    # Calculate association metrics
    association_data = []
    total_yew_sites = len(yew_sites)
    total_yew_trees = len(yew_trees)

    for species, count in species_counts.items():
        # Calculate percentage of yew sites where this species occurs
        pct_occurrence = (count / total_yew_sites) * 100

        # Calculate average DBH for this species (if available)
        avg_dbh = np.mean(species_dbh[species]
                          ) if species_dbh[species] else np.nan
        max_dbh = np.max(species_dbh[species]
                         ) if species_dbh[species] else np.nan

        # Calculate average height for this species (if available)
        avg_height = np.mean(
            species_height[species]) if species_height[species] else np.nan

        # Calculate size ratio compared to yew (DBH comparison)
        size_ratio = avg_dbh / \
            yew_avg_dbh if (pd.notna(avg_dbh) and pd.notna(
                yew_avg_dbh) and yew_avg_dbh > 0) else np.nan

        # Calculate abundance ratio (how many of this species vs. yew trees)
        abundance_ratio = species_tree_counts[species] / \
            total_yew_trees if total_yew_trees > 0 else np.nan

        # Calculate how many sites this species occurs in
        site_count = len(species_sites[species])

        # Calculate a combined "ecological association index" - higher values mean stronger association
        # Normalize counts to be between 0-1 by dividing by max possible (total_yew_sites)
        normalized_occurrence = count / total_yew_sites

        # Simple ecological association score - can be refined with more ecological knowledge
        ecological_score = normalized_occurrence * 10  # Scale to 0-10

        association_data.append({
            'Species': species,
            'Co-occurrence_Count': count,
            'Sites': site_count,
            'Co-occurrence_Percentage': pct_occurrence,
            'Tree_Count': species_tree_counts[species],
            'Average_DBH': avg_dbh,
            'Maximum_DBH': max_dbh,
            'Average_Height': avg_height,
            'Size_Ratio_to_Yew': size_ratio,
            'Abundance_Ratio_to_Yew': abundance_ratio,
            'Ecological_Association_Score': ecological_score,
            'Site_Status': site_status
        })

    # Convert to DataFrame and sort by co-occurrence count (descending)
    associations_df = pd.DataFrame(association_data)

    if len(associations_df) > 0:
        associations_df = associations_df.sort_values(
            'Co-occurrence_Count', ascending=False).reset_index(drop=True)

        # Print some additional insights about the most strongly associated species
        top_species = associations_df.iloc[0]['Species']
        print(
            f"Most strongly associated species: {top_species} (found in {associations_df.iloc[0]['Co-occurrence_Percentage']:.1f}% of Pacific Yew sites)")

    return associations_df


def find_bec_zones(header_data, yew_sites, tree_data=None):
    """
    Identify biogeoclimatic zones, subzones, and variants where Pacific Yew is commonly found.
    Normalizes counts by plot area in each zone if plot header data is available.

    Args:
        header_data: DataFrame containing site header information
        yew_sites: Dictionary of sites with yew trees
        tree_data: DataFrame containing all tree data (optional, for tree counts)

    Returns:
        tuple: (zone_df, subzone_df, variant_df) - DataFrames for zones, subzones and variants
    """
    if header_data is None or not yew_sites:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    print("\nAnalyzing biogeoclimatic zones for Pacific Yew...")

    # Filter header data for sites with yew trees
    yew_site_ids = list(yew_sites.keys())
    yew_site_headers = header_data[header_data['SITE_IDENTIFIER'].isin(
        yew_site_ids)]

    if len(yew_site_headers) == 0:
        print("No header data found for yew sites")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Add site status information for analysis
    site_status_info = ""
    if 'SITE_STATUS' in yew_site_headers.columns:
        status_counts = yew_site_headers['SITE_STATUS'].value_counts()
        site_status_info = f" (Active: {status_counts.get('Active', 0)}, Inactive: {status_counts.get('Inactive', 0)})"

    # ===== ZONE LEVEL ANALYSIS =====
    # Count occurrences of BEC zones
    bec_counts = yew_site_headers['BEC_ZONE'].value_counts()

    # Calculate percentages based on yew sites
    total_sites = len(yew_site_headers)
    bec_percentages = (bec_counts / total_sites) * 100

    # Create basic DataFrame for zones
    bec_df = pd.DataFrame({
        'BEC_Zone': bec_counts.index,
        'Count': bec_counts.values,
        'Percentage': bec_percentages.values
    })

    # Add site status breakdown if available
    if 'SITE_STATUS' in yew_site_headers.columns:
        zone_status_breakdown = []
        for zone in bec_df['BEC_Zone']:
            zone_sites = yew_site_headers[yew_site_headers['BEC_ZONE'] == zone]
            active_count = len(
                zone_sites[zone_sites['SITE_STATUS'] == 'Active'])
            inactive_count = len(
                zone_sites[zone_sites['SITE_STATUS'] == 'Inactive'])
            zone_status_breakdown.append({
                'BEC_Zone': zone,
                'Active_Sites': active_count,
                'Inactive_Sites': inactive_count,
                'Active_Percentage': (active_count / (active_count + inactive_count) * 100) if (active_count + inactive_count) > 0 else 0
            })

        status_df = pd.DataFrame(zone_status_breakdown)
        bec_df = bec_df.merge(status_df, on='BEC_Zone', how='left')

    # Try to load the plot header data for plot area normalization
    try:
        plot_header_data = pd.read_csv("faib_plot_header.csv", dtype={
                                       'SITE_IDENTIFIER': str, 'PLOT_AREA_MAIN': float})

        if 'PLOT_AREA_MAIN' in plot_header_data.columns:
            print("Adding plot area normalization...")

            # Merge plot area data with yew site headers
            yew_plot_data = yew_site_headers.merge(plot_header_data[['SITE_IDENTIFIER', 'PLOT_AREA_MAIN']],
                                                   on='SITE_IDENTIFIER', how='left')

            # Calculate area-normalized metrics for each zone
            zone_area_stats = []

            for zone in bec_df['BEC_Zone']:
                zone_plots = yew_plot_data[yew_plot_data['BEC_ZONE'] == zone]
                valid_areas = zone_plots['PLOT_AREA_MAIN'].dropna()

                if len(valid_areas) > 0:
                    total_area = valid_areas.sum()
                    site_count = len(zone_plots)

                    # Calculate yew density per hectare for this zone
                    if tree_data is not None:
                        zone_site_ids = zone_plots['SITE_IDENTIFIER'].tolist()
                        zone_yew_trees = tree_data[(tree_data['SITE_IDENTIFIER'].isin(zone_site_ids)) &
                                                   (tree_data['SPECIES'] == 'TW')]
                        yew_count_in_zone = len(zone_yew_trees)
                        yew_per_hectare = yew_count_in_zone / total_area if total_area > 0 else 0
                    else:
                        yew_count_in_zone = sum(
                            [yew_sites.get(site_id, 0) for site_id in zone_site_ids])
                        yew_per_hectare = yew_count_in_zone / total_area if total_area > 0 else 0

                    zone_area_stats.append({
                        'BEC_Zone': zone,
                        'Total_Area_Ha': total_area,
                        'Plots_with_Area': len(valid_areas),
                        'Yew_Trees_Per_Hectare': yew_per_hectare,
                        'Normalized_Percentage': (yew_per_hectare / max(0.001, yew_plot_data.groupby('BEC_ZONE')['PLOT_AREA_MAIN'].sum().max() / yew_plot_data.groupby('BEC_ZONE').size().max())) * 100
                    })
                else:
                    zone_area_stats.append({
                        'BEC_Zone': zone,
                        'Total_Area_Ha': 0,
                        'Plots_with_Area': 0,
                        'Yew_Trees_Per_Hectare': 0,
                        'Normalized_Percentage': 0
                    })

            # Merge area statistics
            area_stats_df = pd.DataFrame(zone_area_stats)
            bec_df = bec_df.merge(area_stats_df, on='BEC_Zone', how='left')

    except FileNotFoundError:
        print("Plot area data not available - using count-based analysis only")

    bec_df = bec_df.sort_values(
        'Count', ascending=False).reset_index(drop=True)

    # Track subzone and variant collections for reference
    bec_subzones = defaultdict(Counter)
    bec_variants = defaultdict(list)

    # ===== SUBZONE LEVEL ANALYSIS =====
    # Create a dataframe for subzone analysis
    subzone_data = []

    # Analyze subzones within each zone
    for bec_zone, zone_df in yew_site_headers.groupby('BEC_ZONE'):
        subzone_counts = zone_df['BEC_SBZ'].value_counts()
        total_zone_sites = len(zone_df)

        for subzone, count in subzone_counts.items():
            if pd.notna(subzone) and subzone != '':
                bec_subzones[bec_zone][subzone] = count
                subzone_data.append({
                    'BEC_Zone': bec_zone,
                    'BEC_Subzone': subzone,
                    'Count': count,
                    'Zone_Percentage': (count / total_zone_sites) * 100,
                    'Overall_Percentage': (count / total_sites) * 100
                })

    # Convert to DataFrame
    subzone_df = pd.DataFrame(subzone_data)
    if len(subzone_df) > 0:
        subzone_df = subzone_df.sort_values(['BEC_Zone', 'Count'], ascending=[
                                            True, False]).reset_index(drop=True)

    # ===== VARIANT LEVEL ANALYSIS =====
    # Create a dataframe for variant analysis
    variant_data = []

    # Check if variant column exists
    has_variants = 'BEC_VAR' in yew_site_headers.columns

    if has_variants:
        for bec_zone, zone_df in yew_site_headers.groupby('BEC_ZONE'):
            for subzone, subzone_df in zone_df.groupby('BEC_SBZ'):
                if pd.notna(subzone) and subzone != '':
                    variant_counts = subzone_df['BEC_VAR'].value_counts()

                    for variant, count in variant_counts.items():
                        if pd.notna(variant) and variant != '':
                            bec_variants[bec_zone].append(
                                (subzone, variant, count))
                            variant_data.append({
                                'BEC_Zone': bec_zone,
                                'BEC_Subzone': subzone,
                                'BEC_Variant': variant,
                                'Full_BEC_Code': f"{bec_zone}{subzone}{variant}",
                                'Count': count,
                                'Overall_Percentage': (count / total_sites) * 100
                            })

    # Convert to DataFrame
    variant_df = pd.DataFrame(variant_data)
    if len(variant_df) > 0:
        variant_df = variant_df.sort_values(['BEC_Zone', 'BEC_Subzone', 'Count'],
                                            ascending=[True, True, False]).reset_index(drop=True)

    # Add subzone and variant summaries to the zone dataframe for backward compatibility
    bec_df['Common_Subzones'] = bec_df['BEC_Zone'].apply(
        lambda x: ', '.join([f"{sz}({count})" for sz, count in
                             bec_subzones[x].most_common(3)])  # Top 3 subzones
    )

    bec_df['Variant_Details'] = bec_df['BEC_Zone'].apply(
        lambda x: ', '.join([f"{sz}{var}({count})" for sz, var, count in
                             sorted(bec_variants.get(x, []), key=lambda item: item[2], reverse=True)[:3]])  # Top 3 variants
    )

    print(f"Found {len(bec_df)} BEC zones, {len(subzone_df)} subzones, and {len(variant_df)} variants{site_status_info}")

    # Continue with detailed analysis below...

    print("\nAnalyzing biogeoclimatic zones for Pacific Yew...")

    # Filter header data for sites with yew trees
    yew_site_ids = list(yew_sites.keys())
    yew_site_headers = header_data[header_data['SITE_IDENTIFIER'].isin(
        yew_site_ids)]

    if len(yew_site_headers) == 0:
        print("No matching site headers found for yew sites!")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # ===== ZONE LEVEL ANALYSIS =====
    # Count occurrences of BEC zones
    bec_counts = yew_site_headers['BEC_ZONE'].value_counts()

    # Calculate percentages based on yew sites
    total_sites = len(yew_site_headers)
    bec_percentages = (bec_counts / total_sites) * 100

    # Create basic DataFrame for zones
    bec_df = pd.DataFrame({
        'BEC_Zone': bec_counts.index,
        'Count': bec_counts.values,
        'Percentage': bec_percentages.values
    })

    # Try to load the plot header data for plot area normalization
    try:
        # Try both relative and absolute paths
        try:
            plot_header_data = pd.read_csv("faib_plot_header.csv", dtype={
                                           'SITE_IDENTIFIER': str, 'PLOT_AREA_MAIN': float})
        except FileNotFoundError:
            # Try absolute path
            plot_header_data = pd.read_csv(r"C:\Users\jericho1\Downloads\faib_plot_header.csv", dtype={
                                           'SITE_IDENTIFIER': str, 'PLOT_AREA_MAIN': float})
        print("Loaded plot header data for area normalization")

        # Create a mapping of site IDs to BEC zones
        site_to_bec = header_data.set_index('SITE_IDENTIFIER')[
            'BEC_ZONE'].to_dict()

        # Add BEC zone information to the plot header data
        plot_header_with_bec = plot_header_data.copy()
        plot_header_with_bec['BEC_ZONE'] = plot_header_data['SITE_IDENTIFIER'].map(
            site_to_bec)

        # Calculate total area by BEC zone
        # Group by BEC_ZONE and sum the PLOT_AREA_MAIN
        area_by_bec = plot_header_with_bec.groupby(
            'BEC_ZONE')['PLOT_AREA_MAIN'].sum()

        # Count Pacific Yew trees in each BEC zone if tree_data is provided
        if tree_data is not None and 'SITE_IDENTIFIER' in tree_data.columns:
            # Add BEC zone information to the tree data
            tree_data_with_bec = tree_data.copy()
            tree_data_with_bec['BEC_ZONE'] = tree_data['SITE_IDENTIFIER'].map(
                site_to_bec)

            # Count total trees in each BEC zone
            total_trees_by_bec = tree_data_with_bec.groupby('BEC_ZONE').size()

            # Count Pacific Yew trees in each BEC zone
            yew_trees_by_bec = tree_data_with_bec[tree_data_with_bec['SPECIES'] == 'TW'].groupby(
                'BEC_ZONE').size()

            # Add the tree counts to the DataFrame
            bec_df['Total_Trees'] = bec_df['BEC_Zone'].map(total_trees_by_bec)
            bec_df['Yew_Trees'] = bec_df['BEC_Zone'].map(yew_trees_by_bec)
            bec_df['Trees_Per_Site'] = bec_df['Total_Trees'] / bec_df['Count']

        # Add the plot area information to the DataFrame
        bec_df['Total_Plot_Area'] = bec_df['BEC_Zone'].map(area_by_bec)

        # Calculate yew density per hectare (1 hectare = 10000 m²)
        # Assuming PLOT_AREA_MAIN is in hectares, adjust multiplier if it's in different units
        if 'Yew_Trees' in bec_df.columns:
            # Calculate yew trees per hectare
            bec_df['Yew_Trees_Per_Hectare'] = bec_df['Yew_Trees'] / \
                bec_df['Total_Plot_Area']

            # Create normalized percentage based on area
            total_yew = bec_df['Yew_Trees'].sum()
            total_area = bec_df['Total_Plot_Area'].sum()
            expected_yews = bec_df['Total_Plot_Area'] * \
                (total_yew / total_area)
            bec_df['Normalized_Percentage'] = (
                bec_df['Yew_Trees'] / expected_yews) * 100

            # For backward compatibility
            bec_df['Yew_Density_Per_1000'] = bec_df['Yew_Trees_Per_Hectare'] * 1000

            print("Calculated normalized distribution by plot area in each BEC zone")

    except FileNotFoundError:
        print("Plot header file not found. Normalizing by tree counts instead...")

        # Fallback to tree count normalization if plot area data is not available
        if tree_data is not None and 'SITE_IDENTIFIER' in tree_data.columns:
            print(
                "Calculating normalized distribution by total trees in each BEC zone...")

            # Create a mapping of site IDs to BEC zones
            site_to_bec = header_data.set_index('SITE_IDENTIFIER')[
                'BEC_ZONE'].to_dict()

            # Add BEC zone information to the tree data
            tree_data_with_bec = tree_data.copy()
            tree_data_with_bec['BEC_ZONE'] = tree_data['SITE_IDENTIFIER'].map(
                site_to_bec)

            # Count total trees in each BEC zone
            total_trees_by_bec = tree_data_with_bec.groupby('BEC_ZONE').size()

            # Count Pacific Yew trees in each BEC zone
            yew_trees_by_bec = tree_data_with_bec[tree_data_with_bec['SPECIES'] == 'TW'].groupby(
                'BEC_ZONE').size()

            # Add the normalized counts to the DataFrame
            bec_df['Total_Trees'] = bec_df['BEC_Zone'].map(total_trees_by_bec)
            bec_df['Yew_Trees'] = bec_df['BEC_Zone'].map(yew_trees_by_bec)
            bec_df['Trees_Per_Site'] = bec_df['Total_Trees'] / bec_df['Count']

            # Calculate percentage of yew trees compared to total trees in each BEC zone
            bec_df['Normalized_Percentage'] = (
                bec_df['Yew_Trees'] / bec_df['Total_Trees']) * 100

            # Calculate density: yew trees per 1000 trees in the zone
            bec_df['Yew_Density_Per_1000'] = (
                bec_df['Yew_Trees'] / bec_df['Total_Trees']) * 1000

    bec_df = bec_df.sort_values(
        'Count', ascending=False).reset_index(drop=True)

    # Track subzone and variant collections for reference
    bec_subzones = defaultdict(list)
    bec_variants = defaultdict(list)

    # ===== SUBZONE LEVEL ANALYSIS =====
    # Create a dataframe for subzone analysis
    subzone_data = []

    # Analyze subzones within each zone
    for bec_zone, zone_df in yew_site_headers.groupby('BEC_ZONE'):
        # Track subzones for this zone
        subzones = zone_df['BEC_SBZ'].value_counts().to_dict()
        bec_subzones[bec_zone] = subzones

        # Add each subzone to our detailed analysis
        for subzone, count in subzones.items():
            if pd.isna(subzone) or subzone == '':
                subzone_name = 'Unspecified'
            else:
                subzone_name = subzone

            subzone_data.append({
                'BEC_Zone': bec_zone,
                'BEC_Subzone': subzone_name,
                'Count': count,
                'Percentage_Within_Zone': (count / len(zone_df)) * 100,
                'Percentage_Overall': (count / total_sites) * 100
            })

    # Convert to DataFrame
    subzone_df = pd.DataFrame(subzone_data)
    if len(subzone_df) > 0:
        subzone_df = subzone_df.sort_values(['BEC_Zone', 'Count'], ascending=[
                                            True, False]).reset_index(drop=True)

    # ===== VARIANT LEVEL ANALYSIS =====
    # Create a dataframe for variant analysis
    variant_data = []

    # Check if variant column exists
    has_variants = 'BEC_VAR' in yew_site_headers.columns

    if has_variants:
        # Group by zone, subzone and variant
        variant_groups = yew_site_headers.groupby(
            ['BEC_ZONE', 'BEC_SBZ', 'BEC_VAR']).size()

        for (zone, subzone, variant), count in variant_groups.items():
            if pd.isna(subzone) or subzone == '':
                subzone_name = 'Unspecified'
            else:
                subzone_name = subzone

            if pd.isna(variant) or variant == '':
                variant_name = 'Unspecified'
            else:
                variant_name = variant

            # Calculate the number of sites with this zone and subzone
            zone_subzone_count = len(yew_site_headers[(yew_site_headers['BEC_ZONE'] == zone) &
                                                      (yew_site_headers['BEC_SBZ'] == subzone)])

            variant_data.append({
                'BEC_Zone': zone,
                'BEC_Subzone': subzone_name,
                'BEC_Variant': variant_name,
                'Count': count,
                'Percentage_Within_Subzone': (count / zone_subzone_count) * 100 if zone_subzone_count > 0 else 0,
                'Percentage_Overall': (count / total_sites) * 100
            })

            # Add to our tracking collection
            bec_variants[zone].append((subzone, variant, count))

    # Convert to DataFrame
    variant_df = pd.DataFrame(variant_data)
    if len(variant_df) > 0:
        variant_df = variant_df.sort_values(
            ['BEC_Zone', 'BEC_Subzone', 'Count'],
            ascending=[True, True, False]
        ).reset_index(drop=True)

    # Add subzone and variant summaries to the zone dataframe for backward compatibility
    bec_df['Common_Subzones'] = bec_df['BEC_Zone'].apply(
        lambda x: ', '.join([f"{sz}({count})" for sz, count in
                             sorted(bec_subzones[x].items(
                             ), key=lambda item: item[1], reverse=True)
                             if not pd.isna(sz) and sz != ''])
    )

    bec_df['Variant_Details'] = bec_df['BEC_Zone'].apply(
        lambda x: ', '.join([f"{sz}{var}({count})" for sz, var, count in
                             sorted(bec_variants.get(x, []),
                                    key=lambda item: item[2], reverse=True)
                             if not pd.isna(sz) and not pd.isna(var) and sz != '' and var != ''])
    )

    print(
        f"Found {len(bec_df)} BEC zones, {len(subzone_df)} subzones, and {len(variant_df)} variants")

    return bec_df, subzone_df, variant_df


def analyze_yew_characteristics(tree_data):
    """
    Analyze characteristics of Pacific Yew trees in the dataset.

    Args:
        tree_data: DataFrame containing tree data

    Returns:
        Dictionary with yew statistics
    """
    if tree_data is None:
        return {}

    print("\nAnalyzing Pacific Yew characteristics...")

    # Filter for Pacific Yew trees
    yew_trees = tree_data[tree_data['SPECIES'] == 'TW']

    if len(yew_trees) == 0:
        print("No Pacific Yew trees found in the dataset!")
        return {}

    # Calculate statistics
    stats = {
        'count': len(yew_trees),
        'avg_dbh': yew_trees['DBH'].astype(float).mean() if 'DBH' in yew_trees.columns else None,
        'avg_height': yew_trees['HEIGHT'].astype(float).mean() if 'HEIGHT' in yew_trees.columns else None,
        'max_dbh': yew_trees['DBH'].astype(float).max() if 'DBH' in yew_trees.columns else None,
        'max_height': yew_trees['HEIGHT'].astype(float).max() if 'HEIGHT' in yew_trees.columns else None
    }

    return stats


def visualize_results(associations_df, bec_data, yew_stats, tree_data=None):
    """
    Create visualizations of the results.

    Args:
        associations_df: DataFrame of species associations
        bec_data: Tuple of (bec_df, subzone_df, variant_df) DataFrames
        yew_stats: Dictionary with yew statistics
        tree_data: Original tree data for additional visualizations
    """
    # Unpack BEC zone data
    bec_df, subzone_df, variant_df = bec_data
    print("\nGenerating visualizations...")

    # Set up the figure layout
    plt.figure(figsize=(15, 12))

    # 1. Top species associations
    if len(associations_df) > 0:
        plt.subplot(2, 2, 1)
        top_n = min(10, len(associations_df))

        # Create bar chart of top associated species
        top_species = associations_df.head(top_n)

        # Use common names if available, otherwise use species codes
        if 'Common_Name' in top_species.columns:
            # Truncate long names for better display
            species_labels = top_species['Common_Name'].apply(
                lambda x: x[:35] + '...' if len(x) > 35 else x)
        else:
            species_labels = top_species['Species']

        # Use a green color palette for the bars
        bars = plt.barh(species_labels, top_species['Co-occurrence_Count'],
                        color=plt.cm.Greens(np.linspace(0.5, 0.8, len(top_species))))
        plt.xlabel('Number of Sites', fontweight='bold')
        plt.ylabel('Tree Species', fontweight='bold')
        plt.title(f'Top {top_n} Species Associated with Pacific Yew',
                  fontsize=12, fontweight='bold')
        plt.gca().invert_yaxis()  # Highest count at the top

        # Add percentage labels to bars
        for i, (_, row) in enumerate(top_species.iterrows()):
            plt.text(row['Co-occurrence_Count'] + 0.1, i,
                     f"{row['Co-occurrence_Percentage']:.1f}%",
                     va='center', fontweight='bold')

    # 2. BEC Zone distribution
    if len(bec_df) > 0:
        plt.subplot(2, 2, 2)

        # Use full zone names if available
        zone_labels = bec_df['Zone_Full_Name'] if 'Zone_Full_Name' in bec_df.columns else bec_df['BEC_Zone']

        # Create shorter labels for the pie chart but keep full names for the legend
        full_zone_labels = zone_labels.copy() if isinstance(
            zone_labels, list) else zone_labels.tolist()
        short_labels = [f"Zone {i+1}" for i in range(len(zone_labels))]

        # Custom colors for better visualization
        colors = plt.cm.Greens(np.linspace(0.4, 0.8, len(bec_df)))

        # Determine which values to use for the pie chart
        # If normalized data is available, use it instead of raw counts
        if 'Normalized_Percentage' in bec_df.columns and 'Yew_Trees_Per_Hectare' in bec_df.columns:
            pie_values = bec_df['Yew_Trees_Per_Hectare']
            chart_title = 'Pacific Yew Density by BEC Zone (trees per hectare)'
            legend_title = "BEC Zone Legend (Density per hectare)"

            # Add normalized values to the legend
            legend_labels = []
            for i, (_, row) in enumerate(bec_df.iterrows()):
                label = full_zone_labels[i]
                if 'Total_Plot_Area' in row and pd.notna(row['Total_Plot_Area']) and 'Yew_Trees' in row and pd.notna(row['Yew_Trees']):
                    legend_text = f"Zone {i+1}: {label} - {row['Yew_Trees_Per_Hectare']:.2f} Yew/hectare ({row['Yew_Trees']} Yew in {row['Total_Plot_Area']:.2f} ha)"
                elif 'Yew_Trees' in row and pd.notna(row['Yew_Trees']):
                    legend_text = f"Zone {i+1}: {label} - {row['Yew_Trees_Per_Hectare']:.2f} Yew/hectare ({row['Yew_Trees']} Yew)"
                else:
                    legend_text = f"Zone {i+1}: {label} - {row['Yew_Trees_Per_Hectare']:.2f} Yew/hectare"
                legend_labels.append(legend_text)
        else:
            pie_values = bec_df['Count']
            chart_title = 'Pacific Yew Distribution by BEC Zone'
            legend_title = "BEC Zone Legend"
            legend_labels = [f"Zone {i+1}: {label}" for i,
                             label in enumerate(full_zone_labels)]

        wedges, texts, autotexts = plt.pie(
            pie_values,
            labels=short_labels,
            autopct='%1.1f%%',
            colors=colors,
            explode=[0.05] * len(bec_df),  # Slightly explode all slices
            shadow=True,
            startangle=90
        )

        # Enhance the appearance of labels
        for text in texts:
            text.set_fontsize(9)
            text.set_fontweight('bold')

        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_fontweight('bold')

        # Add a custom legend with full zone names and normalized values
        plt.legend(wedges, legend_labels,
                   title=legend_title,
                   loc="lower center",
                   bbox_to_anchor=(0.5, -0.3),
                   fontsize=8,
                   frameon=True)

        plt.title(chart_title, fontsize=12, fontweight='bold')
        plt.axis('equal')

    # 3. Pacific Yew characteristics
    if yew_stats:
        plt.subplot(2, 2, 3)

        # Create a table-like format for statistics with improved styling
        stats_text = "\n".join([
            f"Total Pacific Yew Trees: {yew_stats['count']}",
            f"Average DBH: {yew_stats['avg_dbh']:.2f} cm" if yew_stats['avg_dbh'] else "Average DBH: N/A",
            f"Maximum DBH: {yew_stats['max_dbh']:.2f} cm" if yew_stats['max_dbh'] else "Maximum DBH: N/A",
            f"Average Height: {yew_stats['avg_height']:.2f} m" if yew_stats['avg_height'] else "Average Height: N/A",
            f"Maximum Height: {yew_stats['max_height']:.2f} m" if yew_stats['max_height'] else "Maximum Height: N/A"
        ])

        # Add an image background for better presentation
        plt.axhspan(0, 1, facecolor='lightgreen', alpha=0.2)
        plt.text(0.1, 0.5, stats_text, fontsize=12, fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
        plt.axis('off')
        plt.title('Pacific Yew Characteristics',
                  fontsize=12, fontweight='bold')

    # 4. DBH Distribution of Pacific Yew Trees (if tree data is available)
    if tree_data is not None:
        plt.subplot(2, 2, 4)

        # Filter for Pacific Yew trees
        yew_trees = tree_data[tree_data['SPECIES'] == 'TW']

        if 'DBH' in yew_trees.columns and len(yew_trees) > 0:
            # Create a histogram of DBH distribution
            dbh_values = yew_trees['DBH'].dropna()

            if len(dbh_values) > 0:
                # Use KDE plot for smoother distribution
                sns.histplot(dbh_values, kde=True, color='darkgreen')
                plt.xlabel('Diameter (cm)', fontweight='bold')
                plt.ylabel('Frequency', fontweight='bold')
                plt.title('Size Distribution of Pacific Yew Trees',
                          fontsize=12, fontweight='bold')

                # Add vertical line for average DBH
                if yew_stats and yew_stats['avg_dbh']:
                    plt.axvline(x=yew_stats['avg_dbh'], color='red', linestyle='--',
                                label=f"Avg: {yew_stats['avg_dbh']:.1f} cm")
                    plt.legend()
            else:
                plt.text(0.5, 0.5, "No DBH data available",
                         ha='center', va='center', fontsize=12)
                plt.axis('off')
        else:
            plt.text(0.5, 0.5, "No DBH data available",
                     ha='center', va='center', fontsize=12)
            plt.axis('off')

    # Add a main title to the figure
    plt.suptitle('Pacific Yew (Taxus brevifolia) Ecological Analysis',
                 fontsize=16, fontweight='bold', y=0.98)

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the main title
    # Higher DPI for better quality
    plt.savefig('pacific_yew_analysis.png', dpi=300)
    print("Visualizations saved to 'pacific_yew_analysis.png'")

    # Create a separate figure for detailed BEC zone analysis
    visualize_bec_details(bec_df, subzone_df, variant_df)


def visualize_bec_details(bec_df, subzone_df, variant_df):
    """
    Create detailed visualizations of BEC zones, subzones and variants.
    Includes area-normalized density plots if plot area data is available.

    Args:
        bec_df: DataFrame with BEC zone data
        subzone_df: DataFrame with subzone data
        variant_df: DataFrame with variant data
    """
    if len(subzone_df) == 0:
        print("No subzone data available for visualization")
        return

    print("\nGenerating detailed BEC zone visualizations...")

    # Create a new figure for BEC details
    plt.figure(figsize=(15, 15))  # Increased height for additional plots

    # Add normalized density comparison if available
    if 'Normalized_Percentage' in bec_df.columns and 'Yew_Trees_Per_Hectare' in bec_df.columns:
        plt.subplot(3, 1, 1)  # Make room for 3 plots

        # Sort by normalized density
        sorted_df = bec_df.sort_values(
            'Yew_Trees_Per_Hectare', ascending=False)

        # Use full zone names if available
        zone_labels = sorted_df['Zone_Full_Name'] if 'Zone_Full_Name' in sorted_df.columns else sorted_df['BEC_Zone']

        # Create the bar chart of normalized densities
        bars = plt.bar(
            zone_labels, sorted_df['Yew_Trees_Per_Hectare'], color='darkgreen')

        plt.title('Pacific Yew Density by BEC Zone (trees per hectare)',
                  fontsize=14, fontweight='bold')
        plt.ylabel('Yew Trees per Hectare', fontweight='bold')
        plt.xlabel('BEC Zone', fontweight='bold')
        # Add density labels on top of bars
        plt.xticks(rotation=45, ha='right')
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.1,
                f'{height:.2f}',
                ha='center',
                fontweight='bold'
            )
        plt.tight_layout()

    # 1. Create a heatmap of subzones within zones (or 2nd plot if normalized data exists)
    if len(subzone_df) > 0:
        plt_idx = 2 if 'Normalized_Percentage' in bec_df.columns else 1
        plt.subplot(
            3 if 'Normalized_Percentage' in bec_df.columns else 2, 1, plt_idx)

        # Prepare data for the heatmap - pivot to get zones as rows and subzones as columns
        # Limit to top zones for readability
        top_zones = bec_df.head(6)['BEC_Zone'].tolist()
        filtered_subzone_df = subzone_df[subzone_df['BEC_Zone'].isin(
            top_zones)]

        # Create a pivot table
        pivot_data = filtered_subzone_df.pivot_table(
            index='BEC_Zone',
            columns='BEC_Subzone',
            values='Count',
            fill_value=0
        )

        # Create the heatmap
        sns.heatmap(
            pivot_data,
            cmap="Greens",
            annot=True,
            fmt="g",
            linewidths=.5,
            cbar_kws={'label': 'Number of Sites'}
        )

        plt.title('Distribution of Pacific Yew across BEC Subzones',
                  fontsize=14, fontweight='bold')
        plt.ylabel('BEC Zone', fontweight='bold')
        plt.xlabel('BEC Subzone', fontweight='bold')

    # 2. Create a detailed breakdown of top subzone+variant combinations (or 3rd plot if normalized data exists)
    plt_idx = 3 if 'Normalized_Percentage' in bec_df.columns else 2
    plt.subplot(
        3 if 'Normalized_Percentage' in bec_df.columns else 2, 1, plt_idx)

    # Combine zone, subzone and variant into a single descriptive name
    if len(variant_df) > 0:
        variant_df['Full_Description'] = variant_df.apply(
            lambda row: f"{row['BEC_Zone']} {row['BEC_Subzone']}{row['BEC_Variant']}", axis=1
        )

        # Get the top 15 combinations by count
        top_variants = variant_df.sort_values(
            'Count', ascending=False).head(15)

        # Create the bar chart
        sns.barplot(
            x='Count',
            y='Full_Description',
            data=top_variants,
            palette='Greens_d'
        )

        # Add count labels to the bars
        for i, row in enumerate(top_variants.itertuples()):
            plt.text(
                row.Count + 0.1,
                i,
                f"{row.Count} ({row.Percentage_Overall:.1f}%)",
                va='center',
                fontweight='bold'
            )

        plt.title('Top BEC Zone+Subzone+Variant Combinations for Pacific Yew',
                  fontsize=14, fontweight='bold')
        plt.ylabel('BEC Zone+Subzone+Variant', fontweight='bold')
        plt.xlabel('Number of Sites', fontweight='bold')
        plt.tight_layout()

    # Add a main title
    plt.suptitle('Detailed Biogeoclimatic Analysis of Pacific Yew Distribution',
                 fontsize=16, fontweight='bold', y=0.98)

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the main title
    plt.savefig('pacific_yew_bec_details.png', dpi=300)
    print("Detailed BEC visualizations saved to 'pacific_yew_bec_details.png'")


def generate_report(associations_df, bec_data, yew_stats, mapping_data=None):
    """
    Generate a comprehensive textual report of the Pacific Yew analysis results.

    Args:
        associations_df: DataFrame of species associations
        bec_data: Tuple of (bec_df, subzone_df, variant_df) DataFrames
        yew_stats: Dictionary with yew statistics
        mapping_data: Tuple containing reference mapping dictionaries (species_map, bec_zone_map, etc.)
    """
    # Unpack BEC data
    bec_df, subzone_df, variant_df = bec_data

    # Unpack mapping data if provided
    species_map = None
    bec_zone_map = None
    bec_subzone_map = None
    bec_variant_map = None

    if mapping_data:
        species_map, bec_zone_map, bec_subzone_map, bec_variant_map = mapping_data
    """
    Args:
        associations_df: DataFrame of species associations
        bec_df: DataFrame of BEC zone occurrences
        yew_stats: Dictionary with yew statistics
    """
    # Create a more visually distinct report header
    print("\n" + "="*70)
    print(" " * 15 + "PACIFIC YEW (Taxus brevifolia) ANALYSIS REPORT")
    print(" " * 10 + "Understanding Ecological Associations and Distribution Patterns")
    print("="*70)

    # Add timestamp
    from datetime import datetime
    print(f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Yew characteristics section with improved formatting
    if yew_stats:
        print("\n" + "-"*60)
        print("PACIFIC YEW CHARACTERISTICS:")
        print("-"*60)
        print(
            f"• Total Pacific Yew trees identified in dataset: {yew_stats['count']}")

        if yew_stats['avg_dbh']:
            print(f"• Average diameter (DBH): {yew_stats['avg_dbh']:.2f} cm")
            print(f"• Maximum diameter (DBH): {yew_stats['max_dbh']:.2f} cm")

        if yew_stats['avg_height']:
            print(f"• Average height: {yew_stats['avg_height']:.2f} m")
            print(f"• Maximum height: {yew_stats['max_height']:.2f} m")

    # Species associations section with ecological context
    if len(associations_df) > 0:
        print("\n" + "-"*60)
        print("TOP ASSOCIATED TREE SPECIES:")
        print("(Trees most commonly found growing alongside Pacific Yew)")
        print("-"*60)

        for i, (_, row) in enumerate(associations_df.head(10).iterrows()):
            # Use common name if available
            species_name = row['Common_Name'] if 'Common_Name' in associations_df.columns else row['Species']
            print(f"{i+1}. {species_name}")
            print(
                f"   • Found in {row['Co-occurrence_Count']} sites ({row['Co-occurrence_Percentage']:.1f}% of all Pacific Yew sites)")

            # Add average DBH information if available
            if pd.notna(row['Average_DBH']):
                print(
                    f"   • Average diameter (DBH): {row['Average_DBH']:.2f} cm")

            print()  # Add space between entries

    # BEC zone distribution with ecological context
    if len(bec_df) > 0:
        print("\n" + "-"*60)
        if 'Normalized_Percentage' in bec_df.columns:
            print("BIOGEOCLIMATIC ZONE DISTRIBUTION (NORMALIZED BY TREE COUNT):")
            print("(Relative abundance of Pacific Yew across ecological zones)")
        else:
            print("BIOGEOCLIMATIC ZONE DISTRIBUTION:")
            print("(Ecological zones where Pacific Yew is predominantly found)")
        print("-"*60)

        for i, (_, row) in enumerate(bec_df.iterrows()):
            # Use full zone name if available
            zone_name = row['Zone_Full_Name'] if 'Zone_Full_Name' in bec_df.columns else row['BEC_Zone']
            subzones = row['Common_Subzones'] if row['Common_Subzones'] else "None specified"

            print(f"{i+1}. {zone_name}")
            print(
                f"   • Contains {row['Count']} Pacific Yew sites ({row['Percentage']:.1f}% of total)")

            # Add normalized statistics if available
            if 'Normalized_Percentage' in row and pd.notna(row['Normalized_Percentage']):
                if 'Yew_Trees_Per_Hectare' in row and pd.notna(row['Yew_Trees_Per_Hectare']):
                    print(
                        f"   • Normalized density: {row['Yew_Trees_Per_Hectare']:.2f} Yew trees per hectare")
                    if 'Total_Plot_Area' in row and pd.notna(row['Total_Plot_Area']):
                        print(
                            f"   • Total plot area in zone: {row['Total_Plot_Area']:.2f} hectares")
                    if 'Yew_Trees' in row and pd.notna(row['Yew_Trees']):
                        print(f"   • Yew trees in zone: {row['Yew_Trees']}")
                    print(
                        f"   • Area-based comparative index: {row['Normalized_Percentage']:.1f}%")
                elif 'Yew_Density_Per_1000' in row and pd.notna(row['Yew_Density_Per_1000']):
                    print(
                        f"   • Normalized density: {row['Yew_Density_Per_1000']:.2f} Yew trees per 1000 trees")
                    print(f"   • Total trees in zone: {row['Total_Trees']}")
                    print(f"   • Yew trees in zone: {row['Yew_Trees']}")
                    print(
                        f"   • Average trees per site: {row['Trees_Per_Site']:.1f}")

            print(f"   • Common subzones: {subzones}")

            # Print variant details if available
            if 'Variant_Details' in row and row['Variant_Details']:
                print(f"   • Variants: {row['Variant_Details']}")

            print()  # Add space between entries

    # Add detailed subzone and variant information
    if len(subzone_df) > 0:
        print("\n" + "-"*60)
        print("DETAILED BIOGEOCLIMATIC SUBZONE ANALYSIS:")
        print("(Specific subzones where Pacific Yew is predominantly found)")
        print("-"*60)

        # Check if the expected column exists
        if 'BEC_Zone' in subzone_df.columns:
            # Group by zone for easier reading
            for zone in subzone_df['BEC_Zone'].unique():
                zone_full = zone_name = bec_zone_map.get(
                    zone, zone) if bec_zone_map else zone
                print(f"\n{zone} - {zone_full}:")

                zone_subzones = subzone_df[subzone_df['BEC_Zone'] == zone].sort_values(
                    'Count', ascending=False)

                for i, (_, row) in enumerate(zone_subzones.iterrows()):
                    if row['BEC_Subzone'] != 'Unspecified':
                        print(f"   • Subzone {row['BEC_Subzone']}: {row['Count']} sites " +
                              f"({row['Percentage_Within_Zone']:.1f}% of {zone} sites, " +
                              f"{row['Percentage_Overall']:.1f}% of all Pacific Yew sites)")
        else:
            print("ERROR: 'BEC_Zone' column not found in subzone DataFrame")
            print(f"Available columns: {list(subzone_df.columns)}")
    else:
        print("\nNo subzone data available for detailed analysis.")

        # Also include top variant combinations
        if len(variant_df) > 0:
            print("\n" + "-"*60)
            print("TOP 10 BIOGEOCLIMATIC ZONE+SUBZONE+VARIANT COMBINATIONS:")
            print("-"*60)

            top_variants = variant_df.sort_values(
                'Count', ascending=False).head(10)
            for i, (_, row) in enumerate(top_variants.iterrows()):
                if row['BEC_Variant'] != 'Unspecified' and row['BEC_Subzone'] != 'Unspecified':
                    full_combo = f"{row['BEC_Zone']} {row['BEC_Subzone']}{row['BEC_Variant']}"
                    zone_full = bec_zone_map.get(
                        row['BEC_Zone'], row['BEC_Zone']) if bec_zone_map else row['BEC_Zone']
                    print(f"{i+1}. {full_combo} ({zone_full})")
                    print(
                        f"   • {row['Count']} sites ({row['Percentage_Overall']:.1f}% of all Pacific Yew sites)")

            print(
                "\nNote: A detailed visualization of biogeoclimatic zones, subzones, and variants")
            print("has been saved to 'pacific_yew_bec_details.png'.")

    # Summary and ecological implications
    print("\n" + "-"*60)
    print("ECOLOGICAL SUMMARY:")
    print("-"*60)

    # Generate some ecological insights based on the data
    if len(associations_df) > 0 and len(bec_df) > 0:
        top_species = associations_df.iloc[0]['Common_Name'] if 'Common_Name' in associations_df.columns else associations_df.iloc[0]['Species']

        # If normalized data is available, use it to determine the top zone by density
        if 'Normalized_Percentage' in bec_df.columns:
            density_field = 'Yew_Trees_Per_Hectare' if 'Yew_Trees_Per_Hectare' in bec_df.columns else 'Yew_Density_Per_1000'
            density_sorted_df = bec_df.sort_values(
                density_field, ascending=False)
            top_zone_by_count = bec_df.iloc[0]['Zone_Full_Name'] if 'Zone_Full_Name' in bec_df.columns else bec_df.iloc[0]['BEC_Zone']
            top_zone_by_density = density_sorted_df.iloc[0][
                'Zone_Full_Name'] if 'Zone_Full_Name' in density_sorted_df.columns else density_sorted_df.iloc[0]['BEC_Zone']

            print(
                f"Pacific Yew appears to be most strongly associated with {top_species}.")
            print(
                f"By raw count, it is most commonly found in the {top_zone_by_count} biogeoclimatic zone.")

            if 'Yew_Trees_Per_Hectare' in bec_df.columns:
                print(
                    f"However, when normalized by plot area, it is most prevalent in the {top_zone_by_density} zone.")
                print(
                    f"This difference suggests that while absolute numbers may be higher in {top_zone_by_count},")
                print(
                    f"Pacific Yew has a higher spatial density in the {top_zone_by_density} zone.")
                print(
                    "These patterns indicate specific ecological preferences and may inform management decisions.")
            else:
                print(
                    f"However, when normalized by tree density, it is most prevalent in the {top_zone_by_density} zone.")
                print(
                    f"This difference suggests that while absolute numbers may be higher in {top_zone_by_count},")
                print(
                    f"Pacific Yew represents a larger proportion of the forest composition in {top_zone_by_density}.")
                print(
                    "These patterns suggest specific ecological niches and may inform conservation strategies.")
        else:
            top_zone = bec_df.iloc[0]['Zone_Full_Name'] if 'Zone_Full_Name' in bec_df.columns else bec_df.iloc[0]['BEC_Zone']

            print(
                f"Pacific Yew appears to be most strongly associated with {top_species}")
            print(
                f"and is predominantly found in the {top_zone} biogeoclimatic zone.")
            print("This association suggests specific ecological requirements and may")
            print(
                "inform conservation and management strategies for this important species.")

    print("\n" + "="*70)
    print("Report generated successfully. Analysis data saved to CSV files.")
    print("="*70)


def main():
    """Main analysis function."""
    # Print a visually appealing banner
    print("\n" + "="*70)
    print(" "*20 + "PACIFIC YEW ANALYSIS TOOL")
    print(" "*10 + "Ecological Association & Distribution Study with Site Status Analysis")
    print("-"*70)
    print(" Pacific Yew (Taxus brevifolia) - Species Code: TW")
    print(" Analysis of tree associations and biogeoclimatic zone distribution")
    print(" Latest visits only • Active vs Inactive site comparison")
    print("="*70)
    print("Starting Pacific Yew (Taxus brevifolia) analysis...")

    # Load reference data for species and BEC zone translations
    species_map, bec_zone_map, bec_subzone_map, bec_variant_map = load_reference_data()

    # Load data
    tree_data, header_data = load_data()
    if tree_data is None:
        print("Error: Could not load tree data. Analysis aborted.")
        return

    # Identify sites with Pacific Yew (now returns active, inactive, and all sites)
    active_yew_sites, inactive_yew_sites, all_yew_sites = identify_yew_sites(
        tree_data, header_data)
    if not all_yew_sites:
        print("No Pacific Yew sites found. Analysis aborted.")
        return

    # Analyze species associations for all sites combined
    print("\n" + "="*50)
    print("OVERALL ANALYSIS (All Sites)")
    print("="*50)
    associations_all = find_associated_species(tree_data, all_yew_sites, "All")

    # Analyze species associations for active sites only
    associations_active = pd.DataFrame()
    if active_yew_sites:
        print("\n" + "="*50)
        print("ACTIVE SITES ANALYSIS")
        print("="*50)
        associations_active = find_associated_species(
            tree_data, active_yew_sites, "Active")

    # Analyze species associations for inactive sites only
    associations_inactive = pd.DataFrame()
    if inactive_yew_sites:
        print("\n" + "="*50)
        print("INACTIVE SITES ANALYSIS")
        print("="*50)
        associations_inactive = find_associated_species(
            tree_data, inactive_yew_sites, "Inactive")

    # Analyze BEC zones, subzones, and variants for all sites
    bec_df_all, subzone_df_all, variant_df_all = find_bec_zones(
        header_data, all_yew_sites, tree_data)

    # Analyze BEC zones for active sites only
    bec_df_active, subzone_df_active, variant_df_active = pd.DataFrame(
    ), pd.DataFrame(), pd.DataFrame()
    if active_yew_sites:
        bec_df_active, subzone_df_active, variant_df_active = find_bec_zones(
            header_data, active_yew_sites, tree_data)

    # Analyze BEC zones for inactive sites only
    bec_df_inactive, subzone_df_inactive, variant_df_inactive = pd.DataFrame(
    ), pd.DataFrame(), pd.DataFrame()
    if inactive_yew_sites:
        bec_df_inactive, subzone_df_inactive, variant_df_inactive = find_bec_zones(
            header_data, inactive_yew_sites, tree_data)

    # Analyze Yew characteristics
    yew_stats = analyze_yew_characteristics(tree_data)

    # Add human-readable names to the results before visualization
    for associations_df in [associations_all, associations_active, associations_inactive]:
        if len(associations_df) > 0 and species_map:
            associations_df['Common_Name'] = associations_df['Species'].apply(
                lambda x: get_species_name(x, species_map))

    for bec_df in [bec_df_all, bec_df_active, bec_df_inactive]:
        if len(bec_df) > 0 and bec_zone_map and 'Common_Subzones' in bec_df.columns:
            # Add full zone names to the BEC zone data
            bec_df['Zone_Full_Name'] = bec_df.apply(
                lambda row: get_bec_zone_name(row['BEC_Zone'],
                                              row['Common_Subzones'].split(
                                                  '(')[0] if '(' in row['Common_Subzones'] else None,
                                              None, bec_zone_map, bec_subzone_map, bec_variant_map),
                axis=1)

    # Also add full names to subzone and variant data if available
    for subzone_df in [subzone_df_all, subzone_df_active, subzone_df_inactive]:
        if len(subzone_df) > 0 and bec_zone_map and 'BEC_Zone' in subzone_df.columns:
            subzone_df['Zone_Full_Name'] = subzone_df['BEC_Zone'].apply(
                lambda x: bec_zone_map.get(x, x))

    # Pack BEC data for functions that expect it
    bec_data_all = (bec_df_all, subzone_df_all, variant_df_all)
    bec_data_active = (bec_df_active, subzone_df_active, variant_df_active)
    bec_data_inactive = (
        bec_df_inactive, subzone_df_inactive, variant_df_inactive)

    # Pack mapping data for reference
    mapping_data = (species_map, bec_zone_map,
                    bec_subzone_map, bec_variant_map)

    # Generate report and visualizations with the translated names
    print("\n" + "="*70)
    print("GENERATING COMPREHENSIVE REPORTS")
    print("="*70)

    # Generate separate reports for each analysis
    print("\n--- OVERALL ANALYSIS REPORT ---")
    generate_report(associations_all, bec_data_all, yew_stats, mapping_data)

    if len(associations_active) > 0:
        print("\n--- ACTIVE SITES ANALYSIS REPORT ---")
        generate_report(associations_active, bec_data_active,
                        yew_stats, mapping_data)

    if len(associations_inactive) > 0:
        print("\n--- INACTIVE SITES ANALYSIS REPORT ---")
        generate_report(associations_inactive,
                        bec_data_inactive, yew_stats, mapping_data)

    # Generate visualizations (using overall data)
    visualize_results(associations_all, bec_data_all, yew_stats, tree_data)

    # Save results to CSV with site status information
    print("\n" + "="*70)
    print("SAVING ANALYSIS RESULTS")
    print("="*70)

    # Combine all associations into one file with status column
    all_associations = []
    if len(associations_all) > 0:
        associations_all['Analysis_Type'] = 'All_Sites'
        all_associations.append(associations_all)
    if len(associations_active) > 0:
        associations_active['Analysis_Type'] = 'Active_Sites_Only'
        all_associations.append(associations_active)
    if len(associations_inactive) > 0:
        associations_inactive['Analysis_Type'] = 'Inactive_Sites_Only'
        all_associations.append(associations_inactive)

    if all_associations:
        combined_associations = pd.concat(all_associations, ignore_index=True)
        combined_associations.to_csv(
            'pacific_yew_associations.csv', index=False)
        print("Species associations saved to 'pacific_yew_associations.csv'")

    # Save BEC zone data
    if len(bec_df_all) > 0:
        bec_df_all.to_csv('pacific_yew_bec_zones.csv', index=False)
        print("BEC zone data saved to 'pacific_yew_bec_zones.csv'")

    if len(bec_df_active) > 0:
        bec_df_active.to_csv('pacific_yew_bec_zones_active.csv', index=False)
        print("Active sites BEC zone data saved to 'pacific_yew_bec_zones_active.csv'")

    if len(bec_df_inactive) > 0:
        bec_df_inactive.to_csv(
            'pacific_yew_bec_zones_inactive.csv', index=False)
        print("Inactive sites BEC zone data saved to 'pacific_yew_bec_zones_inactive.csv'")

    # Save the detailed subzone and variant data
    if len(subzone_df_all) > 0:
        subzone_df_all.to_csv('pacific_yew_bec_subzones.csv', index=False)
        print("BEC subzone data saved to 'pacific_yew_bec_subzones.csv'")

    if len(variant_df_all) > 0:
        variant_df_all.to_csv('pacific_yew_bec_variants.csv', index=False)
        print("BEC variant data saved to 'pacific_yew_bec_variants.csv'")

    # Generate summary comparison
    print("\n" + "="*70)
    print("SITE STATUS COMPARISON SUMMARY")
    print("="*70)
    print(f"Total yew sites found: {len(all_yew_sites)}")
    print(
        f"Active sites: {len(active_yew_sites)} ({len(active_yew_sites)/len(all_yew_sites)*100:.1f}%)")
    print(
        f"Inactive sites: {len(inactive_yew_sites)} ({len(inactive_yew_sites)/len(all_yew_sites)*100:.1f}%)")
    print(f"Total yew trees (active sites): {sum(active_yew_sites.values())}")
    print(
        f"Total yew trees (inactive sites): {sum(inactive_yew_sites.values())}")

    print("\nAnalysis completed successfully!")
    print("Note: Analysis filtered to latest visit only for each site.")
    print("Results include separate analyses for active and inactive sites.")


if __name__ == "__main__":
    main()
