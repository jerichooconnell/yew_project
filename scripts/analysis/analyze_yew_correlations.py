import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, pearsonr, spearmanr
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Load the bc_sample_data and data dictionary"""
    print("Loading BC sample data...")

    # Load the main data
    bc_data = pd.read_csv('bc_sample_data-2025-10-09/bc_sample_data.csv')
    print(f"Loaded {len(bc_data)} records")

    # Load the data dictionary
    data_dict = pd.read_csv('bc_sample_data-2025-10-09/data_dictionary.csv')

    # Create a mapping from column names to descriptions
    desc_map = {}
    for _, row in data_dict.iterrows():
        if pd.notna(row['Attribute']) and pd.notna(row['Description']):
            desc_map[row['Attribute']] = row['Description']

    return bc_data, desc_map


def identify_yew_presence(bc_data):
    """Identify sites with Pacific Yew presence"""
    print("\nIdentifying Pacific Yew presence...")

    # Pacific Yew has species code 'TW' - check SPB_CPCT_LS column for yew presence
    bc_data['has_yew'] = 0

    # Check if Pacific Yew (TW) appears in the species composition string
    yew_mask = bc_data['SPB_CPCT_LS'].str.contains('TW', na=False)
    bc_data.loc[yew_mask, 'has_yew'] = 1

    yew_sites = bc_data[bc_data['has_yew'] == 1]
    no_yew_sites = bc_data[bc_data['has_yew'] == 0]

    print(
        f"Sites with Pacific Yew: {len(yew_sites)} ({len(yew_sites)/len(bc_data)*100:.1f}%)")
    print(
        f"Sites without Pacific Yew: {len(no_yew_sites)} ({len(no_yew_sites)/len(bc_data)*100:.1f}%)")

    return bc_data


def analyze_categorical_correlations(bc_data, desc_map):
    """Analyze correlations for categorical variables"""
    print("\nAnalyzing categorical variable correlations with Pacific Yew presence...")

    categorical_cols = [
        'BEC_ZONE', 'BECLABEL', 'SPECIES_CLASS', 'SPC_LIVE_1', 'SAMPLE_ESTABLISHMENT_TYPE',
        'PROJECT_DESIGN', 'PSP_STATUS', 'YSM_MAIN_FM', 'MAT_MAIN_FM', 'FIRST_MSMT', 'LAST_MSMT',
        'SAMP_TYP', 'TSA_DESC', 'OWN_SCHED_DESCRIP'
    ]

    categorical_results = []

    for col in categorical_cols:
        if col in bc_data.columns:
            # Remove missing values
            temp_data = bc_data[[col, 'has_yew']].dropna()

            if len(temp_data) > 0:
                try:
                    # Create contingency table
                    contingency_table = pd.crosstab(
                        temp_data[col], temp_data['has_yew'])

                    if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                        # Chi-square test
                        chi2, p_value, dof, expected = chi2_contingency(
                            contingency_table)

                        # Calculate Cramér's V (effect size)
                        n = contingency_table.sum().sum()
                        cramers_v = np.sqrt(
                            chi2 / (n * (min(contingency_table.shape) - 1)))

                        # Calculate yew percentage for each category
                        yew_percentages = {}
                        for category in contingency_table.index:
                            total_in_category = contingency_table.loc[category].sum(
                            )
                            yew_in_category = contingency_table.loc[category,
                                                                    1] if 1 in contingency_table.columns else 0
                            yew_percentages[category] = (
                                yew_in_category / total_in_category) * 100 if total_in_category > 0 else 0

                        categorical_results.append({
                            'variable': col,
                            'description': desc_map.get(col, 'No description available'),
                            'chi2_statistic': chi2,
                            'p_value': p_value,
                            'cramers_v': cramers_v,
                            'n_categories': len(contingency_table.index),
                            'yew_percentages': yew_percentages,
                            'sample_size': len(temp_data)
                        })
                except Exception as e:
                    print(f"Error analyzing {col}: {e}")

    # Sort by Cramér's V (effect size)
    categorical_results.sort(key=lambda x: x['cramers_v'], reverse=True)

    return categorical_results


def analyze_numerical_correlations(bc_data, desc_map):
    """Analyze correlations for numerical variables"""
    print("\nAnalyzing numerical variable correlations with Pacific Yew presence...")

    numerical_cols = [
        'VISIT_NUMBER', 'MEAS_YR', 'NO_PLOTS', 'UTIL', 'BA_HA_LS', 'BA_HA_DS',
        'STEMS_HA_LS', 'STEMS_HA_DS', 'VHA_WSV_LS', 'VHA_WSV_DS', 'VHA_NTWB_LS',
        'VHA_NTWB_DS', 'SI_M_TLSO', 'HT_TLSO', 'AGEB_TLSO', 'AGET_TLSO',
        'IP_EAST', 'IP_NRTH', 'BC_ALBERS_X', 'BC_ALBERS_Y'
    ]

    numerical_results = []

    for col in numerical_cols:
        if col in bc_data.columns:
            # Remove missing values and infinite values
            temp_data = bc_data[[col, 'has_yew']].replace(
                [np.inf, -np.inf], np.nan).dropna()

            if len(temp_data) > 10:  # Need reasonable sample size
                try:
                    # Pearson correlation
                    pearson_r, pearson_p = pearsonr(
                        temp_data[col], temp_data['has_yew'])

                    # Spearman correlation (rank-based, more robust)
                    spearman_r, spearman_p = spearmanr(
                        temp_data[col], temp_data['has_yew'])

                    # Calculate means for yew vs no-yew sites
                    yew_mean = temp_data[temp_data['has_yew'] == 1][col].mean()
                    no_yew_mean = temp_data[temp_data['has_yew']
                                            == 0][col].mean()

                    numerical_results.append({
                        'variable': col,
                        'description': desc_map.get(col, 'No description available'),
                        'pearson_r': pearson_r,
                        'pearson_p': pearson_p,
                        'spearman_r': spearman_r,
                        'spearman_p': spearman_p,
                        'yew_mean': yew_mean,
                        'no_yew_mean': no_yew_mean,
                        'sample_size': len(temp_data)
                    })
                except Exception as e:
                    print(f"Error analyzing {col}: {e}")

    # Sort by absolute Spearman correlation (more robust for non-linear relationships)
    numerical_results.sort(key=lambda x: abs(x['spearman_r']), reverse=True)

    return numerical_results


def generate_correlation_report(categorical_results, numerical_results):
    """Generate a comprehensive report of correlations"""
    print("\n" + "="*80)
    print("PACIFIC YEW SITE CORRELATIONS ANALYSIS")
    print("="*80)

    print("\n" + "="*60)
    print("TOP CATEGORICAL VARIABLE CORRELATIONS")
    print("="*60)
    print("Ranked by Cramér's V (effect size: 0.1=small, 0.3=medium, 0.5=large)")
    print("-"*60)

    for i, result in enumerate(categorical_results[:10], 1):
        print(f"\n{i}. {result['variable']}")
        print(f"   Description: {result['description'][:80]}...")
        print(f"   Cramér's V: {result['cramers_v']:.4f}")
        print(f"   Chi-square p-value: {result['p_value']:.2e}")
        print(f"   Categories: {result['n_categories']}")
        print(f"   Sample size: {result['sample_size']}")

        # Show top categories with highest yew percentages
        top_yew_categories = sorted(result['yew_percentages'].items(),
                                    key=lambda x: x[1], reverse=True)[:3]
        print(f"   Top yew categories:")
        for cat, pct in top_yew_categories:
            if pct > 0:
                print(f"     • {cat}: {pct:.1f}% yew presence")

    print("\n" + "="*60)
    print("TOP NUMERICAL VARIABLE CORRELATIONS")
    print("="*60)
    print("Ranked by absolute Spearman correlation coefficient")
    print("-"*60)

    for i, result in enumerate(numerical_results[:15], 1):
        print(f"\n{i}. {result['variable']}")
        print(f"   Description: {result['description'][:80]}...")
        print(f"   Spearman correlation: {result['spearman_r']:.4f}")
        print(f"   Spearman p-value: {result['spearman_p']:.2e}")
        print(f"   Pearson correlation: {result['pearson_r']:.4f}")
        print(f"   Mean (yew sites): {result['yew_mean']:.2f}")
        print(f"   Mean (no-yew sites): {result['no_yew_mean']:.2f}")
        print(f"   Sample size: {result['sample_size']}")


def create_normalized_histograms(bc_data, desc_map):
    """Create normalized histograms comparing yew vs non-yew populations for numerical variables"""
    print("\nGenerating normalized histograms for numerical variables...")

    # Define numerical columns
    numerical_cols = [
        'VISIT_NUMBER', 'MEAS_YR', 'NO_PLOTS', 'BA_HA_LS', 'BA_HA_DS',
        'STEMS_HA_LS', 'STEMS_HA_DS', 'VHA_WSV_LS', 'VHA_WSV_DS', 'VHA_NTWB_LS',
        'VHA_NTWB_DS', 'SI_M_TLSO', 'HT_TLSO', 'AGEB_TLSO', 'AGET_TLSO',
        'IP_EAST', 'IP_NRTH', 'BC_ALBERS_X', 'BC_ALBERS_Y'
    ]

    # Filter to only columns with meaningful variation
    valid_cols = []
    for col in numerical_cols:
        if col in bc_data.columns:
            temp_data = bc_data[col].dropna()
            if len(temp_data) > 10 and temp_data.nunique() > 5:  # Has variation
                valid_cols.append(col)

    # Calculate number of rows needed (4 columns per row)
    n_cols = len(valid_cols)
    n_rows = (n_cols + 3) // 4  # Ceiling division

    # Create figure
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5*n_rows))
    fig.suptitle('Normalized Histograms: Yew vs Non-Yew Populations',
                 fontsize=16, fontweight='bold')

    # Flatten axes array for easier indexing
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    # Create histograms for each numerical variable
    for i, col in enumerate(valid_cols):
        ax = axes_flat[i]

        # Get data for yew and non-yew sites
        yew_data = bc_data[bc_data['has_yew'] == 1][col].dropna()
        no_yew_data = bc_data[bc_data['has_yew'] == 0][col].dropna()

        if len(yew_data) > 0 and len(no_yew_data) > 0:
            # Calculate bins based on combined data range
            combined_data = pd.concat([yew_data, no_yew_data])
            bins = np.linspace(combined_data.min(), combined_data.max(), 30)

            # Create normalized histograms
            ax.hist(no_yew_data, bins=bins, alpha=0.7, density=True,
                    label=f'No Yew (n={len(no_yew_data)})', color='lightblue', edgecolor='black')
            ax.hist(yew_data, bins=bins, alpha=0.7, density=True,
                    label=f'With Yew (n={len(yew_data)})', color='red', edgecolor='black')

            # Formatting with descriptive titles
            if col in desc_map:
                # Use description from data dictionary as title
                title = desc_map[col]
                # Truncate very long titles
                if len(title) > 60:
                    title = title[:60] + '...'
                ax.set_title(title, fontsize=10, fontweight='bold')
                # Add the column name as subtitle
                ax.text(0.5, 0.95, f'({col})', transform=ax.transAxes,
                        fontsize=8, ha='center', va='top', style='italic', alpha=0.7)
            else:
                # Fallback to column name if no description available
                ax.set_title(f'{col}', fontsize=10, fontweight='bold')

            ax.set_xlabel('Value')
            ax.set_ylabel('Normalized Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(len(valid_cols), len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout()
    plt.savefig('pacific_yew_numerical_histograms.png',
                dpi=300, bbox_inches='tight')
    print("Normalized histograms saved to 'pacific_yew_numerical_histograms.png'")

    return valid_cols


def create_visualizations(bc_data, categorical_results, numerical_results):
    """Create visualizations of the strongest correlations"""
    print("\nGenerating correlation visualizations...")

    # Set up the figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Pacific Yew Site Correlations Analysis',
                 fontsize=16, fontweight='bold')

    # 1. Top categorical correlation - BEC Zone distribution
    if len(categorical_results) > 0:
        top_cat = categorical_results[0]
        ax1 = axes[0, 0]

        # Create a more focused plot for BEC zones
        bec_yew_data = []
        for bec_zone, pct in top_cat['yew_percentages'].items():
            zone_data = bc_data[bc_data[top_cat['variable']] == bec_zone]
            if len(zone_data) >= 10:  # Only include zones with sufficient data
                bec_yew_data.append({
                    'BEC_Zone': bec_zone,
                    'Yew_Percentage': pct,
                    'Total_Sites': len(zone_data)
                })

        bec_df = pd.DataFrame(bec_yew_data).sort_values(
            'Yew_Percentage', ascending=True)

        if len(bec_df) > 0:
            bars = ax1.barh(range(len(bec_df)), bec_df['Yew_Percentage'])
            ax1.set_yticks(range(len(bec_df)))
            ax1.set_yticklabels(bec_df['BEC_Zone'])
            ax1.set_xlabel('Pacific Yew Presence (%)')
            ax1.set_title(
                f'{top_cat["variable"]} vs Yew Presence\n(Cramér\'s V = {top_cat["cramers_v"]:.3f})')

            # Color bars by yew percentage
            for i, bar in enumerate(bars):
                bar.set_color(plt.cm.viridis(
                    bec_df.iloc[i]['Yew_Percentage'] / bec_df['Yew_Percentage'].max()))

    # 2. Top numerical correlation
    if len(numerical_results) > 0:
        top_num = numerical_results[0]
        ax2 = axes[0, 1]

        plot_data = bc_data[[top_num['variable'], 'has_yew']].dropna()
        if len(plot_data) > 0:
            # Box plot
            yew_data = plot_data[plot_data['has_yew']
                                 == 1][top_num['variable']]
            no_yew_data = plot_data[plot_data['has_yew']
                                    == 0][top_num['variable']]

            box_data = [no_yew_data, yew_data]
            ax2.boxplot(box_data, labels=['No Yew', 'With Yew'])
            ax2.set_ylabel(top_num['variable'])
            ax2.set_title(
                f'{top_num["variable"]} vs Yew Presence\n(Spearman r = {top_num["spearman_r"]:.3f})')

    # 3. Species composition analysis
    ax3 = axes[1, 0]

    # Extract leading species and their yew association
    species_yew = defaultdict(lambda: {'total': 0, 'with_yew': 0})

    for _, row in bc_data.iterrows():
        if pd.notna(row['SPC_LIVE_1']) and row['SPC_LIVE_1'] != '':
            species = row['SPC_LIVE_1']
            species_yew[species]['total'] += 1
            if row['has_yew'] == 1:
                species_yew[species]['with_yew'] += 1

    # Calculate percentages and filter for species with enough data
    species_data = []
    for species, counts in species_yew.items():
        if counts['total'] >= 20:  # Only species with 20+ occurrences
            pct = (counts['with_yew'] / counts['total']) * 100
            species_data.append({
                'species': species,
                'yew_percentage': pct,
                'total_sites': counts['total']
            })

    species_df = pd.DataFrame(species_data).sort_values(
        'yew_percentage', ascending=True)

    if len(species_df) > 0:
        # Take top and bottom species
        top_bottom = pd.concat(
            [species_df.head(5), species_df.tail(5)]).drop_duplicates()

        bars = ax3.barh(range(len(top_bottom)), top_bottom['yew_percentage'])
        ax3.set_yticks(range(len(top_bottom)))
        ax3.set_yticklabels(top_bottom['species'])
        ax3.set_xlabel('Pacific Yew Co-occurrence (%)')
        ax3.set_title('Leading Species vs Yew Co-occurrence')

        # Color bars
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.RdYlGn(
                top_bottom.iloc[i]['yew_percentage'] / top_bottom['yew_percentage'].max()))

    # 4. Geographic distribution
    ax4 = axes[1, 1]

    geo_data = bc_data[['BC_ALBERS_X', 'BC_ALBERS_Y', 'has_yew']].dropna()
    if len(geo_data) > 0:
        # Scatter plot
        no_yew = geo_data[geo_data['has_yew'] == 0]
        yew = geo_data[geo_data['has_yew'] == 1]

        ax4.scatter(no_yew['BC_ALBERS_X'], no_yew['BC_ALBERS_Y'],
                    c='lightblue', alpha=0.5, s=1, label='No Yew')
        ax4.scatter(yew['BC_ALBERS_X'], yew['BC_ALBERS_Y'],
                    c='red', alpha=0.7, s=3, label='With Yew')
        ax4.set_xlabel('BC Albers X')
        ax4.set_ylabel('BC Albers Y')
        ax4.set_title('Geographic Distribution of Yew Sites')
        ax4.legend()

    plt.tight_layout()
    plt.savefig('pacific_yew_correlations.png', dpi=300, bbox_inches='tight')
    print("Correlations visualization saved to 'pacific_yew_correlations.png'")


def main():
    """Main analysis function"""
    # Load data
    bc_data, desc_map = load_data()

    # Identify yew presence
    bc_data = identify_yew_presence(bc_data)

    # Analyze correlations
    categorical_results = analyze_categorical_correlations(bc_data, desc_map)
    numerical_results = analyze_numerical_correlations(bc_data, desc_map)

    # Generate report
    generate_correlation_report(categorical_results, numerical_results)

    # Create visualizations
    create_visualizations(bc_data, categorical_results, numerical_results)

    # Create normalized histograms for numerical variables
    create_normalized_histograms(bc_data, desc_map)

    # Save detailed results to CSV
    print("\nSaving detailed results...")

    # Save categorical results
    cat_df = pd.DataFrame(categorical_results)
    cat_df.to_csv('yew_categorical_correlations.csv', index=False)

    # Save numerical results
    num_df = pd.DataFrame(numerical_results)
    num_df.to_csv('yew_numerical_correlations.csv', index=False)

    print("Detailed results saved to:")
    print("- yew_categorical_correlations.csv")
    print("- yew_numerical_correlations.csv")
    print("- pacific_yew_correlations.png")
    print("- pacific_yew_numerical_histograms.png")


if __name__ == "__main__":
    main()
