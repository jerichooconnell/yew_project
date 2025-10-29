import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, ks_2samp
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


def create_enhanced_histograms(bc_data, desc_map):
    """Create enhanced normalized histograms with statistical tests"""
    print("\nGenerating enhanced normalized histograms with statistical analysis...")

    # Define numerical columns of interest (excluding coordinates and visit info)
    numerical_cols = [
        'MEAS_YR', 'BA_HA_LS', 'BA_HA_DS', 'STEMS_HA_LS', 'STEMS_HA_DS',
        'VHA_WSV_LS', 'VHA_WSV_DS', 'VHA_NTWB_LS', 'VHA_NTWB_DS',
        'SI_M_TLSO', 'HT_TLSO', 'AGEB_TLSO', 'AGET_TLSO'
    ]

    # Filter to only columns with meaningful variation
    valid_cols = []
    statistical_results = []

    for col in numerical_cols:
        if col in bc_data.columns:
            yew_data = bc_data[bc_data['has_yew'] == 1][col].dropna()
            no_yew_data = bc_data[bc_data['has_yew'] == 0][col].dropna()

            if len(yew_data) > 5 and len(no_yew_data) > 5 and yew_data.nunique() > 3:
                valid_cols.append(col)

                # Perform statistical tests
                try:
                    # Mann-Whitney U test (non-parametric)
                    mw_stat, mw_pval = mannwhitneyu(
                        yew_data, no_yew_data, alternative='two-sided')

                    # Kolmogorov-Smirnov test (distribution comparison)
                    ks_stat, ks_pval = ks_2samp(yew_data, no_yew_data)

                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(yew_data) - 1) * yew_data.var() +
                                          (len(no_yew_data) - 1) * no_yew_data.var()) /
                                         (len(yew_data) + len(no_yew_data) - 2))
                    cohens_d = (yew_data.mean() - no_yew_data.mean()
                                ) / pooled_std if pooled_std > 0 else 0

                    statistical_results.append({
                        'variable': col,
                        'yew_mean': yew_data.mean(),
                        'no_yew_mean': no_yew_data.mean(),
                        'yew_median': yew_data.median(),
                        'no_yew_median': no_yew_data.median(),
                        'mann_whitney_p': mw_pval,
                        'ks_test_p': ks_pval,
                        'cohens_d': cohens_d,
                        'yew_n': len(yew_data),
                        'no_yew_n': len(no_yew_data)
                    })
                except:
                    statistical_results.append({
                        'variable': col,
                        'yew_mean': yew_data.mean(),
                        'no_yew_mean': no_yew_data.mean(),
                        'yew_median': yew_data.median(),
                        'no_yew_median': no_yew_data.median(),
                        'mann_whitney_p': np.nan,
                        'ks_test_p': np.nan,
                        'cohens_d': np.nan,
                        'yew_n': len(yew_data),
                        'no_yew_n': len(no_yew_data)
                    })

    # Calculate number of rows needed (3 columns per row)
    n_cols = len(valid_cols)
    n_rows = (n_cols + 2) // 3  # Ceiling division

    # Create figure
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 6*n_rows))
    fig.suptitle('Forest Structure Comparison: Sites with vs without Pacific Yew',
                 fontsize=16, fontweight='bold', y=0.98)

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

        # Get statistical results for this variable
        stats_result = next(
            (r for r in statistical_results if r['variable'] == col), None)

        if len(yew_data) > 0 and len(no_yew_data) > 0:
            # Calculate bins based on combined data range, removing extreme outliers
            combined_data = pd.concat([yew_data, no_yew_data])
            q1, q99 = combined_data.quantile([0.01, 0.99])

            # Filter extreme outliers for visualization
            yew_filtered = yew_data[(yew_data >= q1) & (yew_data <= q99)]
            no_yew_filtered = no_yew_data[(
                no_yew_data >= q1) & (no_yew_data <= q99)]

            if len(yew_filtered) > 0 and len(no_yew_filtered) > 0:
                bins = np.linspace(min(yew_filtered.min(), no_yew_filtered.min()),
                                   max(yew_filtered.max(), no_yew_filtered.max()), 25)

                # Create normalized histograms
                ax.hist(no_yew_filtered, bins=bins, alpha=0.6, density=True,
                        label=f'No Yew (n={len(no_yew_data)})', color='lightblue',
                        edgecolor='navy', linewidth=0.5)
                ax.hist(yew_filtered, bins=bins, alpha=0.8, density=True,
                        label=f'With Yew (n={len(yew_data)})', color='darkred',
                        edgecolor='darkred', linewidth=0.5)

                # Add vertical lines for means
                ax.axvline(no_yew_data.mean(), color='blue', linestyle='--', alpha=0.8,
                           label=f'No Yew Mean: {no_yew_data.mean():.1f}')
                ax.axvline(yew_data.mean(), color='red', linestyle='--', alpha=0.8,
                           label=f'Yew Mean: {yew_data.mean():.1f}')

                # Formatting with descriptive titles
                if col in desc_map:
                    # Use description from data dictionary as title
                    title = desc_map[col]
                    # Truncate very long titles
                    if len(title) > 50:
                        title = title[:50] + '...'
                    ax.set_title(title, fontsize=12, fontweight='bold')
                    # Add the column name as subtitle
                    ax.text(0.5, 0.92, f'({col})', transform=ax.transAxes,
                            fontsize=9, ha='center', va='top', style='italic', alpha=0.7)
                else:
                    # Fallback to formatted column name if no description available
                    title = col.replace('_', ' ').title()
                    ax.set_title(f'{title}', fontsize=12, fontweight='bold')

                ax.set_xlabel('Value')
                ax.set_ylabel('Normalized Frequency')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

                # Add statistical significance if available
                if stats_result and not np.isnan(stats_result['mann_whitney_p']):
                    p_val = stats_result['mann_whitney_p']
                    significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                    effect_size = abs(stats_result['cohens_d']) if not np.isnan(
                        stats_result['cohens_d']) else 0

                    # Add text box with statistics
                    textstr = f'p={p_val:.3f} {significance}\nCohen\'s d={effect_size:.3f}'
                    props = dict(boxstyle='round',
                                 facecolor='wheat', alpha=0.8)
                    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
                            verticalalignment='top', bbox=props)

    # Hide unused subplots
    for i in range(len(valid_cols), len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('pacific_yew_detailed_histograms.png',
                dpi=300, bbox_inches='tight')
    print("Enhanced histograms saved to 'pacific_yew_detailed_histograms.png'")

    # Create summary table
    stats_df = pd.DataFrame(statistical_results)
    stats_df['significant'] = stats_df['mann_whitney_p'] < 0.05
    stats_df['effect_size_magnitude'] = stats_df['cohens_d'].abs()
    stats_df = stats_df.sort_values('effect_size_magnitude', ascending=False)

    print("\n" + "="*80)
    print("STATISTICAL COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Variable':<15} {'Yew Mean':<10} {'No-Yew Mean':<12} {'p-value':<10} {'Effect Size':<12} {'Significant':<12}")
    print("-"*80)

    for _, row in stats_df.head(10).iterrows():
        sig_symbol = "***" if row['mann_whitney_p'] < 0.001 else "**" if row['mann_whitney_p'] < 0.01 else "*" if row['mann_whitney_p'] < 0.05 else ""
        print(f"{row['variable']:<15} {row['yew_mean']:<10.1f} {row['no_yew_mean']:<12.1f} "
              f"{row['mann_whitney_p']:<10.4f} {row['cohens_d']:<12.3f} {sig_symbol:<12}")

    # Save statistical results
    stats_df.to_csv('yew_statistical_comparison.csv', index=False)
    print("\nStatistical comparison saved to 'yew_statistical_comparison.csv'")

    return valid_cols, statistical_results


def main():
    """Main analysis function"""
    # Load data
    bc_data, desc_map = load_data()

    # Identify yew presence
    bc_data = identify_yew_presence(bc_data)

    # Create enhanced histograms
    valid_cols, stats_results = create_enhanced_histograms(bc_data, desc_map)

    print(
        f"\nAnalysis complete. Generated histograms for {len(valid_cols)} numerical variables.")


if __name__ == "__main__":
    main()
