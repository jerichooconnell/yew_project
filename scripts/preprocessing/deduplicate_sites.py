#!/usr/bin/env python3
"""
Deduplicate Site Data - Keep Most Recent Measurement
====================================================

Filters the BC sample data to keep only the most recent measurement
for each SITE_IDENTIFIER.

Some sites have multiple measurements over time (repeated visits).
This script keeps only the latest visit based on:
1. MEAS_YR (measurement year) - most recent year
2. VISIT_NUMBER - highest visit number if same year

Author: Analysis Tool
Date: November 7, 2025
"""

import pandas as pd
from pathlib import Path


def deduplicate_sites(input_path, output_path):
    """
    Remove duplicate site measurements, keeping only the most recent.

    Args:
        input_path: Path to input CSV file
        output_path: Path to save deduplicated CSV file
    """
    print("="*80)
    print("DEDUPLICATING SITE MEASUREMENTS")
    print("="*80)

    # Load data
    print(f"\nLoading data from: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"  Total records: {len(df):,}")
    print(f"  Unique sites: {df['SITE_IDENTIFIER'].nunique():,}")

    # Check for duplicates
    duplicated = df['SITE_IDENTIFIER'].duplicated(keep=False)
    n_duplicated = duplicated.sum()
    n_unique_duplicated_sites = df[duplicated]['SITE_IDENTIFIER'].nunique()

    print(f"\nDuplicate Analysis:")
    print(f"  Records with duplicate SITE_IDENTIFIER: {n_duplicated:,}")
    print(
        f"  Unique sites with multiple measurements: {n_unique_duplicated_sites:,}")

    if n_duplicated == 0:
        print("\n✓ No duplicates found! All sites have single measurements.")
        print(f"  Saving data to: {output_path}")
        df.to_csv(output_path, index=False)
        return df

    # Show example of duplicated sites
    print(f"\nExample duplicated sites (first 5):")
    duplicated_sites = df[duplicated]['SITE_IDENTIFIER'].unique()[:5]
    for site_id in duplicated_sites:
        site_data = df[df['SITE_IDENTIFIER'] == site_id][[
            'SITE_IDENTIFIER', 'VISIT_NUMBER', 'MEAS_YR', 'MEAS_DT']]
        print(f"\n  Site {site_id}:")
        print(site_data.to_string(index=False, max_rows=10))

    # Sort by SITE_IDENTIFIER, then by MEAS_YR (descending), then by VISIT_NUMBER (descending)
    # This ensures most recent measurement is first for each site
    print("\nDeduplicating...")
    df_sorted = df.sort_values(
        by=['SITE_IDENTIFIER', 'MEAS_YR', 'VISIT_NUMBER'],
        ascending=[True, False, False]
    )

    # Keep first occurrence (most recent) for each SITE_IDENTIFIER
    df_dedup = df_sorted.drop_duplicates(
        subset='SITE_IDENTIFIER', keep='first')

    print(f"\n✓ Deduplication complete!")
    print(f"  Original records: {len(df):,}")
    print(f"  Deduplicated records: {len(df_dedup):,}")
    print(f"  Records removed: {len(df) - len(df_dedup):,}")
    print(f"  Unique sites: {df_dedup['SITE_IDENTIFIER'].nunique():,}")

    # Verify no duplicates remain
    remaining_duplicates = df_dedup['SITE_IDENTIFIER'].duplicated().sum()
    if remaining_duplicates > 0:
        print(f"\n⚠ WARNING: {remaining_duplicates} duplicates still remain!")
    else:
        print("\n✓ Verified: No duplicate SITE_IDENTIFIERs in output")

    # Show year distribution
    print("\nMeasurement Year Distribution (after deduplication):")
    year_counts = df_dedup['MEAS_YR'].value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"  {year}: {count:,} sites")

    # Save deduplicated data
    print(f"\nSaving deduplicated data to: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_dedup.to_csv(output_path, index=False)
    print("✓ Saved successfully!")

    return df_dedup


def main():
    """Main execution."""
    # Paths
    input_path = 'data/raw/bc_sample_data-2025-10-09/bc_sample_data.csv'
    output_path = 'data/processed/bc_sample_data_deduplicated.csv'

    # Run deduplication
    df_dedup = deduplicate_sites(input_path, output_path)

    print("\n" + "="*80)
    print("DEDUPLICATION COMPLETE")
    print("="*80)
    print(f"\nNext steps:")
    print(f"  1. Update your analysis scripts to use:")
    print(f"     {output_path}")
    print(f"  2. Re-run the data exploration notebook")
    print(f"  3. Retrain models with deduplicated data")


if __name__ == "__main__":
    main()
