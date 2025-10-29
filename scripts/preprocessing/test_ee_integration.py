#!/usr/bin/env python3
"""
Test Earth Engine Data Integration
====================================

Tests the integration of extracted satellite imagery data with the forest
inventory data for model training.

This script:
1. Loads the 100-sample Earth Engine extraction
2. Merges it with the forest inventory data
3. Analyzes the combined dataset
4. Prepares features for model training
5. Tests a small training run

Author: Analysis Tool
Date: October 2025
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# File paths
EE_DATA_PATH = 'data/ee_imagery/temp_extraction_patch_100.csv'
FOREST_DATA_PATH = 'data/raw/bc_sample_data-2025-10-09/bc_sample_data.csv'
OUTPUT_DIR = Path('data/processed')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_and_merge_data():
    """
    Load Earth Engine data and forest inventory, then merge them.
    """
    print("="*70)
    print("Loading and Merging Data")
    print("="*70)

    # Load Earth Engine extraction
    print(f"\nLoading Earth Engine data from {EE_DATA_PATH}...")
    ee_df = pd.read_csv(EE_DATA_PATH)
    print(f"  Loaded {len(ee_df)} plots")
    print(f"  Success rate: {ee_df['success'].mean()*100:.1f}%")

    # Filter to successful extractions
    ee_df = ee_df[ee_df['success'] == True].copy()
    print(f"  After filtering: {len(ee_df)} successful plots")

    # Load forest inventory
    print(f"\nLoading forest inventory from {FOREST_DATA_PATH}...")
    forest_df = pd.read_csv(FOREST_DATA_PATH, low_memory=False)
    print(f"  Loaded {len(forest_df)} plots")

    # Merge on SITE_IDENTIFIER (plot_id in EE data)
    print("\nMerging datasets on SITE_IDENTIFIER...")
    merged_df = forest_df.merge(
        ee_df,
        left_on='SITE_IDENTIFIER',
        right_on='plot_id',
        how='inner'
    )
    print(f"  Merged dataset: {len(merged_df)} plots")

    # Display merge statistics
    print("\nMerge Statistics:")
    print(f"  EE data plots: {len(ee_df)}")
    print(f"  Forest data plots: {len(forest_df)}")
    print(f"  Merged plots: {len(merged_df)}")
    print(f"  Match rate: {len(merged_df)/len(ee_df)*100:.1f}%")

    return merged_df


def analyze_combined_data(df):
    """
    Analyze the combined dataset.
    """
    print("\n" + "="*70)
    print("Combined Dataset Analysis")
    print("="*70)

    # Basic statistics
    print(f"\nDataset size: {len(df)} plots")
    print(f"Columns: {len(df.columns)}")

    # Site status
    print("\nSite Status:")
    print(
        f"  Active sites: {df['is_active_site'].sum()} ({df['is_active_site'].mean()*100:.1f}%)")
    print(
        f"  Inactive sites: {(~df['is_active_site']).sum()} ({(~df['is_active_site']).mean()*100:.1f}%)")

    # BEC Zones
    print("\nBiogeoclimatic Zones (top 5):")
    print(df['BEC_ZONE'].value_counts().head())

    # Satellite data statistics
    print("\nSatellite Data Statistics:")
    print(f"  NDVI: {df['ndvi'].mean():.3f} ± {df['ndvi'].std():.3f}")
    print(f"  EVI: {df['evi'].mean():.3f} ± {df['evi'].std():.3f}")
    print(
        f"  Elevation: {df['elevation'].mean():.0f} ± {df['elevation'].std():.0f} m")
    print(f"  Slope: {df['slope'].mean():.1f} ± {df['slope'].std():.1f}°")

    # Forest metrics
    print("\nForest Metrics:")
    print(
        f"  Basal Area (live): {df['BA_HA_LS'].mean():.1f} ± {df['BA_HA_LS'].std():.1f} m²/ha")
    print(
        f"  Stems/ha (live): {df['STEMS_HA_LS'].mean():.0f} ± {df['STEMS_HA_LS'].std():.0f}")

    # Check for Pacific Yew
    # Assuming yew is in tree detail - we'll create a simplified target
    # For now, use presence in the dataset as a proxy
    print("\nTarget Variable (Yew presence):")
    if 'YEW_PRESENT' in df.columns:
        print(f"  Yew present: {df['YEW_PRESENT'].sum()}")
        print(f"  Yew absent: {(~df['YEW_PRESENT']).sum()}")
    else:
        print("  Note: Yew presence indicator not yet created")


def prepare_features(df):
    """
    Prepare features for model training.
    """
    print("\n" + "="*70)
    print("Feature Preparation")
    print("="*70)

    # Define feature columns

    # Numerical features from forest inventory
    numerical_forest = [
        'BA_HA_LS',      # Basal area (live stems)
        'STEMS_HA_LS',   # Stems per hectare (live)
        'VHA_WSV_LS',    # Volume per hectare (live)
    ]

    # Numerical features from Earth Engine
    numerical_ee = [
        'ndvi',
        'evi',
        'elevation',
        'slope',
        'aspect'
    ]

    # Categorical features
    categorical = [
        'BEC_ZONE',
        'SPC_LIVE_1',    # Primary species
    ]

    # Combine all numerical features
    numerical_features = numerical_forest + numerical_ee

    print(f"\nFeatures defined:")
    print(f"  Numerical (forest): {len(numerical_forest)}")
    print(f"  Numerical (satellite): {len(numerical_ee)}")
    print(f"  Categorical: {len(categorical)}")
    print(f"  Total: {len(numerical_features) + len(categorical)}")

    # Check for missing values
    print("\nMissing values:")
    for col in numerical_features + categorical:
        if col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                print(f"  {col}: {missing} ({missing/len(df)*100:.1f}%)")

    # Fill missing numerical values with median
    for col in numerical_features:
        if col in df.columns:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)

    # Fill missing categorical with 'UNKNOWN'
    for col in categorical:
        if col in df.columns:
            df[col].fillna('UNKNOWN', inplace=True)

    print("\n✓ Missing values handled")

    return df, numerical_features, categorical


def test_preprocessing(df, numerical_features, categorical):
    """
    Test preprocessing pipeline.
    """
    print("\n" + "="*70)
    print("Testing Preprocessing Pipeline")
    print("="*70)

    # Extract features
    X_numerical = df[numerical_features].values

    # Scale numerical features
    print("\nScaling numerical features...")
    scaler = StandardScaler()
    X_numerical_scaled = scaler.fit_transform(X_numerical)
    print(f"  Shape: {X_numerical_scaled.shape}")
    print(f"  Mean: {X_numerical_scaled.mean(axis=0)[:3]}... (first 3)")
    print(f"  Std: {X_numerical_scaled.std(axis=0)[:3]}... (first 3)")

    # Encode categorical features
    print("\nEncoding categorical features...")
    X_categorical = pd.DataFrame()
    encoders = {}

    for col in categorical:
        if col in df.columns:
            le = LabelEncoder()
            X_categorical[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            print(f"  {col}: {len(le.classes_)} unique values")

    print(f"  Categorical shape: {X_categorical.shape}")

    # Save preprocessors
    preprocessor_path = OUTPUT_DIR / 'ee_test_preprocessor.pkl'
    with open(preprocessor_path, 'wb') as f:
        pickle.dump({
            'scaler': scaler,
            'encoders': encoders,
            'numerical_features': numerical_features,
            'categorical_features': categorical
        }, f)
    print(f"\n✓ Saved preprocessors to {preprocessor_path}")

    return X_numerical_scaled, X_categorical, scaler, encoders


def create_summary_report(df, merged_df):
    """
    Create a summary report of the integration test.
    """
    print("\n" + "="*70)
    print("Creating Summary Report")
    print("="*70)

    report = []
    report.append("# Earth Engine Integration Test Report")
    report.append(f"\n**Date**: October 16, 2025")
    report.append(f"\n## Data Summary")
    report.append(f"\n- **EE Extraction file**: {EE_DATA_PATH}")
    report.append(f"- **Forest inventory**: {FOREST_DATA_PATH}")
    report.append(f"- **Successful EE extractions**: {len(df)}")
    report.append(f"- **Merged plots**: {len(merged_df)}")

    report.append(f"\n## Site Status")
    report.append(
        f"\n- **Active sites**: {merged_df['is_active_site'].sum()} ({merged_df['is_active_site'].mean()*100:.1f}%)")
    report.append(
        f"- **Inactive sites**: {(~merged_df['is_active_site']).sum()}")

    report.append(f"\n## Satellite Data Quality")
    report.append(f"\n| Metric | Mean | Std | Min | Max |")
    report.append(f"|--------|------|-----|-----|-----|")

    for col in ['ndvi', 'evi', 'elevation', 'slope']:
        if col in merged_df.columns:
            report.append(
                f"| {col.upper()} | {merged_df[col].mean():.3f} | "
                f"{merged_df[col].std():.3f} | {merged_df[col].min():.3f} | "
                f"{merged_df[col].max():.3f} |"
            )

    report.append(f"\n## BEC Zone Distribution")
    report.append(f"\n```")
    for zone, count in merged_df['BEC_ZONE'].value_counts().head(10).items():
        report.append(
            f"{zone}: {count} plots ({count/len(merged_df)*100:.1f}%)")
    report.append(f"```")

    report.append(f"\n## Next Steps")
    report.append(f"\n1. Create yew density target variable")
    report.append(f"2. Download image patches from URLs")
    report.append(f"3. Update model training script with EE features")
    report.append(f"4. Train and evaluate model")
    report.append(f"5. Compare performance with/without EE data")

    report_path = OUTPUT_DIR / 'ee_integration_test_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"✓ Saved report to {report_path}")


def main():
    """
    Main test pipeline.
    """
    print("\n" + "="*70)
    print("EARTH ENGINE INTEGRATION TEST")
    print("Testing with 100-sample extraction")
    print("="*70)

    # Step 1: Load and merge data
    merged_df = load_and_merge_data()

    # Step 2: Analyze combined data
    analyze_combined_data(merged_df)

    # Step 3: Prepare features
    merged_df, numerical_features, categorical = prepare_features(merged_df)

    # Step 4: Test preprocessing
    X_num, X_cat, scaler, encoders = test_preprocessing(
        merged_df, numerical_features, categorical
    )

    # Step 5: Save processed data
    output_path = OUTPUT_DIR / 'ee_test_data_100.csv'
    merged_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved processed data to {output_path}")

    # Step 6: Create summary report
    ee_df = pd.read_csv(EE_DATA_PATH)
    ee_df = ee_df[ee_df['success'] == True]
    create_summary_report(ee_df, merged_df)

    print("\n" + "="*70)
    print("TEST COMPLETE!")
    print("="*70)
    print(f"\nFiles created:")
    print(f"  - {output_path}")
    print(f"  - {OUTPUT_DIR / 'ee_test_preprocessor.pkl'}")
    print(f"  - {OUTPUT_DIR / 'ee_integration_test_report.md'}")
    print(f"\nNext: Review the report and update the training script")
    print("="*70)


if __name__ == "__main__":
    main()
