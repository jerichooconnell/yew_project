#!/usr/bin/env python3
"""
Analyze Image Review Results
=============================

Process the manual review results and filter the dataset based on quality checks.

Author: GitHub Copilot
Date: November 14, 2025
"""

import pandas as pd
import json
from pathlib import Path
import shutil


def load_reviews():
    """Load review results."""
    review_file = Path('data/ee_imagery/image_review_results.json')

    if not review_file.exists():
        print("✗ No review file found. Run review_inat_images.py first.")
        return None

    with open(review_file, 'r') as f:
        reviews = json.load(f)

    return reviews


def analyze_reviews(reviews, metadata_file):
    """Analyze review results and create filtered dataset."""

    # Load metadata
    metadata = pd.read_csv(metadata_file)

    # Add review status to metadata
    metadata['review_status'] = metadata['observation_id'].astype(
        str).map(reviews)
    metadata['review_status'] = metadata['review_status'].fillna(
        'not_reviewed')

    print("="*80)
    print("REVIEW ANALYSIS")
    print("="*80)
    print()

    # Overall stats
    total = len(metadata)
    reviewed = (metadata['review_status'] != 'not_reviewed').sum()
    good = (metadata['review_status'] == 'good').sum()
    bad = (metadata['review_status'] == 'bad').sum()
    uncertain = (metadata['review_status'] == 'uncertain').sum()
    not_reviewed = total - reviewed

    print(f"Total images: {total}")
    print(f"Reviewed: {reviewed} ({reviewed/total*100:.1f}%)")
    print(f"  Good: {good} ({good/total*100:.1f}%)")
    print(f"  Bad: {bad} ({bad/total*100:.1f}%)")
    print(f"  Uncertain: {uncertain} ({uncertain/total*100:.1f}%)")
    print(f"  Not reviewed: {not_reviewed} ({not_reviewed/total*100:.1f}%)")
    print()

    # Quality metrics for reviewed images
    if reviewed > 0:
        print("Quality metrics for reviewed images:")
        reviewed_df = metadata[metadata['review_status'] != 'not_reviewed']

        print(f"\nPositional accuracy (meters):")
        print(
            f"  Good images: {reviewed_df[reviewed_df['review_status']=='good']['positional_accuracy'].mean():.1f}m")
        print(
            f"  Bad images: {reviewed_df[reviewed_df['review_status']=='bad']['positional_accuracy'].mean():.1f}m")

        print(f"\nNumber of source images:")
        print(
            f"  Good images: {reviewed_df[reviewed_df['review_status']=='good']['num_source_images'].mean():.1f}")
        print(
            f"  Bad images: {reviewed_df[reviewed_df['review_status']=='bad']['num_source_images'].mean():.1f}")

    # Create filtered datasets
    print()
    print("="*80)
    print("FILTERED DATASETS")
    print("="*80)

    # Dataset 1: Only "good" images
    good_df = metadata[metadata['review_status'] == 'good']
    if len(good_df) > 0:
        output_file = 'data/ee_imagery/image_patches_64x64/inat_yew_filtered_good.csv'
        good_df.to_csv(output_file, index=False)
        print(f"\n✓ Good images only: {len(good_df)} images")
        print(f"  Saved to: {output_file}")

    # Dataset 2: Good + Uncertain (conservative approach)
    acceptable_df = metadata[metadata['review_status'].isin(
        ['good', 'uncertain'])]
    if len(acceptable_df) > 0:
        output_file = 'data/ee_imagery/image_patches_64x64/inat_yew_filtered_acceptable.csv'
        acceptable_df.to_csv(output_file, index=False)
        print(f"\n✓ Good + Uncertain: {len(acceptable_df)} images")
        print(f"  Saved to: {output_file}")

    # Dataset 3: Exclude only "bad" (include not_reviewed)
    not_bad_df = metadata[metadata['review_status'] != 'bad']
    if len(not_bad_df) > 0:
        output_file = 'data/ee_imagery/image_patches_64x64/inat_yew_filtered_not_bad.csv'
        not_bad_df.to_csv(output_file, index=False)
        print(f"\n✓ Excluding bad only: {len(not_bad_df)} images")
        print(f"  Saved to: {output_file}")

    # Common issues found
    print()
    print("="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print()

    if bad > 0:
        bad_df = metadata[metadata['review_status'] == 'bad']
        avg_accuracy_bad = bad_df['positional_accuracy'].mean()

        print(f"Issues identified in {bad} bad images:")
        print(f"  - Average GPS accuracy: {avg_accuracy_bad:.1f}m")

        if avg_accuracy_bad > 20:
            print(f"  → Consider lowering max accuracy threshold from 50m to 20m")

        # Check for patterns
        bad_years = bad_df['observation_year'].value_counts()
        if len(bad_years) > 0:
            print(f"  - Bad images by year: {bad_years.to_dict()}")

    print()
    print("Next steps:")
    print("  1. Use 'good' dataset for high-confidence training")
    print("  2. Use 'acceptable' dataset if you need more samples")
    print("  3. Re-run extraction with stricter filters if needed")
    print("  4. Consider extracting more observations if dataset is too small")


def main():
    """Main execution."""
    reviews = load_reviews()
    if reviews is None:
        return

    metadata_file = 'data/ee_imagery/image_patches_64x64/inat_yew_image_metadata.csv'
    analyze_reviews(reviews, metadata_file)


if __name__ == '__main__':
    main()
