#!/usr/bin/env python3
"""
Create train/val splits containing ONLY iNat yew positives.

Negatives will be supplied separately via --gee-negatives (FAIB tree inventory).
This ensures clean separation of data sources.

Output:
  data/processed/inat_yew_positives_train.csv  (80% of positives)
  data/processed/inat_yew_positives_val.csv    (20% of positives)
"""

import pandas as pd
from pathlib import Path


def main():
    src = Path('data/processed/inat_yew_filtered_good.csv')
    df = pd.read_csv(src)
    print(f"Loaded {len(df)} yew-positive records from {src}")
    assert (df['has_yew'] == True).all(), "Expected all has_yew=True"

    # 80/20 split, shuffled
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx]
    val = df.iloc[split_idx:]

    train_out = Path('data/processed/inat_yew_positives_train.csv')
    val_out = Path('data/processed/inat_yew_positives_val.csv')
    train.to_csv(train_out, index=False)
    val.to_csv(val_out, index=False)

    print(f"Train: {len(train)} yew positives -> {train_out}")
    print(f"Val:   {len(val)} yew positives  -> {val_out}")


if __name__ == '__main__':
    main()
