#!/usr/bin/env python3
"""
Classify center-pixel embeddings inside a geographic bbox and save overlay.

- Trains a final SVM (RBF) on combined train+val using center pixel features.
- Scans `embedding_dir` for files named `embedding_{lat:.6f}_{lon:.6f}.npy`.
- Filters to points inside bbox and predicts yew probability for each.
- Saves CSV of predictions and a scatter overlay PNG using an optional composite image.

Usage examples:
python scripts/prediction/classify_bbox_embeddings.py \
    --embedding-dir data/ee_imagery/embedding_patches_64x64 \
    --train-path data/processed/train_split_filtered.csv \
    --val-path data/processed/val_split_filtered.csv \
    --bbox 48.0 50.9 -125.9 -123.0 \
    --output-dir results/predictions/vancouver_island
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--embedding-dir',
                   default='data/ee_imagery/embedding_patches_64x64')
    p.add_argument(
        '--train-path', default='data/processed/train_split_filtered.csv')
    p.add_argument(
        '--val-path', default='data/processed/val_split_filtered.csv')
    p.add_argument('--bbox', nargs=4, type=float,
                   help='lat_min lat_max lon_min lon_max', default=None)
    p.add_argument('--pred-csv', type=str, default=None,
                   help='Optional predictions CSV to derive bbox')
    p.add_argument(
        '--output-dir', default='results/predictions/vancouver_island')
    p.add_argument('--limit', type=int, default=0,
                   help='Limit number of tiles processed (0 = all)')
    return p.parse_args()


def load_embeddings_in_dir(emb_dir):
    emb_dir = Path(emb_dir)
    files = list(emb_dir.glob('embedding_*.npy'))
    coords = []
    for f in files:
        name = f.stem  # embedding_{lat}_{lon}
        try:
            _, lat_str, lon_str = name.split('_')
            lat = float(lat_str)
            lon = float(lon_str)
            coords.append((f, lat, lon))
        except Exception:
            continue
    return coords


def extract_center_pixel_from_file(path, patch_size=64):
    try:
        arr = np.load(path)  # expected shape (64, H, W) or (C, H, W)
        c = patch_size // 2
        feat = arr[:, c, c]
        return feat
    except Exception:
        return None


def train_final_svm(train_csv, val_csv, emb_dir):
    # load train+val
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df = pd.concat([df_train, df_val], ignore_index=True)

    X = []
    y = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Extracting train/val features'):
        lat = row.get('lat') or row.get('latitude')
        lon = row.get('lon') or row.get('longitude')
        if pd.isna(lat) or pd.isna(lon):
            continue
        feat = extract_center_pixel_from_file(
            Path(emb_dir) / f'embedding_{lat:.6f}_{lon:.6f}.npy')
        if feat is None:
            continue
        X.append(feat)
        y.append(int(row['has_yew']))

    X = np.array(X)
    y = np.array(y)

    # handle nan/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = SVC(kernel='rbf', probability=True, random_state=42)
    clf.fit(Xs, y)

    return clf, scaler


def main():
    args = parse_args()
    embedding_dir = Path(args.embedding_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine bbox
    if args.bbox:
        lat_min, lat_max, lon_min, lon_max = args.bbox
    elif args.pred_csv:
        dfp = pd.read_csv(args.pred_csv)
        dfp = dfp[dfp['yew_probability'].notna()]
        lat_min, lat_max = dfp['latitude'].min(), dfp['latitude'].max()
        lon_min, lon_max = dfp['longitude'].min(), dfp['longitude'].max()
    else:
        # Default Vancouver Island bbox used in prior scripts
        lat_min, lat_max, lon_min, lon_max = 48.0, 50.9, -125.9, -123.0

    print(f'Using bbox: lat {lat_min}..{lat_max}, lon {lon_min}..{lon_max}')

    # Train final model on combined train+val
    print('Training final SVM on combined train+val (center pixel)...')
    clf, scaler = train_final_svm(
        args.train_path, args.val_path, embedding_dir)
    print('Model trained.')

    # Gather embeddings
    all_emb = load_embeddings_in_dir(embedding_dir)
    print(f'Found {len(all_emb)} embedding files in {embedding_dir}')

    results = []
    processed = 0
    limit = args.limit if args.limit > 0 else None

    for fpath, lat, lon in tqdm(all_emb, desc='Scanning embeddings'):
        if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
            continue
        feat = extract_center_pixel_from_file(fpath)
        if feat is None:
            continue
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        feat_s = scaler.transform(feat.reshape(1, -1))
        prob = float(clf.predict_proba(feat_s)[:, 1][0])
        results.append(
            {'latitude': lat, 'longitude': lon, 'yew_probability': prob})
        processed += 1
        if limit and processed >= limit:
            break

    df_out = pd.DataFrame(results).sort_values(
        'yew_probability', ascending=False)
    out_csv = out_dir / f'center_pixel_bbox_predictions.csv'
    df_out.to_csv(out_csv, index=False)
    print(f'Saved predictions: {out_csv} (n={len(df_out)})')

    # Visualization: scatter overlay on any available composite in out_dir
    composite_candidates = list(out_dir.glob('*.png'))
    composite_path = None
    if composite_candidates:
        # prefer composite image names
        for c in composite_candidates:
            if 'composite' in c.name or 'satellite' in c.name:
                composite_path = c
                break
        if not composite_path:
            composite_path = composite_candidates[0]

    # Plot scatter
    fig, ax = plt.subplots(figsize=(12, 10))
    if composite_path is not None:
        try:
            img = plt.imread(composite_path)
            ax.imshow(img, extent=[lon_min, lon_max,
                      lat_min, lat_max], aspect='auto')
            print(f'Overlaying on composite image: {composite_path.name}')
        except Exception:
            composite_path = None

    if len(df_out):
        sc = ax.scatter(df_out['longitude'], df_out['latitude'], c=df_out['yew_probability'],
                        cmap='RdYlBu_r', s=20, vmin=0, vmax=1, alpha=0.8)
        cbar = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label('Yew probability')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Center-pixel predicted yew probability (bbox)')
    out_png = out_dir / 'center_pixel_bbox_overlay.png'
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print(f'Saved overlay: {out_png}')


if __name__ == '__main__':
    main()
