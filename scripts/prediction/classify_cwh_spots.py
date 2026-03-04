#!/usr/bin/env python3
"""
Extract GEE embeddings for a 10km×10km area and classify with a saved model.
Supports multiple classifier types: MLP, Random Forest, kNN, Logistic Regression.
Generates an interactive HTML map for each area plus a summary comparison page.

Usage:
    python scripts/prediction/classify_cwh_spots.py --classifier rf_raw --force-reclassify
    python scripts/prediction/classify_cwh_spots.py --classifier logistic_raw
"""

import argparse
import base64
import io
import json
import pickle
import time
from datetime import datetime
from io import BytesIO
from math import ceil, cos, degrees, radians
from pathlib import Path

import ee
import folium
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio.features
import rasterio.transform as rtransform
import requests
import torch
import torch.nn as nn
import xgboost as xgb
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image as PILImage
from pyproj import Transformer as ProjTransformer
from sklearn.preprocessing import StandardScaler


# ── Constants ─────────────────────────────────────────────────────────────────
GEE_PROJECT       = 'carbon-storm-206002'
MODEL_DIR         = Path('results/predictions/south_vi_large')
OUTPUT_DIR        = Path('results/analysis/cwh_spot_comparisons')
SCALE_M           = 10          # 10 m/pixel
AREA_KM           = 10          # 10 km × 10 km
BAND_NAMES        = [f'A{i:02d}' for i in range(64)]
GEE_LIMIT_BYTES   = int(50_331_648 * 0.35)   # 35% safety margin (actual downloads run ~2× larger than probe)

# ── BC Vegetation Resource Inventory ─────────────────────────────────────────
VEG_COMP_GDB   = Path('data/VEG_COMP_LYR_R1_POLY_2024.gdb')
VEG_COMP_LAYER = 'VEG_COMP_LYR_R1_POLY'

# Logging-category RGBA colours matching the annotation_tool forestry overlay
# cat 1=water/non-forest, 2=logged<20yr, 3=logged 20-40yr, 4=logged 40-80yr,
# 5=forest 80-150yr, 6=alpine/barren, 7=old-growth >150yr
LOG_RGBA = {
    1: (30,  100, 220, 180),   # water / non-forest
    2: (220, 50,  50,  170),   # logged  <20 yr
    3: (230, 120, 30,  150),   # logged 20–40 yr
    4: (220, 200, 50,  110),   # logged 40–80 yr
    5: (100, 200, 100,  70),   # forest 80–150 yr
    6: (175, 155, 125, 160),   # alpine / barren (rock, rubble, non-treed veg)
    7: (20,  100,  40,  70),   # old-growth >150 yr
}
LOG_LABELS = {
    1: 'Water / non-forest',
    2: 'Logged  < 20 yr',
    3: 'Logged 20–40 yr',
    4: 'Logged 40–80 yr',
    5: 'Forest 80–150 yr',
    6: 'Alpine / barren',
    7: 'Old-growth > 150 yr',
}
# ── CSS hex equivalents (for HTML legend)
LOG_HEX = {
    1: '#1e64dc',
    2: '#dc3232',
    3: '#e6781e',
    4: '#dcc832',
    5: '#64c864',
    6: '#af9b7d',
    7: '#146428',
}

# PROJ_AGE_CLASS_CD_1 midpoint ages (years).  BC VRI age class codes:
#   1=1-20yr  2=21-40yr  3=41-60yr  4=61-80yr  5=81-100yr
#   6=101-120yr  7=121-140yr  8=141-250yr  9=251+yr
_AGE_CLASS_MIDPOINT = {
    '1': 10, '2': 30, '3': 50, '4': 70,
    '5': 90, '6': 110, '7': 130, '8': 195, '9': 300,
}

# ── 15 Study areas across the BC CWH zone ────────────────────────────────────
# (lat_centre, lon_centre, name, description)
STUDY_AREAS = [
    # Original 10
    (48.440, -124.160, "Carmanah-Walbran",      "South VI old-growth CWH"),
    (48.600, -123.800, "Sooke Hills",            "South VI montane CWH"),
    (49.315, -124.980, "Clayoquot Sound",        "West central VI CWH"),
    (50.020, -125.240, "Campbell River Uplands", "North-central VI CWH"),
    (50.700, -127.100, "Quatsino Sound",         "Northern VI CWH"),
    (49.700, -123.150, "Squamish Highlands",     "Coast Mountains south CWH"),
    (50.720, -124.000, "Desolation Sound",       "Sunshine Coast north CWH"),
    (52.330, -126.600, "Bella Coola Valley",     "Central coast CWH"),
    (54.150, -129.700, "Prince Rupert Hills",    "North coast CWH"),
    (53.500, -128.600, "Kitimat Ranges",         "Skeena CWH fringe"),
    # 5 new areas
    (49.900, -125.550, "Strathcona Highlands",   "Central VI CWH/MH boundary"),
    (49.860, -122.680, "Garibaldi Foothills",    "Mainland coast CWH near Whistler"),
    (50.830, -124.920, "Bute Inlet Slopes",      "Deep fjord CWH, Coast Mountains"),
    (49.020, -124.200, "Nanaimo Lakes",          "South VI mid-elevation CWH"),
    (51.400, -127.700, "Rivers Inlet",           "Central BC outer coast CWH"),
    # 20 additional coastal BC sites
    (48.550, -124.420, "Port Renfrew",           "SW VI Pacific Rim old-growth CWH"),
    (48.820, -124.050, "Cowichan Uplands",       "South VI lower-elevation CWH"),
    (49.620, -125.100, "Comox Uplands",          "Central VI mid-elevation CWH"),
    (49.780, -126.020, "Gold River Forest",      "West-central VI inner CWH"),
    (50.720, -127.500, "Port Hardy Forest",      "North VI CWH valley bottoms"),
    (49.400, -123.720, "Sunshine Coast South",   "Lower Sunshine Coast CWH"),
    (49.520, -123.420, "Howe Sound East",        "Howe Sound montane CWH"),
    (49.780, -124.550, "Powell River Forest",    "Upper Sunshine Coast CWH"),
    (50.100, -124.060, "Jervis Inlet Slopes",    "Jervis Inlet fjord CWH"),
    (51.020, -124.480, "Toba Inlet Slopes",      "Remote fjord CWH, northern Sunshine Coast"),
    (51.080, -125.680, "Knight Inlet",           "Deep fjord, mainland Coast Mountains CWH"),
    (50.760, -126.480, "Broughton Archipelago",  "Outer archipelago CWH"),
    (51.220, -126.020, "Kingcome Inlet",         "Remote fjord valley, inner CWH"),
    (51.640, -126.520, "Owikeno Lake",           "Rivers Inlet drainage, interior CWH"),
    (52.090, -126.840, "Burke Channel",          "Outer fjord near Bella Coola"),
    (52.380, -127.680, "Ocean Falls",            "Outer coast CWH, high precipitation"),
    (52.720, -126.560, "Dean Channel",           "Dean River CWH, coast-interior edge"),
    (52.900, -128.700, "Princess Royal Island",  "Outer island CWH, spirit bear range"),
    (52.510, -128.580, "Milbanke Sound",         "Outer mid-coast CWH"),
    (54.820, -130.120, "Portland Inlet",         "Far north coast CWH near Nisga'a"),
    # 10 gap-filling tiles (March 2026)
    (50.250, -125.750, "Muchalat Valley",        "Central VI CWHmm1/xm2 valley"),
    (49.250, -122.250, "Stave Lake",             "Lower Fraser Valley CWHvm1/dm"),
    (53.81907, -132.43530, "Haida Gwaii South",      "Moresby Island CWHvh3 outer coast"),
    (49.250, -121.750, "Chilliwack Uplands",     "Fraser Valley east CWHdm/ms1"),
    (55.250, -130.750, "Stewart Lowlands",       "Far north CWHvh3 near Stewart BC"),
    (51.250, -127.250, "Smith Sound",            "Mid-coast CWHvm2 mainland fjord"),
    (49.250, -125.250, "Alberni Valley",         "South-central VI CWHmm1/vm2"),
    (49.750, -123.750, "Sechelt Peninsula",      "Central Sunshine Coast CWHvm2/dm"),
    (52.750, -128.250, "Klemtu Forest",          "Mid-coast CWHvh2/vm2 inner islands"),
    (51.750, -127.750, "Namu Lowlands",          "Central coast CWHvh2 old-growth"),
]


# ── Colormap ──────────────────────────────────────────────────────────────────
# Full 0→1 probability range; near-zero pixels made transparent.
YEW_VMIN = 0.0
YEW_VMAX = 1.0
YEW_TRANSPARENT_BELOW = 0.02   # hide noise floor

def _make_yewcmap():
    return LinearSegmentedColormap.from_list(
        'yew',
        [
            (0.00, (0.20, 0.70, 0.20, 0.70)),   # low  – green
            (0.17, (0.45, 0.85, 0.05, 0.80)),   # lime
            (0.33, (1.00, 0.90, 0.00, 0.88)),   # yellow
            (0.50, (1.00, 0.60, 0.00, 0.90)),   # orange-yellow
            (0.67, (0.90, 0.40, 0.10, 0.93)),   # orange
            (0.83, (0.80, 0.15, 0.30, 0.95)),   # red-orange
            (1.00, (0.65, 0.00, 0.45, 0.96)),   # high – magenta
        ],
        N=256,
    )

YEWCMAP = _make_yewcmap()


# ── YewMLP (must match training definition exactly) ───────────────────────────
class YewMLP(nn.Module):
    def __init__(self, input_dim=64, hidden_dims=(128, 64, 32)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── GEE helpers ───────────────────────────────────────────────────────────────
def download_npy(image, region, scale, max_retries=4):
    for attempt in range(max_retries):
        try:
            url = image.getDownloadURL({
                'region':    region,
                'scale':     scale,
                'format':    'NPY',
                'crs':       'EPSG:4326',
            })
            resp = requests.get(url, timeout=300)
            if resp.status_code != 200:
                raise ValueError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            data = np.load(BytesIO(resp.content), allow_pickle=True)
            if data.dtype.names is not None:
                data = np.stack([data[n] for n in data.dtype.names], axis=-1)
            return data.astype(np.float32)
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"    Retry {attempt+1}/{max_retries} in {wait}s: {str(e)[:80]}")
                time.sleep(wait)
            else:
                raise


def download_embeddings_chunked(region, year, scale, cache_path):
    """Download 64-band satellite embedding for a region. Caches result."""
    if cache_path.exists():
        print(f"  ↩ Loaded from cache: {cache_path.name}")
        return np.load(cache_path)

    ee_emb = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
               .filterDate(f'{year}-01-01', f'{year+1}-01-01')
               .filterBounds(region)
               .mosaic()
               .select(BAND_NAMES)
               .toFloat())

    # Probe dimensions with first band
    print(f"  Probing dimensions...")
    first = download_npy(ee_emb.select(['A00']), region, scale)
    h, w = first.shape[0], first.shape[1]
    single_bytes = h * w * 4
    bands_per_chunk = max(1, int(GEE_LIMIT_BYTES // single_bytes))
    bands_per_chunk = min(bands_per_chunk, 64)
    print(f"  Tile {h}×{w} px, {single_bytes/1e6:.1f} MB/band → {bands_per_chunk} bands/chunk")

    all_data = [first[:, :, np.newaxis] if first.ndim == 2 else first.reshape(h, w, 1)]
    remaining = BAND_NAMES[1:]

    i = 0
    while i < len(remaining):
        chunk = remaining[i:i + bands_per_chunk]
        print(f"  Bands {chunk[0]}–{chunk[-1]} ({len(chunk)})...", end='', flush=True)
        t0 = time.time()
        try:
            data = download_npy(ee_emb.select(chunk), region, scale)
        except Exception as e:
            if 'request size' in str(e).lower() or 'too large' in str(e).lower() or 'must be less' in str(e).lower():
                bands_per_chunk = max(1, bands_per_chunk // 2)
                print(f" ⚠ Too large, retrying with {bands_per_chunk} bands/chunk")
                continue
            raise
        if data.ndim == 2:
            data = data[:, :, np.newaxis]
        all_data.append(data)
        print(f" ✓ {time.time()-t0:.1f}s")
        i += len(chunk)

    result = np.concatenate(all_data, axis=-1)   # (H, W, 64)
    np.save(cache_path, result)
    print(f"  ✓ Saved {result.shape} to {cache_path.name}")
    return result


def download_rgb(region, year, scale, cache_path):
    """Download RGB composite for the region."""
    if cache_path.exists():
        return np.load(cache_path)

    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(region)
          .filterDate(f'{year}-06-01', f'{year}-09-30')
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
          .median()
          .select(['B4', 'B3', 'B2'])
          .toFloat())

    data = download_npy(s2, region, scale)
    if data.ndim == 2:
        data = data[:, :, np.newaxis]

    for i in range(min(3, data.shape[-1])):
        band = data[:, :, i]
        valid = band > 0
        if valid.any():
            p2, p98 = np.percentile(band[valid], [2, 98])
            data[:, :, i] = np.clip((band - p2) / (p98 - p2 + 1e-8), 0, 1)

    np.save(cache_path, data)
    return data


# ── Bounding box from centre ──────────────────────────────────────────────────
def centre_to_bbox(lat, lon, km=10):
    """Return (south, north, west, east) for a km×km box centred at lat/lon."""
    half_lat = (km * 1000 / 2) / 111320.0
    half_lon = (km * 1000 / 2) / (111320.0 * cos(radians(lat)))
    return lat - half_lat, lat + half_lat, lon - half_lon, lon + half_lon


# ── Inference ─────────────────────────────────────────────────────────────────

# Classifier types:
#   mlp_scaled  — MLP with StandardScaler (original production method)
#   mlp_raw     — MLP on raw embeddings (no scaler)
#   rf_raw      — Random Forest on raw embeddings
#   knn3_raw    — k-Nearest Neighbors (k=3) on raw embeddings
#   logistic_raw — Logistic Regression on raw embeddings (linear probe)
SKLEARN_CLASSIFIERS = {'rf_raw', 'rf_raw_expanded', 'knn3_raw', 'logistic_raw'}
XGB_CLASSIFIERS     = {'xgb_raw_expanded'}
MLP_CLASSIFIERS     = {'mlp_scaled', 'mlp_raw', 'mlp_raw_expanded'}
ALL_CLASSIFIERS     = SKLEARN_CLASSIFIERS | XGB_CLASSIFIERS | MLP_CLASSIFIERS


def classify_grid(emb_array, classifier, scaler, device, batch_size=500_000,
                  classifier_type='mlp_scaled'):
    """
    Run inference on (H, W, 64) embedding array.
    Returns (H, W) probability grid.

    For sklearn classifiers: uses predict_proba on CPU (batched for memory).
    For MLP classifiers:     uses GPU inference with optional scaler.
    """
    h, w, _ = emb_array.shape
    flat = emb_array.reshape(-1, 64).astype(np.float32)
    flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)

    if classifier_type in SKLEARN_CLASSIFIERS:
        # sklearn classifiers work on CPU, predict_proba gives [P(0), P(1)]
        probs = np.zeros(len(flat), dtype=np.float32)
        for start in range(0, len(flat), batch_size):
            end = min(start + batch_size, len(flat))
            batch = flat[start:end]
            probs[start:end] = classifier.predict_proba(batch)[:, 1]
        return probs.reshape(h, w)

    elif classifier_type in XGB_CLASSIFIERS:
        # XGBoost - GPU-accelerated inference
        dmat = xgb.DMatrix(flat)
        probs = classifier.predict(dmat)
        return probs.reshape(h, w)

    elif classifier_type == 'mlp_scaled':
        # Original method: StandardScaler → MLP on GPU
        flat_s = scaler.transform(flat)
        probs = np.zeros(len(flat), dtype=np.float32)
        classifier.eval()
        with torch.no_grad():
            for start in range(0, len(flat_s), batch_size):
                batch = torch.tensor(flat_s[start:start + batch_size], device=device)
                logits = classifier(batch)
                probs[start:start + batch_size] = torch.sigmoid(logits).cpu().numpy()
        return probs.reshape(h, w)

    elif classifier_type in ('mlp_raw', 'mlp_raw_expanded'):
        # MLP on raw embeddings (no scaler)
        probs = np.zeros(len(flat), dtype=np.float32)
        classifier.eval()
        with torch.no_grad():
            for start in range(0, len(flat), batch_size):
                batch = torch.tensor(flat[start:start + batch_size], device=device)
                logits = classifier(batch)
                probs[start:start + batch_size] = torch.sigmoid(logits).cpu().numpy()
        return probs.reshape(h, w)

    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")


# ── Visualisation ─────────────────────────────────────────────────────────────
def grid_to_png_b64(grid, cmap, threshold=None, vmin=None, vmax=None):
    """Render probability grid to RGBA PNG (base64).  Values below *threshold*
    (default: YEW_TRANSPARENT_BELOW) are fully transparent."""
    if vmin is None:
        vmin = YEW_VMIN
    if vmax is None:
        vmax = YEW_VMAX
    if threshold is None:
        threshold = YEW_TRANSPARENT_BELOW
    normed = np.clip((grid - vmin) / (vmax - vmin), 0, 1)
    rgba = cmap(normed)
    rgba[grid < threshold] = [0, 0, 0, 0]
    img = PILImage.fromarray((rgba * 255).astype(np.uint8), mode='RGBA')
    buf = io.BytesIO()
    img.save(buf, format='PNG', compress_level=6)
    return base64.b64encode(buf.getvalue()).decode()


def rgb_to_png_b64(rgb):
    if rgb.shape[-1] >= 3:
        arr = (np.clip(rgb[:, :, :3], 0, 1) * 255).astype(np.uint8)
    else:
        arr = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    img = PILImage.fromarray(arr, mode='RGB')
    buf = io.BytesIO()
    img.save(buf, format='PNG', compress_level=6)
    return base64.b64encode(buf.getvalue()).decode()


# ── VRI logging-overlay helpers ───────────────────────────────────────────────

_CUR_YEAR_2D = datetime.now().year % 100
_CUR_CENTURY = (datetime.now().year // 100) * 100


def _decode_7b_year(code_2d):
    """Decode a 2-digit year from LINE_7B into a 4-digit calendar year."""
    y = int(code_2d)
    return _CUR_CENTURY + y if y <= _CUR_YEAR_2D else (_CUR_CENTURY - 100) + y


def _parse_7b_min_age(field_val, current_year):
    """
    Parse LINE_7B_DISTURBANCE_HISTORY (e.g. 'B23', 'L14;B23') and return the
    age (years since disturbance) of the MOST RECENT event, or None.
    Format: 1-letter code (L/B/W/I/D/…) + 2-digit year, semicolon-separated.
    All event types are treated equally: any stand-resetting event counts.
    """
    if not field_val:
        return None
    min_age = None
    for part in str(field_val).split(';'):
        part = part.strip()
        if len(part) >= 3 and part[0].isalpha() and part[1:3].isdigit():
            try:
                event_year = _decode_7b_year(part[1:3])
                age = current_year - event_year
                if age >= 0:
                    min_age = age if min_age is None else min(min_age, age)
            except Exception:
                pass
    return min_age


def _classify_vri_row(row, current_year):
    """Return 1-6 logging / land-cover category for a single VRI polygon row."""
    bclcs1 = str(row.get('BCLCS_LEVEL_1') or '').strip()
    bclcs2 = str(row.get('BCLCS_LEVEL_2') or '').strip()

    # Water (L1='W', or L1='N' with L2='W')
    if bclcs1 == 'W' or (bclcs1 == 'N' and bclcs2 == 'W'):
        return 1

    # Non-vegetated land: rock, rubble, exposed soil, alpine barren (L1='N', L2='L')
    if bclcs1 == 'N' and bclcs2 == 'L':
        return 6

    # Officially designated alpine zone (ALPINE_DESIGNATION='A')
    if str(row.get('ALPINE_DESIGNATION') or '').strip() == 'A':
        return 6

    # Collect candidate ages from every available source; take the minimum
    # (= most recent stand-resetting event).
    ages = []

    # 1. PROJ_AGE_1 — explicit stand age (58% polygon coverage).
    pa1 = row.get('PROJ_AGE_1')
    if pa1 is not None:
        try:
            ages.append(int(pa1))
        except (ValueError, TypeError):
            pass

    # 1b. PROJ_AGE_CLASS_CD_1 — age class code (92% coverage, much better than
    #     PROJ_AGE_1).  Convert to midpoint age so it slots into the same bins.
    pac = str(row.get('PROJ_AGE_CLASS_CD_1') or '').strip()
    if pac in _AGE_CLASS_MIDPOINT:
        ages.append(_AGE_CLASS_MIDPOINT[pac])

    # 2. LINE_7B_DISTURBANCE_HISTORY — most recent coded event (L=log, B=burn,
    #    W=wind, I=insect, D=disease, …); this captures post-survey disturbances
    #    (e.g. fires) that are NOT yet reflected in PROJ_AGE_1.
    dist7b = _parse_7b_min_age(row.get('LINE_7B_DISTURBANCE_HISTORY'), current_year)
    if dist7b is not None:
        ages.append(dist7b)

    # 3. HARVEST_DATE — explicit logging date (fallback for null PROJ_AGE_1)
    hdate = row.get('HARVEST_DATE')
    if hdate:
        try:
            hy = hdate.year if hasattr(hdate, 'year') else int(str(hdate)[:4])
            ages.append(current_year - hy)
        except Exception:
            pass

    # 4. OPENING_IND='Y' — aerially confirmed harvest via BC RESULTS system.
    #    If no harvest date was extracted yet, treat as recently disturbed
    #    (conservative: assume <20 yr, i.e. age=0 placeholder = cat2).
    if str(row.get('OPENING_IND') or '').strip() == 'Y' and not ages:
        ages.append(0)

    # 5. OPENING_SOURCE in {3,4,7,11} — polygon is a logged cutblock opening
    #    (3=BC RESULTS cutblock, 4=mapped clearing, 7=non-RESULTS cutblock,
    #    11=silviculture mapping).  If no age has been established from other
    #    sources, flag as recently disturbed so it doesn't pass as old forest.
    opening_src = row.get('OPENING_SOURCE')
    try:
        opening_src_int = int(opening_src) if opening_src is not None else None
    except (ValueError, TypeError):
        opening_src_int = None
    if opening_src_int in {3, 4, 7, 11} and not ages:
        ages.append(0)

    if not ages:
        # No age info: non-treed vegetation without harvest history
        # = naturally non-forested (alpine meadow, subalpine shrub, etc.)
        if bclcs2 == 'N':
            return 6
        return 5   # unknown / assume old forest

    age = min(ages)   # most recent disturbance wins
    if age < 20:
        return 2
    if age < 40:
        return 3
    if age < 80:
        return 4
    if age < 150:
        return 5   # maturing forest 80–150 yr
    return 7       # old-growth >150 yr


def extract_logging_grid(bbox, grid_h, grid_w, cache_path=None):
    """
    Clip VEG_COMP_LYR_R1_POLY_2024.gdb to *bbox* (south, north, west, east in
    WGS84) and rasterise logging-status categories onto a grid_h × grid_w array.
    Returns np.uint8 array (0 = no data, 1-5 = LOG_RGBA categories) or None.
    """
    if cache_path and Path(cache_path).exists():
        return np.load(str(cache_path))

    if not VEG_COMP_GDB.exists():
        return None

    south, north, west, east = bbox
    current_year = datetime.now().year

    # Reproject WGS84 corners to EPSG:3005 for the GDB spatial filter
    t4326_3005 = ProjTransformer.from_crs('EPSG:4326', 'EPSG:3005', always_xy=True)
    x_min, y_min = t4326_3005.transform(west, south)
    x_max, y_max = t4326_3005.transform(east, north)

    print(f"    Reading VEG_COMP for bbox "
          f"({south:.3f}\u00b0N, {west:.3f}\u00b0E) \u2013 ({north:.3f}\u00b0N, {east:.3f}\u00b0E)...")
    try:
        gdf = gpd.read_file(
            str(VEG_COMP_GDB),
            bbox=(x_min, y_min, x_max, y_max),
            layer=VEG_COMP_LAYER,
            columns=['BCLCS_LEVEL_1', 'BCLCS_LEVEL_2', 'PROJ_AGE_1',
                     'PROJ_AGE_CLASS_CD_1', 'HARVEST_DATE',
                     'LINE_7B_DISTURBANCE_HISTORY', 'OPENING_IND',
                     'OPENING_SOURCE', 'ALPINE_DESIGNATION', 'geometry'],
        )
    except Exception as e:
        print(f"    \u26a0  VEG_COMP read failed: {e}")
        return None

    if gdf.empty:
        print(f"    No VEG_COMP polygons in bbox \u2014 returning empty raster")
        return np.zeros((grid_h, grid_w), dtype=np.uint8)

    print(f"    {len(gdf):,} polygons \u2014 reprojecting to WGS84 & classifying...")
    gdf = gdf.to_crs('EPSG:4326')
    gdf['cat'] = [_classify_vri_row(row, current_year) for _, row in gdf.iterrows()]

    # Rasterise in WGS84 space to align with the probability grid
    transform = rtransform.from_bounds(west, south, east, north, grid_w, grid_h)
    shapes = [
        (geom, int(cat))
        for geom, cat in zip(gdf.geometry, gdf['cat'])
        if geom is not None and not geom.is_empty
    ]
    raster = rasterio.features.rasterize(
        shapes,
        out_shape=(grid_h, grid_w),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )

    if cache_path:
        np.save(str(cache_path), raster)
    print(f"    \u2713 Logging raster: {int((raster > 0).sum()):,} classified pixels")
    return raster


def logging_to_png_b64(log_grid):
    """Convert uint8 category raster to semi-transparent RGBA PNG (base64)."""
    h, w = log_grid.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for cat, color in LOG_RGBA.items():
        mask = log_grid == cat
        rgba[mask] = color
    img = PILImage.fromarray(rgba, 'RGBA')
    buf = BytesIO()
    img.save(buf, format='PNG', compress_level=6)
    return base64.b64encode(buf.getvalue()).decode()


# Suppression factors by VRI logging category
# cat 0 = no data (unchanged), 1 = water/non-forest, 2 = logged <20 yr,
# 3 = logged 20–40 yr, 4 = logged 40–80 yr, 5 = forest 80–150 yr,
# 6 = alpine / barren, 7 = old-growth >150 yr
LOG_SUPPRESS = {
    1: 0.00,   # water / non-forest → zero out completely
    2: 0.00,   # logged  <20 yr     → zero out (bare ground, no yew)
    3: 0.00,   # logged 20–40 yr    → zero out (young second-growth, yew absent)
    4: 0.50,   # logged 40–80 yr    → moderately suppressed
    5: 0.35,   # forest 80–150 yr   → partial suppression (maturing second-growth)
    6: 0.00,   # alpine / barren    → zero out (trees don't grow here)
    7: 1.00,   # old-growth >150 yr → unchanged
}


def apply_logging_mask(grid, log_grid):
    """
    Return a copy of *grid* with probabilities suppressed according to VRI
    logging categories.  Category-0 pixels (outside VRI coverage) are
    left unchanged.
    """
    masked = grid.copy()
    for cat, factor in LOG_SUPPRESS.items():
        where = log_grid == cat
        masked[where] *= factor
    return masked


# ── Per-area interactive map ──────────────────────────────────────────────────

def make_area_map(area_info, output_html, log_b64=None):
    lat, lon, name, desc, grid, rgb, bbox, stats = area_info
    south, north, west, east = bbox

    m = folium.Map(location=[lat, lon], zoom_start=12, control_scale=True,
                   tiles='OpenStreetMap')

    folium.TileLayer(
        'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='Satellite', overlay=False, control=True,
    ).add_to(m)
    folium.TileLayer(
        'https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='Terrain', overlay=False, control=True,
    ).add_to(m)

    # RGB base overlay
    if rgb is not None:
        rgb_b64 = rgb_to_png_b64(rgb)
        folium.raster_layers.ImageOverlay(
            image=f'data:image/png;base64,{rgb_b64}',
            bounds=[[south, west], [north, east]],
            origin='upper', opacity=0.9, name='RGB (Sentinel-2)',
            show=False, zindex=1,
        ).add_to(m)

    # Yew probability overlay
    yew_b64 = grid_to_png_b64(grid, YEWCMAP)
    folium.raster_layers.ImageOverlay(
        image=f'data:image/png;base64,{yew_b64}',
        bounds=[[south, west], [north, east]],
        origin='upper', opacity=1.0, name='Yew Probability',
        show=True, zindex=2,
    ).add_to(m)

    # Logging / forestry overlay (hidden by default, toggled via LayerControl)
    if log_b64 is not None:
        folium.raster_layers.ImageOverlay(
            image=f'data:image/png;base64,{log_b64}',
            bounds=[[south, west], [north, east]],
            origin='upper', opacity=0.85, name='🌳 Logging / Forestry',
            show=False, zindex=3,
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # ── Yew probability legend ─────────────────────────────────────────────
    p50_ha = stats['p50_px'] * SCALE_M * SCALE_M / 1e4
    log_legend_rows = ''
    if log_b64 is not None:
        log_legend_rows = '''
      <hr style="margin:7px 0">
      <b style="font-size:12px;">🌳 Logging / Forestry</b>
      <div style="margin:3px 0 1px">
        <span style="background:#1e64dc;padding:1px 8px;opacity:0.7;">▇</span> Water / non-forest</div>
      <div style="margin:1px 0">
        <span style="background:#dc3232;padding:1px 8px;opacity:0.7;">▇</span> Logged  &lt; 20 yr</div>
      <div style="margin:1px 0">
        <span style="background:#e6781e;padding:1px 8px;opacity:0.7;">▇</span> Logged 20–40 yr</div>
      <div style="margin:1px 0">
        <span style="background:#dcc832;padding:1px 8px;opacity:0.7;">▇</span> Logged 40–80 yr</div>
      <div style="margin:1px 0">
        <span style="background:#64c864;padding:1px 8px;opacity:0.7;">▇</span> Forest  > 80 yr</div>
      <div style="margin:1px 0">
        <span style="background:#af9b7d;padding:1px 8px;opacity:0.7;">▇</span> Alpine / barren</div>'''
    legend = f'''
    <div style="position:fixed;bottom:20px;right:10px;width:240px;
                background:#ffffffee;border:2px solid #555;z-index:9999;
                font-size:12px;padding:10px;border-radius:6px;">
      <b style="font-size:13px;">Yew Probability</b>
      <div style="font-size:10px;color:#777;margin:2px 0 4px">(range 0.70 – 1.00)</div>
      <div style="margin:2px 0">
        <span style="background:#33b233;padding:1px 8px;">▇</span> 0.70–0.75</div>
      <div style="margin:2px 0">
        <span style="background:#73d90d;padding:1px 8px;">▇</span> 0.75–0.80</div>
      <div style="margin:2px 0">
        <span style="background:#ffe600;padding:1px 8px;">▇</span> 0.80–0.85</div>
      <div style="margin:2px 0">
        <span style="background:#ff9900;padding:1px 8px;">▇</span> 0.85–0.90</div>
      <div style="margin:2px 0">
        <span style="background:#e66619;padding:1px 8px;">▇</span> 0.90–0.95</div>
      <div style="margin:2px 0">
        <span style="background:#a6004d;padding:1px 8px;">▇</span> 0.95–1.00</div>
      <hr style="margin:7px 0">
      <span style="font-size:11px;color:#444;">
        Grid: {stats["h"]}×{stats["w"]} px<br>
        Mean prob: {stats["mean"]:.3f}<br>
        Median prob: {stats["median"]:.3f}<br>
        Max prob: {stats["max"]:.3f}<br>
        P≥0.3: {stats["p30_px"]:,} px ({stats["p30_px"]*SCALE_M*SCALE_M/1e4:.1f} ha)<br>
        P≥0.5: {stats["p50_px"]:,} px ({p50_ha:.1f} ha)
      </span>
      {log_legend_rows}
    </div>'''
    m.get_root().html.add_child(folium.Element(legend))

    title = f'''
    <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
                width:460px;background:#ffffffee;border:2px solid #555;
                z-index:9999;font-size:14px;padding:8px;text-align:center;
                border-radius:6px;">
      <b>{name}</b><br>
      <span style="font-size:11px;color:#555;">{desc} · 10 km×10 km · FAIB-negatives model</span>
    </div>'''
    m.get_root().html.add_child(folium.Element(title))

    # ── Annotation panel (click to add positive / negative samples) ──────────
    slug_js = name.replace('"', '').replace("'", '')
    annotation_html = f'''
    <div id="anno-panel" style="position:fixed;top:80px;left:10px;width:270px;
         background:#fffffff0;border:2px solid #2d5a27;z-index:9999;
         font-size:12px;padding:10px;border-radius:8px;box-shadow:2px 2px 6px #0004;">
      <b style="font-size:13px;color:#2d5a27;">🌿 Field Sampling</b>
      <div style="margin:8px 0 4px;display:flex;gap:6px;">
        <button id="btn-pos" onclick="setMode('positive')"
          style="flex:1;padding:5px;background:#e8f5e9;border:2px solid #4caf50;
                 border-radius:4px;cursor:pointer;font-weight:bold;color:#1b5e20;">
          ✔ Yew Present
        </button>
        <button id="btn-neg" onclick="setMode('negative')"
          style="flex:1;padding:5px;background:#ffebee;border:2px solid #ef5350;
                 border-radius:4px;cursor:pointer;font-weight:bold;color:#b71c1c;">
          ✘ Yew Absent
        </button>
      </div>
      <div id="anno-mode" style="font-size:11px;color:#666;margin:2px 0 6px;">Click a button then click the map</div>
      <div style="max-height:180px;overflow-y:auto;border:1px solid #ddd;border-radius:4px;">
        <table id="anno-table" style="width:100%;border-collapse:collapse;font-size:11px;">
          <thead style="background:#f5f5f5;position:sticky;top:0;">
            <tr><th style="padding:3px 5px;text-align:left;">Type</th>
                <th style="padding:3px 5px;">Lat</th>
                <th style="padding:3px 5px;">Lon</th>
                <th style="padding:3px 5px;"></th></tr>
          </thead>
          <tbody id="anno-tbody"></tbody>
        </table>
      </div>
      <div style="margin-top:6px;display:flex;gap:5px;">
        <button onclick="clearSamples()"
          style="flex:1;padding:4px;background:#f5f5f5;border:1px solid #bbb;
                 border-radius:4px;cursor:pointer;font-size:11px;">Clear All</button>
        <button onclick="downloadCSV()"
          style="flex:1;padding:4px;background:#2d5a27;color:white;
                 border:none;border-radius:4px;cursor:pointer;font-size:11px;
                 font-weight:bold;">⬇ Download CSV</button>
      </div>
    </div>

    <script>
      var annoMode = null;
      var annoSamples = [];
      var annoMarkers = [];
      var areaName = "{slug_js}";

      function setMode(mode) {{
        if (annoMode === mode) {{
          annoMode = null;
          document.getElementById('btn-pos').style.boxShadow = '';
          document.getElementById('btn-neg').style.boxShadow = '';
          document.getElementById('anno-mode').innerText = 'Click a button then click the map';
          document.getElementById('anno-mode').style.color = '#666';
        }} else {{
          annoMode = mode;
          if (mode === 'positive') {{
            document.getElementById('btn-pos').style.boxShadow = '0 0 0 3px #4caf50';
            document.getElementById('btn-neg').style.boxShadow = '';
            document.getElementById('anno-mode').innerText = '▶ Click map to mark YEW PRESENT';
            document.getElementById('anno-mode').style.color = '#1b5e20';
          }} else {{
            document.getElementById('btn-neg').style.boxShadow = '0 0 0 3px #ef5350';
            document.getElementById('btn-pos').style.boxShadow = '';
            document.getElementById('anno-mode').innerText = '▶ Click map to mark YEW ABSENT';
            document.getElementById('anno-mode').style.color = '#b71c1c';
          }}
        }}
      }}

      function refreshTable() {{
        var tbody = document.getElementById('anno-tbody');
        tbody.innerHTML = '';
        annoSamples.forEach(function(s, i) {{
          var color = s.cls === 'positive' ? '#1b5e20' : '#b71c1c';
          var label = s.cls === 'positive' ? '✔ Yew' : '✘ Absent';
          tbody.innerHTML += '<tr style="border-bottom:1px solid #eee;">' +
            '<td style="padding:2px 5px;color:' + color + ';font-weight:bold;">' + label + '</td>' +
            '<td style="padding:2px 4px;text-align:right;">' + s.lat.toFixed(5) + '</td>' +
            '<td style="padding:2px 4px;text-align:right;">' + s.lon.toFixed(5) + '</td>' +
            '<td style="padding:2px 4px;"><span onclick="removeSample(' + i + ')" ' +
            'style="cursor:pointer;color:#999;font-size:14px;" title="Remove">✕</span></td>' +
            '</tr>';
        }});
      }}

      function removeSample(i) {{
        if (annoMarkers[i]) {{ annoMarkers[i].remove(); }}
        annoSamples.splice(i, 1);
        annoMarkers.splice(i, 1);
        refreshTable();
      }}

      function clearSamples() {{
        annoMarkers.forEach(function(m) {{ m.remove(); }});
        annoSamples = [];
        annoMarkers = [];
        refreshTable();
      }}

      function downloadCSV() {{
        if (annoSamples.length === 0) {{ alert('No samples to download.'); return; }}
        var lines = ['has_yew,lat,lon,area'];
        annoSamples.forEach(function(s) {{
          lines.push((s.cls === 'positive' ? '1' : '0') + ',' + s.lat.toFixed(6) + ',' + s.lon.toFixed(6) + ',' + areaName);
        }});
        var blob = new Blob([lines.join('\\n')], {{type: 'text/csv'}});
        var a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = areaName.replace(/ /g, '_') + '_annotations.csv';
        a.click();
      }}

      // Attach map click handler after map is ready
      document.addEventListener('DOMContentLoaded', function() {{
        setTimeout(function() {{
          var mapEl = document.querySelector('.folium-map') ||
                      document.querySelector('#map') ||
                      document.querySelector('[id^="map_"]');
          if (!mapEl) return;
          var mapId = mapEl.id;
          var leafletMap = window[mapId];
          if (!leafletMap) {{
            // Try iterating over window keys
            for (var key in window) {{
              if (window[key] && window[key]._leaflet_id !== undefined && typeof window[key].on === 'function') {{
                leafletMap = window[key];
                break;
              }}
            }}
          }}
          if (!leafletMap) return;

          leafletMap.on('click', function(e) {{
            if (!annoMode) return;
            var lat = e.latlng.lat;
            var lon = e.latlng.lng;
            var color = annoMode === 'positive' ? '#4caf50' : '#ef5350';
            var label = annoMode === 'positive' ? '✔ Yew Present' : '✘ Yew Absent';
            var marker = L.circleMarker([lat, lon], {{
              radius: 8,
              color: color,
              fillColor: color,
              fillOpacity: 0.85,
              weight: 2,
            }}).addTo(leafletMap);
            marker.bindPopup('<b>' + label + '</b><br>Lat: ' + lat.toFixed(5) + '<br>Lon: ' + lon.toFixed(5));
            annoSamples.push({{cls: annoMode, lat: lat, lon: lon}});
            annoMarkers.push(marker);
            refreshTable();
          }});
        }}, 600);
      }});
    </script>
    '''
    m.get_root().html.add_child(folium.Element(annotation_html))

    m.save(str(output_html))


# ── Summary comparison page ───────────────────────────────────────────────────
def make_summary_page(all_stats, output_html):
    rows = ''
    for s in sorted(all_stats, key=lambda x: -x['p50_px']):
        p50_ha = s['p50_px'] * SCALE_M * SCALE_M / 1e4
        p30_ha = s['p30_px'] * SCALE_M * SCALE_M / 1e4
        bar_pct = min(100, int(s['mean'] * 400))   # scale mean for bar width
        rows += f'''
        <tr>
          <td><a href="{s["html_file"]}">{s["name"]}</a></td>
          <td style="font-size:11px;color:#666;">{s["desc"]}</td>
          <td style="text-align:right;">{s["mean"]:.3f}</td>
          <td style="text-align:right;">{s["median"]:.4f}</td>
          <td style="text-align:right;">{s["max"]:.3f}</td>
          <td style="text-align:right;">{p30_ha:.1f}</td>
          <td style="text-align:right;">{p50_ha:.1f}</td>
          <td>
            <div style="background:#e0e0e0;width:120px;height:12px;display:inline-block;border-radius:3px;">
              <div style="background:#33b233;width:{bar_pct}px;height:12px;border-radius:3px;"></div>
            </div>
          </td>
        </tr>'''

    # Probability distribution thumbnail grid
    thumb_cells = ''
    for s in all_stats:
        p50_ha = s['p50_px'] * SCALE_M * SCALE_M / 1e4
        thumb_cells += f'''
        <div style="display:inline-block;margin:8px;text-align:center;vertical-align:top;width:200px;">
          <a href="{s["html_file"]}">
            <img src="{s["thumb_file"]}" width="200" height="200"
                 style="border:2px solid #ccc;border-radius:4px;"
                 alt="{s["name"]}">
          </a>
          <div style="font-size:12px;font-weight:bold;margin-top:4px;">{s["name"]}</div>
          <div style="font-size:11px;color:#666;">P≥0.5: {p50_ha:.1f} ha · max={s["max"]:.2f}</div>
        </div>'''

    html = f'''<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>CWH Yew Probability — 15 Area Comparison</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; }}
    h1 {{ color: #2d5a27; }}
    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
    th {{ background: #2d5a27; color: white; padding: 8px; text-align: left; }}
    td {{ padding: 6px 8px; border-bottom: 1px solid #ddd; }}
    tr:hover {{ background: #f5f5f5; }}
    a {{ color: #2d5a27; }}
    .note {{ font-size: 12px; color: #666; margin: 10px 0; }}
  </style>
</head>
<body>
  <h1>Pacific Yew Habitat Probability — CWH Zone Spot Comparison</h1>
  <p class="note">
    15 areas × 10 km² sampled across the Coastal Western Hemlock (CWH) biogeoclimatic zone of BC.<br>
    Model: FAIB-negatives MLP (acc=95.8%, F1=0.854, AUC=0.985) · 10 m GEE embeddings · 2024.<br>
    Click on an area name or thumbnail to open its interactive map.
  </p>

  <h2>Probability Maps</h2>
  <div>{thumb_cells}</div>

  <h2>Summary Statistics</h2>
  <table>
    <thead>
      <tr>
        <th>Area</th><th>Description</th>
        <th>Mean P</th><th>Median P</th><th>Max P</th>
        <th>P≥0.3 (ha)</th><th>P≥0.5 (ha)</th><th>Mean probability</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>

  <p class="note">Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}.</p>
</body>
</html>'''

    output_html.write_text(html)
    print(f"✓ Summary page: {output_html}")


# ── Thumbnail ─────────────────────────────────────────────────────────────────
def save_thumbnail(grid, path, size=200):
    normed = np.clip((grid - YEW_VMIN) / (YEW_VMAX - YEW_VMIN), 0, 1)
    rgba = YEWCMAP(normed)
    rgba[grid < YEW_TRANSPARENT_BELOW] = [0.85, 0.85, 0.85, 1.0]   # grey for near-zero
    img = PILImage.fromarray((rgba * 255).astype(np.uint8), mode='RGBA')
    img = img.resize((size, size), PILImage.LANCZOS)
    img.save(str(path), format='PNG')


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year',             type=int,  default=2024)
    parser.add_argument('--device',           type=str,  default='auto')
    parser.add_argument('--reuse-cache',      action='store_true', default=True)
    parser.add_argument('--force-reclassify', action='store_true',
                        help='Delete classification grid caches and re-classify with the current model')
    parser.add_argument('--skip-logging',     action='store_true',
                        help='Skip reading VEG_COMP GDB (faster, no logging overlay)')
    parser.add_argument('--classifier',       type=str, default='mlp_scaled',
                        choices=sorted(ALL_CLASSIFIERS),
                        help='Classifier type (default: mlp_scaled)')
    parser.add_argument('--tiles',            type=str, default=None,
                        help='Comma-separated list of tile indices (1-based) or name substrings to process (default: all)')
    args = parser.parse_args()

    classifier_type = args.classifier

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = OUTPUT_DIR / 'tile_cache'
    cache_dir.mkdir(exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Classifier: {classifier_type}")
    print(f"Vis range: VMIN={YEW_VMIN}, VMAX={YEW_VMAX}")

    # ── Load classifier ───────────────────────────────────────────────────
    print(f"\nLoading classifier from {MODEL_DIR}...")
    scaler = None

    CLASSIFIER_FILES = {
        'mlp_scaled':       ('mlp_scaled_model.pth', 'mlp_scaler.pkl'),
        'mlp_raw':          ('mlp_raw_model.pth',    None),
        'mlp_raw_expanded': ('mlp_raw_model_expanded.pth', None),
        'rf_raw':           ('rf_raw_model.pkl',      None),
        'rf_raw_expanded':  ('rf_raw_model_expanded.pkl', None),
        'xgb_raw_expanded': ('xgb_raw_model_expanded.json', None),
        'knn3_raw':         ('knn3_raw_model.pkl',    None),
        'logistic_raw':     ('logistic_raw_model.pkl', None),
    }
    model_file, scaler_file = CLASSIFIER_FILES[classifier_type]
    model_path = MODEL_DIR / model_file

    if not model_path.exists():
        # Fallback: try the original MLP model for backwards compatibility
        if classifier_type == 'mlp_scaled':
            model_path = MODEL_DIR / 'mlp_model.pth'
            scaler_file = 'mlp_scaler.pkl'

    if not model_path.exists():
        print(f"  ✗ Model file not found: {model_path}")
        print(f"  Run: python scripts/training/compare_classifiers.py")
        return

    if classifier_type in MLP_CLASSIFIERS:
        classifier = YewMLP()
        classifier.load_state_dict(torch.load(model_path, map_location=device))
        classifier = classifier.to(device)
        classifier.eval()
        if scaler_file:
            with open(MODEL_DIR / scaler_file, 'rb') as f:
                scaler = pickle.load(f)
        print(f"  ✓ MLP loaded ({model_path.name})"
              + (f" + {scaler_file}" if scaler_file else ""))
    elif classifier_type in XGB_CLASSIFIERS:
        classifier = xgb.Booster()
        classifier.load_model(str(model_path))
        # Use GPU inference if available
        xgb_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        classifier.set_param({'device': xgb_device})
        print(f"  ✓ XGBoost model loaded ({model_path.name}), device={xgb_device}")
    else:
        with open(model_path, 'rb') as f:
            classifier = pickle.load(f)
        print(f"  ✓ sklearn classifier loaded ({model_path.name})")

    # ── GEE init ──────────────────────────────────────────────────────────
    print("\nInitialising Google Earth Engine...")
    ee.Initialize(project=GEE_PROJECT)
    print("  ✓ GEE ready")

    # ── Process each area ─────────────────────────────────────────────────
    all_stats = []

    # Filter tiles if --tiles was specified
    if args.tiles:
        tokens = [t.strip() for t in args.tiles.split(',')]
        selected_indices = set()
        for tok in tokens:
            if tok.isdigit():
                idx = int(tok) - 1  # 1-based → 0-based
                if 0 <= idx < len(STUDY_AREAS):
                    selected_indices.add(idx)
            else:
                # Match by name substring (case-insensitive)
                for j, (_, _, nm, _) in enumerate(STUDY_AREAS):
                    if tok.lower() in nm.lower():
                        selected_indices.add(j)
        areas_to_process = [(i, STUDY_AREAS[i]) for i in sorted(selected_indices)]
        print(f"\nProcessing {len(areas_to_process)} of {len(STUDY_AREAS)} tiles")
    else:
        areas_to_process = list(enumerate(STUDY_AREAS))

    for i, (lat, lon, name, desc) in areas_to_process:
        slug  = name.lower().replace(' ', '_').replace('-', '_')
        print(f"\n{'='*68}")
        print(f"[{i+1}/{len(STUDY_AREAS)}] {name}  ({lat:.3f}, {lon:.3f})")
        print(f"  {desc}")
        print(f"{'='*68}")

        south, north, west, east = centre_to_bbox(lat, lon, km=AREA_KM)
        region = ee.Geometry.Rectangle([west, south, east, north])

        emb_cache  = cache_dir / f'{slug}_emb.npy'
        rgb_cache  = cache_dir / f'{slug}_rgb.npy'
        grid_cache = cache_dir / f'{slug}_grid.npy'
        log_cache  = cache_dir / f'{slug}_logging.npy'
        html_path  = OUTPUT_DIR / f'{slug}.html'
        thumb_path = OUTPUT_DIR / f'{slug}_thumb.png'

        # ── Force reclassify: clear grid cache so it re-runs inference ────
        if args.force_reclassify and grid_cache.exists():
            grid_cache.unlink()
            print("  ↺ Grid cache deleted (force-reclassify)")

        # ── Download embeddings ───────────────────────────────────────────
        print("  Downloading embeddings...")
        try:
            emb = download_embeddings_chunked(region, args.year, SCALE_M, emb_cache)
            h, w, _ = emb.shape
            print(f"  Embedding shape: {h}×{w}×64")
        except Exception as e:
            print(f"  ✗ Embedding download failed: {e}")
            continue

        # ── Download RGB ──────────────────────────────────────────────────
        print("  Downloading RGB...")
        try:
            rgb = download_rgb(region, args.year, SCALE_M, rgb_cache)
        except Exception as e:
            print(f"  ⚠ RGB download failed (continuing): {e}")
            rgb = None

        # ── Classify ──────────────────────────────────────────────────────
        if grid_cache.exists():
            print("  ↩ Loaded grid from cache")
            grid = np.load(grid_cache)
        else:
            print(f"  Classifying ({classifier_type})...")
            t0 = time.time()
            grid = classify_grid(emb, classifier, scaler, device,
                                 classifier_type=classifier_type)
            np.save(grid_cache, grid)   # raw, un-masked — always kept as-is
            print(f"  ✓ Done in {time.time()-t0:.1f}s")

        # ── Logging / forestry overlay (extracted before stats) ───────────
        log_b64  = None
        log_grid = None
        if not args.skip_logging:
            bbox = (south, north, west, east)
            log_grid = extract_logging_grid(bbox, h, w, cache_path=log_cache)
            if log_grid is not None and log_grid.max() > 0:
                log_b64 = logging_to_png_b64(log_grid)

        # ── Apply logging suppression to probability grid ─────────────────
        # Raw grid stays on disk; display_grid is used for stats + maps.
        if log_grid is not None:
            display_grid = apply_logging_mask(grid, log_grid)
            zeroed = int((grid > 0.02).sum()) - int((display_grid > 0.02).sum())
            print(f"  Logging mask: {zeroed:,} pixels zeroed / suppressed")
        else:
            display_grid = grid

        # ── Statistics ────────────────────────────────────────────────────
        stats = {
            'name':      name,
            'desc':      desc,
            'lat':       lat,
            'lon':       lon,
            'h':         h,
            'w':         w,
            'mean':      float(display_grid.mean()),
            'median':    float(np.median(display_grid)),
            'std':       float(display_grid.std()),
            'max':       float(display_grid.max()),
            'p30_px':    int((display_grid >= 0.30).sum()),
            'p50_px':    int((display_grid >= 0.50).sum()),
            'p70_px':    int((display_grid >= 0.70).sum()),
            'html_file': html_path.name,
            'thumb_file': thumb_path.name,
        }
        print(f"  Stats: mean={stats['mean']:.3f} max={stats['max']:.3f} "
              f"P>=0.5={stats['p50_px']:,}px "
              f"({stats['p50_px']*SCALE_M*SCALE_M/1e4:.1f} ha)")

        # ── Thumbnail ─────────────────────────────────────────────────────
        save_thumbnail(display_grid, thumb_path)

        # ── Interactive map ───────────────────────────────────────────────
        print(f"  Building map → {html_path.name}...")
        area_info = (lat, lon, name, desc, display_grid, rgb,
                     (south, north, west, east), stats)
        make_area_map(area_info, html_path, log_b64=log_b64)
        print(f"  ✓ Map: {html_path.stat().st_size/1024:.0f} KB")

        all_stats.append(stats)

    # ── Summary page (merge with any previously-saved stats) ────────────
    # Load existing stats from a prior run, then update/add current results.
    stats_json = OUTPUT_DIR / 'spot_stats.json'
    merged = {}
    if stats_json.exists():
        try:
            with open(stats_json) as f:
                for s in json.load(f):
                    merged[s['name']] = s
        except Exception:
            pass
    # Overwrite with freshly-computed stats from this run
    for s in all_stats:
        merged[s['name']] = s
    merged_list = list(merged.values())

    # Persist full merged stats
    with open(stats_json, 'w') as f:
        json.dump(merged_list, f, indent=2)

    if merged_list:
        make_summary_page(merged_list, OUTPUT_DIR / 'index.html')

    print(f"\n{'='*68}")
    n_requested = len(areas_to_process)
    print(f"ALL DONE — {len(all_stats)}/{n_requested} areas processed")
    print(f"Index includes {len(merged_list)} total tiles")
    print(f"Each map has a Field Sampling panel - click Yew Present/Absent then the map")
    print(f"Open: file://{(OUTPUT_DIR / 'index.html').absolute()}")


if __name__ == '__main__':
    main()
