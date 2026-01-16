#!/usr/bin/env python3
"""
Large Area Yew Classification

Download a single large embedding image from Google Earth Engine and classify
every pixel with SVM. No tile stitching required!

Uses the high-accuracy StandardScaler+SVM approach from predict_center_pixel_map.py
(91.58% validation accuracy) combined with direct GEE embedding download.

Usage:
    python scripts/prediction/classify_large_area.py \
        --bbox 48.44 48.47 -124.11 -124.002 \
        --output-dir results/large_area \
        --year 2024 \
        --scale 10

Author: GitHub Copilot
Date: January 2026
"""

import argparse
import json
import tempfile
from datetime import datetime
from pathlib import Path

import ee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Classify yew presence in a large area using GEE embeddings and SVM',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--bbox', nargs=4, type=float, required=True,
                        metavar=('LAT_MIN', 'LAT_MAX', 'LON_MIN', 'LON_MAX'),
                        help='Bounding box: lat_min lat_max lon_min lon_max')
    parser.add_argument('--year', type=int, default=2024,
                        help='Year for embeddings (default: 2024)')
    parser.add_argument('--scale', type=int, default=10,
                        help='Pixel resolution in meters (default: 10)')
    parser.add_argument('--train-csv', type=str, 
                        default='data/processed/train_split_filtered.csv',
                        help='Path to training CSV')
    parser.add_argument('--val-csv', type=str,
                        default='data/processed/val_split_filtered.csv',
                        help='Path to validation CSV')
    parser.add_argument('--embedding-dir', type=str,
                        default='data/ee_imagery/embedding_patches_64x64',
                        help='Directory with training embedding patches')
    parser.add_argument('--output-dir', type=str,
                        default='results/predictions/large_area',
                        help='Output directory for results')
    parser.add_argument('--gee-project', type=str,
                        default='carbon-storm-206002',
                        help='Google Earth Engine project ID')
    parser.add_argument('--skip-rgb', action='store_true',
                        help='Skip downloading Sentinel-2 RGB')
    parser.add_argument('--batch-size', type=int, default=50000,
                        help='Pixels per batch for classification (default: 50000)')
    return parser.parse_args()


# =============================================================================
# GEE Download Functions
# =============================================================================

def download_embedding_chunk(embedding, region, scale, bands):
    """Download a single chunk of embedding data."""
    url = embedding.getDownloadURL({
        'region': region,
        'scale': scale,
        'format': 'NPY',
        'crs': 'EPSG:4326'
    })
    
    response = requests.get(url, timeout=300)
    if response.status_code != 200:
        raise ValueError(f"Download failed: {response.status_code}")
    
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name
    
    chunk_data = np.load(tmp_path)
    Path(tmp_path).unlink()
    
    # Handle structured array
    if chunk_data.dtype.names is not None:
        data_list = [chunk_data[band] for band in bands]
        chunk_data = np.stack(data_list, axis=-1)
    elif chunk_data.ndim == 3 and chunk_data.shape[0] == 64:
        chunk_data = np.transpose(chunk_data, (1, 2, 0))
    
    return chunk_data.astype(np.float32)


def compute_grid_params(lat_min, lat_max, lon_min, lon_max, scale, max_pixels_per_chunk):
    """
    Compute consistent grid parameters for chunked downloads.
    
    Returns a dict with all parameters needed for consistent downloads.
    """
    meters_per_deg_lat = 111000.0
    meters_per_deg_lon = 111000.0 * np.cos(np.radians((lat_min + lat_max) / 2.0))
    
    height_pixels = int((lat_max - lat_min) * meters_per_deg_lat / scale)
    width_pixels = int((lon_max - lon_min) * meters_per_deg_lon / scale)
    total_pixels = height_pixels * width_pixels
    
    n_chunks = max(1, int(np.ceil(total_pixels / max_pixels_per_chunk)))
    
    if n_chunks <= 1:
        n_row_chunks = 1
        n_col_chunks = 1
    else:
        aspect = width_pixels / height_pixels
        n_col_chunks = max(1, int(np.ceil(np.sqrt(n_chunks * aspect))))
        n_row_chunks = max(1, int(np.ceil(n_chunks / n_col_chunks)))
        
        pixels_per_chunk = (width_pixels / n_col_chunks) * (height_pixels / n_row_chunks)
        while pixels_per_chunk > max_pixels_per_chunk:
            if width_pixels / n_col_chunks > height_pixels / n_row_chunks:
                n_col_chunks += 1
            else:
                n_row_chunks += 1
            pixels_per_chunk = (width_pixels / n_col_chunks) * (height_pixels / n_row_chunks)
    
    return {
        'height_pixels': height_pixels,
        'width_pixels': width_pixels,
        'total_pixels': total_pixels,
        'n_row_chunks': n_row_chunks,
        'n_col_chunks': n_col_chunks,
        'meters_per_deg_lat': meters_per_deg_lat,
        'meters_per_deg_lon': meters_per_deg_lon,
    }


def find_alignment_offset(ref_strip, new_strip, max_shift=20):
    """
    Find the best alignment offset between two overlapping strips using cross-correlation.
    
    Args:
        ref_strip: Reference strip from existing data (H, W, C) or (H, W)
        new_strip: New strip to align (H, W, C) or (H, W)
        max_shift: Maximum pixel shift to search
    
    Returns:
        (dy, dx): Best offset to apply to new_strip
    """
    # Use first channel or average for correlation
    if ref_strip.ndim == 3:
        ref = ref_strip[:, :, 0]
        new = new_strip[:, :, 0]
    else:
        ref = ref_strip
        new = new_strip
    
    # Normalize
    ref = (ref - np.mean(ref)) / (np.std(ref) + 1e-8)
    new = (new - np.mean(new)) / (np.std(new) + 1e-8)
    
    h, w = ref.shape
    best_score = -np.inf
    best_offset = (0, 0)
    
    # Search over possible offsets
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            # Calculate overlap region
            r_y0 = max(0, dy)
            r_y1 = min(h, h + dy)
            r_x0 = max(0, dx)
            r_x1 = min(w, w + dx)
            
            n_y0 = max(0, -dy)
            n_y1 = min(h, h - dy)
            n_x0 = max(0, -dx)
            n_x1 = min(w, w - dx)
            
            if r_y1 <= r_y0 or r_x1 <= r_x0:
                continue
            
            ref_region = ref[r_y0:r_y1, r_x0:r_x1]
            new_region = new[n_y0:n_y1, n_x0:n_x1]
            
            # Cross-correlation score
            score = np.sum(ref_region * new_region)
            
            if score > best_score:
                best_score = score
                best_offset = (dy, dx)
    
    return best_offset


def blend_overlap(base, overlay, axis, blend_width=10):
    """
    Blend overlapping regions with linear gradient.
    
    Args:
        base: Base array (H, W, C)
        overlay: Overlay array (H, W, C) 
        axis: 0 for vertical blend (top-bottom), 1 for horizontal blend (left-right)
        blend_width: Width of blend region in pixels
    
    Returns:
        Blended array
    """
    result = base.copy()
    h, w = base.shape[:2]
    
    blend_width = min(blend_width, h if axis == 0 else w)
    
    if axis == 0:  # Vertical blend (overlay comes from below)
        for i in range(blend_width):
            alpha = i / blend_width
            result[-(blend_width - i), :, :] = (
                (1 - alpha) * base[-(blend_width - i), :, :] +
                alpha * overlay[-(blend_width - i), :, :]
            )
    else:  # Horizontal blend (overlay comes from right)
        for i in range(blend_width):
            alpha = i / blend_width
            result[:, -(blend_width - i), :] = (
                (1 - alpha) * base[:, -(blend_width - i), :] +
                alpha * overlay[:, -(blend_width - i), :]
            )
    
    return result


def download_large_embedding(lat_min, lat_max, lon_min, lon_max, year, scale=10):
    """
    Download a large embedding image covering the entire bounding box.
    Automatically chunks if too large for a single request.
    
    Uses overlapping chunks with cross-correlation alignment and blending
    to ensure seamless stitching.
    
    Returns: (embedding_data, grid_params) where grid_params can be passed to RGB download
    """
    MAX_PIXELS_PER_CHUNK = 35_000
    OVERLAP_PIXELS = 30  # Overlap for alignment
    BLEND_WIDTH = 15  # Blending region width
    
    print(f"Downloading embeddings for bbox...")
    print(f"  lat: [{lat_min}, {lat_max}]")
    print(f"  lon: [{lon_min}, {lon_max}]")
    print(f"  year: {year}, scale: {scale}m")
    
    # Compute grid parameters
    grid_params = compute_grid_params(lat_min, lat_max, lon_min, lon_max, scale, MAX_PIXELS_PER_CHUNK)
    height_pixels = grid_params['height_pixels']
    width_pixels = grid_params['width_pixels']
    n_row_chunks = grid_params['n_row_chunks']
    n_col_chunks = grid_params['n_col_chunks']
    meters_per_deg_lat = grid_params['meters_per_deg_lat']
    meters_per_deg_lon = grid_params['meters_per_deg_lon']
    
    print(f"  Image dimensions: {width_pixels} × {height_pixels} = {grid_params['total_pixels']:,} pixels")
    
    dataset = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
    start_date = f'{year}-01-01'
    end_date = f'{year + 1}-01-01'
    
    region = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
    image = (dataset
             .filterDate(start_date, end_date)
             .filterBounds(region)
             .first())
    
    if image is None:
        raise ValueError(f"No embedding image found for {year}")
    
    bands = [f'A{i:02d}' for i in range(64)]
    embedding = image.select(bands)
    
    total_chunks = n_row_chunks * n_col_chunks
    
    if total_chunks <= 1:
        print(f"  Single chunk download...")
        full_data = download_embedding_chunk(embedding, region, scale, bands)
    else:
        print(f"  Splitting into {n_row_chunks} rows × {n_col_chunks} cols = {total_chunks} chunks")
        print(f"  Using {OVERLAP_PIXELS}px overlap with cross-correlation alignment")
        
        # Use accumulator arrays for averaging overlaps
        full_data = np.zeros((height_pixels, width_pixels, 64), dtype=np.float32)
        weight_sum = np.zeros((height_pixels, width_pixels), dtype=np.float32)
        
        # Calculate overlap in degrees
        overlap_lat = OVERLAP_PIXELS * scale / meters_per_deg_lat
        overlap_lon = OVERLAP_PIXELS * scale / meters_per_deg_lon
        
        # Base step (without overlap)
        base_lat_step = (lat_max - lat_min) / n_row_chunks
        base_lon_step = (lon_max - lon_min) / n_col_chunks
        
        # Store chunks for alignment
        chunks = {}
        
        chunk_num = 0
        for row in range(n_row_chunks):
            for col in range(n_col_chunks):
                chunk_num += 1
                
                # Calculate chunk bounds with overlap
                c_lat_max = lat_max - row * base_lat_step
                c_lat_min = lat_max - (row + 1) * base_lat_step
                c_lon_min = lon_min + col * base_lon_step
                c_lon_max = lon_min + (col + 1) * base_lon_step
                
                # Add overlap (except at edges)
                if row > 0:
                    c_lat_max += overlap_lat
                if row < n_row_chunks - 1:
                    c_lat_min -= overlap_lat
                if col > 0:
                    c_lon_min -= overlap_lon
                if col < n_col_chunks - 1:
                    c_lon_max += overlap_lon
                
                # Clamp to original bounds
                c_lat_max = min(c_lat_max, lat_max)
                c_lat_min = max(c_lat_min, lat_min)
                c_lon_min = max(c_lon_min, lon_min)
                c_lon_max = min(c_lon_max, lon_max)
                
                print(f"  Chunk {chunk_num}/{total_chunks} (row={row}, col={col})...", end=" ", flush=True)
                
                chunk_region = ee.Geometry.Rectangle([c_lon_min, c_lat_min, c_lon_max, c_lat_max])
                chunk_data = download_embedding_chunk(embedding, chunk_region, scale, bands)
                
                print(f"{chunk_data.shape}", flush=True)
                
                # Calculate nominal position in output grid
                nominal_y = int((lat_max - c_lat_max) * meters_per_deg_lat / scale)
                nominal_x = int((c_lon_min - lon_min) * meters_per_deg_lon / scale)
                
                # Try to align with existing data using cross-correlation
                offset_y, offset_x = 0, 0
                chunk_h, chunk_w = chunk_data.shape[:2]
                
                # Check overlap with already-placed chunks
                if row > 0 or col > 0:
                    # Get the region that would overlap with existing data
                    test_y = max(0, nominal_y)
                    test_x = max(0, nominal_x)
                    test_h = min(OVERLAP_PIXELS, chunk_h, height_pixels - test_y)
                    test_w = min(OVERLAP_PIXELS, chunk_w, width_pixels - test_x)
                    
                    if test_h > 5 and test_w > 5:
                        existing_region = full_data[test_y:test_y + test_h, test_x:test_x + test_w, :]
                        existing_weight = weight_sum[test_y:test_y + test_h, test_x:test_x + test_w]
                        
                        if np.sum(existing_weight > 0) > 0.5 * test_h * test_w:
                            # Normalize existing by weight
                            mask = existing_weight > 0
                            for c in range(64):
                                existing_region[:, :, c] = np.where(
                                    mask,
                                    existing_region[:, :, c] / np.maximum(existing_weight, 1e-8),
                                    0
                                )
                            
                            new_region = chunk_data[:test_h, :test_w, :]
                            offset_y, offset_x = find_alignment_offset(
                                existing_region, new_region, max_shift=10
                            )
                            
                            if offset_y != 0 or offset_x != 0:
                                print(f"    → Alignment offset: dy={offset_y}, dx={offset_x}")
                
                # Apply offset to position
                y_start = max(0, nominal_y + offset_y)
                x_start = max(0, nominal_x + offset_x)
                
                y_end = min(y_start + chunk_h, height_pixels)
                x_end = min(x_start + chunk_w, width_pixels)
                
                src_y_start = max(0, -nominal_y - offset_y)
                src_x_start = max(0, -nominal_x - offset_x)
                
                # Extract chunk region first to get actual dimensions
                actual_h = y_end - y_start
                actual_w = x_end - x_start
                src_y_end = src_y_start + actual_h
                src_x_end = src_x_start + actual_w
                
                # Make sure we don't exceed source bounds
                if src_y_end > chunk_h:
                    actual_h -= (src_y_end - chunk_h)
                    src_y_end = chunk_h
                    y_end = y_start + actual_h
                if src_x_end > chunk_w:
                    actual_w -= (src_x_end - chunk_w)
                    src_x_end = chunk_w
                    x_end = x_start + actual_w
                
                if actual_h <= 0 or actual_w <= 0:
                    print(f"    → Skipping chunk (no valid region)")
                    continue
                
                chunk_region_data = chunk_data[src_y_start:src_y_end, src_x_start:src_x_end, :]
                
                # Create weight mask matching actual extracted region size
                local_h, local_w = chunk_region_data.shape[:2]
                weight = np.ones((local_h, local_w), dtype=np.float32)
                
                # Feather edges for smooth blending
                feather = min(BLEND_WIDTH, local_h // 4, local_w // 4)
                if feather > 1:
                    # Top edge (if not first row)
                    if row > 0:
                        for i in range(feather):
                            weight[i, :] *= i / feather
                    # Bottom edge (if not last row)
                    if row < n_row_chunks - 1:
                        for i in range(feather):
                            weight[-(i + 1), :] *= i / feather
                    # Left edge (if not first col)
                    if col > 0:
                        for i in range(feather):
                            weight[:, i] *= i / feather
                    # Right edge (if not last col)
                    if col < n_col_chunks - 1:
                        for i in range(feather):
                            weight[:, -(i + 1)] *= i / feather
                
                # Accumulate weighted data
                for c in range(64):
                    full_data[y_start:y_start + local_h, x_start:x_start + local_w, c] += chunk_region_data[:, :, c] * weight
                weight_sum[y_start:y_start + local_h, x_start:x_start + local_w] += weight
        
        # Normalize by total weight
        print("  Normalizing accumulated data...")
        mask = weight_sum > 0
        for c in range(64):
            full_data[:, :, c] = np.where(mask, full_data[:, :, c] / np.maximum(weight_sum, 1e-8), 0)
    
    print(f"  ✓ Final shape: {full_data.shape}")
    print(f"  Value range: [{full_data.min():.4f}, {full_data.max():.4f}]")
    
    return full_data, grid_params


def download_rgb_chunk(rgb_image, region, scale, bands):
    """Download a single chunk of RGB data."""
    url = rgb_image.getDownloadURL({
        'region': region,
        'scale': scale,
        'format': 'NPY',
        'crs': 'EPSG:4326'
    })
    
    response = requests.get(url, timeout=300)
    if response.status_code != 200:
        raise ValueError(f"Download failed: {response.status_code}")
    
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name
    
    chunk_data = np.load(tmp_path)
    Path(tmp_path).unlink()
    
    # Handle structured array
    if chunk_data.dtype.names is not None:
        data_list = [chunk_data[band] for band in bands]
        chunk_data = np.stack(data_list, axis=-1)
    
    return chunk_data.astype(np.float32)


def download_sentinel2_rgb(lat_min, lat_max, lon_min, lon_max, year, scale=10, grid_params=None):
    """
    Download Sentinel-2 RGB composite for visualization.
    Uses the EXACT same grid as embeddings for perfect alignment.
    
    Args:
        grid_params: If provided, use these exact grid parameters from embedding download.
                    This ensures RGB and embedding have identical dimensions.
    """
    OVERLAP_PIXELS = 30
    BLEND_WIDTH = 15
    
    print("Downloading Sentinel-2 RGB...")
    print(f"  lat: [{lat_min}, {lat_max}]")
    print(f"  lon: [{lon_min}, {lon_max}]")
    print(f"  year: {year}, scale: {scale}m")
    
    # Use provided grid_params or compute new ones
    if grid_params is None:
        MAX_PIXELS_PER_CHUNK = 35_000  # Same as embeddings for consistency
        grid_params = compute_grid_params(lat_min, lat_max, lon_min, lon_max, scale, MAX_PIXELS_PER_CHUNK)
    
    height_pixels = grid_params['height_pixels']
    width_pixels = grid_params['width_pixels']
    n_row_chunks = grid_params['n_row_chunks']
    n_col_chunks = grid_params['n_col_chunks']
    meters_per_deg_lat = grid_params['meters_per_deg_lat']
    meters_per_deg_lon = grid_params['meters_per_deg_lon']
    
    print(f"  Target dimensions: {width_pixels} × {height_pixels} (matching embeddings)")
    
    region = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
    
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(region)
          .filterDate(f'{year}-06-01', f'{year}-09-30')
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
          .median())
    
    bands = ['B4', 'B3', 'B2']
    rgb = s2.select(bands)
    
    total_chunks = n_row_chunks * n_col_chunks
    
    if total_chunks <= 1:
        print(f"  Single chunk download...")
        full_data = download_rgb_chunk(rgb, region, scale, bands)
        # Resize to match target if needed
        if full_data.shape[0] != height_pixels or full_data.shape[1] != width_pixels:
            print(f"  Resizing from {full_data.shape[:2]} to ({height_pixels}, {width_pixels})...")
            from scipy.ndimage import zoom
            zoom_factors = (height_pixels / full_data.shape[0], width_pixels / full_data.shape[1], 1)
            full_data = zoom(full_data, zoom_factors, order=1)
    else:
        print(f"  Splitting into {n_row_chunks} rows × {n_col_chunks} cols = {total_chunks} chunks")
        print(f"  Using {OVERLAP_PIXELS}px overlap with cross-correlation alignment")
        
        full_data = np.zeros((height_pixels, width_pixels, 3), dtype=np.float32)
        weight_sum = np.zeros((height_pixels, width_pixels), dtype=np.float32)
        
        overlap_lat = OVERLAP_PIXELS * scale / meters_per_deg_lat
        overlap_lon = OVERLAP_PIXELS * scale / meters_per_deg_lon
        
        base_lat_step = (lat_max - lat_min) / n_row_chunks
        base_lon_step = (lon_max - lon_min) / n_col_chunks
        
        chunk_num = 0
        for row in range(n_row_chunks):
            for col in range(n_col_chunks):
                chunk_num += 1
                
                c_lat_max = lat_max - row * base_lat_step
                c_lat_min = lat_max - (row + 1) * base_lat_step
                c_lon_min = lon_min + col * base_lon_step
                c_lon_max = lon_min + (col + 1) * base_lon_step
                
                if row > 0:
                    c_lat_max += overlap_lat
                if row < n_row_chunks - 1:
                    c_lat_min -= overlap_lat
                if col > 0:
                    c_lon_min -= overlap_lon
                if col < n_col_chunks - 1:
                    c_lon_max += overlap_lon
                
                c_lat_max = min(c_lat_max, lat_max)
                c_lat_min = max(c_lat_min, lat_min)
                c_lon_min = max(c_lon_min, lon_min)
                c_lon_max = min(c_lon_max, lon_max)
                
                print(f"  Chunk {chunk_num}/{total_chunks} (row={row}, col={col})...", end=" ", flush=True)
                
                chunk_region = ee.Geometry.Rectangle([c_lon_min, c_lat_min, c_lon_max, c_lat_max])
                chunk_data = download_rgb_chunk(rgb, chunk_region, scale, bands)
                
                print(f"{chunk_data.shape}", flush=True)
                
                nominal_y = int((lat_max - c_lat_max) * meters_per_deg_lat / scale)
                nominal_x = int((c_lon_min - lon_min) * meters_per_deg_lon / scale)
                
                offset_y, offset_x = 0, 0
                chunk_h, chunk_w = chunk_data.shape[:2]
                
                if row > 0 or col > 0:
                    test_y = max(0, nominal_y)
                    test_x = max(0, nominal_x)
                    test_h = min(OVERLAP_PIXELS, chunk_h, height_pixels - test_y)
                    test_w = min(OVERLAP_PIXELS, chunk_w, width_pixels - test_x)
                    
                    if test_h > 5 and test_w > 5:
                        existing_region = full_data[test_y:test_y + test_h, test_x:test_x + test_w, :]
                        existing_weight = weight_sum[test_y:test_y + test_h, test_x:test_x + test_w]
                        
                        if np.sum(existing_weight > 0) > 0.5 * test_h * test_w:
                            mask = existing_weight > 0
                            for c in range(3):
                                existing_region[:, :, c] = np.where(
                                    mask,
                                    existing_region[:, :, c] / np.maximum(existing_weight, 1e-8),
                                    0
                                )
                            
                            new_region = chunk_data[:test_h, :test_w, :]
                            offset_y, offset_x = find_alignment_offset(
                                existing_region, new_region, max_shift=10
                            )
                            
                            if offset_y != 0 or offset_x != 0:
                                print(f"    → Alignment offset: dy={offset_y}, dx={offset_x}")
                
                y_start = max(0, nominal_y + offset_y)
                x_start = max(0, nominal_x + offset_x)
                
                y_end = min(y_start + chunk_h, height_pixels)
                x_end = min(x_start + chunk_w, width_pixels)
                
                src_y_start = max(0, -nominal_y - offset_y)
                src_x_start = max(0, -nominal_x - offset_x)
                
                actual_h = y_end - y_start
                actual_w = x_end - x_start
                src_y_end = src_y_start + actual_h
                src_x_end = src_x_start + actual_w
                
                if src_y_end > chunk_h:
                    actual_h -= (src_y_end - chunk_h)
                    src_y_end = chunk_h
                    y_end = y_start + actual_h
                if src_x_end > chunk_w:
                    actual_w -= (src_x_end - chunk_w)
                    src_x_end = chunk_w
                    x_end = x_start + actual_w
                
                if actual_h <= 0 or actual_w <= 0:
                    print(f"    → Skipping chunk (no valid region)")
                    continue
                
                chunk_region_data = chunk_data[src_y_start:src_y_end, src_x_start:src_x_end, :]
                
                local_h, local_w = chunk_region_data.shape[:2]
                weight = np.ones((local_h, local_w), dtype=np.float32)
                
                feather = min(BLEND_WIDTH, local_h // 4, local_w // 4)
                if feather > 1:
                    if row > 0:
                        for i in range(feather):
                            weight[i, :] *= i / feather
                    if row < n_row_chunks - 1:
                        for i in range(feather):
                            weight[-(i + 1), :] *= i / feather
                    if col > 0:
                        for i in range(feather):
                            weight[:, i] *= i / feather
                    if col < n_col_chunks - 1:
                        for i in range(feather):
                            weight[:, -(i + 1)] *= i / feather
                
                for c in range(3):
                    full_data[y_start:y_start + local_h, x_start:x_start + local_w, c] += chunk_region_data[:, :, c] * weight
                weight_sum[y_start:y_start + local_h, x_start:x_start + local_w] += weight
        
        print("  Normalizing accumulated data...")
        mask = weight_sum > 0
        for c in range(3):
            full_data[:, :, c] = np.where(mask, full_data[:, :, c] / np.maximum(weight_sum, 1e-8), 0)
    
    # Normalize RGB to 0-1 range
    print("  Normalizing RGB values...")
    for i in range(3):
        band = full_data[:, :, i]
        valid = band[band > 0]
        if len(valid) > 0:
            p2, p98 = np.percentile(valid, [2, 98])
            full_data[:, :, i] = np.clip((band - p2) / (p98 - p2 + 1e-8), 0, 1)
    
    print(f"  ✓ Final shape: {full_data.shape}")
    return full_data


# =============================================================================
# Training Functions (from predict_center_pixel_map.py)
# =============================================================================

def extract_center_pixel(lat, lon, emb_dir, patch_size=64):
    """
    Extract single center pixel from embedding.
    Returns shape: (64,) for 64 channels
    
    NOTE: This matches the predict_center_pixel_map.py extraction method
    which expects (C, H, W) format: img[:, center, center]
    """
    emb_path = emb_dir / f'embedding_{lat:.6f}_{lon:.6f}.npy'

    if not emb_path.exists():
        return None

    try:
        img = np.load(emb_path)
        center = patch_size // 2

        # Extract center pixel - handles both (C, H, W) and (H, W, C)
        if img.ndim == 3:
            if img.shape[0] == 64:  # (C, H, W)
                return img[:, center, center]
            elif img.shape[2] == 64:  # (H, W, C)
                return img[center, center, :]
        return None
    except Exception as e:
        print(f'Error loading {emb_path}: {e}')
        return None


def extract_features_from_split(df, emb_dir):
    """
    Extract center pixel features and labels from a data split.
    Matches predict_center_pixel_map.py implementation.
    """
    features = []
    labels = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc='  Extracting features'):
        if pd.notna(row['lat']) and pd.notna(row['lon']):
            center_data = extract_center_pixel(row['lat'], row['lon'], emb_dir)

            if center_data is not None:
                features.append(center_data)
                labels.append(int(row['has_yew']))

    return np.array(features), np.array(labels)


def train_svm_with_validation(train_csv, val_csv, emb_dir):
    """
    Train SVM classifier with StandardScaler on labeled center pixels.
    
    This uses the SAME training approach as predict_center_pixel_map.py
    which achieves 91.58% validation accuracy.
    
    Returns: (classifier, scaler, metrics_dict)
    """
    print("Training SVM classifier with StandardScaler...")
    print("  (Using predict_center_pixel_map.py methodology)")
    
    emb_dir = Path(emb_dir)
    
    # Load data
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    
    print(f"  Train CSV: {len(train_df)} rows")
    print(f"  Val CSV: {len(val_df)} rows")
    
    # Extract features
    X_train, y_train = extract_features_from_split(train_df, emb_dir)
    X_val, y_val = extract_features_from_split(val_df, emb_dir)
    
    print(f"  Train features: {len(X_train)} (Yew: {y_train.sum()})")
    print(f"  Val features: {len(X_val)} (Yew: {y_val.sum()})")
    
    if len(X_train) == 0:
        raise ValueError("No training features found!")
    
    # Handle inf/nan
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Combine for final model (same as predict_center_pixel_map.py)
    X_all = np.vstack([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])
    
    print(f"  Combined training: {len(X_all)} samples")
    
    # Fit StandardScaler on all data
    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)
    
    print(f"  ✓ StandardScaler fitted")
    print(f"    Mean range: [{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}]")
    print(f"    Std range: [{scaler.scale_.min():.4f}, {scaler.scale_.max():.4f}]")
    
    # Train SVM on all data (same as predict_center_pixel_map.py)
    clf = SVC(kernel='rbf', probability=True, random_state=42)
    clf.fit(X_all_scaled, y_all)
    print(f"  ✓ SVM trained on {len(X_all)} samples")
    
    # Evaluate on validation set for metrics
    X_val_scaled = scaler.transform(X_val)
    y_val_pred = clf.predict(X_val_scaled)
    y_val_prob = clf.predict_proba(X_val_scaled)[:, 1]
    
    metrics = {
        'accuracy': float(accuracy_score(y_val, y_val_pred)),
        'f1_score': float(f1_score(y_val, y_val_pred)),
        'roc_auc': float(roc_auc_score(y_val, y_val_prob)),
        'n_train': int(len(X_all)),
        'n_val': int(len(X_val)),
        'train_positive_rate': float(y_all.mean()),
        'val_positive_rate': float(y_val.mean()),
    }
    
    print(f"\n  Validation Performance:")
    print(f"    Accuracy: {metrics['accuracy']:.4f}")
    print(f"    F1 Score: {metrics['f1_score']:.4f}")
    print(f"    ROC-AUC:  {metrics['roc_auc']:.4f}")
    
    return clf, scaler, metrics


# =============================================================================
# Classification Functions
# =============================================================================

def classify_image(embedding_img, classifier, scaler, batch_size=50000):
    """
    Classify every pixel in the embedding image.
    
    Args:
        embedding_img: (H, W, 64) array
        classifier: trained SVM
        scaler: fitted StandardScaler
        batch_size: pixels per batch (for memory efficiency)
    
    Returns:
        (H, W) probability array
    """
    H, W, C = embedding_img.shape
    total_pixels = H * W
    
    print(f"Classifying {total_pixels:,} pixels...")
    
    flat = embedding_img.reshape(-1, C)
    flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"  Applying StandardScaler...")
    flat_scaled = scaler.transform(flat)
    
    probs = np.zeros(total_pixels, dtype=np.float32)
    
    n_batches = (total_pixels + batch_size - 1) // batch_size
    for i in tqdm(range(0, total_pixels, batch_size), desc="  Predicting", total=n_batches):
        end = min(i + batch_size, total_pixels)
        batch = flat_scaled[i:end]
        probs[i:end] = classifier.predict_proba(batch)[:, 1]
    
    prob_grid = probs.reshape(H, W)
    
    print(f"  ✓ Classification complete")
    print(f"  Probability range: [{prob_grid.min():.3f}, {prob_grid.max():.3f}]")
    print(f"  Mean: {prob_grid.mean():.3f}, Median: {np.median(prob_grid):.3f}")
    
    return prob_grid


# =============================================================================
# Visualization Functions
# =============================================================================

def create_visualizations(prob_grid, rgb_image, embedding_image, output_dir, 
                          lat_min, lat_max, lon_min, lon_max, scale, metrics):
    """Create and save visualization figures."""
    cmap = LinearSegmentedColormap.from_list(
        'yew_prob', ['#2166ac', '#92c5de', '#f7f7f7', '#f4a582', '#d6604d', '#b2182b'], N=256
    )
    
    extent = [lon_min, lon_max, lat_min, lat_max]
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    # Three-panel view (vertical)
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    
    if rgb_image is not None:
        axes[0].imshow(rgb_image, extent=extent, aspect='equal')
        axes[0].set_title('Sentinel-2 RGB Composite', fontsize=14, fontweight='bold')
    else:
        proxy_rgb = embedding_image[:, :, :3]
        proxy_rgb = (proxy_rgb - proxy_rgb.min()) / (proxy_rgb.max() - proxy_rgb.min() + 1e-8)
        axes[0].imshow(proxy_rgb, extent=extent, aspect='equal')
        axes[0].set_title('Embedding Channels 0-2 (RGB proxy)', fontsize=14, fontweight='bold')
    
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    
    im = axes[1].imshow(prob_grid, extent=extent, cmap=cmap, 
                        vmin=0, vmax=1, aspect='equal')
    axes[1].set_title('Yew Probability (StandardScaler + SVM)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    plt.colorbar(im, ax=axes[1], fraction=0.046, label='P(yew)')
    
    if rgb_image is not None:
        axes[2].imshow(rgb_image, extent=extent, aspect='equal')
    else:
        axes[2].imshow(proxy_rgb, extent=extent, aspect='equal')
    
    im2 = axes[2].imshow(prob_grid, extent=extent, cmap=cmap, alpha=0.5,
                         vmin=0, vmax=1, aspect='equal')
    axes[2].set_title('Overlay (α=0.5)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Longitude')
    axes[2].set_ylabel('Latitude')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, label='P(yew)')
    
    plt.suptitle(f'Yew Detection: {lat_min:.4f}°N to {lat_max:.4f}°N, '
                 f'{abs(lon_min):.4f}°W to {abs(lon_max):.4f}°W\n'
                 f'Model: SVM with StandardScaler (Val Accuracy: {metrics["accuracy"]:.1%})',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figures_dir / 'three_panel_view.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved three_panel_view.png")
    
    # Histogram
    valid_probs = prob_grid.flatten()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(valid_probs, bins=50, edgecolor='black', alpha=0.7, color='#2ca02c')
    ax.axvline(valid_probs.mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {valid_probs.mean():.3f}')
    ax.axvline(np.median(valid_probs), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {np.median(valid_probs):.3f}')
    ax.axvline(0.5, color='purple', linestyle=':', linewidth=2, label='Threshold: 0.5')
    ax.set_xlabel('Yew Probability', fontsize=12)
    ax.set_ylabel('Pixel Count', fontsize=12)
    ax.set_title('Distribution of Yew Probability Predictions', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.savefig(figures_dir / 'probability_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved probability_histogram.png")
    
    # High probability map
    fig, ax = plt.subplots(figsize=(12, 10))
    
    if rgb_image is not None:
        ax.imshow(rgb_image, extent=extent, aspect='equal')
    
    # Only show high probability areas
    prob_masked = np.ma.masked_where(prob_grid < 0.7, prob_grid)
    im = ax.imshow(prob_masked, extent=extent, cmap='YlOrRd', 
                   vmin=0.7, vmax=1.0, aspect='equal', alpha=0.8)
    ax.set_title(f'High Probability Yew Locations (P ≥ 0.7)\n'
                 f'{(prob_grid >= 0.7).sum():,} pixels', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.colorbar(im, ax=ax, fraction=0.046, label='P(yew)')
    plt.savefig(figures_dir / 'high_probability_map.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved high_probability_map.png")
    
    return valid_probs


def generate_pdf_report(prob_grid, rgb_image, embedding_image, output_dir,
                        lat_min, lat_max, lon_min, lon_max, scale, metrics, valid_probs):
    """Generate a comprehensive PDF report."""
    cmap = LinearSegmentedColormap.from_list(
        'yew_prob', ['#2166ac', '#92c5de', '#f7f7f7', '#f4a582', '#d6604d', '#b2182b'], N=256
    )
    extent = [lon_min, lon_max, lat_min, lat_max]
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pdf_path = output_dir / f'yew_detection_report_{timestamp}.pdf'
    
    with PdfPages(pdf_path) as pdf:
        # Title page
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        
        title_text = "Yew Detection Report\n(Large Area Classification)"
        ax.text(0.5, 0.7, title_text, ha='center', va='center', fontsize=24, fontweight='bold')
        
        info_text = (
            f"Region: {lat_min:.4f}°N to {lat_max:.4f}°N\n"
            f"        {abs(lon_max):.4f}°W to {abs(lon_min):.4f}°W\n\n"
            f"Scale: {scale}m per pixel\n"
            f"Total Pixels: {len(valid_probs):,}\n\n"
            f"Model: SVM with StandardScaler\n"
            f"Validation Accuracy: {metrics['accuracy']:.1%}\n"
            f"Validation F1 Score: {metrics['f1_score']:.1%}\n"
            f"Validation ROC-AUC: {metrics['roc_auc']:.3f}\n\n"
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        ax.text(0.5, 0.35, info_text, ha='center', va='center', fontsize=12, family='monospace')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Statistics page
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        
        pixel_area_m2 = scale ** 2
        
        stats_text = "Detection Statistics\n" + "=" * 40 + "\n\n"
        stats_text += f"Total pixels analyzed: {len(valid_probs):,}\n"
        stats_text += f"Total area: {len(valid_probs) * pixel_area_m2 / 10000:.2f} ha\n\n"
        stats_text += f"Mean probability: {valid_probs.mean():.4f}\n"
        stats_text += f"Median probability: {np.median(valid_probs):.4f}\n"
        stats_text += f"Std deviation: {valid_probs.std():.4f}\n\n"
        
        stats_text += "Threshold Analysis:\n"
        for thresh in [0.3, 0.5, 0.7, 0.9]:
            n_above = (valid_probs >= thresh).sum()
            pct = 100 * n_above / len(valid_probs)
            area_ha = n_above * pixel_area_m2 / 10000
            stats_text += f"  P(yew) ≥ {thresh:.1f}: {n_above:>8,} pixels ({pct:>5.2f}%) = {area_ha:>8.2f} ha\n"
        
        ax.text(0.1, 0.9, stats_text, ha='left', va='top', fontsize=11, family='monospace',
                transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Three-panel visualization
        fig, axes = plt.subplots(3, 1, figsize=(11, 14))
        
        if rgb_image is not None:
            axes[0].imshow(rgb_image, extent=extent, aspect='equal')
            axes[0].set_title('Sentinel-2 RGB')
        else:
            proxy_rgb = embedding_image[:, :, :3]
            proxy_rgb = (proxy_rgb - proxy_rgb.min()) / (proxy_rgb.max() - proxy_rgb.min() + 1e-8)
            axes[0].imshow(proxy_rgb, extent=extent, aspect='equal')
            axes[0].set_title('Embedding Proxy RGB')
        
        im = axes[1].imshow(prob_grid, extent=extent, cmap=cmap, vmin=0, vmax=1, aspect='equal')
        axes[1].set_title('Yew Probability')
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        
        if rgb_image is not None:
            axes[2].imshow(rgb_image, extent=extent, aspect='equal')
        else:
            axes[2].imshow(proxy_rgb, extent=extent, aspect='equal')
        axes[2].imshow(prob_grid, extent=extent, cmap=cmap, alpha=0.5, vmin=0, vmax=1, aspect='equal')
        axes[2].set_title('Overlay')
        
        for ax in axes:
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Histogram page
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.hist(valid_probs, bins=50, edgecolor='black', alpha=0.7, color='#2ca02c')
        ax.axvline(valid_probs.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {valid_probs.mean():.3f}')
        ax.axvline(np.median(valid_probs), color='orange', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(valid_probs):.3f}')
        ax.set_xlabel('Yew Probability')
        ax.set_ylabel('Pixel Count')
        ax.set_title('Distribution of Predictions')
        ax.legend()
        ax.grid(alpha=0.3)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    print(f"  ✓ Saved PDF report: {pdf_path.name}")
    return pdf_path


def print_statistics(valid_probs, scale):
    """Print detection statistics."""
    print("\n" + "=" * 60)
    print("YEW DETECTION STATISTICS")
    print("=" * 60)
    print(f"Total pixels: {len(valid_probs):,}")
    print(f"Mean probability: {valid_probs.mean():.4f}")
    print(f"Median probability: {np.median(valid_probs):.4f}")
    print(f"Std deviation: {valid_probs.std():.4f}")
    print()
    
    pixel_area_m2 = scale ** 2
    
    for thresh in [0.3, 0.5, 0.7, 0.9]:
        n_above = (valid_probs >= thresh).sum()
        pct = 100 * n_above / len(valid_probs)
        area_ha = n_above * pixel_area_m2 / 10000
        print(f"P(yew) ≥ {thresh:.1f}: {n_above:>8,} pixels ({pct:>5.2f}%) = {area_ha:>8.2f} ha")
    
    print("=" * 60)


def save_results(prob_grid, embedding_image, rgb_image, valid_probs, output_dir,
                 lat_min, lat_max, lon_min, lon_max, year, scale, metrics):
    """Save all results to disk."""
    np.save(output_dir / 'prob_grid.npy', prob_grid)
    print(f"  ✓ Saved prob_grid.npy: {prob_grid.shape}")
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'bbox': {
            'lat_min': lat_min,
            'lat_max': lat_max,
            'lon_min': lon_min,
            'lon_max': lon_max
        },
        'year': year,
        'scale_m': scale,
        'image_shape': list(embedding_image.shape),
        'prob_grid_shape': list(prob_grid.shape),
        'model': {
            'type': 'SVM with StandardScaler',
            'kernel': 'rbf',
            'training_samples': metrics['n_train'],
            'validation_accuracy': metrics['accuracy'],
            'validation_f1': metrics['f1_score'],
            'validation_roc_auc': metrics['roc_auc'],
        },
        'statistics': {
            'mean': float(valid_probs.mean()),
            'median': float(np.median(valid_probs)),
            'std': float(valid_probs.std()),
            'min': float(valid_probs.min()),
            'max': float(valid_probs.max()),
            'pixels_above_30': int((valid_probs >= 0.3).sum()),
            'pixels_above_50': int((valid_probs >= 0.5).sum()),
            'pixels_above_70': int((valid_probs >= 0.7).sum()),
            'pixels_above_90': int((valid_probs >= 0.9).sum()),
            'area_ha_above_70': float((valid_probs >= 0.7).sum() * scale * scale / 10000),
        }
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Saved metadata.json")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    
    lat_min, lat_max, lon_min, lon_max = args.bbox
    
    # Initialize Earth Engine
    print("Initializing Earth Engine...")
    ee.Initialize(project=args.gee_project)
    print("✓ Earth Engine initialized")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("LARGE AREA YEW CLASSIFICATION")
    print(f"{'='*60}")
    print(f"Bbox: lat [{lat_min}, {lat_max}], lon [{lon_min}, {lon_max}]")
    print(f"Scale: {args.scale}m per pixel")
    print(f"Output: {output_dir}\n")
    
    # Compute grid parameters once for both embedding and RGB
    MAX_PIXELS_PER_CHUNK = 35_000
    grid_params = compute_grid_params(lat_min, lat_max, lon_min, lon_max, args.scale, MAX_PIXELS_PER_CHUNK)
    
    # Save grid params for caching consistency
    grid_params_cache = output_dir / 'grid_params.json'
    if not grid_params_cache.exists():
        with open(grid_params_cache, 'w') as f:
            json.dump(grid_params, f, indent=2)
    
    # Step 1: Download or load embedding image
    embedding_cache = output_dir / 'embedding_image.npy'
    if embedding_cache.exists():
        print("Loading cached embedding image...")
        embedding_image = np.load(embedding_cache)
        print(f"✓ Loaded from cache: {embedding_image.shape}\n")
    else:
        embedding_image, grid_params = download_large_embedding(
            lat_min, lat_max, lon_min, lon_max, args.year, args.scale
        )
        print(f"\n✓ Embedding image shape: {embedding_image.shape}")
        np.save(embedding_cache, embedding_image)
        print(f"✓ Cached to {embedding_cache}\n")
        # Update grid params cache
        with open(grid_params_cache, 'w') as f:
            json.dump(grid_params, f, indent=2)
    
    # Step 2: Download or load RGB for visualization (optional)
    rgb_image = None
    if not args.skip_rgb:
        rgb_cache = output_dir / 'rgb_image.npy'
        if rgb_cache.exists():
            print("Loading cached RGB image...")
            rgb_image = np.load(rgb_cache)
            print(f"✓ Loaded from cache: {rgb_image.shape}")
            # Check if dimensions match embedding
            if rgb_image.shape[:2] != embedding_image.shape[:2]:
                print(f"  ⚠ RGB shape {rgb_image.shape[:2]} differs from embedding {embedding_image.shape[:2]}")
                print(f"  → Deleting cache and re-downloading with matching grid...")
                rgb_cache.unlink()
                rgb_image = None
            else:
                print()
        
        if rgb_image is None:
            try:
                # Load grid_params from cache if available
                if grid_params_cache.exists():
                    with open(grid_params_cache, 'r') as f:
                        grid_params = json.load(f)
                
                rgb_image = download_sentinel2_rgb(
                    lat_min, lat_max, lon_min, lon_max, args.year, args.scale,
                    grid_params=grid_params
                )
                print(f"✓ RGB image shape: {rgb_image.shape}")
                np.save(rgb_cache, rgb_image)
                print(f"✓ Cached to {rgb_cache}\n")
            except Exception as e:
                print(f"⚠ Could not download RGB: {e}\n")
    
    # Step 3: Train SVM with StandardScaler (predict_center_pixel_map.py method)
    print("\n" + "-" * 60)
    clf, scaler, metrics = train_svm_with_validation(
        args.train_csv, args.val_csv, args.embedding_dir
    )
    print("-" * 60 + "\n")
    
    # Step 4: Classify image
    prob_grid = classify_image(embedding_image, clf, scaler, args.batch_size)
    print()
    
    # Step 5: Create visualizations
    print("Creating visualizations...")
    valid_probs = create_visualizations(
        prob_grid, rgb_image, embedding_image, output_dir,
        lat_min, lat_max, lon_min, lon_max, args.scale, metrics
    )
    print()
    
    # Step 6: Generate PDF report
    print("Generating PDF report...")
    generate_pdf_report(
        prob_grid, rgb_image, embedding_image, output_dir,
        lat_min, lat_max, lon_min, lon_max, args.scale, metrics, valid_probs
    )
    print()
    
    # Step 7: Print statistics
    print_statistics(valid_probs, args.scale)
    
    # Step 8: Save results
    print("\nSaving results...")
    save_results(
        prob_grid, embedding_image, rgb_image, valid_probs, output_dir,
        lat_min, lat_max, lon_min, lon_max, args.year, args.scale, metrics
    )
    
    print(f"\n✓ All results saved to: {output_dir}")
    print(f"\nKey outputs:")
    print(f"  - {output_dir}/prob_grid.npy")
    print(f"  - {output_dir}/figures/three_panel_view.png")
    print(f"  - {output_dir}/figures/high_probability_map.png")
    print(f"  - {output_dir}/yew_detection_report_*.pdf")


if __name__ == '__main__':
    main()
