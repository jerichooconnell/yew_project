#!/usr/bin/env python3
"""
Create KMZ file for Google Earth from yew probability predictions.
"""

import argparse
import json
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import io


def create_probability_colormap():
    """Create blue-white-red colormap matching the web map."""
    cmap = np.zeros((256, 4), dtype=np.uint8)
    
    # Blue (low) -> White (mid) -> Red (high)
    colors = [
        (0, (33, 102, 172)),      # Blue
        (64, (146, 197, 222)),    # Light blue
        (128, (247, 247, 247)),   # White
        (192, (244, 165, 130)),   # Light red
        (255, (214, 96, 77)),     # Red
    ]
    
    for i in range(len(colors) - 1):
        idx1, c1 = colors[i]
        idx2, c2 = colors[i + 1]
        
        for j in range(idx1, idx2 + 1):
            t = (j - idx1) / (idx2 - idx1) if idx2 != idx1 else 0
            cmap[j, 0] = int(c1[0] * (1 - t) + c2[0] * t)
            cmap[j, 1] = int(c1[1] * (1 - t) + c2[1] * t)
            cmap[j, 2] = int(c1[2] * (1 - t) + c2[2] * t)
            cmap[j, 3] = 255  # Full opacity
    
    return cmap


def apply_colormap_with_threshold(prob_grid, cmap, threshold=0.0):
    """Apply colormap to probability grid with optional threshold."""
    h, w = prob_grid.shape
    
    # Normalize to 0-255
    prob_normalized = np.clip(prob_grid, 0, 1)
    prob_idx = (prob_normalized * 255).astype(np.uint8)
    
    # Apply colormap
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = cmap[prob_idx, 0]
    rgba[:, :, 1] = cmap[prob_idx, 1]
    rgba[:, :, 2] = cmap[prob_idx, 2]
    rgba[:, :, 3] = cmap[prob_idx, 3]
    
    # Apply threshold - make pixels below threshold transparent
    if threshold > 0:
        mask = prob_grid < threshold
        rgba[mask, 3] = 0
    
    return rgba


def create_kml(name, description, lat_min, lat_max, lon_min, lon_max, image_filename):
    """Create KML content for ground overlay."""
    kml = f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{name}</name>
    <description>{description}</description>
    
    <Style id="legendStyle">
      <BalloonStyle>
        <text><![CDATA[
          <h2>Yew Probability Legend</h2>
          <p><span style="color: #2166ac;">■</span> Low (0.0 - 0.3)</p>
          <p><span style="color: #92c5de;">■</span> Low-Medium (0.3 - 0.5)</p>
          <p><span style="color: #f7f7f7; background: #ccc;">■</span> Medium (0.5)</p>
          <p><span style="color: #f4a582;">■</span> Medium-High (0.5 - 0.7)</p>
          <p><span style="color: #d6604d;">■</span> High (0.7 - 1.0)</p>
        ]]></text>
      </BalloonStyle>
    </Style>
    
    <Folder>
      <name>Yew Detection</name>
      <open>1</open>
      
      <GroundOverlay>
        <name>Yew Probability</name>
        <description>Pacific Yew (Taxus brevifolia) detection probability from Sentinel-2 imagery using SVM classifier.</description>
        <color>c0ffffff</color>
        <Icon>
          <href>{image_filename}</href>
        </Icon>
        <LatLonBox>
          <north>{lat_max}</north>
          <south>{lat_min}</south>
          <east>{lon_max}</east>
          <west>{lon_min}</west>
        </LatLonBox>
      </GroundOverlay>
      
      <ScreenOverlay>
        <name>Legend</name>
        <Icon>
          <href>legend.png</href>
        </Icon>
        <overlayXY x="0" y="1" xunits="fraction" yunits="fraction"/>
        <screenXY x="0.02" y="0.98" xunits="fraction" yunits="fraction"/>
        <size x="0" y="0" xunits="fraction" yunits="fraction"/>
      </ScreenOverlay>
      
    </Folder>
  </Document>
</kml>'''
    return kml


def create_legend_image():
    """Create a legend image for the colormap."""
    width = 200
    height = 180
    
    img = Image.new('RGBA', (width, height), (255, 255, 255, 230))
    
    # We'll create a simple legend using PIL
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    
    # Title
    draw.text((10, 10), "Yew Probability", fill=(0, 0, 0, 255))
    
    # Colorbar
    cmap = create_probability_colormap()
    bar_x, bar_y = 20, 40
    bar_width, bar_height = 20, 100
    
    for i in range(bar_height):
        # Map position to color (bottom = 0, top = 1)
        prob = 1.0 - (i / bar_height)
        idx = int(prob * 255)
        color = tuple(cmap[idx, :3])
        draw.rectangle([bar_x, bar_y + i, bar_x + bar_width, bar_y + i + 1], fill=color)
    
    # Labels
    draw.text((bar_x + bar_width + 10, bar_y - 5), "1.0 High", fill=(0, 0, 0, 255))
    draw.text((bar_x + bar_width + 10, bar_y + bar_height//2 - 5), "0.5", fill=(0, 0, 0, 255))
    draw.text((bar_x + bar_width + 10, bar_y + bar_height - 5), "0.0 Low", fill=(0, 0, 0, 255))
    
    # Border
    draw.rectangle([0, 0, width-1, height-1], outline=(100, 100, 100, 255))
    
    return img


def _remove_tile_duplicates(prob_grid, tile_info_path):
    """Remove duplicate boundary pixels from a stitched tile grid.

    When GEE tiles are downloaded with adjacent bboxes (tile-N east == tile-N+1
    west), GEE includes the boundary pixel in BOTH downloads.  Naïve
    concatenation therefore duplicates one column at each vertical boundary and
    one row at each horizontal boundary.

    This function removes those duplicates so that every pixel in the output
    maps to a unique geographic position.

    Returns (deduped_grid, n_tile_rows, n_tile_cols) or the original grid
    unchanged if tile_info is not available.
    """
    if not Path(tile_info_path).exists():
        return prob_grid, 1, 1

    with open(tile_info_path) as f:
        tile_info = json.load(f)

    n_tile_rows = tile_info.get('n_rows', 1)
    n_tile_cols = tile_info.get('n_cols', 1)
    tiles = tile_info['tiles']

    if n_tile_rows <= 1 and n_tile_cols <= 1:
        return prob_grid, 1, 1

    # Build list of per-tile pixel heights/widths (from tile_info shapes)
    tile_shapes = {}
    for t in tiles:
        r, c = t['row'], t['col']
        tile_shapes[(r, c)] = tuple(t['shape'])  # (rows, cols)

    h, w = prob_grid.shape

    # --- Remove duplicate columns (vertical boundaries) ---
    # Each tile in a row contributes its full width.  At boundary between
    # col c and col c+1 there is 1 shared column.  We strip the LAST column
    # of every tile except the rightmost one.
    if n_tile_cols > 1:
        col_widths = [tile_shapes.get((0, c), (0, 0))[1] for c in range(n_tile_cols)]
        keep_cols = np.ones(w, dtype=bool)
        offset = 0
        for c in range(n_tile_cols - 1):  # skip last tile
            offset += col_widths[c]
            # The last column of this tile (= first column of next tile) is
            # at index offset-1 in the stitched array.
            if offset - 1 < w:
                keep_cols[offset - 1] = False
        prob_grid = prob_grid[:, keep_cols]

    h, w = prob_grid.shape

    # --- Remove duplicate rows (horizontal boundaries) ---
    if n_tile_rows > 1:
        row_heights = [tile_shapes.get((r, 0), (0, 0))[0] for r in range(n_tile_rows)]
        keep_rows = np.ones(h, dtype=bool)
        offset = 0
        for r in range(n_tile_rows - 1):  # skip last tile
            offset += row_heights[r]
            if offset - 1 < h:
                keep_rows[offset - 1] = False
        prob_grid = prob_grid[keep_rows, :]

    return prob_grid, n_tile_rows, n_tile_cols


def create_kmz(input_dir, output_path=None, threshold=0.0, name=None):
    """Create KMZ file from prediction results."""
    input_dir = Path(input_dir)
    
    # Load data
    print("Loading data...")
    prob_grid = np.load(input_dir / 'prob_grid.npy')
    print(f"  Probability grid (raw stitched): {prob_grid.shape}")
    
    with open(input_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    bbox = metadata.get('bbox', {})
    # Support both metadata formats (SVM uses lat_min/lon_min, GPU uses south/north/west/east)
    if 'lat_min' in bbox:
        lat_min, lat_max = bbox['lat_min'], bbox['lat_max']
        lon_min, lon_max = bbox['lon_min'], bbox['lon_max']
    else:
        lat_min, lat_max = bbox.get('south', 0), bbox.get('north', 0)
        lon_min, lon_max = bbox.get('west', 0), bbox.get('east', 0)
    
    print(f"  Requested bbox: {lat_min:.6f}°N to {lat_max:.6f}°N, "
          f"{lon_min:.6f}°E to {lon_max:.6f}°E")

    # ----------------------------------------------------------------
    # Remove duplicate boundary pixels from tile stitching.
    #
    # When GEE tiles are downloaded with adjacent bboxes, the pixel at
    # the shared boundary is included in BOTH tiles.  Naïve concatenation
    # duplicates these pixels.  For a 4×7 tile grid, that's 6 extra
    # columns and 3 extra rows.  At ~10 m/pixel this causes ~40 m of
    # cumulative stretch error in the KMZ overlay.
    # ----------------------------------------------------------------
    # Try to find tile_info.json in the source prediction directory
    tile_info_path = None
    for candidate in [input_dir / 'tile_info.json',
                      input_dir.parent / input_dir.name.replace('_forestry', '') / 'tile_info.json']:
        if candidate.exists():
            tile_info_path = candidate
            break
    # Also check metadata for tile_info_dir hint
    if tile_info_path is None:
        src = metadata.get('source_dir', '')
        if src:
            candidate = Path(src) / 'tile_info.json'
            if candidate.exists():
                tile_info_path = candidate

    raw_h, raw_w = prob_grid.shape
    if tile_info_path and tile_info_path.exists():
        prob_grid, n_tile_rows, n_tile_cols = _remove_tile_duplicates(
            prob_grid, tile_info_path)
        dup_rows = raw_h - prob_grid.shape[0]
        dup_cols = raw_w - prob_grid.shape[1]
        if dup_rows > 0 or dup_cols > 0:
            print(f"  Removed {dup_rows} duplicate rows, {dup_cols} duplicate cols "
                  f"from tile stitching")
            print(f"  Deduplicated grid: {prob_grid.shape}")
    else:
        print(f"  tile_info.json not found — assuming no tile duplicates")

    h, w = prob_grid.shape

    # ----------------------------------------------------------------
    # Compute GEE EPSG:4326 edge-to-edge bounding box.
    #
    # GEE uses a fixed global pixel grid with WGS84-exact spacing:
    #   px_deg = scale_m / (2π × 6378137 / 360)
    # Pixel edges at n * px_deg, centres at (n + 0.5) * px_deg.
    #
    # GEE includes pixels whose centres fall inside [west..east] ×
    # [south..north] (inclusive, with small floating-point tolerance).
    # The KML LatLonBox needs the outer EDGES, not centres, so we
    # expand by half a pixel on every side.
    # ----------------------------------------------------------------
    import math
    scale_m = metadata.get('scale_m', 10)
    WGS84_R = 6378137.0
    px_deg = scale_m / (2.0 * math.pi * WGS84_R / 360.0)  # 0.0000898315284...°

    # Use the IMAGE DIMENSIONS to compute the edge-to-edge extent.
    # The NW pixel centre is the first GEE pixel inside the requested bbox.
    # Snap to GEE's global grid: centres at (n + 0.5) * px_deg.
    #
    # GEE has ~1 m tolerance at boundaries, so use floor/ceil with a tiny
    # nudge (1 m ≈ 1e-5°) to match GEE's inclusive behaviour.
    tol = 1e-5  # ~1 m tolerance in degrees
    n_west  = math.ceil ((lon_min - tol) / px_deg - 0.5)   # westernmost pixel index
    n_north = math.floor((lat_max + tol) / px_deg - 0.5)   # northernmost pixel index

    # The image has h rows and w cols; the SE pixel index follows directly:
    n_east  = n_west + (w - 1)
    n_south = n_north - (h - 1)

    # Pixel centres
    nw_lon = (n_west  + 0.5) * px_deg
    nw_lat = (n_north + 0.5) * px_deg
    se_lon = (n_east  + 0.5) * px_deg
    se_lat = (n_south + 0.5) * px_deg

    # Edge-to-edge bbox
    half_px = px_deg / 2.0
    edge_lon_min = nw_lon - half_px  # west
    edge_lon_max = se_lon + half_px  # east
    edge_lat_max = nw_lat + half_px  # north
    edge_lat_min = se_lat - half_px  # south

    mid_lat = (lat_min + lat_max) / 2.0
    lon_m_per_deg = 111320 * math.cos(math.radians(mid_lat))
    print(f"  WGS84 pixel size: {px_deg:.14f}° ({scale_m} m)")
    print(f"  Deduped grid: {h} rows × {w} cols")
    print(f"  NW pixel centre: lon={nw_lon:.10f}, lat={nw_lat:.10f}")
    print(f"  SE pixel centre: lon={se_lon:.10f}, lat={se_lat:.10f}")
    print(f"  Edge-to-edge bbox: N={edge_lat_max:.10f}, S={edge_lat_min:.10f}, "
          f"W={edge_lon_min:.10f}, E={edge_lon_max:.10f}")
    print(f"  Shift from request: "
          f"N={(edge_lat_max - lat_max) * 111320:+.1f} m, "
          f"S={(edge_lat_min - lat_min) * 111320:+.1f} m, "
          f"W={(edge_lon_min - lon_min) * lon_m_per_deg:+.1f} m, "
          f"E={(edge_lon_max - lon_max) * lon_m_per_deg:+.1f} m")

    # Create colormap and apply
    print("Creating probability overlay...")
    cmap = create_probability_colormap()
    rgba = apply_colormap_with_threshold(prob_grid, cmap, threshold)

    # Convert to PIL Image
    prob_img = Image.fromarray(rgba, 'RGBA')

    # Downsample if very large (Google Earth handles ~4096px well, >10K can be slow)
    max_dim = 8192
    if max(h, w) > max_dim:
        scale_factor = max_dim / max(h, w)
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        print(f"  Downscaling for KMZ: {w}×{h} → {new_w}×{new_h}")
        prob_img = prob_img.resize((new_w, new_h), Image.LANCZOS)
    
    # Create legend
    print("Creating legend...")
    legend_img = create_legend_image()
    
    # Create KML  (use corrected edge-to-edge bbox)
    area_name = name or input_dir.name
    kml_content = create_kml(
        name=f"Yew Probability - {area_name}",
        description=f"Pacific Yew detection probability map. Threshold: {threshold}",
        lat_min=edge_lat_min,
        lat_max=edge_lat_max,
        lon_min=edge_lon_min,
        lon_max=edge_lon_max,
        image_filename="overlay.png"
    )
    
    # Create KMZ (ZIP file)
    if output_path is None:
        output_path = input_dir / f'{area_name}_yew_probability.kmz'
    else:
        output_path = Path(output_path)
    
    print(f"Creating KMZ: {output_path}")
    print(f"  Encoding overlay PNG ({prob_img.size[0]}x{prob_img.size[1]})...")
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as kmz:
        # Add KML
        kmz.writestr('doc.kml', kml_content)
        
        # Add overlay image (optimize PNG for speed)
        img_buffer = io.BytesIO()
        prob_img.save(img_buffer, format='PNG', optimize=False, compress_level=1)
        kmz.writestr('overlay.png', img_buffer.getvalue())
        
        # Add legend
        legend_buffer = io.BytesIO()
        legend_img.save(legend_buffer, format='PNG')
        kmz.writestr('legend.png', legend_buffer.getvalue())
    
    print(f"✓ Created: {output_path}")
    print(f"\nTo use:")
    print(f"  1. Download Google Earth Pro (free) or use Google Earth Web")
    print(f"  2. Open the KMZ file (double-click or File > Open)")
    print(f"  3. The yew probability layer will appear on the map")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Create KMZ file for Google Earth')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory with prob_grid.npy and metadata.json')
    parser.add_argument('--output', type=str, default=None,
                        help='Output KMZ file path')
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='Probability threshold (pixels below become transparent)')
    parser.add_argument('--name', type=str, default=None,
                        help='Name for the layer')
    args = parser.parse_args()
    
    create_kmz(args.input_dir, args.output, args.threshold, args.name)


if __name__ == '__main__':
    main()
