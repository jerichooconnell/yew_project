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


def create_kmz(input_dir, output_path=None, threshold=0.0, name=None):
    """Create KMZ file from prediction results."""
    input_dir = Path(input_dir)
    
    # Load data
    print("Loading data...")
    prob_grid = np.load(input_dir / 'prob_grid.npy')
    print(f"  Probability grid: {prob_grid.shape}")
    
    with open(input_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    bbox = metadata.get('bbox', {})
    lat_min = bbox.get('lat_min', 0)
    lat_max = bbox.get('lat_max', 0)
    lon_min = bbox.get('lon_min', 0)
    lon_max = bbox.get('lon_max', 0)
    
    print(f"  Bounds: {lat_min:.4f}°N to {lat_max:.4f}°N, {lon_min:.4f}°W to {lon_max:.4f}°W")
    
    # Create colormap and apply
    print("Creating probability overlay...")
    cmap = create_probability_colormap()
    rgba = apply_colormap_with_threshold(prob_grid, cmap, threshold)
    
    # Convert to PIL Image
    prob_img = Image.fromarray(rgba, 'RGBA')
    
    # Create legend
    print("Creating legend...")
    legend_img = create_legend_image()
    
    # Create KML
    area_name = name or input_dir.name
    kml_content = create_kml(
        name=f"Yew Probability - {area_name}",
        description=f"Pacific Yew detection probability map. Threshold: {threshold}",
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        image_filename="overlay.png"
    )
    
    # Create KMZ (ZIP file)
    if output_path is None:
        output_path = input_dir / f'{area_name}_yew_probability.kmz'
    else:
        output_path = Path(output_path)
    
    print(f"Creating KMZ: {output_path}")
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as kmz:
        # Add KML
        kmz.writestr('doc.kml', kml_content)
        
        # Add overlay image
        img_buffer = io.BytesIO()
        prob_img.save(img_buffer, format='PNG')
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
