#!/usr/bin/env python3
"""
Interactive Image Review Tool
==============================

Review satellite images alongside iNaturalist observations to verify
data quality and alignment.

Author: GitHub Copilot
Date: November 14, 2025

Usage:
    python scripts/visualization/review_inat_images.py
    
Controls:
    - Left/Right arrows: Navigate between images
    - Y: Mark as good
    - N: Mark as bad
    - U: Mark as uncertain
    - S: Skip (no decision)
    - Q: Quit and save
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from pathlib import Path
import webbrowser
from PIL import Image
import json


class ImageReviewer:
    """Interactive tool to review satellite images against iNaturalist data."""

    def __init__(self, metadata_file, image_dir):
        """
        Initialize the reviewer.

        Args:
            metadata_file: Path to metadata CSV
            image_dir: Directory containing .npy files
        """
        self.metadata = pd.read_csv(metadata_file)
        self.image_dir = Path(image_dir)
        self.current_idx = 0
        self.reviews = {}
        self.review_file = Path('data/ee_imagery/image_review_results.json')

        # Load existing reviews if they exist
        if self.review_file.exists():
            with open(self.review_file, 'r') as f:
                self.reviews = json.load(f)
            print(f"Loaded {len(self.reviews)} existing reviews")

        # Load iNaturalist data for cross-reference
        self.inat_df = pd.read_csv(
            'data/inat_observations/observations-558049.csv')

        # Setup figure
        self.setup_figure()

    def setup_figure(self):
        """Create the matplotlib figure and controls."""
        self.fig = plt.figure(figsize=(16, 10))

        # Create subplots
        gs = self.fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Satellite images
        self.ax_rgb = self.fig.add_subplot(gs[0:2, 0])
        self.ax_false = self.fig.add_subplot(gs[0:2, 1])

        # Info panel
        self.ax_info = self.fig.add_subplot(gs[0:2, 2])
        self.ax_info.axis('off')

        # Controls
        self.ax_status = self.fig.add_subplot(gs[2, :])
        self.ax_status.axis('off')

        # Buttons
        button_y = 0.02
        button_h = 0.04
        button_w = 0.08

        self.btn_prev = Button(
            plt.axes([0.1, button_y, button_w, button_h]), 'Previous')
        self.btn_next = Button(
            plt.axes([0.2, button_y, button_w, button_h]), 'Next')
        self.btn_good = Button(
            plt.axes([0.35, button_y, button_w, button_h]), 'Good (Y)')
        self.btn_bad = Button(
            plt.axes([0.45, button_y, button_w, button_h]), 'Bad (N)')
        self.btn_uncertain = Button(
            plt.axes([0.55, button_y, button_w, button_h]), 'Uncertain (U)')
        self.btn_skip = Button(
            plt.axes([0.65, button_y, button_w, button_h]), 'Skip (S)')
        self.btn_url = Button(
            plt.axes([0.75, button_y, button_w, button_h]), 'Open iNat')
        self.btn_save = Button(
            plt.axes([0.85, button_y, button_w, button_h]), 'Save & Quit')

        # Connect buttons
        self.btn_prev.on_clicked(lambda e: self.navigate(-1))
        self.btn_next.on_clicked(lambda e: self.navigate(1))
        self.btn_good.on_clicked(lambda e: self.mark('good'))
        self.btn_bad.on_clicked(lambda e: self.mark('bad'))
        self.btn_uncertain.on_clicked(lambda e: self.mark('uncertain'))
        self.btn_skip.on_clicked(lambda e: self.navigate(1))
        self.btn_url.on_clicked(lambda e: self.open_url())
        self.btn_save.on_clicked(lambda e: self.save_and_quit())

        # Keyboard shortcuts
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def normalize_band(self, band, percentile_clip=2):
        """Normalize a band for display."""
        p_low = np.percentile(band, percentile_clip)
        p_high = np.percentile(band, 100 - percentile_clip)
        band_clipped = np.clip(band, p_low, p_high)
        band_min = band_clipped.min()
        band_max = band_clipped.max()
        if band_max > band_min:
            return ((band_clipped - band_min) / (band_max - band_min) * 255).astype(np.uint8)
        return np.zeros_like(band_clipped, dtype=np.uint8)

    def load_current_image(self):
        """Load and display the current image."""
        if self.current_idx >= len(self.metadata):
            print("End of dataset!")
            return

        row = self.metadata.iloc[self.current_idx]
        obs_id = row['observation_id']

        # Load .npy file
        img_path = self.image_dir / row['image_path']
        img = np.load(img_path)

        # Create RGB and false color
        blue = self.normalize_band(img[0])
        green = self.normalize_band(img[1])
        red = self.normalize_band(img[2])
        nir = self.normalize_band(img[3])

        rgb = np.stack([red, green, blue], axis=2)
        false_color = np.stack([nir, red, green], axis=2)

        # Display images
        self.ax_rgb.clear()
        self.ax_rgb.imshow(rgb)
        self.ax_rgb.set_title('RGB (True Color)',
                              fontsize=12, fontweight='bold')
        self.ax_rgb.axis('off')

        self.ax_false.clear()
        self.ax_false.imshow(false_color)
        self.ax_false.set_title('False Color (NIR-R-G)',
                                fontsize=12, fontweight='bold')
        self.ax_false.axis('off')

        # Get iNaturalist info
        inat_row = self.inat_df[self.inat_df['id'] == obs_id]

        # Display info
        self.ax_info.clear()
        self.ax_info.axis('off')

        info_text = []
        info_text.append(
            f"═══ IMAGE {self.current_idx + 1} / {len(self.metadata)} ═══\n")
        info_text.append(f"Observation ID: {obs_id}")

        if len(inat_row) > 0:
            obs = inat_row.iloc[0]
            info_text.append(f"\n--- iNaturalist Data ---")
            info_text.append(f"Date: {obs.get('observed_on', 'N/A')}")
            info_text.append(f"Location: {obs.get('place_guess', 'N/A')}")
            info_text.append(
                f"Accuracy: {row.get('positional_accuracy', 'N/A'):.1f}m")
            info_text.append(f"Lat/Lon: {row['lat']:.5f}, {row['lon']:.5f}")
            info_text.append(f"\nQuality: {obs.get('quality_grade', 'N/A')}")
            info_text.append(f"User: {obs.get('user_login', 'N/A')}")

            # Description if available
            desc = obs.get('description', '')
            if pd.notna(desc) and desc:
                info_text.append(f"\nNotes: {desc[:100]}...")

        info_text.append(f"\n--- Satellite Data ---")
        info_text.append(f"Images used: {row['num_source_images']}")
        info_text.append(f"Date range: 2020-2024")

        # Review status
        status = self.reviews.get(str(obs_id), 'Not reviewed')
        info_text.append(f"\n--- Review Status ---")
        info_text.append(f"Status: {status}")

        # Quality checks
        info_text.append(f"\n--- Quality Checks ---")
        # Check for spatial variation
        stds = [img[i].std() for i in range(4)]
        avg_std = np.mean(stds)
        info_text.append(
            f"✓ Spatial variation: {avg_std:.1f}" if avg_std > 10 else f"✗ Low variation: {avg_std:.1f}")

        # Check accuracy
        accuracy = row.get('positional_accuracy', 999)
        info_text.append(
            f"✓ GPS accuracy: {accuracy:.1f}m" if accuracy <= 50 else f"⚠ GPS accuracy: {accuracy:.1f}m")

        self.ax_info.text(0.05, 0.95, '\n'.join(info_text),
                          transform=self.ax_info.transAxes,
                          fontsize=9, verticalalignment='top',
                          family='monospace')

        # Update status bar
        self.ax_status.clear()
        self.ax_status.axis('off')

        # Count reviews
        good = sum(1 for v in self.reviews.values() if v == 'good')
        bad = sum(1 for v in self.reviews.values() if v == 'bad')
        uncertain = sum(1 for v in self.reviews.values() if v == 'uncertain')

        status_text = (f"Progress: {self.current_idx + 1}/{len(self.metadata)} | "
                       f"Reviewed: {len(self.reviews)} | "
                       f"Good: {good} | Bad: {bad} | Uncertain: {uncertain}")

        self.ax_status.text(0.5, 0.5, status_text,
                            transform=self.ax_status.transAxes,
                            fontsize=11, ha='center', va='center',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.draw()

    def navigate(self, delta):
        """Move to next/previous image."""
        self.current_idx = max(
            0, min(len(self.metadata) - 1, self.current_idx + delta))
        self.load_current_image()

    def mark(self, status):
        """Mark current image with a status."""
        row = self.metadata.iloc[self.current_idx]
        obs_id = str(row['observation_id'])
        self.reviews[obs_id] = status
        print(f"Marked observation {obs_id} as: {status}")
        self.navigate(1)  # Move to next

    def open_url(self):
        """Open iNaturalist URL in browser."""
        row = self.metadata.iloc[self.current_idx]
        obs_id = row['observation_id']
        url = f"https://www.inaturalist.org/observations/{obs_id}"
        print(f"Opening: {url}")
        webbrowser.open(url)

    def on_key(self, event):
        """Handle keyboard shortcuts."""
        if event.key == 'right':
            self.navigate(1)
        elif event.key == 'left':
            self.navigate(-1)
        elif event.key == 'y':
            self.mark('good')
        elif event.key == 'n':
            self.mark('bad')
        elif event.key == 'u':
            self.mark('uncertain')
        elif event.key == 's':
            self.navigate(1)
        elif event.key == 'q':
            self.save_and_quit()

    def save_and_quit(self):
        """Save reviews and exit."""
        with open(self.review_file, 'w') as f:
            json.dump(self.reviews, f, indent=2)

        print(f"\n{'='*60}")
        print(f"REVIEW SUMMARY")
        print(f"{'='*60}")
        print(f"Total reviewed: {len(self.reviews)}/{len(self.metadata)}")

        if self.reviews:
            good = sum(1 for v in self.reviews.values() if v == 'good')
            bad = sum(1 for v in self.reviews.values() if v == 'bad')
            uncertain = sum(1 for v in self.reviews.values()
                            if v == 'uncertain')

            print(f"  Good: {good} ({good/len(self.reviews)*100:.1f}%)")
            print(f"  Bad: {bad} ({bad/len(self.reviews)*100:.1f}%)")
            print(
                f"  Uncertain: {uncertain} ({uncertain/len(self.reviews)*100:.1f}%)")

        print(f"\nReviews saved to: {self.review_file}")
        plt.close()

    def run(self):
        """Start the review process."""
        print("="*60)
        print("IMAGE REVIEW TOOL")
        print("="*60)
        print("\nControls:")
        print("  Arrow keys / Buttons: Navigate")
        print("  Y: Mark as Good")
        print("  N: Mark as Bad")
        print("  U: Mark as Uncertain")
        print("  S: Skip")
        print("  Open iNat: View original observation")
        print("  Q: Save and Quit")
        print("\nLook for:")
        print("  ✓ Forest matching iNat description")
        print("  ✓ Good spatial variation in image")
        print("  ✓ GPS accuracy < 50m")
        print("  ✗ Urban areas, parking lots")
        print("  ✗ Clouds, shadows, artifacts")
        print("  ✗ Water, fields (location errors)")
        print("="*60)

        self.load_current_image()
        plt.show()


def main():
    """Main entry point."""
    metadata_file = 'data/ee_imagery/image_patches_64x64/inat_yew_image_metadata.csv'
    image_dir = 'data/ee_imagery/image_patches_64x64'

    reviewer = ImageReviewer(metadata_file, image_dir)
    reviewer.run()


if __name__ == '__main__':
    main()
