#!/usr/bin/env python3
"""
Generate PDF Report from Markdown
==================================

Converts the yew expert summary markdown to PDF format.

Requirements: pip install markdown pdfkit (or use pandoc)

Author: GitHub Copilot
Date: November 7, 2025
"""

import subprocess
import sys
from pathlib import Path


def convert_with_pandoc(input_md, output_pdf):
    """
    Convert markdown to PDF using pandoc.

    Args:
        input_md: Path to input markdown file
        output_pdf: Path to output PDF file
    """
    try:
        # Check if pandoc is installed
        result = subprocess.run(['pandoc', '--version'],
                                capture_output=True, text=True)

        if result.returncode != 0:
            print("Error: pandoc is not installed")
            print("Install with: sudo apt-get install pandoc")
            return False

        print(f"Using pandoc version: {result.stdout.split()[1]}")

        # Convert to PDF with nice formatting
        cmd = [
            'pandoc',
            str(input_md),
            '-o', str(output_pdf),
            '--pdf-engine=pdflatex',
            '-V', 'geometry:margin=1in',
            '-V', 'fontsize=11pt',
            '-V', 'documentclass=article',
            '--toc',
            '--toc-depth=2',
            '--number-sections',
            '--highlight-style=tango'
        ]

        print(f"\nConverting {input_md.name} to PDF...")
        print("This may take a moment...")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"\n✓ PDF created successfully: {output_pdf}")
            return True
        else:
            print(f"\n✗ Conversion failed:")
            print(result.stderr)

            # Check if it's a LaTeX issue
            if 'pdflatex not found' in result.stderr:
                print("\nPandoc needs LaTeX to create PDFs.")
                print(
                    "Install with: sudo apt-get install texlive-latex-base texlive-fonts-recommended")

            return False

    except FileNotFoundError:
        print("✗ pandoc not found")
        print("\nTo install pandoc:")
        print("  Ubuntu/Debian: sudo apt-get install pandoc texlive-latex-base")
        print("  macOS: brew install pandoc basictex")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Main execution."""
    print("="*80)
    print("GENERATING PDF REPORT")
    print("="*80)
    print()

    # Paths
    input_md = Path('SUMMARY_FOR_YEW_EXPERT.md')
    output_pdf = Path('Pacific_Yew_Distribution_Report.pdf')

    # Check input exists
    if not input_md.exists():
        print(f"✗ Input file not found: {input_md}")
        sys.exit(1)

    # Convert
    success = convert_with_pandoc(input_md, output_pdf)

    if success:
        # Show file size
        size_mb = output_pdf.stat().st_size / (1024 * 1024)
        print(f"\nFile size: {size_mb:.2f} MB")
        print(f"Location: {output_pdf.absolute()}")
        print("\n✓ Done!")
    else:
        print("\n✗ PDF generation failed")
        print("\nAlternative: You can manually convert using:")
        print(f"  pandoc {input_md} -o {output_pdf}")
        sys.exit(1)


if __name__ == '__main__':
    main()
