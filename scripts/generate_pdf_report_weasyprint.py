#!/usr/bin/env python3
"""
Generate Professional PDF Report
=================================

Converts the yew distribution markdown report to a formatted PDF.

Author: GitHub Copilot
Date: November 7, 2025
"""

import markdown
from weasyprint import HTML, CSS
from pathlib import Path
import re


def preprocess_markdown(md_content):
    """
    Preprocess markdown to handle image paths and formatting.

    Args:
        md_content: Raw markdown string

    Returns:
        Processed markdown string
    """
    # Convert relative image paths to absolute paths
    md_content = re.sub(
        r'!\[(.*?)\]\((results/figures/.*?)\)',
        r'![\1](file://' + str(Path.cwd()) + r'/\2)',
        md_content
    )

    return md_content


def create_pdf(input_md, output_pdf):
    """
    Convert markdown to PDF with professional styling.

    Args:
        input_md: Path to markdown file
        output_pdf: Path to output PDF
    """
    print(f"Reading {input_md.name}...")

    # Read markdown
    with open(input_md, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Preprocess
    md_content = preprocess_markdown(md_content)

    # Convert to HTML
    print("Converting to HTML...")
    md = markdown.Markdown(extensions=[
        'extra',
        'codehilite',
        'tables',
        'toc'
    ])
    html_content = md.convert(md_content)

    # Wrap in HTML document with styling
    html_doc = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Pacific Yew Distribution Report</title>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    # Custom CSS for professional appearance
    css = CSS(string="""
        @page {
            size: letter;
            margin: 1in;
            @bottom-center {
                content: counter(page);
                font-size: 9pt;
                color: #666;
            }
        }
        
        body {
            font-family: 'Georgia', 'Times New Roman', serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #333;
        }
        
        h1 {
            font-size: 20pt;
            font-weight: bold;
            color: #2c5f2d;
            border-bottom: 3px solid #2c5f2d;
            padding-bottom: 10px;
            margin-top: 30px;
            margin-bottom: 20px;
            page-break-after: avoid;
        }
        
        h2 {
            font-size: 16pt;
            font-weight: bold;
            color: #2c5f2d;
            border-bottom: 2px solid #97bc62;
            padding-bottom: 8px;
            margin-top: 24px;
            margin-bottom: 16px;
            page-break-after: avoid;
        }
        
        h3 {
            font-size: 13pt;
            font-weight: bold;
            color: #2c5f2d;
            margin-top: 18px;
            margin-bottom: 12px;
            page-break-after: avoid;
        }
        
        h4 {
            font-size: 11pt;
            font-weight: bold;
            color: #555;
            margin-top: 14px;
            margin-bottom: 10px;
        }
        
        p {
            text-align: justify;
            margin-bottom: 12px;
        }
        
        ul, ol {
            margin-left: 25px;
            margin-bottom: 12px;
        }
        
        li {
            margin-bottom: 6px;
        }
        
        strong {
            font-weight: bold;
            color: #2c5f2d;
        }
        
        em {
            font-style: italic;
        }
        
        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
            border: 1px solid #ddd;
            padding: 5px;
            background: white;
            page-break-inside: avoid;
        }
        
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            font-size: 10pt;
            page-break-inside: avoid;
        }
        
        th {
            background-color: #2c5f2d;
            color: white;
            font-weight: bold;
            padding: 10px;
            text-align: left;
            border: 1px solid #2c5f2d;
        }
        
        td {
            padding: 8px;
            border: 1px solid #ddd;
        }
        
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        
        code {
            font-family: 'Courier New', monospace;
            background-color: #f5f5f5;
            padding: 2px 4px;
            border-radius: 3px;
            font-size: 10pt;
        }
        
        hr {
            border: none;
            border-top: 1px solid #ddd;
            margin: 30px 0;
        }
        
        blockquote {
            border-left: 4px solid #97bc62;
            padding-left: 20px;
            margin-left: 0;
            color: #666;
            font-style: italic;
        }
        
        .page-break {
            page-break-before: always;
        }
    """)

    # Generate PDF
    print("Generating PDF...")
    HTML(string=html_doc, base_url=str(Path.cwd())).write_pdf(
        output_pdf,
        stylesheets=[css]
    )

    print(f"\n✓ PDF created: {output_pdf}")

    # Show file size
    size_mb = output_pdf.stat().st_size / (1024 * 1024)
    print(f"  Size: {size_mb:.2f} MB")
    print(f"  Location: {output_pdf.absolute()}")


def main():
    """Main execution."""
    print("="*80)
    print("GENERATING PDF REPORT")
    print("="*80)
    print()

    input_md = Path('SUMMARY_FOR_YEW_EXPERT.md')
    output_pdf = Path('Pacific_Yew_Distribution_Report.pdf')

    if not input_md.exists():
        print(f"✗ Error: {input_md} not found")
        return

    try:
        create_pdf(input_md, output_pdf)
        print("\n✓ Done!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
