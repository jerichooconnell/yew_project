#!/usr/bin/env python3
"""Render docs/PAPER_FINAL_DRAFT.md -> docs/PAPER_FINAL_DRAFT.pdf with academic styling."""
import re
from pathlib import Path
import markdown
from weasyprint import HTML, CSS

ROOT = Path("/home/jericho/yew_project")
SRC = ROOT / "docs" / "PAPER_FINAL_DRAFT.md"
OUT = ROOT / "docs" / "PAPER_FINAL_DRAFT.pdf"

# Paper caption number -> generated figure file. The generator uses an older
# fig1..fig19 naming that does NOT match the caption numbers (see CLAUDE.md).
_PFIG = ROOT / "results" / "figures" / "paper"
_FIG = ROOT / "results" / "figures"
FIGURE_MAP = {
    "Figure 1": _PFIG / "fig1_zone_comparison.png",
    "Figure 2": _PFIG / "fig12_suppression_waterfall.png",
    "Figure 3": _PFIG / "fig2_cwh_subzones.png",
    "Figure 4": _PFIG / "fig5_percent_decline.png",
    "Figure 5": _PFIG / "fig18_study_area_map.png",
    "Figure 6": _PFIG / "fig14_model_performance.png",
    "Figure 7": _FIG / "tree_size_distribution.png",
    "Figure S1": _PFIG / "fig3_ich_subzones.png",
    "Figure S2": _PFIG / "fig4_cdf_detail.png",
    "Figure S3": _PFIG / "fig6_overall_pie.png",
    "Figure S4": _PFIG / "fig7_three_zone_comparison.png",
    "Figure S5": _PFIG / "fig8_logging_vs_decline.png",
    "Figure S6": _PFIG / "fig9_og_yew_rate.png",
    "Figure S7": _PFIG / "fig10_fire_decades.png",
    "Figure S8": _PFIG / "fig11_example_tiles.png",
    "Figure S9": _PFIG / "fig13_logging_age_classes.png",
    "Figure S10": _PFIG / "fig15_threats_summary.png",
    "Figure S11": _PFIG / "fig16_heatmap.png",
    "Figure S12": _PFIG / "fig17_og_logged_ratio.png",
    "Figure S13": _PFIG / "fig19_zone_waterfall.png",
}


def embed_figures(md: str) -> str:
    """Insert each figure image directly above its '**Figure N.**' caption."""
    def repl(m):
        label = m.group(1)                       # e.g. "Figure 1" / "Figure S3"
        path = FIGURE_MAP.get(label)
        if not path or not path.exists():
            return m.group(0)
        img = (f'<figure class="paper-fig"><img src="file://{path}"/></figure>\n\n')
        return img + m.group(0)
    # Caption lines look like:  **Figure 1.** Caption text…
    return re.sub(r"^\*\*(Figure S?\d+)\.\*\*", repl, md, flags=re.M)


def latex_to_html(expr: str) -> str:
    """Best-effort conversion of simple inline/display LaTeX to HTML."""
    s = expr
    # \text{...} -> ... ; underscores inside \text are literal, protect them
    s = re.sub(r"\\text\{([^}]*)\}",
               lambda m: m.group(1).replace(r"\_", "\0").replace("_", "\0"), s)
    # any remaining escaped underscore \_ -> literal (protect from subscript)
    s = s.replace(r"\_", "\0")
    # \frac{A}{B} -> (A) / (B)
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1) / (\2)", s)
    # \clip / \operatorname not used; \propto, \pi, \quad, \cdot
    s = s.replace(r"\propto", "∝").replace(r"\pi", "π")
    s = s.replace(r"\cdot", "·").replace(r"\times", "×")
    s = s.replace(r"\quad", "    ").replace(r"\,", " ").replace(r"\\", " ")
    s = s.replace(r"\left", "").replace(r"\right", "")
    # superscripts ^{...} or ^x
    s = re.sub(r"\^\{([^}]*)\}", r"<sup>\1</sup>", s)
    s = re.sub(r"\^(\w)", r"<sup>\1</sup>", s)
    # subscripts _{...} or _x
    s = re.sub(r"_\{([^}]*)\}", r"<sub>\1</sub>", s)
    s = re.sub(r"_(\w)", r"<sub>\1</sub>", s)
    s = s.replace("{", "").replace("}", "")
    # restore protected literal underscores
    s = s.replace("\0", "_")
    return s.strip()


def preprocess(md: str) -> str:
    md = embed_figures(md)
    # absolute image paths (none expected, but keep parity with existing script)
    md = re.sub(r"!\[(.*?)\]\((results/figures/.*?)\)",
                r"![\1](file://" + str(ROOT) + r"/\2)", md)
    # display math $$...$$ (possibly multiline) -> centered div
    def disp(m):
        return ('\n\n<div class="math-display">' +
                latex_to_html(m.group(1).strip()) + '</div>\n\n')
    md = re.sub(r"\$\$(.+?)\$\$", disp, md, flags=re.S)
    # inline math $...$ -> italic span (avoid matching currency; require no newline)
    def inl(m):
        return '<span class="math-inline">' + latex_to_html(m.group(1)) + '</span>'
    md = re.sub(r"\$([^$\n]+?)\$", inl, md)
    return md


CSS_TEXT = """
@page { size: A4; margin: 22mm 20mm 20mm 20mm;
        @bottom-center { content: counter(page); font-size: 9pt; color: #666; } }
body { font-family: 'Georgia','Times New Roman',serif; font-size: 10.5pt;
       line-height: 1.45; color: #1a1a1a; text-align: justify; }
h1 { font-size: 18pt; line-height: 1.25; text-align: left; color: #14532d;
     border-bottom: 2px solid #14532d; padding-bottom: 6px; margin: 0 0 14px; }
h2 { font-size: 14pt; color: #14532d; margin: 20px 0 8px; border-bottom: 1px solid #cbd5cb;
     padding-bottom: 3px; }
h3 { font-size: 12pt; color: #1f3a26; margin: 14px 0 6px; }
h4 { font-size: 10.8pt; color: #333; margin: 11px 0 4px; font-style: italic; }
p { margin: 0 0 8px; }
strong { color: #111; }
hr { border: none; border-top: 1px solid #ccc; margin: 16px 0; }
a { color: #1d4ed8; text-decoration: none; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 12px; font-size: 9pt;
        text-align: left; }
th, td { border: 1px solid #b8c4b8; padding: 4px 7px; vertical-align: top; }
th { background: #e8f0e8; color: #14532d; }
tr:nth-child(even) td { background: #f6f9f6; }
code { background: #fff3cd; color: #8a6d00; padding: 1px 4px; border-radius: 3px;
       font-family: 'Courier New',monospace; font-size: 9pt; }
figure.paper-fig { margin: 10px 0 4px; text-align: center; break-inside: avoid; }
figure.paper-fig img { max-width: 100%; max-height: 200mm; height: auto; }
.math-display { text-align: center; font-style: italic; font-family: 'Cambria Math','Georgia',serif;
                margin: 10px 0; font-size: 11pt; }
.math-inline { font-style: italic; }
sup, sub { font-size: 70%; }
"""


def main():
    md_text = preprocess(SRC.read_text(encoding="utf-8"))
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
    )
    html = (f"<!DOCTYPE html><html><head><meta charset='utf-8'></head>"
            f"<body>{html_body}</body></html>")
    HTML(string=html, base_url=str(ROOT)).write_pdf(
        str(OUT), stylesheets=[CSS(string=CSS_TEXT)])
    print(f"Wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
