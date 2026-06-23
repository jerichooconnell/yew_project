---
name: rebuild-paper-pdf
description: Render docs/PAPER_FINAL_DRAFT.md to docs/PAPER_FINAL_DRAFT.pdf. Use whenever the paper markdown changes and a fresh PDF is needed.
---

# Rebuild the paper PDF

The paper source of truth is `docs/PAPER_FINAL_DRAFT.md`. Render it with:

```bash
/home/jericho/anaconda3/bin/python scripts/paper/render_paper_pdf.py
```

Notes:
- Uses **anaconda base** python (has `weasyprint` + `markdown`); the `yew_pytorch`
  env does NOT have weasyprint. pandoc is not installed.
- The script converts the few `$$…$$` LaTeX equations to HTML best-effort and styles
  tables/headings with academic CSS. `TODO:` markers render as highlighted code.
- Figures are NOT embedded — the markdown carries a caption list only.
- After rendering, sanity-check: `pdfinfo docs/PAPER_FINAL_DRAFT.pdf | grep Pages`
  and grep the PDF text for stray `$$`, `\frac`, or `\text` (should be none):
  `pdftotext docs/PAPER_FINAL_DRAFT.pdf - | grep -E '\\\\frac|\\\\text|\$\$'`
