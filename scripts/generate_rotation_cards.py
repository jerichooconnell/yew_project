#!/usr/bin/env python3
"""
Generate half-page rotation cards PDF for the hosted table-rotation event.

Casual-wedding style: warm ivory card, elegant serif name, simple numbered
list showing each person's table colour per round.  Table leaders are
shown as colours only — no names on the cards.

    Will → Red    Anne → Orange    Karen → Yellow
    Sophia → Green    Aaron → Blue    Lilli → Purple

Output: results/rotation_cards.pdf
    12 pages of 2 cards each (24 personal cards) + 1 colour reference page.
    Cut along the dashed midline of each sheet.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import HexColor, white
from reportlab.pdfgen import canvas as rl_canvas

# ── Page / card geometry ──────────────────────────────────────────────────────
PAGE_W, PAGE_H = letter   # 612 × 792 pt
CARD_H = PAGE_H / 2       # 396 pt  (5.5 in — half page)
CARD_W = PAGE_W            # 612 pt

# ── Colour palette (warm, muted — wedding-appropriate) ────────────────────────
TABLE_COLORS = {
    "Red":    HexColor("#9E3030"),
    "Orange": HexColor("#C4702A"),
    "Yellow": HexColor("#A8922A"),
    "Green":  HexColor("#3D6B4F"),
    "Blue":   HexColor("#2B5F8A"),
    "Purple": HexColor("#6E4A8A"),
}

TABLE_HOSTS = ["Red", "Orange", "Yellow", "Green", "Blue", "Purple"]
HOST_NAMES  = ["Will", "Anne", "Karen", "Sophia", "Aaron", "Lilli"]
ROUND_WORDS = ["one", "two", "three", "four", "five", "six"]

# ── Card colour tokens ────────────────────────────────────────────────────────
BG     = HexColor("#FEFCF7")   # warm ivory
GOLD   = HexColor("#C9A84C")   # decorative gold
DARK   = HexColor("#2C1810")   # near-black warm (primary text)
MEDIUM = HexColor("#8B6F4E")   # warm brown (secondary text)
LIGHT  = HexColor("#E4D9C8")   # very light warm (dividers / dots)

# ── Per-person schedule: 6 rounds → colour name ──────────────────────────────
SCHEDULE = {
    "Ellen":    ["Orange", "Yellow", "Blue",   "Purple", "Blue",   "Purple"],
    "Sam":      ["Red",    "Yellow", "Purple", "Yellow", "Orange", "Yellow"],
    "Lana":     ["Red",    "Red",    "Orange", "Blue",   "Green",  "Purple"],
    "Art":      ["Yellow", "Blue",   "Purple", "Blue",   "Red",    "Orange"],
    "Peter":    ["Blue",   "Orange", "Green",  "Red",    "Yellow", "Purple"],
    "Esther":   ["Blue",   "Blue",   "Yellow", "Yellow", "Green",  "Green"],
    "Joan":     ["Green",  "Yellow", "Yellow", "Orange", "Red",    "Blue"],
    "Dan O":    ["Blue",   "Green",  "Red",    "Orange", "Orange", "Red"],
    "Noah":     ["Purple", "Blue",   "Green",  "Orange", "Blue",   "Yellow"],
    "Rachel G": ["Yellow", "Red",    "Green",  "Purple", "Purple", "Green"],
    "Char":     ["Orange", "Purple", "Green",  "Blue",   "Orange", "Blue"],
    "Jericho":  ["Purple", "Orange", "Blue",   "Yellow", "Red",    "Red"],
    "Kirsten":  ["Green",  "Green",  "Orange", "Green",  "Blue",   "Green"],
    "Solomon":  ["Orange", "Red",    "Purple", "Green",  "Yellow", "Red"],
    "Misty":    ["Purple", "Green",  "Purple", "Red",    "Green",  "Blue"],
    "Aja":      ["Orange", "Orange", "Orange", "Orange", "Purple", "Orange"],
    "Isobel":   ["Red",    "Blue",   "Blue",   "Green",  "Purple", "Blue"],
    "Shane":    ["Yellow", "Purple", "Yellow", "Red",    "Blue",   "Red"],
    "Sora":     ["Blue",   "Purple", "Orange", "Purple", "Red",    "Yellow"],
    "Isaac":    ["Purple", "Yellow", "Red",    "Blue",   "Yellow", "Green"],
    "Jordan":   ["Red",    "Green",  "Yellow", "Purple", "Yellow", "Orange"],
    "Dan S":    ["Green",  "Red",    "Blue",   "Red",    "Orange", "Orange"],
    "Rachel":   ["Yellow", "Orange", "Red",    "Green",  "Green",  "Yellow"],
    "Cathe":    ["Green",  "Purple", "Red",    "Yellow", "Purple", "Purple"],
}


# ── Drawing helpers ─────────────────────────────────────────────────────────

def _petal(cv, cx, cy, r, horiz=False):
    """Filled almond/petal bezier shape (vertical or horizontal)."""
    k = 0.55  # bezier approximation of a narrow oval
    p = cv.beginPath()
    if not horiz:
        p.moveTo(cx, cy + r)
        p.curveTo(cx + r*k, cy + r*k, cx + r*k, cy - r*k, cx, cy - r)
        p.curveTo(cx - r*k, cy - r*k, cx - r*k, cy + r*k, cx, cy + r)
    else:
        p.moveTo(cx + r, cy)
        p.curveTo(cx + r*k, cy + r*k, cx - r*k, cy + r*k, cx - r, cy)
        p.curveTo(cx - r*k, cy - r*k, cx + r*k, cy - r*k, cx + r, cy)
    p.close()
    cv.drawPath(p, fill=1, stroke=0)


def _flower(cv, cx, cy, r=7):
    """Four-petal flower (N/S/E/W petals) with a small dot centre."""
    cv.setFillColor(GOLD)
    _petal(cv, cx, cy, r, horiz=False)
    _petal(cv, cx, cy, r, horiz=True)
    cv.setFillColor(HexColor("#A07830"))
    cv.circle(cx, cy, r * 0.22, fill=1, stroke=0)


def _coloured_flower(cv, cx, cy, color, r=9):
    """Four-petal flower in a given fill colour, gold outline and gold centre."""
    k = 0.55
    cv.setStrokeColor(GOLD)
    cv.setLineWidth(0.6)
    cv.setFillColor(color)
    for horiz in (False, True):
        p = cv.beginPath()
        if not horiz:
            p.moveTo(cx, cy + r)
            p.curveTo(cx + r*k, cy + r*k, cx + r*k, cy - r*k, cx, cy - r)
            p.curveTo(cx - r*k, cy - r*k, cx - r*k, cy + r*k, cx, cy + r)
        else:
            p.moveTo(cx + r, cy)
            p.curveTo(cx + r*k, cy + r*k, cx - r*k, cy + r*k, cx - r, cy)
            p.curveTo(cx - r*k, cy - r*k, cx + r*k, cy - r*k, cx + r, cy)
        p.close()
        cv.drawPath(p, fill=1, stroke=1)
    # gold centre dot
    cv.setFillColor(GOLD)
    cv.circle(cx, cy, r * 0.22, fill=1, stroke=0)


def _sprig(cv, cx, cy, size=5):
    """Two small curved leaves branching upward from (cx, cy)."""
    cv.setFillColor(GOLD)
    for side in (-1, 1):
        p = cv.beginPath()
        p.moveTo(cx, cy)
        p.curveTo(cx + side*size, cy + size*0.5,
                  cx + side*size*1.5, cy + size,
                  cx + side*size*0.5, cy + size*1.3)
        p.curveTo(cx + side*size*0.2, cy + size*0.7,
                  cx, cy + size*0.3, cx, cy)
        p.close()
        cv.drawPath(p, fill=1, stroke=0)


def _vine_line(cv, x1, x2, y, amp=3):
    """Gentle wavy gold line from x1 to x2 at height y."""
    cv.setStrokeColor(GOLD)
    cv.setLineWidth(0.5)
    segs = 10
    sw   = (x2 - x1) / segs
    path = cv.beginPath()
    path.moveTo(x1, y)
    for i in range(segs):
        sx = x1 + i * sw
        ex = sx + sw
        sign = 1 if i % 2 == 0 else -1
        path.curveTo(sx + sw*0.33, y + sign*amp,
                     sx + sw*0.67, y + sign*amp,
                     ex, y)
    cv.drawPath(path, stroke=1, fill=0)


def _floral_border(cv, abs_y, x1=32, x2=None):
    """Vine line with flowers at ends and centre, sprigs between."""
    if x2 is None:
        x2 = CARD_W - 32
    mid   = (x1 + x2) / 2
    left  = x1 + 14
    right = x2 - 14
    midL  = (left + mid) / 2
    midR  = (mid + right) / 2

    _vine_line(cv, left, right, abs_y)
    _flower(cv, left,  abs_y, r=8)
    _flower(cv, mid,   abs_y, r=9)
    _flower(cv, right, abs_y, r=8)
    _sprig(cv, midL, abs_y - 3, size=5)
    _sprig(cv, midR, abs_y - 3, size=5)


# ── Card drawing ──────────────────────────────────────────────────────────────

def draw_personal_card(cv, name: str, rounds: list, y0: float):
    """
    Draw one half-page personal card.
    y0 = bottom-left y of the card in page coordinates.
    """
    # Background
    cv.setFillColor(BG)
    cv.rect(0, y0, CARD_W, CARD_H, fill=1, stroke=0)

    # Top floral border
    _floral_border(cv, y0 + CARD_H - 26)

    # Person's name  – large italic serif for a script-like feel
    cv.setFillColor(DARK)
    cv.setFont("Times-BoldItalic", 46)
    cv.drawCentredString(CARD_W / 2, y0 + CARD_H - 82, name)

    # Subtitle
    # Divider below header
    cv.setStrokeColor(LIGHT)
    cv.setLineWidth(0.6)
    cv.line(36, y0 + CARD_H - 118, CARD_W - 36, y0 + CARD_H - 118)

    # ── Six round rows ────────────────────────────────────────────────────────
    ROW_TOP  = CARD_H - 146   # first row baseline, measured from card bottom
    ROW_GAP  = 33             # pitch between rows
    LABEL_X  = 58             # "round N" left edge
    DOT_X0   = 198            # leader-dot start x
    DOT_X1   = 412            # leader-dot end x
    CIRCLE_X = 434            # colour indicator circle centre x
    NAME_X   = 454            # colour name text left x

    for i, (word, colour) in enumerate(zip(ROUND_WORDS, rounds)):
        ry = y0 + ROW_TOP - i * ROW_GAP   # baseline for this row

        # "round N" label
        cv.setFillColor(DARK)
        cv.setFont("Times-Italic", 14)
        cv.drawString(LABEL_X, ry, f"round  {word}")

        # Leader dots
        x = DOT_X0
        cv.setFillColor(LIGHT)
        while x <= DOT_X1:
            cv.circle(x, ry + 5, 1.3, fill=1, stroke=0)
            x += 9

        # Colour flower indicator
        c = TABLE_COLORS[colour]
        _coloured_flower(cv, CIRCLE_X, ry + 5, c, r=9)

        # Colour name, drawn in the table colour — sans-serif
        cv.setFillColor(c)
        cv.setFont("Helvetica", 14)
        cv.drawString(NAME_X, ry, colour)

    # Bottom floral border
    _floral_border(cv, y0 + 26)


def draw_reference_card(cv, y0: float):
    """Draw a colour-to-host reference card (one per event, top of last page)."""
    cv.setFillColor(BG)
    cv.rect(0, y0, CARD_W, CARD_H, fill=1, stroke=0)

    _floral_border(cv, y0 + CARD_H - 26)

    cv.setFillColor(DARK)
    cv.setFont("Times-BoldItalic", 38)
    cv.drawCentredString(CARD_W / 2, y0 + CARD_H - 76, "Table Colours")

    cv.setStrokeColor(LIGHT)
    cv.setLineWidth(0.6)
    cv.line(36, y0 + CARD_H - 114, CARD_W - 36, y0 + CARD_H - 114)

    ROW_TOP = CARD_H - 142
    ROW_GAP = 36

    for i, (colour, host) in enumerate(zip(TABLE_HOSTS, HOST_NAMES)):
        ry  = y0 + ROW_TOP - i * ROW_GAP
        c   = TABLE_COLORS[colour]

        # Swatch flower indicator
        _coloured_flower(cv, 88, ry + 7, c, r=11)

        # Colour name
        cv.setFillColor(c)
        cv.setFont("Helvetica-Bold", 15)
        cv.drawString(108, ry, colour)

        # Host name
        cv.setFillColor(DARK)
        cv.setFont("Helvetica", 15)
        cv.drawString(208, ry, f"\u2013  {host}'s table")

    _floral_border(cv, y0 + 26)


def draw_cut_line(cv, y: float):
    """Dashed cut line at the page midpoint with a small centred flower."""
    cv.setStrokeColor(HexColor("#BBBBBB"))
    cv.setLineWidth(0.5)
    cv.setDash([6, 4])
    cv.line(18, y, CARD_W - 18, y)
    cv.setDash([])
    # tiny flower centred on the cut line
    _flower(cv, CARD_W / 2, y, r=5)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "rotation_cards.pdf")

    cv = rl_canvas.Canvas(out_path, pagesize=letter)
    people = list(SCHEDULE.keys())   # 24 people → 12 pages

    for i, name in enumerate(people):
        if i % 2 == 0 and i > 0:
            cv.showPage()

        y0 = CARD_H if i % 2 == 0 else 0
        draw_personal_card(cv, name, SCHEDULE[name], y0)

        if i % 2 == 0:
            draw_cut_line(cv, CARD_H)

    # Page 13: colour reference (top half) + blank matching bottom half
    cv.showPage()
    draw_reference_card(cv, CARD_H)
    draw_cut_line(cv, CARD_H)
    cv.setFillColor(BG)
    cv.rect(0, 0, CARD_W, CARD_H, fill=1, stroke=0)

    cv.showPage()
    cv.save()
    print(f"Saved {len(people)} personal cards + 1 reference  →  {out_path}")


if __name__ == "__main__":
    main()
