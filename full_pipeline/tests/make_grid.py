"""
tests/make_grid.py
------------------
Generates a printable A3 navigation grid as an HTML file.

Output: tests/results/nav_grid_A3.html

HOW TO PRINT
------------
1. Open nav_grid_A3.html in Chrome or Edge
2. Ctrl+P
3. Paper: A3  |  Orientation: Landscape  |  Margins: None  |  Scale: 100%
4. Print

The @page CSS forces A3 landscape and zero margins so circles print at
exactly 50 mm diameter regardless of browser zoom.

The top-left corner REF (0,0) is the robot reference point.

Usage
-----
    cd full_pipeline
    python -m tests.make_grid
"""

from pathlib import Path

# ── Grid geometry (mm) ───────────────────────────────────────────────────────
A3_W, A3_H = 420, 297       # landscape A3
DIAM        = 50             # circle diameter = bottle footprint (mm)
GAP         = 100            # edge-to-edge gap (mm)
STEP        = DIAM + GAP     # centre-to-centre = 150 mm

MARGIN_MM   = DIAM / 2       # 25 mm — minimum border from paper edge
COLS        = int((A3_W - 2 * MARGIN_MM) // STEP) + 1   # = 3
ROWS        = int((A3_H - 2 * MARGIN_MM) // STEP) + 1   # = 2

# Centre the grid on the paper
GRID_W      = (COLS - 1) * STEP + DIAM
GRID_H      = (ROWS - 1) * STEP + DIAM
OX          = (A3_W - GRID_W) / 2    # left offset
OY          = (A3_H - GRID_H) / 2    # top  offset

OUTPUT = Path(__file__).resolve().parent / "results" / "nav_grid_A3.html"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# ── Circle positions ──────────────────────────────────────────────────────────
positions = []
for row in range(ROWS):
    for col in range(COLS):
        cx = OX + DIAM / 2 + col * STEP
        cy = OY + DIAM / 2 + row * STEP
        positions.append((cx, cy, row * COLS + col + 1))


# ── Build HTML ────────────────────────────────────────────────────────────────
def make_html():
    r   = DIAM / 2
    els = []
    for cx, cy, label in positions:
        shade = "#E8F4FE" if ((label - 1) // COLS + (label - 1) % COLS) % 2 == 0 \
                else "#FFF8E8"
        els += [
            f'  <!-- P{label} ({cx:.1f},{cy:.1f})mm -->',
            f'  <circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r}"'
            f' fill="{shade}" stroke="#1A73E8" stroke-width="0.8"/>',
            f'  <line x1="{cx-7:.1f}" y1="{cy}" x2="{cx+7:.1f}" y2="{cy}"'
            f' stroke="#1A73E8" stroke-width="0.35" stroke-dasharray="2,2"/>',
            f'  <line x1="{cx}" y1="{cy-7:.1f}" x2="{cx}" y2="{cy+7:.1f}"'
            f' stroke="#1A73E8" stroke-width="0.35" stroke-dasharray="2,2"/>',
            f'  <text x="{cx:.1f}" y="{cy - r - 2:.1f}" text-anchor="middle"'
            f' font-size="6" font-weight="bold" fill="#1A73E8"'
            f' font-family="sans-serif">P{label}</text>',
            f'  <text x="{cx:.1f}" y="{cy + r + 7:.1f}" text-anchor="middle"'
            f' font-size="5" fill="#555" font-family="monospace">'
            f'({cx:.0f},{cy:.0f})mm</text>',
        ]

    circles = '\n'.join(els)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Navigation Grid A3</title>
<style>
  @page {{ size: 420mm 297mm; margin: 0; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ width: 420mm; height: 297mm; background: white; overflow: hidden; }}
  svg  {{ display: block; width: 420mm; height: 297mm; }}
</style>
</head>
<body>
<svg xmlns="http://www.w3.org/2000/svg"
     viewBox="0 0 {A3_W} {A3_H}" width="{A3_W}mm" height="{A3_H}mm">

  <rect width="{A3_W}" height="{A3_H}" fill="white"/>
  <rect width="{A3_W}" height="{A3_H}" fill="none" stroke="#CCC" stroke-width="0.4"/>

  <!-- REF corner -->
  <line x1="0" y1="0" x2="20" y2="0" stroke="#E53935" stroke-width="1.5"/>
  <line x1="0" y1="0" x2="0" y2="20" stroke="#E53935" stroke-width="1.5"/>
  <text x="2" y="13" font-size="6" fill="#E53935" font-family="sans-serif"
        font-weight="bold">REF (0,0)</text>

{circles}

  <!-- Footer -->
  <text x="{A3_W/2:.1f}" y="{A3_H - 5:.1f}" text-anchor="middle"
        font-size="5.5" fill="#999" font-family="sans-serif">
    Navigation Grid  |  {COLS}x{ROWS} = {COLS*ROWS} positions  |
    D={DIAM}mm  |  {STEP}mm pitch ({GAP}mm gap)  |  Top-left corner = REF (0,0)
  </text>

</svg>
</body>
</html>"""


if __name__ == "__main__":
    OUTPUT.write_text(make_html(), encoding="utf-8")
    print(f"Saved -> {OUTPUT}")
    print()
    print(f"Grid : {COLS}x{ROWS} = {COLS*ROWS} positions  |  "
          f"{STEP}mm pitch  |  centred on A3")
    print()
    print("Positions (mm from top-left corner of paper):")
    print(f"  {'Label':6}  {'CX':>8}  {'CY':>8}")
    for cx, cy, lbl in positions:
        print(f"  P{lbl:<5}  {cx:>8.1f}  {cy:>8.1f}")
    print()
    print("PRINT: Open nav_grid_A3.html in Chrome -> Ctrl+P")
    print("       Paper: A3  | Landscape  | Margins: None  | Scale: 100%")
