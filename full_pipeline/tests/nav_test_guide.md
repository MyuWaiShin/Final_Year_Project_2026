# Navigation Accuracy Testing — Detailed Guide

## Experimental Design

| | |
|---|---|
| **Controls (fixed)** | Scan pose height (the fixed one used in the real pipeline), robot arm, camera, manual focus, ArUco marker ID |
| **Variable** | Object pose (12 positions on the A3 grid, 100 mm pitch) |
| **Trials** | 12 (one per grid circle) — ArUco mode + YOLO mode separately |

> The test uses only the **scan height used in the real pipeline**. Scan height is NOT varied.

---

## How to Run

```bash
cd full_pipeline

# ArUco mode (normal — tag visible on bottle)
python -m tests.test_nav --mode aruco

# YOLO mode (cover the ArUco tag with card)
python -m tests.test_nav --mode yolo

# Resume from a specific trial if interrupted
python -m tests.test_nav --mode aruco --start 5
```

---

## What Gets Saved

Each trial writes one row to `tests/results/nav_grid_results.csv`:

| Column | What it is |
|--------|------------|
| `trial_id` | Trial number (1–12) |
| `grid_pos` | Grid circle label (P1–P12) |
| `grid_cx_mm` | Circle X from A3 corner (mm) |
| `grid_cy_mm` | Circle Y from A3 corner (mm) |
| `detection_mode` | `aruco` or `yolo_only` |
| `hover_x/y/z` | TCP position after navigate() (metres, base frame) |
| `pick_z` | Z the robot descended to (hover_z − 70 mm) |
| `aligned` | **1** = gripper visually over bottle, **0** = misaligned, **−1** = skipped |
| `notes` | Free text per trial |

**Analyse:**
```bash
python -m tests.analyse nav
```


Measure how precisely the robot's TCP hovers above the actual object after `navigate.py` completes. This validates the full chain: ArUco PnP → hand-eye transform `T_tag2base` → calibration offsets → hover move → horizontal centring correction.

---

## What's Being Measured

| Symbol | Meaning |
|--------|---------|
| `hover_xy` | TCP XY position after navigate() completes (read from robot state) |
| `actual_xy` | True tag position measured with a ruler from your table reference point |
| **XY error** | `√((hover_x − actual_x)² + (hover_y − actual_y)²)` in mm |

---

## Equipment You Need

- **ArUco marker ID 13** (the one already on your object, or a spare mounted flat on a board)
- **Steel rule or calipers**, measured from a fixed corner of the table
- **Notepad** (or trust the script to log for you — it will prompt)
- Robot with arm free to move, E-stop to hand

---

## Step-by-Step Protocol

### 1. Define Your Table Reference Point

Pick a fixed corner of the table as origin (0, 0). Mark it clearly.  
All your "actual" measurements will be distances from this point *in the robot base frame X and Y directions* (you need to align your ruler axes to the robot base).

> **Tip:** Place the robot at its initial position, move the TCP to the reference corner manually and note the base-frame XY from the robot teach pendant. This gives you the offset to convert ruler measurements to base-frame coordinates.

---

### 2. Set Up Your Test Grid

Lay out **15–20 positions** on the table where you'll place the tag. A 4×5 grid with 50 mm spacing works well. Mark them with tape crosses or a printed grid.

Suggested grid (relative to reference, in mm):

```
  ┌───────────────────────────────┐
  │  P1    P2    P3    P4         │
  │  P5    P6    P7    P8   ...   │
  │  P9    P10   P11   P12        │
  │  P13   P14   P15   P16        │
  └───────────────────────────────┘
       ↑ 50mm spacing
```

Record the `actual_x, actual_y` (in metres, base frame) for each position in a reference table before you start. You'll type them in when prompted.

---

### 3. Run the Test Script

```bash
cd full_pipeline
python -m tests.test_nav_grid --mode aruco
```

Each trial:
1. Script says `Place bottle on circle P1 …` → press ENTER
2. `navigate()` runs — robot detects ArUco, hovers, centres
3. Robot descends 70 mm (gripper stays **open**, no grasp)
4. Camera window: check alignment → **ENTER** = pass | **N + ENTER** = fail
5. Robot rises → loops to P2

For YOLO mode, cover the ArUco tag:
```bash
python -m tests.test_nav_grid --mode yolo
```

---

### 4. Run Trials — ArUco Mode (15–20 trials)

- Place tag at a grid position (tag face-up, flat on table)
- Press ENTER in the terminal
- Watch the robot navigate and hover
- When prompted, enter the `actual_x` and `actual_y` (metres)
- Enter any notes (e.g. "camera auto-focus flickered", "tag at edge of FOV")
- Repeat across all grid positions

> **Vary:** cover different areas of the workspace — near-centre should be most accurate, edges may degrade.

---

### 5. Run Trials — YOLO-Only Fallback Mode (10–15 trials)

To force YOLO-only mode, **cover the ArUco marker** with a piece of card while the object is still visible to YOLO detection.

- In `navigate.py`, the system falls back to YOLO + OAK-D stereo depth when ArUco is not detected
- Run the same procedure as above
- The `detection_mode` column in the CSV will show `"yolo_only"`

> **Note:** YOLO-only has lower expected accuracy because it relies on depth estimation rather than PnP. This comparison is a key result.

---

### 6. Compare Yaw Alignment (Optional, Qualitative)

`navigate.py` rotates the EEF to match the tag's yaw. After hovering:
- Note whether the gripper jaws are visually aligned with the object's long axis
- Score each trial: ✓ (aligned) / ~ (rough) / ✗ (misaligned)
- Add to the notes field

---

### 7. Analyse Results

```bash
python -m tests.analyse nav
```

This produces:

| Figure | What it shows |
|--------|---------------|
| `nav_scatter.png` | Hover TCP vs. ground truth for all trials, error lines |
| `nav_error_dist.png` | Box plot by mode + CDF with P50/P90 markers |
| `nav_error_vs_trial.png` | Error over trial number (drift/warm-up check) |

Console output includes:
- Mean, median, std, P90 error per detection mode
- Mann-Whitney U test comparing ArUco vs YOLO-only (requires ≥5 trials each)

---

## Interpreting Results

| Error range | Interpretation |
|-------------|---------------|
| < 5 mm | Excellent — gripper should reliably centre over object |
| 5–15 mm | Acceptable — centring correction loop should compensate |
| 15–30 mm | Marginal — calibration offsets (`CALIB_X_OFFSET_M`, `CALIB_Y_OFFSET_M`) should be tuned |
| > 30 mm | Poor — hand-eye calibration likely needs redoing |

The **centring correction** in navigate adds up to 3 incremental moves to bring the tag within 40px of frame centre. If the initial hover error is large, check whether the centring loop is actually correcting it (watch the `centering_iters` column — consistently hitting 3 means it's working hard).

---

## What to Report

In your dissertation/report, you'll want:

1. **Table:** Mean ± std error for ArUco mode vs YOLO-only mode
2. **Figure 1:** Scatter plot (nav_scatter.png) — where are misses concentrated?
3. **Figure 2:** CDF (nav_error_dist.png) — what % of trials are under 5mm? 10mm?
4. **Figure 3:** Error vs. trial to show consistency (no drift)
5. **Key claim:** "The ArUco-based navigation achieved a median XY error of X mm (P90 = Y mm), sufficient for reliable grasp."

---

## Tuning Guidance (if errors are large)

| Problem | Fix |
|---------|-----|
| Consistent X bias | Adjust `CALIB_X_OFFSET_M` in `navigate.py` |
| Consistent Y bias | Adjust `CALIB_Y_OFFSET_M` |
| Large random error | Re-run `Calibration/eye_in_hand/save_calibration.py` |
| YOLO-only much worse | Expected — ArUco PnP is more accurate than depth+YOLO |
| Edge-of-workspace worse | Consider constraining the scan pose sweep to keep tag in FOV |
