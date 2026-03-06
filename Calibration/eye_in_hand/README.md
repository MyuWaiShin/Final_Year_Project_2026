# Eye-in-Hand Calibration (Checkerboard)

Solves the rigid transform **camera → TCP** for the UR10 + OAK-D Lite.  
Camera is mounted ~40° downward, viewing the gripper and workspace simultaneously.

---

## Files

| File | Purpose |
|---|---|
| `collect_poses.py` | Live capture — jog robot manually, press `C` to record each pose |
| `solve_calibration.py` | Offline solver — runs OpenCV `calibrateHandEye()`, saves result |
| `replay_positions.py` | Auto-replay — moves robot to saved poses for repeatable recalibration |
| `data/poses.json` | Auto-generated: images + TCP poses from a collection session |
| `handeye_calibration.json` | Auto-generated: final calibration result |

---

## Step 1 — Edit config in `collect_poses.py`

Open `collect_poses.py` and set:

```python
BOARD_COLS  = 9      # inner corners along long axis
BOARD_ROWS  = 6      # inner corners along short axis
SQUARE_M    = 0.025  # square size in METRES
ROBOT_IP    = "192.168.8.102"
```

> Count **inner** corners only (not the outer edge). A standard A3 9×6 board has `BOARD_COLS=9, BOARD_ROWS=6`.

---

## Step 2 — Collect poses

```bash
cd Calibration/eye_in_hand
python collect_poses.py
```

**What to do in the live window:**

1. Jog the robot to a new position where the checkerboard is clearly visible
2. Wait for the HUD to show **BOARD: OK ✓** (green)
3. Press **`C`** to capture
4. Repeat from different angles/heights — aim for **20–40 samples**
5. Press **`S`** when done → saves `data/poses.json` + `data/images/`

**Good diversity means:**
- Move the robot laterally (X and Y) across the workspace
- Vary height (Z) — lower and higher
- Tilt 10–20° left/right and forward/backward
- Don't just translate — also rotate the wrist

**Controls:**

| Key | Action |
|---|---|
| `C` | Capture frame + robot pose (only works when board detected) |
| `D` | Delete the last captured sample |
| `S` | Save all samples and exit |
| `Q` | Quit without saving |

---

## Step 3 — Solve

```bash
python solve_calibration.py
```

This runs 5 different hand-eye methods (Tsai, Park, Horaud, Andreff, Daniilidis) and picks the most consistent result.

**Output:** `handeye_calibration.json`

```json
{
  "R_cam2tcp": [[...], [...], [...]],
  "t_cam2tcp": [dx, dy, dz],
  "best_method": "TSAI",
  "board_reproj_mean_px": 1.3,
  "n_samples": 22
}
```

**What to look for:**
- `board_reproj_mean_px` < 2.0 is excellent, < 5.0 is acceptable
- `t_cam2tcp` values should be in the range 0.05–0.20 m (5–20 cm) — matching the physical camera-to-TCP offset
- The script prints a pitch estimate and warns if it's far from ~40°

---

## Step 4 — Recalibrate automatically (replay)

After your first successful collection you can redo calibration any time:

```bash
python replay_positions.py
```

The robot will move to each saved position automatically, capture, and re-run the solver.  

> ⚠️ **Always supervise and keep your hand on the E-stop.**  
> Speed is capped at `MOVE_SPEED = 0.05 m/s`. Do not increase above 0.10 m/s.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Board not detected | Improve lighting; ensure the board is flat; try reducing motion blur |
| `board_reproj_mean_px` > 5 | Remove blurry/partial samples with `D` during collection |
| `t_cam2tcp` looks wrong (> 0.5 m) | Check BOARD_COLS/BOARD_ROWS match your actual board; check SQUARE_M |
| Robot not connecting | Check `ROBOT_IP`; ensure robot is in Remote Control mode |
| All solver methods fail | Need more diverse samples — add more tilt variation |

---

## Using the result

Load `handeye_calibration.json` in your detection/pick scripts:

```python
import json, numpy as np

with open("Calibration/eye_in_hand/handeye_calibration.json") as f:
    cal = json.load(f)

R_cam2tcp = np.array(cal["R_cam2tcp"])   # 3×3 rotation
t_cam2tcp = np.array(cal["t_cam2tcp"])   # 3-vector, metres

# Transform a point from camera frame to TCP frame:
p_tcp = R_cam2tcp @ p_cam + t_cam2tcp

# To get to the robot base frame:
# p_base = R_tcp2base @ p_tcp + t_tcp2base   (from live robot pose)
```
