# Safety System — How It Works and How to Use It

---

## Overview

The robot workspace is bounded by a **safe volume** defined by X/Y/Z limits in the robot's base frame. Two layers enforce this:

1. **Static limits** — XYZ bounds recorded once by jogging to each extreme (`record_safe_limits.py`)
2. **Dynamic floor limit** — OAK-D stereo depth measures the actual table surface in real time, enforcing a 2 mm clearance

---

## Files

| File | What it does |
|---|---|
| `Perception/safe_limits.json` | Stored workspace bounds. Do not edit manually. |
| `Perception/record_safe_limits.py` | Jog the robot to each axis extreme and press a key to record each limit. |
| `UR10/show_safe_limits.py` | **Run this to see your limits live.** Colour-coded terminal viewer. |
| `UR10/floor_depth_monitor.py` | **Run this to verify floor detection.** Jog the robot down while watching clearance. |
| `UR10/safety_guard.py` | Shared Python module imported by all robot scripts. |

---

## How Safety Limits Are Measured

Run `python Perception/record_safe_limits.py` and follow the prompts:

| Key | What to record |
|---|---|
| `1` | Jog to the **leftmost** X position |
| `2` | Jog to the **rightmost** X position |
| `3` | Jog to the **furthest forward** Y (max reach toward table) |
| `4` | Jog to the **furthest back** Y (nearest robot base) |
| `5` | Jog to the **lowest** Z (just above the mat surface) |
| `6` | Jog to the **highest** Z (max lift) |
| `S` | Save and generate the JSON |

A 10 mm margin is automatically applied inside all limits. The result is saved to `Perception/safe_limits.json`.

---

## Viewing Your Current Limits

```bash
python UR10/show_safe_limits.py
```

Run this in a second terminal while jogging the robot. It shows:
- 🟢 **Green** — inside safe zone
- 🟡 **Yellow** — within 20 mm of a limit  
- 🔴 **Red** — outside limit, VIOLATION

This is a read-only viewer. It does not move the robot.

---

## Floor Depth Detection

The OAK-D stereo camera can measure the floor depth in real time using the left/right mono cameras.

**Key fact about depth accuracy (OAK-D Lite, 75 mm baseline):**

| Camera height above floor | Depth error |
|---|---|
| 35 cm (min, extended disparity enabled) | ~0.9 mm |
| **50–70 cm (recommended)** | **~2–4 mm** |
| 100 cm | ~7.5 mm |

**Extended disparity** is always enabled in `floor_depth_monitor.py` — this pushes the minimum reliable depth from 70 cm down to ~35 cm.

### How to run the floor depth verification

```bash
python UR10/floor_depth_monitor.py
```

1. Position the camera roughly **50–70 cm above the table**
2. The monitor shows live depth and robot Z position
3. Jog the robot **down slowly** with the pendant
4. Watch the `Clearance` readout:
   - 🟡 Yellow at 20 mm from floor  
   - 🔴 Red = VIOLATION (below 2 mm clearance)
5. Press **`F`** (or click the camera window and press `f`) to record the current floor Z
6. Press **`Q`** to quit — a final summary is printed

The script prints a line you can paste directly into the pipeline:
```python
FLOOR_Z = -0.4504  # metres — paste into safety_guard.py or failure_detection_pipeline.py
```

---

## How `safety_guard.py` Works

Import in any robot script:

```python
from UR10.safety_guard import SafetyGuard, SafetyViolation

guard = SafetyGuard()          # loads Perception/safe_limits.json

# Hard check — raises SafetyViolation if out of bounds
guard.check(x, y, z)

# Soft clamp — returns adjusted pose, prints warning if clamped
x, y, z = guard.clamp(x, y, z)

# Dynamic floor — updated by DepthFloorDetector in the pipeline
print(guard.floor_z)           # current detected floor Z (metres)
print(guard.effective_z_min()) # max(static_limit, floor_z + 2mm)
```

### Where it is enforced

- **`replay_positions.py`** — every `movel_pose()` is checked before sending to the robot
- **`failure_detection_pipeline.py`** — every `_movel()` and `_movej()` is checked; pick Z is clamped to `floor_z + 2mm`

---

## Re-recording Limits After Moving the Setup

If the robot, table, or camera mount position changes:

1. Re-run `python Perception/record_safe_limits.py`
2. Re-run `python UR10/floor_depth_monitor.py` to verify floor Z
3. No code changes needed — all scripts load from `safe_limits.json` at startup
