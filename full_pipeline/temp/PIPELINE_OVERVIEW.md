# Full Pipeline — Technical Overview

> **Project:** Final Year Project 2026  
> **Folder:** `full_pipeline/`  
> **Last updated:** 2026-04-06

---

## Table of Contents
1. [Pipeline at a glance](#1-pipeline-at-a-glance)
2. [main.py — Orchestrator](#2-mainpy--orchestrator)
3. [explore.py — Stage 1: Object Search](#3-explorepy--stage-1-object-search)
4. [navigate.py — Stage 2: TCP Alignment](#4-navigatepy--stage-2-tcp-alignment)
5. [How TCP alignment works (PnP)](#5-how-tcp-alignment-works-pnp)
6. [grasp.py — Stage 3: Pick](#6-grasppy--stage-3-pick)
7. [verify.py — Stage 4: Visual Verification](#7-verifypy--stage-4-visual-verification)
8. [How YOLO classifies](#8-how-yolo-classifies)
9. [transit.py — Stage 5: Carry to Drop Zone](#9-transitpy--stage-5-carry-to-drop-zone)
10. [release.py — Stage 6: Release](#10-releasepy--stage-6-release)
11. [recover.py — Recovery Strategy](#11-recoverpy--recovery-strategy)
12. [Communication architecture](#12-communication-architecture)

---

## 1. Pipeline at a glance

```
python main.py [--autonomous]
        │
        ▼
  [Stage 1]  explore.py    ─── J0 sweep + YOLO detect, J0 recentre  (runs ONCE)
        │
        └─── retry loop (max 5 total failures) ────────────────────────────────┐
             │                                                                  │
        [Stage 2]  navigate.py  ── ArUco or YOLO+depth → hover TCP             │
             │                      derives clearance_z = hover_z + 0.40 m    │
             │                                                                  │
        [Stage 3]  grasp.py     ── Descend, close, width+force check           │
             │  missed? ─────────────── recover.py ──────────────────────────→┘
             │                                                                  │
        [Stage 4]  verify.py    ── Rise to clearance_z, YOLO26n classify       │
             │  empty?  ─────────────── recover.py ──────────────────────────→┘
             │                                                                  │
        [Stage 5]  transit.py   ── Move at clearance_z, dual slip monitor      │
             │  slip/empty? ──────── recover.py ─────────────────────────────→┘
             │
        [Stage 6]  release.py   ── Descend, open gripper, return to scan pose
```

**`clearance_z`** is a shared Z height (hover_z + 0.40 m) computed by `navigate.py` and passed through every subsequent stage. All transit motion and recovery happen at this height.

---

## 2. main.py — Orchestrator

**File:** `main.py`

`main.py` owns the top-level state machine. It imports each stage as a function and calls them in sequence, sharing a failure budget across all stages.

### Key constants

| Constant | Value | Meaning |
|---|---|---|
| `MAX_RETRIES` | 5 | Shared failure budget across grasp + verify + transit failures |

### Flow

```python
# Pre-load YOLO classify model in background (so verify has zero loading latency)
threading.Thread(target=load_models).start()

found = explore(autonomous=autonomous)   # Stage 1 — runs once

while True:
    nav_result = navigate(autonomous=autonomous)
    hover_pose  = nav_result["hover_pose"]
    clearance_z = nav_result["clearance_z"]      # hover_z + 0.40 m

    grasp_result = grasp(close_only=(attempt > 1), autonomous=autonomous)
    if grasp_result["result"] == "missed":
        recover(clearance_z); continue

    verify_result = verify(clearance_z=clearance_z, models=preloaded_models)
    if verify_result["result"] == "empty":
        recover(clearance_z); continue

    transit_result = transit(clearance_z=clearance_z, models={"cls": yolo_model})
    if transit_result["result"] != "arrived":
        recover(clearance_z); continue

    release(clearance_z=clearance_z)
    break   # success
```

### Protective stop handling

If any stage's `movel` detects a protective stop (sub-packet type 0, byte at offset 17), it raises `RuntimeError("PROTECTIVE STOP")`. `main.py` catches this and prints clearly:

```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  PROTECTIVE STOP DETECTED
  Re-enable on pendant, reposition to safe pose, restart.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

### Modes

```bash
python main.py               # debug mode — prompts before each stage
python main.py --autonomous  # fully autonomous — no prompts
```

---

## 3. explore.py — Stage 1: Object Search

> **Changed from v1:** ArUco tag detection removed from sweep. Explore now runs **YOLO-only** detection during the sweep. J0 active recentering added. Retry descent is now camera-compensated rather than a raw joint offset.

**Purpose:** Find the object on the table, stop the sweep when it's centred in frame, and fine-tune J0 so `navigate.py` can find it immediately.

### What it does step-by-step

1. **Load calibration** — `calibration/camera_matrix.npy` and `calibration/T_cam2flange.npy`
2. **Load scan pose** — joint angles from `data/scan_pose.json`
3. **Move to scan pose** — `movej` (blocking, polls joint positions)
4. **Open gripper** — skips if already open (checks AI2 voltage-derived width ≥ 80mm)
5. **Sweep** — J0 sweeps ±0.5 rad in a background thread while the main thread runs YOLO on every frame. On first YOLO detection ≥ 80% confidence, `stopj(0.8)` fires and the sweep thread exits
6. **Active J0 recentering** — up to 6 proportional corrections:
   ```
   delta_j0 = -offset_px × 0.0008 rad/px
   ```
   Stops when object is within ±20px of frame centre
7. **Retry with camera-compensated descent** — if no detection after sweep 1, descends 10cm while preserving the camera's look-at point on the table (using `T_cam2flange` to recompute X, Y so the view stays centred)
8. **Max 2 sweeps** total before giving up

### Returns

```python
True   # object found and centred
None   # not found after all sweeps
```

### Key config

| Param | Value |
|---|---|
| `SWEEP_START_RAD / END_RAD` | −0.5 / +0.5 rad |
| `SWEEP_SPEED` | 0.2 rad/s |
| `CONF_THRESHOLD` | 0.80 (YOLO) |
| `CENTER_TOL_PX` | ±80 px (sweep stop) |
| `RECENTER_TOL_PX` | ±20 px (J0 correction) |
| `RECENTER_GAIN` | 0.0008 rad/px |
| `SCAN_MAX_SWEEPS` | 2 |
| `RETRY_LOWER_M` | 0.10 m |

---

## 4. navigate.py — Stage 2: TCP Alignment

> **Changed from v1:** Now opens both an RGB **and a stereo depth stream** from the OAK-D. Added YOLO+depth fallback when ArUco is not visible. Added post-hover centring correction (horizontal TCP shift using depth-derived pixel→metre projection). `clearance_z` changed from `hover_z + 0.60 m` to `hover_z + 0.40 m`. Wrist orientation can now align to the ArUco tag X-axis instead of being fixed.

**Purpose:** Move the robot TCP to hover directly above the object and compute `clearance_z` for all downstream stages.

### Detection hierarchy

Tried in order on every frame:

| Priority | Method | How |
|---|---|---|
| 1st | ArUco ID 13 | `solvePnP` → tag in camera frame → base frame via hand-eye |
| 2nd | YOLO + OAK-D depth | Bbox centre + median depth → deproject to base frame |

YOLO fallback is also used when `force_yolo=True` is passed.

### Dual camera streams

```python
# RGB
cam (1280×720, manual focus=46) → videoQueue

# Stereo depth aligned to RGB
mono_L + mono_R → StereoDepth (HIGH_DENSITY, aligned to CAM_A) → depthQueue
```

Depth is used for `pixel_to_base_frame()` when YOLO is the active detector.

### Stability filter (autonomous mode)

Detection must remain stable within ±5mm for **8 consecutive frames** before the hover move is triggered. Manual mode: user presses SPACE.

### Orientation

**ArUco mode:** wrist yaw aligns to the ArUco tag's X-axis (proportional rotation of the baseline `[RX=2.225, RY=2.170, RZ=0.022]` Rodrigues vector).

**YOLO mode:** fixed orientation `[RX=2.225, RY=2.170, RZ=0.022]` (straight down).

### Post-hover centring correction

After arriving at hover, a second centring loop runs up to 3 times. Each iteration moves the TCP **horizontally** to correct residual pixel offset:

```python
delta_x_cam = delta_px × depth_z / fx  # pixel → metres using depth
delta_base  = R_cam2base @ [delta_x_cam, 0, 0]  # rotate into base frame
new_x = tcp[0] + delta_base[0]
new_y = tcp[1] + delta_base[1]
# send movel to corrected X, Y
```

### Calibration offsets

| Offset | Value |
|---|---|
| `CALIB_X_OFFSET_M` | −0.005 m |
| `CALIB_Y_OFFSET_M` | −0.050 m |
| `CALIB_Z_OFFSET_M` | 0.000 m |

### Returns

```python
{
    "hover_pose":      [x, y, z, rx, ry, rz],   # 6-DOF hover TCP
    "clearance_z":     float,                    # hover_z + 0.40 m
    "detection_mode":  "aruco" | "yolo_only"
}
None   # user quit
```

---

## 5. How TCP Alignment Works (PnP)

This is the core of `navigate.py`'s spatial reasoning (ArUco path):

### Step 1 — Detect tag corners in 2D

`ArucoDetector.detectMarkers()` → 4 pixel corners of ArUco tag ID 13.

### Step 2 — solvePnP → tag in camera frame

```python
cv2.solvePnP(
    objectPoints,   # known 3D corners in tag frame (MARKER_SIZE = 0.036 m)
    imagePoints,    # detected 2D corners in image
    K,              # camera intrinsics
    dist_coeffs     # zeros (camera calibrated separately)
) → rvec, tvec    # tag pose relative to camera
```

### Step 3 → Step 4 — Camera → Flange → Base

```python
T_tag2cam  = [R_tag | tvec]           # 4×4 from solvePnP
T_tcp2base = tcp_to_matrix(tcp_pose)  # FK from state stream
T_tag2base = T_tcp2base @ T_cam2flange @ T_tag2cam
```

`T_cam2flange` is the hand-eye calibration result (Tsai-Lenz algorithm, from `07_hand_eye_calibration.py`).

### Step 5 — Compute hover target

```python
hover_X = T_tag2base[0,3] + CALIB_X_OFFSET_M
hover_Y = T_tag2base[1,3] + CALIB_Y_OFFSET_M
hover_Z = T_tag2base[2,3] + CALIB_Z_OFFSET_M
hover_orientation = compute_hover_orientation(R_tag_base, R_hover_baseline)
```

---

## 6. grasp.py — Stage 3: Pick

> **Changed from v1:** Descent reduced from 150mm to **70mm** (both normal and recovery mode). Force contact (DI8) is now logged but **not** a gate for missed detection — width alone determines miss. Gripper wait logic has been upgraded to a two-phase poll (wait for movement to start, then wait for it to stop). Protective stop detection added to `movel`.

**Purpose:** Descend from hover, close the gripper, run Layer 1 failure detection.

### Motion

| Move | Speed |
|---|---|
| Descent (both modes) | 0.02 m/s, 0.008 m/s² |
| Rise on miss | 0.04 m/s, 0.01 m/s² |
| Descent depth | 70 mm below hover Z |

### Layer 1 failure detection

**Check 1 — Jaw width (gate):**
```
raw_mm = (voltage / 3.7) × 110
width_mm = raw_mm × slope + offset
   slope  = (91.0 − 10.5) / (65.8 − 8.5)
   offset = 10.5 − (8.5 × slope)

width < 11.0 mm → MISSED
width ≥ 11.0 mm → something between fingers → proceed
```

**Check 2 — Force contact (logged only):**
```python
force = bool(di_word & (1 << 17))   # RG2 force limit reached
# HIGH = solid contact confirmed
# LOW  = soft object or sensor lag — not fatal, just logged
```

### Returns

```python
{"result": "holding", "width_mm": float, "force": bool, "tcp_pose": list}
{"result": "missed"}
```

---

## 7. verify.py — Stage 4: Visual Verification

> **Changed from v1:** Completely redesigned. **CLIP and YOLO fusion removed.** Now uses **YOLO26n classification model only** (`yolo26n_cls_V1`). No wrist tilt (J3 rotation removed). Rises straight to `clearance_z` (0.40 m above hover). Confidence threshold raised to **90%**. Model is pre-loaded by `main.py` at startup to eliminate loading latency.

**Purpose:** After grasping, rise to clearance height and run the YOLO classifier on a single camera frame.

### Motion sequence

| Step | Action |
|---|---|
| 1 | Rise straight to `clearance_z` (movel, same X/Y, same wrist) |
| (bg) | YOLO26n pre-loaded via `load_models()` called at pipeline startup |
| 2 | Open OAK-D camera (`1920×1080`), warm up 8 frames, grab one |
| 3 | Run YOLO26n classify → `p_holding` probability |
| 4 | Threshold: `p_holding ≥ 0.90` → "holding", else → "empty" |

### Returns

```python
{"result": "holding", "yolo_conf": float}
{"result": "empty",   "yolo_conf": float}
```

---

## 8. How YOLO Classifies

**Model:** `yolo26n_cls_V1` — YOLOv8n **classification** model trained on two classes: `empty` and `holding`.

### Inference

```python
results = model(frame, imgsz=224, verbose=False)
r = results[0]
probs = {r.names[i]: float(r.probs.data[i]) for i in range(len(r.names))}
p_holding = probs.get("holding", 0.0)
p_empty   = probs.get("empty",   0.0)
```

- Output: `p_holding` in `[0.0, 1.0]`
- Threshold: `p_holding ≥ 0.90` → "holding"

> **Note:** Fusion with CLIP was removed. YOLO classification alone is used at this height because the camera looks straight down at the closed gripper and the classifier is reliable from `clearance_z`.

---

## 9. transit.py — Stage 5: Carry to Drop Zone

> **Changed from v1:** Transit now uses `clearance_z` (from navigate) as the travel height rather than a fixed `TRANSIT_Z_M`. Added **Layer 2 YOLO classify monitor** — a background thread that runs the same YOLO26n classifier on live camera frames and stops the arm after 3 consecutive "empty" predictions. Layer 1 gripper-thread slip detection `SLIP_DETECT` flag defaults to `False` (disabled by default — enable if needed).

**Purpose:** Move from pick position to drop zone at clearance height, monitoring for slip.

### Drop zone

Loaded from `data/drop_zone.json` (x, y only — Z stays at `clearance_z`).

### Two slip detection layers

**Layer 1 — Gripper width (optional, `SLIP_DETECT = False` by default):**

When enabled, sends a multi-line URScript program that runs `movel` and a gripper re-close thread concurrently:

```python
def transit():
  thread KeepClosed:
    while True:
      rg2_grip(0, 40)        # re-closes every 0.2s
      sleep(0.2)
    end
  end
  run KeepClosed()
  movel(p[target_x, target_y, clearance_z, ...])
end
```

A background Python thread monitors AI2 width. If width drops below 11mm → `stopl()` → returns `{"result": "slip", "layer": 1}`.

**Layer 2 — YOLO classify (always active):**

A background thread reads frames from the OAK-D camera queue and runs `yolo26n_cls_V1`. After **3 consecutive "empty" predictions** → `stopl()` → returns `{"result": "empty", "layer": 2}`.

```python
if p_holding < 0.90:
    consec_empty += 1
else:
    consec_empty = 0
if consec_empty >= 3:
    stop → return "empty"
```

### Returns

```python
{"result": "arrived"}             # reached drop zone cleanly
{"result": "slip",  "layer": 1}   # gripper width drop
{"result": "empty", "layer": 2}   # YOLO classify empty
```

---

## 10. release.py — Stage 6: Release

> **New in current version:** `release.py` is now fully implemented (was `[TODO]` in the previous overview).

**Purpose:** Descend to drop height, open gripper, rise, return to scan pose.

### Motion sequence

| Step | Action |
|---|---|
| 1 | Descend `DROP_HEIGHT_M = 0.50 m` below `clearance_z` (slow: 0.02 m/s) |
| 2 | Open gripper via `open_gripper.urp` dashboard URP |
| 3 | Rise back to `clearance_z` |
| 4 | `movej` to scan pose joints (from `data/scan_pose.json`) |

### Returns

```python
True   # always — pipeline complete
```

---

## 11. recover.py — Recovery Strategy

> **Changed from v1:** Recovery rises to **scan height** (from `scan_pose.json`'s `tcp_pose.z`) rather than `clearance_z + 0.40 m`. Search circle radius changed from 25mm to **60mm**. ArUco detection removed from circle — circle uses **YOLO-only**. After circle stops, active **J0 recentering** runs (same as explore.py, ±20px, up to 6 iterations) before handing back to navigate.

Called by `main.py` when grasp, verify, or transit fails.

### Recovery steps

```
FAILURE
  │
  ├─ 1. Open gripper (idempotent — skips if already open ≥ 80mm)
  │
  ├─ 2. Rise to scan height Z (from scan_pose.json tcp_pose)
  │       with scan wrist orientation — undoes any wrist tilt
  │
  ├─ 3. 60mm XY search circle at scan height
  │       5 waypoints: +R, +R (90°), -R (270°), -R (180°), centre
  │       circle thread runs movel while main thread reads YOLO on every frame
  │       First YOLO detection ≥ 80% → stopj(0.8), stop circle
  │
  ├─ 4. Active J0 recentering (±20px, up to 6 moves, 0.0008 rad/px gain)
  │
  └─ 5. Return True → main.py re-enters navigate → grasp → verify
```

### Why scan height (not clearance_z)?

The scan pose was chosen to give the camera a clear overhead view of the full table. Recovery ascending to scan height restores that wide field of view, giving YOLO a better chance of re-detecting the object than from the lower clearance height.

---

## 12. Communication Architecture

All robot communication uses **raw TCP sockets — NO RTDE library**. RTDE caused unpredictable 10-second reconnect hangs on this setup (documented in `pipeline_dev/RTDE_debug_log.md`).

### Port layout

| Port | Class | Direction | Purpose |
|---|---|---|---|
| **30002** | `RobotStateReader` | Robot → PC | Background thread parses secondary-client binary packets: TCP pose (type 4), joint angles (type 1), AI2 voltage/width (type 2), DI word/force (type 3), protective stop flag (type 0) |
| **30002** | `URScriptSender` | PC → Robot | Persistent socket sends `movel`, `movej`, `stopj`, `stopl` as plain URScript text. Drain thread prevents OS recv buffer from filling |
| **29999** | `dashboard_cmd()` | PC → Robot | Short-lived connections per command: `load <urp>`, `play`, `stop`, `running` — gripper control only |

### Why two sockets on 30002?

The UR robot allows multiple simultaneous connections to port 30002. One socket is read-only (state parsing) and one is write-only (commands). The write socket has a drain thread to discard the 125Hz data the robot sends back, preventing the OS receive buffer from filling and blocking `sendall()`.

### Why not RTDE?

`RTDEControlInterface` (C++ binding) auto-reconnects when the connection drops. On this machine, calling the dashboard (`load`/`play`) between RTDE motion commands caused the RTDE C++ layer to enter a blocking reconnect loop on a Windows synchronisation primitive — hanging the entire Python process for 10+ seconds with no Ctrl+C response. Removing RTDE entirely and using port 30002 directly has been completely stable.

### Protective stop detection

`RobotStateReader` in `grasp.py` and `verify.py` parses sub-packet type 0, byte at offset 17:

```python
self._protective_stop = bool(data[off + 17])
```

If this flag is set when `movel` times out, it raises `RuntimeError("PROTECTIVE STOP")` which propagates up to `main.py`.

### Why Is Accuracy Better When Centred?

Yes, lens distortion is one reason, but there are actually three independent reasons stacked on top of each other:

1. Radial lens distortion (your guess — correct)
Every camera lens has barrel/pincushion distortion that's zero at the centre and grows toward the edges. Your calibration (camera_matrix.npy) corrects for this, but the correction itself has residual error — the further from centre, the larger that residual error is. Dead centre = zero distortion to correct for = zero correction error.

2. Perspective / oblique viewing angle
When the object is off-centre, the camera is looking at it at an angle, not straight down. This means:

The object appears slightly foreshortened
The apparent X/Y of the object in the image is shifted from its true X/Y on the table
The further off-centre, the worse this angle gets
When the camera is directly overhead, you're looking straight down — no angular distortion, the pixel X/Y maps directly to real-world X/Y.

Camera off-centre:          Camera centred:
    ↘  (looking at angle)       ↓  (looking straight down)
     [object]                  [object]
  → apparent X is wrong    → apparent X is correct
3. ArUco / PnP corner detection accuracy
solvePnP estimates pose from the 4 corner points of the ArUco tag. When the tag is off-centre, you're seeing it from an angle, so the corners are foreshortened and closer together in image space. Smaller projected area = larger relative error in pixel-level corner detection = noisier PnP output. When the tag is centred and square-on, the corners are spread wide in the image = most accurate PnP.

Practically speaking:
If you don't recentre and the object is 200px off to the side, you might get 5–10mm of systematic position error just from the oblique angle. 70mm descent with 8mm of lateral error = gripper misses or clips the edge of the object
