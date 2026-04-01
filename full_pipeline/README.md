# full_pipeline

Autonomous pick-and-place pipeline for UR10 + RG2 gripper + OAK-D Lite camera.

## Running

```bash
# Debug mode — prompts before each stage, requires ENTER/YES confirmation
python full_pipeline/main.py

# Autonomous mode — no prompts, auto-triggers on stable detections
python full_pipeline/main.py --autonomous
```

## Stages

```
1. explore()   ── runs once ──────────────────────────────────────────┐
2. navigate()  ──┐                                                     │
3. grasp()     ──┼── retry loop (max 3 grasp fails + 3 verify fails)  │
4. verify()    ──┘                                                     │
   recover()   ── called on any failure; re-enters loop at step 2 ────┘
5. transit()   ── TODO
6. release()   ── TODO
```

CLIP + YOLO models are pre-loaded in a background thread during stages 1–2 so verify() has zero loading latency.

---

## Stage details

### Stage 1 — explore.py

Moves to a saved scan pose (`data/scan_pose.json`), then sweeps joint J0 across a ±0.5 rad arc at 0.2 rad/s while looking for ArUco tag ID 13. Stops the sweep immediately on detection.

- Up to **2 sweep attempts** (`SCAN_MAX_SWEEPS = 2`). On retry, descends 10 cm with camera-offset-compensated EEF XY so the camera still covers the same table footprint from closer range.
- Dwell of 0.6 s at each sweep endpoint — detection is unreliable while the arm is moving.
- Returns `tag_pos` as `[x, y, z]` in the robot base frame, or `None` if not found (pipeline aborts).
- Opens the gripper via `open_gripper.urp` before sweeping.

To record or update the scan pose:
```bash
python full_pipeline/temp/capture_scan_pose.py
```

### Stage 2 — navigate.py

Aligns the TCP directly above the tag at a fixed hover height. XY tracks the base-frame tag position plus empirical calibration offsets (`CALIB_X_OFFSET_M = -5 mm`, `CALIB_Y_OFFSET_M = -50 mm`). Z is kept at whatever height the tag was detected at (not a fixed Z).

- **Orientation**: smart yaw alignment — extracts the tag's X-axis yaw in the base XY plane and rotates the baseline hover orientation (`rx=2.225, ry=2.170, rz=0.022`) to match. Picks the smaller of the two equivalent rotations (gripper is symmetric, so ±180° are equivalent).
- **Autonomous mode**: triggers when the tag position is stable within 5 mm for 8 consecutive frames (~0.25 s at 30 fps).
- **Debug mode**: press SPACE in the camera window, then type `YES` to execute the move.
- **Horizontal centering** (after hover move): up to 3 correction moves to bring the tag within 40 px of the image horizontal midline. This aligns the gripper gap with the object's long axis.
- Returns full `[x, y, z, rx, ry, rz]` hover pose.

### Stage 3 — grasp.py

Descends and closes the gripper, then validates the grasp with two sensor checks.

| Parameter | Value |
|-----------|-------|
| First-attempt descent | 150 mm (`DESCEND_OFFSET`) |
| Recovery-attempt descent | 10 mm (`RECOVERY_DESCEND_OFFSET`) |
| Descend speed | 0.02 m/s |
| Width threshold (miss) | ≤ 11 mm |

**Check 1 — jaw width**: AI2 voltage → mm via two-point linear calibration:
```
raw_mm = (voltage / 3.7) * 110.0
calibrated = raw_mm * slope + offset    # slope/offset from (8.5→10.5, 65.8→91.0) mm pair
```
If calibrated width ≤ 11 mm → gripper fully closed → missed → opens gripper, rises back to hover Z, returns `{"result": "missed"}`.

**Check 2 — force contact**: DI8 bit 17 of the masterboard data (Type 3 sub-packet). HIGH = RG2 force limit reached. LOW is logged but not fatal (soft objects or sensor lag).

Returns:
```python
{"result": "holding", "width_mm": float, "force": bool, "tcp_pose": list}
{"result": "missed"}
```

### Stage 4 — verify.py

Lifts 186 mm (matching the descent), tilts J3 by -1.2 rad (~69°) so the camera can see into the gripper jaws, then classifies the frame with two models.

**YOLO classifier** (`yolo_binary_classifiers/empty_holding_v1/weights/best.pt`):
- Binary classification: `empty` vs `holding`
- Input: 224×224

**CLIP linear probe** (`../CLIP_post_grab/clip_probe.pkl`):
- ViT-B/32 backbone + sklearn `LinearProbe`
- Crops bottom-centre of 1920×1080 frame (1400×600 px)
- Classes: `{0: "Empty", 1: "Holding"}`

**Fusion**:
```
score = 0.7 × YOLO_p_holding + 0.3 × CLIP_p_holding
result = "holding" if score >= 0.5 else "empty"
```

Decision table:

| YOLO | CLIP | Score | Result |
|------|------|-------|--------|
| holding | holding | ~1.00 | pass |
| holding | empty   | ~0.66 | pass |
| empty   | holding | ~0.34 | fail  |
| empty   | empty   | ~0.00 | fail  |

Shows a debug window for 2 s, then returns:
```python
{"result": "holding"/"empty", "score": float, "yolo_conf": float, "clip_conf": float}
```

### recover.py

Called by main.py when grasp or verify fails.

1. Opens gripper (idempotent).
2. Snaps orientation back to hover (tool pointing down) without descending further.
3. Rises to `hover_Z + 0.40 m` (40 cm clearance).
4. Runs a **4-point XY search circle** (60 mm radius): right → forward → left → back → centre. Circle centre is camera-offset compensated so the camera FOV covers the original tag position.
5. Stops the circle immediately if ArUco tag ID 13 is spotted.
6. If tag found: does a centering move to put the tag in the camera crosshairs.
7. Returns `True`; main.py re-enters the navigate → grasp → verify loop.

---

## Robot communication

All stages use raw URScript over port 30002. **RTDE is not used** — both `RTDEControlInterface` and `RTDEReceiveInterface` caused unpredictable 10-second reconnect hangs on this setup (see `pipeline_dev/RTDE_debug_log.md`).

Each stage creates two port-30002 connections:
- **`RobotStateReader`** — background thread that parses the secondary-client binary stream (~10 Hz). Sub-packets decoded:
  - Type 0: Robot Mode Data — protective stop flag (byte at offset 17)
  - Type 1: Joint Data — 6 × `q_actual` doubles (each joint is 41 bytes)
  - Type 2: Tool Data — AI2 voltage at offset 7 → gripper width
  - Type 3: Masterboard Data — 64-bit DI word at offset 5 → force bit
  - Type 4: Cartesian Info — 6 doubles at offset 5 → TCP pose
- **`URScriptSender`** — persistent socket with a drain thread (discards incoming bytes to prevent OS buffer fill blocking `sendall()`). Auto-reconnects on failure.

Gripper is actuated by loading and playing `.urp` programs via Dashboard port 29999:
- `open_gripper.urp` → `/programs/myu/open_gripper.urp`
- `close_gripper.urp` → `/programs/myu/close_gripper.urp`

Robot IP: `192.168.8.102`

---

## Calibration files

| File | Contents |
|------|----------|
| `calibration/camera_matrix.npy` | 3×3 intrinsic matrix K |
| `calibration/T_cam2flange.npy` | 4×4 hand-eye transform (camera → TCP flange) |

`T_cam2flange` is applied as:
```
T_tag2base = T_tcp2base @ T_cam2flange @ T_tag2cam
```

To recalibrate: see `Calibration/eye_in_hand/` and copy the result here.

---

## ArUco config

All stages share the same hardcoded detector:
- **Tag ID**: 13
- **Dictionary**: `DICT_6X6_250`
- **Marker size**: 21 mm

---

## Key constants (tuning reference)

| Constant | File | Default | Effect |
|----------|------|---------|--------|
| `SWEEP_START_RAD / SWEEP_END_RAD` | explore.py | ±0.5 rad | J0 sweep arc |
| `SCAN_MAX_SWEEPS` | explore.py | 2 | Max scan attempts |
| `RETRY_LOWER_M` | explore.py | 0.10 m | Descent on retry |
| `CALIB_X_OFFSET_M` | navigate.py | -0.005 m | Residual X correction |
| `CALIB_Y_OFFSET_M` | navigate.py | -0.050 m | Residual Y correction |
| `STABLE_FRAMES_NEEDED` | navigate.py | 8 | Frames for auto-trigger |
| `DESCEND_OFFSET` | grasp.py | 0.150 m | Full descent depth |
| `WIDTH_CLOSED_MM` | grasp.py | 11.0 mm | Gripper miss threshold |
| `LIFT_HEIGHT_M` | verify.py | 0.186 m | Lift before verify |
| `WRIST1_TILT_RAD` | verify.py | -1.2 rad | J3 tilt for camera |
| `YOLO_WEIGHT / CLIP_WEIGHT` | verify.py | 0.7 / 0.3 | Fusion weights |
| `FUSION_THRESHOLD` | verify.py | 0.5 | Pass/fail threshold |
| `RECOVERY_HEIGHT_M` | recover.py | 0.40 m | Rise height |
| `SEARCH_RADIUS_M` | recover.py | 0.060 m | Circle radius |

---

## Adding transit / release (stages 5–6)

In `main.py`, uncomment the imports and add calls after the verify success block:
```python
# from transit  import main as transit
# from release  import main as release
```
The `hover_pos` and `verify_result` are already in scope at that point.
