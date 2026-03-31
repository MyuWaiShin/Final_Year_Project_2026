# Full Pipeline — Technical Overview

> **Project:** Final Year Project 2026  
> **Folder:** `full_pipeline/`  
> **Last updated:** 2026-03-31

---

## Table of Contents
1. [Pipeline at a glance](#1-pipeline-at-a-glance)
2. [main.py — Orchestrator](#2-mainpy--orchestrator)
3. [explore.py — Stage 1: Tag Search](#3-explorepy--stage-1-tag-search)
4. [navigate.py — Stage 2: TCP Alignment](#4-navigatepy--stage-2-tcp-alignment)
5. [How TCP alignment works (PnP)](#5-how-tcp-alignment-works-pnp)
6. [grasp.py — Stage 3: Pick](#6-grasppy--stage-3-pick)
7. [verify.py — Stage 4: Visual Verification](#7-verifypy--stage-4-visual-verification)
8. [How YOLO classifies](#8-how-yolo-classifies)
9. [How CLIP classifies](#9-how-clip-classifies)
10. [How the fusion score is calculated](#10-how-the-fusion-score-is-calculated)
11. [transit.py — Stage 5: Carry to Drop Zone](#11-transitpy--stage-5-carry-to-drop-zone)
12. [recover.py — Recovery Strategy](#12-recoverpy--recovery-strategy)
13. [Communication architecture](#13-communication-architecture)

---

## 1. Pipeline at a glance

```
python main.py
        │
        ▼
  [Stage 1]  explore.py    ─── Sweep base joint, find ArUco tag  (runs ONCE)
        │
        └─── retry loop (max 3 attempts) ──────────────────────────────┐
             │                                                          │
        [Stage 2]  navigate.py  ─── PnP → hover TCP above tag         │
             │                                                          │
        [Stage 3]  grasp.py     ─── Descend, close, width+force check  │
             │  missed? ─────────────────── recover.py ────────────────┘
             │                                                          │
        [Stage 4]  verify.py    ─── Lift, tilt wrist, YOLO+CLIP fuse   │
             │  empty?  ─────────────────── recover.py ────────────────┘
             │
        [Stage 5]  transit.py   ─── Carry to drop zone (slip monitor)
             │
        [Stage 6]  release.py   ─── Open gripper  [TODO]
```

---

## 2. main.py — Orchestrator

**File:** `main.py`

`main.py` owns the top-level state machine. It imports each stage as a
function and calls them in sequence, handling all retry/abort logic.

### Key constants

| Constant | Value | Meaning |
|---|---|---|
| `MAX_RETRIES` | 3 | Max attempts for the navigate→grasp→verify loop |
| `RECOVER_DESCEND_M` | 0.10 m | Descent used in grasp on retry attempts (shorter) |

### Flow

```python
tag_pos = explore()          # Stage 1 — runs once

for attempt in 1..MAX_RETRIES:

    hover_pos = navigate()   # Stage 2

    result = grasp(descend_m = 0.15 if attempt==1 else 0.10)  # Stage 3
    if result == "missed":
        recover(hover_pos)   # open gripper, rise 40cm, search circle
        continue             # → back to navigate

    result = verify()        # Stage 4
    if result == "empty":
        recover(hover_pos)   # same recovery
        continue

    break                    # success — exit retry loop

transit()                    # Stage 5
release()                    # Stage 6  [TODO]
```

### Protective stop handling

If `grasp()` or `verify()` raises a `RuntimeError` containing
`"PROTECTIVE STOP"`, `main.py` catches it and prints:

```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  PROTECTIVE STOP DETECTED
  Re-enable the robot on the pendant and reposition
  it to a safe pose, then restart the pipeline.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

No stacktrace. The protective stop flag is detected from the raw robot
state stream (subpacket type 0, byte at offset 17).

---

## 3. explore.py — Stage 1: Tag Search

**Purpose:** Find the physical ArUco tag (ID 13) and return its 3-D
position in the robot base frame.

### What it does step-by-step

1. **Load calibration** — reads `calibration/camera_matrix.npy` and
   `calibration/T_cam2flange.npy` (the eye-in-hand transform).
2. **Load scan pose** — reads `data/scan_pose.json` (a pre-recorded
   joint configuration where the camera has a clear overhead view).
3. **Open gripper** — ensures the RG2 is open before any motion.
4. **Move to scan pose** — user presses ENTER, robot moves.
5. **Sweep** — J0 (base joint) sweeps ±0.5 rad while the camera streams
   frames. Each frame is checked for tag ID 13.
6. **Detect** — when the tag is found, `stopj(acc=0.8)` is sent so the
   robot brakes hard and stops quickly right where the tag is visible.
7. **Retry sweep** — if no tag is found, J1 (shoulder) is adjusted by
   −0.15 rad and J3 (wrist1) by −0.15 rad to get a higher + compensated
   viewpoint, then a second sweep runs automatically (up to 2 sweeps
   total).

### Returns

```python
[x, y, z]   # tag position in robot base frame (metres)
None         # if still not found after all sweeps
```

---

## 4. navigate.py — Stage 2: TCP Alignment

**Purpose:** Move the robot TCP (tool centre point) to hover directly
above the ArUco tag, gripper pointing straight down.

### What it does step-by-step

1. **Open camera, load calibration.**
2. **Live detection loop** — every frame, detect tag 13 and estimate its
   3-D position in the base frame using PnP + the hand-eye transform
   (see §5).
3. **Stability check** — the computed hover target must be stable within
   5 mm for 8 consecutive frames (~0.25 s) before SPACE becomes active.
4. **User confirms** — if in manual mode, user types `YES`; in autonomous
   mode, 8 stable frames auto-trigger the move.
5. **`movel` to hover** — robot moves to the computed target using
   raw URScript over port 30002. Arrival is polled from the TCP pose
   stream until within 3 mm.

### Calibration offsets

Small empirical corrections applied on top of the PnP result:

| Offset | Value | Reason |
|---|---|---|
| `CALIB_X_OFFSET_M` | −0.005 m | Residual X error from hand-eye calibration |
| `CALIB_Y_OFFSET_M` | −0.050 m | Residual Y error |
| `CALIB_Z_OFFSET_M` | 0.000 m | Z comes from PnP directly |

### TCP orientation

Fixed to `[RX=2.225, RY=2.170, RZ=0.022]` (Rodrigues vector), which
corresponds to the gripper pointing **straight down** regardless of tag
orientation.

### Returns

```python
[x, y, z, rx, ry, rz]   # full 6-DOF hover TCP pose
None                     # if user quit (Q key)
```

---

## 5. How TCP Alignment Works (PnP)

This is the core of `navigate.py`'s spatial reasoning.

### Step 1 — Detect tag corners in 2D

OpenCV's `ArucoDetector` finds the four 2-D pixel corners of the tag in
the camera image.

### Step 2 — solvePnP → tag in camera frame

`cv2.solvePnP` takes:
- The 4 known 3-D corner positions of the tag in the **tag's own frame**
  (based on `MARKER_SIZE = 0.021 m`)
- The 4 detected 2-D corner positions in the **image**
- The camera intrinsic matrix `K` and distortion coefficients

It outputs a **rotation vector `rvec`** and **translation vector `tvec`**
that describe the tag's pose **relative to the camera**.

```
tag_in_camera = [rvec, tvec]
```

### Step 3 — Camera frame → Flange frame

The hand-eye calibration matrix `T_cam2flange` (loaded from
`calibration/T_cam2flange.npy`) transforms tag position from the camera
optical frame into the robot flange frame:

```
tag_in_flange = T_cam2flange @ tag_in_camera
```

### Step 4 — Flange frame → Base frame

The robot reports its current TCP pose (flange position + orientation) as
`T_flange2base`. Multiplying gives the tag in the world/base frame:

```
tag_in_base = T_flange2base @ tag_in_flange
```

### Step 5 — Compute hover target

```
hover_X = tag_in_base.X  +  CALIB_X_OFFSET_M
hover_Y = tag_in_base.Y  +  CALIB_Y_OFFSET_M
hover_Z = tag_in_base.Z  +  CALIB_Z_OFFSET_M   (tag surface ≈ 0 correction)
hover_orientation = [HOVER_RX, HOVER_RY, HOVER_RZ]  (straight down, fixed)
```

The combination of PnP + hand-eye transform + empirical offset reliably
positions the TCP within ~3–5 mm of the tag centre.

---

## 6. grasp.py — Stage 3: Pick

**Purpose:** Descend from hover, close the gripper, check Layer 1 failure
(width + force).

### Motion

| Move | Value |
|---|---|
| Descent (attempt 1) | 150 mm below hover Z |
| Descent (retry attempts) | 100 mm below hover Z |
| Descent speed | 0.03 m/s |
| Rise speed (on miss) | 0.06 m/s |

### Layer 1 failure detection

After closing the gripper, two checks run:

**Check 1 — Jaw width:**
The RG2 reports gripper width via an `AI2` analogue voltage signal.
A two-point linear calibration converts voltage → mm:

```
raw_mm = (voltage / 3.7) × 110
width_mm = raw_mm × slope + offset
  where slope  = (91.0 − 10.5) / (65.8 − 8.5)   # physical calibration points
        offset = 10.5 − (8.5 × slope)
```

If `width_mm < 11.0 mm` → the jaws are fully closed → object missed.

**Check 2 — Force contact (DI8):**
The RG2 signals force-limit reached via bit 17 of the masterboard
64-bit digital input word (read from subpacket type 3, port 30002).

```python
force_detected = bool(di_word & (1 << 17))
```

`True` = the gripper made solid contact and reached its force limit.

### Returns

```python
{"result": "holding", "width_mm": float, "force": bool, "tcp_pose": list}
{"result": "missed"}
```

---

## 7. verify.py — Stage 4: Visual Verification

**Purpose:** After grasping, lift the object and check with two AI models
whether something is actually in the gripper.

### Motion sequence

| Step | Action |
|---|---|
| 1 | Lift 186 mm straight up (back to hover height) |
| 2 | Tilt wrist1 (J3) by −1.2 rad (~69°) to expose gripper to camera |
| (bg) | YOLO + CLIP models load in a **background thread** during steps 1–2 |
| 3 | Open OAK-D camera, warm up 8 frames, grab one clean frame |
| 4 | Run YOLO inference |
| 5 | Run CLIP inference |
| 6 | Fuse scores, decide holding/empty |

### Returns

```python
{"result": "holding", "score": float, "yolo_conf": float, "clip_conf": float}
{"result": "empty",   "score": float, "yolo_conf": float, "clip_conf": float}
```

---

## 8. How YOLO Classifies

**Model:** `empty_holding_v1` — a fine-tuned YOLOv8 **classification**
model (not detection). It was trained from scratch on two classes:
`empty` and `holding`.

### Training data

Videos of the gripper were recorded using `temp/collect_videos.py`:
- Label `holding` — object gripped, lifted, tilted toward camera
- Label `empty` — same pose with no object

Frames were extracted and organised into `train/` and `val/` folders per
class using `Data_Preparation_V3/`.

### Training

```bash
yolo classify train  model=yolov8n-cls.pt  data=<data_dir>  epochs=50  imgsz=224
```

YOLOv8 classification fine-tunes the backbone + classification head on
your two-class dataset. The output is a `best.pt` weights file.

### Inference in verify.py

```python
model = YOLO("weights/empty_holding_v1/best.pt", task="classify")
result = model(frame)[0]
top1       = result.probs.top1      # index of predicted class (0 or 1)
top1conf   = result.probs.top1conf  # confidence of that class (0.0–1.0)

# Map class names to holding probability
classes = result.names              # {0: "empty", 1: "holding"} or similar
yolo_holding_prob = top1conf if classes[top1] == "holding" else 1 - top1conf
```

The output is a single probability `[0.0, 1.0]` that the gripper is
holding something.

---

## 9. How CLIP Classifies

**Model:** OpenAI `ViT-B/32` CLIP backbone + a trained **linear probe**
(logistic regression) on top of 512-D image embeddings.

### What CLIP is

CLIP (Contrastive Language-Image Pretraining) is a vision model that maps
images into a 512-dimensional embedding space. It was trained to align
images with text descriptions across hundreds of millions of pairs.

The CLIP backbone itself is **frozen** — its weights are never changed.
Only a small **linear probe** (logistic regression classifier) is trained
on top of the embeddings.

### Training the probe

1. For each training image (holding / empty), CLIP extracts a 512-D
   embedding vector using `clip.encode_image()`.
2. These vectors + labels are used to train a `sklearn` logistic
   regression model: `sklearn.linear_model.LogisticRegression`.
3. The trained classifier is saved as `clip_probe.pkl`.

This means you only need to train the linear head — far fewer parameters
and much faster training than fine-tuning CLIP itself.

### Inference in verify.py

```python
# 1. Preprocess image
img_tensor = clip_preprocess(PIL_image).unsqueeze(0).to(device)

# 2. Extract 512-D embedding (frozen CLIP backbone)
with torch.no_grad():
    embedding = clip_model.encode_image(img_tensor)        # shape: [1, 512]
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # L2 normalise

# 3. Linear probe predicts class probabilities
probs = clip_clf.predict_proba(embedding.cpu().numpy())    # [[p_empty, p_holding]]
clip_holding_prob = probs[0][class_index_of_holding]
```

The output is a probability `[0.0, 1.0]` that the gripper is holding
something — from a completely different feature space than YOLO.

---

## 10. How the Fusion Score is Calculated

The two probabilities are combined using a **weighted average**:

```
score = YOLO_WEIGHT × yolo_holding_prob
      + CLIP_WEIGHT × clip_holding_prob

where:
  YOLO_WEIGHT = 0.7
  CLIP_WEIGHT = 0.3
  (sum to 1.0)
```

| Scenario | YOLO | CLIP | Score | Decision |
|---|---|---|---|---|
| Both say holding | 1.00 | 1.00 | **1.00** | ✅ HOLDING |
| YOLO holding, CLIP unsure | 1.00 | 0.47 | **0.84** | ✅ HOLDING |
| YOLO holding, CLIP says empty | 1.00 | 0.00 | **0.70** | ✅ HOLDING |
| YOLO empty, CLIP holding | 0.00 | 1.00 | **0.30** | ❌ EMPTY |
| Both say empty | 0.00 | 0.00 | **0.00** | ❌ EMPTY |

**Threshold:** if `score ≥ 0.5` → `"holding"`, else `"empty"`.

YOLO gets the higher weight (0.7) because it was trained specifically on
images of this exact gripper and object, while CLIP is a general-purpose
model that may be less precise for this specific setup.

---

## 11. transit.py — Stage 5: Carry to Drop Zone

**Purpose:** Move the object from the pick site to a pre-recorded drop
zone, monitoring for slip during the journey.

### What it does

1. **Loads `data/drop_pose.json`** — a pre-recorded 6-DOF TCP pose at
   the drop zone (recorded once with
   `temp/capture_drop_pose.py`). If missing, returns `"no_drop_pose"`.
2. **Reads current TCP** — picks up wherever verify.py left the robot.
3. **Move sequence (3 steps):**

| Step | Move |
|---|---|
| 1 | Rise to `TRANSIT_Z_M = 0.15 m` (absolute base-frame height) at current X, Y |
| 2 | Move horizontally to above drop zone X, Y at transit height |
| 3 | Descend to drop Z |

4. **Slip detection** — a background thread polls `get_width_mm()` every
   0.1 s. If the width was above 11 mm (object present) and drops below
   it (object fell), `"slipped"` is returned immediately and the transit
   aborts.

### Returns

```python
{"result": "arrived", "tcp_pose": list}   # proceed to release.py
{"result": "slipped"}                     # recovery needed
{"result": "no_drop_pose"}               # config missing
```

> **Stage 6 (release.py):** Not yet implemented. Will open the RG2
> gripper at the drop zone and confirm the object was released.

---

## 12. recover.py — Recovery Strategy

Called by `main.py` whenever **grasp** returns `"missed"` or **verify**
returns `"empty"`.

### Recovery steps

```
FAILURE (grasp miss or verify empty)
  │
  ├─ 1. Open gripper (RG2 via dashboard URP — idempotent)
  │
  ├─ 2. Snap orientation to hover (tool pointing straight down)
  │       without going lower than current Z.
  │       Fixes wrist tilt left over from verify stage.
  │
  ├─ 3. Rise to  hover_Z + 0.40 m  (40cm above hover = ~58cm above object)
  │
  ├─ 4. Open OAK-D camera + do 25mm XY search circle at recovery height
  │       waypoints: right → forward → left → back → centre
  │       ─── If ArUco tag 13 is spotted at ANY point → stop circle immediately
  │
  └─ 5. Return True → main.py re-runs navigate → grasp → verify
```

### Why each step exists

| Step | Reason |
|---|---|
| Open gripper | Safe for re-approach; prevents object being dragged |
| Snap orientation | verify.py tilts J3 by −1.2 rad; this undoes it before rising |
| Rise 40cm | Gives the camera a wider field of view; prevents clipping object on next descent |
| Search circle | Sweeps the camera over a 25mm radius region; increases chance of tag re-acquisition from recovery height |
| Stop on tag | No point completing the circle; navigate() can take over |

### Descent depth on retry

| Attempt | Descent |
|---|---|
| 1st | 150 mm (full default) |
| 2nd, 3rd | 100 mm (shorter — navigate re-measures Z more accurately from recovery height) |

---

## 13. Communication Architecture

All robot communication uses **raw TCP sockets** — NO RTDE library.
RTDE caused unpredictable 10-second reconnect hangs on this setup.

### Port 30002 — Secondary Client (2 uses simultaneously)

| Use | Direction | What |
|---|---|---|
| `RobotStateReader` | Robot → PC | Background thread reads TCP pose, joint angles, AI2 voltage, masterboard DI word, robot mode (for protective stop) |
| `URScriptSender` | PC → Robot | Sends `movel(...)`, `movej(...)`, `stopj(...)` commands as plain text strings |

One persistent socket per purpose, with a drain thread to prevent the OS
receive buffer filling up.

### Port 29999 — Dashboard

Used for gripper control only. The RG2 is operated via UR programs
(`.urp`) on the pendant:

```
load /programs/myu/open_gripper.urp
play
```

The dashboard is a simple command/response protocol — a new socket is
opened per command and immediately closed.

### Why not RTDE?

RTDE (`RTDEControlInterface`) internally uses a watchdog that resets the
connection if not polled frequently enough. On this machine, competing
USB I/O from the OAK-D camera caused watchdog timeouts and 10-second
hangs. The raw secondary-client approach has no watchdog and has been
completely stable.
