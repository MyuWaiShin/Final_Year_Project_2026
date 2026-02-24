# OAK-D Depth Testing

## How Depth Works on the OAK-D Lite

The OAK-D Lite has **two black-and-white cameras** (stereo pair) and **one RGB camera**.

```
 Left mono │   RGB   │ Right mono
   (CAM_B) │  (CAM_A)│   (CAM_C)
     ───────┼─────────┼───────
      ←75mm→         ←75mm→
             baseline ≈ 75mm
```

### Stereo Depth (How it works)

Same as human eyes — uses **parallax** (the shift of an object when viewed from two slightly different positions):

```
  Left camera sees object at pixel X=300
  Right camera sees same object at pixel X=280
  Disparity = 300 - 280 = 20 pixels

  Depth (mm) = (focal_length_px × baseline_mm) / disparity
             = (280 × 75) / 20
             = 1050 mm  (about 1 metre away)
```

- **More disparity** = object is **closer**
- **Less disparity** = object is **further**
- **Zero disparity** = cannot measure (too far away or featureless area)

### Depth Range for OAK-D Lite

| Mode | Min | Max |
|------|-----|-----|
| Normal | ~20 cm | ~10 m |
| Extended disparity | ~10 cm | ~5 m |
| Practical for robots | **30 cm** | **3–4 m** |

### Why Depth Fails Sometimes

- **Textureless surfaces** (plain white/blue table) → hard to match pixels
- **Reflective surfaces** (shiny metal) → wrong matches
- **Transparent objects** (glass) → sees through
- **Very close objects** (<20cm) → out of stereo range
- **Edges** → depth discontinuities cause noise

---

## Scripts

### 1. `calibration_check.py` — Start Here!
Reads the **factory calibration** stored inside the OAK-D and saves it.

```
py -3.11 calibration_check.py
```

Creates `calibration_data.npz` containing:
- **fx, fy** — focal lengths in pixels
- **cx, cy** — principal point (image center)
- **baseline** — distance between stereo cameras
- **distortion coefficients** — lens distortion

### 2. `depth_explorer.py` — Visualise Depth
Interactive viewer with colour depth map.

```
py -3.11 depth_explorer.py
```

Controls:
| Key | Action |
|-----|--------|
| **Left-click** | Print depth (mm) + 3D coordinates at that pixel |
| `h` | Toggle depth histogram |
| `c` | Toggle confidence map |
| `s` | Save current frame to `depth_frames/` |
| `q` | Quit |

**What you'll see:**
- JET colour map: 🔵 Blue = far, 🔴 Red = close
- Min/Max/Median depth overlay
- Click any pixel → get exact mm + XYZ in metres

### 3. `depth_to_3d.py` — The Robot Maths
Draw bounding boxes and get 3D object positions.

```
py -3.11 depth_to_3d.py
```

- **Left click + drag** → draw a box around an object
- Script computes the 3D position (X, Y, Z) in metres
- This is exactly what you'd send to the robot!
- Press `c` to clear boxes, `q` to quit

---

## The Key Maths (Pixel → 3D)

```python
# From camera calibration:
fx, fy  = focal lengths in pixels
cx, cy  = principal point (≈ image centre)

# From depth sensor:
depth_mm = depth_frame[v, u]   # depth at pixel (u, v)

# Convert to 3D (in metres):
Z = depth_mm / 1000.0
X = (u - cx) * Z / fx          # left/right:  + = right
Y = (v - cy) * Z / fy          # up/down:     + = down
# Z                             # forward:     + = away from camera
```

### Coordinate System:
```
        Y↓
        │
        │    Z (away from camera)
        │   /
        │  /
    ────┼──────→ X (right)
        │ (camera at origin)
```

---

## Getting Robust Depth for Robot Grasping

Don't just use a single pixel — use the **centre region** of the bounding box:

```python
# Get the bounding box ROI
roi = depth_frame[y1:y2, x1:x2]

# Filter out invalid pixels
valid = roi[roi > 100]

# Use low percentile (gets closest point, ignores background noise)
depth_mm = np.percentile(valid, 25)
```

Why 25th percentile?
- Ignores background pixels (too far)
- Ignores noise/holes (zero pixels)
- Gets the **object surface**, not the table behind it

---

## Recommended Run Order

```
1.  py -3.11 calibration_check.py    ← save camera params
2.  py -3.11 depth_explorer.py       ← understand depth data
3.  py -3.11 depth_to_3d.py          ← practice 3D projection
```

After this you'll be ready to integrate depth into your detection pipeline!
