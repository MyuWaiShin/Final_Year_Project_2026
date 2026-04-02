"""
test_obb_live.py
----------------
Live oriented bounding box test using the laptop webcam.

For each YOLO detection, crops the box, runs Canny edges, fits
cv2.minAreaRect on the largest contour → real rotation angle.
Draws XYZ axes and reports gripper angle + aspect ratio.

A small debug window shows the edge mask for the last detection.

Controls
--------
    Q  →  quit
"""

import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH  = (SCRIPT_DIR.parent / "models" / "detection"
               / "yolo26n_detect_V1" / "weights" / "best.pt")

BOX_COLOURS = [
    (0, 220, 0),    # class 0 — green
    (0, 100, 255),  # class 1 — orange
]
CONF_THRESHOLD = 0.25
PADDING        = 6     # px to add around YOLO crop


# ── Oriented box via Canny ─────────────────────────────────────────────────

def fit_rotated_box(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int):
    """
    Crop to YOLO bbox, run Canny edges, fit minAreaRect on largest contour.

    Returns (valid, cx, cy, angle_deg, aspect_ratio, box_pts, debug_mask)
    where box_pts are the 4 corners in full-frame coordinates.
    """
    h_frame, w_frame = frame.shape[:2]
    cx1 = max(0, x1 - PADDING)
    cy1 = max(0, y1 - PADDING)
    cx2 = min(w_frame - 1, x2 + PADDING)
    cy2 = min(h_frame - 1, y2 + PADDING)
    crop = frame[cy1:cy2, cx1:cx2]

    if crop.size == 0:
        return False, 0, 0, 0.0, 1.0, None, None

    crop_area = max((cx2 - cx1) * (cy2 - cy1), 1)

    # ── Edge mask ──────────────────────────────────────────────────────────
    gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 20, 80)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask  = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # ── Largest contour ────────────────────────────────────────────────────
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, 0, 0, 0.0, 1.0, None, mask

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 0.02 * crop_area:   # 2% min
        return False, 0, 0, 0.0, 1.0, None, mask

    # ── Rotated rect ───────────────────────────────────────────────────────
    rect = cv2.minAreaRect(largest)
    (rect_cx, rect_cy), (rw, rh), angle = rect

    # Normalise: width = longer side
    if rw < rh:
        angle += 90
        rw, rh = rh, rw

    aspect_ratio  = round(rw / rh, 2) if rh > 0 else 1.0
    gripper_angle = angle % 180

    full_cx = int(rect_cx + cx1)
    full_cy = int(rect_cy + cy1)
    box_pts = (cv2.boxPoints(rect).astype(np.intp)
               + np.array([cx1, cy1], dtype=np.intp))

    return True, full_cx, full_cy, gripper_angle, aspect_ratio, box_pts, mask


# ── Drawing helpers ────────────────────────────────────────────────────────

def draw_axes(frame, cx, cy, angle_deg, axis_len=70):
    rad = np.radians(angle_deg)

    # X — red
    x_end = (int(cx + axis_len * np.cos(rad)),
              int(cy + axis_len * np.sin(rad)))
    cv2.arrowedLine(frame, (cx, cy), x_end, (0, 0, 220), 2, tipLength=0.25)
    cv2.putText(frame, "X", (x_end[0] + 4, x_end[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 2)

    # Y — green (90° CCW from X)
    y_rad = rad - np.pi / 2
    y_end = (int(cx + axis_len * np.cos(y_rad)),
              int(cy + axis_len * np.sin(y_rad)))
    cv2.arrowedLine(frame, (cx, cy), y_end, (0, 200, 0), 2, tipLength=0.25)
    cv2.putText(frame, "Y", (y_end[0] + 4, y_end[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)

    # Z — blue dot (out of plane)
    cv2.circle(frame, (cx, cy), 8, (220, 100, 0), -1)
    cv2.circle(frame, (cx, cy), 8, (255, 180, 60), 1)
    cv2.putText(frame, "Z", (cx + 10, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 100, 0), 2)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print(f"Loading model: {MODEL_PATH.name} …")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Weights not found: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH), task="detect")
    print(f"Classes: {model.names}\n")

    print("Opening webcam …")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0)")
    print("Webcam ready.  Press Q to quit.\n")

    fps_t, fps_count, fps = time.time(), 0, 0.0
    last_mask = None   # shown in debug window

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, imgsz=640, conf=CONF_THRESHOLD, verbose=False)
        r = results[0]

        det_count = 0
        if r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                colour = BOX_COLOURS[cls_id % len(BOX_COLOURS)]
                label  = model.names[cls_id]
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                valid, cx, cy, angle, ar, box_pts, mask = fit_rotated_box(
                    frame, x1, y1, x2, y2)

                if mask is not None:
                    last_mask = mask

                if valid:
                    cv2.drawContours(frame, [box_pts], 0, colour, 2)
                    draw_axes(frame, cx, cy, angle)
                    info = (f"{label}  {conf*100:.0f}%  "
                            f"angle={angle:.1f}deg  AR={ar:.2f}")
                    cv2.putText(frame, info, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50, colour, 2)
                else:
                    # Contour failed — plain box so at least detection is visible
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                    cv2.putText(frame, f"{label}  {conf*100:.0f}%  (no contour)",
                                (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50, colour, 2)

                det_count += 1

        # Detection count
        cv2.putText(frame, f"{det_count} detection{'s' if det_count != 1 else ''}",
                    (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 220, 0) if det_count else (0, 80, 220), 2)

        # FPS
        fps_count += 1
        now = time.time()
        if now - fps_t >= 1.0:
            fps = fps_count / (now - fps_t)
            fps_count = 0
            fps_t = now
        cv2.putText(frame, f"{fps:.1f} fps",
                    (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

        cv2.imshow("OBB test — Q to quit", frame)

        # Debug window: edge mask for last detection
        if last_mask is not None:
            debug = cv2.resize(cv2.cvtColor(last_mask, cv2.COLOR_GRAY2BGR),
                               (240, 180))
            cv2.putText(debug, "edge mask", (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 220), 1)
            cv2.imshow("debug mask", debug)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
