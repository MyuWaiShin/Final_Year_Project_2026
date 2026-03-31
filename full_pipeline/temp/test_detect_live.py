"""
test_detect_live.py
-------------------
Live inference test for the YOLO cylinder detector.

Opens the OAK-D camera and runs the detection model on every frame,
drawing bounding boxes, class labels, and confidence scores.

Usage
-----
    python temp/test_detect_live.py

    # Lower confidence threshold (default: 0.25)
    python temp/test_detect_live.py --conf 0.4

Controls
--------
    Q  →  quit
"""

import argparse
import time
from pathlib import Path

import cv2
import depthai as dai
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
WEIGHTS     = SCRIPT_DIR.parent.parent / "Train" / "weights" / "best.pt"

# Colour per class index (cycles if more classes than entries)
BOX_COLOURS = [
    (0, 220, 0),    # class 0 — green
    (0, 100, 255),  # class 1 — orange
    (255, 60, 60),  # class 2 — blue
    (220, 0, 220),  # class 3 — magenta
]


def open_camera():
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setVideoSize(1280, 720)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("video")
    cam.video.link(xout.input)
    device = dai.Device(pipeline)
    queue  = device.getOutputQueue(name="video", maxSize=4, blocking=False)
    return device, queue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.25,
                    help="Detection confidence threshold (default: 0.25)")
    args = ap.parse_args()

    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Weights not found: {WEIGHTS}")

    print(f"Loading model: {WEIGHTS.name} …")
    model = YOLO(str(WEIGHTS), task="detect")
    print(f"Classes: {model.names}\n")

    print("Opening camera …")
    device, queue = open_camera()
    print("Camera ready.  Press Q to quit.\n")

    fps_t     = time.time()
    fps_count = 0
    fps       = 0.0

    while True:
        pkt   = queue.get()
        frame = pkt.getCvFrame()

        # ── Inference ────────────────────────────────────────────────────────
        results = model(frame, imgsz=640, conf=args.conf, verbose=False)
        r = results[0]

        det_count = 0
        if r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label  = f"{model.names[cls_id]}  {conf*100:.1f}%"
                colour = BOX_COLOURS[cls_id % len(BOX_COLOURS)]

                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                det_count += 1

        # ── Detection count overlay ──────────────────────────────────────────
        det_label = f"{det_count} detection{'s' if det_count != 1 else ''}"
        cv2.putText(frame, det_label, (10, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 220, 0) if det_count else (0, 80, 220), 2)

        # ── FPS counter ──────────────────────────────────────────────────────
        fps_count += 1
        now = time.time()
        if now - fps_t >= 1.0:
            fps       = fps_count / (now - fps_t)
            fps_count = 0
            fps_t     = now
        cv2.putText(frame, f"{fps:.1f} fps", (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

        cv2.imshow("YOLO cylinder detector — Q to quit",
                   cv2.resize(frame, (960, 540)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    device.close()
    print("Done.")


if __name__ == "__main__":
    main()
