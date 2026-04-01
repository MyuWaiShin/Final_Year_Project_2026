"""
test_detect_live.py
-------------------
Live inference test for the YOLO object detection models (cube / cylinder).

Opens the OAK-D camera and runs the selected model on every frame,
drawing bounding boxes, class labels, and confidence scores.

Usage
-----
    # Test yolov8n (default)
    python temp/test_detect_live.py

    # Test yolo26n
    python temp/test_detect_live.py --model v26

    # Test both side-by-side (stacked vertically)
    python temp/test_detect_live.py --model both

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
MODELS_DIR  = SCRIPT_DIR.parent / "models" / "detection"

MODEL_PATHS = {
    "v8":  MODELS_DIR / "yolov8n_detect_V1" / "weights" / "best.pt",
    "v26": MODELS_DIR / "yolo26n_detect_V1"  / "weights" / "best.pt",
}

# Colour per class index
BOX_COLOURS = [
    (0, 220, 0),    # class 0 — green
    (0, 100, 255),  # class 1 — orange
    (255, 60, 60),  # class 2 — blue
    (220, 0, 220),  # class 3 — magenta
]


def load_model(key: str) -> YOLO:
    path = MODEL_PATHS[key]
    if not path.exists():
        raise FileNotFoundError(f"Weights not found: {path}")
    print(f"  Loading {key}: {path.parent.parent.name} …")
    return YOLO(str(path), task="detect")


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


def draw_detections(frame, results, model, label_prefix: str = ""):
    """Draw boxes on frame in-place. Returns detection count."""
    r = results[0]
    count = 0
    if r.boxes is None:
        return count
    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        colour = BOX_COLOURS[cls_id % len(BOX_COLOURS)]
        label  = f"{label_prefix}{model.names[cls_id]}  {conf*100:.1f}%"

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        count += 1
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["v8", "v26", "both"], default="v8",
                    help="Which model to test (default: v8)")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="Detection confidence threshold (default: 0.25)")
    args = ap.parse_args()

    print("Loading model(s) …")
    if args.model == "both":
        models = {k: load_model(k) for k in ("v8", "v26")}
    else:
        models = {args.model: load_model(args.model)}
    for key, m in models.items():
        print(f"  {key} classes: {m.names}")
    print()

    print("Opening camera …")
    device, queue = open_camera()
    print("Camera ready.  Press Q to quit.\n")

    fps_t     = time.time()
    fps_count = 0
    fps       = 0.0

    while True:
        pkt   = queue.get()
        frame = pkt.getCvFrame()

        if args.model == "both":
            # Run both, stack results vertically
            frame_v8  = frame.copy()
            frame_v26 = frame.copy()

            res_v8  = models["v8"](frame,  imgsz=640, conf=args.conf, verbose=False)
            res_v26 = models["v26"](frame, imgsz=640, conf=args.conf, verbose=False)

            n_v8  = draw_detections(frame_v8,  res_v8,  models["v8"],  label_prefix="v8:  ")
            n_v26 = draw_detections(frame_v26, res_v26, models["v26"], label_prefix="v26: ")

            cv2.putText(frame_v8, f"v8  — {n_v8} det",
                        (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 220, 0) if n_v8 else (0, 80, 220), 2)
            cv2.putText(frame_v26, f"v26 — {n_v26} det",
                        (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 220, 0) if n_v26 else (0, 80, 220), 2)

            # FPS on top panel only
            fps_count += 1
            now = time.time()
            if now - fps_t >= 1.0:
                fps = fps_count / (now - fps_t)
                fps_count = 0
                fps_t = now
            cv2.putText(frame_v8, f"{fps:.1f} fps",
                        (10, frame_v8.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

            half_h = 270  # 540 / 2
            display = cv2.vconcat([
                cv2.resize(frame_v8,  (960, half_h)),
                cv2.resize(frame_v26, (960, half_h)),
            ])
            cv2.imshow("YOLO detection", display)

        else:
            key   = args.model
            model = models[key]
            res   = model(frame, imgsz=640, conf=args.conf, verbose=False)
            count = draw_detections(frame, res, model)

            det_label = f"{count} detection{'s' if count != 1 else ''}"
            cv2.putText(frame, det_label, (10, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 220, 0) if count else (0, 80, 220), 2)

            fps_count += 1
            now = time.time()
            if now - fps_t >= 1.0:
                fps = fps_count / (now - fps_t)
                fps_count = 0
                fps_t = now
            cv2.putText(frame, f"{fps:.1f} fps",
                        (10, frame.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

            cv2.imshow("YOLO detection", cv2.resize(frame, (960, 540)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    device.close()
    print("Done.")


if __name__ == "__main__":
    main()
