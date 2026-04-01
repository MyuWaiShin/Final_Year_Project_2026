"""
test_cls_live.py
----------------
Live inference test for the YOLO binary classifiers (empty / holding).

Opens the OAK-D camera and runs the selected model on every frame,
overlaying the predicted class and confidence on screen.

Usage
-----
    # Test yolov8n (default)
    python temp/test_cls_live.py

    # Test yolo26n
    python temp/test_cls_live.py --model v26

    # Test both side-by-side
    python temp/test_cls_live.py --model both

Controls
--------
    Q  →  quit
"""

import argparse
import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np
from ultralytics import YOLO

# ── Paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR.parent / "models" / "classification"

MODEL_PATHS = {
    "v8":  MODELS_DIR / "yolov8n_cls_V1" / "weights" / "best.pt",
    "v26": MODELS_DIR / "yolo26n_cls_V1" / "weights" / "best.pt",
}

# Colour per class: green = holding, red = empty
CLASS_COLOURS = {
    "holding": (0, 255, 0),
    "empty":   (0, 0, 255),
}


def load_model(key: str) -> YOLO:
    path = MODEL_PATHS[key]
    if not path.exists():
        raise FileNotFoundError(f"Weights not found: {path}")
    print(f"  Loading {key}: {path.name} …")
    return YOLO(str(path), task="classify")


def run_inference(model: YOLO, frame: np.ndarray):
    """
    Run classify inference on a single BGR frame.
    Returns (class_name, confidence).
    """
    results = model(frame, imgsz=224, verbose=False)
    r = results[0]
    idx  = int(r.probs.top1)
    conf = float(r.probs.top1conf)
    name = r.names[idx]
    return name, conf


def overlay(frame: np.ndarray, label: str, conf: float,
            colour, x: int = 10, y: int = 35):
    text = f"{label.upper()}  {conf*100:.1f}%"
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, colour, 3)


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
    ap.add_argument("--model", choices=["v8", "v26", "both"], default="v8",
                    help="Which model to test (default: v8)")
    args = ap.parse_args()

    print("Loading model(s) …")
    if args.model == "both":
        models = {k: load_model(k) for k in ("v8", "v26")}
    else:
        models = {args.model: load_model(args.model)}
    print("Models ready.\n")

    print("Opening camera …")
    device, queue = open_camera()
    print("Camera ready.  Press Q to quit.\n")

    fps_t = time.time()
    fps_count = 0
    fps = 0.0

    while True:
        pkt   = queue.get()
        frame = pkt.getCvFrame()

        # ── Run inference ────────────────────────────────────────────────
        results = {}
        for key, model in models.items():
            results[key] = run_inference(model, frame)

        # ── Overlay ──────────────────────────────────────────────────────
        if args.model == "both":
            name_v8,  conf_v8  = results["v8"]
            name_v26, conf_v26 = results["v26"]
            col_v8  = CLASS_COLOURS.get(name_v8,  (255, 255, 255))
            col_v26 = CLASS_COLOURS.get(name_v26, (255, 255, 255))
            overlay(frame, f"v8:  {name_v8}",  conf_v8,  col_v8,  x=10, y=45)
            overlay(frame, f"v26: {name_v26}", conf_v26, col_v26, x=10, y=90)
        else:
            key = args.model
            name, conf = results[key]
            colour = CLASS_COLOURS.get(name, (255, 255, 255))
            overlay(frame, name, conf, colour)

        # ── FPS counter ──────────────────────────────────────────────────
        fps_count += 1
        now = time.time()
        if now - fps_t >= 1.0:
            fps = fps_count / (now - fps_t)
            fps_count = 0
            fps_t = now
        cv2.putText(frame, f"{fps:.1f} fps", (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

        cv2.imshow("YOLO classifier", cv2.resize(frame, (960, 540)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    device.close()
    print("Done.")


if __name__ == "__main__":
    main()
