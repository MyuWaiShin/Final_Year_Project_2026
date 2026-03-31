"""
test_cls_live.py
----------------
Live inference test for the YOLO binary classifiers (empty / holding).

Opens the OAK-D camera and runs the selected model on every frame,
overlaying the predicted class and confidence on screen.

Usage
-----
    # Test v1 (default)
    python temp/test_cls_live.py

    # Test v2
    python temp/test_cls_live.py --model v2

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
WEIGHTS_DIR = SCRIPT_DIR.parent / "yolo_binary_classifiers"

MODEL_PATHS = {
    "v1": WEIGHTS_DIR / "empty_holding_v1"      / "weights" / "best.pt",
    "v2": WEIGHTS_DIR / "empty_holding_cls_v2"  / "weights" / "best.pt",
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
    ap.add_argument("--model", choices=["v1", "v2", "both"], default="v1",
                    help="Which model to test (default: v1)")
    args = ap.parse_args()

    print("Loading model(s) …")
    if args.model == "both":
        models = {k: load_model(k) for k in ("v1", "v2")}
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
            name_v1, conf_v1 = results["v1"]
            name_v2, conf_v2 = results["v2"]
            col_v1 = CLASS_COLOURS.get(name_v1, (255, 255, 255))
            col_v2 = CLASS_COLOURS.get(name_v2, (255, 255, 255))
            overlay(frame, f"v1: {name_v1}", conf_v1, col_v1, x=10, y=45)
            overlay(frame, f"v2: {name_v2}", conf_v2, col_v2, x=10, y=90)
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

        cv2.imshow("YOLO classifier — Q to quit", cv2.resize(frame, (960, 540)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    device.close()
    print("Done.")


if __name__ == "__main__":
    main()
