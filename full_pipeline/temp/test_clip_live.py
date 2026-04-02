"""
test_clip_live.py
-----------------
Live inference test for the CLIP linear probe classifiers (empty / holding).

Opens the OAK-D camera, crops the bottom-centre gripper region (same crop
used during training), runs the selected probe, and overlays the result.

Usage
-----
    # Test v1 probe (default)
    python temp/test_clip_live.py

    # Test v2 probe
    python temp/test_clip_live.py --model v2

    # Compare both side-by-side (stacked vertically)
    python temp/test_clip_live.py --model both

    # Change uncertainty threshold (default: 0.85)
    python temp/test_clip_live.py --threshold 0.70

Controls
--------
    Q  →  quit
"""

import argparse
import pickle
import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np
import torch
import clip as openai_clip
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR.parent / "models" / "clip"

MODEL_PATHS = {
    "v1": MODELS_DIR / "v1" / "clip_probe_v1.pkl",
    "v2": MODELS_DIR / "v2" / "clip_probe_v1.pkl",
}

# Crop region — must match training crop exactly
CROP_W, CROP_H = 1400, 600   # pixels, from 1920×1080 frame

# Label map: matches probe classes_ = [0, 1]
LABEL_MAP = {0: "Empty", 1: "Holding"}


# ── Camera ─────────────────────────────────────────────────────────────────
def open_camera():
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setVideoSize(1920, 1080)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("video")
    cam.video.link(xout.input)
    device = dai.Device(pipeline)
    queue  = device.getOutputQueue(name="video", maxSize=4, blocking=False)
    return device, queue


# ── Inference ───────────────────────────────────────────────────────────────
def crop_frame(frame: np.ndarray):
    """Return the bottom-centre crop used during probe training."""
    h, w = frame.shape[:2]
    cx   = w // 2
    cy   = h - (CROP_H // 2) - 10
    x1   = max(0, cx - CROP_W // 2)
    y1   = max(0, cy - CROP_H // 2)
    x2   = min(w, x1 + CROP_W)
    y2   = min(h, y1 + CROP_H)
    return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def encode_frame(frame: np.ndarray, clip_model, preprocess, device: str):
    """BGR frame → normalised 512-d CLIP feature vector (numpy)."""
    pil   = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inp   = preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = clip_model.encode_image(inp)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy()


def run_probe(clf, feature: np.ndarray, threshold: float):
    """
    Returns (label_str, confidence, is_uncertain).
    label_map: {0: "Empty", 1: "Holding"}
    """
    pred      = int(clf.predict(feature)[0])
    probs     = clf.predict_proba(feature)[0]
    conf      = float(probs[pred])
    label     = LABEL_MAP[pred]
    uncertain = conf < threshold
    return label, conf, uncertain


# ── Overlay helpers ─────────────────────────────────────────────────────────
def draw_hud(display, label, conf, uncertain, infer_ms,
             crop_box_scaled, prefix: str = ""):
    """Draw result HUD and crop rectangle onto display frame in-place."""
    x1, y1, x2, y2 = crop_box_scaled

    if uncertain:
        colour     = (0, 165, 255)   # orange
        disp_label = f"Uncertain ({label})"
    elif label == "Holding":
        colour     = (0, 220, 0)     # green
        disp_label = "Holding"
    else:
        colour     = (0, 0, 220)     # red
        disp_label = "Empty"

    cv2.rectangle(display, (x1, y1), (x2, y2), (255, 140, 0), 2)

    top = 40 if not prefix else (40 if prefix.startswith("v1") else 130)
    cv2.putText(display, f"{prefix}{disp_label}",
                (20, top), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 3)
    cv2.putText(display, f"conf: {conf*100:.1f}%  |  {infer_ms:.0f} ms",
                (20, top + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["v1", "v2", "both"], default="v1",
                    help="Which probe version to test (default: v1)")
    ap.add_argument("--threshold", type=float, default=0.85,
                    help="Confidence below this → Uncertain (default: 0.85)")
    args = ap.parse_args()

    keys = ("v1", "v2") if args.model == "both" else (args.model,)

    # ── Load probes ──────────────────────────────────────────────────────────
    print("Loading CLIP probe(s) …")
    probes = {}
    for k in keys:
        path = MODEL_PATHS[k]
        if not path.exists():
            raise FileNotFoundError(f"Probe not found: {path}")
        with open(path, "rb") as f:
            probes[k] = pickle.load(f)
        print(f"  {k}: {path.parent.parent.name}/{path.parent.name}/{path.name}")

    # ── Load CLIP backbone ───────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading CLIP ViT-B/32 on {device} …")
    clip_model, preprocess = openai_clip.load("ViT-B/32", device=device)
    print("CLIP ready.\n")

    # ── Open camera ──────────────────────────────────────────────────────────
    print("Opening camera …")
    cam_device, queue = open_camera()
    print("Camera ready.  Press Q to quit.\n")

    fps_t, fps_count, fps = time.time(), 0, 0.0

    while True:
        frame = queue.get().getCvFrame()
        crop, (x1, y1, x2, y2) = crop_frame(frame)

        # Encode once, run both probes on same feature
        t0      = time.time()
        feature = encode_frame(crop, clip_model, preprocess, device)
        infer_ms = (time.time() - t0) * 1000

        # Scale crop box for 960×540 display
        sx, sy = 960 / frame.shape[1], 540 / frame.shape[0]
        box_scaled = (int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy))

        display = cv2.resize(frame, (960, 540))

        if args.model == "both":
            # Run both probes on the same feature
            for k in keys:
                label, conf, uncertain = run_probe(probes[k], feature, args.threshold)
                draw_hud(display, label, conf, uncertain, infer_ms,
                         box_scaled, prefix=f"{k}: ")
        else:
            k = args.model
            label, conf, uncertain = run_probe(probes[k], feature, args.threshold)
            draw_hud(display, label, conf, uncertain, infer_ms, box_scaled)

        # FPS
        fps_count += 1
        now = time.time()
        if now - fps_t >= 1.0:
            fps = fps_count / (now - fps_t)
            fps_count = 0
            fps_t = now
        cv2.putText(display, f"{fps:.1f} fps",
                    (10, display.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

        cv2.imshow("CLIP probe", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cam_device.close()
    print("Done.")


if __name__ == "__main__":
    main()
