"""
test_verify_live.py
-------------------
Live comparison of CLIP probe v2 vs YOLO26n classifier — the two models
used in the verify stage.

Both run on every frame and their results are overlaid side-by-side so you
can see where they agree and disagree in real time.

Usage
-----
    python temp/test_verify_live.py

    # Change CLIP uncertainty threshold (default: 0.85)
    python temp/test_verify_live.py --threshold 0.70

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
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR      = Path(__file__).resolve().parent
MODELS_DIR      = SCRIPT_DIR.parent / "models"

CLIP_PROBE_PATH = MODELS_DIR / "clip"           / "v2" / "clip_probe_v1.pkl"
YOLO_PATH       = MODELS_DIR / "classification" / "yolo26n_cls_V1" / "weights" / "best.pt"

# CLIP crop — bottom-centre of 1920×1080, must match training exactly
CROP_W, CROP_H  = 1400, 600


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


# ── Inference helpers ───────────────────────────────────────────────────────
def get_crop(frame: np.ndarray):
    h, w = frame.shape[:2]
    cx = w // 2
    cy = h - (CROP_H // 2) - 10
    x1 = max(0, cx - CROP_W // 2)
    y1 = max(0, cy - CROP_H // 2)
    x2 = min(w, x1 + CROP_W)
    y2 = min(h, y1 + CROP_H)
    return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def run_clip(crop, clip_model, preprocess, clf, device, threshold):
    pil  = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    inp  = preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = clip_model.encode_image(inp)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    probs     = clf.predict_proba(feat.cpu().numpy())[0]
    p_holding = float(probs[1])
    p_empty   = float(probs[0])
    label     = "Holding" if p_holding >= 0.5 else "Empty"
    conf      = p_holding if label == "Holding" else p_empty
    uncertain = conf < threshold
    return label, conf, p_holding, uncertain


def run_yolo(frame, yolo_model):
    results   = yolo_model(frame, imgsz=224, verbose=False)
    r         = results[0]
    probs     = {r.names[i]: float(r.probs.data[i]) for i in range(len(r.names))}
    p_holding = probs.get("holding", 0.0)
    p_empty   = probs.get("empty",   0.0)
    label     = "Holding" if p_holding >= 0.5 else "Empty"
    conf      = p_holding if label == "Holding" else p_empty
    return label, conf, p_holding


# ── Overlay ─────────────────────────────────────────────────────────────────
def colour_for(label, uncertain=False):
    if uncertain:
        return (0, 165, 255)   # orange
    return (0, 220, 0) if label == "Holding" else (0, 0, 220)


def draw_panel(display, clip_label, clip_conf, clip_uncertain, clip_p_holding,
               yolo_label, yolo_conf, yolo_p_holding,
               crop_box_scaled, clip_ms, yolo_ms):

    x1, y1, x2, y2 = crop_box_scaled

    # Crop region rectangle
    cv2.rectangle(display, (x1, y1), (x2, y2), (255, 140, 0), 2)

    # ── CLIP block (left side) ────────────────────────────────────────────
    cc = colour_for(clip_label, clip_uncertain)
    clip_disp = f"Uncertain ({clip_label})" if clip_uncertain else clip_label
    cv2.putText(display, "CLIP v2",
                (20, 36), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(display, clip_disp,
                (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 1.2, cc, 3)
    cv2.putText(display, f"conf {clip_conf*100:.1f}%  |  p_hold {clip_p_holding:.3f}",
                (20, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.58, cc, 2)
    cv2.putText(display, f"{clip_ms:.0f} ms",
                (20, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # Divider
    cv2.line(display, (480, 20), (480, 145), (80, 80, 80), 1)

    # ── YOLO block (right side) ───────────────────────────────────────────
    yc = colour_for(yolo_label)
    cv2.putText(display, "YOLO26n",
                (500, 36), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(display, yolo_label,
                (500, 78), cv2.FONT_HERSHEY_SIMPLEX, 1.2, yc, 3)
    cv2.putText(display, f"conf {yolo_conf*100:.1f}%  |  p_hold {yolo_p_holding:.3f}",
                (500, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.58, yc, 2)
    cv2.putText(display, f"{yolo_ms:.0f} ms",
                (500, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # ── Agreement indicator ───────────────────────────────────────────────
    agree = (clip_label == yolo_label) and not clip_uncertain
    agree_col   = (0, 220, 0) if agree else (0, 60, 220)
    agree_label = "AGREE" if agree else "DISAGREE"
    cv2.putText(display, agree_label,
                (display.shape[1] - 150, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, agree_col, 2)


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.85,
                    help="CLIP confidence below this → Uncertain (default: 0.85)")
    args = ap.parse_args()

    # ── Load models ──────────────────────────────────────────────────────────
    print("Loading CLIP probe v2 …")
    if not CLIP_PROBE_PATH.exists():
        raise FileNotFoundError(f"CLIP probe not found: {CLIP_PROBE_PATH}")
    with open(CLIP_PROBE_PATH, "rb") as f:
        clf = pickle.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP ViT-B/32 on {device} …")
    clip_model, preprocess = openai_clip.load("ViT-B/32", device=device)

    print("Loading YOLO26n classifier …")
    if not YOLO_PATH.exists():
        raise FileNotFoundError(f"YOLO weights not found: {YOLO_PATH}")
    yolo_model = YOLO(str(YOLO_PATH), task="classify")
    print(f"  classes: {yolo_model.names}\n")

    # ── Open camera ──────────────────────────────────────────────────────────
    print("Opening camera …")
    cam_device, queue = open_camera()
    print("Camera ready.  Press Q to quit.\n")

    fps_t, fps_count, fps = time.time(), 0, 0.0

    while True:
        frame = queue.get().getCvFrame()
        crop, (x1, y1, x2, y2) = get_crop(frame)

        # CLIP
        t0 = time.time()
        clip_label, clip_conf, clip_p_hold, clip_uncertain = run_clip(
            crop, clip_model, preprocess, clf, device, args.threshold)
        clip_ms = (time.time() - t0) * 1000

        # YOLO (runs on full frame, same as verify.py)
        t0 = time.time()
        yolo_label, yolo_conf, yolo_p_hold = run_yolo(frame, yolo_model)
        yolo_ms = (time.time() - t0) * 1000

        # Scale crop box to display resolution
        display = cv2.resize(frame, (960, 540))
        sx, sy  = 960 / frame.shape[1], 540 / frame.shape[0]
        box_scaled = (int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy))

        draw_panel(display,
                   clip_label, clip_conf, clip_uncertain, clip_p_hold,
                   yolo_label, yolo_conf, yolo_p_hold,
                   box_scaled, clip_ms, yolo_ms)

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

        cv2.imshow("CLIP v2  vs  YOLO26n", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cam_device.close()
    print("Done.")


if __name__ == "__main__":
    main()
