"""
collect_videos.py
-----------------
Records a fixed 30-second clip per class from the OAK-D RGB camera and saves
the raw video to:

    temp/videos/<class_name>/<class_name>_<timestamp>.mp4

Requires:
    pip install depthai opencv-python

Usage
-----
    python collect_videos.py

Controls (in the preview window)
---------------------------------
    SPACE  –  start the 30-second recording
    Q      –  cancel current class / quit
"""

import cv2
import time
import depthai as dai
from pathlib import Path
from datetime import datetime

# ── config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
VIDEO_DIR    = SCRIPT_DIR / "videos"

RECORD_SECS  = 30
FPS          = 30
RESOLUTION   = (1920, 1080)   # OAK-D 1080p; change to (3840, 2160) for 4K
CODEC        = "mp4v"
EXTENSION    = ".mp4"
# ─────────────────────────────────────────────────────────────────────────────


def build_pipeline() -> dai.Pipeline:
    """Build a depthai pipeline that streams the OAK-D RGB colour camera."""
    pipeline = dai.Pipeline()

    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setFps(FPS)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("rgb")
    cam_rgb.video.link(xout.input)

    return pipeline


def record_clip(class_name: str, queue: dai.DataOutputQueue) -> bool:
    """
    Phase 1 – show live preview until SPACE pressed.
    Phase 2 – record RECORD_SECS seconds.
    Returns True on success, False if cancelled.
    """
    class_dir = VIDEO_DIR / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: live preview ─────────────────────────────────────────────────
    print(f"\n[READY] '{class_name}'  –  frame your shot then press SPACE  |  Q = cancel")
    while True:
        packet = queue.get()
        frame  = packet.getCvFrame()
        frame  = cv2.resize(frame, RESOLUTION)

        preview = frame.copy()
        cv2.putText(preview, f"Class: {class_name}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(preview, "SPACE = start recording   Q = cancel", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 0), 2)
        cv2.imshow("OAK-D Collector", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            break
        if key in (ord("q"), ord("Q")):
            print("[CANCEL] Cancelled before recording.")
            return False

    # ── Phase 2: record ───────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = class_dir / f"{class_name}_{timestamp}{EXTENSION}"

    fourcc = cv2.VideoWriter_fourcc(*CODEC)
    writer = cv2.VideoWriter(str(save_path), fourcc, FPS, RESOLUTION)

    total_frames = RECORD_SECS * FPS
    written      = 0
    start_time   = time.time()
    print(f"[REC]  {save_path.name}  ({RECORD_SECS}s)")

    while written < total_frames:
        packet = queue.get()
        frame  = packet.getCvFrame()
        frame  = cv2.resize(frame, RESOLUTION)

        writer.write(frame)
        written += 1

        elapsed   = time.time() - start_time
        remaining = max(0.0, RECORD_SECS - elapsed)
        filled    = int(30 * elapsed / RECORD_SECS)
        bar       = "█" * filled + "░" * (30 - filled)

        display = frame.copy()
        cv2.putText(display, f"Class: {class_name}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(display, f"● REC  {remaining:.1f}s left", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
        cv2.putText(display, bar, (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        cv2.putText(display, "Q = cancel", (10, RESOLUTION[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        cv2.imshow("OAK-D Collector", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            print("[CANCEL] Clip cancelled mid-recording.")
            writer.release()
            save_path.unlink(missing_ok=True)
            return False

    writer.release()
    print(f"[DONE]  {written} frames saved → {save_path}")
    return True


def main():
    print("=" * 55)
    print("  YOLO Data Collection – OAK-D RGB Recorder")
    print("=" * 55)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    pipeline = build_pipeline()

    with dai.Device(pipeline) as device:
        queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        print(f"[INFO] OAK-D connected  |  {RESOLUTION[0]}x{RESOLUTION[1]} @ {FPS}fps")

        try:
            while True:
                class_name = input("\nEnter class name (or ENTER to quit): ").strip()
                if not class_name:
                    print("[EXIT] Done.")
                    break

                record_clip(class_name, queue)

                again = input(f"\nRecord another clip for '{class_name}'? [y/N]: ").strip().lower()
                # "y" → loops back to record_clip for the same class
                # anything else → loops to new class name prompt

        finally:
            cv2.destroyAllWindows()
            print("\n[INFO] Videos saved to:", VIDEO_DIR)
            print("       Run splice_videos.py to extract frames.")


if __name__ == "__main__":
    main()
