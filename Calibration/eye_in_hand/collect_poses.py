"""
collect_poses.py — Eye-in-Hand Calibration: Interactive Data Collection
=======================================================================
Streams RGB from the OAK-D Lite and detects a checkerboard in real time.

CONTROLS
  C ............. Capture current frame + robot TCP pose (only when board detected)
  D ............. Delete the LAST captured sample
  S ............. Save all samples to disk and exit
  Q ............. Quit WITHOUT saving

OUTPUT (in DATA_DIR)
  images/image_000.png, image_001.png, ...
  poses.json   <- everything needed by solve_calibration.py

TIPS
  . Aim for 20-40 samples.
  . Vary XY translation, Z height, wrist tilt +/-20 degrees.
  . The board should fill at least 1/3 of the frame.
  . Keep the board STILL when pressing C -- any motion = blur = bad corners.
"""

import os
os.environ["DEPTHAI_BOOT_TIMEOUT"] = "30000"

import json
import socket
import struct
import threading
import time
from pathlib import Path

import cv2
import cv2.aruco as aruco
import depthai as dai
import numpy as np

# -----------------------------------------------------------------------------
# USER CONFIGURATION -- edit these before running
# -----------------------------------------------------------------------------
ROBOT_IP    = "192.168.8.102"
DATA_DIR    = Path("data")

# ChArUco Board Configuration (7x5 squares)
BOARD_COLS  = 7
BOARD_ROWS  = 5
SQUARE_M    = 0.038  # <- confirmed by user
MARKER_M    = 0.028

# Camera resolution (saved image resolution)
CAM_W, CAM_H = 1280, 720
# -----------------------------------------------------------------------------

# Initialize Charuco Board
ARUCO_DICT   = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
CHARUCO_BRD  = aruco.CharucoBoard((BOARD_COLS, BOARD_ROWS), SQUARE_M, MARKER_M, ARUCO_DICT)
CHARUCO_DETECTOR = aruco.CharucoDetector(CHARUCO_BRD)

# CLAHE for contrast enhancement (helps on grey/low-contrast boards)
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Detection runs on a smaller image for speed, but too small = flickering.
# 854x480 is a good middle ground (1.5x downscale from 720p).
DETECT_W, DETECT_H = 854, 480
SCALE_X = CAM_W / DETECT_W
SCALE_Y = CAM_H / DETECT_H


# -----------------------------------------------------------------------------
# Background TCP reader -- never blocks the camera loop
# -----------------------------------------------------------------------------
class TCPPoller(threading.Thread):
    """Polls the robot TCP pose in a background thread every ~200 ms."""
    def __init__(self, ip=ROBOT_IP):
        super().__init__(daemon=True)
        self.ip    = ip
        self.pose  = None   # [x, y, z, rx, ry, rz] or None
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            self.pose = self._read()
            time.sleep(0.2)

    def _read(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)   # short timeout -- never blocks more than 0.5 s
                s.connect((self.ip, 30003))
                data = s.recv(1200)
                if len(data) < 492:
                    return None
                return list(struct.unpack("!6d", data[444:492]))
        except Exception:
            return None

    def stop(self):
        self._stop.set()


# -----------------------------------------------------------------------------
# OAK-D Lite pipeline (RGB video only)
# -----------------------------------------------------------------------------
def create_pipeline() -> dai.Pipeline:
    p   = dai.Pipeline()
    cam = p.create(dai.node.ColorCamera)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setVideoSize(CAM_W, CAM_H)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(30) # Back to 30 FPS since MJPEG is so lightweight

    # Hardware MJPEG encoding (drastically reduces USB bandwidth to prevent XLink crashes)
    videoEnc = p.create(dai.node.VideoEncoder)
    videoEnc.setDefaultProfilePreset(30, dai.VideoEncoderProperties.Profile.MJPEG)
    videoEnc.setQuality(95) # High quality for calibration
    cam.video.link(videoEnc.input)

    xout = p.create(dai.node.XLinkOut)
    xout.setStreamName("mjpeg")
    videoEnc.bitstream.link(xout.input)
    
    return p


# -----------------------------------------------------------------------------
# HUD
# -----------------------------------------------------------------------------
def draw_hud(frame, charuco_corners, charuco_ids, n_samples, tcp, fps, msg=""):
    disp = frame.copy()
    h, w = disp.shape[:2]

    # Draw detected Charuco corners
    board_ok = False
    if charuco_corners is not None and len(charuco_corners) > 3:
        board_ok = True
        aruco.drawDetectedCornersCharuco(disp, charuco_corners, charuco_ids, (0, 255, 0))

    # Board status
    if board_ok:
        status_txt = f"CHARUCO OK ({len(charuco_corners)} corners) -- press C to capture"
        status_col = (0, 255, 80)
    else:
        status_txt = "CHARUCO NOT FOUND (need >=4 corners)"
        status_col = (0, 60, 220)
    cv2.putText(disp, status_txt, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, status_col, 2)

    # Warn if board clips a frame edge (common cause of detection failure)
    if board_ok and charuco_corners is not None:
        pts = charuco_corners.reshape(-1, 2)
        margin = 20
        if (pts[:, 0].min() < margin or pts[:, 0].max() > w - margin or
                pts[:, 1].min() < margin or pts[:, 1].max() > h - margin):
            cv2.putText(disp, "!! Board too close to edge -- move robot !!",
                        (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 2)

    # FPS -- colour coded
    fps_col = (0, 220, 0) if fps >= 20 else (0, 160, 255) if fps >= 10 else (0, 0, 220)
    cv2.putText(disp, f"FPS: {fps:.1f}",
                (w - 130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, fps_col, 2)

    # Samples + TCP
    cv2.putText(disp, f"Samples: {n_samples}",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 200, 200), 1)
    if tcp:
        x, y, z = tcp[0]*1000, tcp[1]*1000, tcp[2]*1000
        cv2.putText(disp, f"TCP  X:{x:+.1f}  Y:{y:+.1f}  Z:{z:+.1f} mm",
                    (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 200, 255), 1)
    else:
        cv2.putText(disp, "TCP: robot not connected",
                    (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 60, 255), 1)

    # Board config hint
    cv2.putText(disp,
                f"Board config: {BOARD_COLS}x{BOARD_ROWS} inner corners, {SQUARE_M*1000:.0f}mm sq",
                (10, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1)

    # Controls
    cv2.putText(disp, "C=Capture  D=Delete last  S=Save & Exit  Q=Quit",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)

    if msg:
        # Use a contrasting box so the flash message is super clear
        cv2.rectangle(disp, (8, 120), (400, 155), (0, 0, 0), -1)
        cv2.putText(disp, msg, (15, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)

    return disp


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    img_dir = DATA_DIR / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    samples     = []
    flash_msg   = ""
    flash_until = 0.0
    
    # Simple rolling average for FPS
    prev_time   = time.time()
    fps_smooth  = 12.0

    # Start background TCP poller (never stalls the camera loop)
    tcp_poller = TCPPoller(ROBOT_IP)
    tcp_poller.start()
    print("[TCP] Background pose poller started.")

    pipeline = create_pipeline()

    # Allow Auto-Negotiation for USB Speed (SUPER/HIGH)
    cfg = dai.Device.Config()

    print(f"\n=== Eye-in-Hand Calibration: Data Collection ===")
    print(f"  Board: {BOARD_COLS}x{BOARD_ROWS} inner corners, {SQUARE_M*1000:.0f} mm squares")
    print(f"  Detection resolution: {DETECT_W}x{DETECT_H}  (faster than full res)")
    print(f"  C=Capture  D=Delete last  S=Save  Q=Quit\n")

    try:
        with dai.Device(cfg) as device:
            device.startPipeline(pipeline)
            q = device.getOutputQueue("mjpeg", maxSize=2, blocking=False)

            # OAK-D factory intrinsics -- scaled to our specific resolution
            cal      = device.readCalibration()
            M        = cal.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, CAM_W, CAM_H)
            dist     = cal.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A)
            fx, fy, cx, cy = M[0][0], M[1][1], M[0][2], M[1][2]
            print(f"[Camera] fx={fx:.1f}  fy={fy:.1f}  cx={cx:.1f}  cy={cy:.1f}")

            camera_matrix = np.array([[fx,  0, cx],
                                       [ 0, fy, cy],
                                       [ 0,  0,  1]], dtype=np.float64)
            dist_coeffs   = np.array(dist, dtype=np.float64)

            while True:
                in_encoded = q.get()
                frame = cv2.imdecode(in_encoded.getData(), cv2.IMREAD_COLOR)   # Decode MJPEG on CPU

                if frame is None:
                    continue

                # Downscale for fast detection
                small = cv2.resize(frame, (DETECT_W, DETECT_H))
                gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

                # Detect ArUco markers and interpolate Charuco corners
                charuco_corners_small, charuco_ids, marker_corners, marker_ids = CHARUCO_DETECTOR.detectBoard(gray)
                
                charuco_corners_full = None
                found = False

                if charuco_corners_small is not None and len(charuco_corners_small) > 3:
                    found = True
                    # Scale corners back to full resolution
                    charuco_corners_full = charuco_corners_small.copy()
                    charuco_corners_full[:, :, 0] *= SCALE_X
                    charuco_corners_full[:, :, 1] *= SCALE_Y

                tcp = tcp_poller.pose   # non-blocking cached value

                # FPS (Exponential moving average)
                now = time.time()
                dt = now - prev_time
                prev_time = now
                if dt > 0:
                    fps_smooth = (fps_smooth * 0.9) + (1.0 / dt * 0.1)

                cur_msg = flash_msg if now < flash_until else ""

                disp = draw_hud(frame, charuco_corners_full, charuco_ids, len(samples),
                                tcp, fps_smooth, cur_msg)
                cv2.imshow("Collect Poses -- Eye-in-Hand Calibration", disp)

                key = cv2.waitKey(1) & 0xFF

                # C: Capture
                if key == ord('c'):
                    if not found:
                        flash_msg   = "Board NOT visible -- can't capture"
                        flash_until = now + 2.0
                        print("[Capture] Board not detected.")
                    elif tcp is None:
                        flash_msg   = "Robot disconnected -- can't capture"
                        flash_until = now + 2.0
                        print("[Capture] No robot pose.")
                    else:
                        idx   = len(samples)
                        fname = f"image_{idx:03d}.png"
                        cv2.imwrite(str(img_dir / fname), frame)

                        samples.append({
                            "id":              idx,
                            "image":           f"images/{fname}",
                            "charuco_corners": charuco_corners_full.reshape(-1, 2).tolist(),
                            "charuco_ids":     charuco_ids.flatten().tolist(),
                            "tcp_pose":        tcp,
                        })
                        flash_msg   = f"Captured #{idx:03d}   ({len(samples)} total)"
                        flash_until = now + 1.5
                        print(f"[Capture] #{idx:03d}  "
                              f"TCP=({tcp[0]*1000:+.1f},{tcp[1]*1000:+.1f},{tcp[2]*1000:+.1f}) mm")

                # D: Delete last
                elif key == ord('d'):
                    if samples:
                        removed = samples.pop()
                        p = DATA_DIR / removed["image"]
                        if p.exists():
                            p.unlink()
                        flash_msg   = f"Deleted sample #{removed['id']:03d}"
                        flash_until = now + 1.5
                        print(f"[Delete] Removed #{removed['id']:03d}")
                    else:
                        print("[Delete] Nothing to delete.")

                # S: Save and exit
                elif key == ord('s'):
                    if len(samples) < 5:
                        flash_msg   = f"Need >=5 samples (have {len(samples)})"
                        flash_until = now + 2.0
                    else:
                        _save(samples, camera_matrix, dist_coeffs)
                        break

                # Q: Quit
                elif key == ord('q'):
                    print("Quit -- nothing saved.")
                    break

    finally:
        tcp_poller.stop()
        cv2.destroyAllWindows()


def _save(samples, camera_matrix, dist_coeffs):
    poses_path = DATA_DIR / "poses.json"
    with open(poses_path, "w") as f:
        json.dump({
            "board":   {"type": "charuco", "cols": BOARD_COLS, "rows": BOARD_ROWS, 
                        "square_m": SQUARE_M, "marker_m": MARKER_M,
                        "dict": "DICT_6X6_250"},
            "camera":  {"width": CAM_W, "height": CAM_H,
                        "matrix":     camera_matrix.tolist(),
                        "distortion": dist_coeffs.tolist()},
            "samples": samples,
            "timestamp": time.ctime(),
        }, f, indent=2)
    print(f"\n  Saved {len(samples)} samples to {poses_path}")
    print(f"  Images: {DATA_DIR / 'images'}/")
    print("  Next: python solve_calibration.py")


if __name__ == "__main__":
    main()
