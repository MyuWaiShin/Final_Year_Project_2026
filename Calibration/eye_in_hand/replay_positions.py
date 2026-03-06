"""
replay_positions.py — Eye-in-Hand Calibration: Auto-Replay
===========================================================
Loads the TCP poses saved in data/poses.json, moves the robot to each
one automatically, captures the frame, verifies the board is visible,
and then runs the solver.

This lets you REDO calibration without manually repositioning the robot —
just run this script and everything happens automatically.

!! SAFETY !!
  • Always supervise the robot. Keep your hand on the E-stop.
  • The robot uses JOINT-space moves (movej) to avoid Cartesian singularities.
  • Speed is deliberately slow (MOVE_SPEED). Do NOT increase beyond 0.15 m/s.

Run:
    python replay_positions.py

It will:
  1. Connect to robot + camera
  2. Move to each saved pose, wait for settle
  3. Auto-capture if board detected, skip if not
  4. Save new data/poses_replay.json
  5. Run the solver automatically
"""

import os
os.environ["DEPTHAI_BOOT_TIMEOUT"] = "30000"

import json
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import cv2
import cv2.aruco as aruco
import depthai as dai
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
ROBOT_IP   = "192.168.8.102"
ROBOT_PORT_REALTIME = 30003   # read pose
ROBOT_PORT_SCRIPT   = 30002   # send URScript commands

DATA_DIR      = Path("data")
POSES_FILE    = DATA_DIR / "poses.json"
OUT_POSES     = DATA_DIR / "poses_replay.json"
IMG_DIR       = DATA_DIR / "images_replay"

MOVE_SPEED    = 0.05   # m/s — keep low for safety (max recommended 0.10)
MOVE_ACCEL    = 0.3    # m/s² acceleration
SETTLE_TIME   = 1.5    # seconds to wait after reaching a pose
CAPTURE_DELAY = 0.5    # seconds after settle before capturing

CAM_W, CAM_H  = 1280, 720

CAM_W, CAM_H  = 1280, 720
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Robot interface
# ─────────────────────────────────────────────────────────────────────────────
def get_tcp_pose(ip=ROBOT_IP):
    """Read live TCP pose [x,y,z,rx,ry,rz] from port 30003."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            s.connect((ip, ROBOT_PORT_REALTIME))
            data = s.recv(1200)
            if len(data) < 492:
                return None
            return list(struct.unpack("!6d", data[444:492]))
    except Exception as e:
        print(f"[Robot] pose read error: {e}")
        return None


def send_urscript(script: str, ip=ROBOT_IP):
    """Send a URScript string to the robot (port 30002)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(5.0)
            s.connect((ip, ROBOT_PORT_SCRIPT))
            if not script.endswith("\n"):
                script += "\n"
            s.sendall(script.encode("utf-8"))
        return True
    except Exception as e:
        print(f"[Robot] URScript send error: {e}")
        return False


def movel_pose(pose, speed=MOVE_SPEED, accel=MOVE_ACCEL):
    """Move robot to a Cartesian TCP pose (linear move)."""
    x, y, z, rx, ry, rz = pose
    script = (
        f"movel(p[{x:.6f},{y:.6f},{z:.6f},{rx:.6f},{ry:.6f},{rz:.6f}],"
        f"a={accel:.3f},v={speed:.3f})\n"
    )
    return send_urscript(script)


def wait_until_stopped(target_pose, tol_m=0.002, timeout=30.0):
    """
    Wait until the robot TCP is within `tol_m` metres of `target_pose`.
    Returns True if reached, False on timeout.
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        cur = get_tcp_pose()
        if cur is not None:
            dist = np.linalg.norm(np.array(cur[:3]) - np.array(target_pose[:3]))
            if dist < tol_m:
                return True
        time.sleep(0.1)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# OAK-D pipeline (MJPEG encoded to save USB bandwidth)
# ─────────────────────────────────────────────────────────────────────────────
def create_pipeline() -> dai.Pipeline:
    p   = dai.Pipeline()
    cam = p.create(dai.node.ColorCamera)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setVideoSize(CAM_W, CAM_H)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(30)

    # Hardware MJPEG encoding
    videoEnc = p.create(dai.node.VideoEncoder)
    videoEnc.setDefaultProfilePreset(30, dai.VideoEncoderProperties.Profile.MJPEG)
    videoEnc.setQuality(95)
    cam.video.link(videoEnc.input)

    xout = p.create(dai.node.XLinkOut)
    xout.setStreamName("mjpeg")
    videoEnc.bitstream.link(xout.input)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not POSES_FILE.exists():
        print(f"ERROR: {POSES_FILE} not found — run collect_poses.py first.")
        sys.exit(1)

    with open(POSES_FILE) as f:
        data = json.load(f)

    board_cols = data["board"].get("cols", 7)
    board_rows = data["board"].get("rows", 5)
    square_m   = 0.038  # Hardcoded exactly to user 38mm
    marker_m   = 0.028  # Hardcoded exactly to user 28mm
    samples    = data["samples"]
    camera_matrix = np.array(data["camera"]["matrix"],  dtype=np.float64)
    dist_coeffs   = np.array(data["camera"]["distortion"], dtype=np.float64)

    # Initialize Charuco Board
    aruco_dict   = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    charuco_brD  = aruco.CharucoBoard((board_cols, board_rows), square_m, marker_m, aruco_dict)
    charuco_detector = aruco.CharucoDetector(charuco_brD)

    IMG_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Eye-in-Hand Calibration: Auto-Replay ===")
    print(f"  Loaded {len(samples)} saved poses from {POSES_FILE}")
    print(f"  Board: {board_cols}×{board_rows}, {square_m*1000:.0f} mm")
    print(f"  Speed: {MOVE_SPEED} m/s  —  KEEP HAND ON E-STOP\n")

    # Safety confirmation
    confirm = input(f"  Type YES to start moving the robot: ").strip()
    if confirm.upper() != "YES":
        print("Aborted.")
        sys.exit(0)

    # Allow Auto-Negotiation for USB Speed
    cfg = dai.Device.Config()

    new_samples = []

    with dai.Device(cfg) as device:
        device.startPipeline(create_pipeline())
        q = device.getOutputQueue("mjpeg", maxSize=2, blocking=False)

        for idx, s in enumerate(samples):
            target = s["tcp_pose"]
            print(f"\n[{idx+1}/{len(samples)}] Moving to saved pose #{s['id']:03d} …")
            print(f"         TCP target = ({target[0]*1000:+.1f}, {target[1]*1000:+.1f}, {target[2]*1000:+.1f}) mm")

            if not movel_pose(target):
                print("  → URScript send failed, skipping.")
                continue

            reached = wait_until_stopped(target, tol_m=0.003, timeout=30.0)
            if not reached:
                print("  → Timeout reaching pose — skipping.")
                continue

            time.sleep(SETTLE_TIME)

            # Flush camera buffer and grab a fresh frame
            while q.has():
                q.get()
            time.sleep(CAPTURE_DELAY)

            in_encoded = q.get()
            frame = cv2.imdecode(in_encoded.getData(), cv2.IMREAD_COLOR)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect Charuco
            charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)

            if charuco_corners is None or len(charuco_corners) < 4:
                print("  → Board NOT detected — skipping this pose.")
                continue

            tcp_now = get_tcp_pose()
            if tcp_now is None:
                print("  → Could not read robot pose — skipping.")
                continue

            # Save image
            fname  = f"image_{len(new_samples):03d}.png"
            fpath  = IMG_DIR / fname
            dbg    = frame.copy()
            aruco.drawDetectedCornersCharuco(dbg, charuco_corners, charuco_ids, (0, 255, 0))
            cv2.imwrite(str(fpath), dbg)

            new_samples.append({
                "id":              len(new_samples),
                "image":           f"images_replay/{fname}",
                "charuco_corners": charuco_corners.reshape(-1, 2).tolist(),
                "charuco_ids":     charuco_ids.flatten().tolist(),
                "tcp_pose":        tcp_now,
            })
            print(f"  ✓ Captured #{len(new_samples)-1:03d}  "
                  f"TCP=({tcp_now[0]*1000:+.1f},{tcp_now[1]*1000:+.1f},{tcp_now[2]*1000:+.1f}) mm")

            # Show brief preview
            cv2.imshow("Replay Capture", dbg)
            cv2.waitKey(400)

    cv2.destroyAllWindows()

    if len(new_samples) < 5:
        print(f"\nCaptured {len(new_samples)} valid samples — not enough to solve. Abort.")
        sys.exit(1)

    # Save new poses file (same format as collect_poses.py output)
    payload = {
        "board":   {"type": "charuco", "cols": board_cols, "rows": board_rows, 
                    "square_m": 0.038, "marker_m": 0.028,
                    "dict": "DICT_6X6_250"},
        "camera":  data["camera"],
        "samples": new_samples,
        "timestamp": time.ctime(),
    }
    with open(OUT_POSES, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n✓ Saved {len(new_samples)} replay samples to {OUT_POSES}")

    # Automatically run the solver on the new data
    print("\nRunning solver on replay data …\n")
    solver_path = Path(__file__).parent / "solve_calibration.py"
    env = os.environ.copy()
    env["HANDEYE_POSES_FILE"] = str(OUT_POSES)
    subprocess.run(
        [sys.executable, str(solver_path)],
        env=env, check=False)


if __name__ == "__main__":
    main()
