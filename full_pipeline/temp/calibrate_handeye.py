"""
calibrate_handeye.py
--------------------
Hand-eye calibration for the UR10 + OAK-D Lite (eye-in-hand).

Uses the same port-30002 raw URScript / RobotStateReader approach as
the rest of full_pipeline — no RTDE.

Two modes
---------
  MANUAL (default)
    Jog the robot with the teach pendant.
    Press SPACE in the camera window to capture each pose.
    Press ENTER when done (need >= 15 poses, aim for 20–30).
    Joint angles are saved so you can re-run in AUTO mode later.

  AUTO  (--auto)
    Loads saved joint angles from calibration/handeye_poses.json,
    replays each one automatically, and captures.

Board
-----
  ChArUco 7×5, DICT_6X6_250.
  !! MEASURE YOUR PRINTED BOARD AND SET THE CONSTANTS BELOW !!
    SQUARE_SIZE  — side length of one chess square in metres
    MARKER_SIZE  — side length of the ArUco marker inside each square

Output
------
  full_pipeline/calibration/T_cam2flange.npy   ← used by the pipeline
  full_pipeline/calibration/handeye_poses.json ← saved for AUTO replay

Usage
-----
    # Manual — jog robot, SPACE to capture, ENTER when done
    python full_pipeline/temp/calibrate_handeye.py

    # Auto — replay previously saved poses
    python full_pipeline/temp/calibrate_handeye.py --auto
"""

import argparse
import json
import os
import signal
import socket
import struct
import threading
import time
from pathlib import Path

import cv2
import cv2.aruco as aruco
import depthai as dai
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# !! SET THESE AFTER MEASURING YOUR PRINTED A3 BOARD !!
SQUARE_SIZE = 0.052   # metres — measured from printed A3 board
MARKER_SIZE = 0.0385  # metres — measured from printed A3 board
# ═══════════════════════════════════════════════════════════════════════════

CHARUCO_COLS = 7
CHARUCO_ROWS = 5
CHARUCO_DICT = aruco.DICT_6X6_250

MIN_POSES = 15   # minimum captures before ENTER is allowed

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
CALIB_DIR   = SCRIPT_DIR.parent / "calibration"
CALIB_DIR.mkdir(exist_ok=True)

RESULT_PATH = CALIB_DIR / "T_cam2flange.npy"
POSES_PATH  = CALIB_DIR / "handeye_poses.json"
K_PATH      = CALIB_DIR / "camera_matrix.npy"

# ── Robot ───────────────────────────────────────────────────────────────────
ROBOT_IP   = "192.168.8.102"
ROBOT_PORT = 30002

MOVE_SPEED     = 0.3    # rad/s  — auto-replay joint speed
MOVE_ACCEL     = 0.3    # rad/s²
JOINT_TOL_RAD  = 0.01   # arrival tolerance
SETTLE_TIME    = 1.5    # seconds to wait after arriving before capturing


# ── RobotStateReader ────────────────────────────────────────────────────────
class RobotStateReader(threading.Thread):
    """Reads joint positions and TCP pose from port 30002 secondary client."""
    def __init__(self):
        super().__init__(daemon=True)
        self._lock      = threading.Lock()
        self._stop_evt  = threading.Event()
        self._ready_evt = threading.Event()
        self._tcp_pose  = [0.0] * 6
        self._joints    = [0.0] * 6

    def run(self):
        while not self._stop_evt.is_set():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(3.0)
                    s.connect((ROBOT_IP, ROBOT_PORT))
                    while not self._stop_evt.is_set():
                        hdr = self._recvall(s, 4)
                        if hdr is None: break
                        plen = struct.unpack("!I", hdr)[0]
                        body = self._recvall(s, plen - 4)
                        if body is None: break
                        self._parse(body[1:])
            except Exception:
                time.sleep(0.5)

    @staticmethod
    def _recvall(s, n):
        buf = b""
        while len(buf) < n:
            chunk = s.recv(n - len(buf))
            if not chunk: return None
            buf += chunk
        return buf

    def _parse(self, data):
        off, got = 0, False
        while off < len(data):
            if off + 5 > len(data): break
            ps = struct.unpack("!I", data[off:off+4])[0]
            if ps < 5 or off + ps > len(data): break
            pt = data[off + 4]
            if pt == 1 and ps >= 251:
                joints = []
                for j in range(6):
                    base = off + 5 + j * 41
                    joints.append(struct.unpack("!d", data[base:base+8])[0])
                with self._lock:
                    self._joints = joints
                got = True
            elif pt == 4 and ps >= 53:
                pose = list(struct.unpack("!6d", data[off+5:off+53]))
                with self._lock:
                    self._tcp_pose = pose
                got = True
            off += ps
        if got:
            self._ready_evt.set()

    def wait_ready(self, timeout=5.0):
        return self._ready_evt.wait(timeout=timeout)

    def get_tcp_pose(self):
        with self._lock: return list(self._tcp_pose)

    def get_joints(self):
        with self._lock: return list(self._joints)

    def stop(self):
        self._stop_evt.set()


# ── URScriptSender ──────────────────────────────────────────────────────────
class URScriptSender:
    def __init__(self):
        self._lock = threading.Lock()
        self._sock = self._connect()
        threading.Thread(target=self._drain, daemon=True).start()

    def _connect(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5.0)
        s.connect((ROBOT_IP, ROBOT_PORT))
        return s

    def _drain(self):
        while True:
            try: self._sock.recv(4096)
            except Exception: time.sleep(0.01)

    def send(self, script):
        payload = (script.strip() + "\n").encode()
        with self._lock:
            try: self._sock.sendall(payload)
            except Exception:
                try:
                    self._sock = self._connect()
                    self._sock.sendall(payload)
                except Exception as e:
                    print(f"  [URScript] Send failed: {e}")

    def close(self):
        try: self._sock.close()
        except Exception: pass


def movej_and_wait(sender, state, joints, vel=MOVE_SPEED, acc=MOVE_ACCEL,
                   tol=JOINT_TOL_RAD, timeout=30.0):
    cur = state.get_joints()
    if max(abs(c - t) for c, t in zip(cur, joints)) < tol:
        return
    q_str = ",".join(f"{j:.6f}" for j in joints)
    sender.send(f"movej([{q_str}],a={acc:.4f},v={vel:.4f})")
    deadline = time.time() + timeout
    while time.time() < deadline:
        cur = state.get_joints()
        if max(abs(c - t) for c, t in zip(cur, joints)) < tol:
            return
        time.sleep(0.02)
    print("  [movej] Warning: timeout before arrival.")


# ── ChArUco helpers ─────────────────────────────────────────────────────────
def make_board_and_detector():
    dictionary = aruco.getPredefinedDictionary(CHARUCO_DICT)
    board = aruco.CharucoBoard(
        (CHARUCO_COLS, CHARUCO_ROWS),
        SQUARE_SIZE, MARKER_SIZE, dictionary
    )
    detector = aruco.CharucoDetector(
        board, aruco.CharucoParameters(), aruco.DetectorParameters()
    )
    return board, detector


def detect_charuco(img, board, detector, K, dist):
    """Returns (T_board2cam 4×4, annotated_img) or (None, img)."""
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _, _ = detector.detectBoard(grey)
    if ids is None or len(ids) < 4:
        return None, img
    aruco.drawDetectedCornersCharuco(img, corners, ids)
    valid, rvec, tvec = aruco.estimatePoseCharucoBoard(
        corners, ids, board, K, dist,
        np.zeros((3, 1)), np.zeros((3, 1))
    )
    if not valid:
        return None, img
    cv2.drawFrameAxes(img, K, dist, rvec, tvec, SQUARE_SIZE)
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = tvec.flatten()
    return T, img


def tcp_to_matrix(tcp_pose):
    R, _ = cv2.Rodrigues(np.array(tcp_pose[3:]))
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = tcp_pose[:3]
    return T


# ── Camera ──────────────────────────────────────────────────────────────────
MANUAL_FOCUS = 46    # tuned with tune_focus.py

def open_camera():
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setVideoSize(1280, 720)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    # Manual focus — locks the lens so it doesn't hunt during calibration
    ctrl_in = pipeline.create(dai.node.XLinkIn)
    ctrl_in.setStreamName("control")
    ctrl_in.out.link(cam.inputControl)

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("video")
    cam.video.link(xout.input)

    device     = dai.Device(pipeline)
    queue      = device.getOutputQueue(name="video", maxSize=4, blocking=False)
    ctrl_queue = device.getInputQueue("control")

    ctrl = dai.CameraControl()
    ctrl.setManualFocus(MANUAL_FOCUS)
    ctrl_queue.send(ctrl)

    return device, queue


# ── Solve and save ───────────────────────────────────────────────────────────
def solve_and_save(R_g2b, T_g2b, R_t2c, T_t2c, saved_poses):
    n = len(R_g2b)
    print(f"\nRunning hand-eye calibration with {n} poses (Tsai-Lenz) …")
    if n < 4:
        print(f"  Only {n} valid poses — need at least 4. Aborting.")
        return

    R_c2f, T_c2f = cv2.calibrateHandEye(
        R_g2b, T_g2b, R_t2c, T_t2c,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    T_cam2flange = np.eye(4)
    T_cam2flange[:3, :3] = R_c2f
    T_cam2flange[:3, 3]  = T_c2f.flatten()

    print("\nT_cam2flange:")
    print(T_cam2flange)

    np.save(str(RESULT_PATH), T_cam2flange)
    print(f"\nSaved → {RESULT_PATH}")

    if saved_poses:
        with open(POSES_PATH, "w") as f:
            json.dump(saved_poses, f, indent=2)
        print(f"Saved {len(saved_poses)} poses → {POSES_PATH}")

    print("\n── Copy to pipeline ──────────────────────────────────────────────")
    print(f"  Already saved directly to full_pipeline/calibration/")
    print("──────────────────────────────────────────────────────────────────")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    signal.signal(signal.SIGINT, lambda *_: os._exit(1))

    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", action="store_true",
                    help="Auto-replay saved poses from handeye_poses.json")
    args = ap.parse_args()

    # ── Load camera matrix ───────────────────────────────────────────────────
    if not K_PATH.exists():
        print(f"ERROR: {K_PATH} not found.")
        print("Run calibrate_camera.py first.")
        return
    K    = np.load(str(K_PATH))
    dist = np.zeros((4, 1))
    print(f"Loaded camera matrix from {K_PATH.name}")

    # ── Board + detector ─────────────────────────────────────────────────────
    board, detector = make_board_and_detector()
    print(f"ChArUco board: {CHARUCO_COLS}×{CHARUCO_ROWS}  "
          f"square={SQUARE_SIZE*1000:.1f}mm  marker={MARKER_SIZE*1000:.1f}mm\n")

    # ── Robot ────────────────────────────────────────────────────────────────
    print("Connecting to robot …")
    state  = RobotStateReader()
    state.start()
    if not state.wait_ready(timeout=5.0):
        print("ERROR: Robot state reader timed out — is the robot on?")
        return
    print("Robot connected!\n")

    # ── Camera ───────────────────────────────────────────────────────────────
    print("Opening camera …")
    cam_device, queue = open_camera()
    print("Camera ready.\n")

    # ── Calibration data ─────────────────────────────────────────────────────
    R_g2b, T_g2b   = [], []
    R_t2c, T_t2c   = [], []
    saved_poses     = []

    # ════════════════════════════════════════════════════════════════════════
    # AUTO MODE
    # ════════════════════════════════════════════════════════════════════════
    if args.auto:
        if not POSES_PATH.exists():
            print(f"ERROR: {POSES_PATH} not found — run manual mode first.")
            cam_device.close(); state.stop(); return

        with open(POSES_PATH) as f:
            poses_data = json.load(f)
        print(f"Loaded {len(poses_data)} saved poses from {POSES_PATH.name}\n")

        sender = URScriptSender()

        for i, entry in enumerate(poses_data):
            joints = entry["joint_angles"]
            print(f"[{i+1}/{len(poses_data)}] Moving to saved pose …")
            movej_and_wait(sender, state, joints)

            # Settle + show live feed
            settle_dl = time.time() + SETTLE_TIME
            while time.time() < settle_dl:
                pkt = queue.tryGet()
                if pkt is not None:
                    frame = pkt.getCvFrame()
                    cv2.putText(frame, f"Pose {i+1}/{len(poses_data)} — settling …",
                                (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                    cv2.imshow("calibrate_handeye", cv2.resize(frame, (960, 540)))
                    cv2.waitKey(1)

            # Search for board
            found = False
            search_dl = time.time() + 3.0
            while time.time() < search_dl:
                pkt = queue.tryGet()
                if pkt is None: time.sleep(0.01); continue
                frame = pkt.getCvFrame()
                T_b2c, frame = detect_charuco(frame, board, detector, K, dist)
                status = "BOARD ✓" if T_b2c is not None else "searching …"  # noqa
                cv2.putText(frame, f"Pose {i+1} — {status}  ({len(R_g2b)} captured)",
                            (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 220, 0) if T_b2c is not None else (0, 80, 220), 2)
                cv2.imshow("calibrate_handeye", cv2.resize(frame, (960, 540)))
                cv2.waitKey(1)
                if T_b2c is not None:
                    flange_T = tcp_to_matrix(state.get_tcp_pose())
                    R_g2b.append(flange_T[:3, :3]); T_g2b.append(flange_T[:3, 3])
                    R_t2c.append(T_b2c[:3, :3]);    T_t2c.append(T_b2c[:3, 3])
                    print(f"  Captured ✓  ({len(R_g2b)} total)")
                    found = True
                    break

            if not found:
                print(f"  WARNING: Board not detected at pose {i+1}, skipping.")

        sender.close()

    # ════════════════════════════════════════════════════════════════════════
    # MANUAL MODE
    # ════════════════════════════════════════════════════════════════════════
    else:
        print("=" * 60)
        print("  MANUAL MODE")
        print("  1. Jog the robot with the teach pendant")
        print("  2. Hold the ChArUco board in view of the camera")
        print("  3. Click the camera window → press SPACE to capture")
        print(f"  4. Aim for 20–30 poses with varied tilts/positions")
        print(f"  5. Press ENTER when done (need >= {MIN_POSES} poses)")
        print("  SPACE = capture  |  ENTER = finish  |  Q = quit")
        print("=" * 60 + "\n")

        flash_msg   = ""
        flash_until = 0.0

        while True:
            pkt = queue.tryGet()
            if pkt is not None:
                frame = pkt.getCvFrame()
                T_b2c, frame = detect_charuco(frame, board, detector, K, dist)

                n     = len(R_g2b)
                need  = max(0, MIN_POSES - n)
                ready = need == 0

                board_ok = T_b2c is not None
                status = "BOARD OK ✓ — press SPACE" if board_ok else "Move board into view"
                scol   = (0, 220, 0) if board_ok else (0, 80, 220)

                cv2.putText(frame, f"Captured: {n}",
                            (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
                cv2.putText(frame, status,
                            (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.75, scol, 2)
                hint = "Ready — press ENTER to finish" if ready else f"Need {need} more poses"
                cv2.putText(frame, hint,
                            (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
                cv2.putText(frame, "SPACE=capture  ENTER=finish  Q=quit",
                            (10, frame.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1)

                if time.time() < flash_until:
                    cv2.rectangle(frame, (8, 118), (640, 155), (0, 0, 0), -1)
                    cv2.putText(frame, flash_msg, (14, 148),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

                cv2.imshow("calibrate_handeye", cv2.resize(frame, (960, 540)))

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                # Use the T_b2c already detected on the current displayed frame —
                # re-detecting on a tryGet() frame often fails unnecessarily.
                if T_b2c is None:
                    flash_msg   = "Board not detected — reposition and try again"
                    flash_until = time.time() + 2.0
                    print("  Board not detected.")
                else:
                    tcp    = state.get_tcp_pose()
                    joints = state.get_joints()
                    flange_T = tcp_to_matrix(tcp)
                    R_g2b.append(flange_T[:3, :3]); T_g2b.append(flange_T[:3, 3])
                    R_t2c.append(T_b2c[:3, :3]);    T_t2c.append(T_b2c[:3, 3])
                    n = len(R_g2b)
                    saved_poses.append({
                        "pose_number":  n,
                        "joint_angles": joints,
                        "tcp_pose":     tcp,
                    })
                    flash_msg   = f"Pose {n} captured! ({max(0, MIN_POSES-n)} more needed)"
                    flash_until = time.time() + 1.5
                    print(f"  Pose {n} captured ✓")

            elif key == 13:   # Enter
                if len(R_g2b) < MIN_POSES:
                    flash_msg   = f"Need {MIN_POSES - len(R_g2b)} more poses first!"
                    flash_until = time.time() + 2.0
                else:
                    print(f"\nDone — {len(R_g2b)} poses captured.")
                    break

            elif key == ord('q'):
                print("Quit — no calibration saved.")
                cam_device.close(); state.stop()
                cv2.destroyAllWindows()
                return

    # ── Solve ─────────────────────────────────────────────────────────────────
    cv2.destroyAllWindows()
    cam_device.close()
    state.stop()

    solve_and_save(R_g2b, T_g2b, R_t2c, T_t2c, saved_poses)
    print("\nCalibration complete!")


if __name__ == "__main__":
    main()
