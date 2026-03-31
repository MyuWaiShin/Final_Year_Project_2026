"""
navigate.py
-----------
Stage 2 of the full pipeline.

Behaviour
---------
Detects ArUco tag (ID 3, DICT_6X6_250) and aligns the robot TCP
directly above it:
  - X and Y match the tag's base-frame position (with calibration offsets)
  - Z stays at a fixed hover height (HOVER_Z_M, absolute in base frame)
  - Orientation is fixed (pointing straight down, configured below)

When the tag is seen, press SPACE to execute the move.

Run standalone
--------------
    python navigate.py

Motion notes
------------
All robot motion is sent as raw URScript over a persistent socket on
port 30002.  Robot state (TCP pose) is read from the same port's
secondary-client packet stream in a background thread.
No RTDEControlInterface or RTDEReceiveInterface is used — both caused
unpredictable 10-second reconnect hangs on this setup (see
pipeline_dev/RTDE_debug_log.md).
"""

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

# ── Paths ───────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
CALIB_DIR  = SCRIPT_DIR / "calibration"

# ── Robot ───────────────────────────────────────────────────────────────
ROBOT_IP   = "192.168.8.102"
ROBOT_PORT = 30002

# ── ArUco ───────────────────────────────────────────────────────────────
ARUCO_DICT_TYPE = aruco.DICT_6X6_250
ARUCO_TAG_ID    = 13
MARKER_SIZE     = 0.021   # metres – match your printed tag

# ── Hover config ────────────────────────────────────────────────────────
HOVER_Z_M = -0.050

# TCP orientation (Rodrigues vector) — pointing straight down.
HOVER_RX  = 2.225
HOVER_RY  = 2.170
HOVER_RZ  = 0.022

# Calibration offsets (empirically tuned to correct residual X/Y errors)
CALIB_X_OFFSET_M = -0.005
CALIB_Y_OFFSET_M = -0.050

# Move speed / acceleration
MOVE_SPEED = 0.04   # m/s
MOVE_ACCEL = 0.01   # m/s²

# Arrival tolerance
XYZ_TOL_M = 0.003   # metres


# ── Robot state reader (port 30002 secondary client) ────────────────────
class RobotStateReader(threading.Thread):
    """
    Reads the UR secondary-client stream (port 30002) in a background thread.
    Parses three sub-packet types from every Robot State message:

      Type 1  Joint Data       → actual joint positions q[0..5]
      Type 2  Tool Data        → AI2 voltage → gripper width
      Type 4  Cartesian Info   → TCP pose [x,y,z,rx,ry,rz] in base frame

    All values are updated at the robot's broadcast rate (~10 Hz on port 30002).
    Thread reconnects automatically on socket errors.
    """
    def __init__(self, ip: str, port: int = ROBOT_PORT):
        super().__init__(daemon=True)
        self.ip   = ip
        self.port = port
        self._lock      = threading.Lock()
        self._stop_evt  = threading.Event()
        self._ready_evt = threading.Event()
        self._tcp_pose  = [0.0] * 6
        self._joints    = [0.0] * 6
        self._voltage   = 0.0

    def run(self):
        while not self._stop_evt.is_set():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(3.0)
                    s.connect((self.ip, self.port))
                    while not self._stop_evt.is_set():
                        hdr = self._recvall(s, 4)
                        if hdr is None:
                            break
                        plen = struct.unpack("!I", hdr)[0]
                        body = self._recvall(s, plen - 4)
                        if body is None:
                            break
                        self._parse_subpackets(body[1:])
            except Exception:
                time.sleep(0.5)

    @staticmethod
    def _recvall(s, n: int):
        buf = b""
        while len(buf) < n:
            chunk = s.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def _parse_subpackets(self, data: bytes):
        off = 0
        got_something = False
        while off < len(data):
            if off + 5 > len(data):
                break
            ps = struct.unpack("!I", data[off:off+4])[0]
            if ps < 5 or off + ps > len(data):
                break
            pt = data[off + 4]

            if pt == 1 and ps >= 251:
                joints = []
                for j in range(6):
                    base = off + 5 + j * 41
                    q = struct.unpack("!d", data[base:base+8])[0]
                    joints.append(q)
                with self._lock:
                    self._joints = joints
                got_something = True

            elif pt == 2 and ps >= 15:
                ai = struct.unpack("!d", data[off+7:off+15])[0]
                with self._lock:
                    self._voltage = max(ai, 0.0)

            elif pt == 4 and ps >= 53:
                pose = list(struct.unpack("!6d", data[off+5:off+53]))
                with self._lock:
                    self._tcp_pose = pose
                got_something = True

            off += ps

        if got_something:
            self._ready_evt.set()

    def wait_ready(self, timeout: float = 5.0) -> bool:
        return self._ready_evt.wait(timeout=timeout)

    def get_joint_positions(self) -> list:
        with self._lock:
            return list(self._joints)

    def get_tcp_pose(self) -> list:
        with self._lock:
            return list(self._tcp_pose)

    def get_width_mm(self) -> float:
        with self._lock:
            v = self._voltage
        raw_mm = (v / 3.7) * 110.0
        slope  = (91.0 - 10.5) / (65.8 - 8.5)
        offset = 10.5 - (8.5 * slope)
        return max(0.0, round((raw_mm * slope) + offset, 1))

    def stop(self):
        self._stop_evt.set()


# ── URScript sender (port 30002, persistent socket) ─────────────────────
class URScriptSender:
    """
    Persistent TCP socket to port 30002 for sending URScript commands.
    A drain thread discards incoming secondary data to prevent the OS
    recv buffer from filling and blocking sendall().
    """
    def __init__(self, ip: str, port: int = ROBOT_PORT):
        self.ip   = ip
        self.port = port
        self._lock = threading.Lock()
        self._sock = self._connect()
        threading.Thread(target=self._drain, daemon=True).start()

    def _connect(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5.0)
        s.connect((self.ip, self.port))
        return s

    def _drain(self):
        while True:
            try:
                self._sock.recv(4096)
            except Exception:
                time.sleep(0.01)

    def send(self, script: str):
        payload = (script.strip() + "\n").encode()
        with self._lock:
            try:
                self._sock.sendall(payload)
            except Exception:
                try:
                    self._sock = self._connect()
                    self._sock.sendall(payload)
                except Exception as e:
                    print(f"  [URScript] Send failed: {e}")

    def close(self):
        try:
            self._sock.close()
        except Exception:
            pass


# ── Motion helper ────────────────────────────────────────────────────────
def movel(sender: URScriptSender, state: RobotStateReader,
          x, y, z, rx, ry, rz,
          vel: float = MOVE_SPEED, acc: float = MOVE_ACCEL,
          tol: float = XYZ_TOL_M, timeout: float = 30.0):
    """
    Send movel(p[...]) and poll RobotStateReader TCP pose for arrival.
    """
    cur = state.get_tcp_pose()
    if ((cur[0]-x)**2 + (cur[1]-y)**2 + (cur[2]-z)**2) ** 0.5 < tol:
        return   # already there

    sender.send(
        f"movel(p[{x:.6f},{y:.6f},{z:.6f},{rx:.6f},{ry:.6f},{rz:.6f}],a={acc:.4f},v={vel:.4f})"
    )
    deadline = time.time() + timeout
    while time.time() < deadline:
        cur = state.get_tcp_pose()
        if ((cur[0]-x)**2 + (cur[1]-y)**2 + (cur[2]-z)**2) ** 0.5 < tol:
            return
        time.sleep(0.01)


# ── Helper ──────────────────────────────────────────────────────────────
def tcp_to_matrix(tcp_pose):
    R, _ = cv2.Rodrigues(np.array(tcp_pose[3:]))
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = tcp_pose[:3]
    return T


# ── Main ────────────────────────────────────────────────────────────────
def main():
    signal.signal(signal.SIGINT, lambda *_: os._exit(1))

    # Load calibration
    print("Loading calibration …")
    K            = np.load(CALIB_DIR / "camera_matrix.npy")
    dist_coeffs  = np.zeros((4, 1))
    T_cam2flange = np.load(CALIB_DIR / "T_cam2flange.npy")
    print("  camera_matrix.npy  ✓")
    print("  T_cam2flange.npy   ✓\n")

    # ArUco detector
    dictionary = aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    detector   = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())

    # Robot
    print("Connecting to robot …")
    state  = RobotStateReader(ROBOT_IP)
    sender = URScriptSender(ROBOT_IP)
    state.start()
    if not state.wait_ready(timeout=5.0):
        raise RuntimeError("Robot state reader did not receive data within 5 s — "
                           "is the robot reachable at " + ROBOT_IP + "?")
    print("Robot connected!\n")

    # Camera
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
    device     = dai.Device(pipeline)
    videoQueue = device.getOutputQueue(name="video", maxSize=4, blocking=False)
    print("Camera started!\n")
    print("=" * 55)
    print("  SPACE  →  hover TCP directly above the tag")
    print("  Q      →  quit")
    print("=" * 55 + "\n")

    tag_pos_base = None
    tag_detected = False

    while True:
        frame     = videoQueue.get().getCvFrame()
        grey      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(grey)

        tag_detected = False
        tag_pos_base = None

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            for i, mid in enumerate(ids.flatten()):
                if mid != ARUCO_TAG_ID:
                    continue

                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                    corners[i:i+1], MARKER_SIZE, K, dist_coeffs
                )
                rvec, tvec = rvecs[0][0], tvecs[0][0]
                cv2.drawFrameAxes(frame, K, dist_coeffs, rvec, tvec, MARKER_SIZE * 0.5)

                img_pts, _ = cv2.projectPoints(
                    np.array([[0.0, 0.0, 0.0]], dtype=np.float32), rvec, tvec, K, dist_coeffs
                )
                cx_img = int(img_pts[0][0][0]); cy_img = int(img_pts[0][0][1])
                cv2.circle(frame, (cx_img, cy_img), 6, (0, 255, 0), -1)
                cv2.putText(frame, f"pix ({cx_img},{cy_img})", (cx_img + 8, cy_img - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

                R_tag, _ = cv2.Rodrigues(rvec)
                T_tag2cam = np.eye(4); T_tag2cam[:3, :3] = R_tag; T_tag2cam[:3, 3] = tvec
                tcp_pose   = state.get_tcp_pose()
                T_tcp2base = tcp_to_matrix(tcp_pose)
                T_tag2tcp  = T_cam2flange @ T_tag2cam
                T_tag2base = T_tcp2base   @ T_tag2tcp

                tag_pos_base = T_tag2base[:3, 3].copy()
                tag_pos_base[0] += CALIB_X_OFFSET_M
                tag_pos_base[1] += CALIB_Y_OFFSET_M
                tag_detected = True

                bx, by, bz = tag_pos_base
                cx, cy, cz = tvec
                cv2.putText(frame, f"Cam:  ({cx:.3f}, {cy:.3f}, {cz:.3f}) m",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 255), 1)
                cv2.putText(frame, f"Base: ({bx:.3f}, {by:.3f}, {bz:.3f}) m",
                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 1)
                cv2.putText(frame, f"Hover target: ({bx:.3f}, {by:.3f}, {HOVER_Z_M:.3f}) m  [Z fixed]",
                            (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 200), 1)
                break

        label = f"TAG {ARUCO_TAG_ID} DETECTED — press SPACE to hover" if tag_detected \
                else f"Searching for tag ID {ARUCO_TAG_ID} …"
        color = (0, 255, 0) if tag_detected else (0, 0, 255)
        cv2.putText(frame, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.putText(frame, "SPACE = hover above tag  |  Q = quit",
                    (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)

        cv2.imshow("navigate", cv2.resize(frame, (960, 540)))
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord(' '):
            if not tag_detected or tag_pos_base is None:
                print("[WARN] Tag not detected – aim camera at tag first.")
                continue

            bx, by, _ = tag_pos_base
            target_pose = [bx, by, HOVER_Z_M, HOVER_RX, HOVER_RY, HOVER_RZ]
            cur_tcp = state.get_tcp_pose()
            dist_mm = np.linalg.norm(np.array(target_pose[:3]) - np.array(cur_tcp[:3])) * 1000

            print("\n" + "=" * 60)
            print(f"  Tag base frame:  X={bx:.4f}  Y={by:.4f}")
            print(f"  Hover Z (fixed): {HOVER_Z_M:.4f} m")
            print(f"  Target pose:     {[round(v, 4) for v in target_pose]}")
            print(f"  Distance:        {dist_mm:.1f} mm")
            print("=" * 60)
            confirm = input("  Type YES to move (hand on E-stop): ").strip()
            if confirm.upper() != "YES":
                print("  Cancelled.\n")
                continue

            print("  Moving …")
            movel(sender, state, *target_pose, vel=MOVE_SPEED, acc=MOVE_ACCEL)
            print("  Hover reached.")
            break

    # Cleanup
    state.stop()
    sender.close()
    cv2.destroyAllWindows()
    device.close()
    print("\nDone.")
    return tag_pos_base


if __name__ == "__main__":
    main()
