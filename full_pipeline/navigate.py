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

# TCP orientation (Rodrigues vector) — pointing straight down.
# Used as the baseline for orientation computation and as fallback if
# tag orientation cannot be computed.
HOVER_RX  = 2.225
HOVER_RY  = 2.170
HOVER_RZ  = 0.022

# Autonomous mode: how many consecutive stable frames before auto-moving.
STABLE_FRAMES_NEEDED = 8      # ~0.25 s at 30 fps
STABLE_TOL_M         = 0.005  # 5 mm — max position jitter to count as stable

# Calibration offsets (empirically tuned to correct residual X/Y errors)
CALIB_X_OFFSET_M = -0.005
CALIB_Y_OFFSET_M = -0.050
CALIB_Z_OFFSET_M = -0.000   # TCP descends 18.6 cm below detected tag surface

# Move speed / acceleration
MOVE_SPEED = 0.04   # m/s
MOVE_ACCEL = 0.01   # m/s²

# Arrival tolerance
XYZ_TOL_M = 0.003   # metres

# ── Horizontal centering ─────────────────────────────────────────────────
# After hover, correct EEF along the camera's horizontal axis so the tag
# is centred left-right in the image (aligns gripper gap with object).
CENTER_H_TOL_PX  = 40   # pixels — stop when horizontal offset < this
CENTER_H_MAX_ITER = 3    # max correction moves


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


# ── Helpers ─────────────────────────────────────────────────────────────
def tcp_to_matrix(tcp_pose):
    R, _ = cv2.Rodrigues(np.array(tcp_pose[3:]))
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = tcp_pose[:3]
    return T


def compute_hover_orientation(R_tag_base: np.ndarray,
                               R_hover_baseline: np.ndarray) -> tuple:
    """
    Derive EEF orientation aligned to the ArUco tag's X-axis (yaw).

    Strategy
    --------
    1. Extract the tag's yaw: angle its X-axis makes with base-frame X in XY plane.
    2. Extract the baseline yaw from R_hover_baseline (the known good pointing-down
       orientation for this robot).
    3. Compute delta = tag_yaw - baseline_yaw and the ±180° flip candidate
       (RG2 gripper is symmetric so both grasps are equivalent).
    4. Pick the candidate with the smaller absolute delta (J6 safeguard — avoids
       large wrist rotations).
    5. Apply that rotation around the base Z-axis to R_hover_baseline and convert
       back to axis-angle.

    Returns
    -------
    (rx, ry, rz) — Rodrigues vector for use in movel(p[x,y,z,rx,ry,rz]).
    Falls back to (HOVER_RX, HOVER_RY, HOVER_RZ) on any error.
    """
    try:
        def _wrap(a):
            return (a + np.pi) % (2 * np.pi) - np.pi

        # Tag yaw: direction of tag X-axis projected onto base XY
        x_tag    = R_tag_base[:, 0]
        yaw_tag  = np.arctan2(x_tag[1], x_tag[0])

        # Baseline yaw (same projection for current hover orientation)
        yaw_base = np.arctan2(R_hover_baseline[1, 0], R_hover_baseline[0, 0])

        delta_a  = _wrap(yaw_tag - yaw_base)
        delta_b  = _wrap(delta_a + np.pi)          # 180° flip

        chosen   = delta_a if abs(delta_a) <= abs(delta_b) else delta_b

        if abs(chosen) > np.pi / 2:
            print(f"  [Orient] Best delta = {np.degrees(chosen):.1f}° "
                  f"(> 90° — large wrist rotation, proceeding with best option)")

        c, s   = np.cos(chosen), np.sin(chosen)
        R_z    = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
        R_tgt  = R_z @ R_hover_baseline
        rvec, _ = cv2.Rodrigues(R_tgt)
        return tuple(float(v) for v in rvec.flatten())

    except Exception as e:
        print(f"  [Orient] Warning: orientation computation failed ({e}) — using baseline.")
        return (HOVER_RX, HOVER_RY, HOVER_RZ)


# ── Horizontal centering ─────────────────────────────────────────────────
def center_horizontal(videoQueue, detector, K, dist_coeffs, T_cam2flange,
                      state: RobotStateReader, sender: URScriptSender):
    """
    Single-axis correction: move the EEF along the camera's horizontal axis
    until the tag pixel centre is within CENTER_H_TOL_PX of the image midline.

    Only touches EEF XY (via the camera-horizontal direction in base frame).
    Z and orientation are unchanged.
    """
    print(f"  [Centre] Horizontal centering "
          f"(tol {CENTER_H_TOL_PX}px, max {CENTER_H_MAX_ITER} moves) …")

    fx = float(K[0, 0])
    frame_cx = 640.0   # half of 1280

    for i in range(1, CENTER_H_MAX_ITER + 1):
        time.sleep(0.3)   # settle

        # Try up to 15 frames for a clean detection
        result = None
        for _ in range(15):
            frame  = videoQueue.get().getCvFrame()
            grey   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(grey)
            if ids is None:
                continue
            for j, mid in enumerate(ids.flatten()):
                if mid != ARUCO_TAG_ID:
                    continue
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                    corners[j:j+1], MARKER_SIZE, K, dist_coeffs
                )
                tvec    = tvecs[0][0]
                c       = corners[j][0]
                px_cx   = float(c[:, 0].mean())
                result  = (px_cx, tvec)
                break
            if result is not None:
                break

        if result is None:
            print(f"  [Centre {i}] Tag not visible — stopping.")
            break

        px_cx, tvec = result
        delta_px    = px_cx - frame_cx
        print(f"  [Centre {i}]  pixel offset: {delta_px:+.1f} px", end="")

        if abs(delta_px) < CENTER_H_TOL_PX:
            print(f"  → within {CENTER_H_TOL_PX}px — done.")
            break

        # Convert pixel offset → metres in camera frame (horizontal only)
        delta_x_cam = delta_px * tvec[2] / fx

        # Rotate camera-frame [δx, 0, 0] into base frame
        tcp        = state.get_tcp_pose()
        rx, ry, rz = tcp[3], tcp[4], tcp[5]
        R_eef, _   = cv2.Rodrigues(np.array([rx, ry, rz], dtype=np.float64))
        R_cam2base = R_eef @ T_cam2flange[:3, :3]
        delta_base = R_cam2base @ np.array([delta_x_cam, 0.0, 0.0])

        new_x = tcp[0] + delta_base[0]
        new_y = tcp[1] + delta_base[1]
        print(f"  →  dX={delta_base[0]:+.4f}  dY={delta_base[1]:+.4f} m")
        movel(sender, state, new_x, new_y, tcp[2], rx, ry, rz)

    final = state.get_tcp_pose()
    print(f"  [Centre] EEF: X={final[0]:.4f}  Y={final[1]:.4f}")
    return final


# ── Main ────────────────────────────────────────────────────────────────
def main(autonomous: bool = False):
    signal.signal(signal.SIGINT, lambda *_: os._exit(1))

    # Load calibration
    print("Loading calibration …")
    K            = np.load(CALIB_DIR / "camera_matrix.npy")
    dist_coeffs  = np.zeros((4, 1))
    T_cam2flange = np.load(CALIB_DIR / "T_cam2flange.npy")
    print("  camera_matrix.npy  ✓")
    print("  T_cam2flange.npy   ✓\n")

    # Baseline rotation matrix for orientation computation
    R_hover_baseline, _ = cv2.Rodrigues(
        np.array([HOVER_RX, HOVER_RY, HOVER_RZ], dtype=np.float64)
    )

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
    if autonomous:
        print(f"  AUTO  →  moves when tag stable for {STABLE_FRAMES_NEEDED} frames")
    else:
        print("  SPACE  →  hover TCP directly above the tag")
    print("  Q      →  quit")
    print("=" * 55 + "\n")

    tag_pos_base  = None
    tag_orient    = (HOVER_RX, HOVER_RY, HOVER_RZ)   # updated each detected frame
    tag_detected  = False
    stable_count  = 0
    last_stable_pos = None
    target_pose   = None

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
                tag_pos_base[2] += CALIB_Z_OFFSET_M

                # Smart wrist orientation aligned to tag X-axis
                tag_orient  = compute_hover_orientation(T_tag2base[:3, :3], R_hover_baseline)
                tag_detected = True

                bx, by, bz = tag_pos_base
                cx, cy, cz = tvec
                cv2.putText(frame, f"Cam:  ({cx:.3f}, {cy:.3f}, {cz:.3f}) m",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 255), 1)
                cv2.putText(frame, f"Base: ({bx:.3f}, {by:.3f}, {bz:.3f}) m",
                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 1)
                cv2.putText(frame, f"Target: ({bx:.3f}, {by:.3f}, {bz:.3f}) m",
                            (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 200), 1)
                break

        # ── Stability tracking (autonomous only) ────────────────────────
        if autonomous:
            if tag_detected:
                if (last_stable_pos is not None and
                        np.linalg.norm(tag_pos_base - last_stable_pos) < STABLE_TOL_M):
                    stable_count += 1
                else:
                    stable_count = 1
                last_stable_pos = tag_pos_base.copy()
            else:
                stable_count   = 0
                last_stable_pos = None

        # ── Overlay ─────────────────────────────────────────────────────
        if autonomous:
            if tag_detected:
                label = f"TAG {ARUCO_TAG_ID} — stable {stable_count}/{STABLE_FRAMES_NEEDED}"
                color = (0, 200, 255)
            else:
                label = f"Searching for tag ID {ARUCO_TAG_ID} …"
                color = (0, 0, 255)
            hint = "AUTO mode — Q to abort"
        else:
            label = (f"TAG {ARUCO_TAG_ID} DETECTED — press SPACE to hover"
                     if tag_detected else f"Searching for tag ID {ARUCO_TAG_ID} …")
            color = (0, 255, 0) if tag_detected else (0, 0, 255)
            hint  = "SPACE = hover above tag  |  Q = quit"

        cv2.putText(frame, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.putText(frame, hint, (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)

        cv2.imshow("navigate", cv2.resize(frame, (960, 540)))
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        # ── Autonomous: auto-execute when stable ─────────────────────────
        if autonomous and stable_count >= STABLE_FRAMES_NEEDED and tag_pos_base is not None:
            rx, ry, rz  = tag_orient
            target_pose = [*tag_pos_base, rx, ry, rz]
            dist_mm     = np.linalg.norm(
                np.array(target_pose[:3]) - np.array(state.get_tcp_pose()[:3])) * 1000
            print("\n" + "=" * 60)
            print(f"  [AUTO] Tag stable — executing hover")
            print(f"  Base: X={tag_pos_base[0]:.4f}  Y={tag_pos_base[1]:.4f}  Z={tag_pos_base[2]:.4f}")
            print(f"  Orient: rx={rx:.4f}  ry={ry:.4f}  rz={rz:.4f}")
            print(f"  Distance: {dist_mm:.1f} mm")
            print("=" * 60)
            print("  Moving …")
            movel(sender, state, *target_pose, vel=MOVE_SPEED, acc=MOVE_ACCEL)
            print("  Hover reached.")
            refined = center_horizontal(videoQueue, detector, K, dist_coeffs,
                                        T_cam2flange, state, sender)
            target_pose = refined
            break

        # ── Debug: SPACE + YES ───────────────────────────────────────────
        elif not autonomous and key == ord(' '):
            if not tag_detected or tag_pos_base is None:
                print("[WARN] Tag not detected – aim camera at tag first.")
                continue

            rx, ry, rz  = tag_orient
            target_pose = [*tag_pos_base, rx, ry, rz]
            cur_tcp     = state.get_tcp_pose()
            dist_mm     = np.linalg.norm(
                np.array(target_pose[:3]) - np.array(cur_tcp[:3])) * 1000

            print("\n" + "=" * 60)
            print(f"  Base: X={tag_pos_base[0]:.4f}  Y={tag_pos_base[1]:.4f}  Z={tag_pos_base[2]:.4f}")
            print(f"  Orient: rx={rx:.4f}  ry={ry:.4f}  rz={rz:.4f}")
            print(f"  Target: {[round(v, 4) for v in target_pose]}")
            print(f"  Distance: {dist_mm:.1f} mm")
            print("=" * 60)
            confirm = input("  Type YES to move (hand on E-stop): ").strip()
            if confirm.upper() != "YES":
                print("  Cancelled.\n")
                continue

            print("  Moving …")
            movel(sender, state, *target_pose, vel=MOVE_SPEED, acc=MOVE_ACCEL)
            print("  Hover reached.")
            refined = center_horizontal(videoQueue, detector, K, dist_coeffs,
                                        T_cam2flange, state, sender)
            target_pose = refined
            break

    # Cleanup
    state.stop()
    sender.close()
    cv2.destroyAllWindows()
    device.close()
    print("\nDone.")
    return target_pose   # full [x, y, z, rx, ry, rz] — needed by grasp.py


if __name__ == "__main__":
    main()
