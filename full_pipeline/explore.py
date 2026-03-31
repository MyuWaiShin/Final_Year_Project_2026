"""
explore.py
----------
Stage 1 of the full pipeline.

Behaviour
---------
1.  Moves the robot to a predefined scan pose (SCAN_JOINT_POS).
2.  Sweeps the base joint (J0) across a configurable arc while the
    camera looks for ArUco tag ID 3 (DICT_6X6_250).
3.  Stops the sweep and holds position as soon as the tag is detected.
4.  Returns the last detected tag position in the robot base frame so
    the next stage (navigate.py) can use it.

Run standalone
--------------
    python explore.py

The script waits for you to confirm (press ENTER) before moving.

Motion notes
------------
All robot motion is sent as raw URScript over a persistent socket on
port 30002.  Robot state (joint positions, TCP pose) is read from the
same port's secondary-client packet stream in a background thread.
No RTDEControlInterface or RTDEReceiveInterface is used — both caused
unpredictable 10-second reconnect hangs on this setup (see
pipeline_dev/RTDE_debug_log.md).
"""

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

# ── Paths ───────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
CALIB_DIR  = SCRIPT_DIR / "calibration"
DATA_DIR   = SCRIPT_DIR / "data"
SCAN_POSE_FILE = DATA_DIR / "scan_pose.json"

# ── Robot ───────────────────────────────────────────────────────────────
ROBOT_IP      = "192.168.8.102"
ROBOT_PORT    = 30002

# ── Scan pose ───────────────────────────────────────────────────────────
def load_scan_pose():
    if not SCAN_POSE_FILE.exists():
        raise FileNotFoundError(
            f"Scan pose file not found: {SCAN_POSE_FILE}\n"
            "Run  python temp/capture_scan_pose.py  to record the scan pose first."
        )
    with open(SCAN_POSE_FILE) as f:
        data = json.load(f)
    return data["joint_angles"]

# ── Sweep config ────────────────────────────────────────────────────────
SWEEP_START_RAD   = -0.5
SWEEP_END_RAD     =  0.5
SWEEP_SPEED       =  0.2   # rad/s
SWEEP_ACCEL       =  0.1   # rad/s²  (ramp up)
SWEEP_STOP_ACCEL  =  0.8   # rad/s²  (hard stop on tag detect — must be >> SWEEP_ACCEL)

# ── Retry sweep config ───────────────────────────────────────────────────
# On retry: descend RETRY_LOWER_M with camera-offset-compensated XY so the
# camera still covers the same table area but from closer range.
SCAN_MAX_SWEEPS  = 2       # total sweep attempts before giving up
RETRY_LOWER_M    = 0.10    # metres to descend on retry (~10 cm)
SCAN_TABLE_Z_M   = -0.05   # approximate table-top Z in robot base frame (tune if needed)

# Cartesian move speeds for retry descent
RETRY_SPEED_MS  = 0.05    # m/s
RETRY_ACCEL_MS  = 0.02    # m/s²
CARTESIAN_TOL_M = 0.005   # m — arrival tolerance for movel

# ── ArUco ───────────────────────────────────────────────────────────────
ARUCO_DICT_TYPE = aruco.DICT_6X6_250
ARUCO_TAG_ID    = 13
MARKER_SIZE     = 0.021   # metres

# ── Move speeds (to scan pose) ──────────────────────────────────────────
APPROACH_SPEED = 0.5   # rad/s
APPROACH_ACCEL = 0.3   # rad/s²

# ── Arrival tolerances ──────────────────────────────────────────────────
JOINT_TOL_RAD = 0.01   # rad — all joints within this → arrived

# ── Gripper ─────────────────────────────────────────────────────────────
DASHBOARD_PORT   = 29999
GRIP_OPEN_URP    = "/programs/myu/open_gripper.urp"
GRIP_OPEN_MM     = 85.0   # expected open width


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
        self._ready_evt = threading.Event()   # set once first valid packet arrives
        self._tcp_pose  = [0.0] * 6
        self._joints    = [0.0] * 6
        self._voltage   = 0.0

    # ── background reader ────────────────────────────────────────────────
    def run(self):
        while not self._stop_evt.is_set():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(3.0)
                    s.connect((self.ip, self.port))
                    while not self._stop_evt.is_set():
                        # Outer message: [4-byte total length][1-byte msg type][sub-pkts...]
                        hdr = self._recvall(s, 4)
                        if hdr is None:
                            break
                        plen = struct.unpack("!I", hdr)[0]
                        body = self._recvall(s, plen - 4)
                        if body is None:
                            break
                        # body[0] = message type; sub-packets start at body[1]
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

            # ── Type 1: Joint Data ────────────────────────────────────
            # 5-byte header + 6 joints × 41 bytes = 251 bytes total
            # Each joint: q_actual(d8), q_target(d8), qd_actual(d8),
            #             I(f4), V(f4), T_motor(f4), T_micro(f4), mode(u1)
            if pt == 1 and ps >= 251:
                joints = []
                for j in range(6):
                    base = off + 5 + j * 41
                    q = struct.unpack("!d", data[base:base+8])[0]
                    joints.append(q)
                with self._lock:
                    self._joints = joints
                got_something = True

            # ── Type 2: Tool Data  (AI2 → gripper width) ─────────────
            # Bytes off+7..off+14 = Tool Analog Input 0 (double)
            elif pt == 2 and ps >= 15:
                ai = struct.unpack("!d", data[off+7:off+15])[0]
                with self._lock:
                    self._voltage = max(ai, 0.0)

            # ── Type 4: Cartesian Info (TCP pose in base frame) ───────
            # 5-byte header + 6 doubles (48 bytes) = TCP x,y,z,rx,ry,rz
            elif pt == 4 and ps >= 53:
                pose = list(struct.unpack("!6d", data[off+5:off+53]))
                with self._lock:
                    self._tcp_pose = pose
                got_something = True

            off += ps

        if got_something:
            self._ready_evt.set()

    # ── public API ───────────────────────────────────────────────────────
    def wait_ready(self, timeout: float = 5.0) -> bool:
        """Block until first valid joint+TCP packet received."""
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
    Port 30002 floods all connected clients with secondary data — a drain
    thread discards incoming bytes to prevent the OS recv buffer filling
    and blocking sendall().
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


# ── Motion helpers ───────────────────────────────────────────────────────
def movej_joints(sender: URScriptSender, state: RobotStateReader,
                 joint_angles: list, vel: float, acc: float,
                 tol: float = JOINT_TOL_RAD, timeout: float = 30.0,
                 stop_check=None):
    """
    Send movej([j0..j5]) and poll RobotStateReader joint positions for arrival.
    stop_check: optional callable() → True to abort early (e.g. tag found).
    """
    cur = state.get_joint_positions()
    if max(abs(c - t) for c, t in zip(cur, joint_angles)) < tol:
        return   # already there

    # Check BEFORE sending — a movej sent after stopj overrides the stop.
    if stop_check and stop_check():
        return

    q_str = ",".join(f"{j:.6f}" for j in joint_angles)
    sender.send(f"movej([{q_str}],a={acc:.4f},v={vel:.4f})")

    deadline = time.time() + timeout
    while time.time() < deadline:
        if stop_check and stop_check():
            return
        cur = state.get_joint_positions()
        if max(abs(c - t) for c, t in zip(cur, joint_angles)) < tol:
            return
        time.sleep(0.01)


def stopj(sender: URScriptSender, acc: float):
    sender.send(f"stopj({acc:.4f})")


# ── Helpers ─────────────────────────────────────────────────────────────
def tcp_to_matrix(tcp_pose):
    R, _ = cv2.Rodrigues(np.array(tcp_pose[3:]))
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = tcp_pose[:3]
    return T


def _dashboard_cmd(robot_ip: str, cmd: str) -> str:
    """Send a single dashboard command (port 29999) and return the response."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(5.0)
        s.connect((robot_ip, DASHBOARD_PORT))
        s.recv(1024)   # welcome message
        s.sendall((cmd + "\n").encode())
        return s.recv(1024).decode().strip()


def open_gripper(robot_ip: str, state: RobotStateReader):
    """Open the RG2 via dashboard URP. Skips if already open."""
    if state.get_width_mm() >= GRIP_OPEN_MM - 5.0:
        print("  Gripper already open.")
        return
    print("  Opening gripper …")
    _dashboard_cmd(robot_ip, f"load {GRIP_OPEN_URP}")
    time.sleep(0.06)
    _dashboard_cmd(robot_ip, "play")
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if state.get_width_mm() >= GRIP_OPEN_MM - 5.0:
            break
        time.sleep(0.1)
    print(f"  Gripper open: {state.get_width_mm():.1f} mm")


def detect_tag(frame, grey, K, dist_coeffs, T_cam2flange, state, detector):
    """
    Run one frame of ArUco detection.
    Returns (tag_pos_base, annotated_frame) or (None, annotated_frame).
    """
    corners, ids, _ = detector.detectMarkers(grey)
    if ids is None:
        return None, frame

    aruco.drawDetectedMarkers(frame, corners, ids)
    for i, mid in enumerate(ids.flatten()):
        if mid != ARUCO_TAG_ID:
            continue
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners[i:i+1], MARKER_SIZE, K, dist_coeffs
        )
        rvec, tvec = rvecs[0][0], tvecs[0][0]
        cv2.drawFrameAxes(frame, K, dist_coeffs, rvec, tvec, MARKER_SIZE * 0.5)

        R_tag, _ = cv2.Rodrigues(rvec)
        T_tag2cam = np.eye(4); T_tag2cam[:3, :3] = R_tag; T_tag2cam[:3, 3] = tvec
        tcp_pose   = state.get_tcp_pose()
        T_tcp2base = tcp_to_matrix(tcp_pose)
        T_tag2tcp  = T_cam2flange @ T_tag2cam
        T_tag2base = T_tcp2base   @ T_tag2tcp
        return T_tag2base[:3, 3].copy(), frame

    return None, frame


# ── Retry helpers ────────────────────────────────────────────────────────
def _compute_lower_tcp(scan_tcp, T_cam2flange, lower_m, table_z):
    """
    Return a new TCP pose [x,y,z,rx,ry,rz] that is `lower_m` below scan_tcp,
    with XY shifted so the camera optical axis still intersects the same table
    spot it was covering from scan_tcp height.
    Returns None if the geometry is degenerate (camera nearly horizontal).
    """
    hx, hy, hz   = scan_tcp[:3]
    hrx, hry, hrz = scan_tcp[3:]
    new_z = hz - lower_m
    try:
        R_eef, _   = cv2.Rodrigues(np.array([hrx, hry, hrz], dtype=np.float64))
        p_cam      = R_eef @ T_cam2flange[:3, 3]   # camera offset in base frame
        z_cam_base = R_eef @ T_cam2flange[:3, 2]   # camera optical-axis direction in base

        if abs(z_cam_base[2]) < 1e-3:
            return None   # camera nearly horizontal — ray won't hit table plane

        # Where the camera ray hits the table from the original height
        cam_z_orig = hz + p_cam[2]
        t1         = (table_z - cam_z_orig) / z_cam_base[2]
        look_x     = hx + p_cam[0] + t1 * z_cam_base[0]
        look_y     = hy + p_cam[1] + t1 * z_cam_base[1]

        # EEF XY at new_z such that camera still looks at (look_x, look_y)
        cam_z_new  = new_z + p_cam[2]
        t2         = (table_z - cam_z_new) / z_cam_base[2]
        new_x      = look_x - p_cam[0] - t2 * z_cam_base[0]
        new_y      = look_y - p_cam[1] - t2 * z_cam_base[1]
        return [new_x, new_y, new_z, hrx, hry, hrz]
    except Exception:
        return None


def movel_pose(sender: URScriptSender, state: RobotStateReader,
               x, y, z, rx, ry, rz, vel: float, acc: float,
               timeout: float = 30.0):
    """Send movel to a Cartesian pose and wait for TCP arrival."""
    cur = state.get_tcp_pose()
    if ((cur[0]-x)**2 + (cur[1]-y)**2 + (cur[2]-z)**2) ** 0.5 < CARTESIAN_TOL_M:
        return
    sender.send(
        f"movel(p[{x:.6f},{y:.6f},{z:.6f},{rx:.6f},{ry:.6f},{rz:.6f}],"
        f"a={acc:.4f},v={vel:.4f})"
    )
    deadline = time.time() + timeout
    while time.time() < deadline:
        cur = state.get_tcp_pose()
        if ((cur[0]-x)**2 + (cur[1]-y)**2 + (cur[2]-z)**2) ** 0.5 < CARTESIAN_TOL_M:
            return
        time.sleep(0.01)
    print("  [Explore] Warning: movel timeout — continuing anyway.")


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

    # ArUco detector
    dictionary = aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    detector   = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())

    # Robot — two port-30002 connections: one for state reading, one for sending
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

    # ── Load scan pose ──────────────────────────────────────────────────
    print("=" * 55)
    SCAN_JOINT_POS = load_scan_pose()
    print(f"  Scan pose loaded: {[round(j, 3) for j in SCAN_JOINT_POS]}")
    if not autonomous:
        print("  Press ENTER to move to scan pose (hand on E-stop) …")
        input()
    print("  Moving to scan pose …")
    movej_joints(sender, state, SCAN_JOINT_POS, APPROACH_SPEED, APPROACH_ACCEL)
    print("  Scan pose reached.")
    open_gripper(ROBOT_IP, state)
    print()

    # ── Build sweep waypoints (vary only J0 from scan pose) ─────────────
    print(f"  Sweeping J0 from offset {SWEEP_START_RAD:.2f} → {SWEEP_END_RAD:.2f} rad …")
    if not autonomous:
        print("  Press ENTER to start sweep (Q in camera window = abort) …")
        input()

    current_scan_pose = list(SCAN_JOINT_POS)   # may be adjusted on retry
    tag_result = {"pos": None}

    for sweep_num in range(1, SCAN_MAX_SWEEPS + 1):
        if sweep_num > 1:
            # Descend for a closer view with camera-offset-compensated EEF XY
            scan_tcp  = state.get_tcp_pose()
            lower_tcp = _compute_lower_tcp(scan_tcp, T_cam2flange, RETRY_LOWER_M, SCAN_TABLE_Z_M)
            if lower_tcp is None:
                # Fallback: drop Z straight down, no XY shift
                lower_tcp = [scan_tcp[0], scan_tcp[1], scan_tcp[2] - RETRY_LOWER_M,
                             scan_tcp[3], scan_tcp[4], scan_tcp[5]]
                print(f"\n  [Retry {sweep_num}/{SCAN_MAX_SWEEPS}] Camera compensation failed — "
                      f"descending {RETRY_LOWER_M*100:.0f} cm straight down …")
            else:
                print(f"\n  [Retry {sweep_num}/{SCAN_MAX_SWEEPS}] Descending "
                      f"{RETRY_LOWER_M*100:.0f} cm, camera-compensated EEF: "
                      f"X={lower_tcp[0]:.4f}  Y={lower_tcp[1]:.4f}  Z={lower_tcp[2]:.4f} …")
            movel_pose(sender, state, *lower_tcp, RETRY_SPEED_MS, RETRY_ACCEL_MS)
            current_scan_pose = state.get_joint_positions()
            print("  Lower scan pose reached.")

        sweep_start = list(current_scan_pose); sweep_start[0] += SWEEP_START_RAD
        sweep_end   = list(current_scan_pose); sweep_end[0]   += SWEEP_END_RAD
        print(f"  Sweep {sweep_num}: J0 {sweep_start[0]:.3f} → {sweep_end[0]:.3f} rad")

        sweep_done = threading.Event()

        def _sweep(s_start=sweep_start, s_end=sweep_end):
            for waypoint in (s_end, s_start):
                if tag_result["pos"] is not None:
                    break
                movej_joints(sender, state, waypoint, SWEEP_SPEED, SWEEP_ACCEL,
                             stop_check=lambda: tag_result["pos"] is not None)
                # Dwell at each endpoint: the camera needs a settled moment to
                # see any tag that's in FOV — detection is unreliable while the
                # arm is moving because the camera angle sweeps ahead of the arc.
                end_t = time.time() + 0.6
                while time.time() < end_t:
                    if tag_result["pos"] is not None:
                        break
                    time.sleep(0.05)
            sweep_done.set()

        sweep_thread = threading.Thread(target=_sweep, daemon=True)
        sweep_thread.start()

        print("  Sweeping … (tag window open – Q to abort)\n")
        while not sweep_done.is_set():
            pkt   = videoQueue.get()
            frame = pkt.getCvFrame()
            grey  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            tag_pos, frame = detect_tag(
                frame, grey, K, dist_coeffs, T_cam2flange, state, detector)

            if tag_pos is not None and tag_result["pos"] is None:
                tag_result["pos"] = tag_pos
                px, py, pz = tag_pos
                print(f"  ✓ Tag detected at base frame: X={px:.4f}  Y={py:.4f}  Z={pz:.4f}")
                print("  Stopping sweep …")
                stopj(sender, SWEEP_STOP_ACCEL)
                cv2.putText(frame, "TAG FOUND — stopping", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            label = "TAG FOUND" if tag_result["pos"] is not None else f"Sweep {sweep_num}/{SCAN_MAX_SWEEPS} — searching …"
            color = (0, 255, 0) if tag_result["pos"] is not None else (0, 0, 255)
            cv2.putText(frame, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow("explore", cv2.resize(frame, (960, 540)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stopj(sender, SWEEP_STOP_ACCEL)
                sweep_done.set()
                break

        sweep_thread.join(timeout=2.0)

        if tag_result["pos"] is not None:
            break   # found — exit retry loop
        if sweep_num < SCAN_MAX_SWEEPS:
            print(f"  Tag not found on sweep {sweep_num}. Descending for a closer view …")

    cv2.destroyAllWindows()

    # ── Result ──────────────────────────────────────────────────────────
    if tag_result["pos"] is not None:
        px, py, pz = tag_result["pos"]
        print("\n" + "=" * 55)
        print(f"  EXPLORE COMPLETE")
        print(f"  Tag base frame: X={px:.4f}  Y={py:.4f}  Z={pz:.4f}")
        print("=" * 55)
    else:
        print(f"\n  Tag NOT found after {SCAN_MAX_SWEEPS} sweeps.")

    # Cleanup
    state.stop()
    sender.close()
    device.close()
    print("\nDone.")
    return tag_result["pos"]


if __name__ == "__main__":
    main()
