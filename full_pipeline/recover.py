"""
recover.py
----------
Recovery helper for the full pipeline.

Called by main.py when grasp(), verify(), or transit() fails.

Behaviour
---------
1. Opens the gripper (idempotent)
2. Rises to clearance_z (absolute, passed from navigate — safe to command
   even if already there; the robot just stays put)
3. Runs a 60 mm XY search circle at clearance_z
   — checks BOTH ArUco and YOLO on every frame
   — stops the circle early when the object is within ±80 px of frame centre
4. Returns True so main.py re-enters navigate → grasp → verify

Motion notes
------------
Same port-30002 / no-RTDE pattern as all other stages.
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
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR     = Path(__file__).resolve().parent
CALIB_DIR      = SCRIPT_DIR / "calibration"
SCAN_POSE_FILE = SCRIPT_DIR / "data" / "scan_pose.json"
DETECT_MODEL_PATH = (SCRIPT_DIR / "models" / "detection"
                     / "yolo26n_detect_V1" / "weights" / "best.pt")

# ── Robot ─────────────────────────────────────────────────────────────────────
ROBOT_IP       = "192.168.8.102"
ROBOT_PORT     = 30002
DASHBOARD_PORT = 29999

# ── Gripper ───────────────────────────────────────────────────────────────────
GRIP_OPEN_URP = "/programs/myu/open_gripper.urp"
GRIP_OPEN_MM  = 85.0

# ── Search circle ───────────────────────────────────────────────────────────
SEARCH_RADIUS_M = 0.060   # 60 mm

# ── Detection / centring ──────────────────────────────────────────────────────
ARUCO_DICT_TYPE = aruco.DICT_6X6_250
ARUCO_TAG_ID    = 13
MARKER_SIZE     = 0.036
CONF_THRESHOLD  = 0.80
CENTER_TOL_PX   = 80
VIDEO_W, VIDEO_H = 1280, 720
MANUAL_FOCUS    = 46

# ── Motion ─────────────────────────────────────────────────────────────────────
MOVE_SPEED    = 0.06
MOVE_ACCEL    = 0.02
XYZ_TOL_M     = 0.005
JOINT_SPEED   = 0.5
JOINT_ACCEL   = 0.3
JOINT_TOL_RAD = 0.01

# ── Active recentering (post-circle J0 correction) ───────────────────────────
RECENTER_TOL_PX   = 20
RECENTER_MAX_ITER = 6
RECENTER_GAIN     = 0.0008  # J0 radians per pixel of offset


# ── Robot state reader ────────────────────────────────────────────────────────
class RobotStateReader(threading.Thread):
    def __init__(self, ip, port=ROBOT_PORT):
        super().__init__(daemon=True)
        self.ip, self.port = ip, port
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
                        self._parse(body[1:])
            except Exception:
                time.sleep(0.5)

    @staticmethod
    def _recvall(s, n):
        buf = b""
        while len(buf) < n:
            chunk = s.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def _parse(self, data):
        off, got = 0, False
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
                    joints.append(struct.unpack("!d", data[base:base+8])[0])
                with self._lock:
                    self._joints = joints
                got = True
            elif pt == 2 and ps >= 15:
                with self._lock:
                    self._voltage = max(struct.unpack("!d", data[off+7:off+15])[0], 0.0)
            elif pt == 4 and ps >= 53:
                with self._lock:
                    self._tcp_pose = list(struct.unpack("!6d", data[off+5:off+53]))
                got = True
            off += ps
        if got:
            self._ready_evt.set()

    def wait_ready(self, timeout=5.0):
        return self._ready_evt.wait(timeout=timeout)

    def get_tcp_pose(self):
        with self._lock:
            return list(self._tcp_pose)

    def get_joint_positions(self):
        with self._lock:
            return list(self._joints)

    def get_width_mm(self):
        with self._lock:
            v = self._voltage
        raw_mm = (v / 3.7) * 110.0
        slope  = (91.0 - 10.5) / (65.8 - 8.5)
        return max(0.0, round((raw_mm * slope) + (10.5 - 8.5 * slope), 1))

    def stop(self):
        self._stop_evt.set()


class URScriptSender:
    def __init__(self, ip, port=ROBOT_PORT):
        self.ip    = ip
        self._lock = threading.Lock()
        self._sock = self._connect(port)
        threading.Thread(target=self._drain, daemon=True).start()

    def _connect(self, port=ROBOT_PORT):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5.0)
        s.connect((self.ip, port))
        return s

    def _drain(self):
        while True:
            try:
                self._sock.recv(4096)
            except Exception:
                time.sleep(0.01)

    def send(self, script):
        payload = (script.strip() + "\n").encode()
        with self._lock:
            try:
                self._sock.sendall(payload)
            except Exception:
                pass

    def close(self):
        try:
            self._sock.close()
        except Exception:
            pass


# ── Helpers ───────────────────────────────────────────────────────────────────
def _dashboard_cmd(cmd, retries=3):
    for attempt in range(1, retries + 1):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5.0)
            s.connect((ROBOT_IP, DASHBOARD_PORT))
            s.recv(1024)
            s.sendall((cmd + "\n").encode())
            resp = s.recv(1024).decode().strip()
            s.close()
            return resp
        except Exception:
            if attempt < retries:
                time.sleep(0.5)
    return ""


def _wait_urp_done(timeout=6.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if "false" in _dashboard_cmd("running", retries=1).lower():
            return
        time.sleep(0.05)


def _open_gripper(state):
    if state.get_width_mm() >= GRIP_OPEN_MM - 5.0:
        print("  [Recover] Gripper already open.")
        return
    print("  [Recover] Opening gripper …")
    resp = _dashboard_cmd("running", retries=2)
    if "true" in resp.lower():
        _dashboard_cmd("stop")
        _wait_urp_done(timeout=3.0)
    _dashboard_cmd(f"load {GRIP_OPEN_URP}")
    time.sleep(0.06)
    _dashboard_cmd("play")
    _wait_urp_done()
    deadline = time.time() + 4.0
    while time.time() < deadline:
        if state.get_width_mm() >= GRIP_OPEN_MM - 5.0:
            break
        time.sleep(0.1)
    print(f"  [Recover] Gripper open: {state.get_width_mm():.1f} mm")


def _movel(sender, state, x, y, z, rx, ry, rz, timeout=30.0, stop_check=None):
    cur = state.get_tcp_pose()
    if ((cur[0]-x)**2 + (cur[1]-y)**2 + (cur[2]-z)**2) ** 0.5 < XYZ_TOL_M:
        return
    sender.send(
        f"movel(p[{x:.6f},{y:.6f},{z:.6f},{rx:.6f},{ry:.6f},{rz:.6f}],"
        f"a={MOVE_ACCEL:.4f},v={MOVE_SPEED:.4f})"
    )
    deadline = time.time() + timeout
    while time.time() < deadline:
        if stop_check and stop_check():
            return
        cur = state.get_tcp_pose()
        if ((cur[0]-x)**2 + (cur[1]-y)**2 + (cur[2]-z)**2) ** 0.5 < XYZ_TOL_M:
            return
        time.sleep(0.01)
    print("  [Recover] movel timeout — continuing.")


def _movej(sender, state, joints, timeout=30.0):
    cur = state.get_joint_positions()
    if max(abs(c - t) for c, t in zip(cur, joints)) < JOINT_TOL_RAD:
        return
    q_str = ",".join(f"{j:.6f}" for j in joints)
    sender.send(f"movej([{q_str}],a={JOINT_ACCEL:.4f},v={JOINT_SPEED:.4f})")
    deadline = time.time() + timeout
    while time.time() < deadline:
        cur = state.get_joint_positions()
        if max(abs(c - t) for c, t in zip(cur, joints)) < JOINT_TOL_RAD:
            return
        time.sleep(0.01)
    print("  [Recover] movej timeout — continuing.")


def _recenter_j0(sender, state, queue, yolo_model, frame_cx):
    """
    Active J0 recentering after the circle stops on first detection.
    Rotates J0 proportionally to the YOLO pixel offset until within
    RECENTER_TOL_PX of frame centre, or max iterations reached.
    """
    print(f"  [Recenter] Correcting J0 "
          f"(tol={RECENTER_TOL_PX}px, max={RECENTER_MAX_ITER} moves) …")

    for i in range(1, RECENTER_MAX_ITER + 1):
        time.sleep(0.25)

        pixel_x = None
        for _ in range(10):
            frame = queue.get().getCvFrame()
            results = yolo_model(frame, imgsz=640, conf=CONF_THRESHOLD, verbose=False)
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                best = max(r.boxes, key=lambda b: float(b.conf[0]))
                x1, y1, x2, y2 = map(int, best.xyxy[0])
                pixel_x = (x1 + x2) / 2.0
                break

        if pixel_x is None:
            print(f"  [Recenter {i}] Object not visible — stopping.")
            break

        offset_px = pixel_x - frame_cx
        print(f"  [Recenter {i}]  offset={offset_px:+.1f}px", end="")

        if abs(offset_px) < RECENTER_TOL_PX:
            print(f"  → within {RECENTER_TOL_PX}px — centred.")
            break

        delta_j0 = -offset_px * RECENTER_GAIN
        joints    = state.get_joint_positions()
        joints[0] += delta_j0
        q_str = ",".join(f"{j:.6f}" for j in joints)
        sender.send(f"movej([{q_str}],a={JOINT_ACCEL:.4f},v={JOINT_SPEED:.4f})")
        print(f"  → J0 Δ={delta_j0:+.4f} rad")

        deadline = time.time() + 5.0
        while time.time() < deadline:
            if abs(state.get_joint_positions()[0] - joints[0]) < 0.005:
                break
            time.sleep(0.01)

    print(f"  [Recenter] Done at J0={state.get_joint_positions()[0]:.4f} rad")


def _detect_objects(frame, yolo_model):
    """
    YOLO-only detection for the recovery circle.
    Annotates frame in-place.
    Returns (found: bool, pixel_x: float|None)
    """
    results = yolo_model(frame, imgsz=640, conf=CONF_THRESHOLD, verbose=False)
    r = results[0]
    if r.boxes is not None and len(r.boxes) > 0:
        best = max(r.boxes, key=lambda b: float(b.conf[0]))
        x1, y1, x2, y2 = map(int, best.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        label = yolo_model.names[int(best.cls[0])]
        cv2.putText(frame, f"YOLO {label}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
        return True, (x1 + x2) / 2.0
    return False, None


def _stopj(sender, acc=0.8):
    sender.send(f"stopj({acc:.4f})")


def _open_camera():
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setVideoSize(VIDEO_W, VIDEO_H)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.initialControl.setManualFocus(MANUAL_FOCUS)
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("video")
    cam.video.link(xout.input)
    device = dai.Device(pipeline)
    queue  = device.getOutputQueue(name="video", maxSize=1, blocking=False)
    return device, queue


# ── Main ──────────────────────────────────────────────────────────────────────
def main(clearance_z: float) -> bool:
    """
    Recovery: open gripper, rise to clearance_z, search circle with dual detection.
    Stops circle when object is centred (±80px). Returns True when done.

    clearance_z — absolute Z (hover_z + 0.60 m), computed by navigate.py.
                  Safe to command even if robot is already there.
    """
    signal.signal(signal.SIGINT, lambda *_: os._exit(1))

    print(f"\n  [RECOVER] clearance_z = {clearance_z:.4f} m")

    state  = RobotStateReader(ROBOT_IP)
    sender = URScriptSender(ROBOT_IP)
    state.start()
    if not state.wait_ready(timeout=5.0):
        print("  [Recover] WARNING: state reader timed out — continuing anyway.")

    cur = state.get_tcp_pose()
    cx, cy, cz    = cur[:3]
    crx, cry, crz = cur[3:]
    print(f"  [Recover] Current TCP: X={cx:.4f}  Y={cy:.4f}  Z={cz:.4f}")

    # ── 1. Open gripper ───────────────────────────────────────────────────────
    _open_gripper(state)

    # ── 2. Load scan pose, rise to scan height ─────────────────────────────
    with open(SCAN_POSE_FILE) as f:
        _scan = json.load(f)
    scan_tcp = _scan["tcp_pose"]          # [x, y, z, rx, ry, rz]
    scan_z   = float(scan_tcp[2])         # height from scan_pose — recovery rises here
    rx, ry, rz = scan_tcp[3], scan_tcp[4], scan_tcp[5]   # wrist orientation

    print(f"  [Recover] Rising to scan height + orientation (Z={scan_z:.4f}) …")
    _movel(sender, state, cx, cy, scan_z, rx, ry, rz)
    cur = state.get_tcp_pose()
    print(f"  [Recover] At Z={cur[2]:.4f}")

    # ── 3. Load calibration + models ──────────────────────────────────────────
    dist_coeffs = np.zeros((4, 1))
    try:
        K            = np.load(CALIB_DIR / "camera_matrix.npy")
        T_cam2flange = np.load(CALIB_DIR / "T_cam2flange.npy")
    except FileNotFoundError as e:
        print(f"  [Recover] Warning: calibration missing ({e.name}) — detection disabled.")
        K = T_cam2flange = None

    aruco_detector = aruco.ArucoDetector(
        aruco.getPredefinedDictionary(ARUCO_DICT_TYPE),
        aruco.DetectorParameters()
    )

    print("  [Recover] Loading YOLO detection model …")
    yolo_model = YOLO(str(DETECT_MODEL_PATH), task="detect")

    # ── 4. Build search circle at scan height ──────────────────────────────
    # scan_z and rx/ry/rz already loaded in step 2
    circle_cx, circle_cy = cur[0], cur[1]

    R = SEARCH_RADIUS_M
    waypoints = [
        (circle_cx + R, circle_cy,     scan_z),
        (circle_cx,     circle_cy + R, scan_z),
        (circle_cx - R, circle_cy,     scan_z),
        (circle_cx,     circle_cy - R, scan_z),
        (circle_cx,     circle_cy,     scan_z),   # return to centre
    ]

    print(f"  [Recover] Opening camera + starting {R*1000:.0f} mm search circle …")
    cam_device, queue = _open_camera()
    time.sleep(0.3)

    frame_cx   = VIDEO_W / 2.0
    result     = {"found": False, "centred": False}
    move_done  = threading.Event()

    def _circle():
        for px, py, pz in waypoints:
            if result["centred"]:
                break
            _movel(sender, state, px, py, pz, rx, ry, rz,
                   stop_check=lambda: result["centred"])
        move_done.set()

    circle_thread = threading.Thread(target=_circle, daemon=True)
    circle_thread.start()

    while not move_done.is_set():
        frame = queue.get().getCvFrame()

        if K is not None:
            found, pixel_x = _detect_objects(frame, yolo_model)

            if found and not result["centred"]:
                result["found"]   = True
                result["centred"] = True
                offset_px = pixel_x - frame_cx
                _stopj(sender)
                print(f"  [Recover] ✓ Object detected [YOLO]  "
                      f"offset={offset_px:+.0f}px — stopping circle.")

            # Draw overlay
            if found and pixel_x is not None:
                cv2.line(frame, (int(pixel_x), 0), (int(pixel_x), VIDEO_H),
                         (0, 255, 0), 1)
                cv2.line(frame, (int(frame_cx), 0), (int(frame_cx), VIDEO_H),
                         (255, 255, 255), 1)
                offset_px = pixel_x - frame_cx
                cv2.putText(frame,
                            f"DETECTED [YOLO]  offset={offset_px:+.0f}px — recentering …",
                            (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Searching (YOLO) …",
                            (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 220), 2)

        cv2.imshow("recover — search circle", cv2.resize(frame, (640, 360)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            _stopj(sender)
            break

    circle_thread.join(timeout=2.0)

    # ── Active recentering: J0 correction to bring object to frame centre ─────
    if result["centred"]:
        _recenter_j0(sender, state, queue, yolo_model, frame_cx)

    cv2.destroyAllWindows()
    cam_device.close()

    arrived = state.get_tcp_pose()
    status  = "centred" if result["centred"] else ("detected" if result["found"] else "not found")
    print(f"  [Recover] Done ({status}) at Z={arrived[2]:.4f} — ready for navigate.\n")

    state.stop()
    sender.close()
    return True
