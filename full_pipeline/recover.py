"""
recover.py
----------
Recovery helper for the full pipeline.

Called by main.py when grasp() or verify() fails.

Behaviour
---------
1. Opens the gripper (idempotent)
2. Snaps orientation to hover (tool pointing down) without dropping
3. Rises to hover_Z + RECOVERY_HEIGHT_M  (40 cm above hover)
4. Opens OAK-D camera, starts a 4-point XY search circle at recovery height
   — sweeps right → forward → left → back → centre
   — if ArUco tag is detected at any point the circle stops immediately
5. Closes camera, returns True so main.py re-runs navigate → grasp → verify

The tag-detection during the circle is a simple binary check (is the tag
visible?). 3D position is computed properly by navigate() afterwards.

Motion notes
------------
Same port-30002 / no-RTDE pattern as all other pipeline stages.
"""

import os
import signal
import socket
import struct
import threading
import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np
from cv2 import aruco

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
CALIB_DIR  = SCRIPT_DIR / "calibration"

# ── Robot ─────────────────────────────────────────────────────────────────────
ROBOT_IP       = "192.168.8.102"
ROBOT_PORT     = 30002
DASHBOARD_PORT = 29999

# ── Gripper ───────────────────────────────────────────────────────────────────
GRIP_OPEN_URP = "/programs/myu/open_gripper.urp"
GRIP_OPEN_MM  = 85.0

# ── Recovery params ───────────────────────────────────────────────────────────
RECOVERY_HEIGHT_M = 0.40   # metres above hover Z to retreat to
SEARCH_RADIUS_M   = 0.060  # metres — XY circle radius (~60 mm)

# ── Motion ────────────────────────────────────────────────────────────────────
MOVE_SPEED  = 0.06   # m/s
MOVE_ACCEL  = 0.02   # m/s²
XYZ_TOL_M   = 0.005

# ── ArUco ─────────────────────────────────────────────────────────────────────
ARUCO_DICT_TYPE = aruco.DICT_6X6_250
ARUCO_TAG_ID    = 13
MARKER_SIZE     = 0.021   # metres

# ── Centering ─────────────────────────────────────────────────────────────────
CENTER_TOL_PX    = 80     # pixels — tag within this of frame centre = well centred
FRAME_W, FRAME_H = 1280, 720


# ── Robot state reader ────────────────────────────────────────────────────────
class RobotStateReader(threading.Thread):
    def __init__(self, ip, port=ROBOT_PORT):
        super().__init__(daemon=True)
        self.ip, self.port = ip, port
        self._lock      = threading.Lock()
        self._stop_evt  = threading.Event()
        self._ready_evt = threading.Event()
        self._tcp_pose  = [0.0] * 6
        self._voltage   = 0.0

    def run(self):
        while not self._stop_evt.is_set():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(3.0)
                    s.connect((self.ip, self.port))
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
            if pt == 2 and ps >= 15:
                ai = struct.unpack("!d", data[off+7:off+15])[0]
                with self._lock: self._voltage = max(ai, 0.0)
            elif pt == 4 and ps >= 53:
                pose = list(struct.unpack("!6d", data[off+5:off+53]))
                with self._lock: self._tcp_pose = pose
                got = True
            off += ps
        if got: self._ready_evt.set()

    def wait_ready(self, timeout=5.0): return self._ready_evt.wait(timeout=timeout)
    def get_tcp_pose(self):
        with self._lock: return list(self._tcp_pose)
    def get_width_mm(self):
        with self._lock: v = self._voltage
        raw_mm = (v / 3.7) * 110.0
        slope  = (91.0 - 10.5) / (65.8 - 8.5)
        return max(0.0, round((raw_mm * slope) + (10.5 - 8.5 * slope), 1))
    def stop(self): self._stop_evt.set()


class URScriptSender:
    def __init__(self, ip, port=ROBOT_PORT):
        self.ip = ip
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
            try: self._sock.recv(4096)
            except Exception: time.sleep(0.01)

    def send(self, script):
        payload = (script.strip() + "\n").encode()
        with self._lock:
            try: self._sock.sendall(payload)
            except Exception: pass

    def close(self):
        try: self._sock.close()
        except Exception: pass


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
            if attempt < retries: time.sleep(0.5)
    return ""

def _wait_urp_done(timeout=6.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if "false" in _dashboard_cmd("running", retries=1).lower(): return
        time.sleep(0.05)

def _open_gripper(state):
    if state.get_width_mm() >= GRIP_OPEN_MM - 5.0:
        print("  [Recover] Gripper already open.")
        return
    print("  [Recover] Opening gripper ...")
    resp = _dashboard_cmd("running", retries=2)
    if "true" in resp.lower():
        _dashboard_cmd("stop"); _wait_urp_done(timeout=3.0)
    _dashboard_cmd(f"load {GRIP_OPEN_URP}")
    time.sleep(0.06)
    _dashboard_cmd("play"); _wait_urp_done()
    deadline = time.time() + 4.0
    while time.time() < deadline:
        if state.get_width_mm() >= GRIP_OPEN_MM - 5.0: break
        time.sleep(0.1)
    print(f"  [Recover] Gripper open: {state.get_width_mm():.1f} mm")

def _movel(sender, state, x, y, z, rx, ry, rz, timeout=30.0, stop_check=None):
    cur = state.get_tcp_pose()
    if ((cur[0]-x)**2 + (cur[1]-y)**2 + (cur[2]-z)**2) ** 0.5 < XYZ_TOL_M: return
    sender.send(
        f"movel(p[{x:.6f},{y:.6f},{z:.6f},{rx:.6f},{ry:.6f},{rz:.6f}],"
        f"a={MOVE_ACCEL:.4f},v={MOVE_SPEED:.4f})"
    )
    deadline = time.time() + timeout
    while time.time() < deadline:
        if stop_check and stop_check(): return
        cur = state.get_tcp_pose()
        if ((cur[0]-x)**2 + (cur[1]-y)**2 + (cur[2]-z)**2) ** 0.5 < XYZ_TOL_M: return
        time.sleep(0.01)
    print("  [Recover] Warning: movel timeout.")

def _stopj(sender, acc=0.8):
    sender.send(f"stopj({acc:.4f})")

def _open_camera():
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
    device = dai.Device(pipeline)
    queue  = device.getOutputQueue(name="video", maxSize=1, blocking=False)
    return device, queue

def _detect_tag(frame, K, dist_coeffs, T_cam2flange, state, detector):
    """
    Run ArUco detection on frame.
    Returns (tag_base_pos, pixel_center, annotated_frame) or (None, None, frame).
    pixel_center is (px, py) — the tag's centre in image coordinates.
    """
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(grey)
    if ids is None:
        return None, None, frame
    aruco.drawDetectedMarkers(frame, corners, ids)
    for i, mid in enumerate(ids.flatten()):
        if mid != ARUCO_TAG_ID:
            continue
        c = corners[i][0]
        px_cx = float(c[:, 0].mean())
        px_cy = float(c[:, 1].mean())
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners[i:i+1], MARKER_SIZE, K, dist_coeffs
        )
        rvec, tvec = rvecs[0][0], tvecs[0][0]
        R_tag, _ = cv2.Rodrigues(rvec)
        T_tag2cam = np.eye(4)
        T_tag2cam[:3, :3] = R_tag
        T_tag2cam[:3, 3]  = tvec
        tcp = state.get_tcp_pose()
        R_tcp, _ = cv2.Rodrigues(np.array(tcp[3:]))
        T_tcp2base = np.eye(4)
        T_tcp2base[:3, :3] = R_tcp
        T_tcp2base[:3, 3]  = tcp[:3]
        T_tag2base = T_tcp2base @ T_cam2flange @ T_tag2cam
        return T_tag2base[:3, 3].copy(), (px_cx, px_cy), frame
    return None, None, frame


def _compute_centering_xy(tag_base_pos, recovery_z, hrx, hry, hrz, T_cam2flange):
    """
    Compute EEF XY at recovery_z such that the camera optical axis passes
    directly through tag_base_pos. Returns (x, y) or None on failure.
    """
    tx, ty, tz = tag_base_pos
    try:
        R_eef, _   = cv2.Rodrigues(np.array([hrx, hry, hrz], dtype=np.float64))
        p_cam      = R_eef @ T_cam2flange[:3, 3]
        z_cam_base = R_eef @ T_cam2flange[:3, 2]
        if abs(z_cam_base[2]) < 1e-3:
            return None
        t  = (tz - (recovery_z + p_cam[2])) / z_cam_base[2]
        cx = tx - p_cam[0] - t * z_cam_base[0]
        cy = ty - p_cam[1] - t * z_cam_base[1]
        return cx, cy
    except Exception:
        return None


# ── Main ──────────────────────────────────────────────────────────────────────
def main(last_hover_pose: list) -> bool:
    """
    Recovery: open gripper, rise to clearance, search circle with camera.
    Stops circle early if ArUco tag is spotted.
    """
    signal.signal(signal.SIGINT, lambda *_: os._exit(1))

    hx, hy, hz    = last_hover_pose[:3]
    hrx, hry, hrz = last_hover_pose[3:]
    recovery_z    = hz + RECOVERY_HEIGHT_M

    print(f"\n  [RECOVER] Hover Z={hz:.4f}  →  recovery Z={recovery_z:.4f} "
          f"(+{RECOVERY_HEIGHT_M*100:.0f} cm)")

    state  = RobotStateReader(ROBOT_IP)
    sender = URScriptSender(ROBOT_IP)
    state.start()
    if not state.wait_ready(timeout=5.0):
        print("  [Recover] WARNING: state reader timed out — continuing anyway.")

    cur = state.get_tcp_pose()
    print(f"  [Recover] Current TCP: X={cur[0]:.4f}  Y={cur[1]:.4f}  Z={cur[2]:.4f}")

    # 1. Open gripper
    _open_gripper(state)

    # 2. Snap orientation to hover (tool pointing down) without going lower
    snap_z = max(cur[2], hz)
    print(f"  [Recover] Correcting orientation at Z={snap_z:.4f} ...")
    _movel(sender, state, hx, hy, snap_z, hrx, hry, hrz)

    # 3. Rise to recovery height
    print(f"  [Recover] Rising to Z={recovery_z:.4f} ...")
    _movel(sender, state, hx, hy, recovery_z, hrx, hry, hrz)
    print(f"  [Recover] At Z={state.get_tcp_pose()[2]:.4f}")

    # 4. Open camera + search circle
    # ── Load calibration ──────────────────────────────────────────────────────
    dist_coeffs  = np.zeros((4, 1))
    T_cam2flange = None
    K            = None
    try:
        K            = np.load(CALIB_DIR / "camera_matrix.npy")
        T_cam2flange = np.load(CALIB_DIR / "T_cam2flange.npy")
    except FileNotFoundError as e:
        print(f"  [Recover] Warning: calibration file missing ({e.name}) — "
              f"centering disabled, using raw hover XY for circle.")

    detector = aruco.ArucoDetector(
        aruco.getPredefinedDictionary(ARUCO_DICT_TYPE),
        aruco.DetectorParameters()
    )

    # ── Camera-offset compensation for circle centre ───────────────────────────
    circle_cx, circle_cy = hx, hy   # fallback
    if T_cam2flange is not None:
        R_eef, _   = cv2.Rodrigues(np.array([hrx, hry, hrz], dtype=np.float64))
        p_cam_local = R_eef @ T_cam2flange[:3, 3]
        z_cam_base  = R_eef @ T_cam2flange[:3, 2]
        if abs(z_cam_base[2]) > 1e-3:
            cam_z_base = p_cam_local[2] + recovery_z
            t0         = (hz - cam_z_base) / z_cam_base[2]
            circle_cx  = hx - p_cam_local[0] - t0 * z_cam_base[0]
            circle_cy  = hy - p_cam_local[1] - t0 * z_cam_base[1]
            print(f"  [Recover] Camera-compensated circle centre: "
                  f"X={circle_cx:.4f}  Y={circle_cy:.4f}  "
                  f"(raw hover: X={hx:.4f}  Y={hy:.4f})")
        else:
            print("  [Recover] Warning: camera Z nearly horizontal — using raw hover XY.")

    print(f"  [Recover] Opening camera + starting {SEARCH_RADIUS_M*1000:.0f} mm search circle ...")
    cam_device, queue = _open_camera()
    time.sleep(0.5)   # brief warmup

    R = SEARCH_RADIUS_M
    waypoints = [
        (circle_cx + R, circle_cy,     recovery_z),
        (circle_cx,     circle_cy + R, recovery_z),
        (circle_cx - R, circle_cy,     recovery_z),
        (circle_cx,     circle_cy - R, recovery_z),
        (circle_cx,     circle_cy,     recovery_z),   # return to centre
    ]

    tag_found  = {"flag": False, "pos": None}
    move_done  = threading.Event()

    def _circle():
        for px, py, pz in waypoints:
            if tag_found["flag"]: break
            _movel(sender, state, px, py, pz, hrx, hry, hrz,
                   stop_check=lambda: tag_found["flag"])
        move_done.set()

    circle_thread = threading.Thread(target=_circle, daemon=True)
    circle_thread.start()

    while not move_done.is_set():
        pkt   = queue.get()
        frame = pkt.getCvFrame()

        if K is not None and not tag_found["flag"]:
            tag_pos, pix_center, frame = _detect_tag(
                frame, K, dist_coeffs, T_cam2flange, state, detector)
            if tag_pos is not None:
                tag_found["flag"] = True
                tag_found["pos"]  = tag_pos
                _stopj(sender)
                print("  [Recover] ✓ Tag spotted — stopping circle.")

        if tag_found["flag"] and tag_found["pos"] is not None:
            # re-detect for live pixel offset display (best-effort)
            _, pix_center, frame = _detect_tag(
                frame, K, dist_coeffs, T_cam2flange, state, detector)
            if pix_center is not None:
                dx = pix_center[0] - FRAME_W / 2
                dy = pix_center[1] - FRAME_H / 2
                dist_px = (dx**2 + dy**2) ** 0.5
                col   = (0, 220, 0) if dist_px < CENTER_TOL_PX else (0, 200, 220)
                label = f"TAG — {dist_px:.0f}px off centre"
            else:
                col, label = (0, 220, 0), "TAG FOUND"
        else:
            col, label = (0, 80, 220), "Searching ..."

        cv2.putText(frame, label, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
        cv2.imshow("recover — search circle", cv2.resize(frame, (640, 360)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            _stopj(sender)
            break

    circle_thread.join(timeout=2.0)

    # ── Centering move ────────────────────────────────────────────────────────
    if tag_found["pos"] is not None and T_cam2flange is not None:
        print("  [Recover] Centering camera on tag ...")
        cxy = _compute_centering_xy(
            tag_found["pos"], recovery_z, hrx, hry, hrz, T_cam2flange)
        if cxy is not None:
            _movel(sender, state, cxy[0], cxy[1], recovery_z, hrx, hry, hrz)
            print(f"  [Recover] Centred at EEF X={cxy[0]:.4f}  Y={cxy[1]:.4f}")
        else:
            print("  [Recover] Warning: centering computation failed — staying put.")

    cv2.destroyAllWindows()
    cam_device.close()

    arrived = state.get_tcp_pose()
    status  = "tag visible" if tag_found["flag"] else "circle complete"
    print(f"  [Recover] Done ({status}) at Z={arrived[2]:.4f} — ready for next attempt.\n")

    state.stop()
    sender.close()
    return True
