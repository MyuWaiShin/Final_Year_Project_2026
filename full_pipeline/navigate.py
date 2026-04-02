"""
navigate.py
-----------
Stage 2 of the full pipeline.

Detection hierarchy (re-checked on every attempt — no mode flag needed):
  1. ArUco tag ID 13  →  tag pose + gripper yaw alignment
  2. YOLO-only        →  bbox centre + OAK-D depth → 3D position,
                         fixed wrist orientation (straight down)

Returns {'hover_pose': [x,y,z,rx,ry,rz], 'clearance_z': float} or None.
clearance_z = hover_z + 0.60 m — passed downstream to verify, transit, recover.

Motion notes
------------
All robot motion is sent as raw URScript over port 30002.
Robot state is read from the same port in a background thread.
No RTDE — caused 10-second reconnect hangs (see pipeline_dev/RTDE_debug_log.md).
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
from ultralytics import YOLO

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR        = Path(__file__).resolve().parent
CALIB_DIR         = SCRIPT_DIR / "calibration"
HOVER_Z_FILE      = SCRIPT_DIR / "data" / "hover_z.json"
DETECT_MODEL_PATH = (SCRIPT_DIR / "models" / "detection"
                     / "yolo26n_detect_V1" / "weights" / "best.pt")

# ── Robot ────────────────────────────────────────────────────────────────────
ROBOT_IP   = "192.168.8.102"
ROBOT_PORT = 30002

# ── ArUco ────────────────────────────────────────────────────────────────────
ARUCO_DICT_TYPE = aruco.DICT_6X6_250
ARUCO_TAG_ID    = 13
MARKER_SIZE     = 0.036   # metres

# ── Camera ───────────────────────────────────────────────────────────────────
VIDEO_W      = 1280
VIDEO_H      = 720
MANUAL_FOCUS = 46         # locked value from focus calibration

# ── TCP orientation — straight down ──────────────────────────────────────────
HOVER_RX = 2.225
HOVER_RY = 2.170
HOVER_RZ = 0.022

# ── Clearance height ─────────────────────────────────────────────────────────
CLEARANCE_OFFSET_M = 0.60   # metres above hover Z

# ── Stability (autonomous mode) ───────────────────────────────────────────────
STABLE_FRAMES_NEEDED = 8
STABLE_TOL_M         = 0.005

# ── Calibration offsets (zeroed after recalibration — adjust if needed) ──────
CALIB_X_OFFSET_M = -0.005
CALIB_Y_OFFSET_M = -0.050
CALIB_Z_OFFSET_M = 0.000

# ── Motion ───────────────────────────────────────────────────────────────────
MOVE_SPEED = 0.04
MOVE_ACCEL = 0.01
XYZ_TOL_M  = 0.003

# ── Centring correction ───────────────────────────────────────────────────────
CENTER_H_TOL_PX   = 40
CENTER_H_MAX_ITER = 3

# ── YOLO detection confidence threshold ──────────────────────────────────────
CONF_THRESHOLD = 0.80


# ── Robot state reader ────────────────────────────────────────────────────────
class RobotStateReader(threading.Thread):
    """
    Reads the UR secondary-client stream (port 30002) in a background thread.
    Parses sub-packet types:
      Type 1  Joint Data      → joint positions
      Type 2  Tool Data       → AI2 voltage → gripper width
      Type 4  Cartesian Info  → TCP pose [x,y,z,rx,ry,rz]
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
        got = False
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
                ai = struct.unpack("!d", data[off+7:off+15])[0]
                with self._lock:
                    self._voltage = max(ai, 0.0)
            elif pt == 4 and ps >= 53:
                pose = list(struct.unpack("!6d", data[off+5:off+53]))
                with self._lock:
                    self._tcp_pose = pose
                got = True
            off += ps
        if got:
            self._ready_evt.set()

    def wait_ready(self, timeout: float = 5.0) -> bool:
        return self._ready_evt.wait(timeout=timeout)

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


# ── URScript sender ───────────────────────────────────────────────────────────
class URScriptSender:
    """Persistent TCP socket to port 30002 for sending URScript commands."""
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


# ── Motion ────────────────────────────────────────────────────────────────────
def movel(sender, state, x, y, z, rx, ry, rz,
          vel=MOVE_SPEED, acc=MOVE_ACCEL, tol=XYZ_TOL_M, timeout=30.0):
    cur = state.get_tcp_pose()
    if ((cur[0]-x)**2 + (cur[1]-y)**2 + (cur[2]-z)**2) ** 0.5 < tol:
        return
    sender.send(
        f"movel(p[{x:.6f},{y:.6f},{z:.6f},{rx:.6f},{ry:.6f},{rz:.6f}],"
        f"a={acc:.4f},v={vel:.4f})"
    )
    deadline = time.time() + timeout
    while time.time() < deadline:
        cur = state.get_tcp_pose()
        if ((cur[0]-x)**2 + (cur[1]-y)**2 + (cur[2]-z)**2) ** 0.5 < tol:
            return
        time.sleep(0.01)


# ── Helpers ───────────────────────────────────────────────────────────────────
def tcp_to_matrix(tcp_pose):
    R, _ = cv2.Rodrigues(np.array(tcp_pose[3:]))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = tcp_pose[:3]
    return T


def compute_hover_orientation(R_tag_base, R_hover_baseline):
    """Derive EEF yaw aligned to ArUco tag X-axis. Falls back to baseline."""
    try:
        def _wrap(a):
            return (a + np.pi) % (2 * np.pi) - np.pi
        x_tag    = R_tag_base[:, 0]
        yaw_tag  = np.arctan2(x_tag[1], x_tag[0])
        yaw_base = np.arctan2(R_hover_baseline[1, 0], R_hover_baseline[0, 0])
        delta_a  = _wrap(yaw_tag - yaw_base)
        delta_b  = _wrap(delta_a + np.pi)
        chosen   = delta_a if abs(delta_a) <= abs(delta_b) else delta_b
        if abs(chosen) > np.pi / 2:
            print(f"  [Orient] Large wrist rotation {np.degrees(chosen):.1f}° — proceeding.")
        c, s     = np.cos(chosen), np.sin(chosen)
        R_z      = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
        R_tgt    = R_z @ R_hover_baseline
        rvec, _  = cv2.Rodrigues(R_tgt)
        return tuple(float(v) for v in rvec.flatten())
    except Exception as e:
        print(f"  [Orient] Failed ({e}) — using baseline.")
        return (HOVER_RX, HOVER_RY, HOVER_RZ)


def pixel_to_base_frame(u, v, depth_m, K, T_cam2flange, tcp_pose):
    """Deproject (u, v) + depth_m in camera frame → 3D point in base frame."""
    fx, fy   = K[0, 0], K[1, 1]
    cx_k, cy_k = K[0, 2], K[1, 2]
    p_cam    = np.array([(u - cx_k) * depth_m / fx,
                          (v - cy_k) * depth_m / fy,
                          depth_m,
                          1.0])
    T_tcp2base = tcp_to_matrix(tcp_pose)
    p_base     = T_tcp2base @ T_cam2flange @ p_cam
    return p_base[:3]


def get_median_depth(depth_frame, u, v, radius=8):
    """
    Median depth in a small patch around (u, v), converting from OAK-D mm
    to metres.  u, v are in video-frame coordinates and are scaled to the
    depth frame resolution automatically.
    """
    dh, dw = depth_frame.shape
    us = int(u * dw / VIDEO_W)
    vs = int(v * dh / VIDEO_H)
    u1, u2 = max(0, us - radius), min(dw, us + radius)
    v1, v2 = max(0, vs - radius), min(dh, vs + radius)
    patch = depth_frame[v1:v2, u1:u2]
    valid = patch[patch > 0]
    if len(valid) == 0:
        return None
    return float(np.median(valid)) / 1000.0   # mm → m


# ── Centring correction ───────────────────────────────────────────────────────
def _center_horizontal(videoQueue, depthQueue, detect_fn,
                        K, T_cam2flange, state, sender, label=""):
    """
    Generic horizontal centring loop.

    detect_fn(frame, depth_frame) → (pixel_x, depth_z_m) or None
      pixel_x  : horizontal pixel of the target in the VIDEO frame
      depth_z_m: depth to the target in metres (used for px→m conversion)
    """
    fx       = float(K[0, 0])
    frame_cx = VIDEO_W / 2.0

    print(f"  [Centre{label}] tol={CENTER_H_TOL_PX}px  max={CENTER_H_MAX_ITER} moves …")

    for i in range(1, CENTER_H_MAX_ITER + 1):
        time.sleep(0.3)
        result = None
        for _ in range(15):
            frame       = videoQueue.get().getCvFrame()
            depth_frame = depthQueue.get().getFrame() if depthQueue is not None else None
            result      = detect_fn(frame, depth_frame)
            if result is not None:
                break

        if result is None:
            print(f"  [Centre{label} {i}] Target not visible — stopping.")
            break

        pixel_x, depth_z = result
        delta_px = pixel_x - frame_cx
        print(f"  [Centre{label} {i}]  offset={delta_px:+.1f}px", end="")

        if abs(delta_px) < CENTER_H_TOL_PX:
            print(f"  → within {CENTER_H_TOL_PX}px — done.")
            break

        delta_x_cam = delta_px * depth_z / fx
        tcp         = state.get_tcp_pose()
        R_eef, _    = cv2.Rodrigues(np.array(tcp[3:], dtype=np.float64))
        R_cam2base  = R_eef @ T_cam2flange[:3, :3]
        delta_base  = R_cam2base @ np.array([delta_x_cam, 0.0, 0.0])
        new_x = tcp[0] + delta_base[0]
        new_y = tcp[1] + delta_base[1]
        print(f"  →  dX={delta_base[0]:+.4f}  dY={delta_base[1]:+.4f} m")
        movel(sender, state, new_x, new_y, tcp[2], *tcp[3:])

    final = state.get_tcp_pose()
    print(f"  [Centre{label}] EEF: X={final[0]:.4f}  Y={final[1]:.4f}")
    return final


def center_horizontal_aruco(videoQueue, detector, K, dist_coeffs,
                             T_cam2flange, state, sender):
    def detect_fn(frame, _depth):
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(grey)
        if ids is None:
            return None
        for j, mid in enumerate(ids.flatten()):
            if mid != ARUCO_TAG_ID:
                continue
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners[j:j+1], MARKER_SIZE, K, dist_coeffs)
            px_cx = float(corners[j][0][:, 0].mean())
            return (px_cx, float(tvecs[0][0][2]))
        return None

    return _center_horizontal(videoQueue, None, detect_fn,
                               K, T_cam2flange, state, sender, label=" ArUco")


def center_horizontal_yolo(videoQueue, depthQueue, yolo_model,
                            K, T_cam2flange, state, sender):
    def detect_fn(frame, depth_frame):
        if depth_frame is None:
            return None
        results = yolo_model(frame, imgsz=640, conf=CONF_THRESHOLD, verbose=False)
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return None
        best  = max(r.boxes, key=lambda b: float(b.conf[0]))
        x1, y1, x2, y2 = map(int, best.xyxy[0])
        px_cx = (x1 + x2) / 2.0
        px_cy = (y1 + y2) / 2.0
        depth = get_median_depth(depth_frame, int(px_cx), int(px_cy))
        if depth is None:
            return None
        return (px_cx, depth)

    return _center_horizontal(videoQueue, depthQueue, detect_fn,
                               K, T_cam2flange, state, sender, label=" YOLO")


# ── Main ──────────────────────────────────────────────────────────────────────
def main(autonomous: bool = False):
    signal.signal(signal.SIGINT, lambda *_: os._exit(1))

    # ── Load calibration ──────────────────────────────────────────────────────
    print("Loading calibration …")
    K            = np.load(CALIB_DIR / "camera_matrix.npy")
    dist_coeffs  = np.zeros((4, 1))
    T_cam2flange = np.load(CALIB_DIR / "T_cam2flange.npy")
    print("  camera_matrix.npy  ✓")
    print("  T_cam2flange.npy   ✓\n")

    R_hover_baseline, _ = cv2.Rodrigues(
        np.array([HOVER_RX, HOVER_RY, HOVER_RZ], dtype=np.float64))

    # ── ArUco detector ────────────────────────────────────────────────────────
    dictionary = aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    detector   = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())

    # ── YOLO detection model ──────────────────────────────────────────────────
    print(f"Loading YOLO detection model …")
    if not DETECT_MODEL_PATH.exists():
        raise FileNotFoundError(f"Detection model not found: {DETECT_MODEL_PATH}")
    yolo_model = YOLO(str(DETECT_MODEL_PATH), task="detect")
    print(f"  {DETECT_MODEL_PATH.name}  ✓  classes={yolo_model.names}\n")

    # ── Robot ─────────────────────────────────────────────────────────────────
    print("Connecting to robot …")
    state  = RobotStateReader(ROBOT_IP)
    sender = URScriptSender(ROBOT_IP)
    state.start()
    if not state.wait_ready(timeout=5.0):
        raise RuntimeError(f"Robot state reader timed out — is {ROBOT_IP} reachable?")
    print("Robot connected!\n")

    # ── OAK-D pipeline: RGB video + stereo depth ──────────────────────────────
    pipeline = dai.Pipeline()

    cam = pipeline.create(dai.node.ColorCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setVideoSize(VIDEO_W, VIDEO_H)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.initialControl.setManualFocus(MANUAL_FOCUS)

    mono_l = pipeline.create(dai.node.MonoCamera)
    mono_r = pipeline.create(dai.node.MonoCamera)
    mono_l.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_r.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_l.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_r.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    mono_l.out.link(stereo.left)
    mono_r.out.link(stereo.right)

    xout_video = pipeline.create(dai.node.XLinkOut)
    xout_video.setStreamName("video")
    cam.video.link(xout_video.input)

    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    device     = dai.Device(pipeline)
    videoQueue = device.getOutputQueue(name="video", maxSize=4, blocking=False)
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    print("Camera started!\n")

    print("=" * 55)
    if autonomous:
        print(f"  AUTO  →  moves when detection stable {STABLE_FRAMES_NEEDED} frames")
    else:
        print("  SPACE  →  hover above detected object")
    print("  Q      →  quit / abort")
    print("=" * 55 + "\n")

    # ── Detection loop ────────────────────────────────────────────────────────
    obj_pos_base    = None
    obj_orient      = (HOVER_RX, HOVER_RY, HOVER_RZ)
    detection_mode  = None   # "aruco" or "yolo_only"
    stable_count    = 0
    last_stable_pos = None
    target_pose     = None

    while True:
        frame       = videoQueue.get().getCvFrame()
        depth_in    = depthQueue.get()
        depth_frame = depth_in.getFrame()

        grey                = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _     = detector.detectMarkers(grey)

        aruco_found = False
        yolo_found  = False

        # ── 1. Try ArUco ─────────────────────────────────────────────────────
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            for i, mid in enumerate(ids.flatten()):
                if mid != ARUCO_TAG_ID:
                    continue
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                    corners[i:i+1], MARKER_SIZE, K, dist_coeffs)
                rvec, tvec = rvecs[0][0], tvecs[0][0]
                cv2.drawFrameAxes(frame, K, dist_coeffs, rvec, tvec, MARKER_SIZE * 0.5)

                R_tag, _   = cv2.Rodrigues(rvec)
                T_tag2cam  = np.eye(4)
                T_tag2cam[:3, :3] = R_tag
                T_tag2cam[:3, 3]  = tvec
                tcp_pose   = state.get_tcp_pose()
                T_tcp2base = tcp_to_matrix(tcp_pose)
                T_tag2base = T_tcp2base @ T_cam2flange @ T_tag2cam

                pos        = T_tag2base[:3, 3].copy()
                pos[0]    += CALIB_X_OFFSET_M
                pos[1]    += CALIB_Y_OFFSET_M
                pos[2]    += CALIB_Z_OFFSET_M

                obj_pos_base   = pos
                obj_orient     = compute_hover_orientation(T_tag2base[:3, :3], R_hover_baseline)
                detection_mode = "aruco"
                aruco_found    = True

                bx, by, bz = pos
                cv2.putText(frame, f"ArUco base: ({bx:.3f}, {by:.3f}, {bz:.3f}) m",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 1)
                break

        # ── 2. Fall back to YOLO + depth ──────────────────────────────────────
        if not aruco_found:
            results = yolo_model(frame, imgsz=640, conf=CONF_THRESHOLD, verbose=False)
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                best = max(r.boxes, key=lambda b: float(b.conf[0]))
                x1, y1, x2, y2 = map(int, best.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)

                px_cx = (x1 + x2) / 2.0
                px_cy = (y1 + y2) / 2.0
                depth = get_median_depth(depth_frame, int(px_cx), int(px_cy))

                if depth is not None and depth > 0.05:
                    tcp_pose = state.get_tcp_pose()
                    pos      = pixel_to_base_frame(px_cx, px_cy, depth,
                                                   K, T_cam2flange, tcp_pose)
                    pos[0]  += CALIB_X_OFFSET_M
                    pos[1]  += CALIB_Y_OFFSET_M
                    pos[2]  += CALIB_Z_OFFSET_M

                    obj_pos_base   = pos
                    obj_orient     = (HOVER_RX, HOVER_RY, HOVER_RZ)   # fixed orientation
                    detection_mode = "yolo_only"
                    yolo_found     = True

                    label = yolo_model.names[int(best.cls[0])]
                    conf  = float(best.conf[0])
                    bx, by, bz = pos
                    cv2.putText(frame,
                                f"YOLO {label} {conf*100:.0f}%  "
                                f"base: ({bx:.3f}, {by:.3f}, {bz:.3f}) m",
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 165, 255), 1)

        detected = aruco_found or yolo_found

        # ── Stability tracking (autonomous) ───────────────────────────────────
        if autonomous:
            if detected and obj_pos_base is not None:
                if (last_stable_pos is not None and
                        np.linalg.norm(obj_pos_base - last_stable_pos) < STABLE_TOL_M):
                    stable_count += 1
                else:
                    stable_count = 1
                last_stable_pos = obj_pos_base.copy()
            else:
                stable_count    = 0
                last_stable_pos = None

        # ── Overlay ───────────────────────────────────────────────────────────
        mode_str = f"[{detection_mode}]" if detection_mode else ""
        if autonomous:
            label = (f"DETECTED {mode_str} — stable {stable_count}/{STABLE_FRAMES_NEEDED}"
                     if detected else "Searching …")
            color = (0, 200, 255) if detected else (0, 0, 255)
            hint  = "AUTO mode — Q to abort"
        else:
            label = (f"DETECTED {mode_str} — press SPACE to hover"
                     if detected else "Searching (ArUco + YOLO) …")
            color = (0, 255, 0) if detected else (0, 0, 255)
            hint  = "SPACE = hover  |  Q = quit"

        cv2.putText(frame, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.putText(frame, hint,  (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
        cv2.imshow("navigate", cv2.resize(frame, (960, 540)))
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            state.stop(); sender.close()
            cv2.destroyAllWindows(); device.close()
            return None

        # ── Execute hover ─────────────────────────────────────────────────────
        should_move = (autonomous and stable_count >= STABLE_FRAMES_NEEDED
                       and obj_pos_base is not None)
        should_move = should_move or (not autonomous and key == ord(' ')
                                      and detected and obj_pos_base is not None)

        if not autonomous and key == ord(' ') and not detected:
            print("[WARN] No object detected — aim camera at object first.")
            continue

        if should_move:
            rx, ry, rz  = obj_orient
            target_pose = [*obj_pos_base, rx, ry, rz]
            dist_mm     = np.linalg.norm(
                np.array(target_pose[:3]) - np.array(state.get_tcp_pose()[:3])) * 1000

            print("\n" + "=" * 60)
            print(f"  Mode:   {detection_mode}")
            print(f"  Base:   X={obj_pos_base[0]:.4f}  Y={obj_pos_base[1]:.4f}  Z={obj_pos_base[2]:.4f}")
            print(f"  Orient: rx={rx:.4f}  ry={ry:.4f}  rz={rz:.4f}")
            print(f"  Dist:   {dist_mm:.1f} mm")
            print("=" * 60)

            if not autonomous:
                confirm = input("  Type YES to move (hand on E-stop): ").strip()
                if confirm.upper() != "YES":
                    print("  Cancelled.\n")
                    continue

            print("  Moving to hover …")
            movel(sender, state, *target_pose)
            print("  Hover reached.")

            # ── Centring correction ───────────────────────────────────────────
            if detection_mode == "aruco":
                refined = center_horizontal_aruco(
                    videoQueue, detector, K, dist_coeffs,
                    T_cam2flange, state, sender)
            else:
                refined = center_horizontal_yolo(
                    videoQueue, depthQueue, yolo_model,
                    K, T_cam2flange, state, sender)

            target_pose = list(refined)
            break

    # ── Compute clearance_z and return ────────────────────────────────────────
    hover_z     = target_pose[2]
    clearance_z = hover_z + CLEARANCE_OFFSET_M

    src = "ArUco pose estimation" if detection_mode == "aruco" else "depth camera"
    print(f"\n  hover_z={hover_z:.4f} ({src})  clearance_z={clearance_z:.4f}")

    state.stop()
    sender.close()
    cv2.destroyAllWindows()
    device.close()
    print("Navigate done.\n")

    return {"hover_pose": target_pose, "clearance_z": clearance_z,
            "detection_mode": detection_mode}


if __name__ == "__main__":
    result = main()
    if result:
        print(f"hover_pose  = {[round(v, 4) for v in result['hover_pose']]}")
        print(f"clearance_z = {result['clearance_z']:.4f}")
