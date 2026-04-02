"""
explore.py
----------
Stage 1 of the full pipeline.

Behaviour
---------
1.  Move to saved scan pose.
2.  Sweep J0 across ±0.5 rad arc.
3.  On every frame run BOTH ArUco (ID 13) and YOLO detection.
4.  Stop sweep when the detected object/tag is within ±80 px of the
    image horizontal centre.
5.  If neither detector fires after SCAN_MAX_SWEEPS → abort.

Returns True on success, None on abort.
Navigate re-detects the object live on every attempt — no position is
returned here.

Motion notes
------------
All robot motion is sent as raw URScript over port 30002.
No RTDE — caused 10-second reconnect hangs (pipeline_dev/RTDE_debug_log.md).
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

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR     = Path(__file__).resolve().parent
CALIB_DIR      = SCRIPT_DIR / "calibration"
DATA_DIR       = SCRIPT_DIR / "data"
SCAN_POSE_FILE = DATA_DIR / "scan_pose.json"
DETECT_MODEL_PATH = (SCRIPT_DIR / "models" / "detection"
                     / "yolo26n_detect_V1" / "weights" / "best.pt")

# ── Robot ────────────────────────────────────────────────────────────────────
ROBOT_IP       = "192.168.8.102"
ROBOT_PORT     = 30002
DASHBOARD_PORT = 29999

# ── Scan pose ─────────────────────────────────────────────────────────────────
def load_scan_pose():
    if not SCAN_POSE_FILE.exists():
        raise FileNotFoundError(
            f"Scan pose file not found: {SCAN_POSE_FILE}\n"
            "Run  python temp/capture_scan_pose.py  to record it first."
        )
    with open(SCAN_POSE_FILE) as f:
        return json.load(f)["joint_angles"]

# ── Sweep config ──────────────────────────────────────────────────────────────
SWEEP_START_RAD  = -0.5
SWEEP_END_RAD    =  0.5
SWEEP_SPEED      =  0.2   # rad/s
SWEEP_ACCEL      =  0.1   # rad/s²
SWEEP_STOP_ACCEL =  0.8   # rad/s²  (hard stop)

SCAN_MAX_SWEEPS  = 2
RETRY_LOWER_M    = 0.10   # metres to descend on retry
SCAN_TABLE_Z_M   = -0.05  # approximate table Z in base frame

RETRY_SPEED_MS   = 0.05   # m/s
RETRY_ACCEL_MS   = 0.02   # m/s²
CARTESIAN_TOL_M  = 0.005  # m

# ── Detection / centring ──────────────────────────────────────────────────────
ARUCO_DICT_TYPE  = aruco.DICT_6X6_250
ARUCO_TAG_ID     = 13
MARKER_SIZE      = 0.036   # metres
CONF_THRESHOLD   = 0.80    # YOLO confidence
CENTER_TOL_PX    = 80      # px — stop sweep when object within this of frame centre
CENTER_CONSEC    = 3       # consecutive centred frames required before stopping

# ── Camera ────────────────────────────────────────────────────────────────────
VIDEO_W      = 1280
VIDEO_H      = 720
MANUAL_FOCUS = 46

# ── Move speeds (to scan pose) ────────────────────────────────────────────────
APPROACH_SPEED = 0.5   # rad/s
APPROACH_ACCEL = 0.3   # rad/s²
JOINT_TOL_RAD  = 0.01  # rad

# ── Gripper ───────────────────────────────────────────────────────────────────
GRIP_OPEN_URP = "/programs/myu/open_gripper.urp"
GRIP_OPEN_MM  = 85.0


# ── Robot state reader ────────────────────────────────────────────────────────
class RobotStateReader(threading.Thread):
    """
    Reads the UR secondary-client stream (port 30002) in a background thread.
    Type 1 → joint positions, Type 2 → AI2 voltage, Type 4 → TCP pose.
    """
    def __init__(self, ip, port=ROBOT_PORT):
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
    def _recvall(s, n):
        buf = b""
        while len(buf) < n:
            chunk = s.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def _parse_subpackets(self, data):
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

    def get_joint_positions(self):
        with self._lock:
            return list(self._joints)

    def get_tcp_pose(self):
        with self._lock:
            return list(self._tcp_pose)

    def get_width_mm(self):
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
    def __init__(self, ip, port=ROBOT_PORT):
        self.ip    = ip
        self.port  = port
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

    def send(self, script):
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


# ── Motion helpers ────────────────────────────────────────────────────────────
def movej_joints(sender, state, joint_angles, vel, acc,
                 tol=JOINT_TOL_RAD, timeout=30.0, stop_check=None):
    cur = state.get_joint_positions()
    if max(abs(c - t) for c, t in zip(cur, joint_angles)) < tol:
        return
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


def stopj(sender, acc):
    sender.send(f"stopj({acc:.4f})")


def tcp_to_matrix(tcp_pose):
    R, _ = cv2.Rodrigues(np.array(tcp_pose[3:]))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = tcp_pose[:3]
    return T


def _dashboard_cmd(robot_ip, cmd):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(5.0)
        s.connect((robot_ip, DASHBOARD_PORT))
        s.recv(1024)
        s.sendall((cmd + "\n").encode())
        return s.recv(1024).decode().strip()


def open_gripper(robot_ip, state):
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


def movel_pose(sender, state, x, y, z, rx, ry, rz, vel, acc, timeout=30.0):
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
    print("  [Explore] movel timeout — continuing.")


def _compute_lower_tcp(scan_tcp, T_cam2flange, lower_m, table_z):
    hx, hy, hz     = scan_tcp[:3]
    hrx, hry, hrz  = scan_tcp[3:]
    new_z = hz - lower_m
    try:
        R_eef, _   = cv2.Rodrigues(np.array([hrx, hry, hrz], dtype=np.float64))
        p_cam      = R_eef @ T_cam2flange[:3, 3]
        z_cam_base = R_eef @ T_cam2flange[:3, 2]
        if abs(z_cam_base[2]) < 1e-3:
            return None
        cam_z_orig = hz + p_cam[2]
        t1         = (table_z - cam_z_orig) / z_cam_base[2]
        look_x     = hx + p_cam[0] + t1 * z_cam_base[0]
        look_y     = hy + p_cam[1] + t1 * z_cam_base[1]
        cam_z_new  = new_z + p_cam[2]
        t2         = (table_z - cam_z_new) / z_cam_base[2]
        new_x      = look_x - p_cam[0] - t2 * z_cam_base[0]
        new_y      = look_y - p_cam[1] - t2 * z_cam_base[1]
        return [new_x, new_y, new_z, hrx, hry, hrz]
    except Exception:
        return None


# ── Detection (ArUco first, YOLO fallback) ────────────────────────────────────
def detect_objects(frame, K, dist_coeffs, aruco_detector, yolo_model):
    """
    Run ArUco detection first; if not found, run YOLO.
    Annotates frame in-place.

    Returns (found: bool, pixel_x: float|None, mode: str|None)
      pixel_x — horizontal pixel of the target centre in the video frame
      mode    — "aruco" or "yolo_only"
    """
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── ArUco ─────────────────────────────────────────────────────────────────
    corners, ids, _ = aruco_detector.detectMarkers(grey)
    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)
        for i, mid in enumerate(ids.flatten()):
            if mid != ARUCO_TAG_ID:
                continue
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners[i:i+1], MARKER_SIZE, K, dist_coeffs)
            cv2.drawFrameAxes(frame, K, dist_coeffs,
                               rvecs[0][0], tvecs[0][0], MARKER_SIZE * 0.5)
            pixel_x = float(corners[i][0][:, 0].mean())
            return True, pixel_x, "aruco"

    # ── YOLO ─────────────────────────────────────────────────────────────────
    results = yolo_model(frame, imgsz=640, conf=CONF_THRESHOLD, verbose=False)
    r = results[0]
    if r.boxes is not None and len(r.boxes) > 0:
        best = max(r.boxes, key=lambda b: float(b.conf[0]))
        x1, y1, x2, y2 = map(int, best.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        label = yolo_model.names[int(best.cls[0])]
        conf  = float(best.conf[0])
        cv2.putText(frame, f"YOLO {label} {conf*100:.0f}%", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
        pixel_x = (x1 + x2) / 2.0
        return True, pixel_x, "yolo_only"

    return False, None, None


# ── Main ──────────────────────────────────────────────────────────────────────
def main(autonomous: bool = False):
    signal.signal(signal.SIGINT, lambda *_: os._exit(1))

    # ── Calibration ───────────────────────────────────────────────────────────
    print("Loading calibration …")
    K            = np.load(CALIB_DIR / "camera_matrix.npy")
    dist_coeffs  = np.zeros((4, 1))
    T_cam2flange = np.load(CALIB_DIR / "T_cam2flange.npy")
    print("  camera_matrix.npy  ✓")
    print("  T_cam2flange.npy   ✓\n")

    # ── ArUco detector ────────────────────────────────────────────────────────
    dictionary     = aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    aruco_detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())

    # ── YOLO detection model ──────────────────────────────────────────────────
    print("Loading YOLO detection model …")
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

    # ── Camera ────────────────────────────────────────────────────────────────
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
    device     = dai.Device(pipeline)
    videoQueue = device.getOutputQueue(name="video", maxSize=4, blocking=False)
    print("Camera started!\n")

    frame_cx = VIDEO_W / 2.0   # image horizontal centre (pixels)

    # ── Scan pose ─────────────────────────────────────────────────────────────
    print("=" * 55)
    SCAN_JOINT_POS = load_scan_pose()
    print(f"  Scan pose: {[round(j, 3) for j in SCAN_JOINT_POS]}")
    if not autonomous:
        print("  Press ENTER to move to scan pose (hand on E-stop) …")
        input()
    print("  Moving to scan pose …")
    movej_joints(sender, state, SCAN_JOINT_POS, APPROACH_SPEED, APPROACH_ACCEL)
    print("  Scan pose reached.")
    open_gripper(ROBOT_IP, state)
    print()

    if not autonomous:
        print("  Press ENTER to start sweep …")
        input()

    # ── Sweep loop ────────────────────────────────────────────────────────────
    current_scan_pose = list(SCAN_JOINT_POS)

    # Shared result dict — written by camera loop, read by sweep thread
    result = {"found": False, "centred": False, "mode": None}
    consec_centred = [0]   # consecutive centred-frame counter

    for sweep_num in range(1, SCAN_MAX_SWEEPS + 1):

        # Descend on retry
        if sweep_num > 1:
            scan_tcp  = state.get_tcp_pose()
            lower_tcp = _compute_lower_tcp(scan_tcp, T_cam2flange,
                                           RETRY_LOWER_M, SCAN_TABLE_Z_M)
            if lower_tcp is None:
                lower_tcp = [scan_tcp[0], scan_tcp[1], scan_tcp[2] - RETRY_LOWER_M,
                             *scan_tcp[3:]]
                print(f"\n  [Retry {sweep_num}] Descending straight {RETRY_LOWER_M*100:.0f} cm …")
            else:
                print(f"\n  [Retry {sweep_num}] Camera-compensated descent to "
                      f"X={lower_tcp[0]:.4f}  Y={lower_tcp[1]:.4f}  Z={lower_tcp[2]:.4f} …")
            movel_pose(sender, state, *lower_tcp, RETRY_SPEED_MS, RETRY_ACCEL_MS)
            current_scan_pose = state.get_joint_positions()
            print("  Lower scan pose reached.")

        sweep_start = list(current_scan_pose); sweep_start[0] += SWEEP_START_RAD
        sweep_end   = list(current_scan_pose); sweep_end[0]   += SWEEP_END_RAD
        print(f"  Sweep {sweep_num}/{SCAN_MAX_SWEEPS}: "
              f"J0 {sweep_start[0]:.3f} → {sweep_end[0]:.3f} rad")

        sweep_done = threading.Event()
        consec_centred[0] = 0   # reset counter for this sweep

        def _sweep(s_start=sweep_start, s_end=sweep_end):
            for waypoint in (s_end, s_start):
                if result["centred"]:
                    break
                movej_joints(sender, state, waypoint, SWEEP_SPEED, SWEEP_ACCEL,
                             stop_check=lambda: result["centred"])
                # Dwell at endpoint so camera can detect while arm is still
                t_end = time.time() + 0.6
                while time.time() < t_end:
                    if result["centred"]:
                        break
                    time.sleep(0.05)
            sweep_done.set()

        sweep_thread = threading.Thread(target=_sweep, daemon=True)
        sweep_thread.start()

        print("  Sweeping … (Q to abort)\n")

        while not sweep_done.is_set():
            frame = videoQueue.get().getCvFrame()
            found, pixel_x, mode = detect_objects(
                frame, K, dist_coeffs, aruco_detector, yolo_model)

            if found:
                result["found"] = True
                result["mode"]  = mode
                offset_px = pixel_x - frame_cx
                centred   = abs(offset_px) < CENTER_TOL_PX

                if centred:
                    consec_centred[0] += 1
                else:
                    consec_centred[0] = 0

                if consec_centred[0] >= CENTER_CONSEC and not result["centred"]:
                    result["centred"] = True
                    print(f"  ✓ Object centred [{mode}]  offset={offset_px:+.0f}px  "
                          f"({CENTER_CONSEC} consecutive) — stopping sweep")
                    stopj(sender, SWEEP_STOP_ACCEL)

                # Overlay
                cx_draw = int(pixel_x)
                cv2.line(frame, (cx_draw, 0), (cx_draw, VIDEO_H), (0, 255, 0), 1)
                cv2.line(frame, (int(frame_cx), 0), (int(frame_cx), VIDEO_H),
                         (255, 255, 255), 1)
                offset_str = f"offset={offset_px:+.0f}px"
                status = (f"CENTRED [{mode}]  {offset_str}"
                          if result["centred"] else
                          f"DETECTED [{mode}]  {offset_str} — centering …")
                cv2.putText(frame, status, (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            else:
                cv2.putText(frame,
                            f"Sweep {sweep_num}/{SCAN_MAX_SWEEPS} — searching (ArUco + YOLO) …",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            cv2.imshow("explore", cv2.resize(frame, (960, 540)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stopj(sender, SWEEP_STOP_ACCEL)
                sweep_done.set()
                break

        sweep_thread.join(timeout=2.0)

        if result["centred"]:
            break   # done — centred detection achieved
        if sweep_num < SCAN_MAX_SWEEPS:
            print(f"  Not centred on sweep {sweep_num} — descending for closer view …")

    cv2.destroyAllWindows()

    # ── Result ────────────────────────────────────────────────────────────────
    if result["found"]:
        centred_str = "centred" if result["centred"] else "detected (not centred — navigate will correct)"
        print("\n" + "=" * 55)
        print(f"  EXPLORE COMPLETE  [{result['mode']}]  {centred_str}")
        print("=" * 55)
    else:
        print(f"\n  Object NOT found after {SCAN_MAX_SWEEPS} sweeps — aborting.")

    state.stop()
    sender.close()
    device.close()
    print("Explore done.\n")

    return True if result["found"] else None


if __name__ == "__main__":
    main()
