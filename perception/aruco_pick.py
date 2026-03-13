"""
aruco_pick.py
=============
Detects an ArUco marker (stuck on top of an object) using the OAK-D camera,
measures its 3D position using stereo depth, transforms the position to the
robot base frame using the hand-eye calibration, then commands the UR10 to
descend and pick it up.

Prerequisites:
  - Perception/ur10_cam_offset.json   (run calibrate_handeye.py first)

Run:
    python Perception/aruco_pick.py

Keys (live camera window):
  SPACE  — pick the currently detected ArUco marker
  R      — re-detect (refresh)
  Q      — quit
"""

import os
os.environ["DEPTHAI_BOOT_TIMEOUT"] = "30000"

import json
import socket
import struct
import sys
import time
import threading
from pathlib import Path

import cv2
import cv2.aruco as aruco
import depthai as dai
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
ROBOT_IP        = "192.168.8.102"
CALIB_PATH      = Path(__file__).parent.parent / "Calibration" / "eye_in_hand" / "handeye_calibration.json"
LIMITS_PATH     = Path(__file__).parent / "safe_limits.json"

ARUCO_DICT_ID = aruco.DICT_6X6_250     # change to match your tag
TARGET_MARKER_ID = None                # None = pick first detected, or set e.g. 0

APPROACH_HEIGHT = 0.150   # m
PICK_HEIGHT     = 0.050   # m
LIFT_HEIGHT     = 0.200   # m

# ── MANUAL TUNING OVERRIDES ──────────────────────────────────────────────
# These define where the Gripper (TCP) is relative to the Camera.
# Use W/A/S/D to align the Cyan crosshair with the real gripper tip.
TCP_IN_CAM_X = 0.000  # m (Screen Horizontal: A/D)
TCP_IN_CAM_Y = 0.000  # m (Screen Vertical: W/S)
TCP_IN_CAM_Z = 0.200  # m (Distance from lens: R/F)

PITCH_DEG = 45.0      # deg (Tilt Forward/Down)
YAW_DEG   = 180.0     # deg (Rotate around arm)
ROLL_DEG  = 0.0       # deg (Horizon)

USE_MANUAL_CALIB = True  # True = use manual values above
# ─────────────────────────────────────────────────────────────────────────

MOVE_VEL        = 0.08    # m/s
MOVE_ACC        = 0.08    # m/s²

# Depth patch half-size (px) — sample area around tag centre for depth
DEPTH_PATCH_PX  = 15

# ANSI colours
GREEN = "\033[92m"; RED = "\033[91m"; RESET = "\033[0m"; BOLD = "\033[1m"


# ─────────────────────────────────────────────────────────────────────────────
# Robot interface
# ─────────────────────────────────────────────────────────────────────────────
def get_tcp_pose(ip=ROBOT_IP):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            s.connect((ip, 30003))
            data = s.recv(1200)
            if len(data) < 492:
                return None
            return list(struct.unpack("!6d", data[444:492]))
    except Exception:
        return None


def send_urscript(script: str, ip=ROBOT_IP, port=30001):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(5.0)
            s.connect((ip, port))
            s.sendall((script + "\n").encode())
        return True
    except Exception as e:
        print(f"[Robot] send error: {e}")
        return False


def movel(x, y, z, rx, ry, rz, vel=MOVE_VEL, acc=MOVE_ACC):
    script = (
        f"def pick_move():\n"
        f"  movel(p[{x:.6f},{y:.6f},{z:.6f},{rx:.6f},{ry:.6f},{rz:.6f}],"
        f"a={acc},v={vel})\n"
        f"end\n"
    )
    send_urscript(script)


def wait_at(target_xyz, tol=0.004, timeout=25.0):
    """Block until robot TCP is within tol metres of target_xyz."""
    t0 = time.time()
    time.sleep(0.4)   # let motion start
    while time.time() - t0 < timeout:
        pose = get_tcp_pose()
        if pose and np.linalg.norm(np.array(pose[:3]) - np.array(target_xyz)) < tol:
            return True
        time.sleep(0.1)
    print("[Robot] Warning: arrival timeout")
    return False


def rotvec_to_matrix(rx, ry, rz):
    vec   = np.array([rx, ry, rz], dtype=float)
    angle = np.linalg.norm(vec)
    if angle < 1e-9:
        return np.eye(3)
    axis = vec / angle
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


# ─────────────────────────────────────────────────────────────────────────────
# Calibration loader
# ─────────────────────────────────────────────────────────────────────────────
def load_calibration(path):
    with open(path) as f:
        cal = json.load(f)
    # handeye_calibration.json uses R_cam2tcp / t_cam2tcp
    t = np.array(cal["t_cam2tcp"])
    R = np.array(cal["R_cam2tcp"])
    method = cal.get("best_method", "unknown")
    print(f"[Calib] Loaded {path.name}  method={method}")
    print(f"[Calib] camera-to-TCP: "
          f"dx={t[0]*1000:.1f}  dy={t[1]*1000:.1f}  dz={t[2]*1000:.1f} mm")
    return R, t


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate transform: camera → TCP → base
# ─────────────────────────────────────────────────────────────────────────────
def build_full_rot(p, y, r):
    """Combine Pitch (X), Yaw (Y), and Roll (Z) into one rotation matrix."""
    # Rotation matrices
    def rx(a):
        c, s = np.cos(np.radians(a)), np.sin(np.radians(a))
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    def ry(a):
        c, s = np.cos(np.radians(a)), np.sin(np.radians(a))
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    def rz(a):
        c, s = np.cos(np.radians(a)), np.sin(np.radians(a))
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    # Sequence: Pitch then Yaw then Roll
    return rz(r) @ ry(y) @ rx(p)

def cam_to_base(X_cam, Y_cam, Z_cam, tcp_pose, R_cam_tcp, t_cam_tcp):
    global USE_MANUAL_CALIB, TCP_IN_CAM_X, TCP_IN_CAM_Y, TCP_IN_CAM_Z, PITCH_DEG, YAW_DEG, ROLL_DEG
    P_cam  = np.array([X_cam, Y_cam, Z_cam])
    
    if USE_MANUAL_CALIB:
        R_use = build_full_rot(PITCH_DEG, YAW_DEG, ROLL_DEG)
        # Offset from camera center to gripper tip in camera frame
        P_tcp_in_cam = np.array([TCP_IN_CAM_X, TCP_IN_CAM_Y, TCP_IN_CAM_Z])
        # Direct vector from TCP to Object in Camera-aligned orientation
        # (Correcting the offset before applying robot-base rotation)
        P_tcp = R_use @ (P_cam - P_tcp_in_cam)
    else:
        # Standard matrix: P_tcp = R @ P_cam + t
        P_tcp = R_cam_tcp @ P_cam + t_cam_tcp

    R_base_tcp = rotvec_to_matrix(*tcp_pose[3:])
    P_base = R_base_tcp @ P_tcp + np.array(tcp_pose[:3])
    return P_base


# ─────────────────────────────────────────────────────────────────────────────
# TCP overlay — project TCP position into camera preview frame
# ─────────────────────────────────────────────────────────────────────────────
def project_tcp_to_image(tcp_pose, R_cam_tcp, t_cam_tcp,
                         fx, fy, cx0, cy0, dx_scale, dy_scale):
    global USE_MANUAL_CALIB, TCP_IN_CAM_X, TCP_IN_CAM_Y, TCP_IN_CAM_Z
    
    if USE_MANUAL_CALIB:
        # Tuning is in direct camera space: X-Right, Y-Down, Z-Forward
        P_tcp_cam = np.array([TCP_IN_CAM_X, TCP_IN_CAM_Y, TCP_IN_CAM_Z])
    else:
        # Project from R/t matrices: P_cam = -R.T @ t (Position of TCP in Cam frame)
        P_tcp_cam = -R_cam_tcp.T @ t_cam_tcp

    # Force a small positive Z for math safety
    z_vis = max(0.001, P_tcp_cam[2])
    u = (fx * P_tcp_cam[0] / z_vis + cx0)
    v = (fy * P_tcp_cam[1] / z_vis + cy0)
    
    return (int(u / dx_scale), int(v / dy_scale))


def draw_tcp_overlay(img, tcp_proj, label="TCP"):
    """Draw a cyan crosshair + label at the projected TCP position."""
    if tcp_proj is None:
        return
    x, y = tcp_proj
    h, w = img.shape[:2]
    if 0 <= x < w and 0 <= y < h:
        cv2.drawMarker(img, (x, y), (255, 255, 0),
                       cv2.MARKER_CROSS, 30, 2)
        cv2.putText(img, label, (x + 16, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

def draw_3d_axis(img, R, t, K, length=0.05):
    """Draw XYZ axes based on rotation R and translation t in camera frame."""
    # Points in 3D (X, Y, Z, Origin)
    pts = np.array([
        [length, 0, 0], [0, length, 0], [0, 0, length], [0, 0, 0]
    ]).T
    
    # Transform to camera frame
    # If R/t are from Tag-to-Camera, then P_cam = R @ P_tag + t
    # If we are drawing the TCP, R/t must be TCP-to-Camera
    pts_cam = R @ pts + t.reshape(3, 1)
    
    # Project
    proj = []
    # Ensure K is numpy for indexing
    K_np = np.array(K)
    for i in range(4):
        z = pts_cam[2, i]
        if z <= 0: return # Behind camera
        u = int(K_np[0,0] * pts_cam[0,i] / z + K_np[0,2])
        v = int(K_np[1,1] * pts_cam[1,i] / z + K_np[1,2])
        proj.append((u, v))
    
    # Draw (X=Red, Y=Green, Z=Blue)
    p0 = proj[3] # Origin
    cv2.line(img, p0, proj[0], (0, 0, 255), 2) # X
    cv2.line(img, p0, proj[1], (0, 255, 0), 2) # Y
    cv2.line(img, p0, proj[2], (255, 0, 0), 2) # Z


def load_guard(limits_path):
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "UR10"))
        from safety_guard import SafetyGuard, SafetyViolation
        guard = SafetyGuard(limits_path=limits_path)
        return guard, SafetyViolation
    except Exception as e:
        print(f"[Safety] Could not load guard: {e}")
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# OAK-D pipeline — colour preview + stereo depth
# ─────────────────────────────────────────────────────────────────────────────
def create_pipeline():
    p = dai.Pipeline()

    cam = p.create(dai.node.ColorCamera)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setIspScale(2, 3)   # 1080p -> 720p (1920*2/3=1280, 1080*2/3=720)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(15)          # lowered from 20 to 15 to reduce heating

    ml = p.create(dai.node.MonoCamera)
    mr = p.create(dai.node.MonoCamera)
    st = p.create(dai.node.StereoDepth)

    ml.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    ml.setBoardSocket(dai.CameraBoardSocket.LEFT)
    mr.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mr.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    st.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    st.setDepthAlign(dai.CameraBoardSocket.RGB)
    st.setSubpixel(True)

    ml.out.link(st.left)
    mr.out.link(st.right)

    xrgb  = p.create(dai.node.XLinkOut); xrgb.setStreamName("rgb")
    xdep  = p.create(dai.node.XLinkOut); xdep.setStreamName("depth")
    cam.isp.link(xrgb.input)
    st.depth.link(xdep.input)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Pick sequence
# ─────────────────────────────────────────────────────────────────────────────
def pick_aruco(tag_xyz_cam, tcp, R_cam_tcp, t_cam_tcp, guard, SafetyViolation):
    """Full pick sequence for one detected ArUco tag."""
    # Target in base frame using active tuning (cam_to_base handles globals)
    P_obj = cam_to_base(tag_xyz_cam[0], tag_xyz_cam[1], tag_xyz_cam[2],
                        tcp, R_cam_tcp, t_cam_tcp)

    rx, ry, rz = tcp[3], tcp[4], tcp[5]
    print(f"[Pick] Target Base (with offsets): X={P_obj[0]*1000:.1f} Y={P_obj[1]*1000:.1f} Z={P_obj[2]*1000:.1f} mm")

    # Safety check - CHECK BEFORE APPROACH
    approach_z = P_obj[2] + APPROACH_HEIGHT
    pick_z     = P_obj[2] + PICK_HEIGHT
    lift_z     = P_obj[2] + LIFT_HEIGHT

    if guard is not None:
        for label, pz in [("approach", approach_z), ("pick", pick_z)]:
            try:
                guard.check(P_obj[0], P_obj[1], pz)
            except SafetyViolation as e:
                print(f"{RED}[Pick] SAFETY VIOLATION! {label} pose rejected. Check your Z-alignment!{RESET}")
                print(f"       Details: {e}")
                return False

    # ── Motion sequence ──────────────────────────────────────────────────────
    print("[Pick] Opening gripper before approach...")
    _gripper_open()
    time.sleep(2.0)   # wait for gripper to fully open

    print("[Pick] Step 1: Approach (hover above tag)...")
    movel(P_obj[0], P_obj[1], approach_z, rx, ry, rz)
    wait_at([P_obj[0], P_obj[1], approach_z])

    print("[Pick] Step 2: Descend to pick height...")
    movel(P_obj[0], P_obj[1], pick_z, rx, ry, rz, vel=0.04, acc=0.04)
    wait_at([P_obj[0], P_obj[1], pick_z])
    time.sleep(0.3)

    print("[Pick] Step 3: Closing gripper...")
    _gripper_close()
    time.sleep(2.5)

    print("[Pick] Step 4: Lifting...")
    movel(P_obj[0], P_obj[1], lift_z, rx, ry, rz)
    wait_at([P_obj[0], P_obj[1], lift_z])

    print("[Pick] Done!")
    return True


def _gripper_close(ip=ROBOT_IP):
    r1 = _dashboard("stop");            time.sleep(0.5)
    r2 = _dashboard("load grip_close.urp"); time.sleep(0.3)
    r3 = _dashboard("play")
    print(f"  [Gripper] close → stop:{r1!r}  load:{r2!r}  play:{r3!r}")


def _gripper_open(ip=ROBOT_IP):
    r1 = _dashboard("stop");            time.sleep(0.5)
    r2 = _dashboard("load grip_open.urp"); time.sleep(0.3)
    r3 = _dashboard("play")
    print(f"  [Gripper] open → stop:{r1!r}  load:{r2!r}  play:{r3!r}")


def _dashboard(cmd, ip=ROBOT_IP, port=29999):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(3.0)
            s.connect((ip, port))
            s.recv(1024)
            s.sendall((cmd + "\n").encode())
            return s.recv(1024).decode().strip()
    except Exception as e:
        print(f"[Dashboard] {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not CALIB_PATH.exists():
        print(f"ERROR: {CALIB_PATH} not found — run calibrate_handeye.py first.")
        sys.exit(1)

    R_cam_tcp, t_cam_tcp = load_calibration(CALIB_PATH)
    guard, SafetyViolation = load_guard(LIMITS_PATH)

    global TCP_IN_CAM_X, TCP_IN_CAM_Y, TCP_IN_CAM_Z, USE_MANUAL_CALIB, PITCH_DEG, YAW_DEG, ROLL_DEG

    aruco_dict     = aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    aruco_params   = aruco.DetectorParameters()
    
    # AGGRESSIVE detection for speckled background
    aruco_params.adaptiveThreshWinSizeStep = 2
    aruco_params.minMarkerPerimeterRate = 0.02 # Detect smaller tags
    aruco_params.adaptiveThreshConstant = 7
    aruco_params.polygonalApproxAccuracyRate = 0.05
    
    aruco_detector = aruco.ArucoDetector(aruco_dict, aruco_params)

    cfg = dai.Device.Config()
    with dai.Device(cfg) as device:
        device.startPipeline(create_pipeline())

        cal  = device.readCalibration()
        # getDefaultIntrinsics returns values at the FULL sensor resolution.
        # We must scale them down to match the actual preview frame size.
        M_full, w_full, h_full = cal.getDefaultIntrinsics(dai.CameraBoardSocket.RGB)

        q_rgb   = device.getOutputQueue("rgb",   maxSize=4, blocking=False)
        q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)

        _first = q_rgb.get().getCvFrame()
        _depth_first = q_depth.get().getFrame()
        frame_h, frame_w = _first.shape[:2]
        print(f"[Camera] Running at {frame_w}x{frame_h} (aligned depth & RGB)")

        # Get intrinsics at the ACTUAL frame resolution
        # getCameraIntrinsics handles everything correctly
        M = cal.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, frame_w, frame_h)
        fx, fy, cx0, cy0 = M[0][0], M[1][1], M[0][2], M[1][2]
        print(f"[Camera] Intrinsics: fx={fx:.1f}  fy={fy:.1f}  cx={cx0:.1f}  cy={cy0:.1f}")

        print("\n=== ArUco Pick ===")
        print("  SPACE  — pick detected tag")
        print("  O      — open gripper")
        print("  C      — close gripper")
        print("  Q      — quit\n")

        # Initial gripper state: OPEN
        print("[Start] Opening gripper...")
        _gripper_open()

        last_tag = None   # (cx_px, cy_px, X_cam, Y_cam, Z_cam)

        while True:
            # Drain queues: get the LATEST available frame, avoid latency buildup
            img_msg = None
            while q_rgb.has(): img_msg = q_rgb.get()
            if img_msg is None: continue # Wait for first frame
            
            dep_msg = None
            while q_depth.has(): dep_msg = q_depth.get()
            if dep_msg is None: continue

            frame = img_msg.getCvFrame()
            depth = dep_msg.getFrame()
            disp  = frame.copy()
            
            # Simple HUD Background (Opaque for clarity)
            cv2.rectangle(disp, (0, 0), (450, 165), (0, 0, 0), -1) 
            
            tcp_live = get_tcp_pose()
            
            # ── HUD ──
            y0 = 25
            mode_str = "MANUAL TUNING (Use WASD/IKJL)" if USE_MANUAL_CALIB else "FIXED MATRIX (Wait for Space)"
            cv2.putText(disp, f"MODE: {mode_str}", (15, y0), 2, 0.65, (0, 255, 0), 2)
            
            cv2.putText(disp, f"ANGLES: P:{PITCH_DEG:.1f} Y:{YAW_DEG:.1f} R:{ROLL_DEG:.1f}", (15, y0+30), 2, 0.5, (0,255,255), 1)
            cv2.putText(disp, f"TCP OFFSET(mm): X:{TCP_IN_CAM_X*1000:+.0f} Y:{TCP_IN_CAM_Y*1000:+.0f} Z:{TCP_IN_CAM_Z*1000:+.0f}", (15, y0+55), 2, 0.5, (0,255,255), 1)
            cv2.putText(disp, "W/S/A/D: Shift Cross (10mm) | I/K/J/L: Tilt/Pan", (15, y0+80), 2, 0.45, (255,255,255), 1)
            cv2.putText(disp, "T: Toggle Manual Tuning  |  X: Reset Tuning", (15, y0+105), 2, 0.45, (255,255,255), 1)

            # ── TCP overlay ───────────────────────────────────────────────────
            if tcp_live is not None:
                tcp_proj = project_tcp_to_image(tcp_live, R_cam_tcp, t_cam_tcp, fx, fy, cx0, cy0, 1.0, 1.0)
                draw_tcp_overlay(disp, tcp_proj, label="GRIPPER TIP")

                # Axes at TCP
                if USE_MANUAL_CALIB: 
                    t_tcp_cam = np.array([TCP_IN_CAM_X, TCP_IN_CAM_Y, TCP_IN_CAM_Z])
                else: 
                    t_tcp_cam = -R_cam_tcp.T @ t_cam_tcp
                
                draw_3d_axis(disp, np.eye(3), t_tcp_cam, M, length=0.04)
                
                cv2.putText(disp, f"ROBOT TCP BASE: {tcp_live[0]*1000:.0f}, {tcp_live[1]*1000:.0f}, {tcp_live[2]*1000:.0f} mm",
                            (15, disp.shape[0]-20), 2, 0.45, (255, 255, 0), 1)

            # Detect ArUco... (Apply light blur to kill speckle noise)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3,3), 0)
            corners, ids, _ = aruco_detector.detectMarkers(gray)

            last_tag = None
            if ids is not None:
                aruco.drawDetectedMarkers(disp, corners, ids)

                for i, marker_id in enumerate(ids.flatten()):
                    if TARGET_MARKER_ID is not None and marker_id != TARGET_MARKER_ID:
                        continue

                    # Tag centre pixel
                    pts = corners[i][0]
                    cx_px = int(pts[:, 0].mean())
                    cy_px = int(pts[:, 1].mean())

                    # Resolution is 1:1, no mapping needed
                    # Sample depth patch
                    r     = DEPTH_PATCH_PX
                    patch = depth[max(0, cy_px-r):cy_px+r+1,
                                  max(0, cx_px-r):cx_px+r+1]
                    good  = patch[patch > 0]
                    if good.size == 0:
                        cv2.putText(disp, "NO DEPTH", (cx_px, cy_px - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        continue

                    # Robust depth: pick the 10th percentile (closest values)
                    # This avoids the median picking up the background (13m) if the 
                    # tag itself is producing zeros (too close).
                    Z = float(np.percentile(good, 10)) / 1000.0   # mm → metres
                    X = (cx_px - cx0) * Z / fx
                    Y = (cy_px - cy0) * Z / fy

                    # Draw tag info
                    cv2.circle(disp, (cx_px, cy_px), 6, (0, 255, 255), -1)
                    tag_info = f"ID:{marker_id}  Z={Z*1000:.0f}mm  X={X*1000:.0f}mm  Y={Y*1000:.0f}mm"
                    cv2.putText(disp, tag_info, (cx_px - 80, cy_px - 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                    cv2.putText(disp, "SPACE=pick", (cx_px - 40, cy_px + 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1)

                    last_tag = (cx_px, cy_px, X, Y, Z)
                    
                    # Draw 3D Axes on Tag
                    # R=Identity for tag drawing (it's axis-aligned to cam for now)
                    draw_3d_axis(disp, np.eye(3), np.array([X, Y, Z]), M, length=0.04)
                    break   # use first matching tag

                # ── Live Target Base Calc ──
                if tcp_live is not None:
                    p_tag_base = cam_to_base(X, Y, Z, tcp_live, R_cam_tcp, t_cam_tcp)
                    bx, by, bz = p_tag_base[0]*1000, p_tag_base[1]*1000, p_tag_base[2]*1000
                    
                    # Draw on HUD
                    col = (0, 255, 0) # Green
                    if not LIMITS_PATH.exists(): pass
                    else:
                        if not (-1138 < by < -602): col = (0, 0, 255) # Red for Y violation

                    cv2.putText(disp, f"TAG BASE: X={bx:.0f} Y={by:.0f} Z={bz:.0f}mm",
                                (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

            status = ("READY — SPACE to pick" if last_tag
                      else "NO TAG VISIBLE")
            col = (0, 255, 0) if last_tag else (0, 80, 255)
            cv2.putText(disp, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, col, 2)

            cv2.imshow("ArUco Pick", disp)
            key_raw = cv2.waitKey(1)
            if key_raw == -1: continue # No key
            
            # DEBUG KEYPRESS
            print(f"[Key] Raw={key_raw}  Char='{chr(key_raw & 0xFF) if 0 < (key_raw & 0xFF) < 127 else '?'}'")
            
            key = key_raw & 0xFF

            if key == ord('q'):
                break
            elif key == ord('o'):
                _gripper_open()
            elif key == ord('c'):
                _gripper_close()
            # WASD: Pure Screen-Space Movement (10mm steps)
            elif key == ord('w'): TCP_IN_CAM_Y -= 0.010
            elif key == ord('s'): TCP_IN_CAM_Y += 0.010
            elif key == ord('a'): TCP_IN_CAM_X -= 0.010
            elif key == ord('d'): TCP_IN_CAM_X += 0.010
            # RF: Depth
            elif key == ord('r'): TCP_IN_CAM_Z += 0.010
            elif key == ord('f'): TCP_IN_CAM_Z -= 0.010
            # Flip Yaw
            elif key == ord('y'):
                YAW_DEG = 180.0 if YAW_DEG < 90 else 0.0
            # Pitch Angle (I/K)
            elif key == ord('i'): PITCH_DEG += 1.0
            elif key == ord('k'): PITCH_DEG -= 1.0
            # Yaw Angle (J/L)
            elif key == ord('l'): YAW_DEG += 1.0
            elif key == ord('j'): YAW_DEG -= 1.0
            # Roll Angle (U/P)
            elif key == ord('u'): ROLL_DEG += 1.0
            elif key == ord('p'): ROLL_DEG -= 1.0
            # Reset
            elif key == ord('x'): 
                TCP_IN_CAM_X=0.0; TCP_IN_CAM_Y=0.0; TCP_IN_CAM_Z=0.200
                PITCH_DEG=45.0; YAW_DEG=180.0; ROLL_DEG=0.0
                USE_MANUAL_CALIB=True
            elif key == ord('t'): USE_MANUAL_CALIB = not USE_MANUAL_CALIB
            elif key == ord(' '):
                if last_tag is None:
                    print("[Pick] No tag detected — move camera so tag is visible.")
                else:
                    tcp_now = get_tcp_pose()
                    if tcp_now is None:
                        print("[Pick] Cannot read robot TCP — is it connected?")
                    else:
                        cx, cy, X, Y, Z = last_tag
                        print(f"[Pick] Targeting tag at cam XYZ: "
                              f"X={X*1000:.1f}  Y={Y*1000:.1f}  Z={Z*1000:.1f} mm")
                        
                        # Clear queues and stop updates before moving to avoid X_LINK error
                        cv2.destroyAllWindows()
                        print("[Pick] Draining camera queues...")
                        while q_rgb.has(): q_rgb.get()
                        while q_depth.has(): q_depth.get()

                        pick_aruco((X, Y, Z), tcp_now, R_cam_tcp, t_cam_tcp,
                                   guard, SafetyViolation)
                        cv2.namedWindow("ArUco Pick")
            
            # Small sleep to reduce device/host load & heating
            time.sleep(0.01)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
