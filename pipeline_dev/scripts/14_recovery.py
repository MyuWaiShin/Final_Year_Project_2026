"""
14_recovery.py
==============
Recovery script — triggered when 13_failure_detection.py returns "rescan".

Steps:
  1. Move to saved "scan_pos" (a position where the camera sees the workspace)
  2. Open camera, run YOLO + OBB + depth for N frames to detect the object
  3. Convert best detection to robot base frame via eye-in-hand calibration
  4. Move 200 mm above the object, gripper pointing STRAIGHT DOWN,
     rotated to match the object's OBB angle (Rz)
  5. Descend vertically to the object and close gripper

Prerequisites:
  data/saved_positions.json  must contain "scan_pos" (and "pick_pos" for orientation)
  calibration/T_cam2flange.npy   — 4x4 camera-to-flange transform
  calibration/camera_matrix.npy  — 3x3 camera intrinsic matrix K
  data/best.onnx                 — YOLO model

Run:
  python 14_recovery.py
"""

import os
os.environ["DEPTHAI_BOOT_TIMEOUT"] = "30000"

import json
import socket
import time
import numpy as np
import cv2
import depthai as dai
import onnxruntime as ort
from pathlib import Path

import rtde_control
import rtde_receive

# ===========================================================================
#  CONFIG
# ===========================================================================
ROBOT_IP   = "192.168.8.102"
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR   = SCRIPT_DIR.parent

POSITIONS_FILE   = BASE_DIR / "data/saved_positions.json"
MODEL_PATH       = BASE_DIR / "data/best.onnx"
CALIB_DIR        = BASE_DIR / "calibration"
T_CAM2FLANGE_NPY = CALIB_DIR / "T_cam2flange.npy"
CAM_MATRIX_NPY   = CALIB_DIR / "camera_matrix.npy"

# Detection
CONF_THRESHOLD = 0.70
IOU_THRESHOLD  = 0.45
CLASSES        = ["cube", "cylinder"]
COLORS         = {"cube": (0, 255, 0), "cylinder": (255, 0, 0)}

# How many frames to accumulate detections before picking the best
N_DETECT_FRAMES = 30

# Gripper URP paths (Dashboard)
GRIP_CLOSE_URP = "/programs/myu/close_gripper.urp"
GRIP_OPEN_URP  = "/programs/myu/open_gripper.urp"

# Motion
APPROACH_ABOVE_M = 0.200   # metres above object for approach
VEL      = 0.25            # m/s
ACC      = 0.15            # m/s²
VEL_SLOW = 0.08            # m/s  — descend to object
ACC_SLOW = 0.08

# Dashboard port (for gripper URPs)
DASHBOARD_PORT = 29999

# ===========================================================================
#  YOLO Detector  (inline from pose_estimation.py)
# ===========================================================================
class YOLODetector:
    def __init__(self, model_path, conf=0.70, iou=0.45):
        self.conf = conf
        self.iou  = iou
        self.sess = ort.InferenceSession(str(model_path))
        inp       = self.sess.get_inputs()[0]
        self.name = inp.name
        self.h    = inp.shape[2]
        self.w    = inp.shape[3]
        print(f"[YOLO] Loaded model {model_path.name}  input={self.w}x{self.h}")

    def detect(self, frame):
        img  = cv2.resize(frame, (self.w, self.h))
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp  = np.expand_dims(np.transpose(img, (2, 0, 1)), 0)
        outs = self.sess.run(None, {self.name: inp})
        return self._post(outs, frame.shape)

    def _post(self, outputs, shape):
        preds = np.transpose(outputs[0], (0, 2, 1))[0]
        oh, ow = shape[:2]
        dets = []
        for p in preds:
            xc, yc, bw, bh = p[:4]
            scores = p[4:]
            cid  = int(np.argmax(scores))
            conf = float(scores[cid])
            if conf < self.conf:
                continue
            x1 = max(0,    int((xc - bw/2) * ow / self.w))
            y1 = max(0,    int((yc - bh/2) * oh / self.h))
            x2 = min(ow-1, int((xc + bw/2) * ow / self.w))
            y2 = min(oh-1, int((yc + bh/2) * oh / self.h))
            if x2 > x1 and y2 > y1:
                dets.append({"bbox": (x1, y1, x2, y2),
                              "conf": conf, "class_id": cid})
        return self._nms(dets)

    def _nms(self, dets):
        dets = sorted(dets, key=lambda d: d["conf"], reverse=True)
        keep = []
        while dets:
            best = dets.pop(0)
            keep.append(best)
            dets = [d for d in dets
                    if self._iou(best["bbox"], d["bbox"]) < self.iou]
        return keep

    def _iou(self, a, b):
        xi1, yi1 = max(a[0], b[0]), max(a[1], b[1])
        xi2, yi2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, xi2-xi1) * max(0, yi2-yi1)
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / ua if ua else 0.0


# ===========================================================================
#  OBB Analyzer  (inline from pose_estimation.py)
# ===========================================================================
class OrientedBoxAnalyzer:
    def __init__(self, padding=6, hue_window=20):
        self.padding    = padding
        self.hue_window = hue_window

    def _dominant_hue_mask(self, hsv):
        h, s, v = cv2.split(hsv)
        coloured = s > 80
        if coloured.sum() < 50:
            return np.zeros(h.shape, dtype=np.uint8)
        hues     = h[coloured].flatten()
        hist, _  = np.histogram(hues, bins=180, range=(0, 180))
        smooth   = np.convolve(hist, np.ones(7)/7, mode="same")
        dom      = int(np.argmax(smooth))
        lo = (dom - self.hue_window) % 180
        hi = (dom + self.hue_window) % 180
        thr = np.array([80, 50])
        if lo <= hi:
            mask = cv2.inRange(hsv, np.array([lo, 80, 50]),
                               np.array([hi, 255, 255]))
        else:
            m1 = cv2.inRange(hsv, np.array([lo, 80, 50]),
                             np.array([179, 255, 255]))
            m2 = cv2.inRange(hsv, np.array([0,  80, 50]),
                             np.array([hi, 255, 255]))
            mask = cv2.bitwise_or(m1, m2)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask

    def analyze(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        hf, wf = frame.shape[:2]
        cx1 = max(0, x1 - self.padding); cx2 = min(wf-1, x2 + self.padding)
        cy1 = max(0, y1 - self.padding); cy2 = min(hf-1, y2 + self.padding)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return {"valid": False}
        area = max((cx2-cx1)*(cy2-cy1), 1)
        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = self._dominant_hue_mask(hsv)
        if cv2.countNonZero(mask) < 0.05 * area:
            gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 0), 30, 100)
            k     = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
            mask  = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return {"valid": False}
        largest = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(largest) < 0.04 * area:
            return {"valid": False}
        rect = cv2.minAreaRect(largest)
        (rcx, rcy), (rw, rh), angle = rect
        if rw < rh:
            angle += 90
            rw, rh = rh, rw
        gripper_angle = angle % 180
        box_pts = cv2.boxPoints(rect).astype(np.intp) + np.array([cx1, cy1])
        return {
            "valid":         True,
            "center":        (int(rcx + cx1), int(rcy + cy1)),
            "gripper_angle": gripper_angle,
            "box_points":    box_pts,
        }


# ===========================================================================
#  Depth → camera-frame 3-D pose
# ===========================================================================
def depth_to_cam_pose(pixel_cx, pixel_cy, depth_frame, K, gripper_angle):
    """Return {'X','Y','Z','Rz'} in camera frame, or None if no depth."""
    fx, fy = K[0, 0], K[1, 1]
    cx0, cy0 = K[0, 2], K[1, 2]
    r  = 5
    hd, wd = depth_frame.shape
    px1 = max(0, pixel_cx - r); px2 = min(wd, pixel_cx + r + 1)
    py1 = max(0, pixel_cy - r); py2 = min(hd, pixel_cy + r + 1)
    patch = depth_frame[py1:py2, px1:px2]
    good  = patch[patch > 0]
    if good.size == 0:
        return None
    Z = float(np.median(good)) / 1000.0   # mm → m
    X = (pixel_cx - cx0) * Z / fx
    Y = (pixel_cy - cy0) * Z / fy
    return {"X": round(X, 4), "Y": round(Y, 4),
            "Z": round(Z, 4), "Rz": round(gripper_angle, 1)}


# ===========================================================================
#  Camera → base frame transform (eye-in-hand)
# ===========================================================================
def cam_to_base(pose_cam, T_cam2flange, tcp_pose):
    """
    Full 6-DOF eye-in-hand transform: camera → flange → base.
    pose_cam: {'X','Y','Z'} in camera frame (metres)
    T_cam2flange: 4x4 homogeneous transform (camera in flange frame)
    tcp_pose: [x,y,z,rx,ry,rz] current TCP in base frame
    """
    P_cam = np.array([pose_cam["X"], pose_cam["Y"], pose_cam["Z"], 1.0])

    # Camera → flange frame
    P_flange = T_cam2flange @ P_cam

    # Flange → base frame: use TCP rotation vector
    rv = np.array(tcp_pose[3:])
    R_flange2base, _ = cv2.Rodrigues(rv)
    T_flange2base = np.eye(4)
    T_flange2base[:3, :3] = R_flange2base
    T_flange2base[:3, 3]  = tcp_pose[:3]

    P_base = T_flange2base @ P_flange
    return P_base[:3]


# ===========================================================================
#  Robot helpers
# ===========================================================================
def dashboard_cmd(ip, cmd, retries=3):
    for attempt in range(1, retries + 1):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5.0)
            s.connect((ip, DASHBOARD_PORT))
            s.recv(1024)
            s.sendall((cmd + "\n").encode())
            resp = s.recv(1024).decode().strip()
            s.close()
            return resp
        except Exception as e:
            print(f"  [Dashboard] {cmd}: {e}  (attempt {attempt}/{retries})")
            time.sleep(0.3)
    return ""


def stop_and_wait(ip, timeout=4.0):
    dashboard_cmd(ip, "stop")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if "false" in dashboard_cmd(ip, "running", retries=1).lower():
            break
        time.sleep(0.15)


def play_urp(ip, path):
    stop_and_wait(ip)
    dashboard_cmd(ip, f"load {path}")
    time.sleep(0.2)
    dashboard_cmd(ip, "play")


def open_gripper(ip):
    print("  [Gripper] Opening...")
    play_urp(ip, GRIP_OPEN_URP)
    time.sleep(2.5)


def close_gripper(ip):
    print("  [Gripper] Closing...")
    play_urp(ip, GRIP_CLOSE_URP)
    time.sleep(2.5)


# ===========================================================================
#  OBB angle → robot rz  (straight-down approach)
# ===========================================================================
def obb_angle_to_rz(obb_deg, base_rx, base_ry, base_rz):
    """
    Convert an image-plane OBB angle (0-180°) to a robot rz rotation vector.

    Strategy:
      - Keep base_rx and base_ry (the "straight down" tilt) unchanged.
      - Replace rz with the OBB angle converted to radians.
        This sets the gripper jaw direction to match the object orientation.

    Args:
        obb_deg   — OBB gripper_angle from OrientedBoxAnalyzer (degrees, 0-180)
        base_rx/ry/rz — orientation from pick_pos (straight-down baseline)
    Returns:
        (rx, ry, rz) tuple in radians
    """
    rz_new = np.deg2rad(obb_deg)
    return float(base_rx), float(base_ry), float(rz_new)


# ===========================================================================
#  DAI camera pipeline  (Color + StereoDepth, 640×480 preview)
# ===========================================================================
def create_dai_pipeline():
    p = dai.Pipeline()
    cam = p.create(dai.node.ColorCamera)
    cam.setPreviewSize(640, 480)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(30)

    mono_l = p.create(dai.node.MonoCamera)
    mono_r = p.create(dai.node.MonoCamera)
    stereo  = p.create(dai.node.StereoDepth)
    mono_l.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_l.setBoardSocket(dai.CameraBoardSocket.LEFT)
    mono_r.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_r.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
    mono_l.out.link(stereo.left)
    mono_r.out.link(stereo.right)

    xrgb = p.create(dai.node.XLinkOut); xrgb.setStreamName("rgb")
    xdep = p.create(dai.node.XLinkOut); xdep.setStreamName("depth")
    cam.preview.link(xrgb.input)
    stereo.depth.link(xdep.input)
    return p


# ===========================================================================
#  Detection scan  (N frames, returns best pose in camera frame)
# ===========================================================================
def scan_for_object(device, q_rgb, q_depth, detector, obb_analyzer, K,
                    n_frames=N_DETECT_FRAMES):
    """
    Accumulate detections over n_frames, return the candidate with the
    highest cumulative confidence that has a valid depth reading.
    Also shows a live camera window during scanning.
    """
    candidates = []
    win = "14 — Recovery Scan"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 800, 480)

    fps_count = 0
    fps_time  = time.time()
    fps       = 0.0
    frame_n   = 0

    print(f"[Scan] Collecting detections over {n_frames} frames...")
    while frame_n < n_frames:
        frame       = q_rgb.get().getCvFrame()
        depth_frame = q_depth.get().getFrame()
        display     = frame.copy()
        frame_n    += 1

        dets = detector.detect(frame)
        for det in dets:
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox
            cls   = CLASSES[det["class_id"]] if det["class_id"] < len(CLASSES) else "?"
            color = COLORS.get(cls, (200, 200, 200))

            obb = obb_analyzer.analyze(frame.copy(), bbox)
            if not obb["valid"]:
                continue

            angle = obb["gripper_angle"]
            cx, cy = obb["center"]

            pose_cam = depth_to_cam_pose(cx, cy, depth_frame, K, angle)
            if pose_cam is None:
                continue

            candidates.append({
                "conf":     det["conf"],
                "class_id": det["class_id"],
                "pose_cam": pose_cam,
                "bbox":     bbox,
                "obb":      obb,
            })

            # Draw OBB + pose on display
            cv2.drawContours(display, [obb["box_points"]], 0, color, 2)
            rad = np.deg2rad(angle)
            ax  = int(cx + 40 * np.cos(rad))
            ay  = int(cy + 40 * np.sin(rad))
            cv2.arrowedLine(display, (cx, cy), (ax, ay), (0, 255, 255), 2,
                            tipLength=0.3)
            lbl = (f"{cls} {det['conf']*100:.0f}%  "
                   f"X:{pose_cam['X']:+.3f} Y:{pose_cam['Y']:+.3f} "
                   f"Z:{pose_cam['Z']:.3f}m  Rz:{angle:.0f}deg")
            cv2.putText(display, lbl, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

        # FPS
        fps_count += 1
        if fps_count >= 10:
            fps       = fps_count / (time.time() - fps_time)
            fps_count = 0
            fps_time  = time.time()

        cv2.putText(display,
                    f"Scanning... {frame_n}/{n_frames}  FPS:{fps:.1f}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.imshow(win, display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyWindow(win)

    if not candidates:
        return None

    # Pick best: highest confidence with valid depth
    best = max(candidates, key=lambda c: c["conf"])
    print(f"[Scan] Best detection: {CLASSES[best['class_id']]} "
          f"conf={best['conf']*100:.0f}%  "
          f"X={best['pose_cam']['X']:+.4f}  "
          f"Y={best['pose_cam']['Y']:+.4f}  "
          f"Z={best['pose_cam']['Z']:.4f}  "
          f"Rz={best['pose_cam']['Rz']:.1f}deg")
    return best


# ===========================================================================
#  MAIN
# ===========================================================================
def load_positions():
    if not POSITIONS_FILE.exists():
        raise FileNotFoundError(f"Positions file not found: {POSITIONS_FILE}")
    data = json.loads(POSITIONS_FILE.read_text(encoding="utf-8"))
    return data


def main():
    # ── Checks ──────────────────────────────────────────────────────────────
    missing = []
    for p in (POSITIONS_FILE, MODEL_PATH, T_CAM2FLANGE_NPY, CAM_MATRIX_NPY):
        if not p.exists():
            missing.append(str(p))
    if missing:
        print("ERROR: Missing files:")
        for m in missing:
            print(f"  {m}")
        return

    # ── Load calibration ────────────────────────────────────────────────────
    T_cam2flange = np.load(str(T_CAM2FLANGE_NPY))   # 4x4
    K            = np.load(str(CAM_MATRIX_NPY))      # 3x3
    print(f"[Calib] T_cam2flange loaded  shape={T_cam2flange.shape}")
    print(f"[Calib] K loaded  fx={K[0,0]:.1f}  fy={K[1,1]:.1f}")

    # ── Load positions ───────────────────────────────────────────────────────
    positions = load_positions()

    if "scan_pos" not in positions:
        print("\nERROR: 'scan_pos' not found in saved_positions.json")
        print("  → Move the robot to a scan position where the camera")
        print("    has a clear view of the workspace, then save it as")
        print("    'scan_pos' using 06_tcp_test.py and re-run.\n")
        # Fallback: try using pick_pos as the scan position
        if "pick_pos" in positions:
            ans = input("Use 'pick_pos' as scan position? [Y/N]: ").strip().upper()
            if ans != "Y":
                return
            scan_pos = positions["pick_pos"]
            print("[Recovery] Using pick_pos as scan_pos fallback.")
        else:
            return
    else:
        scan_pos = positions["scan_pos"]

    if "pick_pos" not in positions:
        print("ERROR: 'pick_pos' not found — needed for gripper orientation baseline.")
        return

    pick_pos = positions["pick_pos"]
    # Straight-down orientation baseline from pick_pos
    base_rx = pick_pos["rx"]
    base_ry = pick_pos["ry"]
    base_rz = pick_pos["rz"]

    sp = scan_pos
    print(f"\n[Config] Scan pos:  X={sp['x']*1000:.1f}  Y={sp['y']*1000:.1f}  Z={sp['z']*1000:.1f} mm")
    print(f"[Config] Baseline orientation (straight down):  "
          f"rx={base_rx:.3f}  ry={base_ry:.3f}  rz={base_rz:.3f}")
    print(f"[Config] Approach above object: {APPROACH_ABOVE_M*1000:.0f} mm\n")

    # ── Connect RTDE ─────────────────────────────────────────────────────────
    print(f"[RTDE] Connecting to {ROBOT_IP}...")
    rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
    print("[RTDE] Connected.\n")

    # ── YOLO + OBB ──────────────────────────────────────────────────────────
    detector     = YOLODetector(MODEL_PATH, CONF_THRESHOLD, IOU_THRESHOLD)
    obb_analyzer = OrientedBoxAnalyzer()

    # ── Open camera ──────────────────────────────────────────────────────────
    print("[Camera] Opening OAK-D...")
    pipeline = create_dai_pipeline()
    config   = dai.Device.Config()
    config.board.usb.maxSpeed = dai.UsbSpeed.HIGH

    with dai.Device(config) as device:
        device.startPipeline(pipeline)
        print(f"[Camera] Connected: {device.getMxId()}")

        # Read intrinsics from device
        cal  = device.readCalibration()
        M_list, _, _ = cal.getDefaultIntrinsics(dai.CameraBoardSocket.RGB)
        K_device = np.array(M_list, dtype=np.float64).reshape(3, 3)
        print(f"[Camera] Intrinsics from device: "
              f"fx={K_device[0,0]:.1f}  cx={K_device[0,2]:.1f}")

        q_rgb   = device.getOutputQueue("rgb",   maxSize=4, blocking=False)
        q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)

        # ── Step 1: Open gripper ─────────────────────────────────────────────
        print("\n[1] Opening gripper...")
        open_gripper(ROBOT_IP)

        # ── Step 2: Move to scan position ────────────────────────────────────
        print(f"\n[2] Moving to scan position...")
        rtde_c.moveL(
            [sp["x"], sp["y"], sp["z"], sp["rx"], sp["ry"], sp["rz"]],
            VEL, ACC
        )
        print(f"  Arrived at scan pos.")
        time.sleep(0.5)

        # ── Step 3: Scan for object ───────────────────────────────────────────
        tcp_at_scan = rtde_r.getActualTCPPose()
        print(f"\n[3] Scanning for object ({N_DETECT_FRAMES} frames)...")
        best = scan_for_object(device, q_rgb, q_depth,
                               detector, obb_analyzer, K_device)

        if best is None:
            print("\n[Recovery] No object detected — cannot recover.")
            print("  Check that the object is visible from the scan position.")
            rtde_c.stopScript()
            rtde_r.disconnect()
            return

        # ── Step 4: Convert pose to base frame ───────────────────────────────
        print(f"\n[4] Converting pose to base frame...")
        pose_cam = best["pose_cam"]
        P_base   = cam_to_base(pose_cam, T_cam2flange, tcp_at_scan)
        obj_x, obj_y, obj_z = P_base[0], P_base[1], P_base[2]

        print(f"  Object in base frame:")
        print(f"    X={obj_x*1000:.1f}  Y={obj_y*1000:.1f}  Z={obj_z*1000:.1f} mm")
        print(f"  OBB angle: {pose_cam['Rz']:.1f} deg")

        # ── Step 5: Compute gripper orientation ───────────────────────────────
        rx_t, ry_t, rz_t = obb_angle_to_rz(pose_cam["Rz"], base_rx, base_ry, base_rz)
        approach_z = obj_z + APPROACH_ABOVE_M

        print(f"\n  Approach pose:")
        print(f"    X={obj_x*1000:.1f}  Y={obj_y*1000:.1f}  Z={approach_z*1000:.1f} mm")
        print(f"    rx={rx_t:.3f}  ry={ry_t:.3f}  rz={rz_t:.3f}  "
              f"(OBB={pose_cam['Rz']:.1f}deg → rz={np.rad2deg(rz_t):.1f}deg)")

        confirm = input("\n  Type YES to execute pick, anything else to cancel: ").strip()
        if confirm.upper() != "YES":
            print("  Cancelled.")
            rtde_c.stopScript()
            rtde_r.disconnect()
            return

        # ── Step 6: Approach — 200mm above, straight down, OBB-aligned ───────
        print(f"\n[5] Moving to approach ({APPROACH_ABOVE_M*1000:.0f} mm above)...")
        rtde_c.moveJ_IK(
            [obj_x, obj_y, approach_z, rx_t, ry_t, rz_t],
            VEL, ACC
        )
        print("  Arrived at approach.")

        # ── Step 7: Descend to object ─────────────────────────────────────────
        print(f"[6] Descending to object  Z={obj_z*1000:.1f} mm...")
        rtde_c.moveL(
            [obj_x, obj_y, obj_z, rx_t, ry_t, rz_t],
            VEL_SLOW, ACC_SLOW
        )
        print("  Arrived at object.")

        # ── Step 8: Close gripper ─────────────────────────────────────────────
        print("[7] Closing gripper...")
        close_gripper(ROBOT_IP)

        # ── Step 9: Lift to approach height ──────────────────────────────────
        print(f"[8] Lifting to {approach_z*1000:.1f} mm...")
        rtde_c.moveL(
            [obj_x, obj_y, approach_z, rx_t, ry_t, rz_t],
            VEL, ACC
        )

        print("\n" + "=" * 56)
        print("  ✓  Recovery pick complete — hand back to script 13.")
        print("=" * 56 + "\n")

    rtde_c.stopScript()
    rtde_r.disconnect()
    print("[Recovery] Done.")


if __name__ == "__main__":
    main()
