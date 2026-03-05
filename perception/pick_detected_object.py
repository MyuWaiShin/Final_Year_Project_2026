"""
pick_detected_object.py
========================
Detects an object using the OAK-D camera + YOLO, converts the camera-frame
pose (X, Y, Z, Rz) to the robot base frame using the hand-eye calibration,
then commands the UR10 to pick it up.

Prerequisites:
  - ur10_cam_offset.json  (run calibrate_ur10_handseye.py first)
  - grip_close.urp / grip_open.urp on the robot pendant

Run:
    python pick_detected_object.py

Press 'space' to trigger a pick on the best detection.
Press  'q'    to quit.
"""

import os
os.environ["DEPTHAI_BOOT_TIMEOUT"] = "30000"

import json
import socket
import struct
import time
import threading
from pathlib import Path

import cv2
import depthai as dai
import numpy as np
import onnxruntime as ort

from ur10_utils import UR10PoseReader, pose_to_matrix, rotvec_to_matrix

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
ROBOT_IP       = "192.168.8.102"
CALIB_PATH     = Path("ur10_cam_offset.json")
MODEL_PATH     = Path(r"C:\Users\myuwa\OneDrive - Middlesex University\FYP Datasets\runs\train\yolov8n\weights\best.onnx")

CONF_THRESHOLD = 0.80
IOU_THRESHOLD  = 0.45
CLASSES        = ["cube", "cylinder", "arc"]
COLORS         = {"cube": (0,255,0), "cylinder": (255,0,0), "arc": (0,0,255)}

APPROACH_HEIGHT = 0.10   # metres above object before descending
MOVE_VEL        = 0.10   # m/s — slow for picking
MOVE_ACC        = 0.10   # m/s²
GRIPPER_WAIT    = 3.0    # seconds for gripper to finish


# ─────────────────────────────────────────────────────────────────────────────
# Load calibration & compute camera-to-TCP offset
# ─────────────────────────────────────────────────────────────────────────────
def load_calibration(path: Path):
    """
    Loads the camera-to-TCP transform produced by calibrate_handeye.py.

    New format (3-point calibration):
        R_cam_in_tcp  — 3×3 rotation of camera frame in TCP frame
        t_cam_in_tcp  — translation of camera origin in TCP frame (metres)

    Both are used together: P_obj_tcp = R @ P_obj_cam + t
    """
    with open(path) as f:
        cal = json.load(f)

    t_cam_in_tcp = np.array(cal["t_cam_in_tcp"])

    if "R_cam_in_tcp" in cal:
        R_cam_in_tcp = np.array(cal["R_cam_in_tcp"])
        print(f"[Calib] Loaded full 6DOF camera-to-TCP transform")
    else:
        # Backward compat: old single-point calibration had no rotation
        R_cam_in_tcp = np.eye(3)
        print(f"[Calib] Old format — no rotation (R=I). Re-run calibration for best accuracy.")

    print(f"[Calib] t_cam_in_tcp:  dx={t_cam_in_tcp[0]:.4f}m  dy={t_cam_in_tcp[1]:.4f}m  dz={t_cam_in_tcp[2]:.4f}m")
    return R_cam_in_tcp, t_cam_in_tcp


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate transform: camera frame → robot base frame
# ─────────────────────────────────────────────────────────────────────────────
def cam_to_base(X_cam, Y_cam, Z_cam, tcp_pose, R_cam_in_tcp, t_cam_in_tcp):
    """
    Converts a camera-frame observation to the robot base frame.

    Full 6DOF eye-in-hand transform:
        1. Rotate camera observation into TCP frame:  P_tcp = R_cam @ P_cam + t_cam
        2. Rotate + translate from TCP to base frame: P_base = R_tcp @ P_tcp + P_tcp_base

    Args:
        X_cam, Y_cam, Z_cam   – object position in camera frame (metres)
        tcp_pose               – current TCP pose [x,y,z,rx,ry,rz]
        R_cam_in_tcp           – 3×3 rotation of camera in TCP frame
        t_cam_in_tcp           – camera origin in TCP frame (metres)
    """
    P_tcp_base = np.array(tcp_pose[:3])
    R_tcp      = rotvec_to_matrix(*tcp_pose[3:])
    P_obj_cam  = np.array([X_cam, Y_cam, Z_cam])

    # Step 1: camera frame → TCP frame
    P_obj_tcp  = R_cam_in_tcp @ P_obj_cam + t_cam_in_tcp

    # Step 2: TCP frame → base frame
    P_obj_base = R_tcp @ P_obj_tcp + P_tcp_base
    return P_obj_base


# ─────────────────────────────────────────────────────────────────────────────
# Robot motion via URScript (movel)
# ─────────────────────────────────────────────────────────────────────────────
class UR10Mover:
    """Sends movel() URScript commands to move the robot."""

    SCRIPT_PORT    = 30001
    DASHBOARD_PORT = 29999

    def __init__(self, ip):
        self.ip      = ip
        self.pose_rd = UR10PoseReader(ip)

    def get_tcp_pose(self):
        return self.pose_rd.get_actual_tcp_pose()

    def _send_urscript(self, script: str):
        """Send URScript to port 30001 and return immediately."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3.0)
            sock.connect((self.ip, self.SCRIPT_PORT))
            sock.sendall((script + "\n").encode())
            sock.close()
        except Exception as e:
            print(f"[URScript] Error: {e}")

    def movel(self, x, y, z, rx, ry, rz, vel=MOVE_VEL, acc=MOVE_ACC):
        """Move to Cartesian pose (blocking — waits for arrival)."""
        script = (
            f"def move_to():\n"
            f"  movel(p[{x:.6f},{y:.6f},{z:.6f},{rx:.6f},{ry:.6f},{rz:.6f}], "
            f"a={acc}, v={vel})\n"
            f"end\n"
        )
        self._send_urscript(script)

        # Wait until robot arrives (TCP position stable within 5mm)
        target = np.array([x, y, z])
        timeout = time.time() + 20.0
        time.sleep(0.5)   # let it start moving
        while time.time() < timeout:
            pose = self.get_tcp_pose()
            if pose:
                dist = np.linalg.norm(np.array(pose[:3]) - target)
                if dist < 0.005:
                    return True
            time.sleep(0.1)
        print("[movel] Warning: timed out waiting for arrival")
        return False

    def _dashboard_cmd(self, cmd):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(3.0)
            s.connect((self.ip, self.DASHBOARD_PORT))
            s.recv(1024)
            s.sendall((cmd + "\n").encode())
            r = s.recv(1024).decode().strip()
            s.close()
            return r
        except Exception as e:
            print(f"[Dashboard] {e}")
            return ""

    def _stop_and_wait(self):
        self._dashboard_cmd("stop")
        deadline = time.time() + 4.0
        while time.time() < deadline:
            if "false" in self._dashboard_cmd("running").lower():
                break
            time.sleep(0.15)
        time.sleep(1.5)  # buffer drain

    def grip_close(self):
        self._stop_and_wait()
        self._dashboard_cmd("load grip_close.urp")
        time.sleep(0.2)
        self._dashboard_cmd("play")
        time.sleep(GRIPPER_WAIT)

    def grip_open(self):
        self._stop_and_wait()
        self._dashboard_cmd("load grip_open.urp")
        time.sleep(0.2)
        self._dashboard_cmd("play")
        time.sleep(GRIPPER_WAIT)


# ─────────────────────────────────────────────────────────────────────────────
# Pick routine
# ─────────────────────────────────────────────────────────────────────────────
def pick_object(mover: UR10Mover, pose_cam: dict,
                R_cam_in_tcp: np.ndarray, t_cam_in_tcp: np.ndarray):
    """
    Full pick sequence for one detected object.

    Args:
        pose_cam      – {'X','Y','Z','Rz'} in camera frame
        R_cam_in_tcp  – camera rotation in TCP frame (3×3)
        t_cam_in_tcp  – camera translation in TCP frame (3,)
    """
    tcp_now = mover.get_tcp_pose()
    if tcp_now is None:
        print("[Pick] Cannot read TCP pose — aborting.")
        return False

    # Convert to base frame using full 6DOF transform
    P_obj = cam_to_base(pose_cam['X'], pose_cam['Y'], pose_cam['Z'],
                        tcp_now, R_cam_in_tcp, t_cam_in_tcp)

    # Keep current rotation for approach (maintain gripper orientation)
    rx, ry, rz = tcp_now[3], tcp_now[4], tcp_now[5]

    print(f"\n[Pick] Object in base frame: "
          f"X={P_obj[0]:.4f}m  Y={P_obj[1]:.4f}m  Z={P_obj[2]:.4f}m")
    print(f"[Pick] Approach height: {APPROACH_HEIGHT}m above object")

    # 1. Open gripper
    print("[Pick] Opening gripper...")
    mover.grip_open()

    # 2. Move to approach position (above object)
    print("[Pick] Moving to approach position...")
    mover.movel(P_obj[0], P_obj[1], P_obj[2] + APPROACH_HEIGHT, rx, ry, rz)

    # 3. Descend to object
    print("[Pick] Descending to object...")
    mover.movel(P_obj[0], P_obj[1], P_obj[2], rx, ry, rz, vel=0.05, acc=0.05)

    # 4. Close gripper
    print("[Pick] Closing gripper...")
    mover.grip_close()

    # 5. Lift
    print("[Pick] Lifting...")
    mover.movel(P_obj[0], P_obj[1], P_obj[2] + APPROACH_HEIGHT, rx, ry, rz)

    print("[Pick] Done!")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# YOLO detector (same as pose_estimation.py)
# ─────────────────────────────────────────────────────────────────────────────
class YOLODetector:
    def __init__(self, model_path, conf=0.8, iou=0.45):
        self.conf = conf
        self.iou  = iou
        self.sess = ort.InferenceSession(str(model_path))
        self.inp  = self.sess.get_inputs()[0]
        self.h    = self.inp.shape[2]
        self.w    = self.inp.shape[3]

    def detect(self, frame):
        img = cv2.resize(frame, (self.w, self.h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = np.expand_dims(np.transpose(img, (2,0,1)), 0)
        out = self.sess.run(None, {self.inp.name: inp})
        return self._post(out, frame.shape)

    def _post(self, outputs, shape):
        preds = np.transpose(outputs[0], (0,2,1))[0]
        oh, ow = shape[:2]
        dets = []
        for p in preds:
            xc,yc,w,h = p[:4]
            scores = p[4:]
            cid = int(np.argmax(scores))
            conf = float(scores[cid])
            if conf < self.conf: continue
            x1 = int((xc-w/2)*ow/self.w); y1 = int((yc-h/2)*oh/self.h)
            x2 = int((xc+w/2)*ow/self.w); y2 = int((yc+h/2)*oh/self.h)
            dets.append({'bbox':(max(0,x1),max(0,y1),min(ow-1,x2),min(oh-1,y2)),
                         'class_id':cid, 'conf':conf})
        return sorted(dets, key=lambda d: d['conf'], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Pose from depth (same as pose_estimation.py)
# ─────────────────────────────────────────────────────────────────────────────
def get_pose_from_depth(bbox, depth_frame, fx, fy, cx0, cy0):
    x1,y1,x2,y2 = bbox
    px = (x1+x2)//2; py = (y1+y2)//2
    r = 5
    patch = depth_frame[max(0,py-r):py+r+1, max(0,px-r):px+r+1]
    good = patch[patch > 0]
    if good.size == 0:
        return None
    Z = float(np.median(good)) / 1000.0
    X = (px - cx0) * Z / fx
    Y = (py - cy0) * Z / fy
    return {'X': round(X,4), 'Y': round(Y,4), 'Z': round(Z,4)}


# ─────────────────────────────────────────────────────────────────────────────
# Camera pipeline
# ─────────────────────────────────────────────────────────────────────────────
def create_pipeline():
    p = dai.Pipeline()
    p.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_4)
    cam = p.create(dai.node.ColorCamera)
    cam.setPreviewSize(640, 480); cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR); cam.setFps(30)
    ml = p.create(dai.node.MonoCamera); mr = p.create(dai.node.MonoCamera)
    st = p.create(dai.node.StereoDepth)
    ml.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    ml.setBoardSocket(dai.CameraBoardSocket.LEFT)
    mr.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mr.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    st.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    st.setDepthAlign(dai.CameraBoardSocket.RGB)
    ml.out.link(st.left); mr.out.link(st.right)
    xrgb = p.create(dai.node.XLinkOut); xrgb.setStreamName("rgb")
    xdep = p.create(dai.node.XLinkOut); xdep.setStreamName("depth")
    cam.preview.link(xrgb.input); st.depth.link(xdep.input)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not CALIB_PATH.exists():
        print(f"ERROR: {CALIB_PATH} not found — run calibrate_ur10_handseye.py first.")
        return
    if not MODEL_PATH.exists():
        print(f"ERROR: model not found at {MODEL_PATH}")
        return

    R_cam_in_tcp, t_cam_in_tcp = load_calibration(CALIB_PATH)
    detector     = YOLODetector(MODEL_PATH, CONF_THRESHOLD, IOU_THRESHOLD)
    mover        = UR10Mover(ROBOT_IP)
    pipeline     = create_pipeline()

    cfg = dai.Device.Config()
    cfg.board.usb.maxSpeed = dai.UsbSpeed.HIGH

    with dai.Device(cfg) as device:
        device.startPipeline(pipeline)

        # Load intrinsics
        cal  = device.readCalibration()
        M, _, _ = cal.getDefaultIntrinsics(dai.CameraBoardSocket.RGB)
        fx, fy, cx0, cy0 = M[0][0], M[1][1], M[0][2], M[1][2]
        print(f"[Camera] fx={fx:.1f} fy={fy:.1f} cx={cx0:.1f} cy={cy0:.1f}")

        q_rgb   = device.getOutputQueue("rgb",   maxSize=4, blocking=False)
        q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)

        print("\n=== Detect & Pick ===")
        print("  SPACE  →  pick the highest-confidence detection")
        print("  Q      →  quit\n")

        last_detections = []

        while True:
            frame       = q_rgb.get().getCvFrame()
            depth_frame = q_depth.get().getFrame()
            display     = frame.copy()

            dets = detector.detect(frame)

            # Draw and compute pose for each detection
            last_detections = []
            for det in dets:
                x1,y1,x2,y2 = det['bbox']
                label = CLASSES[det['class_id']] if det['class_id'] < len(CLASSES) else "?"
                color = COLORS.get(label, (200,200,200))

                pose = get_pose_from_depth(det['bbox'], depth_frame, fx, fy, cx0, cy0)
                det['pose'] = pose

                cv2.rectangle(display, (x1,y1), (x2,y2), color, 2)
                lbl = f"{label} {int(det['conf']*100)}%"
                cv2.putText(display, lbl, (x1, y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                if pose:
                    info = f"X:{pose['X']:+.3f} Y:{pose['Y']:+.3f} Z:{pose['Z']:.3f}m"
                    cv2.putText(display, info, (x1, y2+16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,255,255), 1)
                    last_detections.append(det)

            cv2.putText(display, f"Objects: {len(dets)} | SPACE=pick  Q=quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.imshow("Detect & Pick", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                if not last_detections:
                    print("[Pick] No detections with valid depth — try again.")
                else:
                    best = last_detections[0]   # highest confidence
                    label = CLASSES[best['class_id']] if best['class_id'] < len(CLASSES) else "?"
                    print(f"\n[Pick] Targeting: {label} | pose={best['pose']}")
                    cv2.destroyAllWindows()
                    pick_object(mover, best['pose'], R_cam_in_tcp, t_cam_in_tcp)
                    print("\nReturning to camera view...")
                    cv2.namedWindow("Detect & Pick")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
