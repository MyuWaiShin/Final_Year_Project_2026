"""
calibrate_handeye.py  — 3-Point Hand-Eye Calibration
======================================================
Solves for the FULL 6DOF camera-to-TCP transform:
  R_cam_in_tcp  (3x3 rotation — accounts for camera tilt/angle)
  t_cam_in_tcp  (3D translation — accounts for lateral offset)

WHY 3 POINTS?
  A single touch calibration can only find the translational offset.
  If the camera is angled or offset (not directly above TCP), a rotation
  term is also needed to correctly map camera coordinates to robot base.
  Three points lets us solve for both R and t via SVD (Procrustes method).

HOW TO USE:
  Place 3 calibration objects in a triangle spread across the mat.
  For each object:
    Step A — Move robot so the object is centred at the crosshair.
             Press '1' to record DETECT pose + auto-read depth.
    Step B — Jog the gripper tip to physically touch that same object.
             Press '2' to record TOUCH pose.
  Repeat for all 3 objects (3 pairs total).

  Press 'S' to solve and save  |  Press 'Q' to quit without saving.
"""

import os
os.environ["DEPTHAI_BOOT_TIMEOUT"] = "30000"

import json
import socket
import struct
import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np
import onnxruntime as ort

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
ROBOT_IP   = "192.168.8.102"
SAVE_PATH  = Path("ur10_cam_offset.json")
MODEL_PATH = Path(r"C:\Users\myuwa\OneDrive - Middlesex University\FYP Datasets\runs\train\yolov8n\weights\best.onnx")

CONF_THRESHOLD = 0.80
IOU_THRESHOLD  = 0.45
CLASSES = ["cube", "cylinder", "arc"]
COLORS  = {"cube": (0,255,0), "cylinder": (255,0,0), "arc": (0,0,255)}

N_POINTS = 3   # number of calibration pairs to record


# ─────────────────────────────────────────────────────────────────────────────
# Robot pose reader (port 30003)
# ─────────────────────────────────────────────────────────────────────────────
def get_tcp_pose(ip=ROBOT_IP):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            s.connect((ip, 30003))
            data = s.recv(1100)
            if len(data) < 444 + 48:
                return None
            return list(struct.unpack("!6d", data[444:492]))
    except Exception as e:
        print(f"[Robot] {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def rotvec_to_matrix(rx, ry, rz):
    """Convert rotation vector (Rodrigues) to 3x3 rotation matrix."""
    vec   = np.array([rx, ry, rz], dtype=float)
    angle = np.linalg.norm(vec)
    if angle < 1e-9:
        return np.eye(3)
    axis = vec / angle
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def depth_at(px, py, depth_frame, r=10):
    patch = depth_frame[max(0, py-r):py+r+1, max(0, px-r):px+r+1]
    good  = patch[patch > 0]
    return float(np.median(good)) / 1000.0 if good.size > 0 else None


def pixel_to_cam(px, py, Z, fx, fy, cx0, cy0):
    return (px - cx0) * Z / fx, (py - cy0) * Z / fy, Z


# ─────────────────────────────────────────────────────────────────────────────
# 3-point Procrustes solve for R and t
# ─────────────────────────────────────────────────────────────────────────────
def solve_cam_to_tcp(pairs):
    """
    Given N calibration pairs, solve for R_cam_in_tcp and t_cam_in_tcp
    using the Procrustes (SVD) method.

    For each pair: object is observed at P_cam in camera frame,
    and the gripper touch gives the object's position in the TCP frame
    (via the detect vs touch TCP poses).

    Returns R (3x3), t (3,) both expressed in TCP frame.
    """
    P_cam_list = []
    P_tcp_list = []

    for p in pairs:
        R_det       = rotvec_to_matrix(*p['pose_detect'][3:])
        P_tcp_det   = np.array(p['pose_detect'][:3])
        P_tcp_touch = np.array(p['pose_touch'][:3])

        # Object position in TCP frame at detection time:
        # R_det.T rotates from base frame to TCP frame
        P_obj_in_tcp = R_det.T @ (P_tcp_touch - P_tcp_det)

        P_cam_list.append(np.array(p['cam_xyz']))
        P_tcp_list.append(P_obj_in_tcp)

    A = np.array(P_cam_list)   # (N,3) camera observations
    B = np.array(P_tcp_list)   # (N,3) TCP-frame object positions

    # Procrustes: find R, t  such that  B ≈ R @ A + t
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean

    H = A_c.T @ B_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Correct reflection (det must be +1 for a proper rotation)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = b_mean - R @ a_mean

    # Report reprojection errors
    print("\n[Solve] Reprojection errors:")
    errs = []
    for i, p in enumerate(pairs):
        pred = R @ np.array(p['cam_xyz']) + t
        err  = np.linalg.norm(pred - B[i]) * 1000
        errs.append(err)
        print(f"  Point {i+1}: {err:.1f} mm")
    print(f"  Mean: {np.mean(errs):.1f} mm")

    return R, t


# ─────────────────────────────────────────────────────────────────────────────
# YOLO detector
# ─────────────────────────────────────────────────────────────────────────────
class YOLODetector:
    def __init__(self, model_path, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD):
        self.conf = conf
        self.iou  = iou
        self.sess = ort.InferenceSession(str(model_path))
        inp       = self.sess.get_inputs()[0]
        self.name = inp.name
        self.h    = inp.shape[2]
        self.w    = inp.shape[3]

    def detect(self, frame):
        img = cv2.resize(frame, (self.w, self.h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x   = np.expand_dims(np.transpose(img, (2,0,1)), 0)
        out = self.sess.run(None, {self.name: x})
        return self._post(out, frame.shape)

    def _post(self, outputs, shape):
        preds = np.transpose(outputs[0], (0,2,1))[0]
        oh, ow = shape[:2]
        dets = []
        for p in preds:
            xc, yc, bw, bh = p[:4]
            scores = p[4:]
            cid    = int(np.argmax(scores))
            conf   = float(scores[cid])
            if conf < self.conf:
                continue
            x1 = max(0,    int((xc - bw/2) * ow / self.w))
            y1 = max(0,    int((yc - bh/2) * oh / self.h))
            x2 = min(ow-1, int((xc + bw/2) * ow / self.w))
            y2 = min(oh-1, int((yc + bh/2) * oh / self.h))
            dets.append({'bbox': (x1, y1, x2, y2), 'class_id': cid, 'conf': conf})
        return sorted(dets, key=lambda d: d['conf'], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Camera pipeline
# ─────────────────────────────────────────────────────────────────────────────
def create_pipeline():
    p   = dai.Pipeline()
    p.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_4)
    cam = p.create(dai.node.ColorCamera)
    cam.setPreviewSize(640, 480)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(30)
    ml = p.create(dai.node.MonoCamera)
    mr = p.create(dai.node.MonoCamera)
    st = p.create(dai.node.StereoDepth)
    ml.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    ml.setBoardSocket(dai.CameraBoardSocket.LEFT)
    mr.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mr.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    st.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    st.setDepthAlign(dai.CameraBoardSocket.RGB)
    ml.out.link(st.left)
    mr.out.link(st.right)
    xr = p.create(dai.node.XLinkOut); xr.setStreamName("rgb")
    xd = p.create(dai.node.XLinkOut); xd.setStreamName("depth")
    cam.preview.link(xr.input)
    st.depth.link(xd.input)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Draw overlay
# ─────────────────────────────────────────────────────────────────────────────
def draw_overlay(frame, depth_frame, dets, fx, fy, cx0, cy0,
                 pairs, current_point, waiting_for, tcp_now):
    disp   = frame.copy()
    h, w   = disp.shape[:2]

    # ── Crosshair — centred in the usable area above the gripper ─────────────
    # Gripper covers ~40% of bottom → usable top 60% → centre = 30% from top
    img_cx = w // 2
    img_cy = int(h * 0.30)      # ← moved up from h//2

    cv2.drawMarker(disp, (img_cx, img_cy), (0, 255, 255), cv2.MARKER_CROSS, 40, 2)
    cv2.circle(disp,    (img_cx, img_cy), 20, (0, 255, 255), 1)

    # ── TCP position (top-left) ───────────────────────────────────────────────
    if tcp_now:
        x, y, z = tcp_now[0]*1000, tcp_now[1]*1000, tcp_now[2]*1000
        cv2.putText(disp, f"TCP  X:{x:.1f}  Y:{y:.1f}  Z:{z:.1f} mm",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,200,255), 1)
    else:
        cv2.putText(disp, "TCP: not connected",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,0,220), 1)

    # ── Progress ──────────────────────────────────────────────────────────────
    n_done = len(pairs)
    cv2.putText(disp, f"Points recorded: {n_done}/{N_POINTS}",
                (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200,200,200), 1)

    if n_done < N_POINTS:
        pt_lbl = f"Point {current_point}/{N_POINTS}"
        if waiting_for == 'detect':
            msg = f"[{pt_lbl}]  Centre object at crosshair, then press '1'"
            col = (0, 200, 255)
        else:
            msg = f"[{pt_lbl}]  DETECT recorded. Jog gripper to touch, then press '2'"
            col = (0, 165, 255)
        cv2.putText(disp, msg, (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1)
    else:
        cv2.putText(disp, "All 3 points done!  Press 'S' to solve & save",
                    (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,255,0), 2)

    # ── Already-recorded pairs (bottom) ──────────────────────────────────────
    for i, pair in enumerate(pairs):
        cx, cy, cz = pair['cam_xyz']
        txt = f"P{i+1} cam=({cx:+.3f},{cy:+.3f},{cz:.3f}m)"
        cv2.putText(disp, txt,
                    (10, h - 14 - (n_done - i - 1) * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0,255,120), 1)

    # ── Live detections ───────────────────────────────────────────────────────
    for det in dets:
        x1, y1, x2, y2 = det['bbox']
        label = CLASSES[det['class_id']] if det['class_id'] < len(CLASSES) else "?"
        color = COLORS.get(label, (200,200,200))
        cv2.rectangle(disp, (x1,y1), (x2,y2), color, 2)

        px = (x1+x2)//2
        py = (y1+y2)//2
        cv2.drawMarker(disp, (px,py), (0,255,255), cv2.MARKER_CROSS, 12, 1)

        err_px = int(np.sqrt((px-img_cx)**2 + (py-img_cy)**2))
        err_col = (0,255,0) if err_px < 15 else (0,165,255) if err_px < 50 else (0,0,255)
        label_err = 'OK' if err_px < 15 else 'centre robot'
        cv2.putText(disp, f"err:{err_px}px  ({label_err})",
                    (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, err_col, 1)

        Z = depth_at(px, py, depth_frame)
        if Z:
            X, Y, _ = pixel_to_cam(px, py, Z, fx, fy, cx0, cy0)
            cv2.putText(disp, f"X:{X:+.3f} Y:{Y:+.3f} Z:{Z:.3f}m",
                        (x1, y2+14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,255,255), 1)
        else:
            cv2.putText(disp, "depth N/A",
                        (x1, y2+14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,200,200), 1)

    return disp


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not MODEL_PATH.exists():
        print(f"ERROR: model not found at {MODEL_PATH}")
        return

    detector = YOLODetector(MODEL_PATH)
    pipeline = create_pipeline()

    pairs          = []     # completed {pose_detect, pose_touch, cam_xyz}
    pending_detect = None   # detect recorded, waiting for touch
    waiting_for    = 'detect'

    cfg = dai.Device.Config()
    cfg.board.usb.maxSpeed = dai.UsbSpeed.HIGH

    with dai.Device(cfg) as device:
        device.startPipeline(pipeline)
        cal = device.readCalibration()
        M, _, _ = cal.getDefaultIntrinsics(dai.CameraBoardSocket.RGB)
        fx, fy, cx0, cy0 = M[0][0], M[1][1], M[0][2], M[1][2]
        print(f"[Camera] fx={fx:.1f}  fy={fy:.1f}  cx={cx0:.1f}  cy={cy0:.1f}")

        q_rgb   = device.getOutputQueue("rgb",   maxSize=4, blocking=False)
        q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)

        print("\n=== 3-Point Hand-Eye Calibration ===")
        print(f"  Place {N_POINTS} objects spread in a triangle across the mat.")
        print("  For each object:")
        print("    1 — Centre it at the crosshair → press '1' (DETECT)")
        print("    2 — Jog gripper to touch it    → press '2' (TOUCH)")
        print("  S — Solve & save  |  Q — Quit\n")

        while True:
            frame       = q_rgb.get().getCvFrame()
            depth_frame = q_depth.get().getFrame()
            dets        = detector.detect(frame)
            tcp_now     = get_tcp_pose()
            current_pt  = len(pairs) + 1

            disp = draw_overlay(frame, depth_frame, dets,
                                fx, fy, cx0, cy0,
                                pairs, current_pt, waiting_for, tcp_now)
            cv2.imshow("Hand-Eye Calibration (3-Point)", disp)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Quit — nothing saved.")
                break

            elif key == ord('1') and len(pairs) < N_POINTS and waiting_for == 'detect':
                tcp = get_tcp_pose()
                if tcp is None:
                    print("[Calib] Cannot read robot pose"); continue
                if not dets:
                    print("[Calib] No object detected — aim crosshair at object first"); continue
                x1, y1, x2, y2 = dets[0]['bbox']
                px, py = (x1+x2)//2, (y1+y2)//2
                Z = depth_at(px, py, depth_frame)
                if not Z:
                    print("[Calib] No depth reading — move object closer"); continue
                X, Y, _ = pixel_to_cam(px, py, Z, fx, fy, cx0, cy0)
                pending_detect = {'pose_detect': tcp, 'cam_xyz': (X, Y, Z)}
                waiting_for = 'touch'
                print(f"[P{current_pt}] DETECT recorded  TCP=({tcp[0]*1000:.1f},{tcp[1]*1000:.1f},{tcp[2]*1000:.1f})mm  "
                      f"cam=({X:+.3f},{Y:+.3f},{Z:.3f}m)")

            elif key == ord('2') and len(pairs) < N_POINTS and waiting_for == 'touch':
                if pending_detect is None:
                    print("[Calib] Press '1' first"); continue
                tcp = get_tcp_pose()
                if tcp is None:
                    print("[Calib] Cannot read robot pose"); continue
                pair = {**pending_detect, 'pose_touch': tcp}
                pairs.append(pair)
                pending_detect = None
                waiting_for    = 'detect'
                print(f"[P{len(pairs)}] TOUCH  recorded  TCP=({tcp[0]*1000:.1f},{tcp[1]*1000:.1f},{tcp[2]*1000:.1f})mm  "
                      f"  [{len(pairs)}/{N_POINTS} done]")

            elif key == ord('s'):
                if len(pairs) < N_POINTS:
                    print(f"[Calib] Need {N_POINTS} pairs — only {len(pairs)} recorded"); continue

                print(f"\nSolving for camera-to-TCP transform from {N_POINTS} points ...")
                R, t = solve_cam_to_tcp(pairs)

                data = {
                    "R_cam_in_tcp": R.tolist(),
                    "t_cam_in_tcp": t.tolist(),
                    "calibration_points": [
                        {
                            "pose_detect": p['pose_detect'],
                            "pose_touch":  p['pose_touch'],
                            "cam_xyz":     list(p['cam_xyz']),
                        }
                        for p in pairs
                    ],
                    "camera_intrinsics": {"fx": fx, "fy": fy, "cx": cx0, "cy": cy0},
                    "timestamp": time.ctime()
                }
                with open(SAVE_PATH, 'w') as f:
                    json.dump(data, f, indent=4)

                print(f"\n Calibration saved to {SAVE_PATH}")
                print(f"  R_cam_in_tcp:\n{np.round(R, 4)}")
                print(f"  t_cam_in_tcp:  dx={t[0]:.4f}m  dy={t[1]:.4f}m  dz={t[2]:.4f}m")
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
