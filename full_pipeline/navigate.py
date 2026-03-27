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
"""

import socket
import struct
import threading
import time
from pathlib import Path

import cv2
import cv2.aruco as aruco
import depthai as dai
import numpy as np
import rtde_control
import rtde_receive

# ── Paths ───────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
CALIB_DIR  = SCRIPT_DIR / "calibration"

# ── Robot ───────────────────────────────────────────────────────────────
ROBOT_IP = "192.168.8.102"

# ── ArUco ───────────────────────────────────────────────────────────────
ARUCO_DICT_TYPE = aruco.DICT_6X6_250
ARUCO_TAG_ID    = 3
MARKER_SIZE     = 0.021   # metres – match your printed tag

# ── Hover config ────────────────────────────────────────────────────────
# Fixed hover Z in base frame (metres). Absolute – ignores the tag's detected Z.
HOVER_Z_M = -0.050

# TCP orientation (Rodrigues vector). Tuned to point straight down at the
# desired wrist angle for the grasp approach.
HOVER_RX  = 2.225
HOVER_RY  = 2.170
HOVER_RZ  = 0.022

# Calibration offsets (empirically tuned to correct residual X/Y errors)
CALIB_X_OFFSET_M = -0.005
CALIB_Y_OFFSET_M = -0.050

# Move speed / acceleration
MOVE_SPEED = 0.04   # m/s
MOVE_ACCEL = 0.01   # m/s²


# ── Gripper Width Reader ────────────────────────────────────────────────
class GripperWidthReader(threading.Thread):
    def __init__(self, ip, port=30002):
        super().__init__(daemon=True)
        self.ip = ip; self.port = port
        self._voltage = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(2.0); s.connect((self.ip, self.port))
                    while not self._stop.is_set():
                        hdr = s.recv(4)
                        if not hdr or len(hdr) < 4: break
                        plen = struct.unpack("!I", hdr)[0]
                        data = s.recv(plen - 4)
                        off = 1
                        while off < len(data):
                            if off + 4 > len(data): break
                            ps = struct.unpack("!I", data[off:off+4])[0]
                            pt = data[off+4]
                            if pt == 2 and off + 15 <= len(data):
                                ai2 = struct.unpack("!d", data[off+7:off+15])[0]
                                with self._lock:
                                    self._voltage = max(ai2, 0.0)
                            if ps == 0: break
                            off += ps
            except Exception:
                time.sleep(0.5)

    def get_width_mm(self):
        with self._lock: v = self._voltage
        raw_mm = (v / 3.7) * 110.0
        slope  = (91.0 - 10.5) / (65.8 - 8.5)
        offset = 10.5 - (8.5 * slope)
        return max(0.0, round((raw_mm * slope) + offset, 1))

    def stop(self): self._stop.set()


# ── Helper ──────────────────────────────────────────────────────────────
def tcp_to_matrix(tcp_pose):
    R, _ = cv2.Rodrigues(np.array(tcp_pose[3:]))
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = tcp_pose[:3]
    return T


# ── Main ────────────────────────────────────────────────────────────────
def main():
    # Load calibration
    print("Loading calibration …")
    K            = np.load(CALIB_DIR / "camera_matrix.npy")
    dist         = np.zeros((4, 1))
    T_cam2flange = np.load(CALIB_DIR / "T_cam2flange.npy")
    print("  camera_matrix.npy  ✓")
    print("  T_cam2flange.npy   ✓\n")

    # ArUco detector
    dictionary = aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    detector   = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())

    # Robot
    print("Connecting to robot …")
    rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
    print("Robot connected!\n")

    gripper = GripperWidthReader(ROBOT_IP)
    gripper.start()
    time.sleep(0.8)

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
    print("  SPACE  →  hover TCP directly above the tag")
    print("  Q      →  quit")
    print("=" * 55 + "\n")

    tag_pos_base = None
    tag_detected = False

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
                    corners[i:i+1], MARKER_SIZE, K, dist
                )
                rvec, tvec = rvecs[0][0], tvecs[0][0]
                cv2.drawFrameAxes(frame, K, dist, rvec, tvec, MARKER_SIZE * 0.5)

                # Project tag centre for visual overlay
                img_pts, _ = cv2.projectPoints(
                    np.array([[0.0, 0.0, 0.0]], dtype=np.float32), rvec, tvec, K, dist
                )
                cx_img = int(img_pts[0][0][0]); cy_img = int(img_pts[0][0][1])
                cv2.circle(frame, (cx_img, cy_img), 6, (0, 255, 0), -1)
                cv2.putText(frame, f"pix ({cx_img},{cy_img})", (cx_img + 8, cy_img - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

                # Transform: camera → flange → base
                R_tag, _ = cv2.Rodrigues(rvec)
                T_tag2cam = np.eye(4); T_tag2cam[:3, :3] = R_tag; T_tag2cam[:3, 3] = tvec
                tcp_pose   = rtde_r.getActualTCPPose()
                T_tcp2base = tcp_to_matrix(tcp_pose)
                T_tag2tcp  = T_cam2flange @ T_tag2cam
                T_tag2base = T_tcp2base   @ T_tag2tcp

                tag_pos_base = T_tag2base[:3, 3].copy()
                tag_pos_base[0] += CALIB_X_OFFSET_M
                tag_pos_base[1] += CALIB_Y_OFFSET_M
                tag_detected = True

                bx, by, bz = tag_pos_base
                cx, cy, cz = tvec
                cv2.putText(frame, f"Cam:  ({cx:.3f}, {cy:.3f}, {cz:.3f}) m",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 255), 1)
                cv2.putText(frame, f"Base: ({bx:.3f}, {by:.3f}, {bz:.3f}) m",
                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 1)
                cv2.putText(frame, f"Hover target: ({bx:.3f}, {by:.3f}, {HOVER_Z_M:.3f}) m  [Z fixed]",
                            (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 200), 1)
                break

        label = f"TAG {ARUCO_TAG_ID} DETECTED — press SPACE to hover" if tag_detected \
                else f"Searching for tag ID {ARUCO_TAG_ID} …"
        color = (0, 255, 0) if tag_detected else (0, 0, 255)
        cv2.putText(frame, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.putText(frame, "SPACE = hover above tag  |  Q = quit",
                    (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)

        cv2.imshow("navigate", cv2.resize(frame, (960, 540)))
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord(' '):
            if not tag_detected or tag_pos_base is None:
                print("[WARN] Tag not detected – aim camera at tag first.")
                continue

            bx, by, _ = tag_pos_base
            target_pose = [bx, by, HOVER_Z_M, HOVER_RX, HOVER_RY, HOVER_RZ]
            current_tcp = rtde_r.getActualTCPPose()
            dist_mm = np.linalg.norm(np.array(target_pose[:3]) - np.array(current_tcp[:3])) * 1000

            print("\n" + "=" * 60)
            print(f"  Tag base frame:  X={bx:.4f}  Y={by:.4f}")
            print(f"  Hover Z (fixed): {HOVER_Z_M:.4f} m")
            print(f"  Target pose:     {[round(v, 4) for v in target_pose]}")
            print(f"  Distance:        {dist_mm:.1f} mm")
            print("=" * 60)
            confirm = input("  Type YES to move (hand on E-stop): ").strip()
            if confirm.upper() != "YES":
                print("  Cancelled.\n")
                continue

            print("  Moving …")
            try:
                rtde_c.moveL(target_pose, MOVE_SPEED, MOVE_ACCEL)
                print("  Hover reached.")
                break   # exit loop after successful navigation
            except Exception as e:
                print(f"  Move failed: {e}")

    # Cleanup
    gripper.stop()
    cv2.destroyAllWindows()
    device.close()
    rtde_r.disconnect()
    rtde_c.stopScript()
    print("\nDone.")
    return tag_pos_base


if __name__ == "__main__":
    main()
