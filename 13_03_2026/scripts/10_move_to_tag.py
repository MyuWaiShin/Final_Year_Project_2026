"""
10_move_to_tag.py
------------------
Detects ArUco tag (ID 3), moves flange to tag XYZ in base frame.
Stops APPROACH_Z_OFFSET above the detected tag Z.
Keeps current orientation.
Press SPACE → confirm → move.
"""

import os
import socket
import struct
import threading
import time
from pathlib import Path
import cv2
import cv2.aruco as aruco
import depthai as dai
import rtde_receive
import rtde_control
import numpy as np

# ── Robust path setup ─────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR   = SCRIPT_DIR.parent

# ── Gripper geometry (RG2) ─────────────────────────────────────
PENDANT_TCP_Z_MM  = 186.6   # TCP offset defined on pendant (mm), at fingertip ~closed
GRIPPER_Z_CLOSED  = 199.0   # fingertip Z from flange at width=0mm
GRIPPER_Z_OPEN    = 161.0   # fingertip Z from flange at width=110mm
GRIPPER_MAX_WIDTH = 110.0

class GripperWidthReader(threading.Thread):
    """Reads AI2 voltage from port 30002 and converts to gripper width mm."""
    def __init__(self, ip, port=30002):
        super().__init__(daemon=True)
        self.ip = ip; self.port = port
        self._voltage = 0.0;  self._lock = threading.Lock()
        self._stop_event = threading.Event()
    def run(self):
        while not self._stop_event.is_set():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(2.0);  s.connect((self.ip, self.port))
                    while not self._stop_event.is_set():
                        header = s.recv(4)
                        if not header or len(header) < 4: break
                        pkt_len  = struct.unpack("!I", header)[0]
                        pkt_data = s.recv(pkt_len - 4)
                        offset = 1
                        while offset < len(pkt_data):
                            if offset + 4 > len(pkt_data): break
                            p_size = struct.unpack("!I", pkt_data[offset:offset+4])[0]
                            p_type = pkt_data[offset+4]
                            if p_type == 2 and offset + 15 <= len(pkt_data):
                                ai2 = struct.unpack("!d", pkt_data[offset+7:offset+15])[0]
                                with self._lock: self._voltage = max(ai2, 0.0)
                            if p_size == 0: break
                            offset += p_size
            except Exception: time.sleep(0.5)
    def get_width_mm(self):
        with self._lock: v = self._voltage
        raw_mm = (v / 3.7) * 110.0
        slope  = (91.0 - 10.5) / (65.8 - 8.5)
        offset = 10.5 - (8.5 * slope)
        return max(0.0, round((raw_mm * slope) + offset, 1))
    def stop(self): self._stop_event.set()

def corrected_tcp_pos(tcp_pose, width_mm):
    """Returns actual fingertip position in base frame, corrected for gripper width."""
    R, _ = cv2.Rodrigues(np.array(tcp_pose[3:]))
    tool_z = R[:, 2]
    tip_z_mm     = GRIPPER_Z_CLOSED + (GRIPPER_Z_OPEN - GRIPPER_Z_CLOSED) * (width_mm / GRIPPER_MAX_WIDTH)
    correction_m = (tip_z_mm - PENDANT_TCP_Z_MM) / 1000.0
    return np.array(tcp_pose[:3]) + correction_m * tool_z

# ── CONFIG ─────────────────────────────────────────────────────────
ROBOT_IP          = "192.168.8.102"
ARUCO_DICT        = aruco.DICT_6X6_250   # must match your printed tag
ARUCO_TAG_ID      = 3
MARKER_SIZE       = 0.021          # metres
APPROACH_Z_OFFSET = 0.00           # stop 5cm above the detected tag Z (in base frame)
MOVE_SPEED        = 0.04           # m/s
MOVE_ACCEL        = 0.01

# ── Manual correction for residual calibration error ──────────────
# If the robot consistently lands offset from the tag, tune these.
# Positive = robot needs to move further in that base-frame direction.
# Start at 0.0 and adjust in 5mm steps after testing.
MANUAL_X_OFFSET_M = 0.000   # metres — tweak if robot lands left/right of tag
MANUAL_Y_OFFSET_M = 0.000   # metres — tweak if robot lands in-front/behind tag

# ── Load calibration ───────────────────────────────────────────────
print("Loading calibration files...")
K            = np.load(BASE_DIR / "calibration/camera_matrix.npy")
dist         = np.zeros((4, 1))
T_cam2flange = np.load(BASE_DIR / "calibration/T_cam2flange.npy")
print("  camera_matrix.npy   ✓")
print("  T_cam2flange.npy    ✓\n")

# ── ArUco detector ─────────────────────────────────────────────────
dictionary      = aruco.getPredefinedDictionary(ARUCO_DICT)
detector_params = aruco.DetectorParameters()
detector        = aruco.ArucoDetector(dictionary, detector_params)

# ── Robot ──────────────────────────────────────────────────────────
print("Connecting to robot...")
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
print("Robot connected!")

print("Starting gripper width reader (port 30002)...")
gripper = GripperWidthReader(ROBOT_IP)
gripper.start()
time.sleep(0.8)
print(f"Gripper width: {gripper.get_width_mm():.1f} mm\n")

# ── Camera (DepthAI API 2.0) ──────────────────────────────────────
print("Starting camera...")
pipeline = dai.Pipeline()
cam      = pipeline.create(dai.node.ColorCamera)
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
print("CONTROLS:")
print("  SPACE  →  move flange to tag XYZ + 5cm above")
print("  Q      →  quit")
print("=" * 55)
print()

# ── Helper ─────────────────────────────────────────────────────────
def tcp_to_matrix(tcp_pose):
    R, _ = cv2.Rodrigues(np.array(tcp_pose[3:]))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = tcp_pose[:3]
    return T

# ── Main loop ──────────────────────────────────────────────────────
while True:
    imgFrame = videoQueue.get()
    frame    = imgFrame.getCvFrame()
    grey     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = detector.detectMarkers(grey)

    tag_detected = False
    tag_pos_base = None
    current_tcp  = None

    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)

        for i, marker_id in enumerate(ids.flatten()):
            if marker_id == ARUCO_TAG_ID:
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                    corners[i:i+1], MARKER_SIZE, K, dist
                )
                rvec = rvecs[0][0]
                tvec = tvecs[0][0]
                cv2.drawFrameAxes(frame, K, dist, rvec, tvec, MARKER_SIZE * 0.5)

                tag_detected = True

                R_tag, _ = cv2.Rodrigues(rvec)
                T_tag2cam = np.eye(4)
                T_tag2cam[:3, :3] = R_tag
                T_tag2cam[:3, 3]  = tvec

                current_tcp   = rtde_r.getActualTCPPose()
                T_tcp2base    = tcp_to_matrix(current_tcp)
                T_tag2tcp     = T_cam2flange @ T_tag2cam        # tag relative to TCP
                T_tag2base    = T_tcp2base   @ T_tag2tcp        # tag in robot world
                tag_pos_tcp   = T_tag2tcp[:3, 3]
                tag_pos_base  = T_tag2base[:3, 3]
                width_mm      = gripper.get_width_mm()
                tip_pos_base  = corrected_tcp_pos(current_tcp, width_mm)

                # ── In camera frame: raw distance from camera ──────────
                cx, cy, cz = tvec
                cv2.putText(frame, f"Cam frame  (dist from camera):     ({cx:.3f}, {cy:.3f}, {cz:.3f})m",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 255), 1)

                # ── In TCP/tool frame: tag relative to robot tool tip ──
                fx, fy, fz = tag_pos_tcp
                cv2.putText(frame, f"TCP frame  (tag from tool tip):     ({fx:.3f}, {fy:.3f}, {fz:.3f})m",
                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 165, 0), 1)

                # ── In base frame: tag absolute in robot world ─────────
                bx, by, bz = tag_pos_base
                cv2.putText(frame, f"Base frame (tag in robot world):    ({bx:.3f}, {by:.3f}, {bz:.3f})m",
                            (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 1)

                # ── Move target: where TCP will go ─────────────────────
                tz_target = bz + APPROACH_Z_OFFSET
                cv2.putText(frame, f"Move target (base, +{APPROACH_Z_OFFSET*1000:.0f}mm Z above tag): ({bx:.3f}, {by:.3f}, {tz_target:.3f})m",
                            (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 200), 1)

                # ── Gripper tip corrected position ─────────────────────
                tx, ty, tz = tip_pos_base
                cv2.putText(frame, f"Gripper tip (width={width_mm:.0f}mm, corrected):  ({tx:.3f}, {ty:.3f}, {tz:.3f})m",
                            (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 1)
                break

    if tag_detected:
        cv2.putText(frame, f"TAG {ARUCO_TAG_ID} DETECTED — press SPACE to move",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    else:
        cv2.putText(frame, f"Searching for tag ID {ARUCO_TAG_ID}...",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    cv2.putText(frame, "SPACE = move to tag  |  Q = quit",
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    cv2.imshow("Move to Tag", cv2.resize(frame, (960, 540)))
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord(' '):
        if not tag_detected or tag_pos_base is None:
            print("Tag not detected — point camera at tag first.")
            continue

        bx, by, bz = tag_pos_base
        fx, fy, fz = tag_pos_tcp
        rx, ry, rz = current_tcp[3], current_tcp[4], current_tcp[5]

        # Apply manual calibration correction
        tx = bx + MANUAL_X_OFFSET_M
        ty = by + MANUAL_Y_OFFSET_M
        tz = bz + APPROACH_Z_OFFSET
        target_pose = [tx, ty, tz, rx, ry, rz]

        # Distance from current TCP to target
        curr_xyz = np.array(current_tcp[:3])
        targ_xyz = np.array([tx, ty, tz])
        dist_mm  = np.linalg.norm(targ_xyz - curr_xyz) * 1000

        print("\n" + "=" * 60)
        print(f"  Tag in CAMERA frame (from camera):       X={tvec[0]:.4f}  Y={tvec[1]:.4f}  Z={tvec[2]:.4f}")
        print(f"  Tag in TCP frame   (from tool tip):      X={fx:.4f}  Y={fy:.4f}  Z={fz:.4f}")
        print(f"  Tag in BASE frame  (robot world coords): X={bx:.4f}  Y={by:.4f}  Z={bz:.4f}")
        print(f"  Manual offset applied:                   dX={MANUAL_X_OFFSET_M*1000:+.1f}mm  dY={MANUAL_Y_OFFSET_M*1000:+.1f}mm")
        print(f"  Move target        (base +{APPROACH_Z_OFFSET*1000:.0f}mm Z):       X={tx:.4f}  Y={ty:.4f}  Z={tz:.4f}")
        print(f"  Move distance:     {dist_mm:.1f} mm")
        if dist_mm < 10.0:
            print(f"  ⚠  Move distance < 10mm — robot is already at target. Move to a new pose first.")
        print(f"  Orientation kept:  RX={rx:.4f}  RY={ry:.4f}  RZ={rz:.4f}")
        print(f"  Speed: {MOVE_SPEED*100:.0f} cm/s")
        print("=" * 60)
        confirm = input("  Type YES to move, anything else to cancel: ").strip()

        if confirm.upper() != "YES":
            print("  Cancelled.\n")
            continue

        print("  Moving... hand on E-stop.")
        try:
            rtde_c.moveL(target_pose, MOVE_SPEED, MOVE_ACCEL)
            print("  Done! How far off is the gripper from above the tag?")
        except Exception as e:
            print(f"  Move failed: {e}")

cv2.destroyAllWindows()
device.close()
rtde_r.disconnect()
rtde_c.stopScript()
print("\nDone.")