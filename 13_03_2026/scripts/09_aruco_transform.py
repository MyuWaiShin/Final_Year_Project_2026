"""
09_aruco_transform.py
----------------------
Detects ArUco tag (ID 3, DICT_6X6_250).
Shows tag position in:
  - Camera frame
  - TCP frame  (pendant TCP = gripper tip at ~186.6mm)
  - Base frame (corrected for live gripper width via AI2 → port 30002)

Press SPACE to print all transforms.
"""

import socket
import struct
import threading
import time
from pathlib import Path

import cv2
import cv2.aruco as aruco
import depthai as dai
import rtde_receive
import numpy as np

# ── Robust path setup ─────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR   = SCRIPT_DIR.parent

ROBOT_IP     = "192.168.8.102"
ARUCO_DICT   = aruco.DICT_6X6_250
ARUCO_TAG_ID = 3
MARKER_SIZE  = 0.021   # metres — measure your actual printed tag!

# ── Gripper TCP geometry (RG2) ────────────────────────────────────
# Pendant TCP offset from flange: Z = 186.6mm (the reference position)
# Fingertip Z from flange:  closed (0mm) = 199mm,  open (110mm) = 161mm
PENDANT_TCP_Z_MM  = 186.6   # Z offset defined on pendant (mm)
GRIPPER_Z_CLOSED  = 199.0   # fingertip Z at width=0mm  (fully closed)
GRIPPER_Z_OPEN    = 161.0   # fingertip Z at width=110mm (fully open)
GRIPPER_MAX_WIDTH = 110.0   # max width in mm


# ── Gripper Width Reader (port 30002, AI2 voltage) ─────────────────
class GripperWidthReader(threading.Thread):
    """
    Background thread: reads AI2 (gripper width voltage) from
    UR Secondary Client port 30002 — same method as grip_control script.
    No RTDE registers needed.
    """
    def __init__(self, ip, port=30002):
        super().__init__(daemon=True)
        self.ip          = ip
        self.port        = port
        self._voltage    = 0.0
        self._lock       = threading.Lock()
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(2.0)
                    s.connect((self.ip, self.port))
                    while not self._stop_event.is_set():
                        header = s.recv(4)
                        if not header or len(header) < 4:
                            break
                        pkt_len  = struct.unpack("!I", header)[0]
                        pkt_data = s.recv(pkt_len - 4)
                        offset = 1
                        while offset < len(pkt_data):
                            if offset + 4 > len(pkt_data):
                                break
                            p_size = struct.unpack("!I", pkt_data[offset:offset+4])[0]
                            p_type = pkt_data[offset+4]
                            if p_type == 2 and offset + 15 <= len(pkt_data):
                                # Tool Data sub-packet: AI2 is 8-byte double at offset+7
                                ai2 = struct.unpack("!d", pkt_data[offset+7:offset+15])[0]
                                with self._lock:
                                    self._voltage = max(ai2, 0.0)
                            if p_size == 0:
                                break
                            offset += p_size
            except Exception:
                time.sleep(0.5)   # retry on disconnection

    def get_width_mm(self):
        """Calibrated gripper width in mm (0=closed, ~110=open)."""
        with self._lock:
            voltage = self._voltage
        raw_mm = (voltage / 3.7) * 110.0
        slope  = (91.0 - 10.5) / (65.8 - 8.5)   # ≈ 1.405
        offset = 10.5 - (8.5 * slope)             # ≈ -1.44
        return max(0.0, round((raw_mm * slope) + offset, 1))

    def stop(self):
        self._stop_event.set()


def fingertip_z_mm(width_mm):
    """Actual fingertip distance from flange in mm, given gripper width."""
    return GRIPPER_Z_CLOSED + (GRIPPER_Z_OPEN - GRIPPER_Z_CLOSED) * (width_mm / GRIPPER_MAX_WIDTH)


def corrected_tcp_pos(tcp_pose, width_mm):
    """
    Returns the actual fingertip position in base frame by correcting the
    pendant TCP (which assumes a fixed gripper length) with the live width.

    tcp_pose : [x, y, z, rx, ry, rz] from getActualTCPPose()
    width_mm : live gripper width in mm
    """
    R, _ = cv2.Rodrigues(np.array(tcp_pose[3:]))
    tool_z_in_base  = R[:, 2]    # tool Z direction in base frame
    tip_z_mm        = fingertip_z_mm(width_mm)
    correction_m    = (tip_z_mm - PENDANT_TCP_Z_MM) / 1000.0   # +ve = further out
    pendant_pos     = np.array(tcp_pose[:3])
    return pendant_pos + correction_m * tool_z_in_base


# ── Load calibration ──────────────────────────────────────────────
K            = np.load(BASE_DIR / "calibration/camera_matrix.npy")
dist         = np.zeros((4, 1))
T_cam2tcp    = np.load(BASE_DIR / "calibration/T_cam2flange.npy")
print("  camera_matrix.npy  ✓")
print("  T_cam2flange.npy   ✓  (used as T_cam2tcp)\n")

dictionary = aruco.getPredefinedDictionary(ARUCO_DICT)
detector   = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())

# ── Connections ───────────────────────────────────────────────────
print("Connecting to robot...")
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
print("RTDE connected!")

print("Starting gripper width reader (port 30002)...")
gripper = GripperWidthReader(ROBOT_IP)
gripper.start()
time.sleep(0.8)   # let it get a first reading
print(f"Gripper width: {gripper.get_width_mm():.1f} mm\n")

# ── Camera Pipeline (DepthAI API 2.0) ────────────────────────────
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
print("SPACE = print transforms  |  Q = quit\n")


def tcp_to_matrix(tcp_pose):
    R, _ = cv2.Rodrigues(np.array(tcp_pose[3:]))
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = tcp_pose[:3]
    return T


while True:
    pkt   = videoQueue.get()
    frame = pkt.getCvFrame()
    grey  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = detector.detectMarkers(grey)
    tag_detected = False
    tag_pos_cam = tag_pos_tcp = tag_pos_base = tag_pos_base_corrected = None

    width_mm = gripper.get_width_mm()

    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)
        for i, mid in enumerate(ids.flatten()):
            if mid == ARUCO_TAG_ID:
                # Estimate pose of the tag
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners[i:i+1], MARKER_SIZE, K, dist)
                rvec, tvec = rvecs[0][0], tvecs[0][0]
                # Draw the 3‑D axes on the image
                cv2.drawFrameAxes(frame, K, dist, rvec, tvec, MARKER_SIZE * 0.5)
                # Project the tag centre (origin) onto the image to obtain pixel coordinates
                img_pts, _ = cv2.projectPoints(np.array([[0.0, 0.0, 0.0]], dtype=np.float32), rvec, tvec, K, dist)
                cx_img, cy_img = int(img_pts[0][0][0]), int(img_pts[0][0][1])
                # Overlay a small circle at the projected centre
                cv2.circle(frame, (cx_img, cy_img), 5, (0, 255, 0), -1)
                # Optional label with pixel coordinates
                cv2.putText(frame, f"Cam pix: ({cx_img},{cy_img})", (cx_img + 10, cy_img - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                tag_detected = True
                tag_pos_cam  = tvec

                R_tag, _ = cv2.Rodrigues(rvec)
                T_tag2cam = np.eye(4); T_tag2cam[:3, :3] = R_tag; T_tag2cam[:3, 3] = tvec

                tcp_pose       = rtde_r.getActualTCPPose()
                T_tcp2base     = tcp_to_matrix(tcp_pose)
                T_tag2tcp      = T_cam2tcp @ T_tag2cam
                tag_pos_tcp    = T_tag2tcp[:3, 3]
                T_tag2base     = T_tcp2base @ T_tag2tcp
                tag_pos_base   = T_tag2base[:3, 3]

                # Gripper-corrected fingertip position
                tip_pos_base = corrected_tcp_pos(tcp_pose, width_mm)

                fx, fy, fz = tag_pos_tcp
                bx, by, bz = tag_pos_base
                tx, ty, tz = tip_pos_base

                cv2.putText(frame, f"TCP frame: ({fx:.3f}, {fy:.3f}, {fz:.3f})m",
                            (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                cv2.putText(frame, f"Base frame: ({bx:.3f}, {by:.3f}, {bz:.3f})m",
                            (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"Gripper tip: ({tx:.3f}, {ty:.3f}, {tz:.3f})m  [width={width_mm:.1f}mm]",
                            (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)
                break

    label = f"TAG {ARUCO_TAG_ID} DETECTED" if tag_detected else f"Tag {ARUCO_TAG_ID} not detected"
    color = (0, 255, 0) if tag_detected else (0, 0, 255)
    cv2.putText(frame, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"Gripper: {width_mm:.1f}mm  |  SPACE=print  Q=quit",
                (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    cv2.imshow("ArUco Transform", cv2.resize(frame, (960, 540)))
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' ') and tag_detected:
        cx, cy, cz = tag_pos_cam
        fx, fy, fz = tag_pos_tcp
        bx, by, bz = tag_pos_base
        tx, ty, tz = corrected_tcp_pos(rtde_r.getActualTCPPose(), width_mm)
        print("\n" + "=" * 60)
        print(f"  Tag in CAMERA frame:        X={cx:.4f}  Y={cy:.4f}  Z={cz:.4f}")
        print(f"  Tag in TCP frame:           X={fx:.4f}  Y={fy:.4f}  Z={fz:.4f}")
        print(f"  Tag in BASE frame:          X={bx:.4f}  Y={by:.4f}  Z={bz:.4f}")
        print(f"  Gripper tip (corrected):    X={tx:.4f}  Y={ty:.4f}  Z={tz:.4f}")
        print(f"  Gripper width:              {width_mm:.1f} mm")
        print(f"  Fingertip Z from flange:    {fingertip_z_mm(width_mm):.1f} mm  (pendant: {PENDANT_TCP_Z_MM} mm)")
        print("=" * 60)

gripper.stop()
device.close()
rtde_r.disconnect()
cv2.destroyAllWindows()
print("\nDone.")