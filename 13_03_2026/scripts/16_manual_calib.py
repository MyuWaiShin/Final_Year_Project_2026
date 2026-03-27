"""
16_manual_calib.py
------------------
Same as 15_hover.py but builds T_cam2flange from hand-measured values
instead of loading T_cam2flange.npy.

Camera mount geometry (edit these constants to match your hardware):
  CAM_TX_M  =  0.075  m  (camera offset +X from flange centre)
  CAM_TY_M  =  0.000  m  (camera offset  Y from flange centre)
  CAM_TZ_M  =  0.170  m  (camera offset +Z from flange, along tool axis)
  CAM_ROT_Y_DEG = 21.0   (camera tilted 21° about Y axis wrt the flange)

Controls
--------
  SPACE  →  move to hover pose above the tag
  Q      →  quit
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
BASE_DIR   = SCRIPT_DIR.parent

# ── Robot / tag config ──────────────────────────────────────────────────
ROBOT_IP     = "192.168.8.102"
ARUCO_DICT   = aruco.DICT_6X6_250
ARUCO_TAG_ID = 3
MARKER_SIZE  = 0.021        # metres – match your printed tag

# ── Manual camera-to-TCP geometry ──────────────────────────────────────
# NOTE: getActualTCPPose() returns the TCP (gripper fingertip), NOT the raw flange.
# The camera is mounted BEHIND the tool tip (closer to the robot body),
# so Z is NEGATIVE relative to the TCP.
CAM_TX_M       =  0.075    # camera X offset from TCP (metres) – positive = to the right
CAM_TY_M       =  0.000    # camera Y offset from TCP (metres)
CAM_TZ_M       = -0.170    # camera Z offset from TCP (metres) – NEGATIVE because camera
                            # is behind the tool tip along the tool axis
CAM_ROT_Y_DEG  = 21.0      # tilt of camera about Y axis relative to the TCP frame (degrees)

def build_T_cam2flange(tx, ty, tz, rot_y_deg):
    """
    Build the 4×4 homogeneous transform T_cam2flange from hand-measured values.
    Rotation is about the Y axis only (toe-in/toe-out tilt of the camera).

      R_y(θ) = [[ cos θ,  0,  sin θ],
                [   0,    1,    0  ],
                [-sin θ,  0,  cos θ]]
    """
    theta = np.radians(rot_y_deg)
    R = np.array([
        [ np.cos(theta), 0, np.sin(theta)],
        [       0,       1,       0      ],
        [-np.sin(theta), 0, np.cos(theta)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = [tx, ty, tz]
    return T

T_cam2flange = build_T_cam2flange(CAM_TX_M, CAM_TY_M, CAM_TZ_M, CAM_ROT_Y_DEG)
print("T_cam2flange (manually constructed):")
np.set_printoptions(precision=5, suppress=True)
print(T_cam2flange)
print()

# ── Hover / move config ─────────────────────────────────────────────────
HOVER_Z_M        = -0.050   # fixed absolute Z in base frame (metres)
HOVER_RX = 2.225
HOVER_RY = 2.170
HOVER_RZ = 0.022
CALIB_X_OFFSET_M = -0.000   # known X calibration correction (metres)
CALIB_Y_OFFSET_M = -0.000   # known Y calibration correction (metres)
MOVE_SPEED       =  0.04    # m/s
MOVE_ACCEL       =  0.01    # m/s²

# ── Gripper Width Reader ────────────────────────────────────────────────
class GripperWidthReader(threading.Thread):
    """Background thread: reads AI2 (gripper width voltage) from port 30002."""
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

# ── Camera intrinsics (factory calibrated via DepthAI) ──────────────────
print("Loading camera intrinsics …")
K    = np.load(BASE_DIR / "calibration/camera_matrix.npy")
dist = np.zeros((4, 1))   # OAK-D Lite distortion is negligible at 1280×720
print("  camera_matrix.npy  ✓\n")

# ── ArUco detector ──────────────────────────────────────────────────────
dictionary = aruco.getPredefinedDictionary(ARUCO_DICT)
detector   = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())

# ── Robot connections ───────────────────────────────────────────────────
print("Connecting to robot …")
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
print("Robot connected!\n")

gripper = GripperWidthReader(ROBOT_IP)
gripper.start()
time.sleep(0.8)

# ── Camera pipeline ─────────────────────────────────────────────────────
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
print(f"  Cam offset: X={CAM_TX_M*1000:.0f}mm  Y={CAM_TY_M*1000:.0f}mm  Z={CAM_TZ_M*1000:.0f}mm")
print(f"  Cam tilt:   {CAM_ROT_Y_DEG}° about Y")
print("  SPACE  →  hover TCP directly above the tag")
print("  Q      →  quit")
print("=" * 55 + "\n")


# ── Helper ──────────────────────────────────────────────────────────────
def tcp_to_matrix(tcp_pose):
    R, _ = cv2.Rodrigues(np.array(tcp_pose[3:]))
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = tcp_pose[:3]
    return T


# ── Main loop ───────────────────────────────────────────────────────────
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

            # Project tag origin onto image
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
            cv2.putText(frame, f"Cam frame:  ({cx:.3f}, {cy:.3f}, {cz:.3f}) m",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 255), 1)
            cv2.putText(frame, f"Base frame: ({bx:.3f}, {by:.3f}, {bz:.3f}) m",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 1)
            cv2.putText(frame, f"Hover target: ({bx:.3f}, {by:.3f}, {HOVER_Z_M:.3f}) m  [Z fixed]",
                        (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 200), 1)
            break

    if tag_detected:
        cv2.putText(frame, f"TAG {ARUCO_TAG_ID} DETECTED — press SPACE to hover",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    else:
        cv2.putText(frame, f"Searching for tag ID {ARUCO_TAG_ID} …",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    cv2.putText(frame, "SPACE = hover above tag  |  Q = quit",
                (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)

    cv2.imshow("16_manual_calib", cv2.resize(frame, (960, 540)))
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord(' '):
        if not tag_detected or tag_pos_base is None:
            print("[WARN] Tag not detected – point the camera at the tag first.")
            continue

        bx, by, _ = tag_pos_base
        target_pose = [bx, by, HOVER_Z_M, HOVER_RX, HOVER_RY, HOVER_RZ]

        current_tcp = rtde_r.getActualTCPPose()
        dist_mm = np.linalg.norm(np.array(target_pose[:3]) - np.array(current_tcp[:3])) * 1000

        print("\n" + "=" * 60)
        print(f"  Tag in base frame:  X={bx:.4f}  Y={by:.4f}")
        print(f"  Corrections:        dX={CALIB_X_OFFSET_M*1000:+.0f}mm  dY={CALIB_Y_OFFSET_M*1000:+.0f}mm")
        print(f"  Hover Z (fixed):    {HOVER_Z_M:.4f} m")
        print(f"  Target pose:        {[round(v, 4) for v in target_pose]}")
        print(f"  Distance:           {dist_mm:.1f} mm")
        print("=" * 60)
        confirm = input("  Type YES to move, anything else to cancel: ").strip()

        if confirm.upper() != "YES":
            print("  Cancelled.\n")
            continue

        print("  Moving … hand on E-stop.")
        try:
            rtde_c.moveL(target_pose, MOVE_SPEED, MOVE_ACCEL)
            print("  Hover reached.")
        except Exception as e:
            print(f"  Move failed: {e}")

# ── Cleanup ─────────────────────────────────────────────────────────────
gripper.stop()
cv2.destroyAllWindows()
device.close()
rtde_r.disconnect()
rtde_c.stopScript()
print("\nDone.")
