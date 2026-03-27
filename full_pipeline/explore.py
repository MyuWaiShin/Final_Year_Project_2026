"""
explore.py
----------
Stage 1 of the full pipeline.

Behaviour
---------
1.  Moves the robot to a predefined scan pose (SCAN_JOINT_POS).
2.  Sweeps the base joint (J0) across a configurable arc while the
    camera looks for ArUco tag ID 3 (DICT_6X6_250).
3.  Stops the sweep and holds position as soon as the tag is detected.
4.  Returns the last detected tag position in the robot base frame so
    the next stage (navigate.py) can use it.

Run standalone
--------------
    python explore.py

The script waits for you to confirm (press ENTER) before moving.
"""

import json
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
DATA_DIR   = SCRIPT_DIR / "data"
SCAN_POSE_FILE = DATA_DIR / "scan_pose.json"

# ── Robot ───────────────────────────────────────────────────────────────
ROBOT_IP = "192.168.8.102"

# ── Scan pose ───────────────────────────────────────────────────────────
# Loaded from data/scan_pose.json (written by temp/capture_scan_pose.py).
# ⚠  Run capture_scan_pose.py first if this file does not exist yet.
def load_scan_pose():
    if not SCAN_POSE_FILE.exists():
        raise FileNotFoundError(
            f"Scan pose file not found: {SCAN_POSE_FILE}\n"
            "Run  python temp/capture_scan_pose.py  to record the scan pose first."
        )
    with open(SCAN_POSE_FILE) as f:
        data = json.load(f)
    return data["joint_angles"]

# ── Sweep config ────────────────────────────────────────────────────────
# The base joint (J0) sweeps from SWEEP_START_RAD to SWEEP_END_RAD.
# Positive angle = counter-clockwise when viewed from above.
SWEEP_START_RAD = -0.5   # start of sweep  (radians from scan pose J0)
SWEEP_END_RAD   =  0.5   # end of sweep
SWEEP_SPEED     =  0.2   # joint speed during sweep (rad/s) – slow and safe
SWEEP_ACCEL     =  0.1   # joint acceleration (rad/s²)

# ── ArUco ───────────────────────────────────────────────────────────────
ARUCO_DICT_TYPE = aruco.DICT_6X6_250
ARUCO_TAG_ID    = 3
MARKER_SIZE     = 0.021   # metres

# ── Move speeds (to scan pose) ──────────────────────────────────────────
APPROACH_SPEED = 0.5    # joint speed to scan pose (rad/s)
APPROACH_ACCEL = 0.3


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


# ── Helpers ─────────────────────────────────────────────────────────────
def tcp_to_matrix(tcp_pose):
    R, _ = cv2.Rodrigues(np.array(tcp_pose[3:]))
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = tcp_pose[:3]
    return T


def detect_tag(frame, grey, K, dist, T_cam2flange, rtde_r, detector):
    """
    Run one frame of ArUco detection.
    Returns (tag_pos_base, annotated_frame) or (None, annotated_frame).
    """
    corners, ids, _ = detector.detectMarkers(grey)
    if ids is None:
        return None, frame

    aruco.drawDetectedMarkers(frame, corners, ids)
    for i, mid in enumerate(ids.flatten()):
        if mid != ARUCO_TAG_ID:
            continue
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners[i:i+1], MARKER_SIZE, K, dist
        )
        rvec, tvec = rvecs[0][0], tvecs[0][0]
        cv2.drawFrameAxes(frame, K, dist, rvec, tvec, MARKER_SIZE * 0.5)

        R_tag, _ = cv2.Rodrigues(rvec)
        T_tag2cam = np.eye(4); T_tag2cam[:3, :3] = R_tag; T_tag2cam[:3, 3] = tvec
        tcp_pose   = rtde_r.getActualTCPPose()
        T_tcp2base = tcp_to_matrix(tcp_pose)
        T_tag2tcp  = T_cam2flange @ T_tag2cam
        T_tag2base = T_tcp2base   @ T_tag2tcp
        return T_tag2base[:3, 3].copy(), frame

    return None, frame


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
    time.sleep(0.5)

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

    # ── Load scan pose ──────────────────────────────────────────────────
    print("=" * 55)
    SCAN_JOINT_POS = load_scan_pose()
    print(f"  Scan pose loaded: {[round(j, 3) for j in SCAN_JOINT_POS]}")
    print("  Press ENTER to move to scan pose (hand on E-stop) …")
    input()
    print("  Moving to scan pose …")
    rtde_c.moveJ(SCAN_JOINT_POS, APPROACH_SPEED, APPROACH_ACCEL)
    print("  Scan pose reached.\n")

    # ── Build sweep waypoints (vary only J0) ────────────────────────────
    base_q      = list(rtde_r.getActualQ())
    sweep_start = list(base_q); sweep_start[0] += SWEEP_START_RAD
    sweep_end   = list(base_q); sweep_end[0]   += SWEEP_END_RAD

    print(f"  Sweeping J0 from {sweep_start[0]:.3f} → {sweep_end[0]:.3f} rad …")
    print("  Press ENTER to start sweep (Q in camera window = abort) …")
    input()

    # ── Start sweep asynchronously then poll camera ─────────────────────
    # moveJ is blocking; we run it in a thread and poll the camera on the main thread.
    tag_result   = {"pos": None}
    sweep_done   = threading.Event()

    def _sweep():
        """Sweep J0: first to SWEEP_END, then back to SWEEP_START. Stops early if tag found."""
        for waypoint in (sweep_end, sweep_start):
            if tag_result["pos"] is not None:
                break   # tag already found, skip remaining waypoints
            try:
                rtde_c.moveJ(waypoint, SWEEP_SPEED, SWEEP_ACCEL)
            except Exception as e:
                print(f"  [sweep] {e}")
                break
        sweep_done.set()

    sweep_thread = threading.Thread(target=_sweep, daemon=True)
    sweep_thread.start()

    print("  Sweeping … (tag window open – Q to abort)\n")
    while not sweep_done.is_set():
        pkt   = videoQueue.get()
        frame = pkt.getCvFrame()
        grey  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        tag_pos, frame = detect_tag(frame, grey, K, dist, T_cam2flange, rtde_r, detector)

        if tag_pos is not None and tag_result["pos"] is None:
            tag_result["pos"] = tag_pos
            px, py, pz = tag_pos
            print(f"  ✓ Tag detected at base frame: X={px:.4f}  Y={py:.4f}  Z={pz:.4f}")
            print("  Stopping sweep …")
            try:
                rtde_c.stopJ(SWEEP_ACCEL)
            except Exception:
                pass
            cv2.putText(frame, "TAG FOUND — stopping", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        label = "TAG FOUND" if tag_result["pos"] is not None else "Searching for tag …"
        color = (0, 255, 0) if tag_result["pos"] is not None else (0, 0, 255)
        cv2.putText(frame, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("explore", cv2.resize(frame, (960, 540)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            try: rtde_c.stopJ(SWEEP_ACCEL)
            except Exception: pass
            break

    sweep_thread.join(timeout=2.0)
    cv2.destroyAllWindows()

    # ── Result ──────────────────────────────────────────────────────────
    if tag_result["pos"] is not None:
        px, py, pz = tag_result["pos"]
        print("\n" + "=" * 55)
        print(f"  EXPLORE COMPLETE")
        print(f"  Tag base frame: X={px:.4f}  Y={py:.4f}  Z={pz:.4f}")
        print("=" * 55)
    else:
        print("\n  Tag NOT found during sweep.")

    # Cleanup
    gripper.stop()
    device.close()
    rtde_r.disconnect()
    rtde_c.stopScript()
    print("\nDone.")
    return tag_result["pos"]


if __name__ == "__main__":
    main()
