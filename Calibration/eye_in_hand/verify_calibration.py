"""
verify_calibration.py — Eye-in-Hand Calibration: Live Verification
================================================================
Loads the final handeye_calibration.json matrix and projects a live
3D coordinate frame (X=Red, Y=Green, Z=Blue) into the real-world
image directly onto the robot's TCP (Tool Center Point).

This lets you visually confirm the calibration: wherever the robot
moves, the animated axes should perfectly track the tip of the
robot arm on screen.

Run:
    python verify_calibration.py
"""

import json
import socket
import struct
import threading
import time
from pathlib import Path
import sys

import cv2
import depthai as dai
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
ROBOT_IP   = "192.168.8.102"
CALIB_FILE = Path("handeye_calibration.json")
CAM_W, CAM_H = 1280, 720


# ─────────────────────────────────────────────────────────────────────────────
# Background TCP reader (keeps camera smooth)
# ─────────────────────────────────────────────────────────────────────────────
class TCPPoller(threading.Thread):
    def __init__(self, ip=ROBOT_IP):
        super().__init__(daemon=True)
        self.ip    = ip
        self.pose  = None
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            self.pose = self._read()
            time.sleep(0.2)  # 5 Hz polling (prevents UR10 network spam)

    def _read(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                s.connect((self.ip, 30003))
                data = s.recv(1200)
                if len(data) < 492:
                    return None
                return list(struct.unpack("!6d", data[444:492]))
        except Exception:
            return None

    def stop(self):
        self._stop.set()


# ─────────────────────────────────────────────────────────────────────────────
# Math Helpers
# ─────────────────────────────────────────────────────────────────────────────
def rotvec_to_matrix(rx, ry, rz):
    vec   = np.array([rx, ry, rz], dtype=np.float64)
    angle = np.linalg.norm(vec)
    if angle < 1e-9:
        return np.eye(3)
    axis = vec / angle
    K = np.array([[    0,    -axis[2],  axis[1]],
                  [ axis[2],     0,    -axis[0]],
                  [-axis[1],  axis[0],     0   ]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def draw_axes(img, rvec, tvec, K, dist, length=0.1):
    """Project and draw 3D RGB axes (length in meters)."""
    # Define XYZ origin and tips of the 3 axes
    axes_3d = np.float32([
        [0, 0, 0],
        [length, 0, 0],  # X
        [0, length, 0],  # Y
        [0, 0, length]   # Z
    ])
    
    # Project 3D points down onto the 2D image plane using camera intrinsics
    img_pts, _ = cv2.projectPoints(axes_3d, rvec, tvec, K, dist)
    img_pts = np.int32(img_pts.reshape(-1, 2))

    origin, x_tip, y_tip, z_tip = img_pts

    # Draw lines (OpenCV uses BGR, we want XYZ = RGB)
    # X axis -> RED
    cv2.line(img, origin, x_tip, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "X", x_tip, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Y axis -> GREEN
    cv2.line(img, origin, y_tip, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "Y", y_tip, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Z axis -> BLUE
    cv2.line(img, origin, z_tip, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "Z", z_tip, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Origin marker
    cv2.circle(img, origin, 4, (255, 255, 255), -1)


# ─────────────────────────────────────────────────────────────────────────────
# OAK-D pipeline
# ─────────────────────────────────────────────────────────────────────────────
def create_pipeline() -> dai.Pipeline:
    p   = dai.Pipeline()
    cam = p.create(dai.node.ColorCamera)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setVideoSize(CAM_W, CAM_H)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(30)
    
    # MJPEG encoding prevents USB crashing
    videoEnc = p.create(dai.node.VideoEncoder)
    videoEnc.setDefaultProfilePreset(30, dai.VideoEncoderProperties.Profile.MJPEG)
    videoEnc.setQuality(90)
    cam.video.link(videoEnc.input)

    xout = p.create(dai.node.XLinkOut)
    xout.setStreamName("mjpeg")
    videoEnc.bitstream.link(xout.input)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Main Loop (Live Verification)
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not CALIB_FILE.exists():
        print(f"ERROR: {CALIB_FILE} not found. Run solve_calibration.py first.")
        sys.exit(1)

    print(f"Loading '{CALIB_FILE}'...")
    with open(CALIB_FILE) as f:
        data = json.load(f)

    # 1. Hand-Eye transform: Camera -> Robot Gripper
    R_cam2tcp = np.array(data["R_cam2tcp"], dtype=np.float64)
    t_cam2tcp = np.array(data["t_cam2tcp"], dtype=np.float64).reshape(3, 1)

    # 2. Camera Intrinsics
    intr = data["intrinsics"]
    K = np.array([[intr["fx"],          0, intr["cx"]],
                  [         0, intr["fy"], intr["cy"]],
                  [         0,          0,          1]], dtype=np.float64)
    # Distortion: Use None (zero) for axes to avoid stretching weirdness
    # (since the axes are close to center anyway)
    
    # 3. Diagnostics
    reproj_err = data.get("board_reproj_mean_px", 0.0)
    best_method_name = data.get("best_method", "Unknown")
    all_methods = data.get("all_methods", {})
    
    current_method_idx = 0
    method_names = list(all_methods.keys())
    # Try to start at the best one
    if best_method_name in method_names:
        current_method_idx = method_names.index(best_method_name)

    # Validation check
    if np.array(data["t_cam2tcp"])[2] > 0:
        # Note: If t_cam2tcp_z is positive, then Camera is forward of TCP.
        # Usually for Eye-in-Hand, Camera is BEHIND TCP, so t_z should be negative.
        print("\n[INFO] Loaded calibration has positive Z translation.")
    
    cv2.namedWindow("Calibration Verification")
    print("\nControls:")
    print("  [1-5] Switch Calibration Methods")
    print("  [q]   Quit\n")

    # Setup Robot Connection
    tcp_poller = TCPPoller(ROBOT_IP)
    tcp_poller.start()
    print("TCP Background Poller Started.")

    # Setup Camera Connection
    cfg = dai.Device.Config()
    pipeline = create_pipeline()

    print("\n==================================")
    print(" LIVE TCP VERIFICATION ACTIVE")
    print("==================================")
    print("Move the robot around with the pendant.")
    print("You should see a 3D coordinate frame (Red/Green/Blue)")
    print("permanently stuck to the physical tip of the robot.")
    print("\nPress Q to quit.\n")

    try:
        with dai.Device(cfg) as device:
            device.startPipeline(pipeline)
            q = device.getOutputQueue("mjpeg", maxSize=2, blocking=False)

            while True:
                in_encoded = q.get()
                frame = cv2.imdecode(in_encoded.getData(), cv2.IMREAD_COLOR)

                if frame is None:
                    continue

                tcp = tcp_poller.pose
                
                # Get current method data
                m_name = method_names[current_method_idx]
                m_data = all_methods[m_name]
                R_cam2tcp = np.array(m_data["R"])
                t_cam2tcp = np.array(m_data["t"]).reshape(3, 1)

                if tcp is not None:
                    # Show the TCP frame in the camera image
                    # Use the inverse to project TCP origin (0,0,0) into Camera pixels
                    R_tcp2cam = R_cam2tcp.T
                    t_tcp2cam = -R_tcp2cam @ t_cam2tcp
                    
                    r_vec, _ = cv2.Rodrigues(R_tcp2cam)
                    
                    # Draw a 3cm access frame with labels and NO distortion
                    draw_axes(frame, r_vec, t_tcp2cam, K, None, length=0.03)

                    text = f"Live TCP: ({tcp[0]*1000:+.0f}, {tcp[1]*1000:+.0f}, {tcp[2]*1000:+.0f}) mm"
                    cv2.putText(frame, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Diagnostic HUD
                    diag_text = f"Method [{current_method_idx+1}/5]: {m_name}"
                    cv2.putText(frame, diag_text, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    info_text = f"Consistency: {reproj_err:.2f} px (Mean)"
                    cv2.putText(frame, info_text, (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                else:
                    cv2.putText(frame, "Waiting for Robot TCP...", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                cv2.imshow("Calibration Verification", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif ord('1') <= key <= ord('5'):
                    idx = key - ord('1')
                    if idx < len(method_names):
                        current_method_idx = idx
                        print(f"Switched to method: {method_names[idx]}")
    finally:
        tcp_poller.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
