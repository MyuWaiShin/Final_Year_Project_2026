"""
00_aruco_info.py
-----------------
Runs for 5 seconds, tries every common ArUco dictionary,
detects any markers in view and prints: ID, dictionary, distance from camera.
"""

import time
from pathlib import Path

import cv2
import cv2.aruco as aruco
import depthai as dai
import numpy as np

# ── Robust path setup ─────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR   = SCRIPT_DIR.parent

K    = np.load(BASE_DIR / "calibration/camera_matrix.npy")
dist = np.zeros((4, 1))
MARKER_SIZE = 0.03

DICT_NAMES = [
    "DICT_4X4_50",  "DICT_4X4_100",  "DICT_4X4_250",
    "DICT_5X5_50",  "DICT_5X5_100",  "DICT_5X5_250",
    "DICT_6X6_50",  "DICT_6X6_100",  "DICT_6X6_250",
    "DICT_7X7_50",  "DICT_7X7_100",  "DICT_7X7_250",
]
detectors = {
    name: aruco.ArucoDetector(
        aruco.getPredefinedDictionary(getattr(aruco, name)),
        aruco.DetectorParameters()
    )
    for name in DICT_NAMES
}

# ── Camera Pipeline (DepthAI API 2.0) ─────────────────────────────
print("Starting camera... will run for 5 seconds.")
pipeline = dai.Pipeline()
cam      = pipeline.create(dai.node.ColorCamera)
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setInterleaved(False)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("video")
cam.video.link(xout.input)

device     = dai.Device(pipeline)
videoQueue = device.getOutputQueue(name="video", maxSize=4, blocking=False)

start = time.time()
found = {}

while time.time() - start < 5.0:
    pkt = videoQueue.tryGet()
    if pkt is not None:
        frame = pkt.getCvFrame()
        grey  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for dict_name, det in detectors.items():
            corners, ids, _ = det.detectMarkers(grey)
            if ids is not None:
                for i, mid in enumerate(ids.flatten()):
                    found.setdefault(dict_name, {})
                    if mid not in found[dict_name]:
                        _, tvecs, _ = aruco.estimatePoseSingleMarkers(
                            corners[i:i+1], MARKER_SIZE, K, dist)
                        found[dict_name][mid] = np.linalg.norm(tvecs[0][0])
        remaining = 5.0 - (time.time() - start)
        cv2.putText(frame, f"Scanning... {remaining:.1f}s", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imshow("ArUco Info", frame)
        cv2.waitKey(1)

device.close()
cv2.destroyAllWindows()

print("\n" + "=" * 50 + "\nRESULTS\n" + "=" * 50)
if not found:
    print("No markers detected!")
else:
    for dict_name, markers in found.items():
        for mid, dist_m in markers.items():
            print(f"  Dictionary : {dict_name}")
            print(f"  Marker ID  : {mid}")
            print(f"  Distance   : {dist_m:.3f}m\n")