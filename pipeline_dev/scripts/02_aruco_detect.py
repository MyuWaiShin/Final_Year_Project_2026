import os
import sys
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

# ── Robust path setup — works from any working directory ─────────
SCRIPT_DIR = Path(__file__).resolve().parent          # .../13_03_2026/scripts
BASE_DIR   = SCRIPT_DIR.parent                        # .../13_03_2026

MARKER_SIZE = 0.03  # 30mm in metres
DICTIONARY  = cv2.aruco.DICT_6X6_250

# Load camera intrinsics
K    = np.load(BASE_DIR / "calibration/camera_matrix.npy")
dist = np.zeros((4, 1))

# Set up ArUco detector
dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY)
detector   = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

# ── Camera Pipeline (DepthAI API 2.0) ─────────────────────────────
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

print("Starting camera...")
print("Hold your ArUco tag in front of the camera")
print("Press Q to quit\n")


while True:
    frame_pkt = videoQueue.tryGet()
    if frame_pkt is not None:
        img  = frame_pkt.getCvFrame()
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = detector.detectMarkers(grey)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(img, corners, ids)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, MARKER_SIZE, K, dist
            )
            for i in range(len(ids)):
                tvec = tvecs[i][0]
                cv2.drawFrameAxes(img, K, dist, rvecs[i], tvecs[i], MARKER_SIZE)
                print(f"Tag ID {ids[i][0]} | "
                      f"X: {tvec[0]:.3f}m  "
                      f"Y: {tvec[1]:.3f}m  "
                      f"Z: {tvec[2]:.3f}m")
        else:
            print("No tag detected...")

        cv2.imshow("ArUco Detection", cv2.resize(img, (960, 540)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

device.close()
cv2.destroyAllWindows()
print("Stopped.")
