import os
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

# ── Robust path setup ─────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR   = SCRIPT_DIR.parent

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
print("Camera started! Press Q to quit.")

while True:
    frame_pkt = videoQueue.tryGet()
    if frame_pkt is not None:
        cv2.imshow("OAK-D Lite - Colour Camera", frame_pkt.getCvFrame())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

device.close()
cv2.destroyAllWindows()
print("Camera stopped.")
