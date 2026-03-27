import os
from pathlib import Path

import cv2
import depthai as dai

# ── Robust path setup ─────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR   = SCRIPT_DIR.parent

# ── FOCUS SETTING ─────────────────────────────────────────────────
# OAK-D Lite manual focus: 0 = very close, 255 = far
# Change this value and re-run to find the sharpest focus:
#   0  – 50  → very close (~10–20cm)
#   50 – 100 → close (~20–40cm)
#   100–150  → medium (~40–80cm)
#   150–220  → far (~80cm+)
MANUAL_FOCUS = None  # <── CHANGE THIS VALUE TO TEST FOCUS
# Set to None to use autofocus instead:
# MANUAL_FOCUS = None

print(f"Starting camera with focus = {MANUAL_FOCUS}  (range: 0=close, 255=far)")
print("Change MANUAL_FOCUS value and re-run to test different values.")
print("Press Q to quit.\n")

# ── Camera Pipeline (DepthAI API 2.0) ─────────────────────────────
pipeline = dai.Pipeline()
cam      = pipeline.create(dai.node.ColorCamera)
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setVideoSize(1280, 720)   # crop sensor output to 720p
cam.setInterleaved(False)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
if MANUAL_FOCUS is not None:
    cam.initialControl.setManualFocus(MANUAL_FOCUS)
# else: autofocus is used by default

xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("video")
cam.video.link(xout.input)

device     = dai.Device(pipeline)
videoQueue = device.getOutputQueue(name="video", maxSize=4, blocking=False)


while True:
    frame_pkt = videoQueue.tryGet()
    if frame_pkt is not None:
        frame = frame_pkt.getCvFrame()
        cv2.putText(frame, f"Focus = {MANUAL_FOCUS}  (0=close, 255=far)",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imshow("OAK-D Lite - Focus Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

device.close()
cv2.destroyAllWindows()
print("Camera stopped.")
