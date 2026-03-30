"""
08_readiness_check.py
------------------------------------------------------
- Shows live camera feed
- Detects ChArUco board and draws overlay if found
- Press SPACE to print current TCP pose to terminal
- Press Q to quit
"""

from pathlib import Path

import cv2
import depthai as dai
import rtde_receive
import numpy as np

# ── Robust path setup ─────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR   = SCRIPT_DIR.parent

ROBOT_IP = "192.168.8.102"

# ── Must match the board you physically have ──────────────────────
CHARUCO_SQUARES_X   = 7
CHARUCO_SQUARES_Y   = 5
CHARUCO_SQUARE_SIZE = 0.038
CHARUCO_MARKER_SIZE = 0.028
CHARUCO_DICT        = cv2.aruco.DICT_6X6_250

dictionary       = cv2.aruco.getPredefinedDictionary(CHARUCO_DICT)
board            = cv2.aruco.CharucoBoard(
    (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
    CHARUCO_SQUARE_SIZE, CHARUCO_MARKER_SIZE, dictionary
)
charuco_detector = cv2.aruco.CharucoDetector(
    board,
    cv2.aruco.CharucoParameters(),
    cv2.aruco.DetectorParameters()
)

K    = np.load(BASE_DIR / "calibration/camera_matrix.npy")
dist = np.zeros((4, 1))

print("Connecting to robot...")
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
print("Robot connected!\n")

# ── Camera Pipeline (DepthAI API 2.0) ─────────────────────────────
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


print("=" * 50)
print("CONTROLS:  SPACE = print TCP  |  Q = quit")
print("=" * 50)

sample_count = 0

while True:
    pkt   = videoQueue.get()
    frame = pkt.getCvFrame()
    grey  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(grey)

    board_detected = False
    corner_count   = 0

    if charuco_ids is not None and len(charuco_ids) >= 4:
        cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)
        valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners, charuco_ids, board, K, dist,
            np.zeros((3, 1)), np.zeros((3, 1))
        )
        if valid:
            cv2.drawFrameAxes(frame, K, dist, rvec, tvec, CHARUCO_SQUARE_SIZE)
            board_detected = True
            corner_count   = len(charuco_ids)

    status_text  = f"BOARD DETECTED ({corner_count} corners)" if board_detected else "Board NOT detected"
    status_color = (0, 255, 0) if board_detected else (0, 0, 255)

    cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
    cv2.putText(frame, "SPACE = print TCP | Q = quit", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"TCP samples: {sample_count}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow("Readiness Check", cv2.resize(frame, (960, 540)))
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):
        tcp = rtde_r.getActualTCPPose()
        sample_count += 1
        board_str = "board visible" if board_detected else "board NOT visible"
        print(f"[Sample {sample_count}] TCP: {[round(v, 6) for v in tcp]}  |  {board_str}")

device.close()
cv2.destroyAllWindows()
rtde_r.disconnect()
print("\nDone.")