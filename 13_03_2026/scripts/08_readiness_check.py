"""
Readiness Check — Camera + Robot + ChArUco Detection
------------------------------------------------------
- Shows live camera feed
- Detects ChArUco board and draws overlay if found
- Press SPACE to print current TCP pose to terminal
- Press Q to quit

All three passing = ready to run calibration.
"""

import cv2
import depthai as dai
import rtde_receive
import numpy as np

ROBOT_IP = "192.168.8.102"

# ── Must match the board you physically have ───────────────────────
CHARUCO_SQUARES_X   = 7
CHARUCO_SQUARES_Y   = 5
CHARUCO_SQUARE_SIZE = 0.038   # metres - measure your printed board
CHARUCO_MARKER_SIZE = 0.028   # metres - measure your printed board
CHARUCO_DICT        = cv2.aruco.DICT_6X6_250

# ── ChArUco board setup ────────────────────────────────────────────
dictionary = cv2.aruco.getPredefinedDictionary(CHARUCO_DICT)
board = cv2.aruco.CharucoBoard(
    (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
    CHARUCO_SQUARE_SIZE,
    CHARUCO_MARKER_SIZE,
    dictionary
)
detector_params  = cv2.aruco.DetectorParameters()
charuco_params   = cv2.aruco.CharucoParameters()
charuco_detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)

# ── Camera intrinsics ──────────────────────────────────────────────
K    = np.load("calibration/camera_matrix.npy")
dist = np.zeros((4, 1))

# ── Robot ──────────────────────────────────────────────────────────
print("Connecting to robot...")
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
print("Robot connected!\n")

# ── Camera ─────────────────────────────────────────────────────────
print("Starting camera...")
pipeline    = dai.Pipeline()
cam         = pipeline.create(dai.node.Camera).build()
videoOutput = cam.requestOutput((1280, 720), type=dai.ImgFrame.Type.BGR888p)
videoQueue  = videoOutput.createOutputQueue()
pipeline.start()
print("Camera started!\n")

print("=" * 50)
print("CONTROLS:")
print("  SPACE  →  print current TCP pose")
print("  Q      →  quit")
print("=" * 50)
print()

sample_count = 0

while True:
    imgFrame = videoQueue.get()
    frame    = imgFrame.getCvFrame()
    grey     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── ChArUco detection ──────────────────────────────────────────
    charuco_corners, charuco_ids, marker_corners, marker_ids = \
        charuco_detector.detectBoard(grey)

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

    # ── Overlay ────────────────────────────────────────────────────
    if board_detected:
        status_text  = f"BOARD DETECTED ({corner_count} corners)"
        status_color = (0, 255, 0)
    else:
        status_text  = "Board NOT detected"
        status_color = (0, 0, 255)

    cv2.putText(frame, status_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
    cv2.putText(frame, "SPACE = print TCP | Q = quit", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"TCP samples printed: {sample_count}", (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow("Readiness Check", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("\nQuitting.")
        break

    elif key == ord(' '):
        tcp = rtde_r.getActualTCPPose()
        sample_count += 1
        board_str = "board visible" if board_detected else "board NOT visible"
        print(f"[Sample {sample_count}] TCP: {[round(v, 6) for v in tcp]}  |  {board_str}")

cv2.destroyAllWindows()
pipeline.stop()
rtde_r.disconnect()
print("\nDone.")