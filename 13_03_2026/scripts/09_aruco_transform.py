"""
09_aruco_transform.py
----------------------
Detects ArUco tag (ID 3, DICT_6X6_250) on the floor.
Press SPACE to print the tag position in all three frames.
"""

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
MARKER_SIZE  = 0.042   # metres

K            = np.load(BASE_DIR / "calibration/camera_matrix.npy")
dist         = np.zeros((4, 1))
T_cam2flange = np.load(BASE_DIR / "calibration/T_cam2flange.npy")
print("  camera_matrix.npy   ✓")
print("  T_cam2flange.npy    ✓\n")

dictionary      = aruco.getPredefinedDictionary(ARUCO_DICT)
detector        = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())

print("Connecting to robot...")
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
print("Robot connected!\n")

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
    tag_pos_cam = tag_pos_flange = tag_pos_base = None

    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)
        for i, mid in enumerate(ids.flatten()):
            if mid == ARUCO_TAG_ID:
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners[i:i+1], MARKER_SIZE, K, dist)
                rvec, tvec = rvecs[0][0], tvecs[0][0]
                cv2.drawFrameAxes(frame, K, dist, rvec, tvec, MARKER_SIZE * 0.5)
                tag_detected = True; tag_pos_cam = tvec

                R_tag, _ = cv2.Rodrigues(rvec)
                T_tag2cam = np.eye(4); T_tag2cam[:3, :3] = R_tag; T_tag2cam[:3, 3] = tvec

                T_flange2base  = tcp_to_matrix(rtde_r.getActualTCPPose())
                T_tag2flange   = T_cam2flange @ T_tag2cam
                tag_pos_flange = T_tag2flange[:3, 3]
                T_tag2base     = T_flange2base @ T_tag2flange
                tag_pos_base   = T_tag2base[:3, 3]

                fx, fy, fz = tag_pos_flange
                bx, by, bz = tag_pos_base
                cv2.putText(frame, f"flange: ({fx:.3f}, {fy:.3f}, {fz:.3f})m", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 165, 0), 2)
                cv2.putText(frame, f"base:   ({bx:.3f}, {by:.3f}, {bz:.3f})m", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
                break

    label = f"TAG {ARUCO_TAG_ID} DETECTED" if tag_detected else f"Tag {ARUCO_TAG_ID} not detected"
    color = (0, 255, 0) if tag_detected else (0, 0, 255)
    cv2.putText(frame, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, "SPACE = print | Q = quit", (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("ArUco Transform", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' ') and tag_detected:
        tcp = rtde_r.getActualTCPPose()
        cx, cy, cz = tag_pos_cam
        fx, fy, fz = tag_pos_flange
        bx, by, bz = tag_pos_base
        print("\n" + "=" * 55)
        print(f"  Tag in CAMERA frame: X={cx:.4f}  Y={cy:.4f}  Z={cz:.4f}")
        print(f"  Tag in FLANGE frame: X={fx:.4f}  Y={fy:.4f}  Z={fz:.4f}")
        print(f"  Tag in BASE frame:   X={bx:.4f}  Y={by:.4f}  Z={bz:.4f}")
        print("=" * 55)

device.close()
cv2.destroyAllWindows()
rtde_r.disconnect()
print("\nDone.")