"""
temp/capture_scan_pose.py
--------------------------
Utility script to record the robot's scan pose.

Usage
-----
1.  Free-drive (or jog) the robot to the position you want it to use
    as the starting pose before the base-joint sweep in explore.py.
2.  Run this script:
        python temp/capture_scan_pose.py
3.  The live camera feed opens so you can see what the robot sees.
4.  Press SPACE to capture and save the current robot pose.
5.  The pose is saved to  full_pipeline/data/scan_pose.json.
6.  explore.py will load this file automatically at startup.

Press Q to quit without saving.
"""

import json
import time
from pathlib import Path

import cv2
import depthai as dai
import rtde_receive

# ── Paths ───────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent          # full_pipeline/temp/
DATA_DIR   = SCRIPT_DIR.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
OUT_FILE   = DATA_DIR / "scan_pose.json"

# ── Robot ───────────────────────────────────────────────────────────────
ROBOT_IP = "192.168.8.102"

# ── Connect ─────────────────────────────────────────────────────────────
print("Connecting to robot …")
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
print("Robot connected!\n")

# ── Camera ──────────────────────────────────────────────────────────────
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
print("Camera live. Jog robot to scan pose then press SPACE to capture.\n")

# ── Live loop ────────────────────────────────────────────────────────────
saved = False

while True:
    frame = videoQueue.get().getCvFrame()

    # Overlay current TCP position
    tcp   = rtde_r.getActualTCPPose()
    q     = rtde_r.getActualQ()
    x, y, z = tcp[0], tcp[1], tcp[2]
    cv2.putText(frame, f"TCP: X={x:.4f}  Y={y:.4f}  Z={z:.4f}  (m)",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(frame, f"J0={q[0]:.3f}  J1={q[1]:.3f}  J2={q[2]:.3f}  J3={q[3]:.3f}  J4={q[4]:.3f}  J5={q[5]:.3f}",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)
    cv2.putText(frame, "SPACE = save pose  |  Q = quit",
                (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    if saved:
        cv2.putText(frame, f"SAVED to {OUT_FILE.name}", (10, frame.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow("capture_scan_pose", cv2.resize(frame, (960, 540)))
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord(' '):
        tcp = rtde_r.getActualTCPPose()
        q   = rtde_r.getActualQ()

        pose_data = {
            "tcp_pose":    [round(v, 6) for v in tcp],   # [x,y,z,rx,ry,rz]
            "joint_angles": [round(v, 6) for v in q]     # [J0..J5]  ← used by explore.py
        }
        with open(OUT_FILE, "w") as f:
            json.dump(pose_data, f, indent=2)

        print("\n" + "=" * 55)
        print(f"  Scan pose saved → {OUT_FILE}")
        print(f"  TCP:    {pose_data['tcp_pose']}")
        print(f"  Joints: {pose_data['joint_angles']}")
        print("=" * 55)
        saved = True

# ── Cleanup ─────────────────────────────────────────────────────────────
cv2.destroyAllWindows()
device.close()
rtde_r.disconnect()
print("\nDone.")
