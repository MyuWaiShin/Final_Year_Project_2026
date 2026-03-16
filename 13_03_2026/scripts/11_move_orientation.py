"""
11_move_orientation.py
-----------------------
Detects ArUco tag (ID 3), moves flange to tag XYZ + 5cm above.

--hard true  → uses fixed orientation from 04_robot_connect.py reading
               RX=-2.2071  RY=-2.1849  RZ=0.0547
--hard false → computes perpendicular orientation from tag normal

Press SPACE → confirm → move.
"""

import cv2
import cv2.aruco as aruco
import depthai as dai
import rtde_receive
import rtde_control
import numpy as np
import argparse

# ── ARGS ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--hard", type=str, choices=["true", "false"],
                    required=True,
                    help="true = use fixed orientation | false = perpendicular from tag")
args = parser.parse_args()
USE_HARD = args.hard.lower() == "true"

# ── CONFIG ─────────────────────────────────────────────────────────
ROBOT_IP          = "192.168.8.102"
ARUCO_DICT        = aruco.DICT_6X6_50
ARUCO_TAG_ID      = 3
MARKER_SIZE       = 0.021
APPROACH_Z_OFFSET = 0.05
MOVE_SPEED        = 0.02
MOVE_ACCEL        = 0.01

# Fixed orientation from 04_robot_connect.py reading
HARD_RX = -2.2071
HARD_RY = -2.1849
HARD_RZ =  0.0547

# ── Load calibration ───────────────────────────────────────────────
print("Loading calibration files...")
K            = np.load("calibration/camera_matrix.npy")
dist         = np.zeros((4, 1))
T_cam2flange = np.load("calibration/T_cam2flange.npy")
print("  camera_matrix.npy   ✓")
print("  T_cam2flange.npy    ✓")
print(f"  Orientation mode:   {'HARD (fixed)' if USE_HARD else 'PERP (from tag normal)'}\n")

# ── ArUco detector ─────────────────────────────────────────────────
dictionary      = aruco.getPredefinedDictionary(ARUCO_DICT)
detector_params = aruco.DetectorParameters()
detector        = aruco.ArucoDetector(dictionary, detector_params)

# ── Robot ──────────────────────────────────────────────────────────
print("Connecting to robot...")
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
print("Robot connected!\n")

# ── Camera ─────────────────────────────────────────────────────────
print("Starting camera...")
pipeline    = dai.Pipeline()
cam         = pipeline.create(dai.node.Camera).build()
videoOutput = cam.requestOutput((1280, 720), type=dai.ImgFrame.Type.BGR888p)
videoQueue  = videoOutput.createOutputQueue()
pipeline.start()
print("Camera started!\n")

print("=" * 55)
print("CONTROLS:")
print("  SPACE  →  move to tag")
print("  Q      →  quit")
print("=" * 55)
print()

# ── Helpers ────────────────────────────────────────────────────────
def tcp_to_matrix(tcp_pose):
    R, _ = cv2.Rodrigues(np.array(tcp_pose[3:]))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = tcp_pose[:3]
    return T

def perpendicular_approach_rvec(T_tag2base):
    R_tag      = T_tag2base[:3, :3]
    tag_z      = R_tag[:, 2]
    approach_z = -tag_z / np.linalg.norm(tag_z)

    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(approach_z, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])

    approach_y = np.cross(approach_z, ref)
    approach_y = approach_y / np.linalg.norm(approach_y)
    approach_x = np.cross(approach_y, approach_z)
    approach_x = approach_x / np.linalg.norm(approach_x)

    R_approach = np.column_stack([approach_x, approach_y, approach_z])
    rvec, _    = cv2.Rodrigues(R_approach)
    return rvec.flatten()

# ── Main loop ──────────────────────────────────────────────────────
while True:
    imgFrame = videoQueue.get()
    frame    = imgFrame.getCvFrame()
    grey     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = detector.detectMarkers(grey)

    tag_detected = False
    tag_pos_base = None
    T_tag2base   = None
    current_tcp  = None

    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)

        for i, marker_id in enumerate(ids.flatten()):
            if marker_id == ARUCO_TAG_ID:
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                    corners[i:i+1], MARKER_SIZE, K, dist
                )
                rvec = rvecs[0][0]
                tvec = tvecs[0][0]
                cv2.drawFrameAxes(frame, K, dist, rvec, tvec, MARKER_SIZE * 0.5)

                tag_detected = True

                R_tag, _ = cv2.Rodrigues(rvec)
                T_tag2cam = np.eye(4)
                T_tag2cam[:3, :3] = R_tag
                T_tag2cam[:3, 3]  = tvec

                current_tcp   = rtde_r.getActualTCPPose()
                T_flange2base = tcp_to_matrix(current_tcp)
                T_tag2base    = T_flange2base @ T_cam2flange @ T_tag2cam
                tag_pos_base  = T_tag2base[:3, 3]

                bx, by, bz = tag_pos_base
                cv2.putText(frame, f"tag base: ({bx:.3f}, {by:.3f}, {bz:.3f})m",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
                cv2.putText(frame, f"target:   ({bx:.3f}, {by:.3f}, {bz+APPROACH_Z_OFFSET:.3f})m",
                            (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                mode_str = "HARD orientation" if USE_HARD else "PERP orientation"
                cv2.putText(frame, mode_str,
                            (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 165, 0), 2)
                break

    if tag_detected:
        cv2.putText(frame, f"TAG {ARUCO_TAG_ID} DETECTED — SPACE to move",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(frame, f"Tag {ARUCO_TAG_ID} not detected",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(frame, "SPACE = move | Q = quit",
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Move to Tag - Orientation", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord(' '):
        if not tag_detected or tag_pos_base is None:
            print("Tag not detected — point camera at tag first.")
            continue

        bx, by, bz = tag_pos_base

        if USE_HARD:
            rx, ry, rz = HARD_RX, HARD_RY, HARD_RZ
            orient_str = "fixed (hard)"
        else:
            rx, ry, rz = perpendicular_approach_rvec(T_tag2base)
            orient_str = "perpendicular (from tag normal)"

        target_pose = [bx, by, bz + APPROACH_Z_OFFSET, rx, ry, rz]

        print("\n" + "=" * 55)
        print(f"  Tag in base frame:  X={bx:.4f}  Y={by:.4f}  Z={bz:.4f}")
        print(f"  Moving to:          X={bx:.4f}  Y={by:.4f}  Z={bz+APPROACH_Z_OFFSET:.4f}")
        print(f"  Orientation:        {orient_str}")
        print(f"  RX={rx:.4f}  RY={ry:.4f}  RZ={rz:.4f}")
        print(f"  Speed: {MOVE_SPEED*100:.0f} cm/s")
        print("=" * 55)
        confirm = input("  Type YES to move, anything else to cancel: ").strip()

        if confirm.upper() != "YES":
            print("  Cancelled.\n")
            continue

        print("  Moving... hand on E-stop.")
        try:
            rtde_c.moveL(target_pose, MOVE_SPEED, MOVE_ACCEL)
            print("  Done!")
        except Exception as e:
            print(f"  Move failed: {e}")

cv2.destroyAllWindows()
pipeline.stop()
rtde_r.disconnect()
rtde_c.stopScript()
print("\nDone.")