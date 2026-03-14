"""
10_move_to_tag.py
------------------
Detects ArUco tag (ID 3) on the floor.
Press SPACE → confirm → flange moves to 10cm above the tag.
No tip offset, no orientation changes — pure flange to tag XYZ.
"""

import cv2
import cv2.aruco as aruco
import depthai as dai
import rtde_receive
import rtde_control
import numpy as np

# ── CONFIG ─────────────────────────────────────────────────────────
ROBOT_IP          = "192.168.8.102"
ARUCO_DICT        = aruco.DICT_6X6_50
ARUCO_TAG_ID      = 3
MARKER_SIZE       = 0.042        # metres
APPROACH_Z_OFFSET = 0.10         # stop 10cm above tag Z
MOVE_SPEED        = 0.02         # 2 cm/s
MOVE_ACCEL        = 0.01

# ── Load calibration ───────────────────────────────────────────────
print("Loading calibration files...")
K            = np.load("calibration/camera_matrix.npy")
dist         = np.zeros((4, 1))
T_cam2flange = np.load("calibration/T_cam2flange.npy")
print("  camera_matrix.npy   ✓")
print("  T_cam2flange.npy    ✓\n")

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
print("  SPACE  →  move FLANGE to 10cm above tag")
print("  Q      →  quit")
print("=" * 55)
print()

# ── Helper ─────────────────────────────────────────────────────────
def tcp_to_matrix(tcp_pose):
    R, _ = cv2.Rodrigues(np.array(tcp_pose[3:]))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = tcp_pose[:3]
    return T

# ── Main loop ──────────────────────────────────────────────────────
while True:
    imgFrame = videoQueue.get()
    frame    = imgFrame.getCvFrame()
    grey     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = detector.detectMarkers(grey)

    tag_detected = False
    tag_pos_base = None
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

                # T_tag2cam
                R_tag, _ = cv2.Rodrigues(rvec)
                T_tag2cam = np.eye(4)
                T_tag2cam[:3, :3] = R_tag
                T_tag2cam[:3, 3]  = tvec

                # tag in base frame
                current_tcp   = rtde_r.getActualTCPPose()
                T_flange2base = tcp_to_matrix(current_tcp)
                T_tag2base    = T_flange2base @ T_cam2flange @ T_tag2cam
                tag_pos_base  = T_tag2base[:3, 3]

                bx, by, bz = tag_pos_base
                cv2.putText(frame, f"tag base XYZ: ({bx:.3f}, {by:.3f}, {bz:.3f})m",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"target Z (+10cm): {bz + APPROACH_Z_OFFSET:.3f}m",
                            (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                break

    if tag_detected:
        cv2.putText(frame, f"TAG {ARUCO_TAG_ID} DETECTED — SPACE to move flange here",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    else:
        cv2.putText(frame, f"Tag {ARUCO_TAG_ID} not detected",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    cv2.putText(frame, "SPACE = move flange to tag | Q = quit",
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Move Flange to Tag", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord(' '):
        if not tag_detected or tag_pos_base is None:
            print("Tag not detected — point camera at tag first.")
            continue

        bx, by, bz = tag_pos_base
        # Keep current orientation, just move XYZ
        rx, ry, rz  = current_tcp[3], current_tcp[4], current_tcp[5]
        target_pose = [bx, by, bz + APPROACH_Z_OFFSET, rx, ry, rz]

        print("\n" + "=" * 55)
        print(f"  Tag in base frame:   X={bx:.4f}  Y={by:.4f}  Z={bz:.4f}")
        print(f"  Moving flange to:    X={bx:.4f}  Y={by:.4f}  Z={bz+APPROACH_Z_OFFSET:.4f}")
        print(f"  Orientation kept:    RX={rx:.4f}  RY={ry:.4f}  RZ={rz:.4f}")
        print(f"  Speed: {MOVE_SPEED*100:.0f} cm/s")
        print("=" * 55)
        confirm = input("  Type YES to move, anything else to cancel: ").strip()

        if confirm.upper() != "YES":
            print("  Cancelled.\n")
            continue

        print("  Moving... hand on E-stop.")
        try:
            rtde_c.moveL(target_pose, MOVE_SPEED, MOVE_ACCEL)
            print("  Done! How close is the flange to being above the tag?")
        except Exception as e:
            print(f"  Move failed: {e}")

cv2.destroyAllWindows()
pipeline.stop()
rtde_r.disconnect()
rtde_c.stopScript()
print("\nDone.")