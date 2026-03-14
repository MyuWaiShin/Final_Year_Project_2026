"""
Stage 6 — ArUco Tag → Full Transform Chain
--------------------------------------------
Detects ArUco tag (ID 3, DICT_6X6_50, 42mm) on the floor.
Press SPACE to print the tag position in three frames:
  1. Camera frame       — raw detection
  2. Flange frame       — relative to robot flange
  3. Robot base frame   — absolute position in robot world

Camera: OAK-D Lite, depthai v3 API
"""

import cv2
import cv2.aruco as aruco
import depthai as dai
import rtde_receive
import numpy as np

# ── CONFIG ─────────────────────────────────────────────────────────
ROBOT_IP     = "192.168.8.102"
ARUCO_DICT   = aruco.DICT_6X6_50
ARUCO_TAG_ID = 3
MARKER_SIZE  = 0.042   # metres

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
print("Robot connected!\n")

# ── Camera — depthai v3 ────────────────────────────────────────────
print("Starting camera...")
pipeline    = dai.Pipeline()
cam         = pipeline.create(dai.node.Camera).build()
videoOutput = cam.requestOutput((1280, 720), type=dai.ImgFrame.Type.BGR888p)
videoQueue  = videoOutput.createOutputQueue()
pipeline.start()
print("Camera started!\n")

print("=" * 55)
print("CONTROLS:")
print("  SPACE  →  print full transform chain")
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

# ── Main loop ──────────────────────────────────────────────────────
while True:
    imgFrame = videoQueue.get()
    frame    = imgFrame.getCvFrame()
    grey     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = detector.detectMarkers(grey)

    tag_detected   = False
    tag_pos_cam    = None
    tag_pos_flange = None
    tag_pos_base   = None

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
                tag_pos_cam  = tvec

                # T_tag2cam
                R_tag, _ = cv2.Rodrigues(rvec)
                T_tag2cam = np.eye(4)
                T_tag2cam[:3, :3] = R_tag
                T_tag2cam[:3, 3]  = tvec

                # Robot state
                tcp           = rtde_r.getActualTCPPose()
                T_flange2base = tcp_to_matrix(tcp)

                # tag in flange frame
                T_tag2flange   = T_cam2flange @ T_tag2cam
                tag_pos_flange = T_tag2flange[:3, 3]

                # tag in base frame
                T_tag2base   = T_flange2base @ T_tag2flange
                tag_pos_base = T_tag2base[:3, 3]

                dist_m        = np.linalg.norm(tvec)
                fx, fy, fz    = tag_pos_flange
                bx, by, bz    = tag_pos_base

                cv2.putText(frame, f"cam dist: {dist_m:.3f}m",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                cv2.putText(frame, f"flange: ({fx:.3f}, {fy:.3f}, {fz:.3f})m",
                            (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 165, 0), 2)
                cv2.putText(frame, f"base:   ({bx:.3f}, {by:.3f}, {bz:.3f})m",
                            (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
                break

    if tag_detected:
        cv2.putText(frame, f"TAG {ARUCO_TAG_ID} DETECTED", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        cv2.putText(frame, f"Tag {ARUCO_TAG_ID} not detected", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.putText(frame, "SPACE = print transforms | Q = quit",
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Stage 6 - ArUco Transform", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):
        if not tag_detected:
            print("Tag not detected — point camera at floor tag and try again.")
        else:
            tcp = rtde_r.getActualTCPPose()
            cx, cy, cz = tag_pos_cam
            fx, fy, fz = tag_pos_flange
            bx, by, bz = tag_pos_base
            print("\n" + "=" * 55)
            print(f"  TCP (flange→base):   {[round(v,4) for v in tcp]}")
            print(f"  Tag in CAMERA frame: X={cx:.4f}  Y={cy:.4f}  Z={cz:.4f}")
            print(f"  Tag in FLANGE frame: X={fx:.4f}  Y={fy:.4f}  Z={fz:.4f}")
            print(f"  Tag in BASE frame:   X={bx:.4f}  Y={by:.4f}  Z={bz:.4f}")
            print("=" * 55)

cv2.destroyAllWindows()
pipeline.stop()
rtde_r.disconnect()
print("\nDone.")