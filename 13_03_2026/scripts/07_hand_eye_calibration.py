"""
07_hand_eye_calibration.py
---------------------------
MANUAL mode  (--pose false):
  Jog the robot using the teach pendant.
  Click the camera window, press SPACE to capture each pose.
  Press ENTER in the window when done (need >=15 poses).
  Poses are saved to data/calibration_poses.json for auto-replay.

AUTO mode  (--pose true):
  Loads saved joint angles from data/calibration_poses.json.
  Replays each pose automatically and captures.

NOTE: Requires stable WIRED ethernet to the robot for auto-replay mode.
      WiFi is unreliable for continuous RTDE control commands.
"""

import time
import json
import argparse
from pathlib import Path

import cv2
import depthai as dai
import numpy as np
import rtde_receive
import rtde_control

# ── Robust path setup ─────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR   = SCRIPT_DIR.parent

# ── CONFIG ────────────────────────────────────────────────────────
ROBOT_IP            = "192.168.8.102"
POSES_FILE          = BASE_DIR / "data/calibration_poses.json"
RESULT_FILE         = BASE_DIR / "calibration/T_cam2flange.npy"

CHARUCO_SQUARES_X   = 7
CHARUCO_SQUARES_Y   = 5
CHARUCO_SQUARE_SIZE = 0.0385   # metres — measure your printed board
CHARUCO_MARKER_SIZE = 0.028    # metres
CHARUCO_DICT        = cv2.aruco.DICT_6X6_250

# ── Camera display window size ────────────────────────────────────
WINDOW_W = 800  # pixels — change to make smaller/larger
WINDOW_H = 450

# ── Camera focus (OAK-D Lite) ─────────────────────────────────────
# Manual focus value: 0 (near) – 255 (far). 130 = ~0.5m, 150 = ~1m
# Locks focus so the camera doesn't hunt/blur during calibration.
MANUAL_FOCUS = 130   # adjust if board appears blurry

# ── ARGUMENT PARSER ───────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--pose", type=str, choices=["true", "false"], required=True,
                    help="true = auto-replay saved poses | false = manual jog mode")
args = parser.parse_args()
USE_SAVED_POSES = args.pose.lower() == "true"

# ── CHARUCO BOARD SETUP ───────────────────────────────────────────
dictionary       = cv2.aruco.getPredefinedDictionary(CHARUCO_DICT)
board            = cv2.aruco.CharucoBoard(
    (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
    CHARUCO_SQUARE_SIZE, CHARUCO_MARKER_SIZE, dictionary
)
charuco_detector = cv2.aruco.CharucoDetector(
    board, cv2.aruco.CharucoParameters(), cv2.aruco.DetectorParameters()
)

# ── CAMERA INTRINSICS ─────────────────────────────────────────────
K    = np.load(BASE_DIR / "calibration/camera_matrix.npy")
dist = np.zeros((4, 1))

# ── ROBOT CONNECTION ──────────────────────────────────────────────
print("Connecting to robot...")
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
print("Robot connected!\n")

# ── CAMERA PIPELINE (DepthAI API 2.0) ─────────────────────────────
pipeline = dai.Pipeline()
cam      = pipeline.create(dai.node.ColorCamera)
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setVideoSize(1280, 720)   # crop sensor output to 720p (matches collect_poses.py)
cam.setInterleaved(False)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
# Autofocus is used by default — same as collect_poses.py which works fine

xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("video")
cam.video.link(xout.input)

device     = dai.Device(pipeline)
videoQueue = device.getOutputQueue(name="video", maxSize=4, blocking=False)
print("Camera started! (focus locked)\n")

# Set up resizable window
WINDOW_NAME = "Hand-Eye Calibration"

# ── HELPERS ───────────────────────────────────────────────────────
def get_flange_matrix():
    pose = rtde_r.getActualTCPPose()
    R, _ = cv2.Rodrigues(np.array(pose[3:]))
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = pose[:3]
    return T, pose

def detect_charuco(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(grey)
    if charuco_ids is not None and len(charuco_ids) >= 4:
        cv2.aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids)
        valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners, charuco_ids, board, K, dist,
            np.zeros((3, 1)), np.zeros((3, 1))
        )
        if valid:
            cv2.drawFrameAxes(img, K, dist, rvec, tvec, CHARUCO_SQUARE_SIZE)
            R, _ = cv2.Rodrigues(rvec)
            T = np.eye(4); T[:3, :3] = R; T[:3, 3] = tvec.flatten()
            return T, img
    return None, img

# ── CALIBRATION DATA LISTS ────────────────────────────────────────
R_gripper2base_list = []
T_gripper2base_list = []
R_target2cam_list   = []
T_target2cam_list   = []
saved_poses         = []

# ══════════════════════════════════════════════════════════════════
# AUTO MODE — replay saved joint angles
# ══════════════════════════════════════════════════════════════════
if USE_SAVED_POSES:
    print(f"Loading saved poses from {POSES_FILE}...")
    with open(POSES_FILE, "r") as f:
        poses_data = json.load(f)
    print(f"Found {len(poses_data)} saved poses\n")

    # FPS tracking
    fps_smooth = 15.0
    prev_time  = time.time()

    for i, pose in enumerate(poses_data):
        print(f"Moving to pose {i+1}/{len(poses_data)}...")

        # ── Move robot — stream live video the whole time ──────────
        move_start = time.time()
        rtde_c.moveJ(pose["joint_angles"], 0.3, 0.3)   # blocking

        # Wait for robot to settle — show live feed during wait
        settle_start = time.time()
        while time.time() - settle_start < 1.5:
            pkt = videoQueue.tryGet()
            if pkt is not None:
                img = pkt.getCvFrame()

                # FPS
                now = time.time()
                dt  = now - prev_time;  prev_time = now
                if dt > 0:
                    fps_smooth = fps_smooth * 0.9 + (1.0 / dt) * 0.1

                cv2.putText(img, f"FPS: {fps_smooth:.1f}",
                            (img.shape[1] - 130, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)
                cv2.putText(img, f"Pose {i+1}/{len(poses_data)} — settling...",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)
                cv2.putText(img, "Live  |  Waiting for board detection",
                            (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                cv2.imshow(WINDOW_NAME, img)
                cv2.waitKey(1)

        # ── Search for board — stream live during search ───────────
        board_found  = False
        search_start = time.time()
        while time.time() - search_start < 3.0:   # up to 3s to find board
            pkt = videoQueue.tryGet()
            if pkt is not None:
                img = pkt.getCvFrame()
                T_board2cam, img = detect_charuco(img)

                now = time.time()
                dt  = now - prev_time;  prev_time = now
                if dt > 0:
                    fps_smooth = fps_smooth * 0.9 + (1.0 / dt) * 0.1

                status = "BOARD DETECTED ✓" if T_board2cam is not None else "Searching for board..."
                color  = (0, 255, 0) if T_board2cam is not None else (0, 100, 255)

                cv2.putText(img, f"FPS: {fps_smooth:.1f}",
                            (img.shape[1] - 130, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)
                cv2.putText(img, f"Pose {i+1}/{len(poses_data)} — {status}",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
                cv2.putText(img, f"Captured so far: {len(R_gripper2base_list)}",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.imshow(WINDOW_NAME, img)
                cv2.waitKey(1)

                if T_board2cam is not None:
                    flange_T, _ = get_flange_matrix()
                    R_gripper2base_list.append(flange_T[:3, :3])
                    T_gripper2base_list.append(flange_T[:3, 3])
                    R_target2cam_list.append(T_board2cam[:3, :3])
                    T_target2cam_list.append(T_board2cam[:3, 3])
                    board_found = True
                    print(f"  Pose {i+1} captured! ✓  ({len(R_gripper2base_list)} total)")
                    break
            else:
                time.sleep(0.01)

        if not board_found:
            print(f"  WARNING: Board not detected at pose {i+1}, skipping...")

# ══════════════════════════════════════════════════════════════════
# MANUAL MODE — jog robot, press SPACE in window to capture
# ══════════════════════════════════════════════════════════════════
else:
    print("MANUAL MODE")
    print("=" * 55)
    print("  1. Jog robot using the teach pendant")
    print("  2. Make sure ChArUco board is visible in camera window")
    print("  3. Click the camera window, then press SPACE to capture")
    print("  4. Aim for 20+ varied poses")
    print("  5. Press ENTER in the window when done (>=15 poses)")
    print("=" * 55)

    pose_count = 0
    flash_msg   = ""
    flash_until = 0.0

    while True:
        pkt = videoQueue.tryGet()
        img_to_show = None

        if pkt is not None:
            img = pkt.getCvFrame()
            T_board2cam, img = detect_charuco(img)

            status = "BOARD OK ✓ — press SPACE to capture" if T_board2cam is not None else "Move board into view"
            color  = (0, 255, 0) if T_board2cam is not None else (0, 80, 255)

            # HUD
            cv2.putText(img, f"Poses captured: {pose_count}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            cv2.putText(img, status,
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            need = max(0, 15 - pose_count)
            hint = f"Need {need} more" if need > 0 else "Ready! Press ENTER to finish"
            cv2.putText(img, hint,
                        (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
            cv2.putText(img, "SPACE = capture  |  ENTER = finish  |  Q = quit",
                        (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)

            # Flash message overlay
            now = time.time()
            if now < flash_until:
                cv2.rectangle(img, (8, 120), (500, 155), (0, 0, 0), -1)
                cv2.putText(img, flash_msg, (15, 147),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            img_to_show = img

        if img_to_show is not None:
            cv2.imshow(WINDOW_NAME, cv2.resize(img_to_show, (960, 540)))

        key = cv2.waitKey(1) & 0xFF

        # ── SPACE — capture ────────────────────────────────────────
        if key == ord(' '):
            # Re-detect on a fresh frame for accuracy
            fresh = videoQueue.tryGet()
            if fresh is not None:
                img = fresh.getCvFrame()
            T_board2cam, _ = detect_charuco(img)

            if T_board2cam is None:
                flash_msg   = "❌ Board not detected! Reposition and try again."
                flash_until = time.time() + 2.0
                print("Board not detected! Make sure it is fully visible.")
            else:
                flange_T, flange_pose = get_flange_matrix()
                R_gripper2base_list.append(flange_T[:3, :3])
                T_gripper2base_list.append(flange_T[:3, 3])
                R_target2cam_list.append(T_board2cam[:3, :3])
                T_target2cam_list.append(T_board2cam[:3, 3])

                joint_angles = list(rtde_r.getActualQ())
                saved_poses.append({
                    "pose_number": pose_count + 1,
                    "joint_angles": joint_angles,
                    "tcp_pose": list(flange_pose)
                })

                pose_count += 1
                flash_msg   = f"✓ Pose {pose_count} captured! ({max(0, 15-pose_count)} more needed)"
                flash_until = time.time() + 1.5
                print(f"Pose {pose_count} captured! ✓")

        # ── ENTER — finish ─────────────────────────────────────────
        elif key == 13:   # 13 = Enter
            if pose_count < 15:
                flash_msg   = f"Need {15 - pose_count} more poses before finishing!"
                flash_until = time.time() + 2.0
                print(f"Only {pose_count} poses. Need at least 15.")
            else:
                print(f"\nDone! {pose_count} poses captured.")
                break

        # ── Q — quit ───────────────────────────────────────────────
        elif key == ord('q'):
            print("Quit — no calibration run.")
            device.close()
            cv2.destroyAllWindows()
            rtde_c.stopScript()
            exit()

# ── RUN CALIBRATION ───────────────────────────────────────────────
print("\nRunning hand-eye calibration (Tsai-Lenz method)...")

if len(R_gripper2base_list) < 4:
    print(f"Only {len(R_gripper2base_list)} valid poses — need at least 4. Exiting.")
    device.close()
    cv2.destroyAllWindows()
    rtde_c.stopScript()
    exit()

R_cam2gripper, T_cam2gripper = cv2.calibrateHandEye(
    R_gripper2base_list, T_gripper2base_list,
    R_target2cam_list,   T_target2cam_list,
    method=cv2.CALIB_HAND_EYE_TSAI
)

T_cam2flange = np.eye(4)
T_cam2flange[:3, :3] = R_cam2gripper
T_cam2flange[:3, 3]  = T_cam2gripper.flatten()

print("\nCalibration result — T_cam2flange:")
print(T_cam2flange)

np.save(str(RESULT_FILE), T_cam2flange)
print(f"\nSaved to {RESULT_FILE}")

if saved_poses:
    with open(POSES_FILE, "w") as f:
        json.dump(saved_poses, f, indent=2)
    print(f"Saved {len(saved_poses)} poses to {POSES_FILE}")

device.close()
cv2.destroyAllWindows()
rtde_c.stopScript()
print("\nCalibration complete!")
