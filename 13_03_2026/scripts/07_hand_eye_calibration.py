import depthai as dai
import cv2
import numpy as np
import rtde_receive
import rtde_control
import json
import argparse
import time

# ── CONFIG ────────────────────────────────────────────────────────
ROBOT_IP        = "192.168.8.102"
POSES_FILE      = "data/calibration_poses.json"
RESULT_FILE     = "calibration/T_cam2flange.npy"

# ChArUco board config - we will generate this board next
CHARUCO_SQUARES_X  = 7
CHARUCO_SQUARES_Y  = 5
CHARUCO_SQUARE_SIZE = 0.038   # 40mm per square - adjust after printing
CHARUCO_MARKER_SIZE = 0.028   # 30mm marker - adjust after printing
CHARUCO_DICT       = cv2.aruco.DICT_6X6_250  # same dict as your tag

# ── ARGUMENT PARSER ───────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--pose",
    type=str,
    choices=["true", "false"],
    required=True,
    help="true = use saved poses file | false = manually jog and capture"
)
args = parser.parse_args()
USE_SAVED_POSES = args.pose.lower() == "true"

# ── CHARUCO BOARD SETUP ───────────────────────────────────────────
dictionary   = cv2.aruco.getPredefinedDictionary(CHARUCO_DICT)
board        = cv2.aruco.CharucoBoard(
    (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
    CHARUCO_SQUARE_SIZE,
    CHARUCO_MARKER_SIZE,
    dictionary
)
detector_params = cv2.aruco.DetectorParameters()
charuco_params  = cv2.aruco.CharucoParameters()
charuco_detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)

# ── CAMERA INTRINSICS ─────────────────────────────────────────────
K    = np.load("calibration/camera_matrix.npy")
dist = np.zeros((4, 1))

# ── ROBOT CONNECTION ──────────────────────────────────────────────
print("Connecting to robot...")
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
print("Robot connected!\n")

# ── CAMERA PIPELINE ───────────────────────────────────────────────
pipeline    = dai.Pipeline()
cam         = pipeline.create(dai.node.Camera).build()
videoOutput = cam.requestOutput((1280, 720), type=dai.ImgFrame.Type.BGR888p)
videoQueue  = videoOutput.createOutputQueue()
pipeline.start()
print("Camera started!\n")

# ── HELPER: get current flange pose as 4x4 matrix ────────────────
def get_flange_matrix():
    pose = rtde_r.getActualTCPPose()
    R, _ = cv2.Rodrigues(np.array(pose[3:]))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = pose[:3]
    return T, pose

# ── HELPER: detect ChArUco and get pose ──────────────────────────
def detect_charuco(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(grey)

    if charuco_ids is not None and len(charuco_ids) >= 4:
        # Draw detected board
        cv2.aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids)

        # Estimate pose
        valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners, charuco_ids, board, K, dist,
            np.zeros((3,1)), np.zeros((3,1))
        )
        if valid:
            cv2.drawFrameAxes(img, K, dist, rvec, tvec, CHARUCO_SQUARE_SIZE)
            R, _ = cv2.Rodrigues(rvec)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3]  = tvec.flatten()
            return T, img
    return None, img

# ── STORAGE ───────────────────────────────────────────────────────
R_gripper2base_list = []
T_gripper2base_list = []
R_target2cam_list   = []
T_target2cam_list   = []
saved_poses         = []

# ── MAIN LOOP ─────────────────────────────────────────────────────
if USE_SAVED_POSES:
    # Load saved poses and move robot to each one automatically
    print(f"Loading saved poses from {POSES_FILE}...")
    with open(POSES_FILE, "r") as f:
        poses_data = json.load(f)
    print(f"Found {len(poses_data)} saved poses\n")

    for i, pose in enumerate(poses_data):
        print(f"Moving to pose {i+1}/{len(poses_data)}...")
        rtde_c.moveJ(pose["joint_angles"], 0.3, 0.3)
        time.sleep(1.5)  # wait for robot to settle

        # Capture frames and look for board
        board_found = False
        attempts = 0
        while not board_found and attempts < 30:
            frame = videoQueue.tryGet()
            if frame is not None:
                img = frame.getCvFrame()
                T_board2cam, img = detect_charuco(img)

                status = "BOARD DETECTED" if T_board2cam is not None else "searching..."
                cv2.putText(img, f"Pose {i+1}/{len(poses_data)} - {status}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.imshow("Hand-Eye Calibration", img)
                cv2.waitKey(1)

                if T_board2cam is not None:
                    board_found = True
                    flange_T, _ = get_flange_matrix()
                    R_gripper2base_list.append(flange_T[:3, :3])
                    T_gripper2base_list.append(flange_T[:3, 3])
                    R_target2cam_list.append(T_board2cam[:3, :3])
                    T_target2cam_list.append(T_board2cam[:3, 3])
                    print(f"  Pose {i+1} captured! ✓")
            attempts += 1
            time.sleep(0.1)

        if not board_found:
            print(f"  WARNING: Board not detected at pose {i+1}, skipping...")

else:
    # Manual mode - jog robot and press Enter to capture
    print("MANUAL MODE")
    print("=" * 50)
    print("Instructions:")
    print("  1. Jog the robot to a position using the teach pendant")
    print("  2. Make sure the ChArUco board is visible in the camera window")
    print("  3. Press ENTER in this terminal to capture the pose")
    print("  4. Repeat for at least 15 poses (20 recommended)")
    print("  5. Type 'done' when finished")
    print("=" * 50)
    print()

    pose_count = 0
    while True:
        # Show camera feed continuously
        frame = videoQueue.tryGet()
        if frame is not None:
            img = frame.getCvFrame()
            T_board2cam, img = detect_charuco(img)

            status = "BOARD DETECTED ✓" if T_board2cam is not None else "No board detected"
            color  = (0, 255, 0) if T_board2cam is not None else (0, 0, 255)
            cv2.putText(img, f"Poses captured: {pose_count} | {status}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(img, "Switch to terminal and press ENTER to capture",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            cv2.imshow("Hand-Eye Calibration", img)
            cv2.waitKey(1)

        # Non-blocking input check
        import msvcrt
        if msvcrt.kbhit():
            key = msvcrt.getwch()
            if key == '\r':  # Enter pressed
                # Capture current frame properly
                time.sleep(0.1)
                frame = videoQueue.tryGet()
                if frame is not None:
                    img = frame.getCvFrame()
                    T_board2cam, img = detect_charuco(img)

                    if T_board2cam is None:
                        print("Board not detected! Make sure the ChArUco board is fully visible, try again.")
                        continue

                    flange_T, flange_pose = get_flange_matrix()
                    R_gripper2base_list.append(flange_T[:3, :3])
                    T_gripper2base_list.append(flange_T[:3, 3])
                    R_target2cam_list.append(T_board2cam[:3, :3])
                    T_target2cam_list.append(T_board2cam[:3, 3])

                    # Save pose for future reuse
                    joint_angles = list(rtde_r.getActualQ())
                    saved_poses.append({
                        "pose_number": pose_count + 1,
                        "joint_angles": joint_angles,
                        "tcp_pose": list(flange_pose)
                    })

                    pose_count += 1
                    print(f"Pose {pose_count} captured! ✓ (need at least 15)")

            elif key == 'd':  # type 'd' to finish
                if pose_count < 15:
                    print(f"Only {pose_count} poses captured. Need at least 15. Keep going!")
                else:
                    print(f"\nFinishing with {pose_count} poses...")
                    break

# ── RUN CALIBRATION ───────────────────────────────────────────────
print("\nRunning hand-eye calibration...")

if len(R_gripper2base_list) < 4:
    print("Not enough poses captured. Need at least 15. Exiting.")
    exit()

R_cam2gripper, T_cam2gripper = cv2.calibrateHandEye(
    R_gripper2base_list, T_gripper2base_list,
    R_target2cam_list,   T_target2cam_list,
    method=cv2.CALIB_HAND_EYE_TSAI
)

T_cam2flange = np.eye(4)
T_cam2flange[:3, :3] = R_cam2gripper
T_cam2flange[:3, 3]  = T_cam2gripper.flatten()

print("\nCalibration result - T_cam2flange:")
print(T_cam2flange)

# Save result
np.save(RESULT_FILE, T_cam2flange)
print(f"\nSaved to {RESULT_FILE}")

# Save poses for reuse
if not USE_SAVED_POSES and saved_poses:
    with open(POSES_FILE, "w") as f:
        json.dump(saved_poses, f, indent=2)
    print(f"Saved {len(saved_poses)} poses to {POSES_FILE} for future use")

cv2.destroyAllWindows()
rtde_c.stopScript()
print("\nCalibration complete!")
