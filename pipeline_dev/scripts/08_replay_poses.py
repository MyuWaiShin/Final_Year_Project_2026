import rtde_receive
import rtde_control
import json
import time

ROBOT_IP   = "192.168.8.102"
POSES_FILE = "data/calibration_poses.json"
SPEED      = 0.3   # rad/s  - adjust if too fast or slow
ACCEL      = 0.3   # rad/s²

print(f"Connecting to robot at {ROBOT_IP}...")
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
print("Connected!\n")

# Load poses
with open(POSES_FILE, "r") as f:
    poses = json.load(f)
print(f"Loaded {len(poses)} poses from {POSES_FILE}\n")

print("WARNING: The robot will move through all poses one by one.")
print("Make sure the workspace is clear.")
print("Press Enter to start, or Ctrl+C to cancel.")
input()

for i, pose in enumerate(poses):
    tcp = pose["tcp_pose"]
    print(f"Moving to pose {i+1}/{len(poses)} (ID {pose['id']})...")

    # moveL moves in a straight line in cartesian space
    # tcp_pose is [x, y, z, rx, ry, rz]
    rtde_c.moveL(tcp, 0.1, 0.1)  # slow speed for safety

    # Wait at each pose so you can observe it
    print(f"  At pose {i+1}. Press Enter for next pose, or type 's' to skip to end.")
    user_input = input("  > ").strip().lower()
    if user_input == 's':
        print("Skipping remaining poses...")
        break

rtde_c.stopScript()
print("\nDone! All poses replayed.")
print("Did you like the poses? If yes, the calibration poses file is ready to use.")