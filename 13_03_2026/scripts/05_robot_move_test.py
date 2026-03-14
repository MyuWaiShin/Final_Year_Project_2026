import rtde_receive
import rtde_control
import time

ROBOT_IP = "192.168.8.102"

print(f"Connecting to UR10 at {ROBOT_IP}...")
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
print("Connected!\n")

# Read current joint angles
current_joints = rtde_r.getActualQ()
print("Current joint angles:")
for i, angle in enumerate(current_joints):
    print(f"  Joint {i+1}: {angle:.4f} rad")

# We will only move Joint 1 (the base) by 5 degrees
# 5 degrees in radians = 0.0873
import math
five_degrees = math.radians(5)

# New target = same as current but base rotated 5 degrees
target_joints = list(current_joints)
target_joints[0] = current_joints[0] + five_degrees

print(f"\nMoving base joint by 5 degrees...")
print(f"From: {math.degrees(current_joints[0]):.2f} degrees")
print(f"To:   {math.degrees(target_joints[0]):.2f} degrees")
print("Moving in 3 seconds... press Ctrl+C to cancel")
time.sleep(3)

# moveJ moves the robot by joint angles
# Speed: 0.1 rad/s (very slow)
# Acceleration: 0.1 rad/s² (very gentle)
rtde_c.moveJ(target_joints, 0.1, 0.1)
print("Move done!\n")

time.sleep(1)

# Move back to original position
print("Moving back to original position...")
rtde_c.moveJ(list(current_joints), 0.1, 0.1)
print("Back to start!\n")

rtde_c.stopScript()
print("Done.")

