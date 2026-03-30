import rtde_receive

ROBOT_IP = "192.168.8.102"

print(f"Connecting to UR10 at {ROBOT_IP}...")

try:
    rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    print("Connected successfully!\n")

    # Read the current TCP position
    # This gives us [x, y, z, rx, ry, rz]
    # x, y, z = position in metres
    # rx, ry, rz = rotation (in radians)
    tcp_pose = rtde_r.getActualTCPPose()
    print("Current TCP position:")
    print(f"  X:  {tcp_pose[0]:.4f} m")
    print(f"  Y:  {tcp_pose[1]:.4f} m")
    print(f"  Z:  {tcp_pose[2]:.4f} m")
    print(f"  RX: {tcp_pose[3]:.4f} rad")
    print(f"  RY: {tcp_pose[4]:.4f} rad")
    print(f"  RZ: {tcp_pose[5]:.4f} rad")

    # Read the joint angles
    joint_angles = rtde_r.getActualQ()
    print("\nCurrent joint angles (radians):")
    for i, angle in enumerate(joint_angles):
        print(f"  Joint {i+1}: {angle:.4f} rad")

    print("\nRobot is ready!")

except Exception as e:
    print(f"Failed to connect: {e}")