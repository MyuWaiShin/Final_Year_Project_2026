import rtde_receive

ROBOT_IP = "192.168.8.102"
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

tests = []

while True:
    label = input("\nEnter a label for this reading (or 'done' to finish): ")
    if label.lower() == 'done':
        break
    
    tcp = rtde_r.getActualTCPPose()
    tests.append((label, tcp))
    print(f"Recorded: X={tcp[0]:.4f} Y={tcp[1]:.4f} Z={tcp[2]:.4f} RX={tcp[3]:.4f} RY={tcp[4]:.4f} RZ={tcp[5]:.4f}")

print("\n\n=== SUMMARY ===")
for label, tcp in tests:
    print(f"\n{label}:")
    print(f"  X={tcp[0]:.4f}  Y={tcp[1]:.4f}  Z={tcp[2]:.4f}")
    print(f"  RX={tcp[3]:.4f}  RY={tcp[4]:.4f}  RZ={tcp[5]:.4f}")

