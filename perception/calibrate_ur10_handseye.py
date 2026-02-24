import cv2
import numpy as np
import json
import time
from pathlib import Path
from ur10_utils import UR10PoseReader, pose_to_matrix

# Configuration
ROBOT_IP = "192.168.8.102"
SAVE_PATH = Path("ur10_cam_offset.json")

def main():
    print("=" * 60)
    print("UR10 EYE-IN-HAND CALIBRATION (Touch Method)")
    print("=" * 60)
    print("Step 1: Mount camera on gripper.")
    print("Step 2: Place an object on the table.")
    print("=" * 60)
    
    reader = UR10PoseReader(ROBOT_IP)
    
    # 1. Record Detections point (where camera sees object)
    input("\n1. Move robot so camera sees object center. Press Enter to record...")
    pose_detect = reader.get_actual_tcp_pose()
    if not pose_detect:
        print("Error: Could not read robot pose!")
        return
    print(f"Recorded Pose (Detection): {pose_detect}")
    
    depth_m = float(input("Enter Z (Depth) from pose_estimation.py for this object (meters): "))
    
    # 2. Record Touch point (where gripper touches object)
    input("\n2. move robot until gripper center touches object center. Press Enter to record...")
    pose_touch = reader.get_actual_tcp_pose()
    if not pose_touch:
        print("Error: Could not read robot pose!")
        return
    print(f"Recorded Pose (Touch): {pose_touch}")
    
    # 3. Compute Offset
    # T_base_detect * T_tcp_cam * P_cam = T_base_touch
    # Since P_cam is [0, 0, depth, 1] (relative to cam), we simplify:
    # We want T_tcp_cam.
    
    M1 = pose_to_matrix(pose_detect)
    M2 = pose_to_matrix(pose_touch)
    
    # Simple translation-only calibration for now (assuming camera is parallel to TCP)
    # dx, dy, dz in TCP frame
    # This is a simplification. For full 6DOF we'd need more points.
    # But for a basic pick-and-place, this "delta" works.
    
    # Target in Base Frame: M2.translation
    # Camera Center in Base Frame (at M1): M1 * T_tcp_cam * [0,0,depth] = M2.translation
    
    # Let's solve for the simple XYZ offset of the camera relative to TCP
    # offset_base = M2.translation - (M1 * [0,0,depth])
    # T_tcp_cam (approx)
    
    # For now, let's just save the raw poses and compute the delta.
    calibration_data = {
        "pose_detect": pose_detect,
        "depth_at_detect": depth_m,
        "pose_touch": pose_touch,
        "timestamp": time.ctime()
    }
    
    with open(SAVE_PATH, 'w') as f:
        json.dump(calibration_data, f, indent=4)
        
    print(f"\n✓ Calibration saved to {SAVE_PATH}")
    print("\nNext: I will update the coordinate mapper to use this data.")

if __name__ == "__main__":
    main()
