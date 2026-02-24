"""
calibration_check.py
====================
Read and display the factory calibration stored inside the OAK-D Lite.

Outputs:
  - Intrinsics per camera (fx, fy, cx, cy)
  - Distortion coefficients
  - Stereo baseline (mm)
  - Extrinsics between left/right cameras
  - Saves calibration to  calibration_data.npz  for later use
"""

import os
os.environ["DEPTHAI_BOOT_TIMEOUT"] = "30000"

import depthai as dai
import numpy as np

def main():
    config = dai.Device.Config()
    config.board.usb.maxSpeed = dai.UsbSpeed.HIGH

    print("Connecting to camera …")
    with dai.Device(config) as device:
        print(f"✓ Connected: {device.getMxId()}\n")

        calib = device.readCalibration()

        # ── helper ───────────────────────────────────
        def get_intrinsics(socket, w, h, name):
            M = calib.getCameraIntrinsics(socket, w, h)
            D = calib.getDistortionCoefficients(socket)
            fx, fy = M[0][0], M[1][1]
            cx, cy = M[0][2], M[1][2]
            print(f"{'─'*55}")
            print(f"{name}  intrinsics  @ {w}×{h}")
            print(f"{'─'*55}")
            print(f"  fx = {fx:.4f}  fy = {fy:.4f}")
            print(f"  cx = {cx:.4f}  cy = {cy:.4f}")
            print(f"  Distortion (k1,k2,p1,p2,k3): "
                  f"{[round(d,6) for d in D[:5]]}")
            return np.array(M), np.array(D), fx, fy, cx, cy

        # RGB
        M_rgb, D_rgb, *_ = get_intrinsics(
            dai.CameraBoardSocket.CAM_A, 1920, 1080, "RGB (CAM_A) 1920×1080")
        M_rgb_640, D_rgb_640, fx, fy, cx, cy = get_intrinsics(
            dai.CameraBoardSocket.CAM_A, 640, 480,  "RGB (CAM_A)  640×480")

        # Left mono
        M_left, D_left, *_ = get_intrinsics(
            dai.CameraBoardSocket.CAM_B, 640, 400,  "Left  (CAM_B) 640×400")

        # Right mono
        M_right, D_right, *_ = get_intrinsics(
            dai.CameraBoardSocket.CAM_C, 640, 400,  "Right (CAM_C) 640×400")

        # ── Stereo baseline ──────────────────────────
        print(f"\n{'─'*55}")
        print("Stereo baseline")
        print(f"{'─'*55}")
        T = calib.getCameraTranslationVector(
            dai.CameraBoardSocket.CAM_B,
            dai.CameraBoardSocket.CAM_C)
        baseline_mm = abs(T[0]) * 10   # DepthAI returns cm → mm
        print(f"  Baseline: {baseline_mm:.2f} mm  ({baseline_mm/10:.2f} cm)")
        print(f"  Translation vector (cm): {[round(t,4) for t in T]}")

        # ── Extrinsics left→right ────────────────────
        R = calib.getCameraExtrinsics(
            dai.CameraBoardSocket.CAM_B,
            dai.CameraBoardSocket.CAM_C)
        print(f"\n  Rotation matrix (Left → Right):")
        for row in R:
            print(f"    {[round(v,6) for v in row]}")

        # ── Depth range from baseline & focal length ─
        print(f"\n{'─'*55}")
        print("Estimated depth range (left@640×400)")
        print(f"{'─'*55}")
        fl = M_left[0][0]   # focal length in pixels
        # OAK-D Lite disparity range: ~4 to 96 pixels
        min_depth = (baseline_mm * fl) / 96
        max_depth = (baseline_mm * fl) / 4
        print(f"  Focal length (left fx): {fl:.1f} px")
        print(f"  Estimated min depth: {min_depth:.0f} mm  ({min_depth/1000:.2f} m)")
        print(f"  Estimated max depth: {max_depth:.0f} mm  ({max_depth/1000:.2f} m)")
        print(f"  Practical range: 200 mm – 10,000 mm")

        # ── Save for use in other scripts ────────────
        np.savez("calibration_data.npz",
                 M_rgb=M_rgb,      D_rgb=D_rgb,
                 M_rgb_640=M_rgb_640, D_rgb_640=D_rgb_640,
                 M_left=M_left,   D_left=D_left,
                 M_right=M_right, D_right=D_right,
                 baseline_mm=np.array([baseline_mm]),
                 fx_640=np.array([fx]), fy_640=np.array([fy]),
                 cx_640=np.array([cx]), cy_640=np.array([cy]))

        print(f"\n✓ Calibration saved to  calibration_data.npz")
        print("  Load it in any script with:")
        print("    cal = np.load('calibration_data.npz')")
        print("    fx, fy = cal['fx_640'][0], cal['fy_640'][0]")
        print("    cx, cy = cal['cx_640'][0], cal['cy_640'][0]")

if __name__ == "__main__":
    main()
