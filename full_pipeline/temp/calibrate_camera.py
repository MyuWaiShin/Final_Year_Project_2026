"""
calibrate_camera.py
-------------------
Reads the OAK-D Lite's factory-calibrated intrinsic matrix and saves it
to full_pipeline/calibration/camera_matrix.npy.

Run this once (or after re-flashing the camera) before running
calibrate_handeye.py.

Usage
-----
    python full_pipeline/temp/calibrate_camera.py
"""

from pathlib import Path

import depthai as dai
import numpy as np

# ── Output path ────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
CALIB_DIR  = SCRIPT_DIR.parent / "calibration"
CALIB_DIR.mkdir(exist_ok=True)

OUT_PATH = CALIB_DIR / "camera_matrix.npy"

# Resolution that the pipeline uses for all camera frames
CAM_W = 1280
CAM_H = 720

# ── Read and save ──────────────────────────────────────────────────────────
print("Connecting to OAK-D …")
with dai.Device() as device:
    calib      = device.readCalibration()
    intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, CAM_W, CAM_H)
    K          = np.array(intrinsics)

print("\nCamera matrix K:")
print(K)
print(f"\n  fx = {K[0,0]:.2f}  fy = {K[1,1]:.2f}")
print(f"  cx = {K[0,2]:.2f}  cy = {K[1,2]:.2f}")

np.save(str(OUT_PATH), K)
print(f"\nSaved → {OUT_PATH}")
print("Done.")
