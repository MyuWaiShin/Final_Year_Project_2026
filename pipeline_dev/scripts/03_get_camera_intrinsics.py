import os
from pathlib import Path

import depthai as dai
import numpy as np

# ── Robust path setup ─────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR   = SCRIPT_DIR.parent
CALIB_DIR  = BASE_DIR / "calibration"
CALIB_DIR.mkdir(exist_ok=True)   # create if missing

print("Reading camera intrinsics from OAK-D Lite...")

with dai.Device() as device:
    calib = device.readCalibration()
    intrinsics = calib.getCameraIntrinsics(
        dai.CameraBoardSocket.CAM_A, 1280, 720
    )
    K = np.array(intrinsics)
    print("\nCamera Matrix (K):")
    print(K)

    out_path = CALIB_DIR / "camera_matrix.npy"
    np.save(str(out_path), K)
    print(f"\nSaved to {out_path}")
