import depthai as dai
import numpy as np

print("Reading camera intrinsics from OAK-D Lite...")

with dai.Device() as device:
    calib = device.readCalibration()
    
    # Get the intrinsics for the colour camera at 1280x720
    intrinsics = calib.getCameraIntrinsics(
        dai.CameraBoardSocket.CAM_A, 1280, 720
    )
    
    K = np.array(intrinsics)
    print("\nCamera Matrix (K):")
    print(K)
    
    # Save to file for use in other scripts
    np.save("calibration/camera_matrix.npy", K)
    print("\nSaved to calibration/camera_matrix.npy")

