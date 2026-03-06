import cv2
import cv2.aruco as aruco
import numpy as np

ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
CHARUCO_BRD = aruco.CharucoBoard((7, 5), 0.029, 0.022, ARUCO_DICT)

# Old way:
# corners, ids, rejected = aruco.detectMarkers(image, ARUCO_DICT)
# new way:
detector_params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(ARUCO_DICT, detector_params)

# For charuco
charuco_detector = aruco.CharucoDetector(CHARUCO_BRD)

print("ArucoDetector available!")
print("CharucoDetector available!")

# Generate a dummy image to test detection
img = CHARUCO_BRD.generateImage((500, 500))

# Test detectBoard
charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(img)
print(f"Detected charuco corners: {len(charuco_corners) if charuco_corners is not None else 0}")
