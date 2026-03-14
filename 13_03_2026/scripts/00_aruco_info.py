"""
00_aruco_info.py
-----------------
Runs for 5 seconds, tries every common ArUco dictionary,
detects any markers in view and prints: ID, dictionary, distance from camera.
Point the camera at your floor tag and let it run.
"""

import cv2
import cv2.aruco as aruco
import depthai as dai
import numpy as np
import time

# ── Camera intrinsics ──────────────────────────────────────────────
K    = np.load("calibration/camera_matrix.npy")
dist = np.zeros((4, 1))

MARKER_SIZE = 0.03  # metres — just a guess for distance estimate, doesn't need to be exact

# ── All dicts to try ───────────────────────────────────────────────
DICT_NAMES = [
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_4X4_250",
    "DICT_5X5_50",
    "DICT_5X5_100",
    "DICT_5X5_250",
    "DICT_6X6_50",
    "DICT_6X6_100",
    "DICT_6X6_250",
    "DICT_7X7_50",
    "DICT_7X7_100",
    "DICT_7X7_250",
]

detectors = {}
for name in DICT_NAMES:
    d   = aruco.getPredefinedDictionary(getattr(aruco, name))
    det = aruco.ArucoDetector(d, aruco.DetectorParameters())
    detectors[name] = det

# ── Camera ─────────────────────────────────────────────────────────
print("Starting camera... will run for 5 seconds.")
print("Point the camera at your floor tag now.\n")

pipeline    = dai.Pipeline()
cam         = pipeline.create(dai.node.Camera).build()
videoOutput = cam.requestOutput((1280, 720), type=dai.ImgFrame.Type.BGR888p)
videoQueue  = videoOutput.createOutputQueue()
pipeline.start()

start = time.time()
found = {}   # dict_name -> {marker_id -> dist_m}

while time.time() - start < 5.0:
    imgFrame = videoQueue.get()
    frame    = imgFrame.getCvFrame()
    grey     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for dict_name, det in detectors.items():
        corners, ids, _ = det.detectMarkers(grey)
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if dict_name not in found:
                    found[dict_name] = {}
                if marker_id not in found[dict_name]:
                    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                        corners[i:i+1], MARKER_SIZE, K, dist
                    )
                    dist_m = np.linalg.norm(tvecs[0][0])
                    found[dict_name][marker_id] = dist_m

    remaining = 5.0 - (time.time() - start)
    cv2.putText(frame, f"Scanning... {remaining:.1f}s", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.imshow("ArUco Info", frame)
    cv2.waitKey(1)

cv2.destroyAllWindows()
pipeline.stop()

# ── Print results ──────────────────────────────────────────────────
print("\n" + "=" * 50)
print("RESULTS")
print("=" * 50)

if not found:
    print("No markers detected! Make sure the tag is clearly visible and well lit.")
else:
    for dict_name, markers in found.items():
        for marker_id, dist_m in markers.items():
            print(f"  Dictionary : {dict_name}")
            print(f"  Marker ID  : {marker_id}")
            print(f"  Distance   : {dist_m:.3f}m (approx, based on {MARKER_SIZE*100:.0f}cm size guess)")
            print()

print("Use the dictionary and ID above in your scripts.")
print("Then physically measure your marker and update MARKER_SIZE.")