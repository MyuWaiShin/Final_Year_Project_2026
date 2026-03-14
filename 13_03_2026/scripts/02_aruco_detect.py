import depthai as dai
import cv2
import numpy as np

MARKER_SIZE = 0.03  # 30mm in metres
DICTIONARY  = cv2.aruco.DICT_6X6_50

# Load camera intrinsics we saved earlier
K    = np.load("calibration/camera_matrix.npy")
dist = np.zeros((4, 1))  # OAK-D Lite already corrects distortion internally

# Set up ArUco detector
dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY)
detector   = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

# Set up camera pipeline
pipeline    = dai.Pipeline()
cam         = pipeline.create(dai.node.Camera).build()
videoOutput = cam.requestOutput((1280, 720), type=dai.ImgFrame.Type.BGR888p)
videoQueue  = videoOutput.createOutputQueue()

print("Starting camera...")
print("Hold your ArUco tag in front of the camera")
print("Press Q to quit\n")
pipeline.start()

while pipeline.isRunning():
    frame = videoQueue.tryGet()
    if frame is not None:
        img  = frame.getCvFrame()
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = detector.detectMarkers(grey)

        if ids is not None:
            # Draw box around tag
            cv2.aruco.drawDetectedMarkers(img, corners, ids)

            # Estimate the 3D pose of the tag
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, MARKER_SIZE, K, dist
            )

            for i in range(len(ids)):
                tvec = tvecs[i][0]  # X, Y, Z in metres from camera
                
                # Draw the axes on the tag so we can see orientation
                cv2.drawFrameAxes(img, K, dist, rvecs[i], tvecs[i], MARKER_SIZE)

                # Print position
                print(f"Tag ID {ids[i][0]} | "
                      f"X: {tvec[0]:.3f}m  "
                      f"Y: {tvec[1]:.3f}m  "
                      f"Z: {tvec[2]:.3f}m (distance from camera)")
        else:
            print("No tag detected...")

        cv2.imshow("ArUco Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("Stopped.")

