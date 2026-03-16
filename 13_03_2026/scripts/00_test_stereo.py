"""
test_stereo_depth.py
---------------------
Quick test to see what stereo depth returns from the OAK-D Lite.
Shows RGB + depth side by side with depth values overlaid.
Press Q to quit.
"""

import cv2
import depthai as dai
import numpy as np

print("Starting camera with stereo depth...")
pipeline    = dai.Pipeline()

# RGB
cam         = pipeline.create(dai.node.Camera).build()
videoOutput = cam.requestOutput((1280, 720), type=dai.ImgFrame.Type.BGR888p)
videoQueue  = videoOutput.createOutputQueue()

# Stereo
stereo      = pipeline.create(dai.node.StereoDepth).build(autoCreateCameras=True)
depthQueue  = stereo.depth.createOutputQueue()

pipeline.start()
print("Started! Move objects in front of the camera.")
print("Press Q to quit.\n")

while True:
    imgFrame   = videoQueue.get()
    frame      = imgFrame.getCvFrame()

    depthFrame = depthQueue.tryGet()

    if depthFrame is not None:
        depth = depthFrame.getFrame()  # uint16, millimetres

        # Stats
        valid = depth[depth > 0]
        if valid.size > 0:
            min_d  = valid.min()
            max_d  = valid.max()
            med_d  = np.median(valid)
            centre = depth[depth.shape[0]//2, depth.shape[1]//2]
            print(f"  min={min_d}mm  max={max_d}mm  median={med_d:.0f}mm  centre={centre}mm")

        # Colorise depth for display
        depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_col = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        # Resize depth to match RGB for display
        depth_col = cv2.resize(depth_col, (frame.shape[1], frame.shape[0]))

        # Centre crosshair on both
        h, w = frame.shape[:2]
        cv2.drawMarker(frame,     (w//2, h//2), (0, 255, 255), cv2.MARKER_CROSS, 40, 2)
        cv2.drawMarker(depth_col, (w//2, h//2), (255, 255, 255), cv2.MARKER_CROSS, 40, 2)

        if depthFrame is not None and valid.size > 0:
            centre_d = depth[depth.shape[0]//2, depth.shape[1]//2]
            cv2.putText(frame, f"centre depth: {centre_d}mm", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        combined = np.hstack([frame, depth_col])
        cv2.imshow("RGB | Depth", combined)
    else:
        cv2.putText(frame, "No depth frame yet...", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("RGB | Depth", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
pipeline.stop()
print("Done.")