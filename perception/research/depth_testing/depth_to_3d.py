"""
depth_to_3d.py
==============
Demonstrates how to convert a depth measurement into real-world (X, Y, Z)
coordinates, which is what you need for robot pick-and-place.

This script:
1. Gets calibration from camera
2. For each detected bounding box centre, computes (X, Y, Z) in metres
3. Shows a live view with 3D coordinates printed on each object
4. Demonstrates the maths clearly

For pick-and-place you'd feed these (X, Y, Z) values to your robot IK solver.
"""

import os
os.environ["DEPTHAI_BOOT_TIMEOUT"] = "30000"

import depthai as dai
import cv2
import numpy as np
import time


# ────────────────────────────────────────────────────────────────────
# THE KEY MATHS - read this first!
# ────────────────────────────────────────────────────────────────────
#
#  A camera maps 3-D world points to 2-D pixels using the pinhole model:
#
#       u  =  fx * (X/Z) + cx
#       v  =  fy * (Y/Z) + cy
#
#  Rearranging to go from pixel (u,v) + depth Z back to 3-D:
#
#       X  =  (u - cx) * Z / fx      ← left/right  (+ = right)
#       Y  =  (v - cy) * Z / fy      ← up/down     (+ = down)
#       Z  =  depth                  ← forward      (+ = away from cam)
#
#  Where:
#    fx, fy  = focal lengths in pixels  (from calibration)
#    cx, cy  = principal point (image centre, usually ≈ W/2, H/2)
#    depth Z = what the OAK-D stereo pair measures (in mm, convert to m)
#
# ────────────────────────────────────────────────────────────────────

def pixel_to_3d(u, v, depth_m, fx, fy, cx, cy):
    """Back-project pixel (u,v) with depth (m) into camera-frame XYZ (m)."""
    X = (u - cx) * depth_m / fx
    Y = (v - cy) * depth_m / fy
    Z = depth_m
    return X, Y, Z


def get_roi_depth(depth_frame, x1, y1, x2, y2, percentile=25):
    """
    Get a robust depth estimate for a bounding box region.
    Uses low percentile to get closest (least occluded) pixels.
    Returns depth in mm.
    """
    roi = depth_frame[y1:y2, x1:x2]
    valid = roi[roi > 100]   # ignore zero/invalid pixels
    if valid.size == 0:
        return 0
    return float(np.percentile(valid, percentile))


def build_pipeline():
    pipeline = dai.Pipeline()
    pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_4)

    # RGB
    rgb = pipeline.create(dai.node.ColorCamera)
    rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    rgb.setPreviewSize(640, 480)
    rgb.setInterleaved(False)
    rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    rgb.setFps(30)

    # Mono
    mono_left  = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    # Stereo
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    # Outputs
    for name, src in [("rgb", rgb.preview), ("depth", stereo.depth)]:
        xout = pipeline.create(dai.node.XLinkOut)
        xout.setStreamName(name)
        src.link(xout.input)

    return pipeline


def main():
    pipeline = build_pipeline()
    config = dai.Device.Config()
    config.board.usb.maxSpeed = dai.UsbSpeed.HIGH

    print("Connecting to camera …")
    with dai.Device(config) as device:
        device.startPipeline(pipeline)

        # Read intrinsics at preview resolution 640×480
        cal = device.readCalibration()
        M   = cal.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 640, 480)
        fx, fy = M[0][0], M[1][1]
        cx, cy = M[0][2], M[1][2]

        print(f"✓ Connected: {device.getMxId()}")
        print(f"  Intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}\n")

        q_rgb   = device.getOutputQueue("rgb",   maxSize=4, blocking=False)
        q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)

        depth_frame = None
        rgb_frame   = None
        click_boxes = []   # list of (x1,y1,x2,y2) drawn by user

        # ── Mouse: draw a bounding box ROI to probe ──
        drawing  = False
        pt_start = (0, 0)

        def on_mouse(event, x, y, flags, _):
            nonlocal drawing, pt_start, click_boxes
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing  = True
                pt_start = (x, y)
            elif event == cv2.EVENT_LBUTTONUP and drawing:
                drawing = False
                x1, y1 = min(pt_start[0], x), min(pt_start[1], y)
                x2, y2 = max(pt_start[0], x), max(pt_start[1], y)
                if x2 - x1 > 5 and y2 - y1 > 5:
                    click_boxes.append((x1, y1, x2, y2))

        cv2.namedWindow("Depth → 3D")
        cv2.setMouseCallback("Depth → 3D", on_mouse)

        print("Draw a bounding box around an object to get its 3D position.")
        print("Press 'c' to clear boxes, 'q' to quit.\n")

        while True:
            if q_rgb.has():
                rgb_frame = q_rgb.get().getCvFrame()
            if q_depth.has():
                depth_frame = q_depth.get().getFrame()

            if rgb_frame is None or depth_frame is None:
                time.sleep(0.005)
                continue

            display = rgb_frame.copy()

            # Draw existing boxes + compute 3D
            for (x1, y1, x2, y2) in click_boxes:
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 200, 50), 2)

                depth_mm = get_roi_depth(depth_frame, x1, y1, x2, y2)
                if depth_mm > 0:
                    # Centre of box
                    uc = (x1 + x2) // 2
                    vc = (y1 + y2) // 2

                    X, Y, Z = pixel_to_3d(uc, vc, depth_mm / 1000.0, fx, fy, cx, cy)

                    label = f"Z={Z:.3f}m"
                    coord = f"X={X:+.3f} Y={Y:+.3f}"
                    cv2.putText(display, label, (x1, y1 - 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 80), 1)
                    cv2.putText(display, coord, (x1, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 80), 1)
                    # Cross-hair at centre
                    cv2.drawMarker(display, (uc, vc), (0, 255, 80),
                                   cv2.MARKER_CROSS, 14, 1)

            cv2.putText(display, "Draw box around object | c=clear | q=quit",
                        (5, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.imshow("Depth → 3D", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                click_boxes.clear()

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
