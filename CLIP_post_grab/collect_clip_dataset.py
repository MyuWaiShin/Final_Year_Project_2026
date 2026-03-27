#!/usr/bin/env python3
"""
Interactive CLIP dataset collection for eye-in-hand gripper images.

Keyboard controls (terminal window must be focused):
  - 'o': open gripper (loads and plays grip_open.urp)
  - 'c': close gripper (loads and plays grip_close.urp, then reads grasp state)
  - 's': save current camera frame into class folder based on grasp state
  - 'q': quit

Frames are saved under a subfolder of this script's directory:
  CLIP_post_grab/clip_dataset/
      holding/
      empty/
      unknown/

Label decision on save:
  - If a recent close command was executed and UR I/O reports an object
    via UR10Controller.is_object_detected(), label = "holding".
  - If a recent close command was executed and no object is detected,
    label = "empty".
  - Otherwise frames are stored under "unknown" (e.g. during motion / open jaw).
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import time
from typing import Optional
import os
import sys

# Add project root to sys.path to resolve module imports like 'Perception'
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from datetime import datetime
from pathlib import Path
import time
from typing import Optional
import os

import cv2

os.environ.setdefault("DEPTHAI_BOOT_TIMEOUT", "30000")

try:
    import depthai as dai
except ImportError:  # depthai is only needed when using the OAK-D
    dai = None  # type: ignore[assignment]

try:
    # Uses the same UR10Controller you already use for gripper_with_detection.
    from UR10.gripper_with_detection import UR10Controller
except Exception as e:  # pragma: no cover - import error only at runtime
    UR10Controller = None  # type: ignore[assignment]
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


WINDOW_NAME = "CLIP Gripper Dataset Collection"


def init_output_dirs(
    root: Optional[str],
) -> tuple[Path, dict, dict, "csv.TextIOWrapper", csv.DictWriter]:
    script_dir = Path(__file__).parent
    base = Path(root).expanduser().resolve() if root else (script_dir / "clip_dataset").resolve()

    crop_dirs = {
        "holding": base / "cropped" / "holding",
        "empty": base / "cropped" / "empty",
        "unknown": base / "cropped" / "unknown",
    }
    
    full_dirs = {
        "holding": base / "full" / "holding",
        "empty": base / "full" / "empty",
        "unknown": base / "full" / "unknown",
    }

    for d in list(crop_dirs.values()) + list(full_dirs.values()):
        d.mkdir(parents=True, exist_ok=True)

    meta_path = base / "metadata.csv"
    is_new = not meta_path.exists()
    meta_file = meta_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        meta_file,
        fieldnames=[
            "timestamp",
            "filename",
            "label",
            "gripper_width_mm",
            "voltage_ai2",
            "force_detected",
            "object_detected",
        ],
    )
    if is_new:
        writer.writeheader()
        meta_file.flush()

    return base, crop_dirs, full_dirs, meta_file, writer


def connect_robot(robot_ip: str) -> Optional[UR10Controller]:
    if UR10Controller is None:
        print("Warning: could not import UR10Controller from perception.ur10_control.gripper_with_detection.")
        print(f"  Import error was: {_IMPORT_ERROR}")
        print("  Robot controls ('o'/'c') will be disabled; only 's' (save) and 'q' will work.")
        return None

    try:
        controller = UR10Controller(robot_ip)
        time.sleep(1.5)
        return controller
    except Exception as e:  # pragma: no cover - hardware dependent
        print(f"Warning: failed to connect to UR10 at {robot_ip}: {e}")
        print("  Robot controls ('o'/'c') will be disabled; only 's' (save) and 'q' will work.")
        return None


def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera at index {index}")
    return cap


def open_oak_camera() -> tuple["dai.Device", "dai.DataOutputQueue"]:
    """
    Open OAK-D Lite using DepthAI and return (device, rgb_queue).
    """
    if dai is None:
        raise SystemExit(
            "depthai is not installed. Install with:\n  pip install depthai"
        )

    pipeline = dai.Pipeline()
    pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_4)

    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setVideoSize(1920, 1080)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(30)

    # 1. Camera Control input
    controlIn = pipeline.create(dai.node.XLinkIn)
    controlIn.setStreamName("control")
    controlIn.out.link(cam_rgb.inputControl)

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("video")
    cam_rgb.video.link(xout.input)

    config = dai.Device.Config()
    config.board.usb.maxSpeed = dai.UsbSpeed.HIGH

    device = dai.Device(config)
    device.startPipeline(pipeline)

    q_rgb = device.getOutputQueue("video", maxSize=4, blocking=False)
    q_control = device.getInputQueue("control")
    
    # Enable continuous auto-focus to keep the gripper sharp as it moves
    ctrl = dai.CameraControl()
    ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
    q_control.send(ctrl)
    print("→ Camera Auto-Focus Enabled.")

    return device, q_rgb


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect gripper images for CLIP fine-tuning (holding vs empty)."
    )
    parser.add_argument(
        "--robot-ip",
        type=str,
        default="192.168.8.102",
        help="UR10 robot IP address (for gripper open/close and grasp feedback).",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index if using a webcam (default: 0).",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["oak", "webcam"],
        default="oak",
        help="Camera source: 'oak' (OAK-D Lite via DepthAI) or 'webcam' (cv2.VideoCapture). Default: oak.",
    )
    # Pose definitions (in meters and radians - rotation vector [rx, ry, rz])
    parser.add_argument(
        "--view-pose",
        type=float,
        nargs=6,
        default=[-0.430, -0.650, 0.400, 3.14, 0.0, 0.0],
        help="Target pose to LIFT and VIEW for clear focus [x, y, z, rx, ry, rz].",
    )
    parser.add_argument(
        "--home-pose",
        type=float,
        nargs=6,
        default=[-0.430, -0.650, 0.200, 3.14, 0.0, 0.0],
        help="Target pose to return to the table [x, y, z, rx, ry, rz].",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Optional output root directory. Default: CLIP_post_grab/clip_dataset",
    )
    args = parser.parse_args()

    (
        base_dir,
        crop_dirs,
        full_dirs,
        meta_file,
        writer,
    ) = init_output_dirs(args.output_root)

    controller = connect_robot(args.robot_ip)

    device = None
    q_rgb = None
    cap = None

    if args.source == "oak":
        print("Using OAK-D (DepthAI) as camera source.")
        device, q_rgb = open_oak_camera()
    else:
        print(f"Using webcam (index {args.camera_index}) as camera source.")
        cap = open_camera(args.camera_index)

    last_label: Optional[str] = None
    last_width: Optional[float] = None
    last_voltage: Optional[float] = None
    last_force: Optional[bool] = None
    last_object: Optional[bool] = None

    print("=" * 70)
    print("CLIP Gripper Dataset Collection")
    print("=" * 70)
    print(f"Saving images under: {base_dir}")
    print("\nControls (focus terminal window for key presses):")
    print("  'o' : open gripper (grip_open.urp)")
    print("  'c' : close gripper (grip_close.urp) and update grasp state")
    print("  'v' : move to VIEW POSE (lift for focus)")
    print("  'h' : move to HOME POSE (return to table)")
    print("  's' : save current frame using sensor's auto-label (green text)")
    print("  '1' : FORCE save as 'holding' (overrides sensor)")
    print("  '0' : FORCE save as 'empty' (overrides sensor)")
    print("  'q' : quit")
    print("-" * 70)

    try:
        while True:
            if args.source == "oak":
                assert q_rgb is not None
                in_rgb = q_rgb.get()
                frame = in_rgb.getCvFrame()
            else:
                assert cap is not None
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame from webcam.")
                    break

            # ROI Cropping: A wide rectangle at the bottom to catch the open gripper
            # Based on the user image: it needs to span almost the whole bottom width
            h, w = frame.shape[:2]
            
            crop_w = 1400  # Wide enough to catch both open fingers
            crop_h = 600   # Tall enough to see the object being grasped, but not too high
            
            # Start the crop at the very bottom edge to ensure we see the tips and base
            cx = w // 2
            cy = h - (crop_h // 2) - 10  
            
            x1 = max(0, cx - crop_w // 2)
            y1 = max(0, cy - crop_h // 2)
            x2 = min(w, x1 + crop_w)
            y2 = min(h, y1 + crop_h)
            
            # Extract the crop for saving
            crop_frame = frame[y1:y2, x1:x2].copy()

            # For display, we show both the full frame (scaled) and the crop
            display = cv2.resize(frame, (640, 360))
            # Draw the ROI box on the display
            scale_x, scale_y = 640/w, 360/h
            cv2.rectangle(display, 
                          (int(x1*scale_x), int(y1*scale_y)), 
                          (int(x2*scale_x), int(y2*scale_y)), 
                          (255, 0, 0), 2)
            
            label_text = last_label if last_label is not None else "unknown"
            status = f"label={label_text}"
            if last_width is not None:
                status += f" | width={last_width:.1f}mm"
            if last_object is not None:
                status += f" | object={last_object}"

            cv2.putText(
                display,
                status,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            
            # Show the actual CROP in a small dedicated window
            # Scale down the preview window (maintaining the wide aspect ratio)
            preview_w = 600
            preview_h = int(preview_w * (crop_h / crop_w))
            display_crop = cv2.resize(crop_frame, (preview_w, preview_h))
            cv2.imshow(f"CLIP CROP ({crop_w}x{crop_h})", display_crop)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Quitting.")
                break

            if key == ord("o"):
                if controller is None:
                    print("Robot not connected; 'o' ignored.")
                    continue
                print("→ Opening gripper...")
                controller.load_program("grip_open.urp")
                time.sleep(0.3)
                controller.play()
                time.sleep(2.5)
                last_width = controller.get_width_mm()
                last_voltage = controller.latest_analog_in2
                last_force = controller.is_force_detected()
                last_object = False
                last_label = "empty"
                print(f"Gripper opened. Width={last_width} mm")

            elif key == ord("c"):
                if controller is None:
                    print("Robot not connected; 'c' ignored.")
                    continue
                print("→ Closing gripper...")
                controller.load_program("grip_close.urp")
                time.sleep(0.3)
                controller.play()
                time.sleep(2.5)

                last_width = controller.get_width_mm()
                last_voltage = controller.latest_analog_in2
                last_force = controller.is_force_detected()
                last_object = controller.is_object_detected()
                last_label = "holding" if last_object else "empty"
                print(
                    f"Close complete. Width={last_width} mm, "
                    f"force={last_force}, object_detected={last_object} -> label='{last_label}'"
                )

            elif key in [ord("s"), ord("1"), ord("0")]:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                
                # Determine label based on key press
                if key == ord("1"):
                    label = "holding"
                elif key == ord("0"):
                    label = "empty"
                else:
                    label = last_label or "unknown"

                crop_dir = crop_dirs.get(label, crop_dirs["unknown"])
                full_dir = full_dirs.get(label, full_dirs["unknown"])

                filename = f"{label}_{timestamp}.png"
                crop_path = crop_dir / filename
                full_path = full_dir / filename
                
                cv2.imwrite(str(crop_path), crop_frame)
                cv2.imwrite(str(full_path), frame)

                writer.writerow(
                    {
                        "timestamp": timestamp,
                        "filename": str(crop_path.relative_to(base_dir)),
                        "label": label,
                        "gripper_width_mm": last_width if last_width is not None else "",
                        "voltage_ai2": f"{last_voltage:.4f}" if last_voltage is not None else "",
                        "force_detected": last_force if last_force is not None else "",
                        "object_detected": last_object if last_object is not None else "",
                    }
                )
                meta_file.flush()
                print(f"Saved crop -> {crop_path} and full -> {full_path} (label={label})")

            elif key == ord("v"):
                if controller is None:
                    print("Robot not connected; 'v' ignored.")
                    continue
                print(f"→ Moving to VIEW POSE: {args.view_pose}")
                controller.movel(args.view_pose, a=0.5, v=0.2)
                # wait is handled by user visually

            elif key == ord("h"):
                if controller is None:
                    print("Robot not connected; 'h' ignored.")
                    continue
                print(f"→ Moving to HOME POSE: {args.home_pose}")
                controller.movel(args.home_pose, a=0.5, v=0.2)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass

        cv2.destroyAllWindows()
        try:
            if controller is not None:
                controller.close()
        except Exception:
            pass
        try:
            if device is not None:
                device.close()
        except Exception:
            pass
        try:
            meta_file.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

