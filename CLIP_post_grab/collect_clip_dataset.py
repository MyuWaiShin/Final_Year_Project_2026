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

import cv2

os.environ.setdefault("DEPTHAI_BOOT_TIMEOUT", "30000")

try:
    import depthai as dai
except ImportError:  # depthai is only needed when using the OAK-D
    dai = None  # type: ignore[assignment]

try:
    # Uses the same UR10Controller you already use for gripper_with_detection.
    from perception.ur10_control.gripper_with_detection import UR10Controller
except Exception as e:  # pragma: no cover - import error only at runtime
    UR10Controller = None  # type: ignore[assignment]
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


WINDOW_NAME = "CLIP Gripper Dataset Collection"


def init_output_dirs(
    root: Optional[str],
) -> tuple[Path, Path, Path, Path, "csv.TextIOWrapper", csv.DictWriter]:
    script_dir = Path(__file__).parent
    base = Path(root).expanduser().resolve() if root else (script_dir / "clip_dataset").resolve()

    holding_dir = base / "holding"
    empty_dir = base / "empty"
    unknown_dir = base / "unknown"

    for d in (holding_dir, empty_dir, unknown_dir):
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

    return base, holding_dir, empty_dir, unknown_dir, meta_file, writer


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

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("video")
    cam_rgb.video.link(xout.input)

    config = dai.Device.Config()
    config.board.usb.maxSpeed = dai.UsbSpeed.HIGH

    device = dai.Device(config)
    device.startPipeline(pipeline)

    q_rgb = device.getOutputQueue("video", maxSize=4, blocking=False)
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
        holding_dir,
        empty_dir,
        unknown_dir,
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
    print("  's' : save current frame using latest grasp label")
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

            # Resize display for manageable window size (640x360)
            display = cv2.resize(frame, (640, 360))
            
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

            elif key == ord("s"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                label = last_label or "unknown"

                if label == "holding":
                    out_dir = holding_dir
                elif label == "empty":
                    out_dir = empty_dir
                else:
                    out_dir = unknown_dir

                filename = f"{label}_{timestamp}.png"
                save_path = out_dir / filename
                cv2.imwrite(str(save_path), frame)

                writer.writerow(
                    {
                        "timestamp": timestamp,
                        "filename": str(save_path.relative_to(base_dir)),
                        "label": label,
                        "gripper_width_mm": last_width if last_width is not None else "",
                        "voltage_ai2": f"{last_voltage:.4f}" if last_voltage is not None else "",
                        "force_detected": last_force if last_force is not None else "",
                        "object_detected": last_object if last_object is not None else "",
                    }
                )
                meta_file.flush()
                print(f"Saved 1080p frame -> {save_path} (label={label})")

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

