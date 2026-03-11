"""
floor_depth_monitor.py
=======================
Interactive floor depth monitor — jog the robot downward slowly while watching
this terminal. The OAK-D stereo depth sensor measures the floor in real time
and warns you before the gripper gets within the 2 mm safety clearance.

Extended Disparity is enabled so the camera can measure depths as close as
~35 cm (vs. ~70 cm in standard mode), which is important when the arm is
nearly fully extended.

HOW TO USE
----------
1.  Position the robot so the camera is roughly 50–70 cm above the table.
    (This is the optimal range for OAK-D stereo depth accuracy.)
2.  Run this script:
        python UR10/floor_depth_monitor.py
3.  Use the pendant to jog the robot downward slowly.
4.  Watch the `Clearance` readout. When it turns RED, stop.
5.  Press  F  to freeze-record the current floor Z as the reference.
6.  Press  Q  to quit.

Keys (terminal, no CV2 window needed):
  F  — freeze/record current depth floor reading
  Q  — quit

Outputs a printed record of the floor Z in robot base frame, ready to paste
into failure_detection_pipeline.py or safety_guard.py if you want a hardcoded
backup.
"""

import os
os.environ["DEPTHAI_BOOT_TIMEOUT"] = "30000"

import json
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
ROBOT_IP          = "192.168.8.102"
LIMITS_PATH       = Path(__file__).parent.parent / "Perception" / "safe_limits.json"
CALIB_PATH        = Path(__file__).parent.parent / "Perception" / "ur10_cam_offset.json"

FLOOR_CLEARANCE_M = 0.002   # 2 mm — stop the robot this far above floor
WARNING_M         = 0.020   # 20 mm — turn yellow before the hard stop

# Depth patch size (px) — centre of image is sampled
PATCH_PX   = 50
# Percentile of depth patch — low percentile = closest surface (the floor)
PERCENTILE = 5

# ANSI colours
GREEN  = "\033[92m"; YELLOW = "\033[93m"; RED = "\033[91m"
BOLD   = "\033[1m";  RESET  = "\033[0m";  CLEAR = "\033[2J\033[H"


# ─────────────────────────────────────────────────────────────────────────────
# Robot pose reader
# ─────────────────────────────────────────────────────────────────────────────
def get_tcp_pose(ip=ROBOT_IP):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            s.connect((ip, 30003))
            data = s.recv(1200)
            if len(data) < 492:
                return None
            return list(struct.unpack("!6d", data[444:492]))
    except Exception:
        return None


def rotvec_to_matrix(rx, ry, rz):
    vec   = np.array([rx, ry, rz], dtype=float)
    angle = np.linalg.norm(vec)
    if angle < 1e-9:
        return np.eye(3)
    axis = vec / angle
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


# ─────────────────────────────────────────────────────────────────────────────
# OAK-D pipeline — RGB + stereo depth with EXTENDED DISPARITY
# ─────────────────────────────────────────────────────────────────────────────
def create_pipeline():
    p = dai.Pipeline()

    cam = p.create(dai.node.ColorCamera)
    cam.setPreviewSize(640, 480)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(30)

    ml = p.create(dai.node.MonoCamera)
    mr = p.create(dai.node.MonoCamera)
    st = p.create(dai.node.StereoDepth)

    ml.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    ml.setBoardSocket(dai.CameraBoardSocket.LEFT)
    mr.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mr.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    st.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    st.setDepthAlign(dai.CameraBoardSocket.RGB)
    # Extended disparity: min reliable depth ~35 cm instead of ~70 cm
    st.initialConfig.setExtendedDisparity(True)
    st.setSubpixel(True)    # sub-pixel disparity for lower noise near floor

    ml.out.link(st.left)
    mr.out.link(st.right)

    xrgb   = p.create(dai.node.XLinkOut); xrgb.setStreamName("rgb")
    xdep   = p.create(dai.node.XLinkOut); xdep.setStreamName("depth")
    cam.preview.link(xrgb.input)
    st.depth.link(xdep.input)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Load camera-to-TCP calibration if available
    R_cam_tcp = np.eye(3)
    t_cam_tcp = np.zeros(3)
    if CALIB_PATH.exists():
        with open(CALIB_PATH) as f:
            cal = json.load(f)
        t_cam_tcp = np.array(cal["t_cam_in_tcp"])
        if "R_cam_in_tcp" in cal:
            R_cam_tcp = np.array(cal["R_cam_in_tcp"])
        print(f"[Calib] Loaded camera-to-TCP offset: "
              f"dx={t_cam_tcp[0]*1000:.1f}  dy={t_cam_tcp[1]*1000:.1f}  "
              f"dz={t_cam_tcp[2]*1000:.1f} mm")
    else:
        print("[Calib] ur10_cam_offset.json not found — floor Z will use raw "
              "camera depth (no rotation correction). Results may be less accurate.")

    # Load static safe limits for reference
    static_z_min = None
    if LIMITS_PATH.exists():
        with open(LIMITS_PATH) as f:
            data = json.load(f)
        static_z_min = data["safe_limits"]["SAFE_Z"][0]
        print(f"[Limits] Static Z min from safe_limits.json: {static_z_min*1000:.1f} mm")

    print(f"\nOpening OAK-D camera with extended disparity (min depth ~35 cm)...")

    # Keyboard input thread
    frozen_floor_z = [None]
    quit_flag      = [False]

    def key_reader():
        try:
            import msvcrt
            while not quit_flag[0]:
                if msvcrt.kbhit():
                    k = msvcrt.getch().decode('utf-8', errors='ignore').lower()
                    if k == 'q':
                        quit_flag[0] = True
                    elif k == 'f':
                        frozen_floor_z[0] = "FREEZE_NOW"
        except ImportError:
            pass   # non-Windows — Q/F via Ctrl-C only

    kt = threading.Thread(target=key_reader, daemon=True)
    kt.start()

    cfg = dai.Device.Config()
    with dai.Device(cfg) as device:
        device.startPipeline(create_pipeline())
        q_rgb   = device.getOutputQueue("rgb",   maxSize=4, blocking=False)
        q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)

        # Intrinsics
        cal  = device.readCalibration()
        M, _, _ = cal.getDefaultIntrinsics(dai.CameraBoardSocket.RGB)
        fx, fy, cx0, cy0 = M[0][0], M[1][1], M[0][2], M[1][2]
        print(f"[Camera] fx={fx:.1f}  fy={fy:.1f}  cx={cx0:.1f}  cy={cy0:.1f}")
        print(f"\n{'='*60}")
        print(f"  Jog the robot DOWN slowly with the pendant.")
        print(f"  F = freeze/record floor Z    Q = quit")
        print(f"{'='*60}\n")

        recorded_floor_zs = []

        while not quit_flag[0]:
            frame = q_rgb.get().getCvFrame()
            depth = q_depth.get().getFrame()   # uint16, mm

            h, w = depth.shape

            # Sample central patch
            cx_px = w // 2;  cy_px = h // 2
            half  = PATCH_PX // 2
            patch = depth[max(0, cy_px-half):cy_px+half,
                          max(0, cx_px-half):cx_px+half]
            good  = patch[patch > 0]

            if good.size == 0:
                floor_depth_mm = None
            else:
                floor_depth_mm = float(np.percentile(good, PERCENTILE))

            # Get robot TCP
            tcp = get_tcp_pose()

            # Convert floor depth to robot base frame Z
            floor_z_base = None
            clearance    = None

            if floor_depth_mm is not None and tcp is not None:
                Z_cam = floor_depth_mm / 1000.0   # metres
                # Back-project centre pixel to camera-centred XYZ
                X_cam = (cx_px - cx0) * Z_cam / fx
                Y_cam = (cy_px - cy0) * Z_cam / fy
                P_cam = np.array([X_cam, Y_cam, Z_cam])

                # Camera → TCP frame
                P_tcp = R_cam_tcp @ P_cam + t_cam_tcp

                # TCP → robot base frame
                R_tcp = rotvec_to_matrix(*tcp[3:])
                P_base = R_tcp @ P_tcp + np.array(tcp[:3])

                floor_z_base = float(P_base[2])
                clearance    = tcp[2] - floor_z_base - FLOOR_CLEARANCE_M

            # Handle freeze request
            if frozen_floor_z[0] == "FREEZE_NOW" and floor_z_base is not None:
                recorded_floor_zs.append(floor_z_base)
                frozen_floor_z[0] = floor_z_base
                print(f"\n  ★ Floor Z RECORDED: {floor_z_base*1000:.1f} mm (robot base frame)")

            # Build display
            lines = [
                f"{BOLD}═══ Floor Depth Monitor ═══{RESET}   {time.strftime('%H:%M:%S')}",
                "",
            ]

            if tcp is not None:
                tcp_z_mm = tcp[2] * 1000
                lines.append(f"  Robot TCP Z      : {tcp_z_mm:+8.1f} mm")
            else:
                lines.append(f"  Robot TCP Z      : {RED}not connected{RESET}")

            if floor_depth_mm is not None:
                lines.append(f"  Depth to floor   : {floor_depth_mm:8.1f} mm  "
                             f"({PERCENTILE}th percentile, {good.size} valid px)")
            else:
                lines.append(f"  Depth to floor   : {RED}no reading{RESET}")

            if floor_z_base is not None:
                lines.append(f"  Floor Z (base)   : {floor_z_base*1000:+8.1f} mm")
            else:
                lines.append(f"  Floor Z (base)   : {RED}N/A{RESET}")

            if clearance is not None:
                if clearance < 0:
                    col = RED;    sym = "⚠  VIOLATION"
                elif clearance < WARNING_M:
                    col = YELLOW; sym = "⚠  WARNING"
                else:
                    col = GREEN;  sym = "✓  OK"
                lines.append(f"  Clearance        : {col}{BOLD}{clearance*1000:+8.1f} mm   {sym}{RESET}")
            else:
                lines.append(f"  Clearance        : {RED}N/A{RESET}")

            if static_z_min is not None:
                lines.append(f"  Static Z min     : {static_z_min*1000:+8.1f} mm  (from safe_limits.json)")

            lines.append("")
            if recorded_floor_zs:
                avg = sum(recorded_floor_zs) / len(recorded_floor_zs)
                lines.append(f"  {GREEN}Recorded floor readings ({len(recorded_floor_zs)}):  "
                             f"avg = {avg*1000:.1f} mm{RESET}")
                lines.append(f"  → Paste into safety_guard / pipeline: "
                             f"FLOOR_Z = {avg:.4f}  # metres")
            else:
                lines.append(f"  Press F to record this floor Z reading.")

            lines.append(f"\n  F = record floor Z   Q = quit")

            print(CLEAR + "\n".join(lines), end="", flush=True)

            # Draw depth patch indicator on frame
            cv2.rectangle(frame,
                          (cx_px - half, cy_px - half),
                          (cx_px + half, cy_px + half),
                          (0, 255, 255), 2)
            cv2.putText(frame, f"Floor depth: {floor_depth_mm:.0f}mm" if floor_depth_mm else "no depth",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Floor Depth Monitor", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                quit_flag[0] = True
            elif k == ord('f') and floor_z_base is not None:
                recorded_floor_zs.append(floor_z_base)
                frozen_floor_z[0] = floor_z_base
                print(f"\n  ★ Floor Z RECORDED: {floor_z_base*1000:.1f} mm (robot base frame)")

    cv2.destroyAllWindows()

    if recorded_floor_zs:
        avg = sum(recorded_floor_zs) / len(recorded_floor_zs)
        print(f"\n{'='*60}")
        print(f"  FINAL FLOOR Z (average of {len(recorded_floor_zs)} recordings)")
        print(f"  {avg*1000:.1f} mm in robot base frame")
        print(f"  Safe pick limit = floor_z + 2mm = {(avg + 0.002)*1000:.1f} mm")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
