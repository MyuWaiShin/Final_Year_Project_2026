"""
depth_explorer.py  (v2 — fixed noise, filtering, UI)
=====================================================
Fixes from v1:
  - Filters out invalid pixels (0 and 65535)
  - Adds spatial + temporal noise filtering
  - Turns off subpixel (causes noise on low-texture scenes)
  - Separate windows: RGB + depth side-by-side, panels separate
  - Fixed histogram (proper range, no 65535 pollution)
  - Confidence shown as colour heat map, not raw grey

Controls:
  Left-click        → print depth (mm) + 3D coords at pixel
  'h'               → toggle histogram window
  'c'               → toggle confidence window
  's'               → save frame
  '+' / '-'         → increase / decrease max display range
  'q'               → quit
"""

import os
os.environ["DEPTHAI_BOOT_TIMEOUT"] = "30000"

import depthai as dai
import cv2
import numpy as np
from pathlib import Path
import time

# ── Tunable constants ──────────────────────────────────
DISPLAY_MIN_MM = 200
DISPLAY_MAX_MM = 3000    # start with 3m, adjust with +/-
RANGE_STEP_MM  = 250     # how much +/- changes the range


# ── Pipeline ───────────────────────────────────────────
def build_pipeline():
    pipeline = dai.Pipeline()
    pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_4)

    # RGB
    rgb = pipeline.create(dai.node.ColorCamera)
    rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    rgb.setPreviewSize(640, 480)
    rgb.setInterleaved(False)
    rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    rgb.setFps(20)

    # Mono pair
    for socket, node_name in [(dai.CameraBoardSocket.CAM_B, "left"),
                               (dai.CameraBoardSocket.CAM_C, "right")]:
        mono = pipeline.create(dai.node.MonoCamera)
        mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono.setBoardSocket(socket)
        if node_name == "left":
            mono_left = mono
        else:
            mono_right = mono

    # Stereo — HIGH_DENSITY, NO subpixel (reduces noise on low-texture scenes)
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(False)
    stereo.setSubpixel(False)          # ← OFF: subpixel causes noise on plain surfaces

    # Spatial noise filter (fills holes, smooths depth)
    config = stereo.initialConfig.get()
    config.postProcessing.spatialFilter.enable = True
    config.postProcessing.spatialFilter.holeFillingRadius = 2
    config.postProcessing.spatialFilter.numIterations = 1
    # Temporal filter (averages over frames — reduces flickering)
    config.postProcessing.temporalFilter.enable = True
    stereo.initialConfig.set(config)

    # Median filter (removes salt-and-pepper spikes)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

    # Align depth to RGB frame
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    # Outputs
    for name, src in [
        ("rgb",        rgb.preview),
        ("depth",      stereo.depth),
        ("confidence", stereo.confidenceMap),
    ]:
        xout = pipeline.create(dai.node.XLinkOut)
        xout.setStreamName(name)
        src.link(xout.input)

    return pipeline


# ── Helpers ────────────────────────────────────────────
def filter_depth(depth_frame):
    """Replace invalid pixels (0 and 65535) with 0."""
    out = depth_frame.copy()
    out[out == 65535] = 0
    return out


def colorise_depth(depth_frame, min_mm, max_mm):
    """
    Convert uint16 depth (mm) → BGR colour image.
    Invalid (0) pixels are rendered black.
    """
    valid_mask = depth_frame > 0
    out = np.zeros_like(depth_frame, dtype=np.float32)
    out[valid_mask] = np.clip(depth_frame[valid_mask].astype(np.float32),
                               min_mm, max_mm)
    # Normalise only valid pixels
    out[valid_mask] = (out[valid_mask] - min_mm) / (max_mm - min_mm) * 255
    grey = out.astype(np.uint8)
    coloured = cv2.applyColorMap(grey, cv2.COLORMAP_TURBO)  # TURBO is cleaner than JET
    coloured[~valid_mask] = 0   # black for invalid
    return coloured


def pixel_to_3d(u, v, depth_mm, calib):
    Z = depth_mm / 1000.0
    X = (u - calib['cx']) * Z / calib['fx']
    Y = (v - calib['cy']) * Z / calib['fy']
    return X, Y, Z


# OAK-D Lite stereo constants (from calibration)
BASELINE_MM = 75.0
FOCAL_LEFT  = 452.7   # left mono fx at 640×400

def depth_resolution_mm(distance_mm):
    """
    Minimum detectable depth difference at a given distance.
    Formula: Δdepth = distance² / (focal_length × baseline)
    This is the fundamental limit of stereo — not a bug!
    """
    return (distance_mm ** 2) / (FOCAL_LEFT * BASELINE_MM)


def depth_accuracy_label(distance_mm):
    """Return a coloured (text, BGR) pair describing accuracy tier."""
    res = depth_resolution_mm(distance_mm)
    if res < 10:
        return f"Res~{res:.1f}mm  EXCELLENT", (0, 255, 80)
    elif res < 25:
        return f"Res~{res:.0f}mm  GOOD", (0, 220, 180)
    elif res < 60:
        return f"Res~{res:.0f}mm  OK (move closer)", (0, 200, 255)
    else:
        return f"Res~{res:.0f}mm  POOR — get within 80cm!", (0, 80, 255)


def draw_histogram(depth_frame, max_mm, width=400, height=180):
    """
    Draw depth histogram using only VALID pixels (>0, <65535, <max_mm+500).
    """
    canvas = np.zeros((height, width, 3), np.uint8)
    upper = min(max_mm + 500, 10000)
    valid = depth_frame[(depth_frame > 100) & (depth_frame < upper)].flatten()

    if valid.size == 0:
        cv2.putText(canvas, "No valid depth pixels", (10, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        return canvas

    hist, edges = np.histogram(valid, bins=60, range=(100, upper))
    bar_max = hist.max()
    if bar_max == 0:
        return canvas
    bar_w = max(1, width // 60)

    for i, h in enumerate(hist):
        if h == 0:
            continue
        bar_h = int((h / bar_max) * (height - 30))
        x0 = i * bar_w
        # Colour bars by distance (same TURBO palette)
        t = i / 60.0
        colour = cv2.applyColorMap(np.array([[[ int(t*255) ]]], np.uint8),
                                    cv2.COLORMAP_TURBO)[0,0].tolist()
        cv2.rectangle(canvas, (x0, height - 30 - bar_h), (x0 + bar_w - 1, height - 30),
                      colour, -1)

    # X axis labels
    cv2.line(canvas, (0, height-30), (width, height-30), (150,150,150), 1)
    cv2.putText(canvas, f"100mm", (2, height-14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1)
    cv2.putText(canvas, f"{upper}mm", (width-60, height-14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1)

    mean_mm = int(valid.mean())
    med_mm  = int(np.median(valid))
    cv2.putText(canvas, f"mean={mean_mm}mm  median={med_mm}mm  n={valid.size:,}",
                (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220,220,220), 1)
    cv2.putText(canvas, "Depth histogram (valid pixels only)",
                (4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160,160,160), 1)
    return canvas


def draw_confidence(conf_frame, width=320, height=240):
    """Render confidence as TURBO colourmap: red=bad, blue/green=good."""
    resized = cv2.resize(conf_frame, (width, height))
    # Confidence in depthai: LOW value = HIGH confidence (counter-intuitive!)
    # Invert so that bright = confident
    inverted = 255 - resized
    coloured = cv2.applyColorMap(inverted, cv2.COLORMAP_TURBO)
    cv2.putText(coloured, "Confidence  (green/blue=good, red=bad)",
                (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1)
    return coloured


# ── Main ───────────────────────────────────────────────
def main():
    global DISPLAY_MAX_MM
    save_dir = Path("depth_frames"); save_dir.mkdir(exist_ok=True)

    pipeline = build_pipeline()
    config   = dai.Device.Config()
    config.board.usb.maxSpeed = dai.UsbSpeed.HIGH

    print("Connecting …")
    with dai.Device(config) as device:
        device.startPipeline(pipeline)
        print(f"✓ {device.getMxId()}  USB:{device.getUsbSpeed().name}")

        cal  = device.readCalibration()
        M    = cal.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 640, 480)
        calib = dict(fx=M[0][0], fy=M[1][1], cx=M[0][2], cy=M[1][2])
        print(f"  fx={calib['fx']:.1f} fy={calib['fy']:.1f} "
              f"cx={calib['cx']:.1f} cy={calib['cy']:.1f}\n")

        q_rgb  = device.getOutputQueue("rgb",        maxSize=4, blocking=False)
        q_dep  = device.getOutputQueue("depth",      maxSize=4, blocking=False)
        q_conf = device.getOutputQueue("confidence", maxSize=4, blocking=False)

        depth_frame = rgb_frame = conf_frame = None
        show_hist   = True
        show_conf   = False
        click_info  = None

        def on_mouse(event, x, y, flags, _):
            nonlocal click_info, depth_frame
            if event == cv2.EVENT_LBUTTONDOWN and depth_frame is not None:
                # Click is on the RIGHT half (depth panel) — offset x by -640
                px = x - 640 if x >= 640 else x
                px = max(0, min(px, depth_frame.shape[1] - 1))
                d = int(depth_frame[y, px])
                if 100 < d < 20000:
                    X, Y, Z = pixel_to_3d(px, y, d, calib)
                    res = depth_resolution_mm(d)
                    click_info = (px, y, d, X, Y, Z)
                    print(f"  pixel=({px},{y})  depth={d}mm  "
                          f"XYZ=({X:+.3f},{Y:+.3f},{Z:.3f})m")
                    print(f"  ↳ Depth resolution at {d}mm: ~{res:.1f}mm "
                          f"(min detectable height difference)")
                    if res > 50:
                        print(f"  ⚠  Too far away! Move camera within ~80cm "
                              f"for accurate small-object detection.")
                else:
                    click_info = (x if x < 640 else x - 640, y, 0, 0, 0, 0)
                    print(f"  ({x},{y})  depth=INVALID (no stereo match here)")

        cv2.namedWindow("Depth Explorer", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Depth Explorer", 1280, 480)
        cv2.setMouseCallback("Depth Explorer", on_mouse)

        print("Controls: click=probe  h=histogram  c=confidence  +/-=range  s=save  q=quit")

        while True:
            if q_rgb.has():  rgb_frame   = q_rgb.get().getCvFrame()
            if q_dep.has():  depth_frame = filter_depth(q_dep.get().getFrame())
            if q_conf.has(): conf_frame  = q_conf.get().getFrame()

            if rgb_frame is None or depth_frame is None:
                time.sleep(0.005); continue

            # ── Build depth colour view ──────────────
            depth_vis = colorise_depth(depth_frame, DISPLAY_MIN_MM, DISPLAY_MAX_MM)
            depth_vis = cv2.resize(depth_vis, (640, 480))

            # ── Stats overlay (valid pixels only) ───
            valid_px = depth_frame[(depth_frame > 100) & (depth_frame < 20000)]
            if valid_px.size:
                med = int(np.median(valid_px))
                cv2.putText(depth_vis,
                    f"Min:{valid_px.min()}  Med:{med}  "
                    f"Max:{valid_px.max()}mm",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

                # ── Live accuracy indicator ──────────
                acc_text, acc_colour = depth_accuracy_label(med)
                cv2.putText(depth_vis, acc_text,
                    (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, acc_colour, 1)
                cv2.putText(depth_vis,
                    "Ideal range: 40–80cm for cubes/cylinders",
                    (6, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1)

            # ── Click probe ─────────────────────────
            if click_info:
                u, v, d, X, Y, Z = click_info
                for img in [rgb_frame, depth_vis]:
                    cv2.circle(img, (u, v), 6, (0, 255, 0), 2)
                if d > 0:
                    res = depth_resolution_mm(d)
                    label  = f"{d}mm  ({X:+.2f},{Y:+.2f},{Z:.2f})m"
                    label2 = f"min detectable diff: ~{res:.0f}mm"
                    cv2.putText(depth_vis, label,  (min(u+8, 580), max(v-14, 14)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)
                    cv2.putText(depth_vis, label2, (min(u+8, 580), max(v+4, 30)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,200,200), 1)

            # ── Side-by-side ─────────────────────────
            combined = np.hstack([rgb_frame, depth_vis])
            cv2.putText(combined, "RGB", (10, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            cv2.putText(combined, "Depth (TURBO: dark=far, bright=close)",
                        (650, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

            cv2.imshow("Depth Explorer", combined)

            # ── Histogram window ─────────────────────
            if show_hist:
                cv2.imshow("Histogram", draw_histogram(depth_frame, DISPLAY_MAX_MM))
            else:
                cv2.destroyWindow("Histogram") if cv2.getWindowProperty(
                    "Histogram", cv2.WND_PROP_VISIBLE) >= 1 else None

            # ── Confidence window ────────────────────
            if show_conf and conf_frame is not None:
                cv2.imshow("Confidence", draw_confidence(conf_frame))
            elif not show_conf:
                try: cv2.destroyWindow("Confidence")
                except: pass

            key = cv2.waitKey(1) & 0xFF
            if   key == ord('q'): break
            elif key == ord('h'): show_hist = not show_hist; print(f"  Histogram {'ON' if show_hist else 'OFF'}")
            elif key == ord('c'): show_conf = not show_conf; print(f"  Confidence {'ON' if show_conf else 'OFF'}")
            elif key == ord('+'): DISPLAY_MAX_MM = min(15000, DISPLAY_MAX_MM + RANGE_STEP_MM); print(f"  Max range: {DISPLAY_MAX_MM}mm")
            elif key == ord('-'): DISPLAY_MAX_MM = max(500,   DISPLAY_MAX_MM - RANGE_STEP_MM); print(f"  Max range: {DISPLAY_MAX_MM}mm")
            elif key == ord('s'):
                ts = time.strftime("%Y%m%d_%H%M%S")
                np.save(save_dir / f"depth_{ts}.npy", depth_frame)
                cv2.imwrite(str(save_dir / f"rgb_{ts}.png"), rgb_frame)
                print(f"  Saved depth_{ts}.npy + rgb_{ts}.png")

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
