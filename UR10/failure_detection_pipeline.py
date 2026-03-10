"""
failure_detection_pipeline.py
==============================
Pick pipeline with dual-layer failure detection for UR10 + RG2 + OAK-D.

DETECTION LAYERS:
  1. Gripper feedback (immediate, ~10ms)
       - width < 11mm after closing  → fully closed, object missed  → RETRY
       - DI8 LOW                     → no force contact at all       → RETRY
       - width ≥ 11mm + DI8 HIGH    → something in gripper          → proceed to CHECK 2

  2. Slip detection during lift (background thread)
       - grip_close.urp runs as a Loop, re-closing every 0.1s
       - Once DI8 HIGH is confirmed + width ≥ 11mm, we had_object = True
       - If width then drops < 11mm while had_object = True → SLIP flagged
       - Slip flag is read after the lift; triggers RETRY before place

  3. CLIP visual check after lifting (single snapshot, not live stream)
       - Capture one frame from OAK-D, crop to gripper region
       - CLIP ViT-B/32 + trained LinearProbe (clip_probe.pkl)
       - If "Holding" with confidence ≥ 0.80 → proceed to PLACE
       - Otherwise → RETRY

PLACE SEQUENCE:
  - Place position = current TCP XY offset 300mm to the LEFT (−Y in robot base)
  - Lower 150mm from approach height before opening gripper
  - Retract back up to approach height after drop

USAGE:
  python failure_detection_pipeline.py             # full run
  python failure_detection_pipeline.py --dry-run   # config print only (no robot/camera)
"""

import socket
import struct
import time
import threading
import argparse
import os
import sys

# ─── Optional heavy imports (guarded for --dry-run) ─────────────────────────
try:
    import torch
    import clip as openai_clip
    from PIL import Image
    import pickle
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

try:
    import depthai as dai
    import cv2
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False

# ============================================================
#  CONFIGURATION — edit these before running
# ============================================================
ROBOT_IP            = "192.168.8.102"

# Path to trained probe (relative to this script's directory)
CLIP_PROBE_PATH     = os.path.join(os.path.dirname(__file__),
                                   "..", "CLIP_post_grab", "clip_probe.pkl")

# Home / initial pick approach position — recorded with read_robot_frame.py.
# The robot should already be near this position before starting.
# Current TCP Z is used as approach height; pick descends APPROACH_OFFSET below.
HOME_X  = -0.0871   # metres
HOME_Y  = -1.0010
HOME_Z  = -0.2505   # approach/hover height
HOME_RX = -2.192    # radians
HOME_RY = -2.151
HOME_RZ = -0.134

# Vertical approach/lift offset above the pick surface
APPROACH_OFFSET     = 0.100        # 100 mm — robot descends this far below home Z to pick

# Fixed place position — recorded with read_robot_frame.py
# Approach from APPROACH_OFFSET above, dip down to PLACE_Z to release.
PLACE_X =  0.0803   # metres
PLACE_Y = -1.0055   # metres
PLACE_Z = -0.4204   # metres  (raised 30 mm from recorded -0.4504 to clear the table)
PLACE_RX = -2.192   # radians
PLACE_RY = -2.151
PLACE_RZ = -0.134

# Width thresholds (mm)
WIDTH_CLOSED_MM     = 11.0         # ≤ this → fully closed / missed
WIDTH_OBJECT_MM     = 12.0         # > this → object confirmed held

# CLIP confidence gate (0–1)
# 0.60 is a good starting point for testing — raise to 0.80 once lighting/angle
# is confirmed to match training conditions.
CLIP_CONFIDENCE     = 0.60

# Maximum pick attempts before giving up
MAX_RETRIES         = 3

# Gripper settle time after closing before we read sensors (seconds)
GRIPPER_SETTLE_S    = 2.5

# Movement speeds
VEL                 = 0.3          # m/s
ACC                 = 0.3          # m/s²

# OAK-D crop parameters (must match training crop in collect_clip_dataset.py)
CROP_W              = 1400
CROP_H              = 600


# ============================================================
#  FailureDetectionPipeline
# ============================================================
class FailureDetectionPipeline:
    """
    Combines UR10 robot control, RG2 gripper feedback, slip detection,
    and CLIP visual verification into a single pick-with-retry pipeline.
    """

    DASHBOARD_PORT   = 29999
    FEEDBACK_PORT    = 30002
    SLIP_CHECK_INTER = 0.3   # seconds between slip monitor ticks

    # ── init ────────────────────────────────────────────────
    def __init__(self, robot_ip: str, clip_probe_path: str):
        self.ip              = robot_ip
        self.running         = True

        # ── Feedback state (port 30002 background thread) ──
        self.latest_analog_in2  = 0.0
        self.latest_digital_in  = 0
        self.tcp_x = self.tcp_y = self.tcp_z = 0.0
        self.tcp_rx = self.tcp_ry = self.tcp_rz = 0.0

        # ── Slip detection state ───────────────────────────
        self._grip_state        = "open"
        self._grip_state_lock   = threading.Lock()
        self._had_object        = False
        self._monitoring_active = False
        self._loop_stopped      = False
        self._slip_detected     = False

        # ── Live camera state (shared between threads) ─────
        self._live_frame   = None          # latest decoded BGR frame (numpy)
        self._frame_lock   = threading.Lock()
        self._pipeline_status = "Initialising..."  # HUD status text
        self._clip_result  = None          # (label, conf) or None

        # ── Connect feedback stream ─────────────────────────
        print(f"[Init] Connecting to UR10 at {self.ip} port {self.FEEDBACK_PORT}...")
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(3.0)
        self._sock.connect((self.ip, self.FEEDBACK_PORT))
        print("[Init] Connected to robot feedback stream.")

        self._recv_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._recv_thread.start()

        # ── Start slip monitor ──────────────────────────────
        self._slip_thread = threading.Thread(target=self._slip_monitor, daemon=True)
        self._slip_thread.start()

        # ── Load CLIP + probe ───────────────────────────────
        if not CLIP_AVAILABLE:
            raise RuntimeError("CLIP not installed. Run: pip install git+https://github.com/openai/CLIP.git")

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Init] Loading CLIP (ViT-B/32) on {device_str}...")
        self._clip_device = device_str
        self._clip_model, self._clip_preprocess = openai_clip.load("ViT-B/32", device=device_str)

        probe_path = os.path.abspath(clip_probe_path)
        if not os.path.exists(probe_path):
            raise FileNotFoundError(f"CLIP probe not found: {probe_path}")
        with open(probe_path, "rb") as f:
            self._clf = pickle.load(f)
        print(f"[Init] CLIP probe loaded from {probe_path}")

        # ── Open OAK-D camera ──────────────────────────────
        if not CAMERA_AVAILABLE:
            raise RuntimeError("depthai / cv2 not installed.")
        print("[Init] Opening OAK-D camera...")
        self._oak_device, self._q_mjpeg = self._open_camera()
        print("[Init] Camera ready.")

        # ── Start background camera decode thread ──────────
        # Continuously drains the MJPEG queue and stores decoded frames.
        # This keeps autofocus running and ensures _live_frame is always fresh.
        self._cam_thread = threading.Thread(target=self._camera_decode_loop, daemon=True)
        self._cam_thread.start()

        # ── Start live display thread ──────────────────────
        # Calls imshow+waitKey(1) continuously — shows the camera window always.
        self._display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self._display_thread.start()

        # ── Wait for feedback + first frame ────────────────
        print("[Init] Waiting for robot feedback and first camera frame...")
        deadline = time.time() + 3.0   # 3 s is plenty; frame usually arrives in <1 s
        while time.time() < deadline:
            with self._frame_lock:
                has_frame = self._live_frame is not None
            if has_frame:
                break
            time.sleep(0.05)
        time.sleep(0.3)   # brief settle for feedback stream
        print("[Init] Initialisation complete.\n")

    # ── Robot feedback loop ──────────────────────────────────
    def _update_loop(self):
        """Background thread: parse I/O + TCP position from port 30002."""
        while self.running:
            try:
                header = self._sock.recv(4)
                if not header or len(header) < 4:
                    continue
                pkt_len  = struct.unpack("!I", header)[0]
                pkt_data = self._sock.recv(pkt_len - 4)

                offset = 1
                while offset < len(pkt_data):
                    if offset + 4 > len(pkt_data):
                        break
                    p_size = struct.unpack("!I", pkt_data[offset:offset+4])[0]
                    p_type = pkt_data[offset+4]

                    if p_type == 2:  # Tool Data → AI2 (width voltage) + DI8 (force)
                        self.latest_analog_in2 = struct.unpack("!d", pkt_data[offset+7:offset+15])[0]
                        self.latest_digital_in = pkt_data[offset + p_size - 1]

                    elif p_type == 4:  # Cartesian Info → TCP pose
                        self.tcp_x  = struct.unpack("!d", pkt_data[offset+5:offset+13])[0]
                        self.tcp_y  = struct.unpack("!d", pkt_data[offset+13:offset+21])[0]
                        self.tcp_z  = struct.unpack("!d", pkt_data[offset+21:offset+29])[0]
                        self.tcp_rx = struct.unpack("!d", pkt_data[offset+29:offset+37])[0]
                        self.tcp_ry = struct.unpack("!d", pkt_data[offset+37:offset+45])[0]
                        self.tcp_rz = struct.unpack("!d", pkt_data[offset+45:offset+53])[0]

                    offset += p_size
            except Exception:
                pass

    # ── Camera decode loop (background) ─────────────────────
    def _camera_decode_loop(self):
        """
        Continuously drains the MJPEG queue and stores the latest decoded frame
        in self._live_frame. Running this non-stop keeps the OAK-D autofocus
        engine active (it needs frames to be consumed to update focus).
        """
        while self.running:
            try:
                pkt = self._q_mjpeg.tryGet()
                if pkt is not None:
                    frame = cv2.imdecode(pkt.getData(), cv2.IMREAD_COLOR)
                    if frame is not None:
                        with self._frame_lock:
                            self._live_frame = frame
                else:
                    time.sleep(0.005)   # yield when queue is empty
            except Exception:
                time.sleep(0.01)

    # ── Live display loop (background) ──────────────────────
    def _display_loop(self):
        """
        Continuously shows the camera feed in a window titled
        'Pipeline Monitor'. Overlays:
          - Blue crop box (what CLIP sees)
          - Green/orange/red CLIP result (last classification)
          - White HUD: gripper width, force, pipeline status
        Runs in a background thread; uses waitKey(1) so Windows
        event loop is serviced and the window stays responsive.
        """
        win = "Pipeline Monitor"
        while self.running:
            with self._frame_lock:
                frame = self._live_frame
            if frame is None:
                time.sleep(0.03)
                continue

            h, w = frame.shape[:2]
            display = cv2.resize(frame, (854, 480))
            sx, sy  = 854 / w, 480 / h

            # Crop box
            cx, cy = w // 2, h - (CROP_H // 2) - 10
            x1 = max(0, cx - CROP_W // 2)
            y1 = max(0, cy - CROP_H // 2)
            x2 = min(w, x1 + CROP_W)
            y2 = min(h, y1 + CROP_H)
            cv2.rectangle(display,
                          (int(x1*sx), int(y1*sy)),
                          (int(x2*sx), int(y2*sy)),
                          (255, 80, 0), 2)

            # CLIP result overlay
            clip_res = self._clip_result
            if clip_res is not None:
                lbl, conf = clip_res
                passed = lbl == "Holding" and conf >= CLIP_CONFIDENCE
                if passed:
                    clip_color, clip_text = (0, 220, 0),   f"CLIP: HOLDING {conf*100:.1f}%  PASS"
                elif lbl == "Holding":
                    clip_color, clip_text = (0, 165, 255), f"CLIP: HOLDING {conf*100:.1f}%  LOW CONF"
                else:
                    clip_color, clip_text = (0, 0, 220),   f"CLIP: EMPTY {conf*100:.1f}%  FAIL"
                cv2.putText(display, clip_text, (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, clip_color, 3)
            else:
                cv2.putText(display, "CLIP: waiting...", (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (160, 160, 160), 2)

            # Gripper HUD
            width = self.get_width_mm()
            force = self.is_force_detected()
            hud = (f"W:{width:.1f}mm  F:{'YES' if force else 'NO'}  "
                   f"| {self._pipeline_status}")
            cv2.putText(display, hud, (20, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

            cv2.imshow(win, display)
            cv2.waitKey(1)

        cv2.destroyAllWindows()
    def _slip_monitor(self):
        """
        Background thread adapted from grip_control_with_slip_detection.py.

        While grip_state == 'closing' and monitoring is active:
          - Track when DI8 HIGH + width > 12 mm  → had_object = True
          - If width then drops < 11 mm           → SLIP → set _slip_detected flag
            (grip_close.urp loop re-closes automatically)
        """
        while self.running:
            with self._grip_state_lock:
                state = self._grip_state

            if state == "closing" and self._monitoring_active:
                force = self.is_force_detected()
                width = self.get_width_mm()

                # Confirm we have an object
                if force and width > WIDTH_OBJECT_MM and not self._had_object:
                    self._had_object = True
                    print(f"  [SlipMonitor] Object confirmed at {width:.1f} mm — monitoring for slip.")

                # Detect slip: had object, now fully closed again
                if force and width < WIDTH_CLOSED_MM and not self._loop_stopped:
                    if self._had_object:
                        print(f"\n  ⚠️  [SlipMonitor] SLIP DETECTED — width dropped to {width:.1f} mm → gripper re-closing.")
                        self._slip_detected = True
                    else:
                        print(f"  [SlipMonitor] Fully closed at {width:.1f} mm, empty grasp — stopping loop.")
                    self._dashboard_cmd("stop")
                    self._loop_stopped = True
                    self._had_object   = False

            elif state != "closing":
                self._had_object        = False
                self._monitoring_active = False
                self._loop_stopped      = False

            time.sleep(self.SLIP_CHECK_INTER)

    # ── Dashboard helpers ─────────────────────────────────────
    def _dashboard_cmd(self, cmd: str) -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(3.0)
            s.connect((self.ip, self.DASHBOARD_PORT))
            s.recv(1024)                      # consume welcome banner
            s.sendall((cmd + "\n").encode())
            resp = s.recv(1024).decode().strip()
            s.close()
            return resp
        except Exception as e:
            print(f"  [Dashboard] Error [{cmd}]: {e}")
            return ""

    def _stop_and_wait(self, timeout: float = 4.0):
        """Stop current URP, poll until robot confirms idle, drain gripper buffer."""
        self._dashboard_cmd("stop")
        deadline = time.time() + timeout
        while time.time() < deadline:
            if "false" in self._dashboard_cmd("running").lower():
                break
            time.sleep(0.15)
        print("  [Gripper] Draining buffer (1.5 s)...")
        time.sleep(1.5)

    def _play_urp(self, program: str):
        """Stop current program, load and play a URP."""
        self._stop_and_wait()
        self._dashboard_cmd(f"load {program}")
        time.sleep(0.2)
        self._dashboard_cmd("play")

    # ── Gripper actions ───────────────────────────────────────
    def open_gripper(self):
        with self._grip_state_lock:
            self._grip_state = "open"
        self._had_object = False
        self._monitoring_active = False
        self._loop_stopped = False
        self._slip_detected = False
        print("  → Opening gripper...")
        self._play_urp("grip_open.urp")

    def close_gripper(self):
        """
        Close gripper with loop (grip_close.urp continuously re-closes).
        Slip monitoring activates after GRIPPER_SETTLE_S seconds.
        """
        with self._grip_state_lock:
            self._grip_state = "closing"
        self._had_object = False
        self._monitoring_active = False
        self._loop_stopped = False
        self._slip_detected = False

        print("  → Closing gripper (loop)...")
        self._play_urp("grip_close.urp")

        # Enable slip monitoring after gripper has settled
        def _enable():
            time.sleep(GRIPPER_SETTLE_S)
            with self._grip_state_lock:
                if self._grip_state == "closing":
                    self._monitoring_active = True
        threading.Thread(target=_enable, daemon=True).start()

    # ── Motion commands ───────────────────────────────────────
    def _movel(self, x, y, z, rx, ry, rz, vel=VEL, acc=ACC, wait_s=5.0):
        """Send a movel (linear) move and block for wait_s seconds."""
        script = (
            f"def move_prog():\n"
            f"  movel(p[{x},{y},{z},{rx},{ry},{rz}], a={acc}, v={vel})\n"
            f"end\n"
        )
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.ip, self.FEEDBACK_PORT))
        s.send(script.encode())
        s.close()
        time.sleep(wait_s)

    def _movej(self, x, y, z, rx, ry, rz, vel=VEL, acc=ACC, wait_s=6.0):
        """Send a movej (joint space) move and block for wait_s seconds."""
        script = (
            f"def move_prog():\n"
            f"  movej(p[{x},{y},{z},{rx},{ry},{rz}], a={acc}, v={vel})\n"
            f"end\n"
        )
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.ip, self.FEEDBACK_PORT))
        s.send(script.encode())
        s.close()
        time.sleep(wait_s)

    # ── Sensor readings ───────────────────────────────────────
    def get_width_mm(self) -> float:
        """Calibrated gripper width from AI2 voltage."""
        voltage = max(self.latest_analog_in2, 0.0)
        raw_mm  = (voltage / 3.7) * 110.0
        slope   = (91.0 - 10.5) / (65.8 - 8.5)   # ≈ 1.405
        offs    = 10.5 - (8.5 * slope)             # ≈ -1.44
        return round((raw_mm * slope) + offs, 1)

    def is_force_detected(self) -> bool:
        """True when DI8 HIGH — gripper jaws touching something."""
        return (self.latest_digital_in & 0b00000001) != 0

    def gripper_has_object(self) -> bool:
        """
        True if an object is properly gripped (not just fully closed empty).
          DI8 HIGH + width >= 11 mm → object held
          DI8 HIGH + width < 11 mm  → fully closed, empty
        """
        if not self.is_force_detected():
            return False
        return self.get_width_mm() >= WIDTH_CLOSED_MM

    # ── CLIP visual check ─────────────────────────────────────
    def _build_dai_pipeline(self):
        """Build the DepthAI pipeline (MJPEG + autofocus control)."""
        pipeline = dai.Pipeline()
        cam_rgb  = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setVideoSize(1920, 1080)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setFps(30)

        ctrl_in = pipeline.create(dai.node.XLinkIn)
        ctrl_in.setStreamName("control")
        ctrl_in.out.link(cam_rgb.inputControl)

        enc = pipeline.create(dai.node.VideoEncoder)
        enc.setDefaultProfilePreset(30, dai.VideoEncoderProperties.Profile.MJPEG)
        enc.setQuality(90)
        cam_rgb.video.link(enc.input)

        xout = pipeline.create(dai.node.XLinkOut)
        xout.setStreamName("mjpeg")
        enc.bitstream.link(xout.input)

        return pipeline

    def _open_camera(self, max_attempts: int = 3, retry_delay: float = 0.5):
        """
        Initialise OAK-D camera with retry logic.

        Uses dai.Device(pipeline) — the preferred constructor in depthai >= 2.18
        which uploads the pipeline before the device fully starts, avoiding the
        USB crash that dai.Device(config) + startPipeline() can trigger.
        Falls back to the Config approach if the first method raises AttributeError.
        """
        pipeline = self._build_dai_pipeline()
        last_err = None

        for attempt in range(1, max_attempts + 1):
            try:
                print(f"  [Camera] Opening OAK-D (attempt {attempt}/{max_attempts})...")
                # Preferred: pass pipeline directly — avoids the USB-crash race
                device = dai.Device(pipeline)

                q_mjpeg   = device.getOutputQueue("mjpeg",   maxSize=4, blocking=False)
                q_control = device.getInputQueue("control")

                ctrl = dai.CameraControl()
                ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
                q_control.send(ctrl)

                print("  [Camera] OAK-D ready.")
                return device, q_mjpeg

            except Exception as e:
                last_err = e
                print(f"  [Camera] Attempt {attempt} failed: {e}")
                if attempt < max_attempts:
                    print(f"  [Camera] Waiting {retry_delay} s for USB to recover...")
                    time.sleep(retry_delay)
                    # Rebuild a fresh pipeline object for the next attempt
                    pipeline = self._build_dai_pipeline()

        raise RuntimeError(
            f"OAK-D failed to open after {max_attempts} attempts. "
            f"Last error: {last_err}\n"
            f"Try: unplug and replug the USB cable, then re-run."
        )

    def clip_classify(self) -> tuple[str, float]:
        """
        Classify the current gripper state using the latest live frame.
        Uses self._live_frame (kept fresh by _camera_decode_loop) so autofocus
        is never interrupted. Updates self._clip_result for the live display.
        Returns: (label, confidence)  where label is "Holding" or "Empty".
        """
        # Grab the most recent decoded frame from the background thread
        with self._frame_lock:
            frame = self._live_frame

        if frame is None:
            print("  [CLIP] WARNING: no frame available yet — returning Empty.")
            return "Empty", 0.0

        # Crop identical to training crop
        h, w   = frame.shape[:2]
        cx, cy = w // 2, h - (CROP_H // 2) - 10
        x1 = max(0, cx - CROP_W // 2)
        y1 = max(0, cy - CROP_H // 2)
        x2 = min(w, x1 + CROP_W)
        y2 = min(h, y1 + CROP_H)
        crop = frame[y1:y2, x1:x2].copy()

        # CLIP inference
        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensor  = self._clip_preprocess(pil_img).unsqueeze(0).to(self._clip_device)

        with torch.no_grad():
            feat = self._clip_model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            feat_np = feat.cpu().numpy()

        pred  = self._clf.predict(feat_np)[0]
        probs = self._clf.predict_proba(feat_np)[0]
        label = {0: "Empty", 1: "Holding"}[pred]
        conf  = float(probs[pred])

        # Push result to display overlay
        self._clip_result = (label, conf)

        return label, conf

    # ── Main pipeline ─────────────────────────────────────────
    def pick_with_failure_detection(self):
        """
        Full pick-and-place with dual-layer failure detection.

        The robot's CURRENT TCP position is used as the APPROACH height.
        The pick happens APPROACH_OFFSET metres below that.
        Place position = approach XY offset 300mm in -Y, same Z.
        """
        # Snapshot current TCP — this IS the approach position.
        # The robot should be hovering above the object before pressing Enter.
        time.sleep(0.3)
        approach_x  = self.tcp_x
        approach_y  = self.tcp_y
        approach_z  = self.tcp_z
        rx          = self.tcp_rx
        ry          = self.tcp_ry
        rz          = self.tcp_rz

        # Descend APPROACH_OFFSET below current Z to reach the object
        pick_x = approach_x
        pick_y = approach_y
        pick_z = approach_z - APPROACH_OFFSET

        # Fixed place: hover APPROACH_OFFSET above, dip down to PLACE_Z
        place_x     = PLACE_X
        place_y     = PLACE_Y
        place_app_z = PLACE_Z + APPROACH_OFFSET   # hover above
        place_dip_z = PLACE_Z                     # dip to table surface
        place_rx    = PLACE_RX
        place_ry    = PLACE_RY
        place_rz    = PLACE_RZ

        print("=" * 60)
        print("  FAILURE DETECTION PICK PIPELINE")
        print("=" * 60)
        print(f"  Current TCP (approach) : X={approach_x*1000:.1f} Y={approach_y*1000:.1f} Z={approach_z*1000:.1f} mm")
        print(f"  Pick Z (descend {APPROACH_OFFSET*1000:.0f}mm) : {pick_z*1000:.1f} mm")
        print(f"  Place position : X={place_x*1000:.1f} Y={place_y*1000:.1f} Z={PLACE_Z*1000:.1f} mm (fixed)")
        print(f"  Place approach Z : {place_app_z*1000:.1f} mm | Dip Z : {place_dip_z*1000:.1f} mm")
        print("=" * 60 + "\n")

        for attempt in range(1, MAX_RETRIES + 1):
            print(f"\n{'─'*50}")
            print(f"  ATTEMPT {attempt} / {MAX_RETRIES}")
            print(f"{'─'*50}")

            # ── 1. Approach: hover above pick site ──────────────
            self._pipeline_status = f"Attempt {attempt}: moving to approach"
            print(f"\n[1] Moving to approach height ({approach_z*1000:.0f} mm)...")
            self._movel(approach_x, approach_y, approach_z, rx, ry, rz, wait_s=4.0)

            # ── 2. Open gripper ─────────────────────────────────
            self._pipeline_status = "Opening gripper"
            self.open_gripper()
            time.sleep(0.5)

            # ── 3. Descend to pick Z ─────────────────────────────
            self._pipeline_status = "Descending to pick"
            print(f"[2] Descending to pick Z ({pick_z*1000:.1f} mm)...")
            self._movel(pick_x, pick_y, pick_z, rx, ry, rz, wait_s=5.0)

            # ── 4. Close gripper ─────────────────────────────────
            self._pipeline_status = "Closing gripper"
            self.close_gripper()
            # Wait for settle + monitoring to activate
            print(f"    Waiting {GRIPPER_SETTLE_S + 0.5:.1f} s for gripper to settle...")
            time.sleep(GRIPPER_SETTLE_S + 0.5)

            # ── CHECK 1: Gripper width / force ───────────────────
            self._pipeline_status = "CHECK 1: width+force"
            width  = self.get_width_mm()
            force  = self.is_force_detected()
            print(f"\n[CHECK 1] Gripper width: {width:.1f} mm | Force: {force}")

            if width < WIDTH_CLOSED_MM or not force:
                print(f"  ✗  MISS — gripper fully closed ({width:.1f} mm). Object not contacted.")
                print("     Retracting for retry...")
                self.open_gripper()
                self._movel(pick_x, pick_y, approach_z, rx, ry, rz, wait_s=4.0)
                continue   # → retry

            print(f"  ✓  Object in gripper at {width:.1f} mm — lifting...")

            # ── 5. Lift to approach height ───────────────────────
            self._pipeline_status = "Lifting to approach"
            print(f"\n[3] Lifting to approach height ({approach_z*1000:.0f} mm)...")
            self._movel(approach_x, approach_y, approach_z, rx, ry, rz, wait_s=4.0)

            # ── CHECK 1b: Slip detected during lift? ─────────────
            if self._slip_detected:
                print("  ✗  SLIP DETECTED during lift — retrying.")
                self.open_gripper()
                continue   # → retry

            # ── CHECK 2: CLIP visual check ───────────────────────
            self._pipeline_status = "CHECK 2: CLIP vision"
            print("\n[CHECK 2] Running CLIP visual classifier...")
            label, conf = self.clip_classify()
            print(f"  CLIP result : {label.upper()}  (confidence {conf*100:.1f}%)")

            if label != "Holding" or conf < CLIP_CONFIDENCE:
                print(f"  ✗  CLIP says {'not holding / uncertain'} — retrying.")
                self.open_gripper()
                self._movel(pick_x, pick_y, approach_z, rx, ry, rz, wait_s=3.0)
                continue   # → retry

            print(f"  ✓  CLIP confirms object held. Proceeding to PLACE.")

            # ── 6. Transfer to place (joint move, slip monitor active) ──
            self._pipeline_status = "Transferring to place"
            print(f"\n[4] Moving to place position (X={place_x*1000:.1f} Y={place_y*1000:.1f})...")
            self._movej(place_x, place_y, place_app_z, place_rx, place_ry, place_rz, wait_s=6.0)

            # ── CHECK 1c: Slip during transfer? ──────────────────
            if self._slip_detected:
                print("  ✗  SLIP DETECTED during transfer — object lost. Retrying from pick.")
                self.open_gripper()
                # Return to pick approach
                self._movej(pick_x, pick_y, approach_z, rx, ry, rz, wait_s=6.0)
                continue   # → retry

            # ── 7. Dip down and release ──────────────────────────
            self._pipeline_status = "Placing object"
            print(f"[5] Dipping to place depth ({place_dip_z*1000:.1f} mm)...")
            self._movel(place_x, place_y, place_dip_z, place_rx, place_ry, place_rz, wait_s=4.0)

            print("[6] Opening gripper — releasing object...")
            self.open_gripper()
            time.sleep(0.5)

            # ── 8. Retract back up ───────────────────────────────
            self._pipeline_status = "Retracting after place"
            print(f"[7] Retracting to approach height ({place_app_z*1000:.1f} mm)...")
            self._movel(place_x, place_y, place_app_z, place_rx, place_ry, place_rz, wait_s=4.0)

            print("\n" + "=" * 60)
            print(f"  ✓  PICK AND PLACE COMPLETE (attempt {attempt})")
            print("=" * 60 + "\n")
            self._pipeline_status = "DONE — success!"
            return True

        # ── All retries exhausted ────────────────────────────
        print("\n" + "=" * 60)
        print(f"  ✗  FAILED after {MAX_RETRIES} attempts. Giving up.")
        print("=" * 60 + "\n")
        self._pipeline_status = f"FAILED after {MAX_RETRIES} attempts"
        self.open_gripper()
        return False

    # ── Cleanup ───────────────────────────────────────────────
    def close(self):
        self.running = False
        try:
            self._sock.close()
        except Exception:
            pass
        try:
            self._oak_device.close()
        except Exception:
            pass
        print("[Pipeline] Closed.")


# ================================================================
#  MAIN
# ================================================================
def print_config():
    """Print configuration summary (used for --dry-run)."""
    print("\n" + "=" * 60)
    print("  FAILURE DETECTION PIPELINE — CONFIG")
    print("=" * 60)
    print(f"  Robot IP           : {ROBOT_IP}")
    print(f"  CLIP probe path    : {os.path.abspath(CLIP_PROBE_PATH)}")
    print(f"  Approach offset    : {APPROACH_OFFSET * 1000:.0f} mm (below current TCP Z)")
    print(f"  Place position     : X={PLACE_X*1000:.1f}  Y={PLACE_Y*1000:.1f}  Z={PLACE_Z*1000:.1f} mm (fixed)")
    print(f"  Place rotation     : Rx={PLACE_RX:.3f}  Ry={PLACE_RY:.3f}  Rz={PLACE_RZ:.3f}")
    print(f"  Width 'closed' thr : {WIDTH_CLOSED_MM} mm")
    print(f"  Width 'object' thr : {WIDTH_OBJECT_MM} mm")
    print(f"  CLIP confidence    : {CLIP_CONFIDENCE * 100:.0f}%")
    print(f"  Max retries        : {MAX_RETRIES}")
    print(f"  Velocity           : {VEL} m/s | Acceleration: {ACC} m/s²")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UR10 Failure Detection Pick Pipeline")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config and exit without connecting to robot or camera.")
    args = parser.parse_args()

    print_config()

    if args.dry_run:
        print("[Dry-run] Exiting without connecting.")
        sys.exit(0)

    pipeline = None
    try:
        pipeline = FailureDetectionPipeline(
            robot_ip       = ROBOT_IP,
            clip_probe_path= CLIP_PROBE_PATH,
        )

        pos = pipeline
        print(f"Current TCP position read from robot:")
        print(f"  X={pos.tcp_x*1000:.1f} mm  Y={pos.tcp_y*1000:.1f} mm  Z={pos.tcp_z*1000:.1f} mm")
        print(f"  Rx={pos.tcp_rx:.3f}  Ry={pos.tcp_ry:.3f}  Rz={pos.tcp_rz:.3f}")
        print(f"\nThis XY + Z will be used as the approach/pick position.")
        print(f"Fixed place: X={PLACE_X*1000:.1f}  Y={PLACE_Y*1000:.1f}  Z={PLACE_Z*1000:.1f} mm\n")

        input("Press Enter to start the pick sequence (Ctrl-C to abort)...")

        pipeline.pick_with_failure_detection()

    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user.")
    except Exception as e:
        print(f"\n[Main] Error: {e}")
        raise
    finally:
        if pipeline:
            pipeline.close()
