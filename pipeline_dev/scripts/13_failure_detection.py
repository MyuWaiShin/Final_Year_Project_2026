"""

13_failure_detection.py

========================

Pick-and-place pipeline with dual-layer failure detection.

Uses RTDE for all robot motion (EtherNet/IP must be OFF on pendant).

Uses port 30002 only to stream gripper sensor data (AI2 + DI8).

POSITIONS:

  Loaded from data/saved_positions.json (use 06_tcp_test.py to record).

  Required keys:  "pick_pos"   — hover position ABOVE the object (approach)

                  "place_pos"  — hover position ABOVE the drop zone (approach)

  The robot descends APPROACH_OFFSET below each approach Z to pick/place.

DETECTION LAYERS:

  1. Gripper width check after close (immediate):

       width < WIDTH_CLOSED_MM → gripper fully snapped shut → missed → retry

       (width alone does NOT confirm grasp — only CLIP does)

  2. Slip detection during transit (background thread):

       close_gripper_timed.urp loops (0.2s intervals) re-closing

       width drops back below WIDTH_CLOSED_MM → SLIP detected → re-verify with CLIP

  3. CLIP visual check after lift:

       CLIP ViT-B/32 + trained LinearProbe (clip_probe.pkl)

       Holding < threshold → safe descend to pick_z → open → rescan

MOTION:

  RTDE (port 30004) via rtde_control + rtde_receive.

  moveL / moveJ block until the robot arrives at the target position.

USAGE:

  python 13_failure_detection.py             # full run

  python 13_failure_detection.py --dry-run   # print config only

"""

import json

import os

import signal

import socket

import struct

import threading

import time

import argparse

import sys

from pathlib import Path

from collections import deque

# ── Robust path setup ──────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent

BASE_DIR   = SCRIPT_DIR.parent

# ── RTDE ──────────────────────────────────────────────────────────────────

try:

    import rtde_control

    import rtde_receive

    RTDE_AVAILABLE = True

except ImportError:

    RTDE_AVAILABLE = False

# ── Optional heavy imports ─────────────────────────────────────────────────

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

    import numpy as np

    CAMERA_AVAILABLE = True

except ImportError:

    CAMERA_AVAILABLE = False

# ===========================================================================

#  CONFIGURATION

# ===========================================================================

ROBOT_IP         = "192.168.8.102"

POSITIONS_FILE   = BASE_DIR / "data/saved_positions.json"

CLIP_PROBE_PATH  = BASE_DIR.parent / "CLIP_post_grab/clip_probe.pkl"

# Approach: robot descends this far below the loaded position Z to pick/place

APPROACH_OFFSET  = 0.100     # metres — descend BELOW saved position Z to pick

LIFT_HEIGHT_M    = 0.200     # metres — lift ABOVE pick Z after grabbing (for CLIP check)

# -- Gripper (RG2 via Dashboard URP) -----------------------------------------

GRIP_CLOSE_ONCE_URP  = "/programs/myu/close_gripper.urp"

GRIP_CLOSE_LOOP_URP  = "/programs/myu/close_gripper_timed.urp"

GRIP_OPEN_URP        = "/programs/myu/open_gripper.urp"

GRIP_OPEN_MM         = 85.0   # expected open width (for _wait_for_gripper)

GRIP_CLOSE_MM        = 0.0    # expected closed width (for _wait_for_gripper)

# Width thresholds

WIDTH_CLOSED_MM = 11.0   # <= this -> gripper fully closed (missed the object)

# CLIP confidence threshold

CLIP_CONFIDENCE  = 0.75

# Max pick attempts

MAX_RETRIES      = 3

# Gripper settle: _wait_for_gripper polls until width stable

GRIPPER_SETTLE_S = 3.5    # max seconds to wait for gripper to settle

# ── RTDE motion speeds ────────────────────────────────────────────────────

VEL  = 0.3   # m/s  — approach / lift movements

ACC  = 0.2   # m/s²

VEL_SLOW = 0.10  # m/s  — descend to object (gentle)

ACC_SLOW = 0.10

# ── OAK-D CLIP crop region (gripper in view at bottom of frame) ───────────

CROP_W = 1400

CROP_H = 600

# How long the CLIP result banner stays after the pipeline moves on

CLIP_RESULT_LINGER_S = 2.0

# On-screen status log: how many lines to keep

LOG_LINES = 5

# ===========================================================================

#  FailureDetectionPipeline

# ===========================================================================

class FailureDetectionPipeline:

    """

    UR10 + RG2 + OAK-D pick-and-place with three failure detection layers.

    Uses RTDE for all motion (EtherNet/IP must be OFF).

    Uses port 30002 only for gripper sensor streaming (AI2 + DI8).

    """

    DASHBOARD_PORT   = 29999

    FEEDBACK_PORT    = 30002

    SLIP_CHECK_INTER = 0.1   # how often Python polls width for slip detection

    # ── init ────────────────────────────────────────────────────────────────

    def __init__(self, robot_ip: str, pick_pos: dict, place_pos: dict, dump_pos: dict):

        self.ip        = robot_ip
        self.pick_pos  = pick_pos
        self.place_pos = place_pos
        self.dump_pos  = dump_pos
        self.running   = True
        # ── Port 30002 — sensor streaming (AI2 + DI8 only) ────────────────
        self.latest_ai2     = 0.0   # AI2 voltage → gripper width
        self.latest_di_word = 0     # 64-bit digital input word → DI8 (bit 17)
        print(f"[Init] Connecting sensor stream ({self.ip}:{self.FEEDBACK_PORT})...")
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(3.0)
        self._sock.connect((self.ip, self.FEEDBACK_PORT))
        threading.Thread(target=self._sensor_loop, daemon=True).start()
        print("[Init] Sensor stream connected.")
        # ── Persistent URScript send socket (port 30002) ───────────────────
        # Reusing the same TCP connection avoids a ~5-20ms handshake per move.
        print(f"[Init] Connecting URScript socket ({self.ip}:{self.FEEDBACK_PORT})...")
        self._urscript_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._urscript_sock.settimeout(5.0)
        self._urscript_sock.connect((self.ip, self.FEEDBACK_PORT))
        self._urscript_lock = threading.Lock()   # protect concurrent sends
        # Drain thread: port 30002 streams secondary data to ALL connected clients.
        # Nobody reads from _urscript_sock, so the OS recv buffer would fill ~0.5s
        # and eventually block sends. Drain thread discards that data silently.
        threading.Thread(target=self._drain_urscript_sock, daemon=True).start()
        print("[Init] URScript socket connected.")
        # ── RTDE Receive (read-only — runs NO script on robot) ────────────
        # RTDEReceiveInterface only reads robot state packets on port 30004.
        # It does NOT upload a script, so dashboard URPs can run freely.
        # Motion commands are sent via port 30002 URScript instead.
        if not RTDE_AVAILABLE:
            raise RuntimeError("rtde_receive not installed.")
        print(f"[Init] Connecting RTDE receive to {self.ip}...")
        self._rtde_r = self._connect_with_timeout(
            lambda: rtde_receive.RTDEReceiveInterface(self.ip),
            label="RTDE receive", timeout=10.0)
        print("[Init] RTDE receive connected.")
        # RTDEControlInterface is NOT used — it uploads a keepalive script that
        # conflicts with dashboard URP calls (_play_urp stopScript drops the
        # connection and the C++ reconnect hangs). All motion is sent as raw
        # URScript via port 30002; RTDEReceiveInterface polls TCP pose.
        # ── Slip / loop-close state ────────────────────────────────────────
        self._grip_state        = "open"
        self._grip_state_lock   = threading.Lock()
        self._had_object        = False
        self._monitoring_active = False
        self._loop_stopped      = False
        self._slip_detected     = False
        self._loop_close_active = False
        threading.Thread(target=self._slip_monitor, daemon=True).start()
        # ── Camera / display state ─────────────────────────────────────────
        self._live_frame      = None
        self._frame_lock      = threading.Lock()
        self._pipeline_status = "Initialising..."
        # CLIP result: (label, conf, stage, confirmed) — or None
        # confirmed=True once pipeline has acted on it (starts linger timer)
        self._clip_result      = None
        self._clip_result_time = None
        # On-screen rolling log
        self._log = deque(maxlen=LOG_LINES)
        # FPS tracking (updated in _camera_decode_loop)
        self._fps       = 0.0
        self._fps_count = 0
        self._fps_time  = time.time()
        # ── Load CLIP + probe ──────────────────────────────────────────────
        if not CLIP_AVAILABLE:
            raise RuntimeError("CLIP not installed: "
                               "pip install git+https://github.com/openai/CLIP.git")
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Init] Loading CLIP ViT-B/32 on {device_str}...")
        self._clip_device = device_str
        self._clip_model, self._clip_preprocess = openai_clip.load(
            "ViT-B/32", device=device_str)
        probe_path = Path(CLIP_PROBE_PATH).resolve()
        if not probe_path.exists():
            raise FileNotFoundError(f"CLIP probe not found: {probe_path}")
        with open(probe_path, "rb") as f:
            self._clf = pickle.load(f)
        print("[Init] CLIP probe loaded.")
        # ── Open OAK-D camera ─────────────────────────────────────────────
        if not CAMERA_AVAILABLE:
            raise RuntimeError("depthai / cv2 not installed.")
        print("[Init] Opening OAK-D camera...")
        self._oak_device, self._videoQueue = self._open_camera()
        print("[Init] Camera ready.")
        threading.Thread(target=self._camera_decode_loop, daemon=True).start()
        cv2.namedWindow("Pipeline Monitor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Pipeline Monitor", 960, 540)
        # Wait for first frame
        print("[Init] Waiting for first camera frame...")
        deadline = time.time() + 5.0
        while time.time() < deadline:
            with self._frame_lock:
                if self._live_frame is not None:
                    break
            time.sleep(0.05)
        time.sleep(0.3)
        print("[Init] Ready.\n")

    # ── RTDE timeout helper ──────────────────────────────────────────────────

    @staticmethod
    def _connect_with_timeout(factory, label: str, timeout: float = 10.0):
        """
        Run *factory()* in a daemon thread and return its result.
        Raises RuntimeError if it doesn't finish within *timeout* seconds.
        The daemon thread is abandoned on timeout (it will eventually error
        or finish on its own — the process itself remains killable).
        """
        result   = [None]
        exc      = [None]
        done_evt = threading.Event()

        def _worker():
            try:
                result[0] = factory()
            except Exception as e:
                exc[0] = e
            finally:
                done_evt.set()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        if not done_evt.wait(timeout=timeout):
            raise RuntimeError(
                f"[Init] Timeout ({timeout:.0f}s): {label} did not connect.\n"
                f"  Check that the robot is reachable, EtherNet/IP is OFF on the pendant,\n"
                f"  and no other program is holding the RTDE port."
            )
        if exc[0] is not None:
            raise exc[0]
        return result[0]

    # ── URScript socket drain ────────────────────────────────────────────────

    def _drain_urscript_sock(self):
        """
        Port 30002 floods ALL connected clients with secondary data at ~125Hz.
        _urscript_sock is write-only for us, but the robot still sends data to it.
        If we don't read, the OS recv buffer fills in ~0.5s and sendall() hangs.
        This thread silently discards everything received on _urscript_sock.
        """
        while self.running:
            try:
                self._urscript_sock.recv(4096)
            except Exception:
                time.sleep(0.01)

    # ── Sensor stream (port 30002) ───────────────────────────────────────────

    def _sensor_loop(self):

        """
        Continuously reads UR Secondary Client packets (port 30002).
        Extracts only:
          sub-type 2 (Tool Data)       → AI2 voltage (gripper width)
          sub-type 3 (Masterboard Data) → 64-bit DI word (bit 17 = RG2 force)
        All motion is handled by RTDE — this loop is sensors only.
        """
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
                    if p_type == 2 and offset + 15 <= len(pkt_data):
                        # Tool Data — AI2 voltage at offset+7 (8-byte double)
                        self.latest_ai2 = struct.unpack(
                            "!d", pkt_data[offset+7:offset+15])[0]
                    elif p_type == 3 and offset + 13 <= len(pkt_data):
                        # Masterboard Data — 64-bit DI word at offset+5
                        self.latest_di_word = struct.unpack(
                            "!Q", pkt_data[offset+5:offset+13])[0]
                    if p_size == 0:
                        break
                    offset += p_size
            except Exception:
                pass

    # ── Camera ───────────────────────────────────────────────────────────────

    def _open_camera(self):

        """Open OAK-D ColorCamera pipeline (1080p, maxSize=1 → always latest)."""
        pipeline = dai.Pipeline()
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setVideoSize(1920, 1080)
        cam.setInterleaved(False)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        xout = pipeline.create(dai.node.XLinkOut)
        xout.setStreamName("video")
        cam.video.link(xout.input)
        device = dai.Device(pipeline)
        # maxSize=1: old frames are dropped → window never freezes during URP calls
        q = device.getOutputQueue(name="video", maxSize=1, blocking=False)
        return device, q

    def _camera_decode_loop(self):

        """Drain the camera queue continuously; update FPS counter."""
        while self.running:
            try:
                pkt = self._videoQueue.tryGet()
                if pkt is not None:
                    frame = pkt.getCvFrame()
                    if frame is not None:
                        with self._frame_lock:
                            self._live_frame = frame
                        # FPS counter
                        self._fps_count += 1
                        if self._fps_count >= 10:
                            elapsed = time.time() - self._fps_time
                            self._fps = self._fps_count / elapsed if elapsed > 0 else 0
                            self._fps_count = 0
                            self._fps_time = time.time()
                else:
                    time.sleep(0.005)
            except Exception:
                time.sleep(0.01)

    def _pump_display(self):

        """
        Refresh the cv2 window. MUST be called from the MAIN THREAD.
        Call every ~50ms during long waits to keep the window live.
        """
        with self._frame_lock:
            frame = self._live_frame
        if frame is None:
            cv2.waitKey(1)
            return
        h, w    = frame.shape[:2]
        display = cv2.resize(frame, (960, 540))
        sx, sy  = 960 / w, 540 / h
        # ── CLIP crop box (orange) ─────────────────────────────────────────
        cx, cy = w // 2, h - (CROP_H // 2) - 10
        x1c = max(0, cx - CROP_W // 2);  y1c = max(0, cy - CROP_H // 2)
        x2c = min(w, x1c + CROP_W);      y2c = min(h, y1c + CROP_H)
        cv2.rectangle(display,
                      (int(x1c*sx), int(y1c*sy)),
                      (int(x2c*sx), int(y2c*sy)),
                      (0, 140, 255), 2)
        # ── CLIP result banner (persistent until linger expires) ───────────
        cr = self._clip_result
        if cr:
            lbl, conf, stage, confirmed = cr
            # Start linger timer only once the pipeline has confirmed it acted
            if confirmed and self._clip_result_time is not None:
                if time.time() - self._clip_result_time > CLIP_RESULT_LINGER_S:
                    self._clip_result = None
                    cr = None
        if cr:
            lbl, conf, stage, _ = cr
            passed    = lbl == "Holding" and conf >= CLIP_CONFIDENCE
            uncertain = lbl == "Holding" and conf < CLIP_CONFIDENCE
            if passed:
                col  = (0, 210, 0)
                txt  = f"  GRASP SUCCESS  {conf*100:.0f}%  [OK]  [{stage}]"
            elif uncertain:
                col  = (0, 165, 255)
                txt  = f"  UNCERTAIN  {conf*100:.0f}%  [?]  [{stage}]"
            else:
                col  = (0, 0, 220)
                txt  = f"  GRASP FAILURE  {conf*100:.0f}%  [X]  [{stage}]"
            # Large filled banner at top of frame
            banner_h = 52
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (960, banner_h), (30, 30, 30), -1)
            cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)
            cv2.putText(display, txt, (10, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, col, 2)
        else:
            cv2.putText(display, "CLIP: standby", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 120, 120), 1)
        # ── On-screen rolling status log (bottom area) ────────────────────
        log_lines = list(self._log)
        panel_h   = len(log_lines) * 22 + 10
        py_start  = display.shape[0] - panel_h - 28
        overlay2  = display.copy()
        cv2.rectangle(overlay2, (0, py_start), (960, display.shape[0] - 28),
                      (20, 20, 20), -1)
        cv2.addWeighted(overlay2, 0.5, display, 0.5, 0, display)
        for i, line in enumerate(log_lines):
            cy2 = py_start + 18 + i * 22
            col = (200, 200, 200) if i < len(log_lines) - 1 else (255, 255, 255)
            cv2.putText(display, line, (8, cy2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1)
        # ── HUD: width + FPS (bottom bar) ─────────────────────────────────
        w_mm = self.get_width_mm()
        hud  = f"W:{w_mm:.1f}mm  |  FPS:{self._fps:.1f}  |  {self._pipeline_status}"
        cv2.putText(display, hud, (8, display.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
        cv2.imshow("Pipeline Monitor", display)
        cv2.waitKey(1)

    def _pump_for(self, duration_s: float):

        """Pump the display for a fixed duration (use during deliberate waits)."""
        deadline = time.time() + duration_s
        while time.time() < deadline:
            self._pump_display()
            time.sleep(0.05)

    # ── On-screen log helper ─────────────────────────────────────────────────

    def _log_status(self, msg: str):

        """Append a message to the on-screen rolling log and print to console."""
        self._log.append(f">> {msg}")   # ASCII only -- OpenCV can't render unicode
        print(f"  [{msg}]")

    # ── Slip monitor ─────────────────────────────────────────────────────────

    def _slip_monitor(self):

        """
        Monitors gripper width while grip_state == 'loop'.
        Sets _slip_detected if object width drops back below CLOSED threshold.
        """
        while self.running:
            with self._grip_state_lock:
                state = self._grip_state
            if state == "loop" and self._monitoring_active:
                width = self.get_width_mm()
                if width > WIDTH_CLOSED_MM and not self._had_object:
                    self._had_object = True
                    print(f"  [Slip] Object between fingers at {width:.1f} mm")
                if width < WIDTH_CLOSED_MM and not self._loop_stopped:
                    if self._had_object:
                        print(f"\n  [!] SLIP -- width dropped to {width:.1f} mm")
                        self._slip_detected = True
                    # DO NOT call dashboard stop -- that kills the transit movel.
                    # Just set the flag. movej_pose poll loop will abort transit.
                    self._loop_stopped = True
                    self._had_object   = False
            elif state != "loop":
                self._had_object = False
                self._monitoring_active = False
                self._loop_stopped = False
            time.sleep(self.SLIP_CHECK_INTER)

    # ── Dashboard (for gripper URPs) ─────────────────────────────────────────

    def _dashboard_cmd(self, cmd: str, retries: int = 3) -> str:

        for attempt in range(1, retries + 1):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(5.0)
                s.connect((self.ip, self.DASHBOARD_PORT))
                s.recv(1024)
                s.sendall((cmd + "\n").encode())
                resp = s.recv(1024).decode().strip()
                s.close()
                return resp
            except Exception as e:
                print(f"  [Dashboard] {cmd}: {e}  (attempt {attempt}/{retries})")
                if attempt < retries:
                    time.sleep(0.5)
        return ""

    def _stop_and_wait(self, timeout=4.0):

        """Stop current URP (only if one is actually running) and wait until idle.
        Skipping stop when idle prevents the dashboard stop from resetting the RTDE
        port — which is what causes the RTDEReceiveInterface EOF stacktrace."""
        # Check first — if nothing is running, skip the stop entirely (no RTDE reset)
        resp = self._dashboard_cmd("running", retries=2)
        if "false" in resp.lower():
            return   # nothing running, no-op
        self._dashboard_cmd("stop")
        deadline = time.time() + timeout
        while time.time() < deadline:
            resp = self._dashboard_cmd("running", retries=1)
            if "false" in resp.lower():
                break
            self._pump_display()
            time.sleep(0.1)

    # ── Gripper (RG2 via Dashboard URP) ─────────────────────────────────────────

    # rg_grip is NOT available via port 30002 URScript (URCap restriction).

    # Dashboard URP is the only working method. _stop_and_wait uses RTDE

    # isProgramRunning() for a fast in-process check instead of a dashboard

    # round-trip, cutting idle detection from ~50ms to <1ms.

    def _is_urp_running(self) -> bool:

        """Fast check via RTDE (no network round-trip overhead)."""
        try:
            return self._rtde_r.isProgramRunning()
        except Exception:
            return False

    def _stop_and_wait(self, timeout=4.0):

        """Stop current URP only if one is actually running (checked via RTDE).
        Skipping stop when idle avoids the dashboard RTDE port reset that
        causes the RTDEReceiveInterface EOF stacktrace."""
        if not self._is_urp_running():
            return   # nothing running -- skip stop entirely
        self._dashboard_cmd("stop")
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self._is_urp_running():
                break
            self._pump_display()
            time.sleep(0.05)

    def _wait_urp_done(self, timeout=6.0):
        """Wait until the dashboard URP has stopped running."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self._is_urp_running():
                break
            self._pump_display()
            time.sleep(0.05)

    def _play_urp(self, program: str):
        """Load and play a gripper URP via dashboard.
        No RTDEControlInterface — nothing to suspend/restore.
        """
        same_prog = getattr(self, "_last_urp", None) == program
        if not same_prog:
            self._stop_and_wait()
            self._dashboard_cmd(f"load {program}")
            self._last_urp = program
            time.sleep(0.06)
        else:
            self._stop_and_wait()
        self._dashboard_cmd("play")
        self._wait_urp_done()


    def _wait_for_gripper(self, target_mm: float, timeout: float = 3.5):

        """
        Poll AI2 width until settled after triggering a gripper URP.
        Phase 0: skip if already at/near target (within 5mm)
        Phase 1: wait for movement to START (width changes > 2mm, max 1.5s)
        Phase 2: wait for movement to STOP (4 consecutive stable readings)
        """
        start_w = self.get_width_mm()
        if abs(start_w - target_mm) < 5.0:
            return
        move_dl = time.time() + 1.5
        while time.time() < move_dl:
            self._pump_display()
            time.sleep(0.06)
            if abs(self.get_width_mm() - start_w) > 2.0:
                break
        prev = self.get_width_mm()
        stable = 0
        dl = time.time() + timeout
        while time.time() < dl:
            self._pump_display()
            time.sleep(0.06)
            cur = self.get_width_mm()
            if abs(cur - prev) < 0.5:
                stable += 1
                if stable >= 3:
                    break
            else:
                stable = 0
            prev = cur

    def open_gripper(self):

        """Open RG2 via dashboard URP. Skips entirely if already open."""
        with self._grip_state_lock:
            self._grip_state = "open"
        self._had_object = False
        self._monitoring_active = False
        self._loop_stopped = False
        self._slip_detected = False
        self._loop_close_active = False
        if self.get_width_mm() >= GRIP_OPEN_MM - 5.0:
            return   # already open
        print("  [Gripper] Opening...")
        self._play_urp(GRIP_OPEN_URP)
        self._wait_for_gripper(GRIP_OPEN_MM)

    def close_gripper_once(self):

        """Single close via dashboard URP. Skips if already fully closed."""
        with self._grip_state_lock:
            self._grip_state = "grab"
        self._had_object = False
        self._monitoring_active = False
        self._loop_stopped = False
        self._slip_detected = False
        self._loop_close_active = False
        if self.get_width_mm() <= GRIP_CLOSE_MM + 3.0:
            return   # already closed
        print("  [Gripper] Closing (single)...")
        self._play_urp(GRIP_CLOSE_ONCE_URP)
        self._wait_for_gripper(GRIP_CLOSE_MM)

    def start_slip_monitor_close(self):

        """Start looping close URP for slip detection during transit.
        Loads and plays close_gripper_timed.urp then returns IMMEDIATELY —
        the URP runs on the robot in background while the arm transits.
        DO NOT call _wait_urp_done here: the URP loops and never finishes
        on its own, so waiting would block for up to _wait_urp_done's timeout.
        Call _stop_and_wait() when transit is done to kill the loop.
        """
        with self._grip_state_lock:
            self._grip_state = "loop"
        self._had_object = True
        self._monitoring_active = True
        self._loop_stopped = False
        self._slip_detected = False
        self._loop_close_active = True
        # Load only if different program (avoids an unnecessary stop+load RTT)
        same_prog = getattr(self, "_last_urp", None) == GRIP_CLOSE_LOOP_URP
        if not same_prog:
            self._stop_and_wait()
            self._dashboard_cmd(f"load {GRIP_CLOSE_LOOP_URP}")
            self._last_urp = GRIP_CLOSE_LOOP_URP
            time.sleep(0.06)   # robot needs ~60ms to process load
        else:
            self._stop_and_wait()
        self._dashboard_cmd("play")
        # Return immediately — URP runs concurrently on robot while arm moves

    # ── Gripper sensors ──────────────────────────────────────────────────────

    def get_width_mm(self) -> float:

        """
        RG2 width from AI2 voltage — tool flange analog input (TAI0).
        The CB3 tool flange connector is always 0-5V range, NOT 0-10V.
        Max measured = 3.7V (voltage divider: 29kΩ UR / (10kΩ RG2 + 29kΩ)).
        Two-point linear calibration corrects RG2 sensor nonlinearity:
          raw_mm = (V / 3.7) * 110
          width  = raw_mm * slope + offset
        Calibration points: raw≈8.5mm → actual 10.5mm; raw≈65.8mm → actual 91.0mm
        """
        voltage   = max(self.latest_ai2, 0.0)
        raw_mm    = (voltage / 3.7) * 110.0
        slope     = (91.0 - 10.5) / (65.8 - 8.5)
        offset    = 10.5 - (8.5 * slope)
        return max(0.0, round((raw_mm * slope) + offset, 1))

    def is_force_detected(self) -> bool:

        """True when RG2 TDI1 (bit 17 of DI word) is HIGH = force limit reached."""
        return bool(self.latest_di_word & (1 << 17))

    def _send_urscript(self, script: str):
        """
        Send a URScript string via the persistent port-30002 socket.
        The socket stays open — no TCP handshake overhead per call.
        Falls back to a one-shot socket if the persistent one has died.
        """
        payload = (script.strip() + "\n").encode()
        with self._urscript_lock:
            try:
                self._urscript_sock.sendall(payload)
            except Exception:
                # Persistent socket died — reconnect once
                try:
                    self._urscript_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self._urscript_sock.settimeout(5.0)
                    self._urscript_sock.connect((self.ip, self.FEEDBACK_PORT))
                    self._urscript_sock.sendall(payload)
                except Exception as e:
                    print(f"  [URScript] Send failed: {e}")

    def movel(self, x, y, z, rx, ry, rz, vel=None, acc=None,

              tol_xyz=0.003, timeout=15.0):
        """
        Linear move via URScript on port 30002.
        Primary exit: RTDE receive position poll (10ms granularity).
        Fallback exit: time estimate (dist/vel + 2s) used when RTDE receive
          is down so we don't wait the full 15s timeout after robot arrived.
        """
        dist = 0.1   # default fallback distance if RTDE read fails
        try:
            cur = self._rtde_r.getActualTCPPose()
            dist = ((cur[0]-x)**2 + (cur[1]-y)**2 + (cur[2]-z)**2) ** 0.5
            if dist < tol_xyz:
                return
        except Exception:
            pass
        v = vel if vel is not None else VEL
        a = acc if acc is not None else ACC
        # Time-based fallback: dist/vel + 2s margin (covers acc/decel ramps)
        fallback = time.time() + (dist / v) + 2.0
        rtde_ok  = True   # pessimistic: flip off if consecutive failures
        rtde_fail_count = 0
        self._send_urscript(
            f"movel(p[{x:.6f},{y:.6f},{z:.6f},{rx:.6f},{ry:.6f},{rz:.6f}],a={a},v={v})")
        deadline = time.time() + timeout
        while time.time() < deadline:
            self._pump_display()
            if self._slip_detected:
                self._send_urscript("stopl(0.5)")
                break
            try:
                cur = self._rtde_r.getActualTCPPose()
                rtde_fail_count = 0
                if ((cur[0]-x)**2 + (cur[1]-y)**2 + (cur[2]-z)**2) ** 0.5 < tol_xyz:
                    break
            except Exception:
                rtde_fail_count += 1
                if rtde_fail_count >= 5 and time.time() >= fallback:
                    break   # RTDE dead, but time says robot should be there
            time.sleep(0.01)

    def movej_pose(self, x, y, z, rx, ry, rz, vel=None, acc=None,

                   tol_xyz=0.005, timeout=20.0):
        """
        Joint move to Cartesian pose via URScript on port 30002.
        movej(p[...]) — robot solves IK internally.
        Same RTDE-poll + time fallback as movel.
        """
        dist = 0.2
        try:
            cur = self._rtde_r.getActualTCPPose()
            dist = ((cur[0]-x)**2 + (cur[1]-y)**2 + (cur[2]-z)**2) ** 0.5
            if dist < tol_xyz:
                return
        except Exception:
            pass
        v = vel if vel is not None else VEL
        a = acc if acc is not None else ACC
        fallback = time.time() + (dist / v) + 2.5
        rtde_fail_count = 0
        self._send_urscript(
            f"movej(p[{x:.6f},{y:.6f},{z:.6f},{rx:.6f},{ry:.6f},{rz:.6f}],a={a},v={v})")
        deadline = time.time() + timeout
        while time.time() < deadline:
            self._pump_display()
            if self._slip_detected:
                self._send_urscript("stopl(0.5)")
                break
            try:
                cur = self._rtde_r.getActualTCPPose()
                rtde_fail_count = 0
                if ((cur[0]-x)**2 + (cur[1]-y)**2 + (cur[2]-z)**2) ** 0.5 < tol_xyz:
                    break
            except Exception:
                rtde_fail_count += 1
                if rtde_fail_count >= 5 and time.time() >= fallback:
                    break
            time.sleep(0.01)

    def get_tcp_pose(self):

        """Return current TCP pose [x,y,z,rx,ry,rz] from RTDE receive."""
        return self._rtde_r.getActualTCPPose()

    def safety_check(self):

        """Print robot safety/program mode from dashboard."""
        mode    = self._dashboard_cmd("robotmode")
        safety  = self._dashboard_cmd("safetymode")
        running = self._dashboard_cmd("running")
        print(f"  [Safety] robot mode  : {mode}")
        print(f"  [Safety] safety mode : {safety}")
        print(f"  [Safety] running     : {running}")
        ok = "normal" in safety.lower() or "reduced" in safety.lower()
        if not ok:
            print("  [Safety] ⚠  Robot may be in a protective/emergency stop.")
            print("  [Safety]    Acknowledge on the teach pendant, then press ENTER.")
        return ok

    # ── CLIP visual check ────────────────────────────────────────────────────

    def clip_classify(self, stage: str = "Visual Check") -> tuple:

        """
        Grab the most recent camera frame and classify with CLIP.
        Does NOT flush/wait -- camera streams at 30fps so a frame is always
        available within ~100ms. Max wait is 0.4s to avoid hanging.
        """
        self._clip_result = None
        deadline = time.time() + 0.4   # was 2.0s -- camera is always streaming
        frame = None
        while time.time() < deadline:
            self._pump_display()
            with self._frame_lock:
                if self._live_frame is not None:
                    frame = self._live_frame
                    break
            time.sleep(0.03)
        if frame is None:
            return "Uncertain", 0.0
        h, w   = frame.shape[:2]
        cx, cy = w // 2, h - (CROP_H // 2) - 10
        x1 = max(0, cx - CROP_W // 2);  y1 = max(0, cy - CROP_H // 2)
        x2 = min(w, x1 + CROP_W);       y2 = min(h, y1 + CROP_H)
        crop    = frame[y1:y2, x1:x2].copy()
        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensor  = self._clip_preprocess(pil_img).unsqueeze(0).to(self._clip_device)
        with torch.no_grad():
            feat = self._clip_model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        pred  = self._clf.predict(feat.cpu().numpy())[0]
        probs = self._clf.predict_proba(feat.cpu().numpy())[0]
        # If neither class exceeds the confidence threshold, call it Uncertain
        if max(probs) < CLIP_CONFIDENCE:
            label = "Uncertain"
            conf  = float(max(probs))
        else:
            label = {0: "Empty", 1: "Holding"}[pred]
            conf  = float(probs[pred])
        # Store result — confirmed=False (banner will NOT linger yet)
        self._clip_result      = (label, conf, stage, False)
        self._clip_result_time = None
        return label, conf

    def _confirm_clip_result(self):

        """
        Call this after the pipeline has acted on the CLIP result.
        Starts the linger timer so the banner fades after CLIP_RESULT_LINGER_S.
        """
        if self._clip_result is not None:
            lbl, conf, stage, _ = self._clip_result
            self._clip_result      = (lbl, conf, stage, True)
            self._clip_result_time = time.time()

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def pick_with_failure_detection(self) -> object:

        """
        Full pick-and-place with 3-layer failure detection.
        Returns:
            True      — pick and place succeeded
            False     — all retries exhausted, object definitely missed
            "rescan"  — CLIP not confident or confirmed empty → trigger script 14
        """
        p  = self.pick_pos
        pl = self.place_pos
        dp = self.dump_pos
        px, py, pz   = p["x"],  p["y"],  p["z"]
        rx, ry, rz   = p["rx"], p["ry"], p["rz"]
        pick_z       = pz - APPROACH_OFFSET
        clip_check_z = pz + LIFT_HEIGHT_M
        plx, ply, plz     = pl["x"],  pl["y"],  pl["z"]
        pl_rx, pl_ry, pl_rz = pl["rx"], pl["ry"], pl["rz"]
        place_dip_z       = plz - APPROACH_OFFSET
        print("=" * 62)
        print("  FAILURE DETECTION PICK PIPELINE")
        print("=" * 62)
        print(f"  Pick approach  : X={px*1000:.1f}  Y={py*1000:.1f}  Z={pz*1000:.1f} mm")
        print(f"  Pick Z (dip)   : {pick_z*1000:.1f} mm  (approach - {APPROACH_OFFSET*1000:.0f} mm)")
        print(f"  CLIP check Z   : {clip_check_z*1000:.1f} mm  (approach + {LIFT_HEIGHT_M*1000:.0f} mm lift)")
        print(f"  Place approach : X={plx*1000:.1f}  Y={ply*1000:.1f}  Z={plz*1000:.1f} mm")
        print(f"  Max retries    : {MAX_RETRIES}")
        print("=" * 62 + "\n")
        for attempt in range(1, MAX_RETRIES + 1):
            print(f"\n{'─'*52}")
            print(f"  ATTEMPT {attempt} / {MAX_RETRIES}")
            print(f"{'─'*52}")
            self._clip_result = None
            # ── 1. Move to pick approach ──────────────────────────────────
            self._pipeline_status = f"Attempt {attempt}: approach"
            self._log_status(f"Attempt {attempt} -- moving to approach")
            self.movel(px, py, pz, rx, ry, rz)
            # -- 2. Open gripper (Phase 0 skips if already open) ----------
            self.open_gripper()
            # ── 3. Descend to pick Z ──────────────────────────────────────
            self._pipeline_status = "Descending"
            self._log_status(f"Descending to pick Z={pick_z*1000:.0f} mm")
            self.movel(px, py, pick_z, rx, ry, rz,
                       vel=VEL_SLOW, acc=ACC_SLOW)
            # ── 4. Close gripper ──────────────────────────────────────────
            self._pipeline_status = "Closing gripper"
            self.close_gripper_once()  # _wait_for_gripper blocks until settled
            # ── CHECK 1: Width (missed = fully closed) ────────────────────
            width = self.get_width_mm()
            print(f"\n[CHECK 1]  Width: {width:.1f} mm")
            if width < WIDTH_CLOSED_MM:
                self._log_status(f"Missed -- gripper closed ({width:.1f} mm)")
                print(f"  X  Gripper fully closed ({width:.1f} mm) -- missed object.")
                self.open_gripper()
                self.movel(px, py, pz, rx, ry, rz)
                continue
            # Something may be between the fingers -- proceed to CLIP
            print(f"  ~  Width {width:.1f} mm -- may have object (CLIP will confirm)")
            # -- 5. Lift to CLIP check height ----------------------------
            self._pipeline_status = "Lifting"
            self._log_status(f"Lifting to {clip_check_z*1000:.0f} mm")
            self.movel(px, py, clip_check_z, rx, ry, rz)
            # Post-lift width check
            width_after_lift = self.get_width_mm()
            if width_after_lift < WIDTH_CLOSED_MM:
                self._log_status(f"Object lost during lift ({width_after_lift:.1f} mm)")
                self.open_gripper()
                continue
            # -- CHECK 2: CLIP before slip monitor (saves ~1s per attempt) --------
            self._pipeline_status = "CLIP check"
            self._log_status("Running CLIP check")
            print("\n[CHECK 2]  Running CLIP visual verification...")
            label, conf = self.clip_classify(stage="Grasp Verification")
            print(f"  Result:  {label.upper()}  ({conf*100:.1f}% confidence)")
            if label == "Holding" and conf >= CLIP_CONFIDENCE:
                # Confirmed -- NOW start slip monitor for transit
                self._log_status(f"GRASP CONFIRMED -- {conf*100:.0f}% Holding")
                print(f"  OK  Grasp confirmed.")
                self._confirm_clip_result()
                self._log_status("Slip monitor active")
                self.start_slip_monitor_close()
            elif label == "Uncertain" or (label == "Holding" and conf < CLIP_CONFIDENCE):
                # UNCERTAIN only -- dump at dump_pos, then retry
                if label == "Holding":
                    reason = f"UNCERTAIN -- {conf*100:.0f}% (below threshold)"
                else:
                    reason = f"UNCERTAIN -- {conf*100:.0f}%"
                self._log_status(f"{reason} -- dumping at dump pos")
                print(f"  ?  {reason}")
                self._confirm_clip_result()
                self._pipeline_status = "Dumping"
                print("  <- Moving to dump position and releasing...")
                self.movej_pose(dp["x"], dp["y"], dp["z"],
                               dp["rx"], dp["ry"], dp["rz"])
                self.open_gripper()
                self._log_status("Dumped -- returning to approach")
                self.movel(px, py, pz, rx, ry, rz)
                continue
            else:
                # EMPTY -- open gripper and retry (no dump needed)
                reason = f"EMPTY -- {conf*100:.0f}%"
                self._log_status(f"{reason} -- releasing and retrying")
                print(f"  X  {reason}")
                self._confirm_clip_result()
                self.open_gripper()
                self.movel(px, py, pz, rx, ry, rz)
                continue
            # ── 6. Transfer to place approach ─────────────────────────────
            self._pipeline_status = "Transferring to place"
            self._log_status(f"Transferring -- X={plx*1000:.0f} Y={ply*1000:.0f}")
            self.movej_pose(plx, ply, plz, pl_rx, pl_ry, pl_rz)
            # ── CHECK: Slip during transfer ───────────────────────────────
            if self._slip_detected:
                self._pipeline_status = "Slip — re-verifying with CLIP"
                self._log_status("Slip detected -- re-verifying...")
                print("  ⚠  Slip event during transfer — re-verifying...")
                label2, conf2 = self.clip_classify(stage="Slip Re-verify")
                print(f"  Re-verify:  {label2.upper()}  ({conf2*100:.1f}%)")
                self._confirm_clip_result()
                if label2 == "Holding" and conf2 >= CLIP_CONFIDENCE:
                    self._log_status("Slip re-verify OK — continuing to place")
                    print("  ✓  Grasp still confirmed — continuing.")
                    self._slip_detected = False
                else:
                    self._log_status("Object lost during transfer — returning to pick")
                    print("  ✗  Object lost during transfer.")
                    self.open_gripper()
                    self.movej_pose(px, py, pz, rx, ry, rz)
                    continue
            # ── 7. Stop loop URP, lower and release ───────────────────────
            self._pipeline_status = "Placing"
            self._stop_and_wait()   # kill close_gripper_timed loop before placing
            with self._grip_state_lock:
                self._grip_state = "grab"
            self._monitoring_active = False
            self._log_status(f"Placing — lowering to Z={place_dip_z*1000:.0f} mm")
            self.movel(plx, ply, place_dip_z, pl_rx, pl_ry, pl_rz,
                       vel=VEL_SLOW, acc=ACC_SLOW)
            self.open_gripper()
            self._pump_for(0.3)
            # ── 8. Retract ────────────────────────────────────────────────
            self._pipeline_status = "Retracting"
            self._log_status("Placed — retracting")
            self.movel(plx, ply, plz, pl_rx, pl_ry, pl_rz)
            print("\n" + "=" * 62)
            print(f"  ✓  PICK AND PLACE COMPLETE  (attempt {attempt})")
            print("=" * 62 + "\n")
            self._pipeline_status = "DONE ✓"
            self._log_status("DONE ✓")
            return True
        # All retries exhausted
        print("\n" + "=" * 62)
        print(f"  ✗  FAILED after {MAX_RETRIES} attempts.")
        print("=" * 62 + "\n")
        self._pipeline_status = f"FAILED after {MAX_RETRIES} attempts"
        self._log_status(f"FAILED after {MAX_RETRIES} attempts")
        self.open_gripper()
        return False

    # ── Cleanup ──────────────────────────────────────────────────────────────

    def close(self):

        self.running = False
        try:
            self._rtde_r.disconnect()
        except Exception:
            pass
        try:
            self._urscript_sock.close()
        except Exception:
            pass
        try:
            self._sock.close()
        except Exception:
            pass
        try:
            self._oak_device.close()
        except Exception:
            pass
        try:
            self._rtde_r.disconnect()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("[Pipeline] Closed.")

# ===========================================================================

#  MAIN

# ===========================================================================

def load_positions():

    if not POSITIONS_FILE.exists():

        raise FileNotFoundError(
            f"Positions file not found: {POSITIONS_FILE}\n"
            f"Run 06_tcp_test.py and save 'pick_pos' and 'place_pos' first."
        )

    data = json.loads(POSITIONS_FILE.read_text(encoding="utf-8"))

    for key in ("pick_pos", "place_pos", "dump_pos"):

        if key not in data:
            raise KeyError(
                f"'{key}' not in {POSITIONS_FILE}.\n"
                f"Run 06_tcp_test.py and save 'pick_pos', 'place_pos', and 'dump_pos'."
            )

    return data["pick_pos"], data["place_pos"], data["dump_pos"]

def print_config(pick_pos, place_pos):

    print("\n" + "=" * 62)

    print("  FAILURE DETECTION PIPELINE — CONFIG")

    print("=" * 62)

    print(f"  Robot IP          : {ROBOT_IP}  (RTDE)")

    print(f"  Positions file    : {POSITIONS_FILE}")

    print(f"  CLIP probe        : {CLIP_PROBE_PATH}")

    print(f"  Pick approach     : X={pick_pos['x']*1000:.1f}  Y={pick_pos['y']*1000:.1f}  Z={pick_pos['z']*1000:.1f} mm")

    print(f"  Place approach    : X={place_pos['x']*1000:.1f}  Y={place_pos['y']*1000:.1f}  Z={place_pos['z']*1000:.1f} mm")

    print(f"  Approach offset   : {APPROACH_OFFSET*1000:.0f} mm (descend below position to pick/place)")

    print(f"  Lift height       : {LIFT_HEIGHT_M*1000:.0f} mm (above approach for CLIP check)")

    print(f"  Width closed thr  : {WIDTH_CLOSED_MM} mm  (AI2 / 3.0V * 110 — 0-10V mode)")

    print(f"  CLIP confidence   : {CLIP_CONFIDENCE*100:.0f}%")

    print(f"  Gripper settle    : {GRIPPER_SETTLE_S} s")

    print(f"  Max retries       : {MAX_RETRIES}")

    print(f"  Velocity (norm)   : {VEL} m/s  /  {ACC} m/s²")

    print(f"  Velocity (slow)   : {VEL_SLOW} m/s  /  {ACC_SLOW} m/s²")

    print("=" * 62 + "\n")

if __name__ == "__main__":

    # Force Ctrl+C to always kill the process, even when stuck inside a
    # C-extension (e.g. RTDE constructor). SIGINT → immediate os._exit.
    def _hard_exit(sig, frame):
        print("\n[Main] Ctrl+C — forcing exit.")
        os._exit(1)
    signal.signal(signal.SIGINT, _hard_exit)

    parser = argparse.ArgumentParser()

    parser.add_argument("--dry-run", action="store_true",

                        help="Print config only — no robot/camera connection.")

    args = parser.parse_args()

    pick_pos, place_pos, dump_pos = load_positions()

    print_config(pick_pos, place_pos)

    if args.dry_run:

        print("[Dry-run] Exiting.")
        sys.exit(0)

    pipeline = None

    try:

        pipeline = FailureDetectionPipeline(
            robot_ip  = ROBOT_IP,
            pick_pos  = pick_pos,
            place_pos = place_pos,
            dump_pos  = dump_pos,
        )
        tcp = pipeline.get_tcp_pose()
        if tcp:
            print(f"Live TCP from robot:")
            print(f"  X={tcp[0]*1000:.1f}  Y={tcp[1]*1000:.1f}  Z={tcp[2]*1000:.1f} mm")
            print(f"  RX={tcp[3]:.3f}  RY={tcp[4]:.3f}  RZ={tcp[5]:.3f}")
        print()
        pipeline.safety_check()
        print()
        input("Press ENTER to start the first pick sequence (Ctrl-C to abort)...\n")
        while True:
            pipeline.safety_check()
            print()
            result = pipeline.pick_with_failure_detection()
            if result == "rescan":
                print("\n[Main] CLIP not confident or EMPTY → trigger 14_recovery.py.")
            elif result:
                print("\n[Main] Pick-and-place succeeded.")
            else:
                print("\n[Main] All attempts exhausted.")
            try:
                input("\nPress ENTER to run again, Ctrl-C to quit...\n")
            except (EOFError, KeyboardInterrupt):
                raise KeyboardInterrupt

    except KeyboardInterrupt:

        print("\n[Main] Stopped by user.")

    except Exception as e:

        print(f"\n[Main] Error: {e}")
        raise

    finally:

        if pipeline:
            pipeline.close()

