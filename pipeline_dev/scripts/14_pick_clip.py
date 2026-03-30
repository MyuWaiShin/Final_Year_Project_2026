"""
14_pick_clip.py
===============
Clean pick pipeline — pure socket URScript motion, no RTDE.

Sequence
--------
  1. Move to scan_pos  (open gripper here)
  2. Move to pick_pos  (approach — saved Z)
  3. Descend 100 mm    (pick_pos.z - 0.100)
  4. Close gripper
  5. Lift 200 mm       (pick_pos.z + 0.200 above approach)
  6. CLIP check        → Holding / Empty / Uncertain  (75% threshold)

Motion
------
  All moves via raw URScript over a persistent TCP socket to port 30002.
  Completion is polled by reading the robot's secondary-client packet stream
  (same port 30002, separate socket) — no RTDE required at all.

Gripper
-------
  RG2 controlled via Dashboard URP on port 29999.
  open_gripper.urp / close_gripper.urp

Display
-------
  OAK-D live feed + status text overlay via OpenCV.
"""

# ── stdlib ──────────────────────────────────────────────────────────────────
import json
import math
import os
import pickle
import signal
import socket
import struct
import sys
import threading
import time
from pathlib import Path
from collections import deque

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR   = SCRIPT_DIR.parent

# ── optional heavy deps ──────────────────────────────────────────────────────
try:
    import torch
    import clip as openai_clip
    from PIL import Image
    CLIP_OK = True
except ImportError:
    CLIP_OK = False

try:
    import depthai as dai
    import cv2
    import numpy as np
    CAM_OK = True
except ImportError:
    CAM_OK = False

# ===========================================================================
#  CONFIG
# ===========================================================================

ROBOT_IP        = "192.168.8.102"
POSITIONS_FILE  = BASE_DIR / "data" / "saved_positions.json"
CLIP_PROBE_PATH = BASE_DIR.parent / "CLIP_post_grab" / "clip_probe.pkl"

DESCEND_M      = 0.100   # how far below pick_pos.z to go for the grab
LIFT_M         = 0.200   # how far above pick_pos.z to lift for CLIP check

WIDTH_CLOSED_MM = 11.0   # gripper fully closed threshold (missed object)
CLIP_THRESHOLD  = 0.75   # 75% confidence required

# URScript speeds
V_FAST = 0.4;  A_FAST = 0.2   # approach / transit
V_SLOW = 0.1; A_SLOW = 0.1  # descend / lift near object

# Dashboard URP paths (on robot controller)
URP_OPEN        = "/programs/myu/open_gripper.urp"
URP_CLOSE       = "/programs/myu/close_gripper.urp"
URP_CLOSE_LOOP  = "/programs/myu/close_gripper_timed.urp"  # loops: re-closes every ~0.2s

# Slip detection
SLIP_DROP_MM  = 8.0    # width must drop this far below initial before slip is flagged
SLIP_POLL_S   = 0.08   # how often the slip monitor samples width

# OAK-D crop for CLIP (centred, bottom of frame where gripper appears)
CROP_W = 1400
CROP_H = 600

DASHBOARD_PORT = 29999
URSCRIPT_PORT  = 30002   # send motion commands here
SENSOR_PORT    = 30002   # read robot state packets here (separate socket)

TOL_XYZ = 0.004          # metres — "arrived" tolerance
MOVE_TIMEOUT = 30.0      # max seconds to wait for any single move

# ===========================================================================
#  ROBOT COMMUNICATION
# ===========================================================================

class Robot:
    """
    Thin wrapper around two persistent TCP connections to the UR10:

      _cmd_sock   — write-only: sends URScript lines to port 30002
      _state_sock — read-only:  parses secondary-client packets to get TCP pose
    """

    def __init__(self, ip: str):
        self.ip      = ip
        self.running = True

        # ── gripper sensor values (updated by _parse_packet) ─────────────────
        self.latest_ai2      = 0.0   # AI2 voltage → gripper width
        self.latest_di_word  = 0     # 64-bit digital input word (DI8 = bit 17)

        # ── state socket (read TCP pose + sensor data from secondary stream) ──
        print(f"[Robot] Connecting state socket ({ip}:{SENSOR_PORT})...")
        self._state_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._state_sock.settimeout(5.0)
        self._state_sock.connect((ip, SENSOR_PORT))
        self._tcp_pose  = [0.0] * 6   # [x,y,z,rx,ry,rz] metres / radians
        self._pose_lock = threading.Lock()
        threading.Thread(target=self._state_loop, daemon=True, name="StateLoop").start()
        print("[Robot] State socket ready.")

        # ── command socket (send URScript) ───────────────────────────────────
        print(f"[Robot] Connecting command socket ({ip}:{URSCRIPT_PORT})...")
        self._cmd_sock  = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._cmd_sock.settimeout(5.0)
        self._cmd_sock.connect((ip, URSCRIPT_PORT))
        self._cmd_lock  = threading.Lock()
        # drain thread: port 30002 pushes data to every connected client;
        # if we never read _cmd_sock, the OS buffer fills and sendall() blocks.
        threading.Thread(target=self._drain_cmd, daemon=True, name="CmdDrain").start()
        print("[Robot] Command socket ready.")

        # wait until we have a valid pose reading
        deadline = time.time() + 5.0
        while time.time() < deadline:
            with self._pose_lock:
                ok = any(v != 0.0 for v in self._tcp_pose)
            if ok:
                break
            time.sleep(0.05)

    # ── internal: secondary client state packets ─────────────────────────────

    def _state_loop(self):
        """Parse UR secondary-client packets to extract TCP Cartesian pose."""
        buf = b""
        while self.running:
            try:
                chunk = self._state_sock.recv(4096)
                if not chunk:
                    time.sleep(0.01)
                    continue
                buf += chunk
                # packets: 4-byte total length + 1-byte type + payload
                while len(buf) >= 5:
                    pkt_len = struct.unpack("!I", buf[:4])[0]
                    if len(buf) < pkt_len:
                        break
                    pkt     = buf[:pkt_len]
                    buf     = buf[pkt_len:]
                    self._parse_packet(pkt)
            except Exception:
                time.sleep(0.01)

    def _parse_packet(self, pkt: bytes):
        """
        Secondary client: outer packet type=16 (Robot State).
        Sub-packet type=4 (Cartesian Info) contains:
          offset 5: TCP x (8-byte double)  — 6 values = 48 bytes
        """
        if len(pkt) < 5:
            return
        outer_type = pkt[4]
        if outer_type != 16:
            return
        # walk sub-packets
        offset = 5
        while offset + 5 <= len(pkt):
            if offset + 4 > len(pkt):
                break
            sp_len  = struct.unpack("!I", pkt[offset:offset+4])[0]
            sp_type = pkt[offset+4]
            if sp_len == 0:
                break
            sp_data = pkt[offset:offset+sp_len]
            if sp_type == 4 and sp_len >= 5 + 48:
                # Cartesian Info — 6 doubles starting at byte 5
                vals = struct.unpack("!6d", sp_data[5:5+48])
                with self._pose_lock:
                    self._tcp_pose = list(vals)
            elif sp_type == 2 and sp_len >= 15:
                # Tool Data — AI2 voltage (8-byte double) at byte 7
                self.latest_ai2 = struct.unpack("!d", sp_data[7:15])[0]
            elif sp_type == 3 and sp_len >= 13:
                # Masterboard Data — 64-bit DI word at byte 5
                self.latest_di_word = struct.unpack("!Q", sp_data[5:13])[0]
            offset += sp_len

    def _drain_cmd(self):
        """Silently discard data the robot pushes to _cmd_sock."""
        while self.running:
            try:
                self._cmd_sock.recv(4096)
            except Exception:
                time.sleep(0.01)

    # ── public API ────────────────────────────────────────────────────────────

    def get_pose(self):
        """Return current TCP pose [x,y,z,rx,ry,rz]."""
        with self._pose_lock:
            return list(self._tcp_pose)

    def get_width_mm(self) -> float:
        """
        RG2 finger width from AI2 voltage (tool flange, 0-3.7V range).
        Two-point calibration: raw~8.5mm->actual 10.5mm, raw~65.8mm->actual 91.0mm.
        """
        voltage = max(self.latest_ai2, 0.0)
        raw_mm  = (voltage / 3.7) * 110.0
        slope   = (91.0 - 10.5) / (65.8 - 8.5)
        offset  = 10.5 - (8.5 * slope)
        return max(0.0, round(raw_mm * slope + offset, 1))

    def is_contact(self) -> bool:
        """True when RG2 TDI1 (bit 17 of DI word) is HIGH = force limit reached."""
        return bool(self.latest_di_word & (1 << 17))

    def send(self, script: str):
        """Send a single-line URScript command."""
        payload = (script.strip() + "\n").encode()
        with self._cmd_lock:
            try:
                self._cmd_sock.sendall(payload)
            except Exception:
                # reconnect once
                try:
                    self._cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self._cmd_sock.settimeout(5.0)
                    self._cmd_sock.connect((self.ip, URSCRIPT_PORT))
                    self._cmd_sock.sendall(payload)
                    threading.Thread(target=self._drain_cmd, daemon=True).start()
                except Exception as e:
                    print(f"[Robot] send failed: {e}")

    def movel(self, x, y, z, rx, ry, rz, v=None, a=None,
              tol=TOL_XYZ, timeout=MOVE_TIMEOUT,
              pump_fn=None):
        """
        Blocking linear move. Polls TCP pose until within tol of target.
        pump_fn (optional): called ~every 50ms to keep a display alive.
        Falls back to time-estimate if pose never updates.
        """
        v = v or V_FAST
        a = a or A_FAST
        t0 = time.time()
        cur = self.get_pose()
        dist = math.sqrt((cur[0]-x)**2 + (cur[1]-y)**2 + (cur[2]-z)**2)
        if dist < tol:
            return
        fallback = t0 + dist / v + 3.0
        self.send(f"movel(p[{x:.6f},{y:.6f},{z:.6f},{rx:.6f},{ry:.6f},{rz:.6f}],a={a},v={v})")
        deadline = t0 + timeout
        while time.time() < deadline:
            if pump_fn:
                pump_fn()
            cur = self.get_pose()
            d = math.sqrt((cur[0]-x)**2 + (cur[1]-y)**2 + (cur[2]-z)**2)
            if d < tol:
                return
            if time.time() >= fallback:
                # time estimate elapsed — assume arrived
                return
            time.sleep(0.02)

    def movej_pose(self, x, y, z, rx, ry, rz, v=None, a=None,
                   tol=TOL_XYZ, timeout=MOVE_TIMEOUT, pump_fn=None,
                   stop_event: threading.Event = None):
        """
        Blocking joint move to Cartesian pose (movej IK).
        stop_event: if set mid-move, sends stopl and returns immediately.
        """
        v = v or V_FAST
        a = a or A_FAST
        t0 = time.time()
        cur = self.get_pose()
        dist = math.sqrt((cur[0]-x)**2 + (cur[1]-y)**2 + (cur[2]-z)**2)
        if dist < tol:
            return
        fallback = t0 + dist / v + 3.0
        self.send(f"movej(p[{x:.6f},{y:.6f},{z:.6f},{rx:.6f},{ry:.6f},{rz:.6f}],a={a},v={v})")
        deadline = t0 + timeout
        while time.time() < deadline:
            if stop_event and stop_event.is_set():
                self.send("stopl(0.5)")
                return
            if pump_fn:
                pump_fn()
            cur = self.get_pose()
            d = math.sqrt((cur[0]-x)**2 + (cur[1]-y)**2 + (cur[2]-z)**2)
            if d < tol:
                return
            if time.time() >= fallback:
                return
            time.sleep(0.02)

    def movej_with_grip_loop(self, x, y, z, rx, ry, rz, v=None, a=None,
                              grip_force=40, grip_interval=0.2,
                              tol=TOL_XYZ, timeout=MOVE_TIMEOUT,
                              pump_fn=None, stop_event=None):
        """
        Joint move to Cartesian pose WITH a concurrent RG2 re-close thread.

        Sends a complete multi-line URScript program to port 30002 that runs:
          - A background thread calling rg_grip(force, 0) every grip_interval s
          - movej to the target pose
        Both run in the SAME interpreter slot, so they coexist without conflict.

        When movej completes the program ends and the grip thread is killed.
        If stop_event fires, sends stopl(0.5) which interrupts the program.
        """
        v = v or V_FAST
        a = a or A_FAST
        t0   = time.time()
        cur  = self.get_pose()
        dist = math.sqrt((cur[0]-x)**2 + (cur[1]-y)**2 + (cur[2]-z)**2)
        if dist < tol:
            return
        fallback = t0 + dist / v + 4.0
        # Complete URScript program: gripper thread + motion in same execution context
        script = (
            "def grip_transit():\n"
            "  thread Tgrip():\n"
            "    while (True):\n"
            f"      rg_grip({grip_force}, 0)\n"
            f"      sleep({grip_interval})\n"
            "    end\n"
            "  end\n"
            "  run Tgrip()\n"
            f"  movej(p[{x:.6f},{y:.6f},{z:.6f},{rx:.6f},{ry:.6f},{rz:.6f}],a={a},v={v})\n"
            "end\n"
            "grip_transit()\n"
        )
        self.send(script)
        deadline = t0 + timeout
        while time.time() < deadline:
            if stop_event and stop_event.is_set():
                self.send("stopl(0.5)")
                return
            if pump_fn:
                pump_fn()
            cur = self.get_pose()
            d = math.sqrt((cur[0]-x)**2 + (cur[1]-y)**2 + (cur[2]-z)**2)
            if d < tol:
                return
            if time.time() >= fallback:
                return
            time.sleep(0.02)

    def close(self):
        self.running = False
        try: self._state_sock.close()
        except: pass
        try: self._cmd_sock.close()
        except: pass


# ===========================================================================
#  DASHBOARD (gripper URP)
# ===========================================================================

def dashboard_cmd(ip: str, cmd: str, retries=3) -> str:
    for attempt in range(1, retries + 1):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5.0)
            s.connect((ip, DASHBOARD_PORT))
            s.recv(1024)                       # greeting
            s.sendall((cmd + "\n").encode())
            resp = s.recv(1024).decode().strip()
            s.close()
            return resp
        except Exception as e:
            print(f"  [Dashboard] {cmd!r}: {e} (attempt {attempt}/{retries})")
            if attempt < retries:
                time.sleep(0.4)
    return ""

_last_urp = [None]

def play_urp(ip: str, program: str, wait_s: float = 2.5):
    """Load + play a URP, then wait wait_s for it to finish."""
    if _last_urp[0] != program:
        # stop whatever is running, load new program
        dashboard_cmd(ip, "stop")
        time.sleep(0.1)
        dashboard_cmd(ip, f"load {program}")
        _last_urp[0] = program
        time.sleep(0.08)
    else:
        dashboard_cmd(ip, "stop")
        time.sleep(0.05)
    dashboard_cmd(ip, "play")
    time.sleep(wait_s)   # RG2 URPs are short — just wait for completion


# ===========================================================================
#  CAMERA + CLIP
# ===========================================================================

def open_camera():
    """Open OAK-D, return (device, queue)."""
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
    q = device.getOutputQueue("video", maxSize=1, blocking=False)
    return device, q


def load_clip(probe_path: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CLIP] Loading ViT-B/32 on {device}…")
    model, preprocess = openai_clip.load("ViT-B/32", device=device)
    with open(probe_path, "rb") as f:
        clf = pickle.load(f)
    print("[CLIP] Probe loaded.")
    return model, preprocess, clf, device


def clip_classify(frame, model, preprocess, clf, device) -> tuple:
    """
    Crop bottom-centre of frame and classify with CLIP.
    Returns (label, confidence):
      label = "Holding" or "Empty"   if winning class >= CLIP_THRESHOLD
      label = "Uncertain"             if BOTH classes are below CLIP_THRESHOLD
                                       (i.e. the model is not confident either way)
    """
    h, w   = frame.shape[:2]
    cx, cy = w // 2, h - CROP_H // 2 - 10
    x1 = max(0, cx - CROP_W // 2)
    y1 = max(0, cy - CROP_H // 2)
    x2 = min(w, x1 + CROP_W)
    y2 = min(h, y1 + CROP_H)
    crop    = frame[y1:y2, x1:x2].copy()
    pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    tensor  = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    pred  = clf.predict(feat.cpu().numpy())[0]
    probs = clf.predict_proba(feat.cpu().numpy())[0]
    if max(probs) < CLIP_THRESHOLD:
        return "Uncertain", float(max(probs))
    label = {0: "Empty", 1: "Holding"}[pred]
    return label, float(probs[pred])


# ===========================================================================
#  DISPLAY
# ===========================================================================

STATUS_COLOURS = {
    "default":   (220, 220, 220),
    "scan":      (255, 200,  50),
    "pick":      (100, 200, 255),
    "descend":   (100, 255, 200),
    "grab":      (255, 160,  50),
    "lift":      (100, 255, 100),
    "clip":      (255, 255, 255),
    "done_hold": ( 50, 220,  50),
    "done_empty":(  50,  50, 220),
    "done_unc":  (  50, 180, 255),
}

STAGE_INFO = {
    "scan":     ("Moving to Scan position",                  "scan"),
    "pick":     ("Moving to Pick pos",                       "pick"),
    "descend":  ("Descending...",                            "descend"),
    "grab":     ("Grabbing object...",                       "grab"),
    "lift":     ("Lifting object...",                        "lift"),
    "clip":     ("Confirming grasp (CLIP)",                  "clip"),
    "hold":     ("HOLDING",                                  "done_hold"),
    "empty":    ("EMPTY",                                    "done_empty"),
    "unc":      ("UNCERTAIN",                                "done_unc"),
    "placing":  ("Moving to Place...",                       "pick"),
    "place_dip":("Releasing at Place...",                    "descend"),
    "dumping":  ("Uncertain grasp - moving to Dump",         "done_unc"),
    "dump_dip": ("Releasing at Dump...",                     "done_unc"),
    "dump_ret": ("Returning to Scan pos...",                 "done_unc"),
    "recovery": ("EMPTY - Recovery triggered (returning to Scan pos)",  "done_empty"),
    "missed":   ("Missed grasp - recovery triggered",        "done_empty"),
    "place_ret":("Returning to Scan...",                     "done_hold"),
    "transit":  ("Transferring to Place (slip monitor ON)",  "done_hold"),
    "slip":     ("SLIP detected - re-verifying...",          "done_unc"),
    "slip_lost":("Object lost during transit",               "done_empty"),
    "slip_unc": ("Transit uncertain - dumping safely",       "done_unc"),
    "idle":     ("Idle",                                    "default"),
}


class Display:
    """Threadsafe live display with status overlay on OAK-D feed."""

    def __init__(self, q):
        self._q         = q
        self._frame     = None
        self._lock      = threading.Lock()
        self._stage     = "idle"
        self._clip_text = ""
        threading.Thread(target=self._decode, daemon=True, name="CamDecode").start()
        cv2.namedWindow("Pick Monitor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Pick Monitor", 960, 540)

    def _decode(self):
        while True:
            try:
                pkt = self._q.tryGet()
                if pkt is not None:
                    f = pkt.getCvFrame()
                    if f is not None:
                        with self._lock:
                            self._frame = f
                else:
                    time.sleep(0.005)
            except Exception:
                time.sleep(0.01)

    def set_stage(self, stage: str, clip_text: str = ""):
        self._stage     = stage
        self._clip_text = clip_text

    def get_frame(self):
        with self._lock:
            return self._frame

    def pump(self):
        """Call from main thread to refresh window. Returns False if 'q' pressed."""
        with self._lock:
            frame = self._frame
        if frame is None:
            cv2.waitKey(1)
            return True

        disp = cv2.resize(frame, (960, 540))
        h, w = disp.shape[:2]

        # ── CLIP crop box (orange) ───────────────────────────────────────────
        fh, fw = frame.shape[:2]
        sx, sy = w / fw, h / fh
        cx, cy = fw // 2, fh - CROP_H // 2 - 10
        bx1 = int(max(0, cx - CROP_W // 2) * sx)
        by1 = int(max(0, cy - CROP_H // 2) * sy)
        bx2 = int(min(fw, cx + CROP_W // 2) * sx)
        by2 = int(min(fh, cy + CROP_H // 2) * sy)
        cv2.rectangle(disp, (bx1, by1), (bx2, by2), (0, 140, 255), 2)

        # ── status banner ────────────────────────────────────────────────────
        st    = self._stage
        label, ckey = STAGE_INFO.get(st, ("...", "default"))
        colour = STATUS_COLOURS.get(ckey, STATUS_COLOURS["default"])

        # semi-transparent dark background at top
        banner_h = 56
        overlay  = disp.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.72, disp, 0.28, 0, disp)

        cv2.putText(disp, label, (14, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2, cv2.LINE_AA)

        # CLIP result sub-text
        if self._clip_text:
            cv2.putText(disp, self._clip_text, (14, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 1, cv2.LINE_AA)

        cv2.imshow("Pick Monitor", disp)
        return cv2.waitKey(1) != ord("q")


# ===========================================================================
#  MAIN PIPELINE
# ===========================================================================

def load_positions():
    if not POSITIONS_FILE.exists():
        raise FileNotFoundError(f"Positions file not found: {POSITIONS_FILE}")
    data = json.loads(POSITIONS_FILE.read_text(encoding="utf-8"))
    for key in ("scan_pos", "pick_pos", "place_pos", "dump_pos"):
        if key not in data:
            raise KeyError(f"'{key}' missing from {POSITIONS_FILE}")
    return data["scan_pos"], data["pick_pos"], data["place_pos"], data["dump_pos"]


def run_once(robot: Robot, display: Display,
             scan: dict, pick: dict, place: dict, dump: dict,
             clip_model, clip_pre, clf, clip_device):

    pump = display.pump    # shortcut

    sx, sy, sz = scan["x"], scan["y"], scan["z"]
    srx, sry, srz = scan["rx"], scan["ry"], scan["rz"]

    px, py, pz  = pick["x"], pick["y"], pick["z"]
    prx, pry, prz = pick["rx"], pick["ry"], pick["rz"]

    plx, ply, plz    = place["x"], place["y"], place["z"]
    pl_rx, pl_ry, pl_rz = place["rx"], place["ry"], place["rz"]

    dx, dy, dz    = dump["x"], dump["y"], dump["z"]
    drx, dry, drz = dump["rx"], dump["ry"], dump["rz"]

    pick_dip_z  = pz - DESCEND_M
    clip_z      = pz + LIFT_M       # above approach
    place_dip_z = plz - DESCEND_M
    dump_dip_z  = dz  - DESCEND_M

    # ── 1. Scan position ─────────────────────────────────────────────────────
    print("\n[1] Moving to scan position…")
    display.set_stage("scan")
    robot.movej_pose(sx, sy, sz, srx, sry, srz,
                     v=V_FAST, a=A_FAST, pump_fn=pump)

    print("    Opening gripper…")
    play_urp(robot.ip, URP_OPEN, wait_s=2.5)

    # ── 2. Pick approach ──────────────────────────────────────────────────────
    print("[2] Moving to pick position…")
    display.set_stage("pick")
    robot.movej_pose(px, py, pz, prx, pry, prz,
                     v=V_FAST, a=A_FAST, pump_fn=pump)

    # ── 3. Descend ────────────────────────────────────────────────────────────
    print(f"[3] Descending {DESCEND_M*1000:.0f} mm to Z={pick_dip_z*1000:.1f} mm…")
    display.set_stage("descend")
    robot.movel(px, py, pick_dip_z, prx, pry, prz,
                v=V_SLOW, a=A_SLOW, pump_fn=pump)

    # ── 4. Close gripper ──────────────────────────────────────────────────────
    print("[4] Closing gripper...")
    display.set_stage("grab")
    play_urp(robot.ip, URP_CLOSE, wait_s=2.5)

    # ── CHECK 1: IO sensors (DI8 contact + AI2 width) ─────────────────────────
    time.sleep(0.2)   # let sensor values settle after gripper stops
    width   = robot.get_width_mm()
    contact = robot.is_contact()
    print(f"  [IO Check]  Width: {width:.1f} mm  |  Contact (DI8): {contact}")

    # PROCEED if: width > 11mm   (object spreading fingers)
    #         OR  contact AND width > 11mm  (force limit hit with object present)
    # MISSED  if: width <= 11mm  (fully closed = missed, OR contact at <=11 = self-contact)
    has_object = width > WIDTH_CLOSED_MM
    self_contact = contact and not has_object   # contact but fingers closed = touching itself

    missed = not has_object   # width > 11mm is the primary gate; contact alone at <=11 = missed
    if missed:
        if self_contact:
            reason = f"width={width:.1f}mm + contact (self-contact, fingers touching)"
        else:
            reason = f"width={width:.1f}mm (fully closed, missed object)"
        print(f"  [IO] Missed grasp: {reason} -- recovery triggered")
        display.set_stage("missed", f"Missed: {reason}")
        time.sleep(0.5)
        play_urp(robot.ip, URP_OPEN, wait_s=2.0)
        print("  [IO] Returning to scan position...")
        robot.movej_pose(sx, sy, sz, srx, sry, srz,
                         v=V_FAST, a=A_FAST, pump_fn=pump)
        display.set_stage("idle")
        return "missed", 0.0

    status = f"width={width:.1f}mm" + ("  contact=True" if contact else "")
    print(f"  [IO] Grasp plausible ({status}) -- proceeding to lift")

    # ── 5. Lift ───────────────────────────────────────────────────────────────
    print(f"[5] Lifting to Z={clip_z*1000:.1f} mm…")
    display.set_stage("lift")
    robot.movel(px, py, clip_z, prx, pry, prz,
                v=V_SLOW, a=A_SLOW, pump_fn=pump)

    # ── 6. CLIP grasp check ───────────────────────────────────────────────────
    print("[6] Running CLIP grasp check…")
    display.set_stage("clip")
    pump()
    time.sleep(0.1)  # tiny settle so camera has a fresh frame

    frame = display.get_frame()
    if frame is None:
        print("    [!] No camera frame — skipping CLIP")
        label, conf = "Uncertain", 0.0
    else:
        label, conf = clip_classify(frame, clip_model, clip_pre, clf, clip_device)

    conf_pct = conf * 100.0
    print(f"\n{'='*50}")
    print(f"  CLIP RESULT:  {label.upper()}  ({conf_pct:.1f}%)")
    print(f"{'='*50}\n")

    clip_text = f"CLIP: {label}  {conf_pct:.1f}%  (thr {CLIP_THRESHOLD*100:.0f}%)"

    if label == "Holding":
        # ── HOLDING → start slip monitor → transfer → place ───────────────
        display.set_stage("hold", clip_text)
        pump(); time.sleep(0.3)

        # Record width now (post-lift, confirmed holding) as baseline for slip
        initial_width = robot.get_width_mm()
        print(f"  [Slip] Baseline width: {initial_width:.1f} mm")

        # Snapshot DI8 contact state at pick (HIGH = contact confirmed)
        initial_di8 = robot.is_contact()

        # ── Slip monitor: fires on width drop OR DI8 contact loss ────────────
        slip_evt = threading.Event()
        slipped  = [False]

        def _slip_monitor():
            # Allow one DI8 wobble before treating LOW as real loss
            di8_low_count = [0]
            while not slip_evt.is_set():
                w   = robot.get_width_mm()
                di8 = robot.is_contact()

                # Trigger 1: width dropped more than threshold
                if initial_width - w > SLIP_DROP_MM:
                    print(f"\n  [Slip] Width dropped {initial_width:.1f} -> {w:.1f} mm -- SLIP!")
                    slipped[0] = True
                    slip_evt.set()
                    return

                # Trigger 2: DI8 contact lost (was HIGH at pick, now LOW)
                if initial_di8 and not di8:
                    di8_low_count[0] += 1
                    if di8_low_count[0] >= 2:   # 2 consecutive lows = real loss
                        print(f"\n  [Slip] DI8 contact lost (width={w:.1f}mm) -- SLIP!")
                        slipped[0] = True
                        slip_evt.set()
                        return
                else:
                    di8_low_count[0] = 0

                time.sleep(SLIP_POLL_S)

        # ── GripReplay: restarts URP whenever it stops (one-shot workaround) ─
        stop_replay = threading.Event()

        def _grip_replay():
            try:
                ds = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                ds.settimeout(1.0)
                ds.connect((robot.ip, DASHBOARD_PORT))
                ds.recv(1024)   # consume greeting
                while not stop_replay.is_set():
                    time.sleep(0.2)
                    if stop_replay.is_set():
                        break
                    try:
                        ds.sendall(b"programState\n")
                        resp = ds.recv(256).decode(errors="ignore").strip()
                        if "STOPPED" in resp:
                            ds.sendall(b"play\n")
                            ds.recv(256)
                    except Exception:
                        break
                try: ds.close()
                except: pass
            except Exception as e:
                print(f"  [GripReplay] error: {e}")

        threading.Thread(target=_slip_monitor, daemon=True, name="SlipMon").start()
        threading.Thread(target=_grip_replay,  daemon=True, name="GripReplay").start()

        # Start close loop URP, wait for first rg_grip cycle, then transit
        print("[7] Starting grip loop URP...")
        dashboard_cmd(robot.ip, "stop")
        time.sleep(0.08)
        if _last_urp[0] != URP_CLOSE_LOOP:
            r = dashboard_cmd(robot.ip, f"load {URP_CLOSE_LOOP}")
            print(f"  [Loop] load: {r!r}")
            _last_urp[0] = URP_CLOSE_LOOP
            time.sleep(0.1)
        r = dashboard_cmd(robot.ip, "play")
        print(f"  [Loop] play: {r!r}")
        time.sleep(0.3)   # let URP fire first rg_grip before arm moves

        print("[8] Transferring to PLACE (loop URP + slip monitor active)...")
        display.set_stage("transit", clip_text)
        robot.movej_pose(plx, ply, plz, pl_rx, pl_ry, pl_rz,
                         v=V_FAST, a=A_FAST, pump_fn=pump,
                         stop_event=slip_evt)

        # Transit done — stop both background threads
        slip_evt.set()
        stop_replay.set()


        if slipped[0]:
            # ── SLIP DETECTED ──────────────────────────────────────────────
            print("\n  [Slip] Slip detected mid-transit -- re-verifying with CLIP...")
            display.set_stage("slip", "Slip detected - re-verifying...")
            pump(); time.sleep(0.15)

            slip_frame = display.get_frame()
            if slip_frame is None:
                slip_label, slip_conf = "Uncertain", 0.0
            else:
                slip_label, slip_conf = clip_classify(
                    slip_frame, clip_model, clip_pre, clf, clip_device)
            print(f"  [Slip] Re-verify: {slip_label.upper()}")
            if slip_label == "Uncertain":
                slip_label = "Empty"  # treat uncertain post-slip as object lost

            if slip_label == "Holding":
                # Still holding — re-close and continue
                slip_text = f"Slip re-verify: HOLDING -- re-closing and continuing"
                display.set_stage("hold", slip_text)
                pump(); time.sleep(0.3)
                print("  [Slip] Grasp still confirmed -- continuing to place")
                # Re-close gripper since it partially slipped
                dashboard_cmd(robot.ip, "stop")
                time.sleep(0.08)
                if _last_urp[0] != URP_CLOSE_LOOP:
                    dashboard_cmd(robot.ip, f"load {URP_CLOSE_LOOP}")
                    _last_urp[0] = URP_CLOSE_LOOP
                    time.sleep(0.1)
                dashboard_cmd(robot.ip, "play")
                time.sleep(1.5)
                do_place = True

            else:  # Empty or Uncertain-treated-as-Empty — object gone
                slip_text = "Object lost during transit -- recovering"
                display.set_stage("slip_lost", slip_text)
                pump(); time.sleep(0.3)
                print("  [Slip] Object lost -- opening gripper, returning to scan to retry")
                play_urp(robot.ip, URP_OPEN, wait_s=2.0)
                robot.movej_pose(sx, sy, sz, srx, sry, srz,
                                 v=V_FAST, a=A_FAST, pump_fn=pump)
                do_place = False
        else:
            do_place = True   # no slip, normal transit complete

        if do_place:
            # ── Place ─────────────────────────────────────────────────────
            print(f"[8] Lowering to place Z={place_dip_z*1000:.1f} mm...")
            display.set_stage("place_dip", clip_text)
            robot.movel(plx, ply, place_dip_z, pl_rx, pl_ry, pl_rz,
                        v=V_SLOW, a=A_SLOW, pump_fn=pump)

            print("[9] Opening gripper (place)...")
            play_urp(robot.ip, URP_OPEN, wait_s=2.0)

            print("[10] Retracting from PLACE...")
            robot.movel(plx, ply, plz, pl_rx, pl_ry, pl_rz,
                        v=V_FAST, a=A_FAST, pump_fn=pump)

            print("[11] Returning to scan position...")
            display.set_stage("place_ret", clip_text)
            robot.movej_pose(sx, sy, sz, srx, sry, srz,
                             v=V_FAST, a=A_FAST, pump_fn=pump)

    elif label == "Uncertain":
        # ── UNCERTAIN → dump ──────────────────────────────────────────────
        display.set_stage("unc", clip_text)
        pump(); time.sleep(0.5)

        print("[7] Uncertain grasp — moving to dump position…")
        display.set_stage("dumping", clip_text)
        robot.movej_pose(dx, dy, dz, drx, dry, drz,
                         v=V_FAST, a=A_FAST, pump_fn=pump)

        print(f"[8] Lowering to dump Z={dump_dip_z*1000:.1f} mm…")
        display.set_stage("dump_dip", clip_text)
        robot.movel(dx, dy, dump_dip_z, drx, dry, drz,
                    v=V_SLOW, a=A_SLOW, pump_fn=pump)

        print("[9] Opening gripper...")
        play_urp(robot.ip, URP_OPEN, wait_s=2.0)

        print("[10] Retracting from dump...")
        robot.movel(dx, dy, dz, drx, dry, drz,
                    v=V_FAST, a=A_FAST, pump_fn=pump)

        print("[11] Returning to scan position...")
        display.set_stage("dump_ret", clip_text)
        robot.movej_pose(sx, sy, sz, srx, sry, srz,
                         v=V_FAST, a=A_FAST, pump_fn=pump)

    else:
        # ── EMPTY → recovery (return to scan) ─────────────────────────────
        display.set_stage("empty", clip_text)
        pump(); time.sleep(0.5)

        print("[7] Empty — Recovery triggered: returning to SCAN position…")
        display.set_stage("recovery", clip_text)
        robot.movej_pose(sx, sy, sz, srx, sry, srz,
                         v=V_FAST, a=A_FAST, pump_fn=pump)

    # brief result hold before next prompt
    t_end = time.time() + 2.0
    while time.time() < t_end:
        pump()
        time.sleep(0.05)

    return label, conf


# ===========================================================================
#  ENTRY POINT
# ===========================================================================

if __name__ == "__main__":

    # Ctrl+C → hard exit even inside C extensions
    def _hard_exit(sig, frame):
        print("\n[Main] Ctrl+C — exiting.")
        os._exit(1)
    signal.signal(signal.SIGINT, _hard_exit)

    # ── dependency checks ─────────────────────────────────────────────────────
    if not CLIP_OK:
        sys.exit("[Error] CLIP not installed. "
                 "pip install git+https://github.com/openai/CLIP.git")
    if not CAM_OK:
        sys.exit("[Error] depthai / cv2 not installed.")

    probe_path = Path(CLIP_PROBE_PATH).resolve()
    if not probe_path.exists():
        sys.exit(f"[Error] CLIP probe not found: {probe_path}")

    # ── load positions ────────────────────────────────────────────────────────
    scan_pos, pick_pos, place_pos, dump_pos = load_positions()

    print("\n" + "="*54)
    print("  PICK + CLIP CHECK — CONFIG")
    print("="*54)
    print(f"  Robot IP    : {ROBOT_IP}")
    print(f"  Scan pos    : X={scan_pos['x']*1000:.1f}  Y={scan_pos['y']*1000:.1f}  Z={scan_pos['z']*1000:.1f} mm")
    print(f"  Pick pos    : X={pick_pos['x']*1000:.1f}  Y={pick_pos['y']*1000:.1f}  Z={pick_pos['z']*1000:.1f} mm")
    print(f"  Place pos   : X={place_pos['x']*1000:.1f}  Y={place_pos['y']*1000:.1f}  Z={place_pos['z']*1000:.1f} mm")
    print(f"  Dump pos    : X={dump_pos['x']*1000:.1f}  Y={dump_pos['y']*1000:.1f}  Z={dump_pos['z']*1000:.1f} mm")
    print(f"  Descend     : {DESCEND_M*1000:.0f} mm  →  pick Z={( pick_pos['z']-DESCEND_M)*1000:.1f} mm")
    print(f"  Lift : {LIFT_M*1000:.0f} mm  →  Z={( pick_pos['z']+LIFT_M)*1000:.1f} mm")
    print(f"  CLIP thresh : {CLIP_THRESHOLD*100:.0f}%")
    print(f"  CLIP probe  : {probe_path}")
    print("="*54 + "\n")

    # ── connect ───────────────────────────────────────────────────────────────
    clip_model, clip_pre, clf, clip_device = load_clip(probe_path)

    print("[Init] Opening camera…")
    oak_dev, vid_q = open_camera()
    display = Display(vid_q)

    print("[Init] Connecting to robot…")
    robot = Robot(ROBOT_IP)

    # wait for first frame
    print("[Init] Waiting for first camera frame…")
    t_end = time.time() + 6.0
    while time.time() < t_end:
        display.pump()
        if display.get_frame() is not None:
            break
        time.sleep(0.05)

    pose = robot.get_pose()
    print(f"\nLive TCP: X={pose[0]*1000:.1f}  Y={pose[1]*1000:.1f}  Z={pose[2]*1000:.1f} mm")
    print("\n[Ready]")

    # ── main loop ─────────────────────────────────────────────────────────────
    try:
        while True:
            try:
                input("\nPress ENTER to run pick sequence  (Ctrl-C to quit)…\n")
            except EOFError:
                break

            label, conf = run_once(
                robot, display,
                scan_pos, pick_pos, place_pos, dump_pos,
                clip_model, clip_pre, clf, clip_device,
            )

            # brief idle label before next prompt
            display.set_stage("idle")

    except KeyboardInterrupt:
        print("\n[Main] Stopped.")

    finally:
        robot.close()
        try:
            oak_dev.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("[Main] Done.")
