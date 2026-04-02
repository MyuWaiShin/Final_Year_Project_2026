"""
transit.py
----------
Stage 5 of the full pipeline.

Behaviour
---------
Moves from current position (at clearance_z) to above the drop zone,
staying at clearance_z.  Two independent slip detection layers:

  Layer 1 — SLIP_DETECT = True/False flag:
    Single URScript with two threads — one executes movel(), one loops
    the gripper close command every 0.2 s.  If the object slips, the
    gripper closes further; width drops below WIDTH_CLOSED_MM → stop → slip.

  Layer 2 — YOLO classify (always on):
    Background thread grabs OAK-D frames and runs YOLO26n classifier.
    N consecutive "empty" predictions → stop → empty.

Drop zone XY loaded from  data/drop_zone.json.  Z stays at clearance_z.

Returns
-------
  {"result": "arrived"}             — reached drop zone cleanly
  {"result": "slip",  "layer": 1}   — gripper width drop (Layer 1)
  {"result": "empty", "layer": 2}   — YOLO classify says empty (Layer 2)

Motion notes
------------
Same port-30002 / no-RTDE pattern as all other stages.

GRIP_CLOSE_CMD — URScript command to close the RG2 inside the thread.
  Adjust this to match your robot's OnRobot installation.
  Common values:
    rg2_grip(0, 40)          ← OnRobot RG2 (some firmware)
    rg_grip(0, 40, False, 0) ← OnRobot SDK via URcap
"""

import json
import os
import signal
import socket
import struct
import threading
import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR     = Path(__file__).resolve().parent
DATA_DIR       = SCRIPT_DIR / "data"
DROP_ZONE_FILE = DATA_DIR / "drop_zone.json"
CLS_MODEL_PATH = (SCRIPT_DIR / "models" / "classification"
                  / "yolo26n_cls_V1" / "weights" / "best.pt")

# ── Robot ─────────────────────────────────────────────────────────────────────
ROBOT_IP       = "192.168.8.102"
ROBOT_PORT     = 30002
DASHBOARD_PORT = 29999

# ── Slip detection ────────────────────────────────────────────────────────────
SLIP_DETECT           = False  # set False to disable Layer 1 (gripper thread)
WIDTH_CLOSED_MM       = 11.0   # width below this → object gone
GRIP_CLOSE_INTERVAL_S = 0.2    # how often the gripper re-closes in the thread

# ─── IMPORTANT: set this to the URScript command that closes your RG2 ─────────
GRIP_CLOSE_CMD = "rg2_grip(0, 40)"

# ── YOLO classify threshold (Layer 2) ────────────────────────────────────────
CLS_THRESHOLD           = 0.90   # p_holding threshold from verify.py
CLS_EMPTY_CONSEC_NEEDED = 3      # consecutive empty frames before triggering

# ── Camera ────────────────────────────────────────────────────────────────────
MANUAL_FOCUS = 46

# ── Motion ────────────────────────────────────────────────────────────────────
MOVE_SPEED = 0.06
MOVE_ACCEL = 0.02
XYZ_TOL_M  = 0.005


# ── Drop zone ─────────────────────────────────────────────────────────────────
def load_drop_zone() -> dict:
    """Load drop zone position from data/drop_zone.json."""
    if not DROP_ZONE_FILE.exists():
        raise FileNotFoundError(
            f"Drop zone file not found: {DROP_ZONE_FILE}\n"
            "Run:  python temp/capture_drop_zone.py  to create it.\n"
            "Expected keys: x, y"
        )
    with open(DROP_ZONE_FILE) as f:
        return json.load(f)


# ── Robot state reader ────────────────────────────────────────────────────────
class RobotStateReader(threading.Thread):
    """
    Reads UR secondary-client stream (port 30002).
    Type 2 → AI2 voltage → gripper width.
    Type 4 → TCP pose.
    """
    def __init__(self, ip, port=ROBOT_PORT):
        super().__init__(daemon=True)
        self.ip, self.port = ip, port
        self._lock      = threading.Lock()
        self._stop_evt  = threading.Event()
        self._ready_evt = threading.Event()
        self._tcp_pose  = [0.0] * 6
        self._voltage   = 0.0

    def run(self):
        while not self._stop_evt.is_set():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(3.0)
                    s.connect((self.ip, self.port))
                    while not self._stop_evt.is_set():
                        hdr = self._recvall(s, 4)
                        if hdr is None:
                            break
                        plen = struct.unpack("!I", hdr)[0]
                        body = self._recvall(s, plen - 4)
                        if body is None:
                            break
                        self._parse(body[1:])
            except Exception:
                time.sleep(0.5)

    @staticmethod
    def _recvall(s, n):
        buf = b""
        while len(buf) < n:
            chunk = s.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def _parse(self, data):
        off, got = 0, False
        while off < len(data):
            if off + 5 > len(data):
                break
            ps = struct.unpack("!I", data[off:off+4])[0]
            if ps < 5 or off + ps > len(data):
                break
            pt = data[off + 4]
            if pt == 2 and ps >= 15:
                with self._lock:
                    self._voltage = max(struct.unpack("!d", data[off+7:off+15])[0], 0.0)
            elif pt == 4 and ps >= 53:
                with self._lock:
                    self._tcp_pose = list(struct.unpack("!6d", data[off+5:off+53]))
                got = True
            off += ps
        if got:
            self._ready_evt.set()

    def wait_ready(self, timeout=5.0):
        return self._ready_evt.wait(timeout=timeout)

    def get_tcp_pose(self):
        with self._lock:
            return list(self._tcp_pose)

    def get_width_mm(self):
        with self._lock:
            v = self._voltage
        raw_mm = (v / 3.7) * 110.0
        slope  = (91.0 - 10.5) / (65.8 - 8.5)
        return max(0.0, round((raw_mm * slope) + (10.5 - 8.5 * slope), 1))

    def stop(self):
        self._stop_evt.set()


class URScriptSender:
    def __init__(self, ip, port=ROBOT_PORT):
        self.ip    = ip
        self._lock = threading.Lock()
        self._sock = self._connect(port)
        threading.Thread(target=self._drain, daemon=True).start()

    def _connect(self, port=ROBOT_PORT):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5.0)
        s.connect((self.ip, port))
        return s

    def _drain(self):
        while True:
            try:
                self._sock.recv(4096)
            except Exception:
                time.sleep(0.01)

    def send(self, script):
        payload = (script.strip() + "\n").encode()
        with self._lock:
            try:
                self._sock.sendall(payload)
            except Exception:
                pass

    def close(self):
        try:
            self._sock.close()
        except Exception:
            pass


# ── Motion helpers ────────────────────────────────────────────────────────────
def _stopl(sender, acc=1.0):
    sender.send(f"stopl({acc:.2f})")


def _wait_arrived(state, x, y, z, stop_flag, timeout=60.0):
    """Poll TCP pose until within XYZ_TOL_M of target or stop_flag is set."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if stop_flag["set"]:
            return False
        cur = state.get_tcp_pose()
        if ((cur[0]-x)**2 + (cur[1]-y)**2 + (cur[2]-z)**2) ** 0.5 < XYZ_TOL_M:
            return True
        time.sleep(0.02)
    return False


# ── Camera helpers ────────────────────────────────────────────────────────────
def _open_camera():
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setVideoSize(1280, 720)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.initialControl.setManualFocus(MANUAL_FOCUS)
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("video")
    cam.video.link(xout.input)
    device = dai.Device(pipeline)
    queue  = device.getOutputQueue(name="video", maxSize=1, blocking=False)
    return device, queue


# ── Main ──────────────────────────────────────────────────────────────────────
def main(clearance_z: float, models: dict = None) -> dict:
    """
    Transit stage: move to drop zone at clearance_z with slip monitoring.

    clearance_z — absolute Z in base frame (from navigate).
    models      — optional pre-loaded {'cls': YOLO} from main.py startup.

    Returns {'result': 'arrived'|'slip'|'empty', ...}
    """
    signal.signal(signal.SIGINT, lambda *_: os._exit(1))

    print("\n" + "=" * 60)
    print("  STAGE 5 — TRANSIT")
    print("=" * 60)
    print(f"  clearance_z : {clearance_z:.4f} m")
    print(f"  SLIP_DETECT : {SLIP_DETECT}")
    print("=" * 60 + "\n")

    # ── Load drop zone ────────────────────────────────────────────────────────
    dz = load_drop_zone()
    target_x  = float(dz["x"])
    target_y  = float(dz["y"])
    target_z  = clearance_z
    print(f"  Drop zone: X={target_x:.4f}  Y={target_y:.4f}  Z={target_z:.4f}\n")

    # ── Load classify model (Layer 2) ─────────────────────────────────────────
    if models is not None and "cls" in models:
        cls_model = models["cls"]
        print("  Using pre-loaded classify model.")
    else:
        print("  Loading classify model …")
        if not CLS_MODEL_PATH.exists():
            raise FileNotFoundError(f"Classify model not found: {CLS_MODEL_PATH}")
        cls_model = YOLO(str(CLS_MODEL_PATH), task="classify")
        print("  Classify model loaded.")

    # ── Robot + camera ────────────────────────────────────────────────────────
    print("Connecting to robot …")
    state  = RobotStateReader(ROBOT_IP)
    sender = URScriptSender(ROBOT_IP)
    state.start()
    if not state.wait_ready(timeout=5.0):
        raise RuntimeError(f"Robot state reader timed out — is {ROBOT_IP} reachable?")
    print("Robot connected!")

    print("Opening camera …")
    cam_device, queue = _open_camera()
    print("Camera ready.\n")

    cur = state.get_tcp_pose()
    width_at_start = state.get_width_mm()
    cur_rx, cur_ry, cur_rz = cur[3], cur[4], cur[5]   # keep current wrist orientation
    print(f"  Current TCP : {[round(v, 4) for v in cur[:3]]}")
    print(f"  Gripper width at start : {width_at_start:.1f} mm\n")

    # ── Shared stop flag ──────────────────────────────────────────────────────
    stop_flag  = {"set": False, "reason": None}

    # ── Layer 1: build and send URScript ──────────────────────────────────────
    if SLIP_DETECT:
        script = f"""
def transit():
  thread KeepClosed:
    while True:
      {GRIP_CLOSE_CMD}
      sleep({GRIP_CLOSE_INTERVAL_S:.2f})
    end
  end
  run KeepClosed()
  movel(p[{target_x:.6f},{target_y:.6f},{target_z:.6f},{cur_rx:.6f},{cur_ry:.6f},{cur_rz:.6f}],a={MOVE_ACCEL:.4f},v={MOVE_SPEED:.4f})
end
transit()
"""
        print("  Sending transit URScript (movel + gripper thread) …")
    else:
        script = (
            f"movel(p[{target_x:.6f},{target_y:.6f},{target_z:.6f},"
            f"{cur_rx:.6f},{cur_ry:.6f},{cur_rz:.6f}],"
            f"a={MOVE_ACCEL:.4f},v={MOVE_SPEED:.4f})"
        )
        print("  Sending plain movel (SLIP_DETECT=False) …")

    sender.send(script)

    # ── Layer 1 monitor: gripper width (background) ───────────────────────────
    def _monitor_width():
        time.sleep(0.5)   # allow gripper to settle after first re-close
        while not stop_flag["set"]:
            w = state.get_width_mm()
            if w <= WIDTH_CLOSED_MM:
                stop_flag["set"]    = True
                stop_flag["reason"] = "slip"
                print(f"\n  [Transit] Layer 1 SLIP — width={w:.1f} mm (<={WIDTH_CLOSED_MM})")
                _stopl(sender)
                return
            time.sleep(0.05)

    if SLIP_DETECT:
        threading.Thread(target=_monitor_width, daemon=True).start()

    # ── Layer 2: YOLO classify monitor (background) ───────────────────────────
    consec_empty = [0]

    def _monitor_classify():
        while not stop_flag["set"]:
            pkt = queue.tryGet()
            if pkt is None:
                time.sleep(0.05)
                continue
            frame   = pkt.getCvFrame()
            results = cls_model(frame, imgsz=224, verbose=False)
            r       = results[0]
            probs   = {r.names[i]: float(r.probs.data[i]) for i in range(len(r.names))}
            p_h     = probs.get("holding", 0.0)

            if p_h < CLS_THRESHOLD:
                consec_empty[0] += 1
            else:
                consec_empty[0] = 0

            if consec_empty[0] >= CLS_EMPTY_CONSEC_NEEDED:
                stop_flag["set"]    = True
                stop_flag["reason"] = "empty"
                print(f"\n  [Transit] Layer 2 EMPTY — p_holding={p_h:.3f} "
                      f"({CLS_EMPTY_CONSEC_NEEDED} consecutive empties)")
                _stopl(sender)
                return

    threading.Thread(target=_monitor_classify, daemon=True).start()

    # ── Wait for arrival or trip ──────────────────────────────────────────────
    arrived = _wait_arrived(state, target_x, target_y, target_z, stop_flag)

    # Ensure width monitor loop exits
    stop_flag["set"] = True

    final = state.get_tcp_pose()
    final_width = state.get_width_mm()

    cam_device.close()
    state.stop()
    sender.close()

    if arrived and stop_flag["reason"] is None:
        print(f"\n  [Transit] Arrived at drop zone.  width={final_width:.1f} mm")
        print("=" * 60)
        print("  TRANSIT COMPLETE")
        print("=" * 60 + "\n")
        return {"result": "arrived"}

    reason = stop_flag.get("reason", "unknown")
    print(f"\n  [Transit] Stopped mid-transit — reason: {reason}")
    print(f"  TCP at stop: {[round(v, 4) for v in final[:3]]}")
    print("=" * 60 + "\n")
    return {"result": reason, "layer": 1 if reason == "slip" else 2}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--clearance-z", type=float, required=True)
    args = parser.parse_args()
    result = main(clearance_z=args.clearance_z)
    print(f"[transit] Returned: {result}")
