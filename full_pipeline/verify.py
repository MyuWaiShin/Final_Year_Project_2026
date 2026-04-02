"""
verify.py
---------
Stage 4 of the full pipeline.

Behaviour
---------
Picks up after grasp.py — gripper is closed at pick Z holding (maybe) the object.

This stage:
  1. Rises straight up to clearance_z (hover_z + 0.60 m, derived by navigate.py)
  2. Grabs a frame from the OAK-D camera (classifier works at this height pointing down)
  3. Runs YOLO26n classifier on the frame (empty vs holding)
       p_holding >= THRESHOLD  →  "holding"
       p_holding <  THRESHOLD  →  "empty"
  4. Returns result dict to main.py

No wrist tilt needed — classifier is reliable from clearance height.
Already at clearance_z if empty → recover can run the search circle immediately.

Return values
-------------
  {"result": "holding", "yolo_conf": float}
  {"result": "empty",   "yolo_conf": float}

Motion notes
------------
All robot motion is sent as raw URScript over port 30002.
Robot state is read from the same port's secondary-client stream.
No RTDE used (see pipeline_dev/RTDE_debug_log.md).
"""

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
SCRIPT_DIR  = Path(__file__).resolve().parent
YOLO_MODEL  = SCRIPT_DIR / "models" / "classification" / "yolo26n_cls_V1" / "weights" / "best.pt"

# ── Robot ─────────────────────────────────────────────────────────────────────
ROBOT_IP      = "192.168.8.102"
ROBOT_PORT    = 30002

# ── Motion ────────────────────────────────────────────────────────────────────
MOVE_SPEED    = 0.04    # m/s
MOVE_ACCEL    = 0.01    # m/s²
XYZ_TOL_M     = 0.003   # arrival tolerance

# ── Classifier settings ───────────────────────────────────────────────────────
THRESHOLD = 0.90  # p_holding >= this → "holding"


# ── Robot state reader ────────────────────────────────────────────────────────
class RobotStateReader(threading.Thread):
    """
    Reads UR secondary-client stream (port 30002) in a background thread.
    Parses:
      Type 1  Joint Data      → joint positions q[0..5]
      Type 4  Cartesian Info  → TCP pose [x,y,z,rx,ry,rz]
    """
    def __init__(self, ip: str, port: int = ROBOT_PORT):
        super().__init__(daemon=True)
        self.ip   = ip
        self.port = port
        self._lock      = threading.Lock()
        self._stop_evt  = threading.Event()
        self._ready_evt = threading.Event()
        self._tcp_pose  = [0.0] * 6
        self._joints    = [0.0] * 6
        self._protective_stop = False

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
                        self._parse_subpackets(body[1:])
            except Exception:
                time.sleep(0.5)

    @staticmethod
    def _recvall(s, n: int):
        buf = b""
        while len(buf) < n:
            chunk = s.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def _parse_subpackets(self, data: bytes):
        off = 0
        got = False
        while off < len(data):
            if off + 5 > len(data):
                break
            ps = struct.unpack("!I", data[off:off+4])[0]
            if ps < 5 or off + ps > len(data):
                break
            pt = data[off + 4]
            # Type 0: Robot Mode Data — isProtectiveStopped at off+17
            if pt == 0 and ps >= 18:
                with self._lock:
                    self._protective_stop = bool(data[off + 17])
            elif pt == 1 and ps >= 251:
                joints = []
                for j in range(6):
                    base = off + 5 + j * 41
                    joints.append(struct.unpack("!d", data[base:base+8])[0])
                with self._lock:
                    self._joints = joints
                got = True
            elif pt == 4 and ps >= 53:
                pose = list(struct.unpack("!6d", data[off+5:off+53]))
                with self._lock:
                    self._tcp_pose = pose
                got = True
            off += ps
        if got:
            self._ready_evt.set()

    def wait_ready(self, timeout: float = 5.0) -> bool:
        return self._ready_evt.wait(timeout=timeout)

    def get_tcp_pose(self) -> list:
        with self._lock:
            return list(self._tcp_pose)

    def get_joint_positions(self) -> list:
        with self._lock:
            return list(self._joints)

    def is_protective_stop(self) -> bool:
        """True when the robot has entered a protective stop."""
        with self._lock:
            return self._protective_stop

    def stop(self):
        self._stop_evt.set()


# ── URScript sender ───────────────────────────────────────────────────────────
class URScriptSender:
    def __init__(self, ip: str, port: int = ROBOT_PORT):
        self.ip    = ip
        self.port  = port
        self._lock = threading.Lock()
        self._sock = self._connect()
        threading.Thread(target=self._drain, daemon=True).start()

    def _connect(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5.0)
        s.connect((self.ip, self.port))
        return s

    def _drain(self):
        while True:
            try:
                self._sock.recv(4096)
            except Exception:
                time.sleep(0.01)

    def send(self, script: str):
        payload = (script.strip() + "\n").encode()
        with self._lock:
            try:
                self._sock.sendall(payload)
            except Exception:
                try:
                    self._sock = self._connect()
                    self._sock.sendall(payload)
                except Exception as e:
                    print(f"  [URScript] Send failed: {e}")

    def close(self):
        try:
            self._sock.close()
        except Exception:
            pass


# ── Motion helpers ────────────────────────────────────────────────────────────
def movel(sender, state, x, y, z, rx, ry, rz,
          vel=MOVE_SPEED, acc=MOVE_ACCEL, tol=XYZ_TOL_M, timeout=30.0):
    cur = state.get_tcp_pose()
    if ((cur[0]-x)**2 + (cur[1]-y)**2 + (cur[2]-z)**2) ** 0.5 < tol:
        return
    sender.send(f"movel(p[{x:.6f},{y:.6f},{z:.6f},{rx:.6f},{ry:.6f},{rz:.6f}],a={acc:.4f},v={vel:.4f})")
    deadline = time.time() + timeout
    while time.time() < deadline:
        cur = state.get_tcp_pose()
        if ((cur[0]-x)**2 + (cur[1]-y)**2 + (cur[2]-z)**2) ** 0.5 < tol:
            return
        time.sleep(0.01)
    if state.is_protective_stop():
        raise RuntimeError(
            "[PROTECTIVE STOP] Robot stopped during movel — "
            "clear the stop on the pendant before continuing."
        )
    print("  [movel] Warning: timeout before arrival tolerance reached.")



# ── Camera ────────────────────────────────────────────────────────────────────
def open_camera():
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
    queue  = device.getOutputQueue(name="video", maxSize=1, blocking=False)
    return device, queue


def grab_frame(queue, warmup_frames: int = 5) -> np.ndarray:
    """Discard a few frames to let the camera settle, then return a fresh one."""
    frame = None
    for _ in range(warmup_frames + 1):
        pkt = queue.get()
        frame = pkt.getCvFrame()
    return frame


# ── Classifiers ───────────────────────────────────────────────────────────────
def run_yolo(model: YOLO, frame: np.ndarray) -> tuple:
    """Returns (p_holding, p_empty) from YOLO26n classifier."""
    results = model(frame, imgsz=224, verbose=False)
    r = results[0]
    probs     = {r.names[i]: float(r.probs.data[i]) for i in range(len(r.names))}
    p_holding = probs.get("holding", 0.0)
    p_empty   = probs.get("empty",   0.0)
    return p_holding, p_empty


# ── Model loader (callable from main.py at startup) ──────────────────────────
def load_models() -> dict:
    """
    Load YOLO26n classifier. Called once in a background thread at pipeline
    startup so verify() has zero loading latency when it runs.
    """
    if not YOLO_MODEL.exists():
        raise FileNotFoundError(f"YOLO weights not found: {YOLO_MODEL}")
    yolo = YOLO(str(YOLO_MODEL), task="classify")
    return {"yolo": yolo}


# ── Main ──────────────────────────────────────────────────────────────────────
def main(clearance_z: float, models: dict = None) -> dict:
    """
    Verify stage: rise to clearance_z, classify with YOLO26n.

    Robot must already be at pick Z with gripper closed (grasp.py left it there).
    clearance_z — absolute Z in base frame (hover_z + 0.60 m from navigate).

    Returns {'result': 'holding'|'empty', 'yolo_conf': float}
    """
    signal.signal(signal.SIGINT, lambda *_: os._exit(1))

    print("\n" + "=" * 60)
    print("  STAGE 4 — VERIFY (YOLO26n)")
    print("=" * 60)
    print(f"  Threshold   : {THRESHOLD}")
    print(f"  clearance_z : {clearance_z:.4f} m")
    print("=" * 60 + "\n")

    # ── Connect ───────────────────────────────────────────────────────────────
    print("Connecting to robot …")
    state  = RobotStateReader(ROBOT_IP)
    sender = URScriptSender(ROBOT_IP)
    state.start()
    if not state.wait_ready(timeout=5.0):
        raise RuntimeError(f"Robot state reader timed out — is {ROBOT_IP} reachable?")
    print("Robot connected!\n")

    tcp = state.get_tcp_pose()
    px, py, pz, prx, pry, prz = tcp
    print(f"  Current TCP : X={px:.4f}  Y={py:.4f}  Z={pz:.4f}")
    print(f"  Rising to   : Z={clearance_z:.4f}\n")

    # ── Load models (or use pre-loaded ones from main.py) ─────────────────────
    if models is not None:
        yolo_model = models["yolo"]
        print("  Using pre-loaded models.\n")
    else:
        _models = {}
        _err    = [None]
        _done   = threading.Event()

        def _load():
            try:
                _models.update(load_models())
            except Exception as e:
                _err[0] = e
            finally:
                _done.set()

        threading.Thread(target=_load, daemon=True).start()
        print("  Loading models in background …\n")

    # ── Rise to clearance_z ───────────────────────────────────────────────────
    print("  [1/2] Rising to clearance_z …")
    movel(sender, state, px, py, clearance_z, prx, pry, prz)
    print(f"  At clearance_z={clearance_z:.4f}")

    # ── Wait for models ───────────────────────────────────────────────────────
    if models is None:
        if not _done.is_set():
            print("  Waiting for model …")
        _done.wait(timeout=60.0)
        if _err[0] is not None:
            raise _err[0]
        yolo_model = _models["yolo"]
    print("  Model ready.")

    # ── Capture frame ─────────────────────────────────────────────────────────
    print("\n  [2/2] Opening camera …")
    cam_device, queue = open_camera()
    frame = grab_frame(queue, warmup_frames=8)
    print("  Frame captured.")

    # ── YOLO26n classify ──────────────────────────────────────────────────────
    p_holding, _ = run_yolo(yolo_model, frame)
    result = "holding" if p_holding >= THRESHOLD else "empty"
    print(f"\n  holding={p_holding:.3f}  threshold={THRESHOLD}  →  {result.upper()}")

    # ── Display ───────────────────────────────────────────────────────────────
    display = cv2.resize(frame, (960, 540))
    col = (0, 220, 0) if result == "holding" else (0, 0, 220)
    cv2.putText(display, f"YOLO26n: {p_holding*100:.1f}%  →  {result.upper()}",
                (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
    cv2.imshow("verify — Stage 4", display)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print(f"  VERIFY COMPLETE  →  {result.upper()}")
    print("=" * 60 + "\n")

    cam_device.close()
    state.stop()
    sender.close()

    return {"result": result, "yolo_conf": round(p_holding, 4)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--clearance-z", type=float, required=True,
                        help="Absolute clearance Z in base frame (metres)")
    args = parser.parse_args()
    result = main(clearance_z=args.clearance_z)
    print(f"[verify] Returned: {result}")
