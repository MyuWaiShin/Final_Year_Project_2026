"""
release.py
----------
Stage 6 of the full pipeline.

Behaviour
---------
1. Descend from clearance_z to DROP_HEIGHT_M below clearance_z
2. Open gripper via dashboard URP
3. Rise back to clearance_z
4. Return to scan pose (joint move)

DROP_HEIGHT_M should be set so the object is just above the drop surface
when the gripper opens.  Adjust to match your drop zone geometry.

Scan pose is loaded from  data/scan_pose.json  (same file as explore.py).

Returns True when complete.

Motion notes
------------
Same port-30002 / no-RTDE pattern as all other stages.
"""

import json
import os
import signal
import socket
import struct
import threading
import time
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR     = Path(__file__).resolve().parent
DATA_DIR       = SCRIPT_DIR / "data"
SCAN_POSE_FILE = DATA_DIR / "scan_pose.json"

# ── Robot ─────────────────────────────────────────────────────────────────────
ROBOT_IP       = "192.168.8.102"
ROBOT_PORT     = 30002
DASHBOARD_PORT = 29999

# ── Gripper ───────────────────────────────────────────────────────────────────
GRIP_OPEN_URP = "/programs/myu/open_gripper.urp"
GRIP_OPEN_MM  = 85.0

# ── Drop geometry ─────────────────────────────────────────────────────────────
DROP_HEIGHT_M = 0.450   # metres to descend below clearance_z
# = CLEARANCE_OFFSET (0.40) + DESCEND_OFFSET (0.07)
# → drop_z = clearance_z - 0.470 = hover_z - 0.070 = same depth as pick_z

# ── Motion ────────────────────────────────────────────────────────────────────
MOVE_SPEED     = 0.08   # m/s — horizontal + rise
MOVE_ACCEL     = 0.02   # m/s²
DESCEND_SPEED  = 0.02   # m/s — slow descent onto surface
DESCEND_ACCEL  = 0.008  # m/s²
JOINT_SPEED    = 0.5    # rad/s — return to scan pose
JOINT_ACCEL    = 0.3    # rad/s²
XYZ_TOL_M      = 0.005
JOINT_TOL_RAD  = 0.01


# ── Robot state reader ────────────────────────────────────────────────────────
class RobotStateReader(threading.Thread):
    def __init__(self, ip, port=ROBOT_PORT):
        super().__init__(daemon=True)
        self.ip, self.port = ip, port
        self._lock      = threading.Lock()
        self._stop_evt  = threading.Event()
        self._ready_evt = threading.Event()
        self._tcp_pose  = [0.0] * 6
        self._joints    = [0.0] * 6
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
            if pt == 1 and ps >= 251:
                joints = []
                for j in range(6):
                    base = off + 5 + j * 41
                    joints.append(struct.unpack("!d", data[base:base+8])[0])
                with self._lock:
                    self._joints = joints
                got = True
            elif pt == 2 and ps >= 15:
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

    def get_joint_positions(self):
        with self._lock:
            return list(self._joints)

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
def _movel(sender, state, x, y, z, rx, ry, rz,
           vel=MOVE_SPEED, acc=MOVE_ACCEL, timeout=30.0):
    cur = state.get_tcp_pose()
    if ((cur[0]-x)**2 + (cur[1]-y)**2 + (cur[2]-z)**2) ** 0.5 < XYZ_TOL_M:
        return
    sender.send(
        f"movel(p[{x:.6f},{y:.6f},{z:.6f},{rx:.6f},{ry:.6f},{rz:.6f}],"
        f"a={acc:.4f},v={vel:.4f})"
    )
    deadline = time.time() + timeout
    while time.time() < deadline:
        cur = state.get_tcp_pose()
        if ((cur[0]-x)**2 + (cur[1]-y)**2 + (cur[2]-z)**2) ** 0.5 < XYZ_TOL_M:
            return
        time.sleep(0.01)
    print("  [Release] movel timeout — continuing.")


def _movej(sender, state, joints, vel=JOINT_SPEED, acc=JOINT_ACCEL, timeout=30.0):
    cur = state.get_joint_positions()
    if max(abs(c - t) for c, t in zip(cur, joints)) < JOINT_TOL_RAD:
        return
    q_str = ",".join(f"{j:.6f}" for j in joints)
    sender.send(f"movej([{q_str}],a={acc:.4f},v={vel:.4f})")
    deadline = time.time() + timeout
    while time.time() < deadline:
        cur = state.get_joint_positions()
        if max(abs(c - t) for c, t in zip(cur, joints)) < JOINT_TOL_RAD:
            return
        time.sleep(0.01)
    print("  [Release] movej timeout — continuing.")


# ── Gripper ───────────────────────────────────────────────────────────────────
def _dashboard_cmd(cmd, retries=3):
    for attempt in range(1, retries + 1):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5.0)
            s.connect((ROBOT_IP, DASHBOARD_PORT))
            s.recv(1024)
            s.sendall((cmd + "\n").encode())
            resp = s.recv(1024).decode().strip()
            s.close()
            return resp
        except Exception:
            if attempt < retries:
                time.sleep(0.5)
    return ""


def _wait_urp_done(timeout=6.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if "false" in _dashboard_cmd("running", retries=1).lower():
            return
        time.sleep(0.05)


def _open_gripper(state):
    if state.get_width_mm() >= GRIP_OPEN_MM - 5.0:
        print("  [Release] Gripper already open.")
        return
    print("  [Release] Opening gripper …")
    
    # Must stop any running program before loading
    resp = _dashboard_cmd("running", retries=2)
    if "true" in resp.lower():
        _dashboard_cmd("stop")
        _wait_urp_done(timeout=3.0)

    _dashboard_cmd(f"load {GRIP_OPEN_URP}")
    time.sleep(0.06)
    _dashboard_cmd("play")
    _wait_urp_done()
    deadline = time.time() + 4.0
    while time.time() < deadline:
        if state.get_width_mm() >= GRIP_OPEN_MM - 5.0:
            break
        time.sleep(0.1)
    print(f"  [Release] Gripper open: {state.get_width_mm():.1f} mm")


# ── Main ──────────────────────────────────────────────────────────────────────
def main(clearance_z: float) -> bool:
    """
    Release stage: descend, open gripper, rise, return to scan pose.

    clearance_z — absolute Z in base frame (from navigate / transit).
    Returns True on completion.
    """
    signal.signal(signal.SIGINT, lambda *_: os._exit(1))

    print("\n" + "=" * 60)
    print("  STAGE 6 — RELEASE")
    print("=" * 60)
    print(f"  clearance_z  : {clearance_z:.4f} m")
    print(f"  drop_height  : {DROP_HEIGHT_M*1000:.0f} mm below clearance_z")
    drop_z = clearance_z - DROP_HEIGHT_M
    print(f"  drop_z       : {drop_z:.4f} m")
    print("=" * 60 + "\n")

    # ── Load scan pose ────────────────────────────────────────────────────────
    if not SCAN_POSE_FILE.exists():
        raise FileNotFoundError(f"Scan pose not found: {SCAN_POSE_FILE}")
    with open(SCAN_POSE_FILE) as f:
        scan_joints = json.load(f)["joint_angles"]

    # ── Connect ───────────────────────────────────────────────────────────────
    print("Connecting to robot …")
    state  = RobotStateReader(ROBOT_IP)
    sender = URScriptSender(ROBOT_IP)
    state.start()
    if not state.wait_ready(timeout=5.0):
        raise RuntimeError(f"Robot state reader timed out — is {ROBOT_IP} reachable?")
    print("Robot connected!\n")

    tcp = state.get_tcp_pose()
    x, y, z, rx, ry, rz = tcp
    print(f"  Current TCP : X={x:.4f}  Y={y:.4f}  Z={z:.4f}\n")

    # ── 1. Descend to drop height ─────────────────────────────────────────────
    print("  [1/4] Descending to drop height …")
    _movel(sender, state, x, y, drop_z, rx, ry, rz,
           vel=DESCEND_SPEED, acc=DESCEND_ACCEL)
    print(f"  At drop_z={state.get_tcp_pose()[2]:.4f}")

    # ── 2. Open gripper ───────────────────────────────────────────────────────
    print("\n  [2/4] Opening gripper …")
    _open_gripper(state)

    # ── 3. Rise back to clearance_z ───────────────────────────────────────────
    print(f"\n  [3/4] Rising back to clearance_z={clearance_z:.4f} …")
    tcp = state.get_tcp_pose()
    _movel(sender, state, tcp[0], tcp[1], clearance_z, rx, ry, rz)
    print(f"  At clearance_z={state.get_tcp_pose()[2]:.4f}")

    # ── 4. Return to scan pose ────────────────────────────────────────────────
    print(f"\n  [4/4] Returning to scan pose …")
    _movej(sender, state, scan_joints)
    print(f"  Scan pose reached: {[round(j, 3) for j in state.get_joint_positions()]}")

    print("\n" + "=" * 60)
    print("  RELEASE COMPLETE — pipeline done.")
    print("=" * 60 + "\n")

    state.stop()
    sender.close()
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--clearance-z", type=float, required=True)
    args = parser.parse_args()
    main(clearance_z=args.clearance_z)
