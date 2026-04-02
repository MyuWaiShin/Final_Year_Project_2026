"""
grasp.py
--------
Stage 3 of the full pipeline.

Behaviour
---------
Picks up exactly where navigate.py leaves off:
  - TCP is hovering above the ArUco tag at hover height
  - Gripper is open

This stage:
  1. Confirms with the user before descending (hand on E-stop)
  2. Descends DESCEND_OFFSET below the hover Z at slow speed
  3. Closes the gripper once (via close_gripper.urp dashboard URP)
  4. CHECK 1 — Jaw width  (AI2 voltage → mm)
       width <  WIDTH_CLOSED_MM  →  fully snapped shut → MISSED → rise, return "missed"
       width >= WIDTH_CLOSED_MM  →  something between fingers → continue
  5. CHECK 2 — Force contact (DI8 bit 17 of masterboard data)
       HIGH  →  RG2 force limit reached → force contact confirmed
       LOW   →  soft object or sensor lag → logged but NOT fatal
  6. Returns a result dict to main.py

Return values
-------------
  {"result": "holding",  "width_mm": float, "force": bool, "tcp_pose": list}
  {"result": "missed"}

Run standalone
--------------
    python grasp.py

  When run standalone a fixed test hover pose (STANDALONE_HOVER_POSE) is used
  so you can test the grasp stage without running the full pipeline.

Motion notes
------------
All robot motion is sent as raw URScript over a persistent socket on
port 30002.  Robot state (TCP pose, gripper width, force) is read from
the same port's secondary-client packet stream in a background thread.
No RTDEControlInterface or RTDEReceiveInterface is used — both caused
unpredictable 10-second reconnect hangs on this setup (see
pipeline_dev/RTDE_debug_log.md).
"""

import os
import signal
import socket
import struct
import threading
import time
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).resolve().parent

# ── Robot ───────────────────────────────────────────────────────────────────
ROBOT_IP      = "192.168.8.102"
ROBOT_PORT    = 30002
DASHBOARD_PORT = 29999

# ── Gripper URPs ─────────────────────────────────────────────────────────────
GRIP_CLOSE_ONCE_URP = "/programs/myu/close_gripper.urp"
GRIP_OPEN_URP       = "/programs/myu/open_gripper.urp"
GRIP_OPEN_MM        = 85.0     # expected open width (used in skip check)
GRIP_CLOSE_MM       = 0.0      # expected closed width (used in skip check)

# ── Layer 1 thresholds ───────────────────────────────────────────────────────
WIDTH_CLOSED_MM = 11.0    # <= this  → gripper fully snapped shut → missed

# ── Motion parameters ────────────────────────────────────────────────────────
DESCEND_OFFSET          = 0.050   # metres — full descent below hover Z (first attempt)
RECOVERY_DESCEND_OFFSET = 0.010   # metres — short descent used in recovery mode
MOVE_SPEED     = 0.04     # m/s  — normal moves (lift back up on miss)
MOVE_ACCEL     = 0.01     # m/s²
DESCEND_SPEED  = 0.02     # m/s  — slow descend onto object
DESCEND_ACCEL  = 0.008    # m/s²
XYZ_TOL_M      = 0.003    # metres arrival tolerance

# ── Standalone test pose ─────────────────────────────────────────────────────
# Only used when running  python grasp.py  directly (not via main.py).
# The robot must already be positioned at a safe hover pose before running.
# No STANDALONE_HOVER_POSE needed — grasp reads current TCP from the robot.


# ── Robot state reader (port 30002 secondary client) ────────────────────────
class RobotStateReader(threading.Thread):
    """
    Reads the UR secondary-client stream (port 30002) in a background thread.
    Parses three sub-packet types from every Robot State message:

      Type 1  Joint Data        → actual joint positions q[0..5]
      Type 2  Tool Data         → AI2 voltage → gripper width
      Type 3  Masterboard Data  → 64-bit DI word → bit 17 = RG2 force limit
      Type 4  Cartesian Info    → TCP pose [x,y,z,rx,ry,rz] in base frame

    All values are updated at the robot's broadcast rate (~10 Hz on port 30002).
    Thread reconnects automatically on socket errors.
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
        self._voltage   = 0.0
        self._di_word   = 0      # 64-bit digital input word
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
        got_something = False
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

            # Type 1: Joint Data — q_actual at off+5, each joint 41 bytes
            elif pt == 1 and ps >= 251:
                joints = []
                for j in range(6):
                    base = off + 5 + j * 41
                    q = struct.unpack("!d", data[base:base+8])[0]
                    joints.append(q)
                with self._lock:
                    self._joints = joints
                got_something = True

            # Type 2: Tool Data — AI2 voltage at off+7 (8-byte double)
            elif pt == 2 and ps >= 15:
                ai = struct.unpack("!d", data[off+7:off+15])[0]
                with self._lock:
                    self._voltage = max(ai, 0.0)

            # Type 3: Masterboard Data — 64-bit DI word at off+5
            elif pt == 3 and ps >= 13:
                di = struct.unpack("!Q", data[off+5:off+13])[0]
                with self._lock:
                    self._di_word = di

            # Type 4: Cartesian Info — TCP pose at off+5 (6 doubles)
            elif pt == 4 and ps >= 53:
                pose = list(struct.unpack("!6d", data[off+5:off+53]))
                with self._lock:
                    self._tcp_pose = pose
                got_something = True

            off += ps

        if got_something:
            self._ready_evt.set()

    def wait_ready(self, timeout: float = 5.0) -> bool:
        return self._ready_evt.wait(timeout=timeout)

    def get_tcp_pose(self) -> list:
        with self._lock:
            return list(self._tcp_pose)

    def get_joint_positions(self) -> list:
        with self._lock:
            return list(self._joints)

    def get_width_mm(self) -> float:
        """
        RG2 width from AI2 voltage.
        Two-point linear calibration corrects RG2 sensor nonlinearity:
          raw 8.5 mm → actual 10.5 mm ; raw 65.8 mm → actual 91.0 mm
        """
        with self._lock:
            v = self._voltage
        raw_mm = (v / 3.7) * 110.0
        slope  = (91.0 - 10.5) / (65.8 - 8.5)
        offset = 10.5 - (8.5 * slope)
        return max(0.0, round((raw_mm * slope) + offset, 1))

    def is_protective_stop(self) -> bool:
        """True when the robot has entered a protective stop."""
        with self._lock:
            return self._protective_stop

    def is_force_detected(self) -> bool:
        """True when RG2 TDI1 (bit 17 of masterboard DI word) is HIGH."""
        with self._lock:
            return bool(self._di_word & (1 << 17))

    def stop(self):
        self._stop_evt.set()


# ── URScript sender (port 30002, persistent socket) ──────────────────────────
class URScriptSender:
    """
    Persistent TCP socket to port 30002 for sending URScript commands.
    A drain thread discards incoming secondary data to prevent the OS
    recv buffer from filling and blocking sendall().
    """
    def __init__(self, ip: str, port: int = ROBOT_PORT):
        self.ip   = ip
        self.port = port
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


# ── Motion helper ────────────────────────────────────────────────────────────
def movel(sender: URScriptSender, state: RobotStateReader,
          x, y, z, rx, ry, rz,
          vel: float = MOVE_SPEED, acc: float = MOVE_ACCEL,
          tol: float = XYZ_TOL_M, timeout: float = 30.0):
    """Send movel(p[...]) and poll TCP pose for arrival."""
    cur = state.get_tcp_pose()
    if ((cur[0]-x)**2 + (cur[1]-y)**2 + (cur[2]-z)**2) ** 0.5 < tol:
        return   # already there
    sender.send(
        f"movel(p[{x:.6f},{y:.6f},{z:.6f},{rx:.6f},{ry:.6f},{rz:.6f}],a={acc:.4f},v={vel:.4f})"
    )
    deadline = time.time() + timeout
    while time.time() < deadline:
        cur = state.get_tcp_pose()
        if ((cur[0]-x)**2 + (cur[1]-y)**2 + (cur[2]-z)**2) ** 0.5 < tol:
            return
        time.sleep(0.01)
    # Timeout — check if due to protective stop
    if state.is_protective_stop():
        raise RuntimeError(
            "[PROTECTIVE STOP] Robot stopped during movel — "
            "clear the stop on the pendant before continuing."
        )
    print("  [movel] Warning: timeout before arrival tolerance reached.")


# ── Dashboard helpers ────────────────────────────────────────────────────────
def _dashboard_cmd(robot_ip: str, cmd: str, retries: int = 3) -> str:
    for attempt in range(1, retries + 1):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5.0)
            s.connect((robot_ip, DASHBOARD_PORT))
            s.recv(1024)   # welcome banner
            s.sendall((cmd + "\n").encode())
            resp = s.recv(1024).decode().strip()
            s.close()
            return resp
        except Exception as e:
            print(f"  [Dashboard] {cmd}: {e}  (attempt {attempt}/{retries})")
            if attempt < retries:
                time.sleep(0.5)
    return ""


def _wait_urp_done(robot_ip: str, timeout: float = 6.0):
    """Poll dashboard 'running' until program stops."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = _dashboard_cmd(robot_ip, "running", retries=1)
        if "false" in resp.lower():
            return
        time.sleep(0.05)


def _play_urp(robot_ip: str, program: str, last_urp: list):
    """
    Load (if different) and play a gripper URP via dashboard.
    last_urp is a 1-element list used as mutable state across calls.
    """
    if last_urp[0] != program:
        # Stop whatever is running before loading a new program
        resp = _dashboard_cmd(robot_ip, "running", retries=2)
        if "true" in resp.lower():
            _dashboard_cmd(robot_ip, "stop")
            _wait_urp_done(robot_ip, timeout=3.0)
        _dashboard_cmd(robot_ip, f"load {program}")
        last_urp[0] = program
        time.sleep(0.06)
    else:
        # Same program — just stop the previous run if still active
        resp = _dashboard_cmd(robot_ip, "running", retries=2)
        if "true" in resp.lower():
            _dashboard_cmd(robot_ip, "stop")
            _wait_urp_done(robot_ip, timeout=3.0)
    _dashboard_cmd(robot_ip, "play")
    _wait_urp_done(robot_ip)


def _wait_for_gripper(state: RobotStateReader, target_mm: float,
                      timeout: float = 3.5):
    """
    Poll width until stable after triggering a gripper URP.
    Phase 0: skip if already at/near target (within 5 mm).
    Phase 1: wait for movement to START (width changes > 2 mm, max 1.5 s).
    Phase 2: wait for movement to STOP (4 consecutive stable readings).
    """
    start_w = state.get_width_mm()
    if abs(start_w - target_mm) < 5.0:
        return
    # Phase 1
    move_dl = time.time() + 1.5
    while time.time() < move_dl:
        time.sleep(0.06)
        if abs(state.get_width_mm() - start_w) > 2.0:
            break
    # Phase 2
    prev   = state.get_width_mm()
    stable = 0
    dl     = time.time() + timeout
    while time.time() < dl:
        time.sleep(0.06)
        cur = state.get_width_mm()
        if abs(cur - prev) < 0.5:
            stable += 1
            if stable >= 3:
                break
        else:
            stable = 0
        prev = cur


def open_gripper(robot_ip: str, state: RobotStateReader, last_urp: list):
    """Open RG2. Skips if already open."""
    if state.get_width_mm() >= GRIP_OPEN_MM - 5.0:
        print("  Gripper already open.")
        return
    print("  [Gripper] Opening ...")
    _play_urp(robot_ip, GRIP_OPEN_URP, last_urp)
    _wait_for_gripper(state, GRIP_OPEN_MM)
    print(f"  [Gripper] Open: {state.get_width_mm():.1f} mm")


def close_gripper_once(robot_ip: str, state: RobotStateReader, last_urp: list):
    """Single close via dashboard URP. Skips if already fully closed."""
    if state.get_width_mm() <= GRIP_CLOSE_MM + 3.0:
        print("  Gripper already closed.")
        return
    print("  [Gripper] Closing ...")
    _play_urp(robot_ip, GRIP_CLOSE_ONCE_URP, last_urp)
    _wait_for_gripper(state, GRIP_CLOSE_MM)
    print(f"  [Gripper] Settled: {state.get_width_mm():.1f} mm")


# ── Main ─────────────────────────────────────────────────────────────────────
def main(descend_m: float = None, close_only: bool = False,
         autonomous: bool = False) -> dict:
    """
    Grasp stage: descend (optional), close, check width + force.

    Parameters
    ----------
    descend_m : float, optional
        How far to descend below hover Z (metres). Defaults to DESCEND_OFFSET
        (150 mm) on first attempt, or RECOVERY_DESCEND_OFFSET (10 mm) when
        close_only=True.
    close_only : bool, optional
        When True (recovery mode) use a shorter 10 mm descent instead of the
        full 150 mm — navigate has already re-positioned the TCP above the tag.

    Returns
    -------
    dict with key "result":
        "holding"  — something is gripped;  also includes "width_mm", "force", "tcp_pose"
        "missed"   — gripper fully closed; nothing grabbed
    """
    signal.signal(signal.SIGINT, lambda *_: os._exit(1))
    if descend_m is not None:
        descent = descend_m
    elif close_only:
        descent = RECOVERY_DESCEND_OFFSET
    else:
        descent = DESCEND_OFFSET

    # ── Connect first so we can read the current TCP pose ───────────────────
    print("Connecting to robot ...")
    state  = RobotStateReader(ROBOT_IP)
    sender = URScriptSender(ROBOT_IP)
    state.start()
    if not state.wait_ready(timeout=5.0):
        raise RuntimeError(
            "Robot state reader did not receive data within 5 s — "
            "is the robot reachable at " + ROBOT_IP + "?"
        )
    print("Robot connected!\n")

    # ── Read hover pose from robot (navigate already left it here) ──────────
    hover_pose = state.get_tcp_pose()
    hx, hy, hz, hrx, hry, hrz = hover_pose

    pick_z = hz - descent

    print("\n" + "=" * 58)
    print("  STAGE 3 — GRASP (Layer 1: width + force check)")
    print("=" * 58)
    print(f"  Hover pose  : X={hx:.4f}  Y={hy:.4f}  Z={hz:.4f}")
    mode_str = f"RECOVERY ({descent*1000:.0f} mm descent)" if close_only else "NORMAL"
    print(f"  Mode        : {mode_str}")
    print(f"  Descend to  : Z={pick_z:.4f}  ({descent*1000:.0f} mm below hover)")
    print(f"  Width miss   threshold: {WIDTH_CLOSED_MM} mm")
    print("=" * 58 + "\n")

    last_urp = [None]   # mutable state for _play_urp program tracking

    # ── Safety gate ─────────────────────────────────────────────────────────
    cur = state.get_tcp_pose()
    print(f"  Current TCP: X={cur[0]:.4f}  Y={cur[1]:.4f}  Z={cur[2]:.4f}")
    print(f"  Gripper width: {state.get_width_mm():.1f} mm")
    print()
    if not autonomous:
        input("  Press ENTER to descend and close gripper (hand on E-stop): ")

    # ── Descend ──────────────────────────────────────────────────────────────
    print(f"\n  Descending {descent*1000:.0f} mm to pick Z={pick_z:.4f} ...")
    movel(sender, state, hx, hy, pick_z, hrx, hry, hrz,
          vel=DESCEND_SPEED, acc=DESCEND_ACCEL)
    print("  At pick Z.")

    # ── Close gripper ────────────────────────────────────────────────────────
    close_gripper_once(ROBOT_IP, state, last_urp)

    # ── CHECK 1: Jaw width ───────────────────────────────────────────────────
    width = state.get_width_mm()
    print(f"\n[CHECK 1]  Jaw width: {width:.1f} mm  (threshold: {WIDTH_CLOSED_MM} mm)")

    if width < WIDTH_CLOSED_MM:
        print(f"  X  Gripper fully closed ({width:.1f} mm) — object MISSED.")
        open_gripper(ROBOT_IP, state, last_urp)
        print(f"  Rising back to hover Z={hz:.4f} ...")
        movel(sender, state, hx, hy, hz, hrx, hry, hrz)
        print("  Back at hover height.\n")
        state.stop()
        sender.close()
        return {"result": "missed"}

    print(f"  ~  Width {width:.1f} mm — something between fingers.")

    # ── CHECK 2: Force contact ───────────────────────────────────────────────
    force = state.is_force_detected()
    print(f"\n[CHECK 2]  Force contact (DI8 bit 17): {'YES' if force else 'NO'}")
    if force:
        print("  OK  RG2 force limit reached — solid contact confirmed.")
    else:
        print("  ~   No force signal (soft object, or sensor lag — not fatal).")

    # ── Result ───────────────────────────────────────────────────────────────
    tcp_pose = state.get_tcp_pose()
    print("\n" + "=" * 58)
    print(f"  GRASP STAGE COMPLETE")
    print(f"  Width: {width:.1f} mm  |  Force: {'YES' if force else 'NO'}")
    print(f"  TCP  : X={tcp_pose[0]:.4f}  Y={tcp_pose[1]:.4f}  Z={tcp_pose[2]:.4f}")
    print("  -> Proceed to Stage 4 (CLIP visual verification)")
    print("=" * 58 + "\n")

    state.stop()
    sender.close()
    return {
        "result":   "holding",
        "width_mm": width,
        "force":    force,
        "tcp_pose": tcp_pose,
    }


if __name__ == "__main__":
    result = main()
    print(f"[grasp] Returned: {result}")
