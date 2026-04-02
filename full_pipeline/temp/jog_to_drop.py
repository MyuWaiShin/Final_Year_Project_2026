"""
jog_to_drop.py
--------------
Moves the robot to the drop zone XY at its current Z, using rx/ry/rz from
the recorded drop zone (data/drop_zone.json).

Run this to confirm the drop zone position is correct before running the pipeline.

Usage:  python temp/jog_to_drop.py
        Press ENTER to confirm and move, Q to abort.
"""

import json
import socket
import struct
import threading
import time
from pathlib import Path

SCRIPT_DIR    = Path(__file__).resolve().parent.parent   # full_pipeline/
DROP_ZONE_FILE = SCRIPT_DIR / "data" / "drop_zone.json"

ROBOT_IP   = "192.168.8.102"
ROBOT_PORT = 30002

MOVE_SPEED = 0.04
MOVE_ACCEL = 0.01
XYZ_TOL_M  = 0.005


class _StateReader(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        self._ready = threading.Event()
        self._tcp   = [0.0] * 6

    def run(self):
        while not self._stop.is_set():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(3.0)
                    s.connect((ROBOT_IP, ROBOT_PORT))
                    while not self._stop.is_set():
                        hdr = self._recv(s, 4)
                        if hdr is None: break
                        plen = struct.unpack("!I", hdr)[0]
                        body = self._recv(s, plen - 4)
                        if body is None: break
                        self._parse(body[1:])
            except Exception:
                time.sleep(0.5)

    @staticmethod
    def _recv(s, n):
        buf = b""
        while len(buf) < n:
            c = s.recv(n - len(buf))
            if not c: return None
            buf += c
        return buf

    def _parse(self, data):
        off = 0
        while off < len(data):
            if off + 5 > len(data): break
            ps = struct.unpack("!I", data[off:off+4])[0]
            if ps < 5 or off + ps > len(data): break
            if data[off+4] == 4 and ps >= 53:
                with self._lock:
                    self._tcp = list(struct.unpack("!6d", data[off+5:off+53]))
                self._ready.set()
            off += ps

    def wait_ready(self, t=5.0): return self._ready.wait(t)
    def get_tcp(self):
        with self._lock: return list(self._tcp)
    def stop(self): self._stop.set()


class _Sender:
    def __init__(self):
        self._lock = threading.Lock()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5.0)
        s.connect((ROBOT_IP, ROBOT_PORT))
        self._s = s
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self):
        while True:
            try: self._s.recv(4096)
            except: time.sleep(0.01)

    def send(self, cmd):
        with self._lock:
            try: self._s.sendall((cmd.strip() + "\n").encode())
            except: pass

    def close(self):
        try: self._s.close()
        except: pass


def main():
    if not DROP_ZONE_FILE.exists():
        print(f"[ERROR] {DROP_ZONE_FILE} not found.")
        print("  Run:  python temp/capture_drop_zone.py  first.")
        return

    with open(DROP_ZONE_FILE) as f:
        dz = json.load(f)

    target_x = float(dz["x"])
    target_y = float(dz["y"])

    print("\n" + "=" * 50)
    print("  JOG TO DROP ZONE")
    print("=" * 50)
    print(f"  Target X : {target_x:.4f}")
    print(f"  Target Y : {target_y:.4f}")
    print(f"  Z        : current robot Z (unchanged)")
    print(f"  Rotation : recorded drop zone rx/ry/rz\n")

    print("Connecting to robot …")
    state  = _StateReader()
    sender = _Sender()
    state.start()
    if not state.wait_ready():
        print("[ERROR] Robot not reachable.")
        return
    print("Connected.\n")

    tcp = state.get_tcp()
    x, y, z, rx, ry, rz = tcp
    print(f"  Current TCP: X={x:.4f}  Y={y:.4f}  Z={z:.4f}")

    # Use rx/ry/rz from drop_zone.json if present, else keep current
    target_rx = float(dz.get("rx", rx))
    target_ry = float(dz.get("ry", ry))
    target_rz = float(dz.get("rz", rz))

    print(f"\n  Will move to: X={target_x:.4f}  Y={target_y:.4f}  Z={z:.4f}")
    print(f"  Rx={target_rx:.4f}  Ry={target_ry:.4f}  Rz={target_rz:.4f}")
    print("\n  Press ENTER to move (hand on E-stop), or type Q + ENTER to abort: ", end="")
    ans = input().strip().lower()
    if ans == "q":
        print("Aborted.")
        state.stop(); sender.close()
        return

    print("Moving …")
    sender.send(
        f"movel(p[{target_x:.6f},{target_y:.6f},{z:.6f},"
        f"{target_rx:.6f},{target_ry:.6f},{target_rz:.6f}],"
        f"a={MOVE_ACCEL:.4f},v={MOVE_SPEED:.4f})"
    )

    deadline = time.time() + 30.0
    while time.time() < deadline:
        cur = state.get_tcp()
        dist = ((cur[0]-target_x)**2 + (cur[1]-target_y)**2) ** 0.5
        if dist < XYZ_TOL_M:
            print(f"  Arrived.  TCP: X={cur[0]:.4f}  Y={cur[1]:.4f}  Z={cur[2]:.4f}")
            break
        time.sleep(0.05)
    else:
        cur = state.get_tcp()
        print(f"  Timeout.  TCP: X={cur[0]:.4f}  Y={cur[1]:.4f}  Z={cur[2]:.4f}")

    state.stop()
    sender.close()


if __name__ == "__main__":
    main()
