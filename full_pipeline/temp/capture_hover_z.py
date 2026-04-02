"""
capture_hover_z.py
------------------
Captures the hover Z for the pipeline by jogging the gripper to the
exact pick position (where it would close on the object), then computing:

    hover_z = pick_z + DESCEND_OFFSET (70 mm)

This fixed hover_z is used by navigate.py so the robot always approaches
at the correct height, independent of depth camera accuracy.

Usage
-----
1. Jog the robot so the gripper is at the exact position it would pick the object.
2. Run:  python temp/capture_hover_z.py
3. Press SPACE to capture.
4. Saves to data/hover_z.json.
"""

import json
import socket
import struct
import sys
import threading
import time
from pathlib import Path

SCRIPT_DIR    = Path(__file__).resolve().parent.parent   # full_pipeline/
OUT_FILE      = SCRIPT_DIR / "data" / "hover_z.json"

ROBOT_IP      = "192.168.8.102"
ROBOT_PORT    = 30002

# Read DESCEND_OFFSET live from grasp.py so it stays in sync automatically
sys.path.insert(0, str(SCRIPT_DIR))
from grasp import DESCEND_OFFSET


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


def main():
    print("\n" + "=" * 55)
    print("  CAPTURE HOVER Z")
    print("=" * 55)
    print(f"  Jog the gripper to the exact pick position,")
    print(f"  then press ENTER.\n")
    print(f"  hover_z = pick_z + {DESCEND_OFFSET*1000:.0f} mm\n")

    print("Connecting to robot …")
    state = _StateReader()
    state.start()
    if not state.wait_ready():
        print("[ERROR] Robot not reachable.")
        return

    print("Connected. Streaming TCP pose. Press ENTER to capture, Q+ENTER to abort.\n")

    # Live print loop until user hits enter
    import msvcrt, sys

    print("  Current TCP Z (live):")
    captured = False
    while True:
        tcp = state.get_tcp()
        print(f"\r    X={tcp[0]:.4f}  Y={tcp[1]:.4f}  Z={tcp[2]:.4f}    ", end="", flush=True)
        time.sleep(0.1)

        if msvcrt.kbhit():
            key = msvcrt.getwch()
            if key == '\r' or key == ' ':   # Enter or Space
                print()
                captured = True
                break
            elif key.lower() == 'q':
                print("\nAborted.")
                break

    if captured:
        tcp     = state.get_tcp()
        pick_z  = tcp[2]
        hover_z = pick_z + DESCEND_OFFSET

        data = {
            "_comment": f"pick_z captured by jogging. hover_z = pick_z + {DESCEND_OFFSET*1000:.0f}mm (DESCEND_OFFSET).",
            "pick_z":  round(pick_z,  6),
            "hover_z": round(hover_z, 6),
        }
        OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_FILE, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\n  pick_z  = {pick_z:.4f} m")
        print(f"  hover_z = {hover_z:.4f} m  (+ {DESCEND_OFFSET*1000:.0f} mm)")
        print(f"\n  Saved → {OUT_FILE}\n")

    state.stop()


if __name__ == "__main__":
    main()
