"""
tests/capture_grid_ref.py
--------------------------
Capture and save the current robot TCP as the grid reference point (0, 0, 0).

Usage
-----
    1. Jog the robot TCP precisely to the top-left corner of the printed A3 grid.
    2. Run:
           cd full_pipeline
           python -m tests.capture_grid_ref
    3. The TCP is saved to  data/grid_ref.json

This reference is used by test_nav.py to convert robot TCP coordinates
into grid-relative XY offsets for logging.
"""

import json
import socket
import struct
import threading
import time
from pathlib import Path

ROBOT_IP   = "192.168.8.102"
ROBOT_PORT = 30002
OUT_FILE   = Path(__file__).resolve().parent.parent / "data" / "grid_ref.json"


class _Reader(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._tcp   = [0.0] * 6
        self._lock  = threading.Lock()
        self._ready = threading.Event()
        self._stop  = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(3.0)
                    s.connect((ROBOT_IP, ROBOT_PORT))
                    while not self._stop.is_set():
                        hdr = b""
                        while len(hdr) < 4:
                            c = s.recv(4 - len(hdr))
                            if not c: break
                            hdr += c
                        plen = struct.unpack("!I", hdr)[0]
                        body = b""
                        while len(body) < plen - 4:
                            c = s.recv(plen - 4 - len(body))
                            if not c: break
                            body += c
                        self._parse(body[1:])
            except Exception:
                time.sleep(0.5)

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

    def get_tcp(self):
        with self._lock:
            return list(self._tcp)

    def wait(self, timeout=5.0): return self._ready.wait(timeout=timeout)
    def stop(self): self._stop.set()


if __name__ == "__main__":
    print("\n  Capture Grid Reference (0, 0, 0)")
    print("  ─────────────────────────────────")
    print("  Jog the robot TCP to the TOP-LEFT corner of the printed A3 grid.")
    print("  That corner is your (0, 0) reference for all 6 circle positions.")
    print()
    input("  TCP at corner? Press ENTER to capture: ")

    reader = _Reader()
    reader.start()
    if not reader.wait(timeout=5.0):
        print("  [!] Could not read robot state — is the robot on and connected?")
        reader.stop()
        exit(1)

    tcp = reader.get_tcp()
    reader.stop()

    ref = {
        "x": round(tcp[0], 6),
        "y": round(tcp[1], 6),
        "z": round(tcp[2], 6),
        "rx": round(tcp[3], 6),
        "ry": round(tcp[4], 6),
        "rz": round(tcp[5], 6),
        "note": "Top-left corner of A3 grid = robot XY reference (0,0)"
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(ref, f, indent=2)

    print(f"\n  Reference saved -> {OUT_FILE}")
    print(f"  X={ref['x']:.5f}  Y={ref['y']:.5f}  Z={ref['z']:.5f}")
    print()
    print("  Grid circle centres (from this reference):")
    circles = {
        1: (60.0, 73.5), 2: (210.0, 73.5), 3: (360.0, 73.5),
        4: (60.0, 223.5), 5: (210.0, 223.5), 6: (360.0, 223.5),
    }
    for label, (gx, gy) in circles.items():
        rx = ref["x"] + gx / 1000.0
        ry = ref["y"] + gy / 1000.0
        print(f"    P{label}: robot X={rx:.5f}  Y={ry:.5f}  (grid {gx}mm, {gy}mm)")
    print()
