"""
capture_drop_zone.py
--------------------
Utility to record the drop zone position for transit.py.

Usage
-----
1. Jog the robot TCP to where you want the gripper to be above the drop box.
2. Run:  python capture_drop_zone.py
3. A live OAK-D feed opens so you can verify what the robot sees.
4. Press SPACE to capture — saves x, y, rx, ry, rz to data/drop_zone.json.
5. Press Q to quit without saving.

Z is NOT saved — transit.py always uses clearance_z at runtime.
"""

import json
import socket
import struct
import threading
import time
from pathlib import Path

import cv2
import depthai as dai

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR     = Path(__file__).resolve().parent.parent   # full_pipeline/
OUT_FILE       = SCRIPT_DIR / "data" / "drop_zone.json"

# ── Robot ─────────────────────────────────────────────────────────────────────
ROBOT_IP   = "192.168.8.102"
ROBOT_PORT = 30002

# ── Camera ────────────────────────────────────────────────────────────────────
MANUAL_FOCUS = 46


# ── Robot state reader (minimal — TCP pose only) ──────────────────────────────
class _StateReader(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._lock     = threading.Lock()
        self._stop     = threading.Event()
        self._ready    = threading.Event()
        self._tcp_pose = [0.0] * 6

    def run(self):
        while not self._stop.is_set():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(3.0)
                    s.connect((ROBOT_IP, ROBOT_PORT))
                    while not self._stop.is_set():
                        hdr = self._recv(s, 4)
                        if hdr is None:
                            break
                        plen = struct.unpack("!I", hdr)[0]
                        body = self._recv(s, plen - 4)
                        if body is None:
                            break
                        self._parse(body[1:])
            except Exception:
                time.sleep(0.5)

    @staticmethod
    def _recv(s, n):
        buf = b""
        while len(buf) < n:
            chunk = s.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def _parse(self, data):
        off = 0
        while off < len(data):
            if off + 5 > len(data):
                break
            ps = struct.unpack("!I", data[off:off+4])[0]
            if ps < 5 or off + ps > len(data):
                break
            if data[off + 4] == 4 and ps >= 53:
                with self._lock:
                    self._tcp_pose = list(struct.unpack("!6d", data[off+5:off+53]))
                self._ready.set()
            off += ps

    def wait_ready(self, timeout=5.0):
        return self._ready.wait(timeout=timeout)

    def get_tcp_pose(self):
        with self._lock:
            return list(self._tcp_pose)

    def stop(self):
        self._stop.set()


def main():
    print("\n" + "=" * 55)
    print("  DROP ZONE CAPTURE")
    print("=" * 55)
    print(f"  Output : {OUT_FILE}")
    print("  Jog the robot above the drop box, then press SPACE.\n")

    # ── Robot ─────────────────────────────────────────────────────────────────
    print("Connecting to robot …")
    state = _StateReader()
    state.start()
    if not state.wait_ready(timeout=5.0):
        raise RuntimeError(f"Robot not reachable at {ROBOT_IP}:{ROBOT_PORT}")
    print("Robot connected!\n")

    # ── Camera ────────────────────────────────────────────────────────────────
    print("Opening camera …")
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
    print("Camera ready.  Window open — press SPACE to capture, Q to quit.\n")

    captured = False

    while True:
        pkt = queue.tryGet()
        if pkt is not None:
            frame = pkt.getCvFrame()
            tcp   = state.get_tcp_pose()
            x, y, z, rx, ry, rz = tcp

            # Overlay current TCP
            info = (f"X={x:.4f}  Y={y:.4f}  Z={z:.4f}"
                    f"  Rx={rx:.4f}  Ry={ry:.4f}  Rz={rz:.4f}")
            cv2.putText(frame, info, (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(frame, "SPACE = capture    Q = quit",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
            cv2.imshow("capture_drop_zone", cv2.resize(frame, (960, 540)))

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            tcp = state.get_tcp_pose()
            x, y = tcp[0], tcp[1]
            data = {
                "_comment": "Drop zone XY only. Z set at runtime to clearance_z. Orientation kept from robot at transit time.",
                "x": round(x, 6),
                "y": round(y, 6),
            }
            OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(OUT_FILE, "w") as f:
                json.dump(data, f, indent=2)
            print(f"\n  ✓ Captured:  X={x:.4f}  Y={y:.4f}")
            print(f"  Saved → {OUT_FILE}\n")
            captured = True
            break

        elif key == ord('q') or key == 27:
            print("\n  Quit — nothing saved.")
            break

    cv2.destroyAllWindows()
    device.close()
    state.stop()

    if captured:
        print("Done. Run the pipeline when ready.")
    return captured


if __name__ == "__main__":
    main()
