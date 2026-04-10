"""
tests/test_verify.py
---------------------
Stage 2 — Binary YOLO Classifier Test

Controls (cv2 window must be focused):
  O      → open gripper   (empty)
  C      → close gripper  (put bottle in first)
  SPACE  → capture frame, run YOLO, prompt for actual label → log
  Q      → quit

Gripper opens/closes in a background thread so the live feed keeps running.
SPACE pauses the loop briefly to collect the terminal label, then resumes.

CSV columns
-----------
  timestamp, trial_id, stage,
  actual,        # "holding" | "empty"  (you type h/e after SPACE)
  predicted,     # "holding" | "empty"  (YOLO output)
  yolo_conf,     # p_holding probability
  tcp_x, tcp_y, tcp_z,
  correct,       # 1 if predicted == actual else 0
  notes

Usage
-----
    cd full_pipeline
    python -m tests.test_verify
"""

import os
import signal
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from verify import open_camera, grab_frame, run_yolo, THRESHOLD, YOLO_MODEL
from tests.logger import TestLogger

ROBOT_IP       = "192.168.8.102"
ROBOT_PORT     = 30002
DASHBOARD_PORT = 29999
GRIP_CLOSE_URP = "/programs/myu/close_gripper.urp"
GRIP_OPEN_URP  = "/programs/myu/open_gripper.urp"

VERIFY_FIELDS  = ["actual", "predicted", "yolo_conf",
                  "tcp_x", "tcp_y", "tcp_z", "correct", "notes"]


# ── Robot state reader ────────────────────────────────────────────────────────
class _StateReader(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._tcp   = [0.0]*6
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
                            c = s.recv(4-len(hdr))
                            if not c: break
                            hdr += c
                        plen = struct.unpack("!I", hdr)[0]
                        body = b""
                        while len(body) < plen-4:
                            c = s.recv(plen-4-len(body))
                            if not c: break
                            body += c
                        off, d = 0, body[1:]
                        while off < len(d):
                            if off+5 > len(d): break
                            ps = struct.unpack("!I", d[off:off+4])[0]
                            if ps < 5 or off+ps > len(d): break
                            if d[off+4] == 4 and ps >= 53:
                                with self._lock:
                                    self._tcp = list(struct.unpack("!6d", d[off+5:off+53]))
                                self._ready.set()
                            off += ps
            except Exception:
                time.sleep(0.5)

    def get_tcp(self):
        with self._lock: return list(self._tcp)
    def wait(self, t=5.0): return self._ready.wait(timeout=t)
    def stop(self): self._stop.set()


# ── Dashboard / gripper ───────────────────────────────────────────────────────
def _dashboard(cmd: str):
    try:
        s = socket.socket(); s.settimeout(5.0)
        s.connect((ROBOT_IP, DASHBOARD_PORT))
        s.recv(1024)
        s.sendall((cmd + "\n").encode())
        s.recv(1024); s.close()
    except Exception as e:
        print(f"  [Dashboard] {cmd!r} failed: {e}")


def _gripper(action: str):
    urp = GRIP_OPEN_URP if action == "open" else GRIP_CLOSE_URP
    print(f"\n  [Gripper] {action.capitalize()}ing …", flush=True)
    _dashboard("stop");  time.sleep(0.1)
    _dashboard(f"load {urp}"); time.sleep(0.1)
    _dashboard("play");  time.sleep(2.0)
    print(f"  [Gripper] {action.capitalize()}ed.\n", flush=True)


# ── Main loop ─────────────────────────────────────────────────────────────────
def run_verify_test():
    signal.signal(signal.SIGINT, lambda *_: os._exit(1))

    print()
    print("=" * 62)
    print("  BINARY CLASSIFICATION TEST  (cv2 window must be focused)")
    print("=" * 62)
    print("  O      → open gripper   (empty)")
    print("  C      → close gripper  (holding)")
    print("  SPACE  → classify + log")
    print("  Q      → quit")
    print()

    # Load model
    print("  Loading YOLO classifier …")
    if not YOLO_MODEL.exists():
        raise FileNotFoundError(f"Model not found: {YOLO_MODEL}")
    model = YOLO(str(YOLO_MODEL), task="classify")
    print("  Model ready.")

    # Start state reader
    state = _StateReader(); state.start()
    if not state.wait(5.0):
        print("  [!] WARNING: Robot TCP unavailable — tcp_x/y/z will be 0.")

    # Open camera
    print("  Opening camera …")
    cam_device, queue = open_camera()
    print("  Camera ready. Focus the cv2 window and press O / C / SPACE.\n")

    gripper_busy = threading.Event()   # don't double-trigger gripper

    with TestLogger("verify", extra_fields=VERIFY_FIELDS) as log:
        trial   = 0
        last_frame = None

        while True:
            # Grab latest frame
            pkt = queue.tryGet()
            if pkt is not None:
                last_frame = pkt.getCvFrame()

            if last_frame is not None:
                disp = cv2.resize(last_frame, (960, 540))
                cv2.imshow("Binary Classification Test", disp)

            key = cv2.waitKey(33) & 0xFF   # ~30 fps poll

            if key == ord('q') or key == 27:      # Q or ESC
                print("\n  Quitting …")
                break

            elif key == ord('o') and not gripper_busy.is_set():
                gripper_busy.set()
                threading.Thread(
                    target=lambda: (_gripper("open"), gripper_busy.clear()),
                    daemon=True
                ).start()

            elif key == ord('c') and not gripper_busy.is_set():
                gripper_busy.set()
                threading.Thread(
                    target=lambda: (_gripper("close"), gripper_busy.clear()),
                    daemon=True
                ).start()

            elif key == ord(' '):
                # ── Classify ──────────────────────────────────────────────────
                trial += 1
                print(f"\n  ── TRIAL {trial} ─────────────────────────────────────")
                tcp = state.get_tcp()
                tx, ty, tz = round(tcp[0],5), round(tcp[1],5), round(tcp[2],5)

                print(f"  TCP: X={tx}  Y={ty}  Z={tz}")
                print("  Running YOLO …", end="", flush=True)
                frame = grab_frame(queue, warmup_frames=3)
                p_holding, p_empty = run_yolo(model, frame)
                predicted = "holding" if p_holding >= THRESHOLD else "empty"
                print(f"\r  YOLO: p_holding={p_holding:.3f}  p_empty={p_empty:.3f}")
                print(f"  Predicted : {predicted.upper()}")

                # ── Get actual label from cv2 window ─────────────────────────────
                print("  Press '1' for HOLDING or '2' for EMPTY in the camera window.")
                actual = None
                while True:
                    # Keep showing frame so window doesn't freeze
                    if last_frame is not None:
                        disp = cv2.resize(last_frame, (960, 540))
                        cv2.putText(disp, "Press 1 (HOLDING) or 2 (EMPTY)", (20, 40), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.imshow("Binary Classification Test", disp)
                        
                    k = cv2.waitKey(33) & 0xFF
                    
                    # Update background queue so it doesn't build up stale frames
                    pkt = queue.tryGet()
                    if pkt is not None:
                        last_frame = pkt.getCvFrame()

                    if k == ord('1'):
                        actual = "holding"
                        break
                    elif k == ord('2'):
                        actual = "empty"
                        break

                correct = 1 if predicted == actual else 0
                print(f"  Actual    : {actual.upper()}")
                print(f"  -> {'CORRECT' if correct else 'WRONG'}")

                log.write(
                    actual    = actual,
                    predicted = predicted,
                    yolo_conf = round(p_holding, 4),
                    tcp_x     = tx, tcp_y = ty, tcp_z = tz,
                    correct   = correct,
                    notes     = "",
                )
                print(f"  Logged — trial {trial}.\n")

    cam_device.close()
    cv2.destroyAllWindows()
    state.stop()


if __name__ == "__main__":
    run_verify_test()
