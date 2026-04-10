"""
tests/test_grasp.py
-------------------
Stage 1 — Gripper IO Failure Detection Test
10 trials total: 5 x air miss (first) + 5 x contact miss

Trials 1-5  (B_air_miss):
    Bottle removed completely before descent.
    Gripper closes on air → width ≈ 0 mm.

Trials 6-10 (A_contact_miss):
    Bottle nudged ~20 mm sideways after hover locked.
    Gripper edge-contacts → width < 11 mm or force fires without clean grasp.

navigate() runs in AUTONOMOUS mode (no YES prompt).
Scenario pre-noted in CSV automatically — no user input required per trial.

CSV columns logged
------------------
  timestamp, trial_id, stage,
  scenario,          # "B_air_miss" | "A_contact_miss"
  hover_x, hover_y, hover_z, pick_z,
  width_mm,          # calibrated jaw width after close
  force_detected,    # 1 | 0
  system_prediction, # "holding" | "missed"
  actual_grasped,    # 0 (always — deliberate miss)
  correct,           # 1 if system_prediction == "missed", else 0
  notes

Usage
-----
    cd full_pipeline
    python -m tests.test_grasp
"""

import json
import os
import signal
import socket
import struct
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import navigate as nav_mod
import grasp   as grasp_mod
from tests.logger import TestLogger

# ── Scan pose (robot returns here after every trial) ──────────────────────────
SCAN_POSE_FILE = ROOT / "data" / "scan_pose.json"
MOVE_SPEED     = 0.05
MOVE_ACCEL     = 0.025
ROBOT_IP       = "192.168.8.102"
ROBOT_PORT     = 30002
DASHBOARD_PORT = 29999
GRIP_OPEN_URP  = "/programs/myu/open_gripper.urp"
XYZ_TOL_M      = 0.003


class _SR(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._tcp   = [0.0]*6; self._lock = threading.Lock()
        self._ready = threading.Event(); self._stop = threading.Event()
    def run(self):
        while not self._stop.is_set():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(3.0); s.connect((ROBOT_IP, ROBOT_PORT))
                    while not self._stop.is_set():
                        hdr = b""
                        while len(hdr) < 4:
                            c = s.recv(4-len(hdr));  hdr += c if c else b""
                        plen = struct.unpack("!I", hdr)[0]
                        body = b""
                        while len(body) < plen-4:
                            c = s.recv(plen-4-len(body)); body += c if c else b""
                        off = 0
                        while off < len(body[1:]):
                            d = body[1:]
                            if off+5 > len(d): break
                            ps = struct.unpack("!I", d[off:off+4])[0]
                            if ps < 5 or off+ps > len(d): break
                            if d[off+4] == 4 and ps >= 53:
                                with self._lock:
                                    self._tcp = list(struct.unpack("!6d", d[off+5:off+53]))
                                self._ready.set()
                            off += ps
            except Exception: time.sleep(0.5)
    def get_tcp(self):
        with self._lock: return list(self._tcp)
    def wait(self, t=5.0): return self._ready.wait(timeout=t)
    def stop(self): self._stop.set()


class _SS:
    def __init__(self):
        s = socket.socket(); s.settimeout(5.0)
        s.connect((ROBOT_IP, ROBOT_PORT)); self._s = s
        threading.Thread(target=self._drain, daemon=True).start()
    def _drain(self):
        while True:
            try: self._s.recv(4096)
            except Exception: break
    def send(self, cmd):
        try: self._s.sendall((cmd.strip()+"\n").encode())
        except Exception: pass
    def close(self):
        try: self._s.close()
        except Exception: pass


def _dashboard(cmd):
    try:
        s = socket.socket(); s.settimeout(5.0)
        s.connect((ROBOT_IP, DASHBOARD_PORT)); s.recv(1024)
        s.sendall((cmd+"\n").encode()); s.recv(1024); s.close()
    except Exception: pass


def _open_gripper():
    """Open gripper via dashboard URP."""
    print("  [Gripper] Opening …")
    _dashboard("stop")
    time.sleep(0.1)
    _dashboard(f"load {GRIP_OPEN_URP}")
    time.sleep(0.1)
    _dashboard("play")
    time.sleep(2.0)   # wait for URP to complete
    print("  [Gripper] Open.")


def _return_to_scan_pose(scan_tcp):
    """Open gripper then movel back to scan pose."""
    _open_gripper()
    sx, sy, sz = scan_tcp[0], scan_tcp[1], scan_tcp[2]
    srx, sry, srz = scan_tcp[3], scan_tcp[4], scan_tcp[5]
    state = _SR(); state.start()
    sender = _SS()
    if not state.wait(5.0):
        print("  [!] Cannot return to scan pose — robot not ready.")
        state.stop(); sender.close(); return
    cur = state.get_tcp()
    dist = sum((a-b)**2 for a,b in zip(cur[:3],[sx,sy,sz]))**0.5
    if dist > XYZ_TOL_M:
        print("  [Return] Moving to scan pose …")
        sender.send(f"movel(p[{sx:.6f},{sy:.6f},{sz:.6f},{srx:.6f},{sry:.6f},{srz:.6f}],"
                    f"a={MOVE_ACCEL:.4f},v={MOVE_SPEED:.4f})")
        deadline = time.time() + 30.0
        while time.time() < deadline:
            cur = state.get_tcp()
            if sum((a-b)**2 for a,b in zip(cur[:3],[sx,sy,sz]))**0.5 < XYZ_TOL_M:
                break
            time.sleep(0.02)
        print("  [Return] At scan pose.")
    state.stop(); sender.close()

# ── Trial plan:  (trial index 1-10, scenario_name, short note auto-logged) ────
TRIALS = (
    [(f"B_air_miss",     "bottle removed — air close") for _ in range(5)] +
    [(f"A_contact_miss", "bottle nudged sideways — edge contact") for _ in range(5)]
)

GRASP_FIELDS = [
    "scenario",
    "hover_x", "hover_y", "hover_z", "pick_z",
    "width_mm",
    "force_detected",
    "system_prediction",
    "actual_grasped",
    "correct",
    "notes",
]


def run_grasp_test():
    signal.signal(signal.SIGINT, lambda *_: os._exit(1))

    print()
    print("=" * 65)
    print("  STAGE 1 — GRIPPER IO FAILURE DETECTION TEST")
    print("  10 trials: trials 1-5 air miss | trials 6-10 contact miss")
    print("=" * 65)
    print()
    print("  Trials 1-5  (B_air_miss):")
    print("    Remove the bottle COMPLETELY before gripper descends.")
    print()
    print("  Trials 6-10 (A_contact_miss):")
    print("    Nudge bottle ~20 mm sideways AFTER hover is locked.")
    print()
    print("  navigate() runs autonomous — NO 'YES' prompt.")
    print("  Each trial: place bottle → ENTER → navigate hovers → you")
    print("  set up failure → ENTER → grasp descends → result logged.")
    print()

    with open(SCAN_POSE_FILE) as f:
        scan_tcp = json.load(f)["tcp_pose"]

    print("  [Setup] Opening gripper + moving to scan pose …")
    _return_to_scan_pose(scan_tcp)   # _return_to_scan_pose opens gripper then moves
    print()

    with TestLogger("grasp", extra_fields=GRASP_FIELDS) as log:
        for trial_num, (scenario_name, auto_note) in enumerate(TRIALS, start=1):
            total = len(TRIALS)
            print(f"\n{'═'*65}")
            print(f"  TRIAL {trial_num}/{total}  [{scenario_name}]")
            print(f"  Auto-note: {auto_note}")
            print(f"{'═'*65}")

            # Section header on first trial of each group
            if trial_num == 1:
                print("\n  >>> TRIALS 1-5: Remove bottle completely before descent <<<")
            elif trial_num == 6:
                print("\n  >>> TRIALS 6-10: Nudge bottle sideways after hover <<<")

            # ── Step 1: place bottle + navigate ───────────────────────────────
            input("\n  Place bottle correctly, press ENTER to navigate: ")
            try:
                nav_result = nav_mod.main(autonomous=False)   # manual: SPACE + YES prompt
            except Exception as e:
                print(f"  [!] navigate() raised: {e}")
                log.write(scenario=scenario_name, system_prediction="ERROR",
                          actual_grasped=0, correct=0, notes=str(e))
                continue

            if nav_result is None:
                print("  [!] navigate() returned None — skipped.")
                log.write(scenario=scenario_name, system_prediction="none",
                          actual_grasped=0, correct=0, notes="navigate returned None")
                continue

            hover_pose = nav_result["hover_pose"]
            hx, hy, hz = hover_pose[:3]
            pick_z     = hz - grasp_mod.DESCEND_OFFSET

            print(f"\n  Hover locked: X={hx:.4f}  Y={hy:.4f}  Z={hz:.4f}")
            print(f"  Pick Z      : {pick_z:.4f}")

            # ── Step 2: user sets up the failure ─────────────────────────────
            if scenario_name == "B_air_miss":
                print("\n  REMOVE the bottle now (robot will close on air).")
            else:
                print("\n  NUDGE the bottle ~20 mm sideways now (edge contact).")
            input("  Done? Press ENTER to run grasp: ")

            # ── Step 3: grasp ────────────────────────────────────────────────
            print("\n  [grasp] Descending and closing …")
            try:
                g = grasp_mod.main(autonomous=True)
            except Exception as e:
                print(f"  [!] grasp() raised: {e}")
                log.write(scenario=scenario_name,
                          hover_x=round(hx,5), hover_y=round(hy,5),
                          hover_z=round(hz,5), pick_z=round(pick_z,5),
                          system_prediction="ERROR",
                          actual_grasped=0, correct=0, notes=str(e))
                continue

            # ── Step 4: log ───────────────────────────────────────────────────
            sys_pred  = g.get("result",   "unknown")
            width_mm  = g.get("width_mm", None)
            force_det = 1 if g.get("force", False) else 0
            correct   = 1 if sys_pred == "missed" else 0

            outcome = "CORRECT (miss detected)" if correct else "WRONG — false positive (said holding)"
            print(f"\n  Width: {width_mm} mm  |  Force: {'YES' if force_det else 'NO'}")
            print(f"  System: {sys_pred.upper()}  ->  {outcome}")

            log.write(
                scenario          = scenario_name,
                hover_x           = round(hx, 5),
                hover_y           = round(hy, 5),
                hover_z           = round(hz, 5),
                pick_z            = round(pick_z, 5),
                width_mm          = width_mm,
                force_detected    = force_det,
                system_prediction = sys_pred,
                actual_grasped    = 0,
                correct           = correct,
                notes             = auto_note,
            )
            print(f"  Logged — trial {trial_num}/{total}.")
            _return_to_scan_pose(scan_tcp)
            print()

    print("\n  Stage 1 test complete.")
    print("  Results -> tests/results/grasp_results.csv")
    print("  Analyse : python -m tests.analyse grasp\n")


if __name__ == "__main__":
    run_grasp_test()
