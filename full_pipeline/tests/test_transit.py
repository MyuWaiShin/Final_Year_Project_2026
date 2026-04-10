"""
tests/test_transit.py
---------------------
Stage 3 — YOLO Live Slip Detection (Transit) Test
15 trials: all deliberate slip events.

SLIP_DETECT must be False in transit.py (only Layer 2 YOLO active).
You yank the bottle out during transit — script checks whether the
system detects the slip and stops.

CSV columns saved
-----------------
  timestamp, trial_id, stage,
  yank_timing,       # "early" | "mid" | "late"  (you enter this)
  system_result,     # "empty" (detected) | "arrived" (missed)
  detected,          # 1 = slip caught, 0 = slip missed
  correct,           # same as detected (all trials are deliberate slips)
  tcp_x_at_stop, tcp_y_at_stop, tcp_z_at_stop,
  notes

Usage
-----
    cd full_pipeline
    python -m tests.test_transit
"""

import argparse
import os
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import navigate as nav_mod
import grasp   as grasp_mod
import transit as transit_mod
from tests.logger import TestLogger

TOTAL_TRIALS      = 15
TRIALS_PER_TIMING = 5   # 5 early + 5 mid + 5 late

TRANSIT_FIELDS = [
    "yank_timing",
    "system_result",
    "detected",
    "correct",
    "tcp_x_at_stop",
    "tcp_y_at_stop",
    "tcp_z_at_stop",
    "notes",
]

TIMING_GROUPS = [
    ("early", "Yank the bottle within the FIRST ~20% of the journey"),
    ("mid",   "Yank the bottle roughly HALFWAY to the drop zone"),
    ("late",  "Yank the bottle in the FINAL ~20% before the drop zone"),
]


def run_transit_test(autonomous_nav: bool = False):
    signal.signal(signal.SIGINT, lambda *_: os._exit(1))

    # Warn if SLIP_DETECT is on
    if transit_mod.SLIP_DETECT:
        print("\n  [!] WARNING: SLIP_DETECT = True in transit.py")
        print("      Set it to False before running Stage 3 tests.")
        print("      Only YOLO Layer 2 should be active for this test.")
        input("  Press ENTER to continue anyway, or Ctrl+C to exit: ")

    print()
    print("=" * 65)
    print("  STAGE 3 — YOLO LIVE SLIP DETECTION TEST (TRANSIT)")
    print("  15 trials, all deliberate slips — SLIP_DETECT = False")
    print("=" * 65)
    print()
    print("  CONTROLS (fixed):")
    print("    - SLIP_DETECT = False (Layer 1 width disabled)")
    print("    - YOLO26n classifier, threshold = 0.90")
    print("    - Consecutive empty frames to trigger = 3 (~300 ms)")
    print("    - Transit speed: 0.06 m/s")
    print("    - Drop zone: fixed position from data/drop_zone.json")
    print()
    print("  VARIABLE:")
    print("    - When during transit you yank the bottle: early / mid / late")
    print()
    print("  Every trial is a deliberate slip.")
    print("  If the robot STOPS mid-transit -> detected (correct).")
    print("  If the robot ARRIVES at the drop zone -> slip missed (wrong).")
    print()

    trial_num = 0

    with TestLogger("transit", extra_fields=TRANSIT_FIELDS) as log:
        for timing, timing_desc in TIMING_GROUPS:
            print(f"\n{'═'*65}")
            print(f"  TIMING: {timing.upper()}")
            print(f"  {timing_desc}")
            print(f"  {TRIALS_PER_TIMING} trials for this timing group")
            print(f"{'═'*65}")
            input("\n  Press ENTER to start this group …\n")

            for i in range(1, TRIALS_PER_TIMING + 1):
                trial_num += 1
                print(f"\n  {'─'*60}")
                print(f"  TRIAL {trial_num}/{TOTAL_TRIALS}  "
                      f"[{timing}  {i}/{TRIALS_PER_TIMING}]")
                print(f"  {'─'*60}")

                # ── Step 1: navigate() + grasp() ──────────────────────────────
                print("\n  [1/2] Getting clean hold (navigate + grasp) …")
                input("  Place bottle correctly, press ENTER: ")

                nav_result = nav_mod.main(autonomous=autonomous_nav)
                if nav_result is None:
                    print("  [!] navigate() failed — skipping trial.")
                    log.write(yank_timing=timing, system_result="nav_fail",
                              detected=0, correct=0, notes="navigate failed")
                    continue

                clearance_z = nav_result["clearance_z"]

                g = grasp_mod.main(autonomous=True)
                if g.get("result") != "holding":
                    print(f"  [!] grasp() returned '{g.get('result')}' — no clean hold.")
                    print("  Retrying this trial — place bottle again.")
                    input("  Press ENTER to retry navigate+grasp, or SKIP: ")
                    # allow retry or continue
                    g = grasp_mod.main(autonomous=True)
                    if g.get("result") != "holding":
                        print("  Still no hold — skipping.")
                        log.write(yank_timing=timing, system_result="grasp_fail",
                                  detected=0, correct=0, notes="grasp failed twice")
                        continue

                print(f"  Gripper width : {g.get('width_mm', '?')} mm  — clean hold.")

                # ── Step 2: transit() with yank ───────────────────────────────
                print(f"\n  [2/2] Transit starting toward drop zone.")
                print(f"  YANK the bottle {timing_desc.lower()} — then watch if it stops.")
                print()

                try:
                    t = transit_mod.main(clearance_z=clearance_z)
                except Exception as e:
                    print(f"  [!] transit() raised: {e}")
                    log.write(yank_timing=timing, system_result="ERROR",
                              detected=0, correct=0, notes=str(e))
                    continue

                system_result = t.get("result", "unknown")
                detected      = 1 if system_result == "empty" else 0
                correct       = detected   # all trials are deliberate slips

                # Get stop TCP from transit if available (transit doesn't return it,
                # so we parse from state directly if possible — log as None otherwise)
                tcp_stop = t.get("tcp_at_stop", [None, None, None])
                tx = round(tcp_stop[0], 5) if tcp_stop[0] is not None else None
                ty = round(tcp_stop[1], 5) if tcp_stop[1] is not None else None
                tz = round(tcp_stop[2], 5) if tcp_stop[2] is not None else None

                # Print outcome
                print(f"\n  ── Trial {trial_num} result {'─'*30}")
                print(f"  System returned : {system_result.upper()}")
                if detected:
                    print("  -> SLIP DETECTED (correct)")
                else:
                    print("  -> SLIP MISSED — robot arrived (incorrect)")

                notes = input("\n  Notes (ENTER to skip): ").strip()

                log.write(
                    yank_timing    = timing,
                    system_result  = system_result,
                    detected       = detected,
                    correct        = correct,
                    tcp_x_at_stop  = tx,
                    tcp_y_at_stop  = ty,
                    tcp_z_at_stop  = tz,
                    notes          = notes,
                )
                print(f"  Logged — trial {trial_num}/{TOTAL_TRIALS}.\n")

    print("\n  Stage 3 test complete.")
    print("  Results -> tests/results/transit_results.csv")
    print("  Analyse : python -m tests.analyse transit\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage 3 YOLO live slip detection test.")
    p.add_argument("--autonomous-nav", action="store_true",
                   help="Run navigate() in autonomous mode.")
    args = p.parse_args()
    run_transit_test(autonomous_nav=args.autonomous_nav)
