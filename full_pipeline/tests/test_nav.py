"""
tests/test_nav.py
-----------------
Navigation evaluation script.

Runs:
  1. Explore (sweep scan, find object)
  2. Navigate (hover over object, centre)
  3. Prompts user for 'can grab?' (Y/N)

Results logged to tests/results/nav_results.csv
"""

import json
import argparse
import os
import signal
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import explore
import navigate as nav_mod
from tests.logger import TestLogger

NAV_FIELDS = [
    "detection_mode",
    "hover_x", "hover_y", "hover_z", "clearance_z",
    "can_grab",   # 1 = yes, 0 = no, -1 = skipped
    "notes"
]


def run_nav_test(start_trial: int = 1, autonomous: bool = True):
    signal.signal(signal.SIGINT, lambda *_: os._exit(1))

    print("\n" + "=" * 60)
    print("  NAVIGATION TARGETING TEST (Explore + Navigate)")
    print("=" * 60)
    print("  Sequence per trial:")
    print("    1. Sweep scan (find object)")
    print("    2. Navigate (hover and centre)")
    print("    3. User visually checks alignment -> input Y/N")
    print()

    with TestLogger("nav", extra_fields=NAV_FIELDS) as log:
        trial_num = start_trial

        while True:
            print(f"\n{'═'*60}")
            print(f"  TRIAL {trial_num}   (Press Q at prompt to quit)")
            print(f"{'═'*60}")
            print()

            raw = input(f"  Press ENTER to start scan (or Q to quit): ").strip().upper()
            if raw == 'Q':
                break

            # ── 1. Explore ──────────────────────────────────────────────────
            print("\n  [STAGE 1] Explore ...")
            explore_result = explore.main(autonomous=autonomous)
            if not explore_result:
                print("  [!] Object not found. Skipping trial.")
                log.write(trial_id=trial_num, stage="nav", failure_type="explore_fail",
                          attempts_taken=1, recovered=0, object_found=0,
                          detection_mode="none", hover_x=0, hover_y=0, hover_z=0, clearance_z=0,
                          can_grab=-1, notes="explore failed")
                trial_num += 1
                continue

            # ── 2. Navigate ─────────────────────────────────────────────────
            print("\n  [STAGE 2] Navigate ...")
            try:
                nav_result = nav_mod.main(autonomous=autonomous, force_yolo=False)
            except Exception as e:
                print(f"  [!] navigate() raised: {e}")
                log.write(trial_id=trial_num, stage="nav", failure_type="navigate_error",
                          attempts_taken=1, recovered=0, object_found=1,
                          detection_mode="ERROR", hover_x=0, hover_y=0, hover_z=0, clearance_z=0,
                          can_grab=-1, notes=str(e))
                trial_num += 1
                continue

            if nav_result is None:
                print("  [!] navigate() aborted.")
                trial_num += 1
                continue

            hover_pose     = nav_result["hover_pose"]
            clearance_z    = nav_result["clearance_z"]
            detection_mode = nav_result.get("detection_mode", "unknown")
            hx, hy, hz     = hover_pose[:3]

            print(f"\n  Hover pose reached: X={hx:.4f}  Y={hy:.4f}  Z={hz:.4f}")

            # ── 3. Evaluate ─────────────────────────────────────────────────
            print("\n  Check alignment:")
            raw = input("  Can it grab? (Y / N): ").strip().upper()
            can_grab = 1 if raw == 'Y' else 0

            notes = input("  Notes (ENTER to skip): ").strip()

            log.write(
                trial_id       = trial_num,
                stage          = "nav",
                failure_type   = "none",
                attempts_taken = 1,
                recovered      = 1,
                object_found   = 1,
                detection_mode = detection_mode,
                hover_x        = round(hx, 5),
                hover_y        = round(hy, 5),
                hover_z        = round(hz, 5),
                clearance_z    = round(clearance_z, 5),
                can_grab       = can_grab,
                notes          = notes,
            )

            print(f"\n  ✓ Logged trial {trial_num}.")
            trial_num += 1

    print("\n  Nav test complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Live Navigation placement test.")
    p.add_argument("--start", type=int, default=1, metavar="N", help="Start trial number")
    args = p.parse_args()
    
    # Force autonomous=True internally as with test_recover.py
    run_nav_test(start_trial=args.start, autonomous=True)
