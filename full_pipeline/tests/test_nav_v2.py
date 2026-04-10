"""
tests/test_nav_v2.py
--------------------
Navigation accuracy test — explore → navigate.

Goal
----
Characterise hover XYZ accuracy across arbitrary workspace positions.

Sequence per repeat
-------------------
  1. Place bottle at labelled position, press ENTER
  2. explore()  — scan + J0 sweep to locate the object
  3. navigate() — hover above the object
  4. You judge: gripper centred above bottle? (Y/N)
  5. TCP hover pose logged → next repeat

Each position is repeated 3 times WITHOUT moving the bottle between repeats.

Results → tests/results/nav_v2_results.csv

Usage
-----
    python -m tests.test_nav_v2
    python -m tests.test_nav_v2 --start 4   # resume from position 4
"""

import argparse
import os
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import explore
import navigate as nav_mod
from tests.logger import TestLogger

# ── Positions ─────────────────────────────────────────────────────────────────
# label → approximate workspace description
# You place the bottle; actual XY is inferred from the TCP hover log
POSITIONS = {
    1:  "P1 — grid near-left",
    2:  "P2 — grid near-centre",
    3:  "P3 — grid near-right",
    4:  "P4 — grid far-left",
    5:  "P5 — grid far-centre",
    6:  "P6 — grid far-right",
    7:  "P7 — extra far-left (~-200mm from ref)",
    8:  "P8 — extra far-right (~+420mm from ref)",
    9:  "P9 — extra far-away (~+300mm from ref)",
}

REPEATS_PER_POS = 3

NAV_V2_FIELDS = [
    "position_id", "position_label", "repeat_num",
    "detection_mode",
    "hover_x", "hover_y", "hover_z", "clearance_z",
    "aligned",    # 1 = yes, 0 = no, -1 = aborted
    "notes",
]


def run_nav_v2_test(start_pos: int = 1):
    signal.signal(signal.SIGINT, lambda *_: os._exit(1))

    total = len(POSITIONS)
    print()
    print("=" * 60)
    print("  NAVIGATION ACCURACY TEST v2  (explore → navigate)")
    print("=" * 60)
    print(f"  {total} positions × {REPEATS_PER_POS} repeats = {total*REPEATS_PER_POS} trials")
    print()
    print("  Per repeat:")
    print("    1. Place bottle at labelled position → ENTER")
    print("    2. explore() scans and locates the object")
    print("    3. navigate() hovers above the object")
    print("    4. Judge alignment → Y / N")
    print("    (Do NOT move bottle between the 3 repeats)")
    print()

    trial_id = 0

    with TestLogger("nav_v2", extra_fields=NAV_V2_FIELDS) as log:
        for pos_id in range(start_pos, total + 1):
            label = POSITIONS[pos_id]

            print(f"\n{'═'*60}")
            print(f"  POSITION {pos_id}/{total}  —  {label}")
            print(f"{'═'*60}")
            print()
            raw = input(f"  Place bottle at {label},  then press ENTER  (or Q to quit): ").strip().upper()
            if raw == 'Q':
                break

            for rep in range(1, REPEATS_PER_POS + 1):
                trial_id += 1
                print(f"\n  ── Repeat {rep}/{REPEATS_PER_POS}  (trial {trial_id}) ──")

                # ── Stage 1: Explore ─────────────────────────────────────────
                print("\n  [1/2] explore() — scanning …")
                found = explore.main(autonomous=True)
                if not found:
                    print("  [!] explore() failed — object not found.")
                    log.write(
                        trial_id=trial_id, position_id=pos_id, position_label=label,
                        repeat_num=rep, detection_mode="none",
                        hover_x=0, hover_y=0, hover_z=0, clearance_z=0,
                        aligned=-1, notes="explore failed"
                    )
                    continue

                # ── Stage 2: Navigate ────────────────────────────────────────
                print("\n  [2/2] navigate() — hovering above object …")
                try:
                    nav_result = nav_mod.main(autonomous=True, force_yolo=False)
                except Exception as e:
                    print(f"  [!] navigate() raised: {e}")
                    log.write(
                        trial_id=trial_id, position_id=pos_id, position_label=label,
                        repeat_num=rep, detection_mode="ERROR",
                        hover_x=0, hover_y=0, hover_z=0, clearance_z=0,
                        aligned=-1, notes=str(e)
                    )
                    continue

                if nav_result is None:
                    print("  [!] navigate() aborted.")
                    log.write(
                        trial_id=trial_id, position_id=pos_id, position_label=label,
                        repeat_num=rep, detection_mode="aborted",
                        hover_x=0, hover_y=0, hover_z=0, clearance_z=0,
                        aligned=-1, notes="navigate aborted"
                    )
                    continue

                hover_pose     = nav_result["hover_pose"]
                clearance_z    = nav_result["clearance_z"]
                detection_mode = nav_result.get("detection_mode", "unknown")
                hx, hy, hz     = hover_pose[:3]

                print(f"\n  Hover pose: X={hx:.4f}  Y={hy:.4f}  Z={hz:.4f}")

                # ── Judge alignment ──────────────────────────────────────────
                raw = input("  Gripper centred above bottle? (Y / N): ").strip().upper()
                aligned = 1 if raw == 'Y' else 0
                notes   = input("  Notes (ENTER to skip): ").strip()

                log.write(
                    trial_id       = trial_id,
                    position_id    = pos_id,
                    position_label = label,
                    repeat_num     = rep,
                    detection_mode = detection_mode,
                    hover_x        = round(hx, 5),
                    hover_y        = round(hy, 5),
                    hover_z        = round(hz, 5),
                    clearance_z    = round(clearance_z, 5),
                    aligned        = aligned,
                    notes          = notes,
                )
                status = "✓ ALIGNED" if aligned else "✗ MISALIGNED"
                print(f"  {status} — logged trial {trial_id}.")

    print("\n  Nav v2 test complete.")
    print("  Results → tests/results/nav_v2_results.csv\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Navigation accuracy test v2.")
    p.add_argument("--start", type=int, default=1, metavar="N",
                   help="Start from position N (default: 1)")
    args = p.parse_args()
    run_nav_v2_test(start_pos=args.start)
