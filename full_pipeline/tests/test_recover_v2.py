"""
tests/test_recover_v2.py
------------------------
Recovery success rate test — recover() → navigate(), no explore().

Goal
----
Measure recovery success rate when the object is at arbitrary positions.
No explore needed — recover() does the search circle from scratch.

Sequence per trial
------------------
  1. Scatter bottle anywhere in workspace (you don't need to record where)
  2. recover() — rises to scan height, runs 60 mm XY search circle
  3. If object found: navigate() → hover above it
  4. You judge: gripper centred above bottle? (Y/N)
  5. Log attempt. If failed, retry up to MAX_ATTEMPTS.

One CSV row per ATTEMPT.
  recovered = 1  → navigate() completed and you confirmed centred
  recovered = 0  → navigate() failed or you marked it misaligned

Analysis: group by trial_id, take last row or first recovered=1 per trial.

Target: 20 trials × up to 5 attempts = ≤100 rows.

Results → tests/results/recover_v2_results.csv

Usage
-----
    python -m tests.test_recover_v2
    python -m tests.test_recover_v2 --start 5   # resume from trial 5
"""

import argparse
import os
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import navigate  as nav_mod
import recover   as recover_mod
from tests.logger import TestLogger

MAX_TRIALS   = 20
MAX_ATTEMPTS = 5

RECOVER_V2_FIELDS = [
    "attempt_num",
    "detection_mode",
    "hover_x", "hover_y", "hover_z", "clearance_z",
    "recovered",   # 1 = success (navigate + aligned confirmed), 0 = failed
    "notes",
]


def run_recover_v2_test(start_trial: int = 1):
    signal.signal(signal.SIGINT, lambda *_: os._exit(1))

    print()
    print("=" * 60)
    print("  RECOVERY SUCCESS RATE TEST v2  (recover → navigate)")
    print("=" * 60)
    print(f"  {MAX_TRIALS} trials × up to {MAX_ATTEMPTS} attempts each")
    print()
    print("  Per trial:")
    print("    1. Scatter bottle somewhere in workspace → ENTER")
    print("    2. recover() — search circle re-finds object")
    print("    3. navigate() — hover above object")
    print("    4. Judge alignment → Y / N")
    print("    5. If N or failed → retry up to 5 attempts total")
    print()

    with TestLogger("recover_v2", extra_fields=RECOVER_V2_FIELDS) as log:
        for trial_num in range(start_trial, MAX_TRIALS + 1):
            print(f"\n{'═'*60}")
            print(f"  TRIAL {trial_num}/{MAX_TRIALS}   (Press Q at prompt to quit)")
            print(f"{'═'*60}")
            print()

            raw = input("  Scatter bottle, then press ENTER  (or Q to quit): ").strip().upper()
            if raw == 'Q':
                break

            recovered = False

            for attempt in range(1, MAX_ATTEMPTS + 1):
                print(f"\n  ── Attempt {attempt}/{MAX_ATTEMPTS} ──")

                # ── Recover: search circle ────────────────────────────────────
                print("\n  [1/2] recover() — search circle …")
                try:
                    recover_mod.main(clearance_z=0.0)   # rises to scan_z internally
                except Exception as e:
                    print(f"  [!] recover() raised: {e}")
                    log.write(
                        trial_id=trial_num, attempt_num=attempt,
                        detection_mode="ERROR",
                        hover_x=0, hover_y=0, hover_z=0, clearance_z=0,
                        recovered=0, notes=f"recover error: {e}"
                    )
                    if attempt == MAX_ATTEMPTS:
                        print("  [ABORT] Max attempts reached.")
                    continue

                # ── Navigate: hover ───────────────────────────────────────────
                print("\n  [2/2] navigate() — hovering …")
                try:
                    nav_result = nav_mod.main(autonomous=True, force_yolo=False)
                except Exception as e:
                    print(f"  [!] navigate() raised: {e}")
                    log.write(
                        trial_id=trial_num, attempt_num=attempt,
                        detection_mode="ERROR",
                        hover_x=0, hover_y=0, hover_z=0, clearance_z=0,
                        recovered=0, notes=f"navigate error: {e}"
                    )
                    if attempt == MAX_ATTEMPTS:
                        print("  [ABORT] Max attempts reached.")
                    continue

                if nav_result is None:
                    print("  [!] navigate() aborted.")
                    log.write(
                        trial_id=trial_num, attempt_num=attempt,
                        detection_mode="aborted",
                        hover_x=0, hover_y=0, hover_z=0, clearance_z=0,
                        recovered=0, notes="navigate aborted"
                    )
                    if attempt == MAX_ATTEMPTS:
                        print("  [ABORT] Max attempts reached.")
                    continue

                hover_pose     = nav_result["hover_pose"]
                clearance_z_v  = nav_result["clearance_z"]
                detection_mode = nav_result.get("detection_mode", "unknown")
                hx, hy, hz     = hover_pose[:3]

                print(f"\n  Hover pose: X={hx:.4f}  Y={hy:.4f}  Z={hz:.4f}")

                # ── Judge ─────────────────────────────────────────────────────
                raw = input("  Gripper centred above bottle? (Y / N): ").strip().upper()
                success = (raw == 'Y')
                notes   = input("  Notes (ENTER to skip): ").strip()

                log.write(
                    trial_id       = trial_num,
                    attempt_num    = attempt,
                    detection_mode = detection_mode,
                    hover_x        = round(hx, 5),
                    hover_y        = round(hy, 5),
                    hover_z        = round(hz, 5),
                    clearance_z    = round(clearance_z_v, 5),
                    recovered      = 1 if success else 0,
                    notes          = notes,
                )

                if success:
                    print(f"  ✓ RECOVERED on attempt {attempt} — logged.")
                    recovered = True
                    break
                else:
                    print(f"  ✗ Misaligned on attempt {attempt}.", end="")
                    if attempt < MAX_ATTEMPTS:
                        print(" Retrying …")
                    else:
                        print(f"\n  [FAIL] All {MAX_ATTEMPTS} attempts failed for trial {trial_num}.")

    print("\n  Recover v2 test complete.")
    print("  Results → tests/results/recover_v2_results.csv\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Recovery success rate test v2.")
    p.add_argument("--start", type=int, default=1, metavar="N",
                   help="Start from trial N (default: 1)")
    args = p.parse_args()
    run_recover_v2_test(start_trial=args.start)
