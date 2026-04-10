"""
tests/test_recover.py
---------------------
Recovery test script matching the exact full pipeline flow (wrapper around `main.py`).

No pre-selected failures. The script runs the exact pipeline. You can induce
failures midway if you want. It detects stage failures naturally, recovers,
and if a failure occurred, asks you for the final placement success at the end.

Usage
-----
    python -m tests.test_recover --autonomous
"""

import argparse
import os
import signal
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from explore  import main as explore
from navigate import main as navigate
from grasp    import main as grasp
from verify   import main as verify, load_models
from transit  import main as transit
from release  import main as release
from recover  import main as recover

from tests.logger import TestLogger

MAX_RETRIES = 5

RECOVER_FIELDS = [
    "failure_type", "attempts_taken", "recovered", "object_found", "notes",
]


def run_recovery_test(autonomous: bool = False):
    signal.signal(signal.SIGINT, lambda *_: os._exit(1))

    print("\n" + "=" * 60)
    print("  RECOVERY LOGGING TEST (Full Pipeline Wrapper)")
    print("=" * 60)

    # ── Pre-load model ────────────────────────────────────────────────────────
    _models   = [None]
    def _load():
        try:
            _models[0] = load_models()
        except:
            pass
    threading.Thread(target=_load, daemon=True).start()

    with TestLogger("recovery", extra_fields=RECOVER_FIELDS) as log:
        trial = 0

        while True:
            trial += 1
            print(f"\n\n{'═'*60}")
            print(f"  TRIAL {trial}   (Press Q at prompt to quit)")
            print(f"{'═'*60}")

            ans = input("  Press ENTER to start autonomous pipeline for this trial (or Q to quit): ").strip().upper()
            if ans == "Q":
                break

            run_single_pipeline_trial(trial, log, _models)

    print("\n  Recovery test complete.")
    print("  Results -> tests/results/recover_results.csv\n")


def run_single_pipeline_trial(trial, log, _models):
    autonomous = True
    # ── Stage 1: Explore ──────────────────────────────────────────────────────
    print("\n[STAGE 1] Explore — scanning for object …")
    found = explore(autonomous=autonomous)
    if found is None:
        print("[ABORT] Object not found during explore. Halting trial.")
        return
    print("[STAGE 1] Complete — object found and centred.\n")

    clearance_z = None
    total_fails = 0

    first_fail_type = None
    object_found_in_recovery = 0

    while True:
        attempt = total_fails + 1
        print(f"{'─'*60}")
        print(f"  ATTEMPT {attempt}  (failures: {total_fails}/{MAX_RETRIES})")
        print(f"{'─'*60}\n")

        # ── Stage 2: Navigate ─────────────────────────────────────────────────
        print("[STAGE 2] Navigate — hovering above object …")
        nav_result = navigate(autonomous=autonomous)
        if nav_result is None:
            print("[ABORT] Navigation cancelled. Halting trial.")
            return

        clearance_z = nav_result["clearance_z"]
        print("[STAGE 2] Complete.\n")

        # ── Stage 3: Grasp ────────────────────────────────────────────────────
        print("[STAGE 3] Grasp — descend, close, check width …")
        is_recovery  = total_fails > 0
        grasp_result = grasp(close_only=is_recovery, autonomous=autonomous)

        if grasp_result["result"] == "missed":
            if first_fail_type is None: first_fail_type = "stage1_miss"
            total_fails += 1
            print(f"[FAIL] Grasp missed (total={total_fails}/{MAX_RETRIES})")
            if total_fails >= MAX_RETRIES: break
            print("  Recovering …\n")
            obj_found = recover(clearance_z)
            object_found_in_recovery = 1 if obj_found else 0
            continue

        print("[STAGE 3] Complete.\n")

        # ── Stage 4: Verify ───────────────────────────────────────────────────
        print("[STAGE 4] Verify — rise to clearance_z, classify …")
        verify_result = verify(clearance_z=clearance_z, models=_models[0])

        if verify_result["result"] == "empty":
            if first_fail_type is None: first_fail_type = "stage2_empty"
            total_fails += 1
            print(f"[FAIL] Verify empty (total={total_fails}/{MAX_RETRIES})")
            if total_fails >= MAX_RETRIES: break
            print("  Recovering …\n")
            obj_found = recover(clearance_z)
            object_found_in_recovery = 1 if obj_found else 0
            continue

        print("[STAGE 4] Complete.\n")

        # ── Stage 5: Transit ──────────────────────────────────────────────────
        print("[STAGE 5] Transit — moving to drop zone …")
        transit_result = transit(
            clearance_z=clearance_z,
            models={"cls": _models[0]["yolo"]} if _models[0] else None
        )

        if transit_result["result"] != "arrived":
            if first_fail_type is None: first_fail_type = "stage3_slip"
            total_fails += 1
            print(f"[FAIL] Transit failed (total={total_fails}/{MAX_RETRIES})")
            if total_fails >= MAX_RETRIES: break
            print("  Recovering …\n")
            obj_found = recover(clearance_z)
            object_found_in_recovery = 1 if obj_found else 0
            continue

        print("[STAGE 5] Complete.\n")

        # ── Stage 6: Release ──────────────────────────────────────────────────
        print("[STAGE 6] Release — place object, return to scan pose …")
        release(clearance_z=clearance_z)
        print("[STAGE 6] Complete.\n")
        break   # Pipeline fully succeeded!

    # ── Logging ───────────────────────────────────────────────────────────────
    # If a failure EVER occurred during this trial, we prompt for success/failure and log.
    if first_fail_type is not None:
        print("\n" + "*" * 50)
        print(f"  A failure occurred in this trial ({first_fail_type}).")
        print("*" * 50)
        while True:
            ans = input("  Did the object successfully PLACE at the end? (Y/N): ").strip().upper()
            if ans in ("Y", "N"):
                recovered_success = 1 if ans == "Y" else 0
                break

        notes = input("  Notes (ENTER to skip): ").strip()

        log.write(
            failure_type   = first_fail_type,
            attempts_taken = total_fails,      # how many failures it hit
            recovered      = recovered_success,
            object_found   = object_found_in_recovery, # result of latest search circle
            notes          = notes,
        )
        print(f"  [LOGGER] Logged recovery data for trial {trial}.")
    else:
        print("\n  [LOGGER] No failures occurred this trial. Nothing to log for recovery.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--autonomous", action="store_true")
    args = p.parse_args()
    
    try:
        run_recovery_test(autonomous=args.autonomous)
    except RuntimeError as e:
        if "PROTECTIVE STOP" in str(e):
            print("\n" + "!" * 60)
            print("  PROTECTIVE STOP DETECTED")
            print("  Re-enable on pendant, reposition to safe pose, restart.")
            print("!" * 60 + "\n")
        else:
            raise
