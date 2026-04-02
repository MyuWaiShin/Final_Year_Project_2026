"""
main.py
-------
Orchestrator for the full pick-and-place pipeline.

Stages
------
1.  explore()   – Scan pose, J0 sweep, ArUco+YOLO detect, centre-lock  [once]
2.  navigate()  – Hover above object (ArUco or YOLO+depth), derive clearance_z ┐
3.  grasp()     – Descend, close gripper, width check                          │
4.  verify()    – Rise to clearance_z, YOLO26n classify                        ├ retry loop
5.  transit()   – Move to drop zone at clearance_z with slip detection         │
6.  release()   – Descend, open gripper, rise, return to scan pose             │
    recover()   – Open gripper, rise to clearance_z, 60mm search circle       ┘

Shared failure budget: MAX_RETRIES across grasp + verify + transit failures.

Usage
-----
    python main.py                 # debug mode (prompts)
    python main.py --autonomous    # fully autonomous
"""

import argparse
import threading

from explore  import main as explore
from navigate import main as navigate
from grasp    import main as grasp
from verify   import main as verify, load_models
from transit  import main as transit
from release  import main as release
from recover  import main as recover

MAX_RETRIES = 5   # shared failure budget across all stages


def run_pipeline(autonomous: bool = False):
    print("\n" + "=" * 60)
    print("  FULL PIPELINE START")
    print("=" * 60 + "\n")

    # ── Pre-load classify model in background ─────────────────────────────────
    _models   = [None]
    _model_err = [None]

    def _load():
        try:
            _models[0] = load_models()
            print("[INFO] Classify model pre-loaded.")
        except Exception as e:
            _model_err[0] = e
            print(f"[WARN] Model pre-load failed: {e} — will load on demand.")

    threading.Thread(target=_load, daemon=True).start()
    print("[INFO] Pre-loading classify model in background …\n")

    # ── Stage 1: Explore (runs once) ──────────────────────────────────────────
    print("[STAGE 1] Explore — scanning for object …")
    found = explore(autonomous=autonomous)
    if found is None:
        print("[ABORT] Object not found during explore. Halting.")
        return
    print("[STAGE 1] Complete — object found and centred.\n")

    # ── Retry loop: Navigate → Grasp → Verify → Transit ──────────────────────
    clearance_z  = None
    total_fails  = 0
    grasp_fails  = 0
    verify_fails = 0
    transit_fails = 0

    while True:
        attempt = total_fails + 1
        print(f"{'─'*60}")
        print(f"  ATTEMPT {attempt}  "
              f"(failures: {total_fails}/{MAX_RETRIES}  "
              f"grasp={grasp_fails}  verify={verify_fails}  transit={transit_fails})")
        print(f"{'─'*60}\n")

        # ── Stage 2: Navigate ─────────────────────────────────────────────────
        print("[STAGE 2] Navigate — hovering above object …")
        nav_result = navigate(autonomous=autonomous)
        if nav_result is None:
            print("[ABORT] Navigation cancelled or failed. Halting.")
            return

        hover_pose      = nav_result["hover_pose"]
        clearance_z     = nav_result["clearance_z"]
        detection_mode  = nav_result.get("detection_mode", "aruco")
        print(f"[STAGE 2] Complete — hover at {[round(v, 4) for v in hover_pose[:3]]}"
              f"  clearance_z={clearance_z:.4f}  mode={detection_mode}\n")

        # ── Stage 3: Grasp ────────────────────────────────────────────────────
        print("[STAGE 3] Grasp — descend, close, check width …")
        is_recovery  = total_fails > 0
        grasp_result = grasp(close_only=is_recovery, autonomous=autonomous)

        if grasp_result["result"] == "missed":
            grasp_fails  += 1
            total_fails  += 1
            print(f"[FAIL] Grasp missed  (grasp={grasp_fails}  total={total_fails}/{MAX_RETRIES})")
            if total_fails >= MAX_RETRIES:
                print("[ABORT] Max retries reached. Halting.")
                return
            print("  Recovering …\n")
            recover(clearance_z)
            continue

        print(f"[STAGE 3] Complete — width={grasp_result['width_mm']:.1f} mm  "
              f"force={'YES' if grasp_result.get('force') else 'NO'}\n")

        # ── Stage 4: Verify ───────────────────────────────────────────────────
        print("[STAGE 4] Verify — rise to clearance_z, classify …")
        verify_result = verify(clearance_z=clearance_z, models=_models[0])

        if verify_result["result"] == "empty":
            verify_fails += 1
            total_fails  += 1
            print(f"[FAIL] Verify empty  YOLO={verify_result['yolo_conf']*100:.1f}%  "
                  f"(verify={verify_fails}  total={total_fails}/{MAX_RETRIES})")
            if total_fails >= MAX_RETRIES:
                print("[ABORT] Max retries reached. Halting.")
                return
            print("  Recovering …\n")
            recover(clearance_z)
            continue

        print(f"[STAGE 4] Complete — holding confirmed  "
              f"YOLO={verify_result['yolo_conf']*100:.1f}%\n")

        # ── Stage 5: Transit ──────────────────────────────────────────────────
        print("[STAGE 5] Transit — moving to drop zone …")
        transit_result = transit(
            clearance_z=clearance_z,
            models={"cls": _models[0]["yolo"]} if _models[0] else None
        )

        if transit_result["result"] != "arrived":
            transit_fails += 1
            total_fails   += 1
            reason = transit_result["result"]
            layer  = transit_result.get("layer", "?")
            print(f"[FAIL] Transit {reason} (layer {layer})  "
                  f"(transit={transit_fails}  total={total_fails}/{MAX_RETRIES})")
            if total_fails >= MAX_RETRIES:
                print("[ABORT] Max retries reached. Halting.")
                return
            print("  Recovering …\n")
            recover(clearance_z)
            continue

        print("[STAGE 5] Complete — at drop zone.\n")

        # ── Stage 6: Release ──────────────────────────────────────────────────
        print("[STAGE 6] Release — place object, return to scan pose …")
        release(clearance_z=clearance_z)
        print("[STAGE 6] Complete.\n")
        break   # pipeline success

    print("=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full pick-and-place pipeline.")
    parser.add_argument("--autonomous", action="store_true",
                        help="Run fully autonomously — no prompts.")
    args = parser.parse_args()

    mode_label = "AUTONOMOUS" if args.autonomous else "DEBUG"
    print(f"[INFO] Mode: {mode_label}")

    try:
        run_pipeline(autonomous=args.autonomous)
    except RuntimeError as e:
        if "PROTECTIVE STOP" in str(e):
            print("\n" + "!" * 60)
            print("  PROTECTIVE STOP DETECTED")
            print("  Re-enable on pendant, reposition to safe pose, restart.")
            print("!" * 60 + "\n")
        else:
            raise
