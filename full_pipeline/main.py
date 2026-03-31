"""
main.py
-------
Orchestrator for the full pick-and-place pipeline.

Stages
------
1.  explore()   – Move to scan pose, sweep base joint, find ArUco tag   [runs once]
2.  navigate()  – Align TCP above detected tag at hover height           ┐
3.  grasp()     – Descend, close gripper, check width + force           ├ retry loop
4.  verify()    – Lift, tilt wrist, YOLO + CLIP dual check              ┘
    recover()   – Open gripper, rise 40 cm above hover, re-run 2-4
5.  transit()   – (TODO) Move to drop zone with slip monitor
6.  release()   – (TODO) Open gripper, confirm drop

Usage
-----
    python main.py
"""

import argparse
import threading

from explore  import main as explore
from navigate import main as navigate
from grasp    import main as grasp
from verify   import main as verify, load_models
from recover  import main as recover

# from transit  import main as transit    # TODO
# from release  import main as release    # TODO

MAX_GRASP_RETRIES  = 3   # max grasp failures before aborting
MAX_VERIFY_RETRIES = 3   # max verify failures before aborting


def run_pipeline(autonomous: bool = False):
    print("\n" + "=" * 60)
    print("  FULL PIPELINE START")
    print("=" * 60 + "\n")

    # ── Pre-load CLIP + YOLO in background while explore/navigate run ────
    _verify_models  = [None]
    _model_err      = [None]

    def _load():
        try:
            _verify_models[0] = load_models()
            print("[INFO] Verify models pre-loaded.")
        except Exception as e:
            _model_err[0] = e
            print(f"[WARN] Model pre-load failed: {e} — verify will load on demand.")

    threading.Thread(target=_load, daemon=True).start()
    print("[INFO] Pre-loading verify models in background …\n")

    # ── Stage 1: Explore (runs once) ────────────────────────────────────
    print("[STAGE 1] Explore – scanning for ArUco tag …")
    tag_pos = explore(autonomous=autonomous)
    if tag_pos is None:
        print("[ABORT] Tag not found during explore. Halting.")
        return
    print(f"[STAGE 1] Complete – tag at {[round(v, 4) for v in tag_pos[:3]]}\n")

    # ── Retry loop: Navigate → Grasp → Verify ───────────────────────────
    hover_pos     = None
    grasp_fails   = 0
    verify_fails  = 0

    while True:
        is_recovery = (grasp_fails + verify_fails) > 0
        attempt     = grasp_fails + verify_fails + 1
        print(f"{'─'*60}")
        print(f"  ATTEMPT {attempt}  "
              f"(grasp failures: {grasp_fails}/{MAX_GRASP_RETRIES}  "
              f"verify failures: {verify_fails}/{MAX_VERIFY_RETRIES})")
        print(f"{'─'*60}\n")

        # ── Stage 2: Navigate ──────────────────────────────────────────
        print("[STAGE 2] Navigate – hovering above tag …")
        hover_pos = navigate(autonomous=autonomous)
        if hover_pos is None:
            print("[ABORT] Navigation cancelled or failed. Halting.")
            return
        print(f"[STAGE 2] Complete – hover at {[round(v, 4) for v in hover_pos[:3]]}\n")

        # ── Stage 3: Grasp ─────────────────────────────────────────────
        print("[STAGE 3] Grasp – close, check width + force …")
        # First-ever attempt: descend onto the object.
        # Recovery attempts: navigate already repositioned TCP — just close.
        grasp_result = grasp(close_only=is_recovery, autonomous=autonomous)

        if grasp_result["result"] == "missed":
            grasp_fails += 1
            print(f"[FAIL] Grasp missed ({grasp_fails}/{MAX_GRASP_RETRIES}).")
            if grasp_fails >= MAX_GRASP_RETRIES:
                print("[ABORT] Max grasp retries reached. Pipeline halted.")
                return
            print("  Recovering — rising to clearance height …\n")
            recover(hover_pos)
            continue

        print(f"[STAGE 3] Complete – holding  "
              f"width={grasp_result['width_mm']:.1f} mm  "
              f"force={'YES' if grasp_result['force'] else 'NO'}\n")

        # ── Stage 4: Verify ────────────────────────────────────────────
        print("[STAGE 4] Verify – lift, tilt wrist, YOLO + CLIP check …")
        verify_result = verify(models=_verify_models[0])

        if verify_result["result"] == "empty":
            verify_fails += 1
            print(f"[FAIL] Verify failed ({verify_fails}/{MAX_VERIFY_RETRIES})  "
                  f"score={verify_result['score']:.3f}  "
                  f"YOLO={verify_result['yolo_conf']:.2f}  "
                  f"CLIP={verify_result['clip_conf']:.2f}.")
            if verify_fails >= MAX_VERIFY_RETRIES:
                print("[ABORT] Max verify retries reached. Pipeline halted.")
                return
            print("  Recovering — opening gripper, rising to clearance height …\n")
            recover(hover_pos)
            continue

        # ── Success ────────────────────────────────────────────────────
        print(f"[STAGE 4] Complete – holding confirmed  "
              f"score={verify_result['score']:.3f}  "
              f"YOLO={verify_result['yolo_conf']:.2f}  "
              f"CLIP={verify_result['clip_conf']:.2f}\n")
        break   # EXIT RETRY LOOP — grasp confirmed

    # ── TODO: Stage 5+ ──────────────────────────────────────────────────
    print("[INFO] Further stages (transit, release) not yet implemented.")
    print("=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full pick-and-place pipeline.")
    parser.add_argument(
        "--autonomous", action="store_true",
        help="Run fully autonomously — no SPACE/YES/ENTER prompts."
    )
    args = parser.parse_args()

    mode_label = "AUTONOMOUS" if args.autonomous else "DEBUG"
    print(f"[INFO] Mode: {mode_label}")

    try:
        run_pipeline(autonomous=args.autonomous)
    except RuntimeError as e:
        if "PROTECTIVE STOP" in str(e):
            print("\n" + "!" * 60)
            print("  PROTECTIVE STOP DETECTED")
            print("  Re-enable the robot on the pendant and reposition")
            print("  it to a safe pose, then restart the pipeline.")
            print("!" * 60 + "\n")
        else:
            raise

