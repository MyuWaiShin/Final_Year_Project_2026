"""
main.py
-------
Orchestrator for the full pick-and-place pipeline.

Stages (work in progress)
--------------------------
1.  explore()   – Move to scan pose, sweep base joint, find ArUco tag
2.  navigate()  – Align TCP above detected tag at hover height
3.  grasp()     – (TODO) Lower, close gripper, lift
4.  verify()    – (TODO) CLIP post-grasp verification
5.  transit()   – (TODO) Move to drop zone with slip monitor
6.  release()   – (TODO) Open gripper, confirm drop
7.  recover()   – (TODO) Unified re-grasp recovery on failure

Usage
-----
    python main.py
"""

from explore  import main as explore
from navigate import main as navigate
from grasp    import main as grasp

# from verify   import main as verify     # TODO
# from transit  import main as transit    # TODO
# from release  import main as release    # TODO
# from recover  import main as recover    # TODO


def run_pipeline():
    print("\n" + "=" * 60)
    print("  FULL PIPELINE START")
    print("=" * 60 + "\n")

    # ── Stage 1: Explore ────────────────────────────────────────────────
    print("[STAGE 1] Explore – scanning for ArUco tag …")
    tag_pos = explore()
    if tag_pos is None:
        print("[ABORT] Tag not found during explore. Halting.")
        return
    print(f"[STAGE 1] Complete – tag at {[round(v, 4) for v in tag_pos]}\n")

    # ── Stage 2: Navigate ───────────────────────────────────────────────
    print("[STAGE 2] Navigate – hovering above tag …")
    hover_pos = navigate()
    if hover_pos is None:
        print("[ABORT] Navigation failed or cancelled. Halting.")
        return
    print(f"[STAGE 2] Complete – hovering at {[round(v, 4) for v in hover_pos]}\n")

    # ── Stage 3: Grasp ──────────────────────────────────────────────────
    print("[STAGE 3] Grasp – descend, close, check width + force …")
    grasp_result = grasp()
    if grasp_result["result"] == "missed":
        print("[ABORT] Gripper closed on air — object missed. Halting.")
        return
    print(f"[STAGE 3] Complete – holding  "
          f"width={grasp_result['width_mm']:.1f} mm  "
          f"force={'YES' if grasp_result['force'] else 'NO'}\n")

    # ── TODO: Stage 4+ ─────────────────────────────────────────────────
    print("[INFO] Further stages (verify, transit, release) not yet implemented.")
    print("=" * 60)
    print("  PIPELINE END")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
