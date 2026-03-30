# Slip Detection Debug Log — UR10 + RG2 + 14_pick_clip.py
**Date:** 2026-03-25  
**Robot:** UR10 @ 192.168.8.102  
**Script:** `scripts/14_pick_clip.py`

---

## Architecture Confirmed by Test (`15_test_grip_urscript.py`)

| Question | Result |
|---|---|
| `rg_grip()` available via port 30002 URScript? | ❌ NO — URCap restriction, only works in dashboard URP interpreter |
| Dashboard loop URP survives a socket `movej()`? | ✅ YES — separate interpreter slots, no conflict |

**Conclusion:**  
- Gripper re-close loop **must** use the dashboard URP (`close_gripper_timed.urp`)  
- `rg_grip()` cannot be called from raw URScript sent to port 30002  
- Dashboard loop URP and socket arm motion coexist safely

---

## Bug 1 — Loop URP Killed Before Dump/Place Transit

### Symptom
After slip detected during transit → uncertain → moves to dump, gripper not re-closing during dump transit.

### Root Cause
```python
# WRONG — called right after movej returned, before any post-slip logic
stop_mon.set()
dashboard_cmd(robot.ip, "stop")   # ← killed loop URP HERE
_last_urp[0] = None
time.sleep(0.1)
# then moved to dump with NO loop URP running
```

### Fix
Removed the early `dashboard_cmd("stop")`. The loop URP now stays alive through all transit (pick→place, or stop→dump on slip). It is only killed by `play_urp(URP_OPEN)` which internally calls `dashboard_cmd("stop")` before switching to the open program.

---

## Bug 2 — `movej_with_grip_loop` Approach Failed

### Symptom
Replaced dashboard loop URP with a multi-line URScript program embedding a `thread` that calls `rg_grip(40, 0)` — gripper still didn't re-close.

### Root Cause
`rg_grip()` is a URCap function registered only in the dashboard/URP interpreter. It is **not** available in the secondary client (port 30002) URScript context. The URScript program compiled fine but `rg_grip()` was a no-op.

### Fix
Reverted to dashboard loop URP. Confirmed by `15_test_grip_urscript.py` Q1/Q2 = NO.

---

## Bug 3 — Loop URP Not Closing During Transit (Timing)

### Symptom
After our fix, loop URP still didn't re-close during transit even though Q3 test showed it should work.

### Root Cause
In the test script there was a `time.sleep(0.1)` between `dashboard("play")` and the `movej`. In the pipeline code, `movej_pose` was called almost immediately after `dashboard("play")`. The URP may not have executed its first `rg_grip` cycle before the movej command started contending for robot resources.

### Fix
Added `time.sleep(0.2)` after `dashboard_cmd("play")` before the `movej_pose` call, giving the loop URP time to execute at least one close cycle before transit begins.

---

## Bug 4 — IO Missed-Grasp Logic Too Strict

### Symptom
```
Width: 36.8 mm  |  Contact (DI8): False  →  MISSED GRASP
```
Width was clearly greater than 11mm (object between fingers) but contact signal was False.

### Root Cause
Original logic: `missed = (width < 11mm) OR (not contact)` — required BOTH width > 11mm AND contact to proceed. This is too strict; contact may not trigger if gripper hasn't reached force limit yet.

### Fix
```python
# Proceed if: width > 11mm  (object spreading fingers — contact optional)
# Missed if:  width <= 11mm (fully closed, OR self-contact at <=11mm)
missed = width <= WIDTH_CLOSED_MM
```

---

## Bug 5 — Unicode Characters in OpenCV Status Labels

### Symptom
```
Moving to pick pos ???
Descending ???
```
Status text showed `???` instead of `...` or `—`.

### Root Cause
Python source had Unicode ellipsis (`…`) and em-dash (`—`) in `STAGE_INFO` strings. OpenCV's `putText()` with `FONT_HERSHEY_*` only renders ASCII — non-ASCII becomes `?`.

### Fix
Replaced all Unicode characters with ASCII equivalents: `…` → `...`, `—` → `-`.

---

## Architecture Summary

### Working Configuration

| Layer | Method | Notes |
|---|---|---|
| Arm motion | URScript via port 30002 (persistent socket) | `movej()`, `movel()`, `stopl()` |
| Gripper open/close (one-shot) | Dashboard URP (port 29999) | `open_gripper.urp`, `close_gripper.urp` |
| Gripper re-close loop (slip) | Dashboard URP (port 29999) | `close_gripper_timed.urp` — loops, survives movej |
| IO sensors | Secondary state stream port 30002 | AI2 → width mm, DI8 bit 17 → contact |
| Slip monitor | Python thread, polls AI2 width every 80ms | Sets `slip_evt` → `stopl(0.5)` |

### Decision Tree

```
Close gripper
    ↓
IO Check (DI8 + AI2):
    width > 11mm?  → YES → lift
                   → NO  → open + recovery → scan

CLIP Check:
    Holding (≥75%) → start loop URP → transfer to place → place → scan
    Uncertain      → dump → scan
    Empty          → scan

During transfer:
    Slip monitor drops > 8mm? → stopl → re-verify CLIP
        Holding   → continue to place
        Uncertain → dump (loop URP still active) → scan
        Empty     → open gripper → scan
```
