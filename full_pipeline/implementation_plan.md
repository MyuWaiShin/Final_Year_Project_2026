# Full Pipeline Implementation Plan

## Context

This pipeline implements **failure detection and recovery** for an autonomous pick-and-place system. The core principle is: the robot must find and attempt to pick the object regardless of its state — even if knocked over, flipped, or tag-obscured. The system only declares failure when the shared attempt budget is exhausted.

---

## Detection Hierarchy

Every stage that requires knowing where the object is follows this priority:

| State | ArUco | YOLO | Action |
|-------|-------|------|--------|
| Best case | ✓ | ✓ | Use ArUco for position |
| Degraded | ✗ | ✓ | Use YOLO + OAK-D depth for position |
| Abort | ✗ | ✗ | Object genuinely not found — abort |

Navigate re-detects on every attempt. It does not rely on a mode flag passed from a previous stage — it checks ArUco first, falls back to YOLO-only if not found.

---

## Full Cycle

```
scan pose → explore → navigate → grasp → verify → transit → release → scan pose
                          ↑
                       recover (shared budget: 5 total failures across all stages)
```

---

## Stage-by-Stage Plan

### Stage 1 — Explore

**Goal:** Confirm the object is on the table and roughly where it is.

**Behaviour:**
- Move to saved scan pose
- Sweep J0 across ±0.5 rad arc
- On every frame, run **both** ArUco detection and YOLO object detection
- Keep sweeping until the detected object/tag is within ±80px of the image horizontal centre
- Stop sweep once centred

**Detection outcome:**
- ArUco found → proceed
- YOLO only → proceed
- Neither after 2 sweeps → abort pipeline

**Notes:**
- Gripper always opened during explore
- No position returned — navigate re-detects live every attempt

---

### Stage 2 — Navigate

**Goal:** Position TCP directly above the object at hover height, ready to descend.

**Behaviour:**
- Check for ArUco tag live
  - Found → compute base-frame position via hand-eye transform, align gripper to tag X-axis
  - Not found → detect YOLO bounding box, use OAK-D depth to get 3D point, transform to base frame via hand-eye, use fixed wrist orientation (straight down)
- Move TCP to hover above object XY at fixed hover Z
- Centring correction: up to 3 moves until object is within 40px of image centre
- Derive `clearance_z = hover_z + 0.60m` — passed to all downstream stages

**Both paths converge to:** TCP hovering above object, `clearance_z` known.

---

### Stage 3 — Grasp

**Goal:** Descend onto object and close gripper, validate contact.

**Behaviour:**
- Descend 100mm below hover Z
- Close gripper via dashboard URP
- **Check — Width:** AI2 voltage → mm. ≤11mm = fully closed = missed

**Outcomes:**
- Width > 11mm → something gripped → proceed to verify
- Width ≤ 11mm → missed → recover

---

### Stage 4 — Verify

**Goal:** Confirm the object is in the gripper.

**Behaviour:**
- Rise straight up to `clearance_z`
- Run **YOLO26n classifier** (empty vs holding)
- Threshold: **90% confidence** required to pass

**Outcomes:**
- p_holding ≥ 0.90 → holding confirmed → transit
- p_holding < 0.90 → empty → recover (already at `clearance_z`)

**Notes:**
- No wrist tilt — classifier works reliably from `clearance_z` pointing straight down

---

### Stage 5 — Transit

**Goal:** Move to drop zone at clearance height with slip monitoring.

**Behaviour:**
- Move from current position to above drop zone, staying at `clearance_z`
- Two independent slip detection layers running during movement:

  **Layer 1 — URScript gripper thread** (`SLIP_DETECT = True/False` flag):
  - Single URScript with two threads: one executes `movel()` to drop zone, one loops close gripper command every 0.2s
  - If width changes between re-close cycles → object slipped → stop robot → recover

  **Layer 2 — YOLO classify (always on):**
  - Background Python thread grabs frames and runs YOLO26n classifier continuously
  - If classification flips from holding → empty → stop robot → recover

**Recovery from mid-transit:**
- Robot stops at current position (already at `clearance_z`)
- Runs 60mm search circle from current position
- Back to navigate

---

### Stage 6 — Release

**Goal:** Place the object at the drop zone.

**Behaviour:**
- Descend to drop height
- Open gripper via dashboard URP
- Rise back to `clearance_z`
- Return to scan pose

**Pipeline complete.**

---

### Recovery

**Goal:** Reset to a safe state and re-attempt from navigate.

**Behaviour:**
1. Open gripper
2. Rise to `clearance_z` (passed in — absolute position, safe to command even if already there)
3. Run 60mm radius XY search circle, looking for ArUco or YOLO detection on every frame
4. Stop circle early if either detector fires and object is centred
5. Re-enter pipeline from **Stage 2 (navigate)**

**Shared failure budget:** 5 total failures (grasp + verify + transit combined) before hard abort.
After abort, return to scan pose.

---

## Key Constants

| Constant | Value | File |
|----------|-------|------|
| Sweep arc | ±0.5 rad | explore.py |
| Centre tolerance (explore) | ±80 px | explore.py |
| Max sweeps | 2 | explore.py |
| Centre tolerance (navigate) | ±40 px | navigate.py |
| Hover orientation | rx=2.225 ry=2.170 rz=0.022 | navigate.py |
| Descend (grasp) | 100 mm | grasp.py |
| Width miss threshold | 11 mm | grasp.py |
| Clearance height | hover_z + 600 mm | navigate.py → all stages |
| Verify threshold | 0.90 | verify.py |
| Search circle radius | 60 mm | recover.py |
| Slip detect flag | SLIP_DETECT = True | transit.py |
| Gripper re-close interval | 0.2 s | transit.py |
| Max retries (shared) | 5 | main.py |
| ArUco tag ID | 13 | all stages |
| ArUco marker size | 36 mm | all stages |
| Robot IP | 192.168.8.102 | all stages |

---

## Implementation Checklist

- [ ] explore.py — dual detect (ArUco + YOLO), centre-lock, 2-sweep limit
- [ ] navigate.py — ArUco path + YOLO-only path, derive clearance_z
- [ ] grasp.py — 100mm descend, width check only
- [ ] verify.py — rise to clearance_z, YOLO26n classify at 90% threshold
- [ ] transit.py — movel + URScript gripper thread + YOLO classify background thread
- [ ] release.py — descend, open gripper, rise, return to scan pose
- [ ] recover.py — rise to clearance_z, 60mm search circle, dual detect
- [ ] main.py — shared 5-attempt budget, thread clearance_z through loop
