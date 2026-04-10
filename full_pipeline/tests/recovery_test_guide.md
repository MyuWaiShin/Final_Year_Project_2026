hol# Recovery Testing Guide

## Experimental Design

| | |
|---|---|
| **Controls (fixed)** | Recovery mechanism (open gripper → rise to scan height → 60mm search circle → YOLO centre → J0 recenter), max 5 retry attempts per trial |
| **Variable** | Which failure stage triggered the recovery (Stage 1 / Stage 2 / Stage 3) |
| **Trials** | 30 total — 10 per failure type |
| **Success** | Recovery completes a successful re-grasp within 5 retry loop iterations |

---

## How to Run

```bash
cd full_pipeline
python -m tests.test_recover
```

Each trial:
1. Script tells you which failure to induce
2. You run the full pipeline up to the failure point
3. Recovery loop starts — up to 5 retry attempts (recover → navigate → grasp)
4. You judge each attempt: did it successfully re-grasp? (Y/N)
5. Script logs attempts taken, final outcome, notes

---

## What Gets Saved

Each trial writes one row to `tests/results/recover_results.csv`:

| Column | What it is |
|--------|------------|
| `trial_id` | Trial number (1–30) |
| `failure_type` | `stage1_miss` / `stage2_empty` / `stage3_slip` |
| `attempts_taken` | How many retry loops ran (1–5) |
| `recovered` | **1** = clean re-grasp achieved within 5 attempts, **0** = all 5 failed |
| `object_found` | **1** = recover() located the object in search circle, **0** = not found |
| `notes` | Free text |

**Analyse:**
```bash
python -m tests.analyse recover
```

---

## Recovery Loop — What Happens Each Attempt

```
recover() → open gripper → rise to scan height
         → 60mm XY search circle (YOLO detection)
         → stop circle when object found (±80px)
         → J0 recentering passes
         → navigate() → grasp()
         → if grasp succeeds → RECOVERED
         → else → attempt N+1 (up to 5)
```

---

## Trial Structure

### Section 1 — Stage 1 Failure: Gripper Miss (10 trials)

**How to induce:** After navigate() hovers, move the bottle 20–30 mm sideways before descent. Gripper closes on edge (width < 11 mm) → stage 1 fires → recovery triggered.

**What you watch:** Does the search circle locate the bottle? Does navigate() + grasp() succeed on retry?

---

### Section 2 — Stage 2 Failure: Visual Empty (10 trials)

**How to induce:** Get a clean grasp, but at clearance_z tilt/remove the bottle before verify() runs so YOLO sees empty → recovery triggered.

**What you watch:** Bottle is already dislodged. Does recovery re-find it and re-grasp?

---

### Section 3 — Stage 3 Failure: Transit Slip (10 trials)

**How to induce:** Clean grasp, then yank the bottle mid-transit. YOLO layer 2 detects empty → robot stops → recovery triggered.

**What you watch:** Robot stops mid-transit, recovery runs from that position. Is the search circle wide enough to find the bottle (which may have fallen some distance away)?

---

## Interpreting Results

| Metric | What to report |
|---|---|
| Recovery success rate | % of 30 trials where `recovered == 1` |
| Per-stage success rate | Recovery rate for stage1 / stage2 / stage3 separately |
| Average attempts taken | Mean `attempts_taken` across recovered trials |
| Object-found rate | % of trials where search circle found the object (even if grasp still failed) |

**Key distinction:**
- `object_found = 1, recovered = 0` → search circle worked but navigate/grasp failed repeatedly
- `object_found = 0, recovered = 0` → bottle not visible in 60mm circle (fell too far, occluded, etc.)

---

## Key Claim for Your Report

> "The recovery mechanism successfully re-grasped the object in X/30 trials (Y%) across all three failure types. Recovery was most reliable for Stage 1 failures (Z/10) and least reliable for Stage 3 slip events (W/10), where the object displacement exceeded the 60mm search radius in N cases."
