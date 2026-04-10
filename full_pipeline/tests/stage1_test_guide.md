# Stage 1 — Gripper IO Failure Detection Testing Guide

## Experimental Design

| | |
|---|---|
| **Controls (fixed)** | Robot, gripper, scan pose height, navigate result (hover pose locked by ArUco PnP before each trial) |
| **Variable** | Failure scenario type (how the miss is induced after the hover pose is locked) |
| **Trials** | 12 total — 6 × Scenario A, 6 × Scenario B |

| Scenario | Description |
|---|---|
| **A — Pose error / edge contact** | After navigate() locks the hover pose, nudge bottle ~20 mm sideways. Gripper edges the bottle on descent — force bit may fire but grasp is not clean. |
| **B — Complete air miss** | After navigate() locks the hover pose, slide bottle completely away. Gripper closes on air. |

> Both scenarios deliberately produce a **failed grasp**. The test measures whether stage 1 correctly identifies each as "missed".

---

## How to Run

```bash
cd full_pipeline
python -m tests.test_grasp
```

Each trial:
1. Script prints the scenario and trial number
2. `navigate()` runs → robot hovers above the bottle
3. You set up the failure condition (nudge or remove the bottle)
4. `grasp()` descends, closes gripper, reads sensors
5. Script shows system prediction, asks for notes → logs and continues

---

## What Gets Saved

Each trial writes one row to `tests/results/grasp_results.csv`:

| Column | What it is |
|--------|------------|
| `trial_id` | Trial number (1–12) |
| `scenario` | `A_pose_error` or `B_air_miss` |
| `hover_x/y/z` | TCP hover position after navigate() (metres, base frame) |
| `pick_z` | Z the gripper descended to (metres) |
| `width_mm` | Calibrated jaw width read after gripper closes (mm). Range 0–85 mm. Close to 0 = nothing grabbed, ~50 = bottle held. |
| `force_detected` | **1** = RG2 force-limit bit fired (contact confirmed), **0** = no contact signal |
| `system_prediction` | What stage 1 returned: `"holding"` or `"missed"` |
| `actual_grasped` | Always **0** in this test (every trial is a deliberate miss) |
| `correct` | **1** if system said `"missed"` (correct), **0** if it said `"holding"` (false positive) |
| `notes` | Free text |

**Analyse (confusion matrix + width histogram):**
```bash
python -m tests.analyse grasp
```


After the robot descends and closes the gripper, `grasp.py` reads two physical sensors:

| Check | Sensor | Logic |
|-------|--------|-------|
| **Width** | AI2 voltage → calibrated mm | `width ≤ 11 mm` → gripper fully closed → **MISSED** |
| **Force** | DI8 bit 17 (masterboard) | HIGH = RG2 force limit reached → **contact confirmed** |

Width is the **primary decision gate** (pass/fail). Force is logged but not fatal.

---

## What You Are Testing

1. **Does the width threshold (11 mm) correctly distinguish a miss from a hold?**
   - Too low → false negatives (real misses pass as holds)
   - Too high → false positives (real holds flagged as misses)

2. **Width sensor calibration accuracy** — does `calibrated_mm` match the actual jaw gap?

3. **Force bit reliability** — does it fire when it should?

---

## Equipment

- The target cylindrical bottle (with ArUco on top)
- A set of **known-width spacers** to test the width calibration curve (optional but useful):
  - e.g. drill bits or gauge blocks: 0 mm (nothing), 15 mm, 25 mm, 40 mm, 60 mm, 85 mm
- Your computer running `grasp.py` in standalone mode
- Robot with gripper ready, E-stop to hand

---

## Step-by-Step Protocol

### Part A — Width Calibration Verification (5 min)

**Goal:** Confirm the calibrated width reading matches reality.

1. Open `grasp.py` and run it standalone (robot must be at a safe pose):
   ```bash
   cd full_pipeline
   python grasp.py
   ```
2. At the prompt, don't press ENTER yet. Instead, manually set the gripper to a known width using the teach pendant or by holding a spacer between the jaws.
3. Note the printed `Gripper width: X.X mm` value.
4. Compare against your measured spacer width.
5. Repeat for 4–5 different widths across the range (0, ~15, ~40, ~65, ~85 mm).

**Record:** `spacer_mm`, `reported_mm`, `error_mm`.

**Acceptance:** Error should be ≤ 3 mm across the range. If systematic, adjust the calibration constants in `grasp.py`:
```python
slope  = (91.0 - 10.5) / (65.8 - 8.5)
offset = 10.5 - (8.5 * slope)
```

---

### Part B — Miss Detection (False Negative Test)

**Goal:** Confirm that a genuine miss (nothing grasped) is correctly detected.

**Setup:** Navigate the robot to hover above the bottle.

**Trial 1 — Deliberate miss (gripper closes on air):**
1. Move the bottle 80 mm sideways — robot hovers over empty table
2. Press ENTER in `grasp.py` to descend and close
3. Expected: `width ≤ 11 mm` → `MISSED` returned
4. ✓ = correct | ✗ = false negative (miss not detected)

**Trial 2 — Object detected correctly:**
1. Place bottle correctly under the gripper
2. Descend and close
3. Expected: `width > 11 mm` (bottle is between jaws) → `holding` returned
4. ✓ = correct | ✗ = false positive (miss wrongly flagged)

Repeat each 5 times. Note the width reading each time.

---

### Part C — False Positive Edge Cases

**Goal:** Find the minimum grasp that the width check can distinguish.

The bottle is 50 mm diameter — the gripper should close to approximately **50 mm** when holding it correctly. The threshold is **11 mm** so there's a large margin. But test:

| Scenario | Expected result | Why |
|----------|----------------|-----|
| Gripper clips the bottle edge (partial grip) | `holding` (width ~20–40 mm) | Marginal grasp — will likely fail verify |
| Gripper closes on bottle cap rim only | `holding` (width depends on cap) | False positive risk |
| Gripper snags the ArUco tag edge | `holding` (width very small, ~1–5 mm) | FALSE POSITIVE — tag is thin |

> **Key risk:** If the robot clips only the paper ArUco tag, width reads very small (like 2–4 mm) but that is less than 11 mm → correctly flagged as **missed**. Good.

---

### Part D — Force Bit Reliability

**Goal:** Check if the force signal is consistent.

Run 5 trials gripping the bottle normally. Record `force=YES/NO` for each.

Expected: `YES` on most trials (RG2 stalls when bottle is gripped firmly).

> **Note:** Force is NOT a decision gate — it's supplementary info only. If it's always `NO`, that's fine as long as width passes. Document the rate.

---



| Column | What to enter |
|--------|--------------|
| `object_desc` | "bottle", "empty air", "edge clip", etc. |
| `true_label` | `holding` or `missed` (what actually happened) |
| `width_mm` | reported calibrated width |
| `force_detected` | 1 or 0 |
| `predicted_label` | what `grasp.py` returned |
| `correct` | 1 if prediction matches true label |
| `notes` | anything notable |

---

## Interpreting Results

| Metric | Target |
|--------|--------|
| Miss detection rate | % of trials where `correct == 1` → target 100% |
| False positive rate | % of trials where system said `"holding"` but actual = miss → target 0% |
| Width when nothing grabbed | `width_mm` for Scenario B (air) — should be < 5 mm |
| Width on edge contact | `width_mm` for Scenario A — watch whether it crosses the 11 mm threshold |
| Force bit in edge contact | `force_detected` for Scenario A — does it fire even on a bad grasp? |

**If miss detection rate < 100%:** Raise `WIDTH_CLOSED_MM` threshold slightly (e.g., 11 → 13 mm).

**If false positive rate > 5%:** The calibration might be reading high — re-check the two-point calibration constants.

---

## Analyse

```bash
python -m tests.analyse grasp
```

Produces:
- **Width histogram** by true label (holding vs missed) — shows the two distributions and whether 11 mm cleanly separates them
- **Confusion matrix** — TPR / FPR at a glance

---

## Key Claim to Make in Your Report

> "Stage 1 (gripper IO) achieved a miss detection rate of X% with 0 false positives across N trials, using a jaw-width threshold of 11 mm. The width sensor calibration error was ≤ Y mm (RMSE)."
