# Stage 2 — YOLO Visual Classifier (verify.py) Testing Guide

## Experimental Design

| | |
|---|---|
| **Controls (fixed)** | Robot, gripper, clearance_z height (same as real pipeline), YOLO model (`yolo26n_cls_V1`), confidence threshold (0.90) |
| **Variable** | Gripper state — what is actually between the jaws when the classifier runs |
| **Trials** | 20 total — 10 × holding, 5 × empty (air miss), 5 × empty (slip) |

| Condition | What is in the gripper | Expected prediction |
|---|---|---|
| **Holding** | Bottle correctly grasped | `holding` |
| **Empty — air miss** | Gripper closed on air, reopened | `empty` |
| **Empty — slip** | Bottle knocked out of gripper before verify runs | `empty` |

---

## How to Run

```bash
cd full_pipeline
python -m tests.test_verify
```

Each trial:
1. Script tells you the condition to set up
2. navigate() + grasp() run to get the gripper into the correct state
3. verify() runs at clearance_z with the YOLO classifier
4. Script shows confidence score, prediction, and whether it was correct
5. Log written automatically → next trial

```bash
# If you already know clearance_z and want to skip navigate+grasp:
python -m tests.test_verify --clearance-z -0.050
```

---

## How to Run

```bash
cd full_pipeline
python -m tests.test_verify
```

Each trial:
1. Script tells you the condition to set up (hold bottle or empty gripper)
2. navigate() + grasp() run to reach clearance_z with gripper in the correct state
3. `verify()` runs the YOLO classifier from clearance_z
4. Script logs prediction, confidence, and actual state → continues to next trial

---

## What Gets Saved

Each trial writes one row to `tests/results/verify_results.csv`:

| Column | What it is |
|--------|------------|
| `trial_id` | Trial number (1–12) |
| `condition` | `holding` or `empty` — what was actually in the gripper |
| `yolo_conf` | Confidence score output for the `holding` class (0.0–1.0). Above 0.90 = system predicts holding. |
| `result` | What verify() returned: `"holding"` or `"empty"` |
| `correct` | **1** if `result == condition`, **0** if wrong |
| `notes` | Free text |

**Analyse (confidence histogram + confusion matrix):**
```bash
python -m tests.analyse verify
```

---

## Step-by-Step Protocol

### Trials 1–6: "Holding" Condition

The goal is to have the robot at `clearance_z` with the bottle correctly grasped — exactly the state verify.py sees in the real pipeline.

1. Run navigate() + grasp() normally until you have a clean hold
2. **Do not run verify.py yet** — just confirm the bottle is gripped by checking `width_mm > 11 mm`
3. The robot is now at `clearance_z` holding the bottle
4. Run the test script — verify() classifies the gripper view
5. Log: `condition = holding`, record `yolo_conf` and `result`

> **Tip:** You can trigger navigate+grasp manually by running the real pipeline but pausing before verify, or use the integrated test script.

---

### Trials 7–12: "Empty" Condition

The goal is to have the robot at `clearance_z` with the gripper open and empty.

1. Move robot to `clearance_z` with gripper fully open (after a missed grasp or manually)
2. Run the test script — verify() classifies the empty gripper view
3. Log: `condition = empty`, record `yolo_conf` and `result`

**Two sub-cases to cover (2–3 trials each):**

| Sub-case | How to set up |
|---|---|
| **Empty — clean miss** | Gripper closed on air, reopened. Camera sees only the table below. |
| **Empty — partial hold** | Gripper barely touched bottle, object slipped out before clearance_z. Camera may see the bottle nearby. |

---

## What the Confidence Score Means

`yolo_conf` is the model's estimated probability that the gripper is `holding`:

- `yolo_conf >= 0.90` → system predicts `"holding"`
- `yolo_conf < 0.90`  → system predicts `"empty"`

You will see this in your results:

```
yolo_conf = 0.97  →  "holding"  (confident hold)
yolo_conf = 0.45  →  "empty"    (clearly empty)
yolo_conf = 0.88  →  "empty"    (just below threshold — borderline)
```

Borderline cases (0.80–0.95) are most interesting to look at.

---

## Interpreting Results

| What to look at | What it tells you |
|---|---|
| `correct` column | Overall accuracy: sum(correct) / 12 |
| `yolo_conf` for holding trials | Should cluster near 1.0 |
| `yolo_conf` for empty trials | Should cluster near 0.0 |
| Any `correct == 0` rows | These are misclassifications — read the notes |

**Confusion matrix from analyse.py:**

```
                  Predicted
                holding | empty
Actual holding |   TP   |  FN   ← missed a real hold
       empty   |   FP   |  TN   ← false positive (most dangerous)
```

- **False Positive (FP):** System says holding when empty → robot will try to transit with nothing
- **False Negative (FN):** System says empty when holding → wastes a good grasp (triggers recovery)

FP is the more dangerous failure — it leads to a transit with no object.

---

## Tuning

If accuracy is poor:

| Problem | Fix |
|---|---|
| Too many FP (empty → holding) | Raise threshold from 0.90 → 0.95 in `verify.py` |
| Too many FN (holding → empty) | Lower threshold from 0.90 → 0.85 — or retrain with more holding samples |
| Confidence clusters overlap (both near 0.5) | Model needs retraining on more representative data |

---

## Key Claim for Your Report

> "Stage 2 (YOLO visual classifier) achieved X% accuracy across 12 trials at clearance_z, with a mean holding confidence of Y ± Z and zero false positives / N false positives. The 0.90 threshold correctly separated holding from empty in X/12 cases."
