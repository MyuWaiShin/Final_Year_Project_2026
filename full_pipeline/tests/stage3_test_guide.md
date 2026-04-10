# Stage 3 — YOLO Live Slip Detection (Transit) Testing Guide

## Experimental Design

| | |
|---|---|
| **Controls (fixed)** | `SLIP_DETECT = False` (width Layer 1 disabled), transit speed (0.06 m/s), YOLO26n classifier, threshold (0.90), consecutive frames to trigger = 3, drop zone position |
| **Variable** | When during transit the bottle is yanked (early / mid / late) |
| **Trials** | 15 total — all deliberate slip events, `SLIP_DETECT = False` throughout |

> Every trial is a **deliberate miss** — you yank the bottle out while the robot is moving. The system should detect `empty` via YOLO and stop. This measures whether Layer 2 reliably catches real slips.

---

## How to Run

```bash
cd full_pipeline
python -m tests.test_transit
```

Make sure `transit.py` has:
```python
SLIP_DETECT = False   # keep off — only testing YOLO Layer 2
```

Each trial:
1. navigate() + grasp() → clean hold confirmed
2. Transit begins toward drop zone
3. You yank the bottle out (hard and fast)
4. System should detect `empty` → robot stops
5. Script logs detection result, YOLO confidence, stop position, notes

---

## What Gets Saved

Each trial writes one row to `tests/results/transit_results.csv`:

| Column | What it is |
|--------|------------|
| `trial_id` | Trial number (1–15) |
| `yank_timing` | When you pulled: `early` / `mid` / `late` (you enter this) |
| `system_result` | What `transit()` returned: `empty` (detected) or `arrived` (missed — failure) |
| `detected` | **1** if system returned `empty`, **0** if it returned `arrived` (slip missed) |
| `correct` | Same as `detected` — every trial is a deliberate slip so correct = detected |
| `tcp_x_at_stop` | X where robot stopped (metres) |
| `tcp_y_at_stop` | Y where robot stopped (metres) |
| `tcp_z_at_stop` | Z where robot stopped (metres) |
| `notes` | Free text (e.g. how hard the yank was) |

**Analyse:**
```bash
python -m tests.analyse transit
```

---

## Protocol

### Before each trial
1. Run navigate() + grasp() normally — confirm clean hold (check width > 11 mm on console)
2. Transit begins automatically

### During transit
- **Yank the bottle** — pull it out of the gripper firmly and cleanly
- Vary **when** you yank:
  - **Early**: within the first 20% of the journey
  - **Mid**: roughly halfway to the drop zone
  - **Late**: in the final 20% before the drop zone

5 trials per timing group (5 × early + 5 × mid + 5 × late = 15 total).

### After each trial
- Note whether the robot stopped mid-transit (`empty` detected) or arrived at the drop zone (slip missed)
- Enter the timing and any notes when prompted

---

## What You're Looking For

| Outcome | What it means |
|---|---|
| Robot stops mid-transit, `result = empty` | ✓ Layer 2 working — slip detected |
| Robot arrives at drop zone, `result = arrived` | ✗ Slip missed — YOLO didn't catch it in time |

**Key question:** Do late yanks (close to drop zone) get caught or does the robot arrive before Layer 2 can react?

Reaction requires 3 consecutive YOLO `empty` frames. At ~10 fps that's ~300 ms. At transit speed 0.06 m/s that = **~18 mm of travel** before stopping. If the drop zone is very close when yank happens, the robot may arrive before detection fires.

---

## Interpreting Results

| Metric | Target |
|---|---|
| Overall detection rate | % of 15 trials where `detected == 1` — target: high as possible |
| Early yank detection | Should be 5/5 — plenty of time to react |
| Late yank detection | Will be lower — robot may arrive first |
| False alarm rate | Not applicable here (no clean transit trials) — test Stage 2 for that |

---

## Key Claim for Your Report

> "Stage 3 (YOLO live slip detection during transit) successfully detected X/15 simulated slip events. Detection rate was Y% for early-transit yanks and Z% for late-transit yanks, reflecting the ~300 ms reaction latency inherent in the 3-frame consecutive detection window."
