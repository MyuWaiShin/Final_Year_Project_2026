# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a final year robotics project implementing a 7-stage autonomous pick-and-place pipeline using:
- **Robot**: Universal Robots UR10 with OnRobot RG2 gripper
- **Vision**: OAK-D Lite camera (DepthAI) + YOLOv8n/YOLO26n object detection + ArUco pose estimation + OpenAI CLIP post-grasp verification
- **Auto-annotation**: Grounding DINO (autodistill) for training data
- **Platform**: Windows 10/11, Python 3.11.9 only (not 3.13; not WSL)

## Environment Setup

Python 3.11.9 is required (not 3.13). Use separate virtual environments for different subsystems:

```bash
# Perception / camera / inference environment
pip install -r Perception/requirements.txt        # depthai 2.28.0, onnxruntime, ultralytics

# Data preparation environment
pip install -r "Object Detection Pipeline/requirements_dataprep.txt"

# Auto-annotation environment (requires CUDA 12.1 for GPU)
pip install -r "Object Detection Pipeline/requirements_annotation.txt"
```

See `SETUP.md` for the full environment setup sequence and troubleshooting table.

## Common Commands

### Training
```bash
# Train all YOLO variants
python Train/train.py --train all

# Train a specific model
python Train/train.py --train v8        # YOLOv8n
python Train/train.py --train v26       # YOLO26n
python Train/train.py --train v5        # YOLOv5n

# Export to ONNX after training
python Train/train.py --train v8 --export

# Compare model performance
python Train/train.py --compare
```
Training outputs to `runs/train/`. Dataset config: `Train/data.yaml` (classes: cube, cylinder).

### Data Pipeline
```bash
python "Object Detection Pipeline/Data_Preparation_V2/scripts/1_extract_frames.py"
python "Object Detection Pipeline/Data_Preparation_V2/scripts/2_split_dataset.py"   # 75/15/15 split
python "Object Detection Pipeline/Data_Preparation_V2/scripts/3_auto_annotation.py" # GroundingDINO
```

### Robot Pipeline
```bash
python full_pipeline/main.py            # Run full 7-stage pipeline
python full_pipeline/explore.py         # Stage 1: scan + ArUco detection
python full_pipeline/navigate.py        # Stage 2: hover TCP above tag
```

### UR10 Control
```bash
python UR10/pick_and_place.py           # Full pick-and-place automation
python UR10/read_robot_frame.py         # Read live TCP position
python UR10/check_voltage.py            # Gripper sensor diagnostics
```

### Camera / Perception
```bash
python Perception/test_camera_windows.py   # Validate OAK-D connection (USB-C)
python Perception/convert_model.py         # Convert YOLO .pt → ONNX
```

### Calibration
```bash
# Hand-eye calibration workflow (20–40 diverse poses needed)
python Calibration/eye_in_hand/collect_poses.py
python Calibration/eye_in_hand/solve_calibration.py   # Outputs handeye_calibration.json
python Calibration/eye_in_hand/replay_positions.py    # Automated recalibration
python Calibration/eye_in_hand/test_aruco.py          # Validate ArUco detection
```

### Component Tests
```bash
python pipeline_dev/scripts/01_camera_test.py
python pipeline_dev/scripts/05_robot_move_test.py
# pipeline_dev/scripts/ has 30+ numbered component tests (00–12)
```

## Architecture

### 7-Stage Pipeline (`full_pipeline/main.py`)
Runs in **DEBUG** mode (prompts before each stage) or **AUTONOMOUS** mode. Each stage retries up to 3 times before calling Recover.

1. **Explore** (`explore.py`) — Robot joint sweep, ArUco tag detection → base-frame tag pose
2. **Navigate** (`navigate.py`) — Align TCP above tag at hover height
3. **Grasp** (`grasp.py`) — Lower, close gripper, validate width + force
4. **Verify** (`verify.py`) — YOLO + CLIP dual check post-grasp
5. **Transit** — Move to drop zone with slip monitoring *(TODO)*
6. **Release** — Open gripper, confirm drop *(TODO)*
7. **Recover** (`recover.py`) — Open gripper, rise 40 cm, re-enter retry loop

### Key Coordinate Transforms
- **Camera → TCP**: solved by hand-eye calibration → `full_pipeline/calibration/T_cam2flange.npy`
- **TCP → Robot Base**: live from RTDE interface
- **Tag → Base**: ArUco detection + calibration chain at runtime

### UR10 Communication (TCP/IP)
- **Port 29999** (Dashboard server): Load/play/stop UR programs
- **Port 30002** (Secondary client): Real-time sensor stream — binary struct packets for force, gripper width, TCP pose
- Gripper width uses linear interpolation to map ADC voltage (AI2) to mm; force detection via DI8

### YOLO Inference on OAK-D
- YOLO runs on CPU via ONNX Runtime (not on-device) because OAK-D Lite doesn't support the required ops
- `numpy` must be `<2.5.0` (DepthAI 2.28.0 compatibility constraint)
- For training: GPU (RTX 5080), batch 32, 640×640, 100 epochs with heavy augmentation

### CLIP Post-Grasp Verification (`CLIP_post_grab/`)
- Zero-shot classification: "holding object" vs "empty gripper"
- `clip_gripper_verify.py` — inference (~50–200ms)
- `train_clip_probe.py` — fine-tune linear probe for improved accuracy

### Safety
- Workspace limits: `Perception/safe_limits.json`
- E-stop checked before movements
- `UR10/safety_guard.py` monitors workspace boundaries continuously
- Slip detection: `UR10/grip_control_with_slip_detection.py`

## Key Files

| File | Purpose |
|------|---------|
| `full_pipeline/main.py` | Top-level pipeline orchestrator (DEBUG/AUTONOMOUS modes) |
| `full_pipeline/explore.py` | Robot scan + ArUco detection (300+ lines) |
| `full_pipeline/navigate.py` | TCP alignment above target (267 lines) |
| `full_pipeline/grasp.py` | Descent, grip, width+force validation |
| `full_pipeline/verify.py` | YOLO + CLIP dual post-grasp check |
| `full_pipeline/recover.py` | Rise, open gripper, re-enter retry loop |
| `UR10/pick_and_place.py` | Multi-threaded pick-and-place with parallel monitoring |
| `UR10/failure_detection_pipeline.py` | Advanced failure detection (800+ lines) |
| `Train/train.py` | Unified YOLO trainer (v5/v8/v26) |
| `Train/data.yaml` | Dataset config — edit `path` for different machines |
| `full_pipeline/calibration/T_cam2flange.npy` | Hand-eye calibration result |
| `Perception/safe_limits.json` | Workspace safety boundaries |
| `SETUP.md` | Full setup guide with troubleshooting |

## Notes

- ArUco config is hardcoded: tag ID **13**, `DICT_6X6_250`, 21mm marker size
- `transformers` must be pinned to `4.38.0` — newer versions break `groundingdino-py` in the annotation env
- `pipeline_dev/` contains 30+ iterative dev/debug scripts — useful as reference for component behavior
- Debug logs in `pipeline_dev/RTDE_debug_log.md` and `slip_detection_debug_log.md`
- Robot movement timing: 3.5–6.0s per operation (see `UR10/README.md`)
- Pick positions spaced 200mm apart; place position and home position configurable in `UR10/pick_and_place.py`
