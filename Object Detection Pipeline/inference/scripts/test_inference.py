"""
test_inference.py
-----------------
Runs test inference on all three trained YOLO models (v8n, v26n, v5n)
against the extracted test images, saves annotated results, and prints
a quick summary table.

Requires:
    pip install ultralytics opencv-python

Usage:
    # Run all models
    python test_inference.py

    # Run a single model
    python test_inference.py --model v8
    python test_inference.py --model v26
    python test_inference.py --model v5

    # Change confidence threshold
    python test_inference.py --conf 0.4

Output:
    inference/results/<model_name>/<image_name>_pred.jpg
"""

import argparse
import sys
import io
import os
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ─────────────────────────────────────────────
# CONFIG — update these paths for this machine
# ─────────────────────────────────────────────
WEIGHTS_DIR = Path(r"C:\Users\myuwa\OneDrive - Middlesex University\FYP Datasets\runs\train")

MODEL_PATHS = {
    "v8":  WEIGHTS_DIR / "yolov8n"  / "weights" / "best.pt",
    "v26": WEIGHTS_DIR / "yolo26n"  / "weights" / "best.pt",
    "v5":  WEIGHTS_DIR / "yolov5n"  / "weights" / "best.pt",
}

# Default confidence thresholds per model (from F1 curve analysis)
DEFAULT_CONF = {
    "v8":  0.566,
    "v26": 0.565,
    "v5":  0.405,
}

SCRIPT_DIR  = Path(__file__).resolve().parent
IMAGES_DIR  = SCRIPT_DIR.parent / "test_images"   # created by extract_test_frames.py
RESULTS_DIR = SCRIPT_DIR.parent / "inference_results"
IMG_SIZE    = 640
CLASS_NAMES = ["cube", "cylinder"]


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
def run_model(model_key: str, conf: float):
    from ultralytics import YOLO
    import cv2

    pt = MODEL_PATHS[model_key]
    if not pt.exists():
        print(f"\n  [SKIP] {model_key} weights not found: {pt}")
        return {}

    print(f"\n{'='*55}")
    print(f"  MODEL: yolo{model_key}n   |   conf={conf:.3f}")
    print(f"{'='*55}")

    model = YOLO(str(pt))
    out_dir = RESULTS_DIR / f"yolo{model_key}n"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gather all test images
    images = list(IMAGES_DIR.rglob("*.jpg")) + list(IMAGES_DIR.rglob("*.png"))
    if not images:
        print(f"  [WARN] No images found in {IMAGES_DIR}")
        print("         Run extract_test_frames.py first.")
        return {}

    detection_counts = {"cube": 0, "cylinder": 0, "total_images": len(images)}

    for img_path in sorted(images):
        results = model.predict(
            source=str(img_path),
            conf=conf,
            imgsz=IMG_SIZE,
            verbose=False,
        )

        result = results[0]
        boxes  = result.boxes

        # Count detections
        for cls_id in (boxes.cls.int().tolist() if boxes is not None else []):
            name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
            detection_counts[name] = detection_counts.get(name, 0) + 1

        # Save annotated image
        annotated = result.plot()
        out_name  = f"{img_path.stem}_pred.jpg"
        import cv2
        cv2.imwrite(str(out_dir / out_name), annotated)

        # Print per-image summary
        n_det = len(boxes) if boxes is not None else 0
        labels = []
        if boxes is not None:
            for cls_id, conf_score in zip(boxes.cls.int().tolist(), boxes.conf.tolist()):
                name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls{cls_id}"
                labels.append(f"{name}:{conf_score:.2f}")
        det_str = ", ".join(labels) if labels else "no detections"
        print(f"  {img_path.parent.name}/{img_path.name:<35}  [{det_str}]")

    print(f"\n  Saved annotated images → {out_dir}")
    return detection_counts


def compare_summary(all_results: dict):
    print(f"\n{'='*60}")
    print(f"  INFERENCE SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<12} {'Images':<8} {'Cubes':<8} {'Cylinders':<12}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*12}")
    for model_key, counts in all_results.items():
        imgs  = counts.get("total_images", 0)
        cubes = counts.get("cube", 0)
        cyls  = counts.get("cylinder", 0)
        print(f"  yolo{model_key}n{'':<7} {imgs:<8} {cubes:<8} {cyls:<12}")
    print(f"{'='*60}")
    print(f"\n  Annotated results saved to: {RESULTS_DIR.absolute()}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test inference on all YOLO models")
    parser.add_argument(
        "--model", choices=["v8", "v26", "v5", "all"], default="all",
        help="Which model to run (default: all)"
    )
    parser.add_argument(
        "--conf", type=float, default=None,
        help="Override confidence threshold for all models (default: per-model optimal)"
    )
    args = parser.parse_args()

    models_to_run = ["v8", "v26", "v5"] if args.model == "all" else [args.model]

    all_results = {}
    for key in models_to_run:
        conf = args.conf if args.conf is not None else DEFAULT_CONF[key]
        all_results[key] = run_model(key, conf)

    if len(all_results) > 1:
        compare_summary(all_results)

    print("\nUSAGE EXAMPLES:")
    print("  python test_inference.py                   # run all 3 models")
    print("  python test_inference.py --model v8        # run only v8n")
    print("  python test_inference.py --conf 0.4        # override confidence")
    print("  python test_inference.py --model v5 --conf 0.3")
