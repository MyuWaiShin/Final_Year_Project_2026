"""
4_check_labels.py
-----------------
Visually inspect auto-annotated labels from 3_auto_annotation.py.

Draws bounding boxes over random images from each split so you can
quickly spot missed detections, false positives, or bad boxes.

Usage:
    cd Data_Preparation_V3
    python scripts/4_check_labels.py

Keys:
    Any key  — next image
    Q        — skip to next split
    Ctrl-C   — quit
"""

import sys
import random
import cv2
from pathlib import Path

# ── DATASET_DIR ───────────────────────────────────────────────────────────────
# Set to your 'dataset' root (with train/ val/ test/ inside), or leave as None
# to get a folder-picker dialog when the script runs.
DATASET_DIR = None
# ─────────────────────────────────────────────────────────────────────────────

SPLITS      = ["train", "val", "test"]
CLASSES     = ["cube", "cylinder"]
COLORS      = [(0, 220, 0), (0, 140, 255)]   # green=cube, orange=cylinder

SAMPLES_PER_SPLIT = 20    # how many images to sample per split (0 = all)
RANDOM_SEED       = 42
# ─────────────────────────────────────────────────────────────────────────────


def draw_labels(img, lbl_path: Path) -> tuple[int, int]:
    """Draw YOLO boxes on img in-place. Returns (n_boxes, n_empty)."""
    h, w = img.shape[:2]
    boxes  = 0
    empty  = 0

    if not lbl_path.exists() or lbl_path.stat().st_size == 0:
        return 0, 1

    for line in lbl_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id       = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:5])
        x1 = int((cx - bw / 2) * w);  y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w);  y2 = int((cy + bh / 2) * h)

        color = COLORS[cls_id % len(COLORS)]
        label = CLASSES[cls_id] if cls_id < len(CLASSES) else str(cls_id)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        boxes += 1

    if boxes == 0:
        empty = 1
    return boxes, empty


def print_summary():
    """Print label coverage table before showing images."""
    col = 10
    print(f"\n  {'Split':<10} {'Images':>{col}} {'Labelled':>{col}} {'Coverage':>{col}}")
    print(f"  {'-'*10} {'-'*col} {'-'*col} {'-'*col}")

    grand_img = grand_lbl = 0
    for split in SPLITS:
        img_dir = DATASET_DIR / split / "images"
        lbl_dir = DATASET_DIR / split / "labels"
        n_img = len(list(img_dir.glob("*.*"))) if img_dir.exists() else 0
        n_lbl = sum(
            1 for f in lbl_dir.glob("*.txt") if f.stat().st_size > 0
        ) if lbl_dir.exists() else 0
        cov = f"{n_lbl / n_img * 100:.1f}%" if n_img else "  --"
        print(f"  {split:<10} {n_img:>{col},} {n_lbl:>{col},} {cov:>{col}}")
        grand_img += n_img
        grand_lbl += n_lbl

    print(f"  {'-'*10} {'-'*col} {'-'*col} {'-'*col}")
    grand_cov = f"{grand_lbl / grand_img * 100:.1f}%" if grand_img else "  --"
    print(f"  {'TOTAL':<10} {grand_img:>{col},} {grand_lbl:>{col},} {grand_cov:>{col}}")
    print()


def main():
    global DATASET_DIR
    random.seed(RANDOM_SEED)

    # ── Resolve dataset folder ────────────────────────────────────────────────
    if DATASET_DIR is None or not Path(DATASET_DIR).exists():
        print()
        print("  Dataset folder not set or not found — opening folder picker...")
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            picked = filedialog.askdirectory(
                title="Select your 'dataset' folder (the one with train/ val/ test/ inside)"
            )
            root.destroy()
            if not picked:
                print("  [Cancelled] No folder selected. Exiting.")
                sys.exit(0)
            DATASET_DIR = Path(picked)
        except Exception as e:
            print(f"  [ERROR] Could not open folder picker: {e}")
            print("  Set DATASET_DIR manually at the top of the script.")
            sys.exit(1)

    DATASET_DIR = Path(DATASET_DIR)

    print()
    print("=" * 58)
    print("  LABEL CHECK — Auto-annotation visualiser")
    print("=" * 58)
    print(f"  Dataset : {DATASET_DIR}")
    print(f"  Splits  : {SPLITS}")
    print(f"  Samples : {SAMPLES_PER_SPLIT} per split  (0 = all)")
    print()

    print_summary()

    print("  Controls: any key = next image | Q = next split | Ctrl-C = quit")
    print("=" * 58)
    print()

    for split in SPLITS:
        img_dir = DATASET_DIR / split / "images"
        lbl_dir = DATASET_DIR / split / "labels"

        imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        if not imgs:
            print(f"  [{split.upper()}]  No images found — skipping.")
            continue

        random.shuffle(imgs)
        sample = imgs if SAMPLES_PER_SPLIT == 0 else imgs[:SAMPLES_PER_SPLIT]

        print(f"  [{split.upper()}]  Showing {len(sample)} / {len(imgs)} images ...")

        skip_split = False
        for img_path in sample:
            if skip_split:
                break

            lbl_path = lbl_dir / (img_path.stem + ".txt")
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"    [!!] Could not read {img_path.name}")
                continue

            n_boxes, n_empty = draw_labels(img, lbl_path)

            status = f"{n_boxes} box(es)" if n_boxes else "NO LABEL"
            title  = f"[{split}] {img_path.name}  —  {status}"

            # Resize for display if image is very large
            dh, dw = img.shape[:2]
            max_dw, max_dh = 1200, 800
            if dw > max_dw or dh > max_dh:
                scale = min(max_dw / dw, max_dh / dh)
                img   = cv2.resize(img, (int(dw * scale), int(dh * scale)))

            cv2.imshow(title, img)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()

            if key == ord('q') or key == ord('Q'):
                skip_split = True

        print()

    print("  Done. All splits checked.")


if __name__ == "__main__":
    main()
