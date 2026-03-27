"""
dataset_stats.py
----------------
Check the state of a YOLO-format dataset at any time.

Expected structure:
    <dataset>/
        train/  images/  labels/
        val/    images/  labels/
        test/   images/  labels/
        data.yaml  (optional)

Usage:
    python dataset_stats.py
    (a folder picker opens if DATA_DIR is not set)
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(r"C:\Users\myuwa\OneDrive - Middlesex University\FYP Datasets\Dataset3 (combined)\dataset")
# ─────────────────────────────────────────────────────────────────────────────

SPLITS           = ["train", "val", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
# ─────────────────────────────────────────────────────────────────────────────


def pick_folder():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        picked = filedialog.askdirectory(
            title="Select your 'dataset' folder (with train/ val/ test/ inside)"
        )
        root.destroy()
        return Path(picked) if picked else None
    except Exception as e:
        print(f"  [ERROR] Folder picker failed: {e}")
        return None


def count_images(split_dir: Path) -> int:
    d = split_dir / "images"
    if not d.exists():
        return 0
    return sum(1 for f in d.iterdir()
               if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)


def count_labels(split_dir: Path):
    """Returns (non_empty_label_files, total_box_count)."""
    d = split_dir / "labels"
    if not d.exists():
        return 0, 0
    non_empty = 0
    total_boxes = 0
    for f in d.iterdir():
        if f.suffix != ".txt":
            continue
        lines = [l for l in f.read_text(encoding="utf-8", errors="replace")
                              .splitlines() if l.strip()]
        if lines:
            non_empty += 1
            total_boxes += len(lines)
    return non_empty, total_boxes


def count_classes(split_dir: Path) -> dict:
    """Returns {class_id: box_count} across all label files in split."""
    d = split_dir / "labels"
    counts = {}
    if not d.exists():
        return counts
    for f in d.iterdir():
        if f.suffix != ".txt":
            continue
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.strip().split()
            if parts:
                cid = int(parts[0])
                counts[cid] = counts.get(cid, 0) + 1
    return counts


def parse_yaml_classes(data_dir: Path) -> list:
    yaml_path = data_dir / "data.yaml"
    if not yaml_path.exists():
        return []
    for line in yaml_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip().startswith("names:"):
            raw = line.split(":", 1)[1].strip()
            names = [n.strip(" []'\"") for n in raw.split(",")]
            return [n for n in names if n]
    return []


def pct(part, total):
    if total == 0:
        return "   --"
    return f"{part / total * 100:5.1f}%"


def main():
    global DATA_DIR

    if DATA_DIR is None or not Path(DATA_DIR).exists():
        print("\n  Dataset folder not set -- opening folder picker...")
        DATA_DIR = pick_folder()
        if DATA_DIR is None:
            print("  Cancelled. Exiting.")
            sys.exit(0)

    DATA_DIR = Path(DATA_DIR)
    class_names = parse_yaml_classes(DATA_DIR)

    print()
    print("=" * 62)
    print("  DATASET STATISTICS  (YOLO format)")
    print(f"  Folder: {DATA_DIR}")
    print("=" * 62)
    if class_names:
        print(f"  Classes ({len(class_names)}): {', '.join(class_names)}")
    print()

    # ── Per-split summary ──────────────────────────────────────────────────────
    col = 10
    print(f"  {'Split':<8} {'Images':>{col}} {'Labelled':>{col}} "
          f"{'Boxes':>{col}} {'Coverage':>{col}}  {'Avg bx/img':>10}")
    print(f"  {'-'*8} {'-'*col} {'-'*col} {'-'*col} {'-'*col}  {'-'*10}")

    grand_img = grand_lbl = grand_box = 0

    for split in SPLITS:
        n_img        = count_images(DATA_DIR / split)
        n_lbl, n_box = count_labels(DATA_DIR / split)
        avg          = f"{n_box / n_img:.2f}" if n_img else "  --"
        flag         = " " if n_img else "!"
        print(f"  {split:<8} {n_img:>{col},} {n_lbl:>{col},} "
              f"{n_box:>{col},} {pct(n_lbl, n_img):>{col}}  {avg:>10}  [{flag}]")
        grand_img += n_img
        grand_lbl += n_lbl
        grand_box += n_box

    print(f"  {'-'*8} {'-'*col} {'-'*col} {'-'*col} {'-'*col}  {'-'*10}")
    grand_avg = f"{grand_box / grand_img:.2f}" if grand_img else "  --"
    print(f"  {'TOTAL':<8} {grand_img:>{col},} {grand_lbl:>{col},} "
          f"{grand_box:>{col},} {pct(grand_lbl, grand_img):>{col}}  {grand_avg:>10}")

    if grand_img > 0 and grand_lbl > grand_img:
        print(f"\n  (* Coverage >100%: some images contain 2+ objects,")
        print(f"     so box count > image count. Not an error.)")

    # ── Per-class breakdown ────────────────────────────────────────────────────
    print()
    print(f"  CLASS BREAKDOWN (box count per split)")
    print(f"  {'Class':<16}", end="")
    for s in SPLITS:
        print(f"  {s.capitalize():>{col}}", end="")
    print(f"  {'Total':>{col}}")
    print(f"  {'-'*16}", end="")
    for _ in SPLITS:
        print(f"  {'-'*col}", end="")
    print(f"  {'-'*col}")

    all_ids = set()
    split_cls = {}
    for split in SPLITS:
        counts = count_classes(DATA_DIR / split)
        split_cls[split] = counts
        all_ids.update(counts.keys())

    for cid in sorted(all_ids):
        name = class_names[cid] if cid < len(class_names) else f"class_{cid}"
        print(f"  {name:<16}", end="")
        row_total = 0
        for split in SPLITS:
            n = split_cls[split].get(cid, 0)
            row_total += n
            print(f"  {n:>{col},}", end="")
        print(f"  {row_total:>{col},}")

    # ── Readiness ─────────────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("  READINESS")
    print("=" * 62)
    yaml_ok = (DATA_DIR / "data.yaml").exists()
    checks = [
        (grand_img > 0,  f"{grand_img:,} images across all splits"),
        (grand_lbl > 0,  f"{grand_lbl:,} labelled images  ({grand_box:,} boxes)"),
        (yaml_ok,        "data.yaml found" if yaml_ok
                         else "data.yaml NOT found  (run 3_auto_annotation.py)"),
    ]
    for ok, msg in checks:
        print(f"  {'[x]' if ok else '[ ]'}  {msg}")
    if grand_img > 0 and grand_lbl > 0 and yaml_ok:
        print(f"\n  --> Ready for YOLO training!")
        print(f"      yolo train model=yolov8n.pt data={DATA_DIR / 'data.yaml'}")
    print()
    print("=" * 62)
    print()


if __name__ == "__main__":
    main()
