"""
3_dataset_stats.py
------------------
Quick-check script — run at ANY point during data prep to see
the current state of the V2 dataset.

Reports:
  • Number of raw videos per class
  • Number of extracted images per class × split (train / val / test)
  • Number of annotation labels per class × split (if annotation has started)
  • Estimated split ratio

Usage:
    python 3_dataset_stats.py
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

# [PATH CONFIGURATION]
# Set this to the absolute path of your data folder. 
# This folder should contain the 'images' and 'labels' subfolders.
DATA_DIR = Path(r"C:\Users\myuwa\OneDrive - Middlesex University\Major Project\Datasets\Data_Preparation_V3\data")

RAW_VIDEO_DIR = DATA_DIR / "raw_videos"
IMAGES_DIR    = DATA_DIR / "images"
LABELS_DIR    = DATA_DIR / "labels"      # populated after annotation

CLASSES = ["cubes", "cylinders"]
SPLITS  = ["train", "val", "test"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
LABEL_EXTENSION  = ".txt"

# ============================================================
# HELPERS
# ============================================================

def count_files(directory: Path, extensions: set) -> int:
    if not directory.exists():
        return 0
    return sum(1 for f in directory.iterdir()
               if f.is_file() and f.suffix.lower() in extensions)


def bar(value: int, maximum: int, width: int = 20) -> str:
    """Simple ASCII progress bar."""
    if maximum == 0:
        return " " * width
    filled = int(round(value / maximum * width))
    return "█" * filled + "░" * (width - filled)


def pct(part: int, total: int) -> str:
    if total == 0:
        return "  —  "
    return f"{part / total * 100:5.1f}%"


# ============================================================
# SECTIONS
# ============================================================

def section_raw_videos():
    print("=" * 62)
    print("  [RAW VIDEOS]")
    print("=" * 62)

    grand_total = 0
    for cls in CLASSES:
        n = count_files(RAW_VIDEO_DIR / cls, VIDEO_EXTENSIONS)
        grand_total += n
        status = "[OK]" if n > 0 else "[!!]"
        print(f"  {status}  {cls:<15}  {n:>3} video(s)")

    print(f"\n  Total recordings : {grand_total}")
    if not RAW_VIDEO_DIR.exists():
        print(f"\n  [!!] Folder not found: {RAW_VIDEO_DIR}")
        print("      --> Run 1_splice_videos.py first.")
    print()


def section_images():
    print("=" * 62)
    print("  [EXTRACTED IMAGES]  (train / val / test)")
    print("=" * 62)

    # Build table
    table   = {}   # table[cls][split] = count
    col_tot = {s: 0 for s in SPLITS}   # per-split grand totals
    row_tot = {}                         # per-class totals
    grand   = 0

    for cls in CLASSES:
        table[cls] = {}
        row_tot[cls] = 0
        for split in SPLITS:
            n = count_files(IMAGES_DIR / split / cls, IMAGE_EXTENSIONS)
            table[cls][split] = n
            row_tot[cls]     += n
            col_tot[split]   += n
            grand            += n

    # Header
    col_w = 9
    print(f"\n  {'Class':<15}", end="")
    for s in SPLITS:
        print(f"  {s.capitalize():>{col_w}}", end="")
    print(f"  {'Total':>{col_w}}  {'Share':>6}")
    print(f"  {'─'*15}", end="")
    for _ in SPLITS:
        print(f"  {'─'*col_w}", end="")
    print(f"  {'─'*col_w}  {'─'*6}")

    for cls in CLASSES:
        print(f"  {cls:<15}", end="")
        for split in SPLITS:
            print(f"  {table[cls][split]:>{col_w},}", end="")
        print(f"  {row_tot[cls]:>{col_w},}  {pct(row_tot[cls], grand)}")

    print(f"  {'─'*15}", end="")
    for _ in SPLITS:
        print(f"  {'─'*col_w}", end="")
    print(f"  {'─'*col_w}  {'─'*6}")

    print(f"  {'TOTAL':<15}", end="")
    for split in SPLITS:
        print(f"  {col_tot[split]:>{col_w},}", end="")
    print(f"  {grand:>{col_w},}  100.0%")

    # Ratio row
    print(f"\n  Split ratio →", end="")
    for split in SPLITS:
        print(f"  {split}:{pct(col_tot[split], grand)}", end="")
    print()

    # Missing-folder warnings
    missing = []
    for cls in CLASSES:
        for split in SPLITS:
            d = IMAGES_DIR / split / cls
            if not d.exists():
                missing.append(str(d))

    if missing:
        print(f"\n  [!!] Missing output directories (run 2_extract_frames.py):")
        for m in missing[:6]:
            print(f"       {m}")
    elif grand == 0:
        print("\n  [!!] No images found -- run 2_extract_frames.py.")
    print()


def section_labels():
    print("=" * 62)
    print("  [ANNOTATION LABELS]  (post auto-annotation)")
    print("=" * 62)

    grand = 0
    any_labels = False

    for split in SPLITS:
        for cls in CLASSES:
            lbl_dir = LABELS_DIR / split / cls
            n = count_files(lbl_dir, {LABEL_EXTENSION})
            img_dir = IMAGES_DIR / split / cls
            n_img   = count_files(img_dir, IMAGE_EXTENSIONS)

            if n > 0:
                any_labels = True
            grand += n

            coverage = pct(n, n_img) if n_img else "  —  "
            b        = bar(n, n_img)
            print(f"  {split}/{cls:<14}  {n:>5} labels / {n_img:>5} images"
                  f"  [{b}]  {coverage}")

    if not any_labels:
        print("\n  [!!] No labels yet -- annotation not started.")
        print("      --> Run auto_annotation.py (coming soon).")
    else:
        print(f"\n  Total labels: {grand:,}")
    print()


def section_summary():
    print("=" * 62)
    print("  [READINESS SUMMARY]")
    print("=" * 62)

    # Videos
    n_videos = sum(
        count_files(RAW_VIDEO_DIR / cls, VIDEO_EXTENSIONS) for cls in CLASSES
    )
    # Images
    n_images = sum(
        count_files(IMAGES_DIR / split / cls, IMAGE_EXTENSIONS)
        for split in SPLITS for cls in CLASSES
    )
    # Labels
    n_labels = sum(
        count_files(LABELS_DIR / split / cls, {LABEL_EXTENSION})
        for split in SPLITS for cls in CLASSES
    )

    steps = [
        ("1_splice_videos.py",   n_videos > 0, f"{n_videos} video(s) ready"),
        ("2_extract_frames.py",  n_images > 0, f"{n_images:,} frames extracted"),
        ("auto_annotation.py",   n_labels > 0, f"{n_labels:,} labels generated"),
    ]

    for script, done, detail in steps:
        icon = "[x]" if done else "[ ]"
        print(f"  {icon}  {script:<28}  {detail}")

    print()

# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 62)
    print("  PDE3802 — MODEL V2 — DATASET STATISTICS")
    print(f"  Data root: {DATA_DIR.absolute()}")
    print("=" * 62)
    print()

    section_raw_videos()
    section_images()
    section_labels()
    section_summary()

    print("=" * 62)
    print()


if __name__ == "__main__":
    main()
