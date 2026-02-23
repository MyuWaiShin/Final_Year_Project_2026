"""
2_split_dataset.py
------------------
Splits extracted frames (from 1_extract_frames.py) into
train / val / test sets using a 75 / 15 / 15 ratio.

Input:
  data/images/<class>/*.jpg

Output:
  data/images/
    train/<class>/*.jpg
    val/<class>/*.jpg
    test/<class>/*.jpg

The raw flat folders (data/images/cubes/, data/images/cylinders/)
are left intact; files are COPIED (not moved) into the split folders.

Usage:
  cd Data_Preparation_V2
  python scripts/2_split_dataset.py
"""

import sys
import io
import random
import shutil
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR.parent / "data"

IMAGES_DIR = DATA_DIR / "images"   # where 1_extract_frames.py saved frames

# Split ratios (must sum to 1.0)
TRAIN_RATIO = 0.75
VAL_RATIO   = 0.15
TEST_RATIO  = 0.10   # float remainder -> ~15% effectively (75+15+10=100, adjust if needed)
# NOTE: with 75/15/15, set TEST_RATIO = 0.10 because the val slice already
#       takes 15% from the remaining 25%, leaving ~10% for test.
#       Actual split is computed correctly below (no remainder lost).

RANDOM_SEED = 42

# Image extensions to include
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ============================================================
# HELPERS
# ============================================================

SPLITS = ["train", "val", "test"]

def split_indices(n: int, train_r: float, val_r: float):
    """Return (train_end, val_end) indices for a list of length n."""
    train_end = round(n * train_r)
    val_end   = train_end + round(n * val_r)
    return train_end, val_end

# ============================================================
# MAIN
# ============================================================

print()
print("=" * 62)
print("  PDE3802 — MODEL V2 — DATASET SPLIT")
print("=" * 62)
print(f"  Images dir  : {IMAGES_DIR}")
print(f"  Split ratio : {int(TRAIN_RATIO*100)} / {int(VAL_RATIO*100)} / "
      f"{int((1 - TRAIN_RATIO - VAL_RATIO)*100)}  (train / val / test)")
print(f"  Random seed : {RANDOM_SEED}")
print()

if not IMAGES_DIR.exists():
    print(f"  [ERROR] Images directory not found: {IMAGES_DIR}")
    print("          Run 1_extract_frames.py first.")
    sys.exit(1)

# Find flat class folders (skip already-created split folders)
class_folders = sorted(
    f for f in IMAGES_DIR.iterdir()
    if f.is_dir() and f.name not in SPLITS
)

if not class_folders:
    print(f"  [ERROR] No class folders found in {IMAGES_DIR}")
    print("          Expected sub-folders like  images/cubes/  images/cylinders/")
    sys.exit(1)

class_names = [f.name for f in class_folders]
print(f"  Found {len(class_names)} class(es): {class_names}")
print()

# Create split/class output directories
for split in SPLITS:
    for cls in class_names:
        (IMAGES_DIR / split / cls).mkdir(parents=True, exist_ok=True)

random.seed(RANDOM_SEED)

summary = {}

for class_folder in class_folders:
    cls = class_folder.name

    print(f"  [CLASS] {cls}")
    print("  " + "-" * 58)

    # Collect images from the flat folder (exclude sub-directories)
    images = sorted(
        f for f in class_folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMG_EXTS
    )

    n = len(images)
    if n == 0:
        print(f"    [!!] No images found — skipping.")
        print()
        continue

    print(f"    Total images: {n:,}")

    # Shuffle deterministically
    shuffled = images[:]
    random.shuffle(shuffled)

    # Compute split boundaries (true 75/15/15)
    train_end = round(n * TRAIN_RATIO)
    val_end   = train_end + round(n * VAL_RATIO)
    # Everything after val_end goes to test

    splits_data = {
        "train": shuffled[:train_end],
        "val":   shuffled[train_end:val_end],
        "test":  shuffled[val_end:],
    }

    counts = {}
    for split_name, files in splits_data.items():
        dest_dir = IMAGES_DIR / split_name / cls
        for f in files:
            shutil.copy2(f, dest_dir / f.name)
        counts[split_name] = len(files)
        print(f"    {split_name:<6} -> {len(files):>5,} images  ({dest_dir})")

    summary[cls] = counts
    print()

# ============================================================
# SUMMARY TABLE
# ============================================================

print("=" * 62)
print("  SPLIT COMPLETE")
print("=" * 62)
print()
col = 10
print(f"  {'Class':<20} {'Train':>{col}} {'Val':>{col}} {'Test':>{col}} {'Total':>{col}}")
print(f"  {'-'*20} {'-'*col} {'-'*col} {'-'*col} {'-'*col}")

gt_train = gt_val = gt_test = 0
for cls, counts in summary.items():
    tr, va, te = counts["train"], counts["val"], counts["test"]
    gt_train += tr; gt_val += va; gt_test += te
    print(f"  {cls:<20} {tr:>{col},} {va:>{col},} {te:>{col},} {tr+va+te:>{col},}")

print(f"  {'-'*20} {'-'*col} {'-'*col} {'-'*col} {'-'*col}")
print(f"  {'TOTAL':<20} {gt_train:>{col},} {gt_val:>{col},} {gt_test:>{col},} "
      f"{gt_train+gt_val+gt_test:>{col},}")

print()
print(f"  Output: {IMAGES_DIR.absolute()}")
print()
print("  --> Next step: run  3_auto_annotation.py")
print("=" * 62)
