"""
2_split_dataset.py
------------------
Splits the combined V3 flat images into train / val / test sets (75/15/15 ratio).
Same logic as V2 but adapted for the massive V3 output count.
"""

import sys
import io
import random
import shutil
from pathlib import Path
from tqdm import tqdm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ============================================================
# CONFIGURATION
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR.parent / "data"
IMAGES_DIR = DATA_DIR / "images"

TRAIN_RATIO = 0.75
VAL_RATIO   = 0.15
RANDOM_SEED = 42

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ["train", "val", "test"]

# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "=" * 62)
    print("  MODEL V3 — DATASET SPLIT (75/15/15)")
    print("=" * 62)
    
    if not IMAGES_DIR.exists():
        print(f"  [ERROR] {IMAGES_DIR} not found. Run 1_extract_and_combine.py first.")
        sys.exit(1)
        
    class_folders = [f for f in IMAGES_DIR.iterdir() if f.is_dir() and f.name not in SPLITS]
    if not class_folders:
        print("  [ERROR] No flat class data found to split.")
        sys.exit(1)
        
    class_names = [f.name for f in class_folders]
    for split in SPLITS:
        for cls in class_names:
            (IMAGES_DIR / split / cls).mkdir(parents=True, exist_ok=True)
            
    random.seed(RANDOM_SEED)
    summary = {}

    for class_folder in class_folders:
        cls = class_folder.name
        print(f"\n  [CLASS] {cls}")
        
        images = sorted([f for f in class_folder.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS])
        n = len(images)
        if n == 0:
            print("    [!] No images found.")
            continue
            
        print(f"    Total images: {n:,}")
        
        shuffled = images[:]
        random.shuffle(shuffled)
        
        train_end = round(n * TRAIN_RATIO)
        val_end   = train_end + round(n * VAL_RATIO)
        
        splits_data = {
            "train": shuffled[:train_end],
            "val":   shuffled[train_end:val_end],
            "test":  shuffled[val_end:]
        }
        
        counts = {}
        for split_name, files in splits_data.items():
            dest_dir = IMAGES_DIR / split_name / cls
            
            pbar = tqdm(total=len(files), desc=f"    Copying {split_name:<5}", unit="img", leave=False)
            for f in files:
                shutil.copy2(f, dest_dir / f.name)
                pbar.update(1)
            pbar.close()
                
            counts[split_name] = len(files)
            print(f"    {split_name:<6} -> {len(files):>5,} images")
            
        summary[cls] = counts

    print("\n" + "=" * 62)
    print("  SPLIT COMPLETE")
    print("=" * 62)
    print(f"  {'Class':<15} {'Train':>10} {'Val':>10} {'Test':>10} {'Total':>10}")
    print("  " + "-" * 60)
    
    grand_totals = {"train": 0, "val": 0, "test": 0, "total": 0}
    for cls, counts in summary.items():
        tr, va, te = counts["train"], counts["val"], counts["test"]
        tot = tr + va + te
        grand_totals["train"] += tr; grand_totals["val"] += va; grand_totals["test"] += te; grand_totals["total"] += tot
        print(f"  {cls:<15} {tr:>10,} {va:>10,} {te:>10,} {tot:>10,}")
        
    print("  " + "-" * 60)
    print(f"  {'TOTAL':<15} {grand_totals['train']:>10,} {grand_totals['val']:>10,} {grand_totals['test']:>10,} {grand_totals['total']:>10,}")
    print("\n  --> Next Step: Data is ready for Model V3 annotation!\n")

if __name__ == "__main__":
    main()
