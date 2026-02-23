"""
1_extract_and_combine.py
------------------------
Extracts EVERY frame from multiple sources and combines them into one dataset.

Sources:
1. New recordings: recordings/cubes, recordings/cylinders
2. V1 raw videos: Data_Preparation_V1/data/raw_videos/Cube, Cylinder (ignores Arc)
3. V2 existing images: Data_Preparation_V2/data/images/cubes, cylinders

Output:
Data_Preparation_V3/data/images/cubes/
Data_Preparation_V3/data/images/cylinders/
"""

import sys
import io
import cv2
import shutil
from pathlib import Path
from tqdm import tqdm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ============================================================
# CONFIGURATION
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
V3_DATA_DIR = SCRIPT_DIR.parent / "data"
OUT_IMAGES_DIR = V3_DATA_DIR / "images"

PROJECT_ROOT = SCRIPT_DIR.parents[2]

# Define the sources
NEW_RECORDINGS_DIR = PROJECT_ROOT / "recordings"
V1_RAW_VIDEOS_DIR = PROJECT_ROOT / "Data_Preparation_V1" / "data" / "raw_videos"
V2_IMAGES_DIR = PROJECT_ROOT / "Data_Preparation_V2" / "data" / "images"

VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv", ".flv"]
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]
JPEG_QUALITY = 95

def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def extract_frames(video_path: Path, class_name: str, start_index: int) -> int:
    """Extract EVERY frame from a video; return number of frames extracted."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"    [ERROR] Could not open {video_path.name}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"    [VIDEO] {video_path.name} ({total_frames} frames)")
    
    out_dir = OUT_IMAGES_DIR / class_name
    ensure_dir(out_dir)
    
    extracted = 0
    pbar = tqdm(total=total_frames, desc="    Extracting", unit="frame", leave=False)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        img_name = f"{class_name}_src_{video_path.stem}_{start_index + extracted:05d}.jpg"
        img_path = out_dir / img_name
        cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        extracted += 1
        pbar.update(1)
        
    pbar.close()
    cap.release()
    return extracted

def copy_images(src_dir: Path, class_name: str, start_index: int) -> int:
    """Copy all images from a directory; return number of images copied."""
    out_dir = OUT_IMAGES_DIR / class_name
    ensure_dir(out_dir)
    
    images = [f for f in src_dir.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS]
    if not images:
        return 0
    
    print(f"    [DIR] {src_dir.name} ({len(images)} images)")
    copied = 0
    pbar = tqdm(total=len(images), desc="    Copying", unit="image", leave=False)
    
    for img_path in images:
        img_name = f"{class_name}_v2src_{start_index + copied:05d}.jpg"
        shutil.copy2(img_path, out_dir / img_name)
        copied += 1
        pbar.update(1)
        
    pbar.close()
    return copied

# ============================================================
# MAIN ORCHESTRATOR
# ============================================================
def main():
    print("=" * 60)
    print("  MODEL V3 — FRAME EXTRACTION & COMBINATION")
    print("=" * 60)
    
    out_counts = {"cubes": 0, "cylinders": 0}

    # 1. NEW RECORDINGS (Extract every frame)
    print("\n--- Phase 1: New Recordings ---")
    for class_name in ["cubes", "cylinders"]:
        src_dir = NEW_RECORDINGS_DIR / class_name
        if src_dir.exists():
            videos = [f for f in src_dir.iterdir() if f.suffix.lower() in VIDEO_EXTS]
            for vid in videos:
                extracted = extract_frames(vid, class_name, out_counts[class_name])
                out_counts[class_name] += extracted
        else:
            print(f"  [WARN] Missing dir: {src_dir}")

    # 2. V1 RAW VIDEOS (Extract every frame, Ignore Arc)
    print("\n--- Phase 2: V1 Raw Videos (Ignoring Arc) ---")
    v1_mapping = {"Cube": "cubes", "Cylinder": "cylinders"}
    for v1_cls, out_cls in v1_mapping.items():
        src_dir = V1_RAW_VIDEOS_DIR / v1_cls
        if src_dir.exists():
            videos = [f for f in src_dir.iterdir() if f.suffix.lower() in VIDEO_EXTS]
            for vid in videos:
                extracted = extract_frames(vid, out_cls, out_counts[out_cls])
                out_counts[out_cls] += extracted
        else:
            print(f"  [WARN] Missing dir: {src_dir}")

    # 3. V2 EXISTING IMAGES (Direct Copy)
    print("\n--- Phase 3: Coping V2 Images ---")
    for class_name in ["cubes", "cylinders"]:
        src_dir = V2_IMAGES_DIR / class_name
        if src_dir.exists():
            copied = copy_images(src_dir, class_name, out_counts[class_name])
            out_counts[class_name] += copied
        else:
            print(f"  [WARN] Missing dir: {src_dir}")

    # SUMMARY
    print("\n" + "=" * 60)
    print("  EXTRACTION & COMBINE COMPLETE")
    print("=" * 60)
    for cls, count in out_counts.items():
        print(f"  {cls:<15}: {count:>8,} images")
    print("=" * 60)
    print(f"  Total Images   : {sum(out_counts.values()):>8,}")
    print(f"  Output Dir     : {OUT_IMAGES_DIR.absolute()}")
    print("  --> Next Step  : run 2_split_dataset.py\n")

if __name__ == "__main__":
    main()
