"""
1_extract_frames.py
-------------------
Extracts frames from raw videos for each class (cubes, cylinders).

Output structure:
  data/images/
    cubes/
      cubes_0001.jpg
      cubes_0002.jpg
      ...
    cylinders/
      cylinders_0001.jpg
      ...

Usage:
  cd Data_Preparation_V2
  python scripts/1_extract_frames.py

Adjust FRAME_SKIP to control how many frames are extracted:
  1  = every frame    (30 FPS video -> 30 imgs/sec) — lots of images, very similar
  3  = every 3rd frame (~10 imgs/sec) — good balance for 20s clips
  5  = every 5th frame (~6  imgs/sec) — faster, fewer images
"""

import cv2
import sys
import io
from pathlib import Path
from tqdm import tqdm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR.parent / "data"

# Input: class subfolders containing raw .mp4 / .avi / .mov videos
VIDEO_DIR  = DATA_DIR / "raw_videos"

# Output: extracted frames, one subfolder per class
IMAGES_DIR = DATA_DIR / "images"

# Extract every Nth frame (1 = all frames, 3 = every 3rd, etc.)
FRAME_SKIP = 1

# JPEG quality (0-100)
JPEG_QUALITY = 95

# Supported video extensions
VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv", ".flv"]

# ============================================================
# MAIN
# ============================================================

print()
print("=" * 62)
print("  PDE3802 — MODEL V2 — FRAME EXTRACTION")
print("=" * 62)
print(f"  Video dir  : {VIDEO_DIR}")
print(f"  Images dir : {IMAGES_DIR}")
print(f"  Frame skip : every {FRAME_SKIP} frame(s)  (FRAME_SKIP={FRAME_SKIP})")
print()

# GPU acceleration check
if cv2.ocl.useOpenCL():
    print("  [OK] OpenCL GPU acceleration enabled")
else:
    print("  [--] GPU acceleration not available — using CPU (this is fine)")
print()

# Validate source directory
if not VIDEO_DIR.exists():
    print(f"  [ERROR] Video directory not found: {VIDEO_DIR}")
    print("          Make sure raw videos are in data/raw_videos/<class>/")
    sys.exit(1)

# Discover class folders
class_folders = sorted([f for f in VIDEO_DIR.iterdir() if f.is_dir()])
if not class_folders:
    print(f"  [ERROR] No class sub-folders found in {VIDEO_DIR}")
    sys.exit(1)

class_names = [f.name for f in class_folders]
print(f"  Found {len(class_names)} class(es): {class_names}")
print()

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

grand_total = 0

for class_folder in class_folders:
    class_name = class_folder.name

    print("=" * 62)
    print(f"  CLASS: {class_name}")
    print("=" * 62)

    # Collect video files
    video_files = sorted(
        f for f in class_folder.iterdir()
        if f.suffix.lower() in VIDEO_EXTS
    )

    if not video_files:
        print(f"  [!!] No videos found in {class_folder} — skipping.")
        continue

    print(f"  Found {len(video_files)} video(s)")

    # Output folder for this class
    class_out = IMAGES_DIR / class_name
    class_out.mkdir(parents=True, exist_ok=True)

    # Global image counter (across all videos for this class)
    img_counter = 1

    for video_file in video_files:
        print(f"\n  [VIDEO] {video_file.name}")

        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            print(f"    [ERROR] Could not open {video_file.name} — skipping.")
            continue

        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration     = total_frames / fps

        print(f"    FPS: {fps:.1f}  |  Frames: {total_frames}  |  Duration: {duration:.1f}s")
        print(f"    Extracting every {FRAME_SKIP} frame(s)  "
              f"(~{total_frames // FRAME_SKIP} images expected)")

        frame_idx        = 0
        extracted_count  = 0

        pbar = tqdm(total=total_frames, desc="    Extracting", unit="frame", leave=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_SKIP == 0:
                img_name = f"{class_name}_{img_counter:04d}.jpg"
                img_path = class_out / img_name
                cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                img_counter    += 1
                extracted_count += 1

            frame_idx += 1
            pbar.update(1)

        pbar.close()
        cap.release()

        print(f"    [OK] Extracted {extracted_count} frames from {video_file.name}")

    class_total = img_counter - 1
    grand_total += class_total
    print(f"\n  [OK] {class_name}: {class_total} total frames saved -> {class_out}")

# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 62)
print("  EXTRACTION COMPLETE")
print("=" * 62)
print(f"\n  {'Class':<20} {'Images':>8}")
print(f"  {'-'*20} {'-'*8}")
for class_name in class_names:
    class_out = IMAGES_DIR / class_name
    n = len(list(class_out.glob("*.jpg"))) if class_out.exists() else 0
    print(f"  {class_name:<20} {n:>8,}")
print(f"  {'-'*20} {'-'*8}")
print(f"  {'TOTAL':<20} {grand_total:>8,}")
print(f"\n  Output: {IMAGES_DIR.absolute()}")
print()
print("  --> Next step: run  2_split_dataset.py")
print("=" * 62)
