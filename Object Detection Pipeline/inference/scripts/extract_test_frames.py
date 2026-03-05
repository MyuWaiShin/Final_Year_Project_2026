"""
extract_test_frames.py
----------------------
Extracts 3 evenly-spaced frames from each recording video and saves them
into a clean test_images/ folder ready for inference testing.

Source: DataPrepV3_recordings/cubes/ and .../cylinders/
Output: test_images/cubes/ and test_images/cylinders/

Usage:
    python extract_test_frames.py
"""

import cv2
import sys
import io
from pathlib import Path
from tqdm import tqdm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
RECORDINGS_DIR = Path(r"C:\Users\myuwa\OneDrive - Middlesex University\Major Project\Final_Year_Project_2026-1\Object Detection Pipeline\DataPrepV3_recordings")
OUTPUT_DIR     = Path(__file__).resolve().parent.parent / "test_images"
FRAMES_PER_VIDEO = 3
JPEG_QUALITY     = 95
VIDEO_EXTS       = {".mp4", ".avi", ".mov", ".mkv"}


def extract_evenly_spaced(video_path: Path, class_name: str, start_index: int) -> int:
    """Extract FRAMES_PER_VIDEO evenly-spaced frames from a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] Could not open {video_path.name}")
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return 0

    # Pick frame indices evenly across the video (avoid very first and last)
    step = total // (FRAMES_PER_VIDEO + 1)
    target_frames = [step * (i + 1) for i in range(FRAMES_PER_VIDEO)]

    out_dir = OUTPUT_DIR / class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    extracted = 0
    for idx in target_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        fname = f"{class_name}_{video_path.stem}_f{idx:05d}.jpg"
        cv2.imwrite(str(out_dir / fname), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        extracted += 1

    cap.release()
    return extracted


def main():
    print("=" * 60)
    print("  TEST FRAME EXTRACTION")
    print(f"  Source : {RECORDINGS_DIR}")
    print(f"  Output : {OUTPUT_DIR}")
    print(f"  Frames per video: {FRAMES_PER_VIDEO}")
    print("=" * 60)

    total_extracted = 0

    for class_name in ["cubes", "cylinders"]:
        src = RECORDINGS_DIR / class_name
        if not src.exists():
            print(f"\n  [WARN] Directory not found: {src}")
            continue

        videos = sorted(f for f in src.iterdir() if f.suffix.lower() in VIDEO_EXTS)
        print(f"\n  [{class_name.upper()}] — {len(videos)} video(s) found")

        idx = 0
        for vid in tqdm(videos, desc=f"  {class_name}", unit="video"):
            n = extract_evenly_spaced(vid, class_name, idx)
            idx += n
            total_extracted += n
            print(f"    {vid.name} → {n} frame(s) saved")

    print("\n" + "=" * 60)
    print(f"  DONE — {total_extracted} test images saved to:")
    print(f"  {OUTPUT_DIR.absolute()}")
    print("=" * 60)
    print("\n  → Now run: python test_inference.py")


if __name__ == "__main__":
    main()
