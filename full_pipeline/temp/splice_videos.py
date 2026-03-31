"""
splice_videos.py
----------------
Reads videos from temp/videos/<class_name>/ and extracts frames into:

    temp/detection/<class_name>/       – if you're doing object detection
    temp/classification/<class_name>/  – if you're doing classification

Frame extraction rate is calculated automatically from the video's FPS
so that roughly TARGET_FPS frames are saved per second of footage.
e.g. a 30fps video → saves every 2nd frame → ~15 unique frames/sec.

No label files are created – raw images only. Label afterwards with
LabelImg, Roboflow, CVAT, etc.

Usage
-----
    python splice_videos.py
"""

import cv2
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR     = Path(__file__).resolve().parent
VIDEO_DIR      = SCRIPT_DIR / "videos"

IMAGE_EXT      = ".jpg"
JPEG_QUALITY   = 95

# Save every Nth frame. step=2 means every other frame.
# 30fps → ~15 unique frames/sec  |  60fps → ~30 unique frames/sec
FRAME_STEP     = 2
# ─────────────────────────────────────────────────────────────────────────────



def choose_mode() -> str:
    print("\n" + "=" * 55)
    print("  What task are you preparing data for?")
    print("=" * 55)
    print("  1 – Object Detection     (detection/)")
    print("  2 – Classification       (classification/)")
    while True:
        c = input("\nEnter 1 or 2: ").strip()
        if c == "1":
            return "detection"
        if c == "2":
            return "classification"
        print("[WARN] Please type 1 or 2.")


def list_class_folders() -> list:
    if not VIDEO_DIR.exists():
        return []
    return sorted([p for p in VIDEO_DIR.iterdir() if p.is_dir()])


def choose_classes(folders: list) -> list:
    if not folders:
        print(f"\n[ERROR] No class folders found in {VIDEO_DIR}")
        print("        Run collect_videos.py first.")
        return []

    print("\nAvailable class folders:")
    for i, f in enumerate(folders, 1):
        n_vids = len(list(f.glob("*.mp4")) + list(f.glob("*.avi")) +
                     list(f.glob("*.mov")) + list(f.glob("*.mkv")))
        print(f"  [{i}] {f.name}  ({n_vids} video(s))")
    print("  [A] All")

    while True:
        raw = input("\nChoose numbers (comma-separated) or A for all: ").strip().lower()
        if raw == "a":
            return folders
        try:
            idx = [int(x.strip()) - 1 for x in raw.split(",")]
            return [folders[i] for i in idx]
        except (ValueError, IndexError):
            print("[WARN] Invalid selection – try again.")


def get_video_files(folder: Path) -> list:
    exts = ["*.mp4", "*.MP4", "*.avi", "*.AVI", "*.mov", "*.MOV", "*.mkv", "*.MKV"]
    vids = []
    for e in exts:
        vids.extend(folder.glob(e))
    return sorted(set(vids))


def splice_class(class_folder: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = get_video_files(class_folder)
    if not videos:
        print(f"  [WARN] No videos in {class_folder}")
        return 0

    # Resume-friendly: start index from existing image count
    existing     = len(list(out_dir.glob(f"*{IMAGE_EXT}")))
    frame_global = existing
    total_saved  = 0

    for vid in videos:
        cap = cv2.VideoCapture(str(vid))
        if not cap.isOpened():
            print(f"  [ERROR] Cannot open {vid.name}")
            continue

        vid_fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step         = FRAME_STEP
        effective    = vid_fps / step

        print(f"  ├─ {vid.name}")
        print(f"  │    FPS={vid_fps:.1f}  step=every {step} frames → ~{effective:.1f} frames/sec saved")

        idx   = 0
        saved = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                name = f"{class_folder.name}_{frame_global:06d}{IMAGE_EXT}"
                cv2.imwrite(str(out_dir / name), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                frame_global += 1
                saved        += 1
            idx += 1

        cap.release()
        total_saved += saved
        print(f"  │    {saved} / {total_frames} frames saved")

    return total_saved


def main():
    print("=" * 55)
    print("  YOLO Data Pipeline – Video Splicer")
    print("=" * 55)

    mode        = choose_mode()
    output_root = SCRIPT_DIR / mode          # temp/detection/ or temp/classification/

    folders  = list_class_folders()
    selected = choose_classes(folders)
    if not selected:
        return

    print(f"\n[INFO] Mode    : {mode.upper()}")
    print(f"[INFO] Output  : {output_root}")
    print(f"[INFO] Classes : {[f.name for f in selected]}")
    print(f"[INFO] Frame step : every {FRAME_STEP} frames (step=2 fixed)")
    input("\nPress ENTER to start …")

    grand_total = 0
    for folder in selected:
        out_dir = output_root / folder.name
        print(f"\n[CLASS] {folder.name}  →  {out_dir}")
        count = splice_class(folder, out_dir)
        grand_total += count
        print(f"  └─ {count} images saved for '{folder.name}'")

    print("\n" + "=" * 55)
    print(f"  DONE — {grand_total} images total  →  {output_root}")
    print("=" * 55)
    print("\nNext: annotate with LabelImg / Roboflow, then train YOLO.")


if __name__ == "__main__":
    main()
