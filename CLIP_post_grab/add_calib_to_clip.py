import cv2
import os
import glob
from pathlib import Path

def main():
    calib_dir = Path(r"c:\Users\myuwa\.gemini\antigravity\scratch\Final_Year_Project_2026\Calibration\eye_in_hand\data\images")
    clip_base = Path(r"c:\Users\myuwa\.gemini\antigravity\scratch\Final_Year_Project_2026\CLIP_post_grab\clip_dataset")
    
    empty_crop_dir = clip_base / "cropped" / "empty"
    empty_full_dir = clip_base / "full" / "empty"
    
    empty_crop_dir.mkdir(parents=True, exist_ok=True)
    empty_full_dir.mkdir(parents=True, exist_ok=True)

    images = glob.glob(str(calib_dir / "*.png"))
    if not images:
        print(f"No images found in {calib_dir}")
        return

    print(f"Found {len(images)} calibration images. Processing...")

    for i, img_path in enumerate(images):
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        
        # Calibration frames were saved at 1280x720. 
        # Our CLIP dataset was captured at 1920x1080.
        # Resize it so the crop math matches perfectly.
        frame_1080 = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_CUBIC)
        
        h, w = frame_1080.shape[:2]
        crop_w = 1400
        crop_h = 600
        
        cx = w // 2
        cy = h - (crop_h // 2) - 10 
        
        x1 = max(0, cx - crop_w // 2)
        y1 = max(0, cy - crop_h // 2)
        x2 = min(w, x1 + crop_w)
        y2 = min(h, y1 + crop_h)
        
        crop_frame = frame_1080[y1:y2, x1:x2]
        
        # Save them as prefixed files so they don't overwrite
        filename = f"calib_empty_{i:03d}.png"
        cv2.imwrite(str(empty_crop_dir / filename), crop_frame)
        cv2.imwrite(str(empty_full_dir / filename), frame_1080)
        
    print(f"Successfully processed and added {len(images)} images to the 'empty' dataset!")

if __name__ == "__main__":
    main()
