import os
from pathlib import Path

def count_images_in_dir(path):
    if not os.path.exists(path):
        return 0
    return len([f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

def main():
    base_dir = Path(__file__).parent / "clip_dataset"
    
    if not base_dir.exists():
        print(f"Dataset directory not found at: {base_dir}")
        return

    print("=" * 40)
    print("CLIP Dataset Image Count")
    print("=" * 40)
    
    total_cropped = 0
    total_full = 0
    
    # Check cropped images
    print("\n--- CROPPED IMAGES ---")
    crop_base = base_dir / "cropped"
    for label in ["holding", "empty", "unknown"]:
        path = crop_base / label
        count = count_images_in_dir(path)
        total_cropped += count
        print(f"  {label:<10}: {count} images")
    print(f"  Total cropped: {total_cropped}")
        
    # Check full frames
    print("\n--- FULL FRAMES ---")
    full_base = base_dir / "full"
    for label in ["holding", "empty", "unknown"]:
        path = full_base / label
        count = count_images_in_dir(path)
        total_full += count
        print(f"  {label:<10}: {count} images")
    print(f"  Total full   : {total_full}")
    
    print("\n" + "=" * 40)
    print(f"GRAND TOTAL    : {total_cropped + total_full} files")
    print("=" * 40)

if __name__ == "__main__":
    main()
