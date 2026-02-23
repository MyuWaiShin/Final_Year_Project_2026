"""
3_auto_annotation.py
--------------------
Automatically annotates split frames with GroundingDINO via autodistill,
then assembles the final YOLO-ready dataset.

Pipeline:
  1. Validates that split images exist (run 2_split_dataset.py first)
  2. Runs GroundingDINO per class across all splits
  3. Remaps class IDs to global order (cubes=0, cylinders=1)
  4. Assembles final  data/dataset/{train,val,test}/{images,labels}/
  5. Writes data.yaml

Compatibility fixes applied vs. V3 original:
  - Uses groundingdino-py (pure-Python wheel) instead of the C++ build
    to avoid MSVC build errors on Windows.
  - Initialises GroundingDINO WITHOUT box_threshold / text_threshold
    kwargs (older autodistill-grounding-dino <0.1.6 doesn't accept them).
    Thresholds are set via environment variables instead.
  - Explicit UTF-8 stdout wrapper for Windows terminals.
  - Graceful fallback if a split folder has 0 images.

Requirements (install in this order):
  1. pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
     (or cu118 / cpu-only — see requirements_annotation.txt)
  2. pip install groundingdino-py
  3. pip install -r requirements_annotation.txt

Usage:
  cd Data_Preparation_V3
  python scripts/3_auto_annotation.py
"""

import sys
import io
import os
import shutil
import random
import importlib.util
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR  = Path(__file__).resolve().parent
DATA_DIR    = SCRIPT_DIR.parent / "data"

IMAGES_DIR  = DATA_DIR / "images"    # output of 2_split_dataset.py
LABELS_DIR  = DATA_DIR / "labels"    # intermediate per-class labels
DATASET_DIR = DATA_DIR / "dataset"   # final YOLO dataset

SPLITS = ["train", "val", "test"]

# ── Class configuration ──────────────────────────────────────
# Map class folder name -> GroundingDINO text prompt
PROMPT_MAP = {
    "cubes":     "cube",
    "cylinders": "cylinder",
}

# Global class order -> class IDs (cubes=0, cylinders=1)
CLASS_ORDER = ["cubes", "cylinders"]

# GroundingDINO confidence thresholds
# Set via env vars to avoid API compatibility issues across autodistill versions
BOX_THRESHOLD  = 0.35
TEXT_THRESHOLD = 0.25

RANDOM_SEED = 42

# ============================================================
# HELPERS
# ============================================================

def ensure_grounding_dino_config():
    """
    autodistill-grounding-dino expects the SwinT config file at:
        ~/.cache/autodistill/groundingdino/GroundingDINO_SwinT_OGC.py

    It is never copied there automatically.  This function finds it inside
    the installed groundingdino package and copies it if it's missing.
    """
    cache_dir   = Path.home() / ".cache" / "autodistill" / "groundingdino"
    config_dest = cache_dir / "GroundingDINO_SwinT_OGC.py"

    if config_dest.exists():
        print("  [OK] GroundingDINO config already in cache")
        return

    # Locate the installed groundingdino package
    spec = importlib.util.find_spec("groundingdino")
    if spec is None or spec.origin is None:
        print("  [!!] groundingdino package not found — skipping config copy")
        return

    pkg_dir    = Path(spec.origin).parent
    config_src = pkg_dir / "config" / "GroundingDINO_SwinT_OGC.py"

    if not config_src.exists():
        print(f"  [!!] Config not found inside package at {config_src}")
        print("       Model load may fail — continuing anyway.")
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_src, config_dest)
    print(f"  [OK] Copied GroundingDINO SwinT config -> {config_dest}")


def import_autodistill():
    """
    Lazily import autodistill so that import errors are readable.

    Windows compatibility note:
      autodistill-grounding-dino can fail to import if the GroundingDINO
      C++ extension is not compiled.  Installing 'groundingdino-py' (pure
      Python wheel) beforehand prevents this.
    """
    try:
        from autodistill_grounding_dino import GroundingDINO
        from autodistill.detection import CaptionOntology
        return GroundingDINO, CaptionOntology
    except ImportError as exc:
        print("\n[ERROR] Could not import autodistill-grounding-dino.")
        print("  Install steps:")
        print("    1. pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121")
        print("    2. pip install groundingdino-py")
        print("    3. pip install scikit-learn roboflow")
        print("    4. pip install autodistill autodistill-grounding-dino supervision")
        print(f"\n  Original error: {exc}")
        sys.exit(1)


def build_model(GroundingDINO, CaptionOntology, prompt: str, class_name: str):
    """
    Initialise GroundingDINO.

    Compatibility note:
      Older autodistill-grounding-dino (<0.1.6) raises TypeError if
      box_threshold / text_threshold are passed as kwargs to the
      constructor.  We attempt both signatures and fall back gracefully.
    """
    ontology = CaptionOntology({prompt: class_name})

    # Try new API (>=0.1.6) first
    try:
        model = GroundingDINO(
            ontology=ontology,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )
        return model
    except TypeError:
        pass

    # Fall back to old API — thresholds via env vars
    os.environ["BOX_THRESHOLD"]  = str(BOX_THRESHOLD)
    os.environ["TEXT_THRESHOLD"] = str(TEXT_THRESHOLD)
    model = GroundingDINO(ontology=ontology)
    return model


def remap_labels(raw_label_files, dest_dir: Path, class_id: int):
    """
    Copy label files from autodistill's temp output into dest_dir,
    remapping class 0 -> class_id.

    Returns (remapped_count, empty_count).
    """
    remapped = 0
    empty    = 0

    for lbl_file in raw_label_files:
        try:
            text  = lbl_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        lines     = text.splitlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                parts[0] = str(class_id)
                new_lines.append(" ".join(parts))

        dest = dest_dir / lbl_file.name
        if new_lines:
            dest.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            remapped += 1
        else:
            dest.write_text("", encoding="utf-8")
            empty += 1

    return remapped, empty


# ============================================================
# STEP 1 — VALIDATE IMAGES
# ============================================================

def validate_images():
    print("=" * 62)
    print("  STEP 1 — Validating split images")
    print("=" * 62)

    total   = 0
    missing = []

    for split in SPLITS:
        for cls in CLASS_ORDER:
            d = IMAGES_DIR / split / cls
            if not d.exists():
                missing.append(f"{split}/{cls}")
                continue
            n = len(list(d.glob("*.jpg"))) + len(list(d.glob("*.png")))
            total += n
            status = "[OK]" if n > 0 else "[!!]"
            print(f"  {status}  {split}/{cls:<15}  {n:>6,} images")
            if n == 0:
                missing.append(f"{split}/{cls}")

    if missing:
        print(f"\n  [ERROR] The following folders are missing or empty:")
        for m in missing:
            print(f"          {m}")
        print("\n  --> Run 2_split_dataset.py first, then re-run this script.")
        sys.exit(1)

    print(f"\n  Total images to annotate: {total:,}")
    print()


# ============================================================
# STEP 2 — ANNOTATE WITH GROUNDING DINO
# ============================================================

def annotate_all(GroundingDINO, CaptionOntology):
    print("=" * 62)
    print("  STEP 2 — Auto-annotating with GroundingDINO")
    print("=" * 62)
    print(f"  box_threshold  = {BOX_THRESHOLD}")
    print(f"  text_threshold = {TEXT_THRESHOLD}")
    print()

    tmp_dir = DATA_DIR / "_tmp_annotations"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for cls in CLASS_ORDER:
        class_id = CLASS_ORDER.index(cls)
        prompt   = PROMPT_MAP.get(cls, cls.rstrip("s"))

        print(f"\n  [CLASS] {cls}  (class_id={class_id}, prompt='{prompt}')")
        print("  " + "-" * 58)

        # Build model once per class (re-using across splits is fine)
        model = build_model(GroundingDINO, CaptionOntology, prompt, cls)

        for split in SPLITS:
            img_dir = IMAGES_DIR / split / cls
            lbl_out = LABELS_DIR / split / cls
            lbl_out.mkdir(parents=True, exist_ok=True)

            imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
            if not imgs:
                print(f"    [{split.upper()}]  0 images — skipped")
                continue

            print(f"\n    [{split.upper()}]  {len(imgs):,} images  ->  {img_dir.relative_to(DATA_DIR)}/")

            tmp_out = tmp_dir / f"{cls}_{split}"
            tmp_out.mkdir(parents=True, exist_ok=True)

            # autodistill writes labels into tmp_out/train/labels/ and tmp_out/valid/labels/
            model.label(
                input_folder=str(img_dir),
                output_folder=str(tmp_out),
                extension="*.jpg",
            )

            # Collect generated label files
            raw_labels = []
            for sub in ("train", "valid"):
                sub_lbl = tmp_out / sub / "labels"
                if sub_lbl.exists():
                    raw_labels.extend(sub_lbl.glob("*.txt"))

            print(f"    [OK] {len(raw_labels)} label files produced by GroundingDINO")

            remapped, empty = remap_labels(raw_labels, lbl_out, class_id)
            print(f"    [OK] {remapped} remapped to class_id={class_id}  "
                  f"({empty} empty / no detection)")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"\n  Temp folder cleaned up.")


# ============================================================
# STEP 3 — ASSEMBLE FINAL YOLO DATASET
# ============================================================

def assemble_dataset():
    print("\n" + "=" * 62)
    print("  STEP 3 — Assembling YOLO dataset")
    print("=" * 62)

    for split in SPLITS:
        (DATASET_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    total_copied  = 0
    missing_lbls  = 0

    for split in SPLITS:
        for cls in CLASS_ORDER:
            img_dir = IMAGES_DIR / split / cls
            lbl_dir = LABELS_DIR / split / cls

            imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
            for img in imgs:
                lbl = lbl_dir / (img.stem + ".txt")

                shutil.copy2(img, DATASET_DIR / split / "images" / img.name)

                if lbl.exists():
                    shutil.copy2(lbl, DATASET_DIR / split / "labels" / lbl.name)
                else:
                    (DATASET_DIR / split / "labels" / (img.stem + ".txt")).write_text("")
                    missing_lbls += 1

                total_copied += 1

    print(f"\n  Copied {total_copied:,} image+label pairs into dataset/")
    if missing_lbls:
        print(f"  [!] {missing_lbls} images had no label — empty .txt written")


# ============================================================
# STEP 4 — CREATE data.yaml
# ============================================================

def create_data_yaml():
    print("\n" + "=" * 62)
    print("  STEP 4 — Writing data.yaml")
    print("=" * 62)

    # Singular display names (cube, cylinder)
    class_display = [cls.rstrip("s") for cls in CLASS_ORDER]

    yaml_content = (
        f"# PDE3802 Model V3 — YOLO Dataset Config\n"
        f"path: {DATASET_DIR.absolute().as_posix()}\n"
        f"train: train/images\n"
        f"val:   val/images\n"
        f"test:  test/images\n"
        f"\n"
        f"nc: {len(CLASS_ORDER)}\n"
        f"names: {class_display}\n"
    )

    yaml_path = DATASET_DIR / "data.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")

    print(f"\n  Written : {yaml_path}")
    print(f"  Classes : {class_display}")


# ============================================================
# SUMMARY
# ============================================================

def print_summary():
    print("\n" + "=" * 62)
    print("  ANNOTATION COMPLETE")
    print("=" * 62)

    col = 10
    print(f"\n  {'Split':<10} {'Images':>{col}} {'Labels':>{col}} {'Coverage':>{col}}")
    print(f"  {'-'*10} {'-'*col} {'-'*col} {'-'*col}")

    grand_img = grand_lbl = 0
    for split in SPLITS:
        img_dir = DATASET_DIR / split / "images"
        lbl_dir = DATASET_DIR / split / "labels"
        n_img = len(list(img_dir.glob("*.*"))) if img_dir.exists() else 0
        n_lbl = sum(
            1 for f in lbl_dir.glob("*.txt") if f.stat().st_size > 0
        ) if lbl_dir.exists() else 0
        cov = f"{n_lbl / n_img * 100:.1f}%" if n_img else " -- "
        print(f"  {split:<10} {n_img:>{col},} {n_lbl:>{col},} {cov:>{col}}")
        grand_img += n_img
        grand_lbl += n_lbl

    print(f"  {'-'*10} {'-'*col} {'-'*col} {'-'*col}")
    grand_cov = f"{grand_lbl / grand_img * 100:.1f}%" if grand_img else " -- "
    print(f"  {'TOTAL':<10} {grand_img:>{col},} {grand_lbl:>{col},} {grand_cov:>{col}}")

    print(f"\n  Dataset folder : {DATASET_DIR.absolute()}")
    print(f"  YAML file      : {(DATASET_DIR / 'data.yaml').absolute()}")
    print("\n  --> Ready for YOLO training!")
    print("      Example:")
    print("        yolo train model=yolov8n.pt data=data/dataset/data.yaml epochs=100")
    print("=" * 62)


# ============================================================
# MAIN
# ============================================================

def main():
    random.seed(RANDOM_SEED)

    print()
    print("=" * 62)
    print("  PDE3802 — MODEL V3 — AUTO ANNOTATION")
    print("  Model: GroundingDINO (via autodistill)")
    print("=" * 62)
    print(f"\n  Images dir  : {IMAGES_DIR}")
    print(f"  Labels dir  : {LABELS_DIR}")
    print(f"  Dataset dir : {DATASET_DIR}")
    print(f"  Classes     : {CLASS_ORDER}")
    print()

    validate_images()

    print("  Checking GroundingDINO config cache...")
    ensure_grounding_dino_config()
    print()

    print("  Loading GroundingDINO model (first run downloads ~700 MB weights)...")
    GroundingDINO, CaptionOntology = import_autodistill()
    print("  [OK] autodistill imported\n")

    annotate_all(GroundingDINO, CaptionOntology)
    assemble_dataset()
    create_data_yaml()
    print_summary()


if __name__ == "__main__":
    main()
