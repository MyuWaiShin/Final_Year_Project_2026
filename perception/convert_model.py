"""
Convert YOLOv8/YOLO PyTorch model to OAK-D blob format

This script converts your trained .pt model to .blob format
so it can run on the OAK-D camera's Myriad X processor.
"""

import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install required packages for conversion"""
    print("Installing conversion tools...")
    packages = [
        "blobconverter",
        "onnx",
        "onnxruntime",
        "ultralytics"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def convert_to_onnx(pt_model_path, output_path):
    """Convert PyTorch model to ONNX format"""
    from ultralytics import YOLO
    
    print(f"\nStep 1: Converting {pt_model_path} to ONNX...")
    model = YOLO(pt_model_path)
    
    # Export to ONNX with OAK-D compatible settings
    model.export(
        format="onnx",
        imgsz=416,  # OAK-D works best with 416x416
        simplify=True,
        opset=12
    )
    
    # The export creates a file with .onnx extension in the same directory
    onnx_path = pt_model_path.with_suffix('.onnx')
    print(f"✓ ONNX model created: {onnx_path}")
    return onnx_path

def convert_to_blob(onnx_path, output_path, num_classes=3):
    """Convert ONNX model to blob format using blobconverter"""
    import blobconverter
    
    print(f"\nStep 2: Converting ONNX to blob format...")
    print("This may take a few minutes...")
    
    blob_path = blobconverter.from_onnx(
        model=str(onnx_path),
        data_type="FP16",
        shaves=6,
        use_cache=False,
        output_dir=str(output_path.parent)
    )
    
    # Rename to expected name
    final_blob_path = output_path
    Path(blob_path).rename(final_blob_path)
    
    print(f"✓ Blob model created: {final_blob_path}")
    return final_blob_path

def main():
    print("=" * 70)
    print("YOLOv8 to OAK-D Blob Converter")
    print("=" * 70)
    
    # Paths
    pt_model = Path("models/yolov8n_100epochs/weights/best.pt")
    blob_output = Path("models/yolov8n_100epochs/weights/best.blob")
    
    if not pt_model.exists():
        print(f"\nERROR: Model not found at {pt_model}")
        print("Please check the path and try again.")
        return
    
    print(f"\nInput:  {pt_model}")
    print(f"Output: {blob_output}")
    print()
    
    # Install requirements
    try:
        install_requirements()
    except Exception as e:
        print(f"\nERROR installing requirements: {e}")
        print("Please install manually: py -3.11 -m pip install blobconverter onnx ultralytics")
        return
    
    # Convert to ONNX
    try:
        onnx_path = convert_to_onnx(pt_model, blob_output)
    except Exception as e:
        print(f"\nERROR converting to ONNX: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure ultralytics is installed: py -3.11 -m pip install ultralytics")
        print("2. Check that your .pt model is valid")
        return
    
    # Convert to blob
    try:
        blob_path = convert_to_blob(onnx_path, blob_output, num_classes=3)
    except Exception as e:
        print(f"\nERROR converting to blob: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure blobconverter is installed: py -3.11 -m pip install blobconverter")
        print("2. Check your internet connection (blobconverter needs to download tools)")
        return
    
    print("\n" + "=" * 70)
    print("✓ CONVERSION COMPLETE!")
    print("=" * 70)
    print(f"\nYour model is ready: {blob_path}")
    print("\nNext step: Run the detection script")
    print("  py -3.11 detect_objects.py")
    print("=" * 70)

if __name__ == "__main__":
    main()
