# OAK-D Lite Perception System

Object detection and depth estimation system using the OAK-D Lite camera for robotic manipulation tasks.

## Hardware Requirements

- **OAK-D Lite Camera** (Luxonis)
- USB-C cable
- Windows 10/11 PC

## Software Requirements

- **Python 3.11.9** (not 3.13!)
- Windows (native, not WSL)

## Quick Start

### 1. Install Python 3.11

Download and install Python 3.11.9 from:
https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe

✅ **Important**: Check "Add Python 3.11 to PATH" during installation

### 2. Install Dependencies

```powershell
py -3.11 -m pip install depthai==2.28.0 opencv-python numpy onnxruntime
```

### 3. Connect Camera

1. Plug in OAK-D Lite via USB-C
2. Wait for Windows to recognize the device
3. Check Device Manager → "Movidius MyriadX" should appear

### 4. Test Camera Connection

```powershell
cd "path\to\Perception"
py -3.11 test_camera_windows.py
```

You should see:
- ✅ "SUCCESS! Connected to: [camera ID]"
- ✅ Live RGB camera feed window
- Press 'q' to quit

### 5. Run Object Detection

First, convert your trained YOLO model to ONNX:

```powershell
py -3.11 convert_model.py
```

Then run live detection:

```powershell
py -3.11 detect_objects.py
```

This will show:
- Live camera feed
- Bounding boxes for detected objects (cubes, cylinders, arcs)
- Confidence scores
- Depth measurements
- FPS counter

Press 'q' to quit.

## Project Structure

```
Perception/
├── detect_objects.py          # Main detection script (CPU-based YOLO)
├── test_camera_windows.py     # Camera connection test
├── convert_model.py           # Convert .pt → .onnx
├── models/                    # Trained YOLO models
│   ├── yolov8n_50epochs/
│   └── yolov8n_100epochs/
├── README.md                  # This file
├── TROUBLESHOOTING.md         # Complete error history & solutions
└── DEPTHAI_VIEWER_GUIDE.md   # Official DepthAI Viewer setup
```

## Key Files

### `detect_objects.py`
Main detection script that:
- Loads ONNX model and runs inference on CPU
- Gets RGB + depth streams from OAK-D camera
- Displays bounding boxes with class labels and depth
- Shows real-time FPS

### `test_camera_windows.py`
Simple camera test to verify connection works.

### `convert_model.py`
Converts PyTorch YOLO models (`.pt`) to ONNX format (`.onnx`) for inference.

## Trained Models

Located in `models/` directory:
- **yolov8n_50epochs**: YOLOv8n trained for 50 epochs
- **yolov8n_100epochs**: YOLOv8n trained for 100 epochs (currently used)

**Classes detected:**
- `cube` - Styrofoam cubes
- `cylinder` - Styrofoam cylinders  

### Why YOLOv8n (not YOLO26n)?

**YOLOv8n** is the standard Ultralytics YOLO architecture:
- ✅ Proven, stable, well-documented
- ✅ Better community support
- ✅ Compatible with most deployment tools
- ✅ Easier to convert and optimize

**YOLO26n** is a custom/experimental variant:
- ❌ Less stable and documented
- ❌ Harder to deploy
- ❌ Not worth the complexity for this use case

**Recommendation:** Always use YOLOv8n for training new models.

## Recording Data with DepthAI Viewer

See `DEPTHAI_VIEWER_GUIDE.md` for instructions on using the official Luxonis viewer to record camera data.

## Recording Training Data with OAK-D

To collect new training data for better model performance:

```powershell
py -3.11 record_data.py
```

This will:
- Record 1080p video at 30fps
- Save as H.264 MP4 format
- Show live preview while recording
- Default 20 second duration (configurable)

**See `DATA_COLLECTION_GUIDE.md` for:**
- Best practices for data collection
- UR10 robot movement strategies
- How to collect diverse training data
- Recommended dataset sizes

## Troubleshooting

If you encounter errors, see `TROUBLESHOOTING.md` for detailed solutions to:
- `X_LINK_DEVICE_NOT_FOUND` errors
- Python version issues
- DepthAI library version conflicts
- Model conversion failures
- WSL USB connection problems

## Common Issues

### Camera not connecting

**Solution:**
1. Try a different USB port (preferably USB 3.0)
2. Restart the camera (unplug and replug)
3. Check that `DEPTHAI_BOOT_TIMEOUT` is set (already done in scripts)

### Detection showing wrong classes

**Solution:**
- Model needs more training data
- Try the 100 epochs model: already set in `detect_objects.py`
- Retrain with more diverse backgrounds and lighting

### Low FPS

**Solution:**
- Normal: 8-15 FPS on CPU (ONNX Runtime)
- For faster inference, would need to convert to `.blob` format (requires model architecture changes)

## Development Notes

### Why CPU inference instead of on-device?

The trained YOLO models use operations (`Tile` with S32 input) not supported by the OAK-D's Myriad X chip. Running on CPU with ONNX Runtime is a practical workaround that still achieves real-time performance.

### Why Python 3.11 instead of 3.13?

Python 3.13 doesn't have pre-built wheels for `depthai 2.28.0`, requiring compilation from source (needs Visual Studio build tools). Python 3.11 has pre-compiled binaries.

### Why depthai 2.28.0 instead of 3.x?

Version 3.3.0 has breaking API changes and boot stability issues. Version 2.28.0 is the stable LTS release.

## Next Steps

1. **Collect more training data** using the OAK-D camera
2. **Retrain models** with diverse backgrounds
3. **Implement pose estimation** using depth + bounding box data
4. **Integrate with ROS2** for robot control (use lab computers with native Linux)

## License

This project is part of a Major Project at Middlesex University.

## Contact

For issues or questions, refer to `TROUBLESHOOTING.md` first.
