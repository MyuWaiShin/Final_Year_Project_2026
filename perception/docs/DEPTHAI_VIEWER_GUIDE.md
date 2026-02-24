# DepthAI Viewer Setup Guide

## What is DepthAI Viewer?

The official **DepthAI Viewer** is a powerful GUI application from Luxonis that lets you:
- View RGB, depth, and all camera streams simultaneously
- Visualize neural network outputs
- See real-time FPS and performance metrics
- Inspect device info and calibration data
- Record and playback data

## Installation

### Option 1: Install via pip (Recommended)

```powershell
py -3.11 -m pip install depthai-viewer
```

### Option 2: Download Standalone Executable

1. Go to: https://github.com/luxonis/depthai-viewer/releases
2. Download the latest Windows `.exe` file
3. Run it directly (no installation needed)

## Running DepthAI Viewer

### If installed via pip:
```powershell
py -3.11 -m depthai_viewer
```

### If using standalone exe:
Just double-click the downloaded `.exe` file

## Features

Once launched, the viewer will:
- Automatically detect your OAK-D camera
- Show all available streams (RGB, left/right mono, depth, etc.)
- Display FPS for each stream
- Allow you to record data for later analysis
- Show device temperature and other diagnostics

## Using with Your Detection Model

To visualize your YOLO detections in the viewer, you would need to:
1. Create a DepthAI pipeline with your model as a blob
2. Run the pipeline
3. The viewer will automatically show the detection outputs

**Note:** Since we're running the model on CPU (not as a blob), the viewer won't show your custom detections. It will only show the raw camera feeds and depth.

## Alternative: Use Our Custom Script

Our `detect_objects.py` script already shows:
- ✅ RGB camera feed
- ✅ Detection bounding boxes
- ✅ Depth measurements
- ✅ FPS counter (now added)
- ✅ Object counts

This is more suitable for your current setup since the model runs on CPU.
