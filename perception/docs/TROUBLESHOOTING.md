# OAK-D Lite Camera Setup & Troubleshooting Guide

## What is X_LINK?

**X_LINK** is the communication protocol used by OAK-D cameras to talk between:
- The **host computer** (your laptop)
- The **Myriad X chip** (the AI processor inside the camera)

Think of it as a "bridge" that allows your Python code to send commands to the camera and receive video/depth data back.

### The X_LINK_DEVICE_NOT_FOUND Error

This error means the host computer **lost connection** to the camera during the boot process. Here's what happens:

1. **Camera plugged in** → Shows up as USB device (VID:PID `03e7:2485`)
2. **Python tries to connect** → Camera starts booting its firmware
3. **Camera reboots** → Briefly disconnects and changes USB ID to bootloader (`03e7:f63b`)
4. **Camera finishes booting** → Should reconnect with new ID (`03e7:2485`)
5. **❌ Problem**: If the OS doesn't recognize the reconnection fast enough → `X_LINK_DEVICE_NOT_FOUND`

### Why This Happens More on WSL/Windows

- **WSL (Windows Subsystem for Linux)**: Adds an extra USB virtualization layer (`usbipd`) that doesn't handle the camera's USB re-enumeration well
- **Windows**: Sometimes USB drivers don't refresh quickly enough
- **Native Linux**: Works best because the kernel handles USB re-enumeration natively

---

## Complete Troubleshooting History

### 1. Initial Approach: WSL Setup

**Why we chose WSL:**
- ROS2 (Robot Operating System) works best on Linux
- WSL allows running Linux tools on Windows without dual-boot
- Seemed like a good compromise for development

**What we tried:**
1. Installed `usbipd-win` to share USB devices with WSL
2. Bound the OAK-D camera to WSL using `usbipd bind` and `usbipd attach`
3. Installed ROS2 Humble in Ubuntu 22.04 (WSL)
4. Set up `depthai` Python library

**Errors encountered:**
```
RuntimeError: Failed to find device after booting, error message: X_LINK_DEVICE_NOT_FOUND
```

**Root cause:**
- The camera changes its USB device ID during boot (from unbooted → bootloader → booted)
- `usbipd` doesn't automatically follow this device re-enumeration
- WSL loses connection during the boot process

**Attempted fixes:**
- Created `force_connect.ps1` PowerShell script to auto-attach the bootloader device
- Increased `DEPTHAI_BOOT_TIMEOUT` to 30 seconds
- Forced USB2 mode (`dai.UsbSpeed.HIGH`)
- Downgraded `depthai` from 3.3.0 to 2.28.0 (version 3.3.0 had breaking API changes)
- Set OpenVINO version to `VERSION_2021_4` for firmware stability

**Result:** ❌ Still unreliable. WSL USB virtualization was the fundamental blocker.

---

### 2. Python Version Issues

**Problem:**
- Python 3.13 (latest) couldn't install `depthai 2.28.0` from source
- Missing build tools (`nmake`, Visual Studio C++ compiler)
- Dependency conflicts with `numpy` versions

**Error:**
```
ERROR: Could not build wheels for depthai
```

**Solution:**
- ✅ Installed Python 3.11.9 (has pre-built wheels for `depthai 2.28.0`)
- Used `py -3.11` launcher to run scripts with the correct Python version

---

### 3. DepthAI Library Version Issues

**Problem:**
- `depthai 3.3.0` had API breaking changes:
  - `DeviceInfo.getMxId()` → `DeviceInfo.deviceId`
  - Pipeline creation triggered immediate device boot (caused crashes)

**Error:**
```
AttributeError: 'depthai.DeviceInfo' object has no attribute 'getMxId'
RuntimeError: Failed to find device after booting (at Pipeline creation)
```

**Solution:**
- ✅ Downgraded to `depthai==2.28.0` (stable version)
- Used Python 3.11 which has pre-compiled wheels

---

### 4. Final Solution: Native Windows

**What worked:**
1. ✅ Installed Python 3.11.9 on Windows
2. ✅ Installed `depthai==2.28.0` using pip (pre-built wheel)
3. ✅ Set `DEPTHAI_BOOT_TIMEOUT=30000` before importing `depthai`
4. ✅ Forced USB2 mode for stability
5. ✅ Used OpenVINO 2021.4 firmware version

**Why it works:**
- No USB virtualization layer (direct USB connection)
- Windows USB drivers handle device re-enumeration properly
- Stable `depthai` version with pre-built binaries

---

## Model Conversion Issues

### YOLO to OAK-D Blob Conversion

**Problem:**
- YOLOv8 models use operations (like `Tile` with S32 input) not supported by Myriad X chip
- `blobconverter` service failed with:
```
Stage node /model.23/Tile_output_0 (Tile) types check error: 
input #0 has type S32, but one of [FP16] is expected
```

**Solution:**
- ✅ Run YOLO model on **CPU** using ONNX Runtime instead of on-device
- Convert `.pt` → `.onnx` (works fine)
- Skip `.onnx` → `.blob` conversion
- Use `onnxruntime` to run inference on the host computer
- OAK-D camera only provides RGB + Depth streams

**Trade-off:**
- ❌ Model doesn't run on camera's Myriad X chip
- ✅ Still get real-time detection (~8-15 FPS on laptop CPU)
- ✅ Can use any YOLO model without blob conversion issues

---

## Summary of All Errors & Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `X_LINK_DEVICE_NOT_FOUND` (WSL) | USB re-enumeration not handled by `usbipd` | ✅ Use native Windows instead of WSL |
| `AttributeError: getMxId` | `depthai 3.3.0` API changes | ✅ Downgrade to `depthai==2.28.0` |
| `Could not build wheels` | Python 3.13 + missing build tools | ✅ Use Python 3.11 (has pre-built wheels) |
| `X_LINK_DEVICE_NOT_FOUND` (Windows) | Boot timeout too short | ✅ Set `DEPTHAI_BOOT_TIMEOUT=30000` |
| Blob conversion failure | Unsupported YOLO operations for Myriad X | ✅ Run model on CPU with ONNX Runtime |
| `X_LINK_DEVICE_NOT_FOUND` (DepthAI Viewer) | Same USB re-enumeration issue | ⚠️ Try different USB port, restart camera |

---

## Key Takeaways

1. **Use native Windows** for OAK-D development (not WSL)
2. **Use Python 3.11** (not 3.13) for stable `depthai` support
3. **Use `depthai==2.28.0`** (not 3.x) for stability
4. **Set environment variables BEFORE importing** `depthai`
5. **Run YOLO on CPU** if blob conversion fails
6. **X_LINK errors** are almost always USB connection/timing issues
