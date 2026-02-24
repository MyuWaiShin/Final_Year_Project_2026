# OAK-D Deployment Options

## Current Setup (Laptop Tethered)

**What you have now:**
- OAK-D connected via USB-C to laptop
- YOLO model runs on **laptop CPU** (ONNX Runtime)
- Camera provides RGB + depth streams
- Laptop does all processing and sends results

**Pros:**
- ✅ Easy to develop and debug
- ✅ Powerful CPU for complex models
- ✅ Can use any model (no conversion needed)

**Cons:**
- ❌ Laptop must be physically connected via USB cable
- ❌ Not portable/standalone
- ❌ Laptop must run continuously

---

## Option 1: On-Device Inference (OAK-D Standalone)

**Can the OAK-D run the model by itself?**

**Short answer: Partially, but not with your current YOLO model.**

### What the OAK-D CAN do standalone:
- ✅ Capture RGB, depth, and stereo images
- ✅ Run neural networks on the **Myriad X chip** (if converted to `.blob` format)
- ✅ Output detection results over USB or network
- ✅ Basic pose estimation using depth + detections

### What the OAK-D CANNOT do standalone:
- ❌ Run ONNX models directly (needs `.blob` format)
- ❌ Run complex post-processing (NMS, tracking, etc.)
- ❌ Display results (no screen)
- ❌ Store data long-term (limited onboard memory)

### The Problem with Your Model:
Your YOLOv8n model uses operations (`Tile` with S32 input) that **aren't supported** by the Myriad X chip. That's why blob conversion failed.

### Possible Solutions:
1. **Retrain with MobileNet-SSD** (fully supported on Myriad X)
2. **Use YOLOv6n** (better Myriad X compatibility than YOLOv8n)
3. **Simplify YOLO architecture** (remove unsupported layers)
4. **Use edge computer** (Raspberry Pi, Jetson Nano, etc.)

---

## Option 2: Raspberry Pi + OAK-D (Recommended ✅)

**Setup:**
- OAK-D connected to Raspberry Pi via USB
- Pi runs detection model (CPU or lightweight on-device)
- Pi communicates with UR10 robot over network
- Laptop only for development/monitoring

### Recommended Hardware:
- **Raspberry Pi 5** (8GB RAM) - Best performance
- **Raspberry Pi 4** (4GB/8GB RAM) - Good enough
- **NVIDIA Jetson Nano** - Better for AI (has GPU)

### Architecture:
```
OAK-D Camera (USB) → Raspberry Pi → Network → UR10 Robot
                         ↓
                    Detection Model
                    Pose Estimation
                    ROS2 Node
```

### Pros:
- ✅ Portable and standalone
- ✅ Can run 24/7
- ✅ Laptop not needed after deployment
- ✅ Can run ROS2 natively (Linux)
- ✅ Network communication with robot

### Cons:
- ❌ Slower inference than laptop (~3-8 FPS on Pi 4)
- ❌ Additional hardware cost (~$50-150)
- ❌ Requires setup and configuration

---

## Option 3: Hybrid (Development + Deployment)

**Best approach for your project:**

### Development Phase (Now):
- Use **laptop tethered** for:
  - Data collection
  - Model training
  - Testing and debugging
  - Visualization

### Deployment Phase (Later):
- Use **Raspberry Pi** for:
  - Running final detection model
  - Pose estimation
  - Robot control integration
  - Autonomous operation

---

## Recommended Deployment Strategy

### For Your UR10 Project:

#### Option A: Tethered Laptop (Simplest)
**Use if:**
- Lab computer is always near robot
- Don't need portability
- Want maximum performance

**Setup:**
1. Mount OAK-D on UR10 end-effector
2. USB cable from camera to lab computer
3. Lab computer runs detection + ROS2
4. Computer sends commands to UR10 over network

**Cable management:**
- Use **3-5m USB-C cable** with active repeater
- Route cable along robot arm (cable management clips)
- Ensure cable has slack for robot movement

---

#### Option B: Raspberry Pi (Portable)
**Use if:**
- Want standalone system
- Need portability
- Lab computer not always available

**Setup:**
1. Mount OAK-D + Raspberry Pi on robot base or nearby
2. Pi connects to OAK-D via short USB cable
3. Pi runs detection model + ROS2
4. Pi communicates with UR10 over Ethernet/WiFi

**Recommended Pi Setup:**
- **Raspberry Pi 5 (8GB)** - $80
- **32GB+ SD card** - $15
- **Power supply** - $12
- **Case with cooling** - $10
- **Total: ~$120**

---

#### Option C: On-Device Inference (Advanced)
**Use if:**
- Need maximum portability
- Willing to retrain model
- Want lowest latency

**Requirements:**
1. Retrain with **MobileNet-SSD** or **YOLOv6n**
2. Convert to `.blob` format successfully
3. Run inference on Myriad X chip
4. Use OAK-D's onboard processing

**Pros:**
- ✅ No external computer needed
- ✅ Lowest latency (~30 FPS)
- ✅ Lowest power consumption

**Cons:**
- ❌ Requires model retraining
- ❌ Limited to supported architectures
- ❌ Still need something to receive results (Pi or laptop)

---

## Performance Comparison

| Setup | FPS | Latency | Portability | Cost |
|-------|-----|---------|-------------|------|
| Laptop (current) | 8-15 | ~100ms | ❌ Tethered | $0 |
| Raspberry Pi 4 | 3-8 | ~200ms | ✅ Portable | ~$100 |
| Raspberry Pi 5 | 8-12 | ~120ms | ✅ Portable | ~$120 |
| Jetson Nano | 15-25 | ~60ms | ✅ Portable | ~$150 |
| On-device (blob) | 20-30 | ~30ms | ⚠️ Semi-portable | $0 |

---

## My Recommendation for Your Project

### Phase 1: Development (Now)
**Use tethered laptop:**
- Collect data with laptop + OAK-D
- Train models on laptop
- Test detection with laptop
- Develop pose estimation with laptop

### Phase 2: Integration (Later)
**Choose based on your needs:**

**If lab computer is always available:**
→ Keep tethered laptop setup (simplest, no extra cost)

**If you need portability or standalone operation:**
→ Get Raspberry Pi 5 (8GB) + run detection on Pi

**If you want maximum performance:**
→ Get NVIDIA Jetson Nano (has GPU for AI)

---

## For Your Specific Use Case

**UR10 with hand-mounted camera:**

### Tethered Laptop is Fine Because:
- ✅ Lab computer is stationary
- ✅ Robot workspace is fixed
- ✅ USB cable can be routed along arm
- ✅ No additional hardware cost
- ✅ Best performance for development

### Cable Routing:
1. Use **5m active USB-C cable**
2. Attach cable to robot arm with **cable clips**
3. Leave **slack at joints** for movement
4. Route to stationary laptop

### When to Switch to Pi:
- If you need to move robot to different locations
- If lab computer is shared/unavailable
- If you want to demo without laptop
- If you're deploying long-term

---

## Summary

**Do you NEED a Raspberry Pi?**
- **No** - Tethered laptop works fine for your use case
- **But** - Pi is nice for portability and standalone operation

**Can OAK-D run standalone?**
- **Partially** - Can run models if converted to `.blob`
- **Your model** - Cannot run standalone (unsupported operations)
- **Solution** - Either use Pi or retrain with compatible architecture

**My advice:**
1. **Stick with tethered laptop** for now (development)
2. **Use 5m USB cable** with proper routing
3. **Consider Pi later** if you need portability
4. **Don't worry about on-device inference** unless you have specific latency requirements

The tethered setup is perfectly fine for a university project!
