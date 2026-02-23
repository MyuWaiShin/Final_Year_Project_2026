# UR10 Pick-and-Place System

A Python-based pick-and-place automation system for the Universal Robots UR10 with OnRobot RG2 gripper, featuring calibrated sensor feedback and real-time object detection.

## 🎯 Project Overview

This system automates the pick-and-place operation for multiple objects using:
- **Robot**: Universal Robots UR10
- **Gripper**: OnRobot RG2 parallel jaw gripper
- **Communication**: TCP/IP sockets (Python)
- **Features**: Real-time object detection, drop monitoring, calibrated sensors

## 📁 Files

### Core Scripts

| File | Purpose |
|------|---------|
| `pick_and_place.py` | Complete pick-and-place automation system |
| `gripper_with_detection.py` | Gripper control with calibrated object detection |
| `dashboard_gripper.py` | Basic gripper control via Dashboard Server |
| `read_base_frame.py` | Real-time TCP position reader |
| `check_voltage.py` | Diagnostic tool for gripper voltage verification |

### Development Journey

1. **dashboard_gripper.py** - Initial connection and basic gripper control
2. **read_base_frame.py** - Position reading for recording coordinates
3. **gripper_with_detection.py** - Calibrated sensor feedback and object detection
4. **pick_and_place.py** - Full automation with parallel monitoring

## 🚀 Quick Start

### Prerequisites

```bash
pip install socket struct threading time
```

### Configuration

Update the robot IP address in each script:
```python
ROBOT_IP = "192.168.8.102"  # Change to your robot's IP
```

### Running the System

**Full pick-and-place automation:**
```bash
python pick_and_place.py
```

**Test gripper with detection:**
```bash
python gripper_with_detection.py
```

**Read current TCP position:**
```bash
python read_base_frame.py
```

**Check gripper voltage:**
```bash
python check_voltage.py
```

## 🔧 Technical Implementation

### Communication: TCP/IP Sockets with URScript Injection

The system uses two communication ports:

**Port 29999 - Dashboard Server** (program control):
```python
sock.connect((robot_ip, 29999))
sock.sendall(("load grip_open.urp\n").encode())
```

**Port 30002 - Secondary Interface** (feedback + URScript):
```python
script = """def move_program():
  movel(p[x, y, z, rx, ry, rz], a=0.5, v=0.5)
end
"""
sock.connect((robot_ip, 30002))
sock.send(script.encode())
```

### Data Parsing: Binary Packet Decoding

Real-time sensor data is extracted using Python's `struct` module:

```python
# Read packet
packet_len = struct.unpack("!I", data)[0]
packet_data = sock.recv(packet_len - 4)

# Extract gripper voltage (Package Type 2)
voltage = struct.unpack("!d", ai2_bytes)[0]

# Extract TCP position (Package Type 4)
tcp_x = struct.unpack("!d", packet_data[offset+5:offset+13])[0]
```

### Threading Architecture

**Daemon thread** for continuous sensor monitoring:
```python
recv_thread = threading.Thread(target=self._update_loop, daemon=True)
recv_thread.start()
```

**Temporary threads** for parallel monitoring during movement:
```python
# Movement in background
lift_thread = threading.Thread(target=lambda: self.move_to(...))
lift_thread.start()

# Monitor grip in parallel
while moving:
    if not self.is_object_detected():
        abort()
```

### Timing Strategy

Calibrated sleep delays ensure movement completion:
- Fast movements (approach/retract): 3.5s
- Normal movements (pick/place): 5.0s
- Joint movements (long distance): 6.0s
- Gripper operations: 4.0s

## 📊 Gripper Sensor Calibration

The RG2 gripper's analog sensor required calibration due to circuit impedance and component tolerances.

### Calibration Process

**Measured reference points:**
- Fully open: Raw = 65.8mm, Actual = 91.0mm
- Fully closed: Raw = 8.5mm, Actual = 10.5mm

**Linear calibration formula:**
```python
slope = (91.0 - 10.5) / (65.8 - 8.5) ≈ 1.405
offset = 10.5 - (8.5 × 1.405) ≈ -1.44
actual_width = (raw_width × 1.405) - 1.44
```

This corrects the ~25mm error when fully open, enabling accurate object detection.

### Object Detection Logic

Combined force sensor + calibrated width:
```python
force_detected = (digital_in & 1) != 0  # DI8 force sensor
width = get_width_mm()

if force_detected and width > 12.0:
    return True  # Object detected
else:
    return False  # No object or fully closed
```

## 🎮 System Configuration

### Pick Positions (200mm spacing)
```python
PICK_POSITIONS = [
    (-0.2606, -1.1581, -0.4592),  # Cube 1
    (-0.0606, -1.1581, -0.4592),  # Cube 2 (X + 200mm)
    (0.1394, -1.1581, -0.4592),   # Cube 3 (X + 400mm)
]
```

### Place Position
```python
PLACE_POSITION = (0.7952, -0.8659, -0.3915)
```

### Home Position
```python
HOME_POSITION = (-0.0645, -0.7940, 1.0128)  # Z = 1012.8mm
HOME_ROTATION = (1.709, -1.750, 0.709)
```

### Movement Parameters
```python
VELOCITY = 0.5      # m/s
ACCELERATION = 0.5  # m/s²
```

## 🔑 Key Features

✅ **Calibrated gripper sensing** - Accurate to ±1mm  
✅ **Real-time object detection** - Force + width combined logic  
✅ **Drop detection during transfer** - Parallel monitoring  
✅ **Configurable speed/acceleration** - Adjustable parameters  
✅ **Safety protocols** - Home position, approach offsets  
✅ **Error recovery** - Continues on failed picks  
✅ **Flexible movement** - Linear (movel) + joint (movej) commands  

## 📐 Coordinate System

All positions are in the **robot base frame**:
- **Origin (0,0,0)**: Center of robot mounting flange
- **X-axis**: Forward from robot
- **Y-axis**: Left from robot
- **Z-axis**: Upward from mounting surface

**Positive Z**: Above base  
**Negative Z**: Below base (reaching down to workspace)

## 🛠️ Prerequisites on Robot

### Required UR Programs
Create these `.urp` programs on the teach pendant:
- `grip_open.urp` - Opens gripper to 110mm
- `grip_close.urp` - Closes gripper with force detection

### Gripper Configuration
- Ensure RG2 gripper is properly installed
- Verify analog output is connected to AI2
- Verify force sensor is connected to DI8
- Check voltage mode (5V or 10V) using `check_voltage.py`

## 📝 Usage Example

```python
from pick_and_place import UR10PickAndPlace

# Initialize
controller = UR10PickAndPlace("192.168.8.102")

# Execute pick and place
success = controller.pick_and_place(
    pick_x=-0.2606, pick_y=-1.1581, pick_z=-0.4592,
    place_x=0.7952, place_y=-0.8659, place_z=-0.3915,
    rx=2.987, ry=0.136, rz=-0.005,
    approach_offset=0.100,  # 100mm safety height
    vel=0.5, acc=0.5
)

# Clean up
controller.close()
```

## 🐛 Troubleshooting

**Gripper not opening fully:**
- Increase wait time in `open_gripper()` to 4.0s
- Verify `.urp` programs are set to 110mm target width
- Check gripper calibration on teach pendant

**Object detection not working:**
- Run `check_voltage.py` to verify sensor readings
- Verify calibration points match your gripper
- Check DI8 connection (force sensor)

**Movement timing issues:**
- Increase sleep delays in `move_to()` functions
- Adjust based on robot load and speed settings

## 📚 Documentation

For detailed explanations, see:
- [Development Journey](../docs/ur10_development_journey.md)
- [Calibration Process](../docs/linear_interpolation_explanation.md)
- [Technical Implementation](../docs/technical_implementation_explained.md)

## 🎓 Learning Outcomes

This project demonstrates:
- TCP/IP socket programming
- Binary protocol parsing
- Sensor calibration techniques
- Multi-threaded programming
- Real-time monitoring systems
- Robotic motion control

## 📄 License

Part of Final Year Project 2026 - Middlesex University

## 🔗 Repository

[https://github.com/MyuWaiShin/Final_Year_Project_2026](https://github.com/MyuWaiShin/Final_Year_Project_2026)
