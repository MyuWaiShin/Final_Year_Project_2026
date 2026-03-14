import os
import sys

# Set timeout BEFORE importing depthai
os.environ["DEPTHAI_BOOT_TIMEOUT"] = "30000"

# Verify it's set
print(f"DEPTHAI_BOOT_TIMEOUT is set to: {os.environ.get('DEPTHAI_BOOT_TIMEOUT')}")

import depthai as dai
import cv2

print(f"DepthAI Version: {dai.__version__}")

# Don't create pipeline yet - just check for devices first
print("Checking for devices...")
devices = dai.Device.getAllAvailableDevices()
print(f"Found {len(devices)} device(s)")
for d in devices:
    print(f"  - {d}")

if len(devices) == 0:
    print("ERROR: No devices found. Make sure camera is plugged in.")
    sys.exit(1)

print("\nCreating pipeline...")
pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_4)

cam = pipeline.create(dai.node.ColorCamera)
cam.setPreviewSize(640, 480)

xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("rgb")
cam.preview.link(xout.input)

print("Connecting to device...")
config = dai.Device.Config()
config.board.usb.maxSpeed = dai.UsbSpeed.HIGH

with dai.Device(config) as device:
    device.startPipeline(pipeline)
    print("SUCCESS! Connected to:", device.getMxId())
    print("USB Speed:", device.getUsbSpeed().name)
    
    q = device.getOutputQueue("rgb", maxSize=4, blocking=False)
    
    print("Streaming... Press 'q' to quit")
    while True:
        frame = q.get().getCvFrame()
        cv2.imshow("RGB", frame)
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
print("Done!")