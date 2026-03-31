"""
minimal_test4.py — lazy control connection
"""
import time
import depthai as dai
import rtde_receive
import rtde_control

ROBOT_IP = "192.168.8.102"

# Camera
print("Starting camera …")
pipeline = dai.Pipeline()
cam = pipeline.create(dai.node.ColorCamera)
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setVideoSize(1280, 720)
cam.setInterleaved(False)
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("video")
cam.video.link(xout.input)
device = dai.Device(pipeline)
queue = device.getOutputQueue(name="video", maxSize=4, blocking=False)
print("Camera ready.")

# Receive only — no control interface yet
print("Connecting RTDE receive …")
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
print("Connected.")

# Let the initial RTDE blip pass
print("Waiting 5s for RTDE to stabilise …")
for i in range(50):
    queue.tryGet()
    rtde_r.getActualQ()
    time.sleep(0.1)
print("Stable.")

# NOW connect control — after the blip has passed
print("Connecting RTDE control …")
rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
print("Control connected.")

# Poll both for 20 seconds
print("Polling for 20 seconds …")
for i in range(200):
    queue.tryGet()
    q = rtde_r.getActualQ()
    if i % 50 == 0:
        print(f"  {i/10:.0f}s — J0={q[0]:.3f}  control={rtde_c.isConnected()}")
    time.sleep(0.1)

print("PASSED.")
device.close()
rtde_r.disconnect()
rtde_c.stopScript()