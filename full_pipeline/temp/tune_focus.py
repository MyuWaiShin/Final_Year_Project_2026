"""
tune_focus.py
-------------
Interactively tune the OAK-D Lite manual focus value.

Controls
--------
    UP   / W  →  focus further  (+1)
    DOWN / S  →  focus closer   (-1)
    RIGHT/ D  →  focus further  (+5)
    LEFT / A  →  focus closer   (-5)
    ENTER / SPACE  →  lock this value and print it
    Q             →  quit without saving

The locked value is printed to the terminal so you can copy it into
MANUAL_FOCUS in calibrate_handeye.py.
"""

import sys
from pathlib import Path

import cv2
import depthai as dai

FOCUS_MIN = 0
FOCUS_MAX = 255
focus     = 130   # starting value


def open_camera():
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setVideoSize(1280, 720)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    ctrl_in = pipeline.create(dai.node.XLinkIn)
    ctrl_in.setStreamName("control")
    ctrl_in.out.link(cam.inputControl)

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("video")
    cam.video.link(xout.input)

    device     = dai.Device(pipeline)
    queue      = device.getOutputQueue(name="video", maxSize=4, blocking=False)
    ctrl_queue = device.getInputQueue("control")
    return device, queue, ctrl_queue


def set_focus(ctrl_queue, value):
    ctrl = dai.CameraControl()
    ctrl.setManualFocus(value)
    ctrl_queue.send(ctrl)


def main():
    global focus

    print("Opening camera …")
    device, queue, ctrl_queue = open_camera()
    set_focus(ctrl_queue, focus)
    print("Camera ready.\n")
    print("  UP/W   = +1   DOWN/S = -1")
    print("  RIGHT/D = +5  LEFT/A = -5")
    print("  ENTER or SPACE = lock & print focus value")
    print("  Q = quit\n")

    while True:
        pkt = queue.get()
        frame = pkt.getCvFrame()

        cv2.putText(frame, f"Focus: {focus}",
                    (20, 44), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 220, 255), 2)
        cv2.putText(frame, "UP/W +1  DOWN/S -1  RIGHT/D +5  LEFT/A -5",
                    (20, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, "ENTER/SPACE = lock  |  Q = quit",
                    (20, frame.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

        cv2.imshow("Focus tuner", cv2.resize(frame, (960, 540)))
        key = cv2.waitKey(1) & 0xFF

        changed = False
        if key in (82, ord('w')):    # UP arrow or W
            focus = min(FOCUS_MAX, focus + 1);  changed = True
        elif key in (84, ord('s')): # DOWN arrow or S
            focus = max(FOCUS_MIN, focus - 1);  changed = True
        elif key in (83, ord('d')): # RIGHT arrow or D
            focus = min(FOCUS_MAX, focus + 5);  changed = True
        elif key in (81, ord('a')): # LEFT arrow or A
            focus = max(FOCUS_MIN, focus - 5);  changed = True
        elif key in (13, ord(' ')): # ENTER or SPACE
            print("=" * 50)
            print(f"  Locked focus value: {focus}")
            print(f"  Set MANUAL_FOCUS = {focus} in calibrate_handeye.py")
            print("=" * 50)
            break
        elif key == ord('q'):
            print("Quit — no focus locked.")
            break

        if changed:
            set_focus(ctrl_queue, focus)
            print(f"  Focus: {focus}")

    cv2.destroyAllWindows()
    device.close()


if __name__ == "__main__":
    main()
