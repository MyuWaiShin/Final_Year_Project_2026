"""
12_move_to_cylinder.py
-----------------------
Detects cylinder using YOLO (best.onnx), transforms its position
to robot base frame, moves flange to it.

--depth stereo  -> use OAK-D Lite stereo depth for Z (falls back to fixed if no depth)
--depth fixed   -> use FIXED_Z directly

Press SPACE -> confirm -> move.
"""

import cv2
import depthai as dai
import rtde_receive
import rtde_control
import numpy as np
import argparse
import onnxruntime as ort

# ── ARGS ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--depth", type=str, choices=["stereo", "fixed"],
                    required=True,
                    help="stereo = OAK-D depth | fixed = use FIXED_Z")
args = parser.parse_args()
USE_STEREO = args.depth.lower() == "stereo"

# ── CONFIG ─────────────────────────────────────────────────────────
ROBOT_IP          = "192.168.8.102"
MODEL_PATH        = "data/best.onnx"
CLASSES           = ["cube", "cylinder"]
TARGET_CLASS      = "cylinder"
CONF_THRESHOLD    = 0.5
APPROACH_Z_OFFSET = 0.02
MOVE_SPEED        = 0.02
MOVE_ACCEL        = 0.01
FIXED_Z           = -0.47  # floor Z in base frame — adjust to your setup

# ── Load calibration ───────────────────────────────────────────────
print("Loading calibration files...")
K            = np.load("calibration/camera_matrix.npy")
dist         = np.zeros((4, 1))
T_cam2flange = np.load("calibration/T_cam2flange.npy")
print("  camera_matrix.npy   ok")
print("  T_cam2flange.npy    ok")
print(f"  Depth mode:         {'STEREO (fallback to fixed)' if USE_STEREO else 'FIXED'}\n")

fx  = float(K[0, 0])
fy  = float(K[1, 1])
cx0 = float(K[0, 2])
cy0 = float(K[1, 2])

# ── YOLO ───────────────────────────────────────────────────────────
print(f"Loading YOLO model from {MODEL_PATH}...")
sess       = ort.InferenceSession(MODEL_PATH)
inp        = sess.get_inputs()[0]
MODEL_W    = inp.shape[3]
MODEL_H    = inp.shape[2]
INPUT_NAME = inp.name
print("  YOLO model loaded ok\n")


def run_yolo(frame):
    img   = cv2.resize(frame, (MODEL_W, MODEL_H))
    img   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x     = np.expand_dims(np.transpose(img, (2, 0, 1)), 0)
    out   = sess.run(None, {INPUT_NAME: x})
    preds = np.transpose(out[0], (0, 2, 1))[0]
    oh, ow = frame.shape[:2]
    dets  = []
    for p in preds:
        xc, yc, bw, bh = p[:4]
        scores = p[4:]
        cid    = int(np.argmax(scores))
        conf   = float(scores[cid])
        if conf < CONF_THRESHOLD:
            continue
        if CLASSES[cid] != TARGET_CLASS:
            continue
        x1 = max(0,    int((xc - bw / 2) * ow / MODEL_W))
        y1 = max(0,    int((yc - bh / 2) * oh / MODEL_H))
        x2 = min(ow-1, int((xc + bw / 2) * ow / MODEL_W))
        y2 = min(oh-1, int((yc + bh / 2) * oh / MODEL_H))
        dets.append({"bbox": (x1, y1, x2, y2), "conf": conf})
    dets.sort(key=lambda d: d["conf"], reverse=True)
    return dets


def depth_at(px, py, depth_frame, r=10):
    patch = depth_frame[max(0, py-r):py+r+1, max(0, px-r):px+r+1]
    good  = patch[patch > 0]
    return float(np.median(good)) / 1000.0 if good.size > 0 else None


def pixel_to_cam(px, py, Z):
    X = (px - cx0) * Z / fx
    Y = (py - cy0) * Z / fy
    return np.array([X, Y, Z])


def tcp_to_matrix(tcp_pose):
    R, _ = cv2.Rodrigues(np.array(tcp_pose[3:]))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = tcp_pose[:3]
    return T


# ── Camera pipeline ────────────────────────────────────────────────
print("Starting camera...")
pipeline    = dai.Pipeline()
cam         = pipeline.create(dai.node.Camera).build()
videoOutput = cam.requestOutput((1280, 720), type=dai.ImgFrame.Type.BGR888p)
videoQueue  = videoOutput.createOutputQueue()

depth_queue = None
if USE_STEREO:
    stereo      = pipeline.create(dai.node.StereoDepth).build(autoCreateCameras=True)
    depth_queue = stereo.depth.createOutputQueue()

pipeline.start()
print("Camera started!\n")

# ── Robot ──────────────────────────────────────────────────────────
print("Connecting to robot...")
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
print("Robot connected!\n")

print("=" * 55)
print("CONTROLS:")
print("  SPACE  -> move to detected cylinder")
print("  Q      -> quit")
print("=" * 55)
print()

# ── Main loop ──────────────────────────────────────────────────────
while True:
    imgFrame    = videoQueue.get()
    frame       = imgFrame.getCvFrame()
    depth_frame = None

    if depth_queue is not None:
        df = depth_queue.get()
        if df is not None:
            depth_frame = df.getFrame()

    dets = run_yolo(frame)

    obj_detected = False
    obj_pos_base = None
    current_tcp  = None

    if dets:
        best        = dets[0]
        x1, y1, x2, y2 = best["bbox"]
        px = (x1 + x2) // 2
        py = (y1 + y2) // 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.drawMarker(frame, (px, py), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(frame, f"cylinder {best['conf']:.2f}",
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Get current TCP for transform
        current_tcp   = rtde_r.getActualTCPPose()
        T_flange2base = tcp_to_matrix(current_tcp)

        # Get depth Z
        Z        = None
        Z_source = ""
        if USE_STEREO:
            if depth_frame is not None:
                Z        = depth_at(px, py, depth_frame)
                Z_source = "stereo"
        else:
            T_cam2base    = T_flange2base @ T_cam2flange
            cam_z_in_base = T_cam2base[2, 3]
            Z             = cam_z_in_base - FIXED_Z
            Z_source      = "fixed"

        if Z is None:
            cv2.putText(frame, "no depth — move closer or improve lighting",
                        (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        if Z is not None and Z > 0:
            obj_cam          = pixel_to_cam(px, py, Z)
            T_obj2cam        = np.eye(4)
            T_obj2cam[:3, 3] = obj_cam
            T_obj2base       = T_flange2base @ T_cam2flange @ T_obj2cam
            obj_pos_base     = T_obj2base[:3, 3]
            obj_detected     = True

            bx, by, bz = obj_pos_base
            cv2.putText(frame, f"base: ({bx:.3f}, {by:.3f}, {bz:.3f})m",
                        (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
            cv2.putText(frame, f"Z: {Z:.3f}m ({Z_source})",
                        (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    if obj_detected:
        cv2.putText(frame, "CYLINDER DETECTED -- SPACE to move",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Cylinder not detected",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(frame, "SPACE = move | Q = quit",
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Move to Cylinder", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord(' '):
        if not obj_detected or obj_pos_base is None:
            print("Cylinder not detected or no depth reading — make sure it is visible.")
            continue

        bx, by, bz  = obj_pos_base
        rx, ry, rz  = current_tcp[3], current_tcp[4], current_tcp[5]
        target_pose = [bx, by, bz + APPROACH_Z_OFFSET, rx, ry, rz]

        print("\n" + "=" * 55)
        print(f"  Cylinder in base:   X={bx:.4f}  Y={by:.4f}  Z={bz:.4f}")
        print(f"  Moving to:          X={bx:.4f}  Y={by:.4f}  Z={bz+APPROACH_Z_OFFSET:.4f}")
        print(f"  Orientation kept:   RX={rx:.4f}  RY={ry:.4f}  RZ={rz:.4f}")
        print(f"  Speed: {MOVE_SPEED*100:.0f} cm/s")
        print("=" * 55)
        confirm = input("  Type YES to move, anything else to cancel: ").strip()

        if confirm.upper() != "YES":
            print("  Cancelled.\n")
            continue

        print("  Moving... hand on E-stop.")
        try:
            rtde_c.moveL(target_pose, MOVE_SPEED, MOVE_ACCEL)
            print("  Done!")
        except Exception as e:
            print(f"  Move failed: {e}")

cv2.destroyAllWindows()
pipeline.stop()
rtde_r.disconnect()
rtde_c.stopScript()
print("\nDone.")