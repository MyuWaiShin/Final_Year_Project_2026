"""
verify.py
---------
Stage 4 of the full pipeline.

Behaviour
---------
Picks up after grasp.py — gripper is closed at pick Z holding (maybe) the object.

This stage:
  1. Lifts LIFT_HEIGHT_M straight up (back to hover Z)
  2. Tilts wrist1 (J3) by WRIST1_TILT_RAD so the camera can see into the gripper jaws
  3. Grabs a frame from the OAK-D camera
  4. Runs two classifiers on the frame:
       YOLO v2   (empty_holding_cls_v2)             — trained on this specific gripper
       CLIP      (ViT-B/32 + LinearProbe)           — general-purpose visual check
  5. Fuses results via weighted average of "holding" probabilities:
       score = YOLO_WEIGHT * p_holding_yolo + CLIP_WEIGHT * p_holding_clip
       score >= FUSION_THRESHOLD  →  "holding"
       score <  FUSION_THRESHOLD  →  "empty"
  6. Returns result dict to main.py

Fusion rules (with default weights YOLO=0.7, CLIP=0.3, threshold=0.5):
  YOLO=holding, CLIP=holding  →  pass   (score ~ 1.0)
  YOLO=holding, CLIP=empty    →  pass   (score ~ 0.66)
  YOLO=empty,   CLIP=holding  →  fail   (score ~ 0.34)
  YOLO=empty,   CLIP=empty    →  fail   (score ~ 0.0)

Return values
-------------
  {"result": "holding", "score": float, "yolo_conf": float, "clip_conf": float}
  {"result": "empty",   "score": float, "yolo_conf": float, "clip_conf": float}

Motion notes
------------
All robot motion is sent as raw URScript over port 30002.
Robot state is read from the same port's secondary-client stream.
No RTDE used (see pipeline_dev/RTDE_debug_log.md).
"""

import os
import pickle
import signal
import socket
import struct
import threading
import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np
import torch
import clip as openai_clip
from PIL import Image
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
YOLO_MODEL  = SCRIPT_DIR / "yolo_binary_classifiers" / "empty_holding_v1" / "weights" / "best.pt"
CLIP_PROBE  = SCRIPT_DIR.parent / "CLIP_post_grab" / "clip_probe.pkl"

# ── Robot ─────────────────────────────────────────────────────────────────────
ROBOT_IP      = "192.168.8.102"
ROBOT_PORT    = 30002

# ── Motion ────────────────────────────────────────────────────────────────────
LIFT_HEIGHT_M    = 0.186     # metres — lift straight up (matches DESCEND_OFFSET)
WRIST1_TILT_RAD  = -1.2     # rad — J3 tilt to expose gripper to camera (~69 deg)
MOVE_SPEED       = 0.04     # m/s
MOVE_ACCEL       = 0.01     # m/s²
JOINT_SPEED      = 0.3      # rad/s — wrist tilt move
JOINT_ACCEL      = 0.2      # rad/s²
XYZ_TOL_M        = 0.003    # arrival tolerance
JOINT_TOL_RAD    = 0.01

# ── Classifier settings ───────────────────────────────────────────────────────
YOLO_WEIGHT      = 0.7      # weight for YOLO "holding" probability
CLIP_WEIGHT      = 0.3      # weight for CLIP "holding" probability
FUSION_THRESHOLD = 0.5      # fused score >= this → "holding"

# CLIP crop region (centre bottom of frame — gripper in view)
CROP_W = 1400
CROP_H = 600


# ── Robot state reader ────────────────────────────────────────────────────────
class RobotStateReader(threading.Thread):
    """
    Reads UR secondary-client stream (port 30002) in a background thread.
    Parses:
      Type 1  Joint Data      → joint positions q[0..5]
      Type 4  Cartesian Info  → TCP pose [x,y,z,rx,ry,rz]
    """
    def __init__(self, ip: str, port: int = ROBOT_PORT):
        super().__init__(daemon=True)
        self.ip   = ip
        self.port = port
        self._lock      = threading.Lock()
        self._stop_evt  = threading.Event()
        self._ready_evt = threading.Event()
        self._tcp_pose  = [0.0] * 6
        self._joints    = [0.0] * 6
        self._protective_stop = False

    def run(self):
        while not self._stop_evt.is_set():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(3.0)
                    s.connect((self.ip, self.port))
                    while not self._stop_evt.is_set():
                        hdr = self._recvall(s, 4)
                        if hdr is None:
                            break
                        plen = struct.unpack("!I", hdr)[0]
                        body = self._recvall(s, plen - 4)
                        if body is None:
                            break
                        self._parse_subpackets(body[1:])
            except Exception:
                time.sleep(0.5)

    @staticmethod
    def _recvall(s, n: int):
        buf = b""
        while len(buf) < n:
            chunk = s.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def _parse_subpackets(self, data: bytes):
        off = 0
        got = False
        while off < len(data):
            if off + 5 > len(data):
                break
            ps = struct.unpack("!I", data[off:off+4])[0]
            if ps < 5 or off + ps > len(data):
                break
            pt = data[off + 4]
            # Type 0: Robot Mode Data — isProtectiveStopped at off+17
            if pt == 0 and ps >= 18:
                with self._lock:
                    self._protective_stop = bool(data[off + 17])
            elif pt == 1 and ps >= 251:
                joints = []
                for j in range(6):
                    base = off + 5 + j * 41
                    joints.append(struct.unpack("!d", data[base:base+8])[0])
                with self._lock:
                    self._joints = joints
                got = True
            elif pt == 4 and ps >= 53:
                pose = list(struct.unpack("!6d", data[off+5:off+53]))
                with self._lock:
                    self._tcp_pose = pose
                got = True
            off += ps
        if got:
            self._ready_evt.set()

    def wait_ready(self, timeout: float = 5.0) -> bool:
        return self._ready_evt.wait(timeout=timeout)

    def get_tcp_pose(self) -> list:
        with self._lock:
            return list(self._tcp_pose)

    def get_joint_positions(self) -> list:
        with self._lock:
            return list(self._joints)

    def is_protective_stop(self) -> bool:
        """True when the robot has entered a protective stop."""
        with self._lock:
            return self._protective_stop

    def stop(self):
        self._stop_evt.set()


# ── URScript sender ───────────────────────────────────────────────────────────
class URScriptSender:
    def __init__(self, ip: str, port: int = ROBOT_PORT):
        self.ip    = ip
        self.port  = port
        self._lock = threading.Lock()
        self._sock = self._connect()
        threading.Thread(target=self._drain, daemon=True).start()

    def _connect(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5.0)
        s.connect((self.ip, self.port))
        return s

    def _drain(self):
        while True:
            try:
                self._sock.recv(4096)
            except Exception:
                time.sleep(0.01)

    def send(self, script: str):
        payload = (script.strip() + "\n").encode()
        with self._lock:
            try:
                self._sock.sendall(payload)
            except Exception:
                try:
                    self._sock = self._connect()
                    self._sock.sendall(payload)
                except Exception as e:
                    print(f"  [URScript] Send failed: {e}")

    def close(self):
        try:
            self._sock.close()
        except Exception:
            pass


# ── Motion helpers ────────────────────────────────────────────────────────────
def movel(sender, state, x, y, z, rx, ry, rz,
          vel=MOVE_SPEED, acc=MOVE_ACCEL, tol=XYZ_TOL_M, timeout=30.0):
    cur = state.get_tcp_pose()
    if ((cur[0]-x)**2 + (cur[1]-y)**2 + (cur[2]-z)**2) ** 0.5 < tol:
        return
    sender.send(f"movel(p[{x:.6f},{y:.6f},{z:.6f},{rx:.6f},{ry:.6f},{rz:.6f}],a={acc:.4f},v={vel:.4f})")
    deadline = time.time() + timeout
    while time.time() < deadline:
        cur = state.get_tcp_pose()
        if ((cur[0]-x)**2 + (cur[1]-y)**2 + (cur[2]-z)**2) ** 0.5 < tol:
            return
        time.sleep(0.01)
    if state.is_protective_stop():
        raise RuntimeError(
            "[PROTECTIVE STOP] Robot stopped during movel — "
            "clear the stop on the pendant before continuing."
        )
    print("  [movel] Warning: timeout before arrival tolerance reached.")


def movej_joints(sender, state, joints,
                 vel=JOINT_SPEED, acc=JOINT_ACCEL, tol=JOINT_TOL_RAD, timeout=15.0):
    cur = state.get_joint_positions()
    if max(abs(c - t) for c, t in zip(cur, joints)) < tol:
        return
    q_str = ",".join(f"{j:.6f}" for j in joints)
    sender.send(f"movej([{q_str}],a={acc:.4f},v={vel:.4f})")
    deadline = time.time() + timeout
    while time.time() < deadline:
        cur = state.get_joint_positions()
        if max(abs(c - t) for c, t in zip(cur, joints)) < tol:
            return
        time.sleep(0.01)
    if state.is_protective_stop():
        raise RuntimeError(
            "[PROTECTIVE STOP] Robot stopped during movej — "
            "clear the stop on the pendant before continuing."
        )
    print("  [movej] Warning: timeout before arrival tolerance reached.")


# ── Camera ────────────────────────────────────────────────────────────────────
def open_camera():
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setVideoSize(1920, 1080)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("video")
    cam.video.link(xout.input)
    device = dai.Device(pipeline)
    queue  = device.getOutputQueue(name="video", maxSize=1, blocking=False)
    return device, queue


def grab_frame(queue, warmup_frames: int = 5) -> np.ndarray:
    """Discard a few frames to let the camera settle, then return a fresh one."""
    frame = None
    for _ in range(warmup_frames + 1):
        pkt = queue.get()
        frame = pkt.getCvFrame()
    return frame


# ── Classifiers ───────────────────────────────────────────────────────────────
def run_yolo(model: YOLO, frame: np.ndarray) -> tuple:
    """
    Returns (p_holding, p_empty) — raw class probabilities.
    class names: model.names  (typically {0:'empty', 1:'holding'} or reversed)
    """
    results = model(frame, imgsz=224, verbose=False)
    r = results[0]
    # Build a name→prob mapping
    probs = {r.names[i]: float(r.probs.data[i]) for i in range(len(r.names))}
    p_holding = probs.get("holding", 0.0)
    p_empty   = probs.get("empty",   0.0)
    return p_holding, p_empty


def run_clip(model, preprocess, clf, frame: np.ndarray, device: str) -> tuple:
    """
    Returns (p_holding, p_empty) from the CLIP linear probe.
    Crops the bottom-centre of the frame (where the gripper sits).
    """
    h, w  = frame.shape[:2]
    cx, cy = w // 2, h - (CROP_H // 2) - 10
    x1 = max(0, cx - CROP_W // 2);  y1 = max(0, cy - CROP_H // 2)
    x2 = min(w, x1 + CROP_W);       y2 = min(h, y1 + CROP_H)
    crop    = frame[y1:y2, x1:x2].copy()
    pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    tensor  = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    probs     = clf.predict_proba(feat.cpu().numpy())[0]
    # LinearProbe was trained with classes {0: "Empty", 1: "Holding"}
    p_empty   = float(probs[0])
    p_holding = float(probs[1])
    return p_holding, p_empty


# ── Model loader (callable from main.py at startup) ──────────────────────────
def load_models() -> dict:
    """
    Load YOLO + CLIP models and return them in a dict.
    Call this once at pipeline startup (e.g. in a background thread) so that
    verify() has zero loading latency when it runs.
    """
    if not YOLO_MODEL.exists():
        raise FileNotFoundError(f"YOLO weights not found: {YOLO_MODEL}")
    yolo = YOLO(str(YOLO_MODEL), task="classify")

    if not CLIP_PROBE.exists():
        raise FileNotFoundError(f"CLIP probe not found: {CLIP_PROBE}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_pre = openai_clip.load("ViT-B/32", device=dev)
    with open(CLIP_PROBE, "rb") as f:
        clip_clf = pickle.load(f)

    return {
        "yolo":         yolo,
        "clip_model":   clip_model,
        "clip_pre":     clip_pre,
        "clip_clf":     clip_clf,
        "clip_device":  dev,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main(models: dict = None) -> dict:
    """
    Verify stage: lift, tilt wrist, classify with YOLO + CLIP.

    Robot must already be at pick Z with gripper closed (grasp.py left it there).
    Reads current TCP and joint positions from the robot — no args needed.

    Returns
    -------
    dict with key "result":
        "holding"  — fused score >= FUSION_THRESHOLD
        "empty"    — fused score <  FUSION_THRESHOLD
      plus "score", "yolo_conf", "clip_conf"
    """
    signal.signal(signal.SIGINT, lambda *_: os._exit(1))

    print("\n" + "=" * 60)
    print("  STAGE 4 — VERIFY (YOLO + CLIP dual check)")
    print("=" * 60)
    print(f"  YOLO weight : {YOLO_WEIGHT}   CLIP weight : {CLIP_WEIGHT}")
    print(f"  Fusion threshold : {FUSION_THRESHOLD}")
    print(f"  Lift : {LIFT_HEIGHT_M*1000:.0f} mm  |  Wrist1 tilt : {WRIST1_TILT_RAD:.2f} rad")
    print("=" * 60 + "\n")

    # ── Connect ──────────────────────────────────────────────────────────────
    print("Connecting to robot ...")
    state  = RobotStateReader(ROBOT_IP)
    sender = URScriptSender(ROBOT_IP)
    state.start()
    if not state.wait_ready(timeout=5.0):
        raise RuntimeError("Robot state reader timed out — is robot reachable at " + ROBOT_IP + "?")
    print("Robot connected!\n")

    # ── Read current pose and joints ─────────────────────────────────────────
    tcp    = state.get_tcp_pose()
    joints = state.get_joint_positions()
    px, py, pz, prx, pry, prz = tcp
    lift_z = pz + LIFT_HEIGHT_M

    print(f"  Current TCP  : X={px:.4f}  Y={py:.4f}  Z={pz:.4f}")
    print(f"  Lift target  : Z={lift_z:.4f}  (+{LIFT_HEIGHT_M*1000:.0f} mm)")
    print(f"  J3 current   : {joints[3]:.4f} rad  →  tilt to {joints[3]+WRIST1_TILT_RAD:.4f} rad\n")

    # ── Load models (or use pre-loaded ones passed in from main.py) ──────────
    if models is not None:
        print("  [bg] Using pre-loaded models.\n")
        yolo_model      = models["yolo"]
        clip_model      = models["clip_model"]
        clip_preprocess = models["clip_pre"]
        clip_clf        = models["clip_clf"]
        clip_device     = models["clip_device"]
    else:
        # Fall back to background loading while the robot moves
        _models   = {}
        _load_err = [None]
        _loaded   = threading.Event()

        def _load_models():
            try:
                _models.update(load_models())
                print(f"  [bg] Models loaded.")
            except Exception as e:
                _load_err[0] = e
            finally:
                _loaded.set()

        threading.Thread(target=_load_models, daemon=True).start()
        print("  [bg] Model loading started in background ...\n")

    # ── 1. Lift straight up ───────────────────────────────────────────────────
    print("  [1/3] Lifting ...")
    movel(sender, state, px, py, lift_z, prx, pry, prz)
    print(f"  At hover Z={lift_z:.4f}")

    # ── 2. Tilt wrist1 (J3) ──────────────────────────────────────────────────
    print(f"\n  [2/3] Tilting wrist1 by {WRIST1_TILT_RAD:.2f} rad ...")
    tilt_joints = list(joints)
    tilt_joints[3] += WRIST1_TILT_RAD
    movej_joints(sender, state, tilt_joints)
    print("  Wrist tilted.")

    # ── Wait for models if we started a background loader ─────────────────────
    if models is None:
        if not _loaded.is_set():
            print("  Waiting for models to finish loading ...")
        _loaded.wait(timeout=60.0)
        if _load_err[0] is not None:
            raise _load_err[0]
        yolo_model      = _models["yolo"]
        clip_model      = _models["clip_model"]
        clip_preprocess = _models["clip_pre"]
        clip_clf        = _models["clip_clf"]
        clip_device     = _models["clip_device"]

    print("  Models ready.")

    # ── 3. Open camera and grab frame ─────────────────────────────────────────
    print("\n  [3/3] Opening camera ...")
    cam_device, queue = open_camera()
    print("  Warming up camera ...")
    frame = grab_frame(queue, warmup_frames=8)
    print("  Frame captured.")

    # ── 5. Run classifiers ────────────────────────────────────────────────────
    print("\n  Running YOLO ...")
    yolo_p_holding, yolo_p_empty = run_yolo(yolo_model, frame)
    print(f"  YOLO   →  holding={yolo_p_holding:.3f}  empty={yolo_p_empty:.3f}")

    print("  Running CLIP ...")
    clip_p_holding, clip_p_empty = run_clip(clip_model, clip_preprocess, clip_clf, frame, clip_device)
    print(f"  CLIP   →  holding={clip_p_holding:.3f}  empty={clip_p_empty:.3f}")

    # ── 6. Fuse ───────────────────────────────────────────────────────────────
    score  = YOLO_WEIGHT * yolo_p_holding + CLIP_WEIGHT * clip_p_holding
    result = "holding" if score >= FUSION_THRESHOLD else "empty"

    print(f"\n  Fused score : {score:.3f}  (threshold {FUSION_THRESHOLD})")
    print(f"  Result      : {result.upper()}")

    # ── Show frame with overlay ───────────────────────────────────────────────
    display = cv2.resize(frame, (960, 540))
    col = (0, 220, 0) if result == "holding" else (0, 0, 220)
    cv2.putText(display,
                f"YOLO: {yolo_p_holding*100:.0f}%  CLIP: {clip_p_holding*100:.0f}%  "
                f"Score: {score:.2f}  →  {result.upper()}",
                (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2)
    # Draw CLIP crop region
    h, w = frame.shape[:2]
    cx, cy = w // 2, h - (CROP_H // 2) - 10
    x1c = max(0, cx - CROP_W // 2);  y1c = max(0, cy - CROP_H // 2)
    x2c = min(w, x1c + CROP_W);      y2c = min(h, y1c + CROP_H)
    sx, sy = 960/w, 540/h
    cv2.rectangle(display,
                  (int(x1c*sx), int(y1c*sy)),
                  (int(x2c*sx), int(y2c*sy)),
                  (0, 140, 255), 2)
    cv2.imshow("verify — Stage 4", display)
    cv2.waitKey(2000)   # show for 2 s then continue
    cv2.destroyAllWindows()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  VERIFY STAGE COMPLETE")
    print(f"  YOLO holding prob : {yolo_p_holding*100:.1f}%")
    print(f"  CLIP holding prob : {clip_p_holding*100:.1f}%")
    print(f"  Fused score       : {score:.3f}")
    print(f"  Result            : {result.upper()}")
    if result == "holding":
        print("  -> Proceed to Stage 5 (transit / place)")
    else:
        print("  -> Grasp failed — abort or retry")
    print("=" * 60 + "\n")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cam_device.close()
    state.stop()
    sender.close()

    return {
        "result":    result,
        "score":     round(score, 4),
        "yolo_conf": round(yolo_p_holding, 4),
        "clip_conf": round(clip_p_holding, 4),
    }


if __name__ == "__main__":
    result = main()
    print(f"[verify] Returned: {result}")
