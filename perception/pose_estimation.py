"""
pose_estimation.py
==================
Detects objects (cube / cylinder / arc) with YOLO, fits oriented bounding
boxes to find their rotation, then uses the OAK-D depth + camera intrinsics
to compute the full 3D pose for each object:

    X   – left / right (metres, positive = right of camera centre)
    Y   – up / down    (metres, positive = below camera centre)
    Z   – depth        (metres, away from camera)
    Rz  – gripper rotation (degrees, 0 = horizontal, 90 = vertical)

This is what the robot arm needs to move to the object and pick it up.

Usage:
    python pose_estimation.py

Press 'q' to quit.
"""

import os
os.environ["DEPTHAI_BOOT_TIMEOUT"] = "30000"

import depthai as dai
import cv2
import numpy as np
from pathlib import Path
import onnxruntime as ort
import time
from collections import deque

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = Path(r"C:\Users\myuwa\OneDrive - Middlesex University\FYP Datasets\runs\train\yolov8n\weights\best.onnx")
CONF_THRESHOLD = 0.8
IOU_THRESHOLD  = 0.45

CLASSES = ["cube", "cylinder", "arc"]
COLORS  = {
    "cube":     (0, 255, 0),   # green
    "cylinder": (255, 0, 0),   # blue
    "arc":      (0, 0, 255),   # red
}


# ─────────────────────────────────────────────────────────────────────────────
# Oriented Bounding Box  (same logic as detect_objects.py)
# ─────────────────────────────────────────────────────────────────────────────
class OrientedBoxAnalyzer:
    """Fits a tight rotated bounding box to the object, returns angle + AR."""

    def __init__(self, padding: int = 6, hue_window: int = 20):
        self.padding    = padding
        self.hue_window = hue_window

    def _dominant_hue_mask(self, hsv_crop: np.ndarray) -> np.ndarray:
        h, s, v   = cv2.split(hsv_crop)
        # Only look at pixels with enough colour — ignore grey/wood/black dots
        sat_thresh = 80  # Increased from 60
        coloured   = s > sat_thresh
        if coloured.sum() < 50:
            return np.zeros(h.shape, dtype=np.uint8)

        hues        = h[coloured].flatten()
        hist, _     = np.histogram(hues, bins=180, range=(0, 180))
        hist_smooth = np.convolve(hist, np.ones(7) / 7, mode='same')
        dom_hue     = int(np.argmax(hist_smooth))

        lo = (dom_hue - self.hue_window) % 180
        hi = (dom_hue + self.hue_window) % 180

        if lo <= hi:
            mask = cv2.inRange(hsv_crop,
                               np.array([lo, sat_thresh, 50]), # Increased V min to 50
                               np.array([hi, 255,        255]))
        else:
            m1 = cv2.inRange(hsv_crop,
                             np.array([lo, sat_thresh, 50]),
                             np.array([179, 255,       255]))
            m2 = cv2.inRange(hsv_crop,
                             np.array([0,  sat_thresh, 50]),
                             np.array([hi, 255,        255]))
            mask = cv2.bitwise_or(m1, m2)

        # Morphological clean-up: slightly larger kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def analyze(self, frame: np.ndarray, bbox: tuple) -> dict:
        x1, y1, x2, y2 = bbox
        h_f, w_f = frame.shape[:2]

        box_w   = max(x2 - x1, 1)
        box_h   = max(y2 - y1, 1)
        bbox_ar = round(max(box_w, box_h) / min(box_w, box_h), 2)

        cx1  = max(0, x1 - self.padding);  cx2 = min(w_f - 1, x2 + self.padding)
        cy1  = max(0, y1 - self.padding);  cy2 = min(h_f - 1, y2 + self.padding)
        crop = frame[cy1:cy2, cx1:cx2]

        if crop.size == 0:
            return {'valid': False, 'bbox_aspect_ratio': bbox_ar}

        crop_area = max((cx2 - cx1) * (cy2 - cy1), 1)
        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = self._dominant_hue_mask(hsv)

        if cv2.countNonZero(mask) < 0.05 * crop_area:
            gray   = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            blur   = cv2.GaussianBlur(gray, (5, 5), 0)
            edges  = cv2.Canny(blur, 30, 100)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask   = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {'valid': False, 'bbox_aspect_ratio': bbox_ar}

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 0.04 * crop_area:
            return {'valid': False, 'bbox_aspect_ratio': bbox_ar}

        rect = cv2.minAreaRect(largest)
        (rcx, rcy), (rw, rh), angle = rect

        full_cx = int(rcx + cx1)
        full_cy = int(rcy + cy1)

        if rw < rh:
            angle += 90
            rw, rh = rh, rw

        aspect_ratio  = round(rw / rh, 2) if rh > 0 else 1.0
        gripper_angle = angle % 180

        box_pts = cv2.boxPoints(rect).astype(np.intp) + np.array([cx1, cy1])

        return {
            'valid':             True,
            'center':            (full_cx, full_cy),
            'gripper_angle':     gripper_angle,
            'aspect_ratio':      aspect_ratio,
            'bbox_aspect_ratio': bbox_ar,
            'box_points':        box_pts,
        }

class DetectionSmoother:
    """Smoothes detections across frames to reduce flickering."""
    def __init__(self, max_age=10, min_hits=3, dist_threshold=50):
        self.max_age = max_age
        self.min_hits = min_hits
        self.dist_threshold = dist_threshold
        self.tracked_objects = []

    def update(self, detections):
        for obj in self.tracked_objects:
            obj['age'] += 1

        matched_indices = set()
        for det in detections:
            det_center = self._get_centroid(det['bbox'])
            best_match = None
            min_dist = float('inf')
            
            for i, obj in enumerate(self.tracked_objects):
                if i in matched_indices or det['class_id'] != obj['class_id']:
                    continue
                obj_center = self._get_centroid(obj['bbox'])
                dist = self._dist(det_center, obj_center)
                if dist < self.dist_threshold and dist < min_dist:
                    min_dist = dist
                    best_match = i
            
            if best_match is not None:
                obj = self.tracked_objects[best_match]
                obj['bbox'] = det['bbox']
                obj['confidence'] = det['confidence']
                obj['hits'] += 1
                obj['age'] = 0
                matched_indices.add(best_match)
            else:
                self.tracked_objects.append({
                    'bbox': det['bbox'], 'class_id': det['class_id'],
                    'confidence': det['confidence'], 'hits': 1, 'age': 0
                })

        self.tracked_objects = [obj for obj in self.tracked_objects if obj['age'] < self.max_age]
        stable_detections = []
        for obj in self.tracked_objects:
            if obj['hits'] >= self.min_hits and obj['age'] < self.max_age:
                stable_detections.append({
                    'bbox': obj['bbox'], 'class_id': obj['class_id'],
                    'confidence': obj['confidence'], 'age': obj['age']
                })
        return stable_detections

    def _get_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _dist(self, p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

class ROISelector:
    """Helper to select ROI via mouse."""
    def __init__(self, window_name):
        self.window_name = window_name
        self.roi = None
        self.drawing = False
        self.ix, self.iy = -1, -1

    def select(self, frame):
        self.roi = None
        clone = frame.copy()
        cv2.putText(clone, "Drag to select ROI. Press 'Space' to confirm.", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow(self.window_name, clone)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        while True:
            temp_frame = clone.copy()
            if self.roi:
                x, y, w, h = self.roi
                cv2.rectangle(temp_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow(self.window_name, temp_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '): break
        cv2.setMouseCallback(self.window_name, lambda *args: None)
        return self.roi

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.roi = (min(self.ix, x), min(self.iy, y), abs(self.ix - x), abs(self.iy - y))
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.roi = (min(self.ix, x), min(self.iy, y), abs(self.ix - x), abs(self.iy - y))


# ─────────────────────────────────────────────────────────────────────────────
# Pose Estimator
# ─────────────────────────────────────────────────────────────────────────────
class PoseEstimator:
    """
    Computes the 3D pose of each detected object using:
      - The OAK-D depth frame  → Z (distance in metres)
      - Camera intrinsics      → X, Y (left/right and up/down in metres)
      - Oriented bounding box  → Rz (gripper rotation in degrees)

    Coordinate frame (camera-centred, right-hand):
        X – positive to the RIGHT
        Y – positive DOWNWARDS
        Z – positive AWAY from the camera  (depth)
        Rz – gripper rotation around the camera's Z axis
    """

    # Number of frames to keep for temporal depth smoothing
    _DEPTH_HISTORY = 8

    def __init__(self):
        self.fx  = None
        self.fy  = None
        self.cx0 = None
        self.cy0 = None
        # Rolling depth history per class_id  →  deque of Z_mm values
        self._z_hist: dict[int, deque] = {}

    def load_intrinsics(self, device):
        """Read real camera intrinsics from the OAK-D calibration storage."""
        calib    = device.readCalibration()
        M, w, h  = calib.getDefaultIntrinsics(dai.CameraBoardSocket.RGB)
        # M = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        self.fx  = M[0][0]
        self.fy  = M[1][1]
        self.cx0 = M[0][2]
        self.cy0 = M[1][2]
        print(f"[PoseEstimator] Intrinsics loaded  "
              f"fx={self.fx:.1f}  fy={self.fy:.1f}  "
              f"cx={self.cx0:.1f}  cy={self.cy0:.1f}")

    def compute(self, pixel_cx: int, pixel_cy: int,
                depth_frame: np.ndarray, gripper_angle: float,
                class_id: int = 0) -> dict:
        """
        Returns the full 3D pose for one detection.

        Args:
            pixel_cx / pixel_cy – object centre in image pixels
            depth_frame         – 16-bit depth map in mm from OAK-D
            gripper_angle       – Rz from OrientedBoxAnalyzer (degrees)

        Returns:
            { 'valid': True/False, 'X', 'Y', 'Z' (metres), 'Rz' (degrees) }
        """
        if depth_frame is None:
            return {'valid': False}

        # ── Spatial patch: median over a 5px radius ────────────────────────
        r   = 5
        h_d, w_d = depth_frame.shape
        px1 = max(0, pixel_cx - r);  px2 = min(w_d, pixel_cx + r + 1)
        py1 = max(0, pixel_cy - r);  py2 = min(h_d, pixel_cy + r + 1)
        patch = depth_frame[py1:py2, px1:px2]

        good = patch[patch > 0]
        if good.size == 0:
            return {'valid': False}

        Z_mm_raw = float(np.median(good))

        # ── Temporal smoothing: rolling median over last N frames ─────────
        if class_id not in self._z_hist:
            self._z_hist[class_id] = deque(maxlen=self._DEPTH_HISTORY)
        self._z_hist[class_id].append(Z_mm_raw)
        Z_mm = float(np.median(self._z_hist[class_id]))
        Z    = Z_mm / 1000.0   # convert mm → metres

        # Camera intrinsics – fall back to sensible defaults if not loaded yet
        fx  = self.fx  or 800.0
        fy  = self.fy  or 800.0
        cx0 = self.cx0 or 320.0
        cy0 = self.cy0 or 240.0

        # Back-project: pixel → real-world metres
        X = (pixel_cx - cx0) * Z / fx
        Y = (pixel_cy - cy0) * Z / fy

        return {
            'valid': True,
            'X':  round(X, 4),    # metres, + = right
            'Y':  round(Y, 4),    # metres, + = down
            'Z':  round(Z, 4),    # metres, depth
            'Rz': round(gripper_angle, 1),  # degrees
        }


# ─────────────────────────────────────────────────────────────────────────────
# YOLO Detector  (same logic as detect_objects.py)
# ─────────────────────────────────────────────────────────────────────────────
class YOLODetector:
    def __init__(self, model_path, conf_threshold=0.5, iou_threshold=0.4):
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self.session    = ort.InferenceSession(str(model_path))
        self.input_name = self.session.get_inputs()[0].name
        inp             = self.session.get_inputs()[0].shape
        self.input_h    = inp[2]
        self.input_w    = inp[3]
        print(f"Model loaded: {self.input_w}x{self.input_h}")

    def detect(self, frame):
        inp    = self._preprocess(frame)
        outs   = self.session.run(None, {self.input_name: inp})
        return self._postprocess(outs, frame.shape)

    def _preprocess(self, img):
        img = cv2.resize(img, (self.input_w, self.input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.expand_dims(np.transpose(img, (2, 0, 1)), 0)

    def _postprocess(self, outputs, orig_shape):
        preds      = np.transpose(outputs[0], (0, 2, 1))[0]
        orig_h, orig_w = orig_shape[:2]
        dets = []

        for p in preds:
            xc, yc, w, h = p[:4]
            scores   = p[4:]
            cls_id   = int(np.argmax(scores))
            conf     = float(scores[cls_id])
            if conf < self.conf_threshold:
                continue
            x1 = int((xc - w / 2) * orig_w / self.input_w)
            y1 = int((yc - h / 2) * orig_h / self.input_h)
            x2 = int((xc + w / 2) * orig_w / self.input_w)
            y2 = int((yc + h / 2) * orig_h / self.input_h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w - 1, x2), min(orig_h - 1, y2)
            if x2 > x1 and y2 > y1:
                dets.append({'bbox': (x1, y1, x2, y2),
                             'confidence': conf, 'class_id': cls_id})

        return self._nms(dets)

    def _nms(self, dets):
        dets = sorted(dets, key=lambda d: d['confidence'], reverse=True)
        keep = []
        while dets:
            best = dets.pop(0)
            keep.append(best)
            dets = [d for d in dets if self._iou(best['bbox'], d['bbox']) < self.iou_threshold]
        return keep

    def _iou(self, a, b):
        xi1, yi1 = max(a[0], b[0]), max(a[1], b[1])
        xi2, yi2 = min(a[2], b[2]), min(a[3], b[3])
        inter    = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / ua if ua else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────────────────────
_obb = OrientedBoxAnalyzer()


def draw_pose(frame, detections, depth_frame, pose_estimator):
    """Draw detections, oriented boxes, and full 3D pose on frame."""
    # Use clean copy for OBB analysis to avoid label/box interference
    analysis_frame = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cls_id = det['class_id']
        label  = CLASSES[cls_id] if cls_id < len(CLASSES) else f"cls{cls_id}"
        color  = COLORS.get(label, (255, 255, 255))

        # ── Oriented box ──────────────────────────────────────────────────────
        obb_result = _obb.analyze(analysis_frame, (x1, y1, x2, y2))
        det['obb'] = obb_result

        if obb_result['valid']:
            # Tight rotated box
            cv2.drawContours(frame, [obb_result['box_points']], 0, color, 2)

            cx, cy        = obb_result['center']
            gripper_angle = obb_result['gripper_angle']
            ar            = obb_result['aspect_ratio']

            # Orientation arrow (cyan)
            rad = np.deg2rad(gripper_angle)
            ax  = int(cx + 45 * np.cos(rad))
            ay  = int(cy + 45 * np.sin(rad))
            cv2.arrowedLine(frame, (cx, cy), (ax, ay), (0, 255, 255), 2, tipLength=0.3)

            # ── 3-D Pose ──────────────────────────────────────────────────────
            pose = pose_estimator.compute(cx, cy, depth_frame, gripper_angle, class_id=cls_id)
            det['pose'] = pose

            if pose['valid']:
                # Show pose values on screen
                # Line 1: X Y Z
                pose_txt1 = f"X:{pose['X']:+.3f}m  Y:{pose['Y']:+.3f}m  Z:{pose['Z']:.3f}m"
                # Line 2: Rz and AR
                pose_txt2 = f"Rz:{pose['Rz']:.0f}deg  AR:{ar:.1f}"

                cv2.putText(frame, pose_txt1, (x1, y2 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)
                cv2.putText(frame, pose_txt2, (x1, y2 + 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)
            else:
                cv2.putText(frame, "depth: N/A", (x1, y2 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 200, 200), 1)
        else:
            det['pose'] = None

        # ── YOLO axis-aligned box (faint) ─────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        # ── Label ─────────────────────────────────────────────────────────────
        lbl_txt       = f"{label} {int(det['confidence']*100)}%"
        (lw, lh), _   = cv2.getTextSize(lbl_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw, y1), color, -1)
        cv2.putText(frame, lbl_txt, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Camera pipeline
# ─────────────────────────────────────────────────────────────────────────────
def create_pipeline():
    pipeline = dai.Pipeline()
    pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_4)

    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(640, 480)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(30)

    mono_l = pipeline.create(dai.node.MonoCamera)
    mono_r = pipeline.create(dai.node.MonoCamera)
    stereo  = pipeline.create(dai.node.StereoDepth)

    mono_l.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_l.setBoardSocket(dai.CameraBoardSocket.LEFT)
    mono_r.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_r.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

    mono_l.out.link(stereo.left)
    mono_r.out.link(stereo.right)

    xout_rgb   = pipeline.create(dai.node.XLinkOut); xout_rgb.setStreamName("rgb")
    xout_depth = pipeline.create(dai.node.XLinkOut); xout_depth.setStreamName("depth")
    cam.preview.link(xout_rgb.input)
    stereo.depth.link(xout_depth.input)

    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not MODEL_PATH.exists():
        print(f"ERROR: model not found at {MODEL_PATH.absolute()}")
        print("Run convert_model.py first.")
        return

    detector      = YOLODetector(MODEL_PATH, CONF_THRESHOLD, IOU_THRESHOLD)
    pose_est      = PoseEstimator()
    pipeline      = create_pipeline()

    config = dai.Device.Config()
    config.board.usb.maxSpeed = dai.UsbSpeed.HIGH

    with dai.Device(config) as device:
        device.startPipeline(pipeline)
        print(f"Connected: {device.getMxId()}  USB: {device.getUsbSpeed().name}")

        # Load real intrinsics from the OAK-D
        pose_est.load_intrinsics(device)

        q_rgb   = device.getOutputQueue("rgb",   maxSize=4, blocking=False)
        q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)

        print("\n" + "=" * 60)
        print("Pose Estimation Started!  Press 'q' to quit.")
        print("Press 'c' to Calibrate ROI.")
        print("=" * 60)
 
        roi = None
        roi_sel = ROISelector("Pose Estimation - Cubes / Cylinders / Arcs")
        # dist_threshold=80: detections within 80px of existing track = same object
        # min_hits=5: must appear in 5 consecutive frames before showing
        # max_age=8: drop track after 8 missed frames (was 15 — shorter avoids ghost boxes)
        smoother = DetectionSmoother(max_age=8, min_hits=5, dist_threshold=80)

        fps_count = 0
        fps_time  = time.time()
        fps       = 0.0

        while True:
            frame       = q_rgb.get().getCvFrame()
            depth_frame = q_depth.get().getFrame()
            
            display_frame = frame.copy()

            if roi:
                rx, ry, rw, rh = roi
                roi_f = frame[ry:ry+rh, rx:rx+rw]
                # Run detection on ROI
                raw_dets = detector.detect(roi_f)
                # Map coordinates back
                for d in raw_dets:
                    x1, y1, x2, y2 = d['bbox']
                    d['bbox'] = (x1 + rx, y1 + ry, x2 + rx, y2 + ry)
                
                detections = smoother.update(raw_dets)
                display_frame = draw_pose(display_frame, detections, depth_frame, pose_est)
                cv2.rectangle(display_frame, (rx, ry), (rx+rw, ry+rh), (255, 255, 0), 2)
            else:
                raw_dets = detector.detect(frame)
                detections = smoother.update(raw_dets)
                display_frame = draw_pose(display_frame, detections, depth_frame, pose_est)

            # Console output for robot integration
            for det in detections:
                pose = det.get('pose')
                if pose and pose.get('valid'):
                    cls = CLASSES[det['class_id']] if det['class_id'] < len(CLASSES) else "?"
                    print(
                        f"[POSE] {cls:10s} | "
                        f"X={pose['X']:+.3f}m  Y={pose['Y']:+.3f}m  Z={pose['Z']:.3f}m  "
                        f"Rz={pose['Rz']:.1f}deg"
                    )

            # FPS
            fps_count += 1
            if fps_count >= 10:
                fps       = fps_count / (time.time() - fps_time)
                fps_count = 0
                fps_time  = time.time()

            cv2.putText(display_frame, f"Objects: {len(detections)} | FPS: {fps:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Pose Estimation - Cubes / Cylinders", display_frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('c'):
                roi = roi_sel.select(frame)
                print(f"[INFO] ROI updated: {roi}")

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
