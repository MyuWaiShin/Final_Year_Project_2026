import os
os.environ["DEPTHAI_BOOT_TIMEOUT"] = "30000"

import depthai as dai
import cv2
import numpy as np
from pathlib import Path
import onnxruntime as ort

# Class labels for your trained model
CLASSES = ["cube", "cylinder", "arc"]

# Colors for each class (BGR format)
COLORS = {
    "cube": (0, 255, 0),      # Green
    "cylinder": (255, 0, 0),  # Blue
    "arc": (0, 0, 255)        # Red
}


class OrientedBoxAnalyzer:
    """
    Fits a tight ROTATED bounding box to the object inside a YOLO detection crop.

    Strategy:
      1. Crop to the YOLO bbox region.
      2. Find the DOMINANT HUE of the crop automatically (no class label needed).
         This means it works even when YOLO mis-classifies the object.
      3. Build an HSV mask around that dominant hue to isolate the object pixels.
      4. Run cv2.minAreaRect on the largest contour → angle + aspect ratio.
      5. Also report the raw YOLO bbox aspect ratio as a quick sanity check.
    """

    def __init__(self, padding: int = 6, hue_window: int = 20):
        """
        padding    – extra pixels around the YOLO crop.
        hue_window – ±degrees around the dominant hue to include in the mask.
        """
        self.padding    = padding
        self.hue_window = hue_window

    # ------------------------------------------------------------------
    def _dominant_hue_mask(self, hsv_crop: np.ndarray) -> np.ndarray:
        """
        Find the most common hue in the crop (ignoring low-saturation / grey pixels
        so the wood-colour table background is ignored), then return a binary mask
        of all pixels within ±hue_window of that hue.
        """
        h, s, v = cv2.split(hsv_crop)

        # Only look at pixels with enough colour — ignore grey/wood/black dots
        sat_thresh = 80  # Increased from 60 to filter more background noise
        coloured   = s > sat_thresh

        if coloured.sum() < 50:          # basically a grey crop, fall back later
            return np.zeros(h.shape, dtype=np.uint8)

        hues = h[coloured].flatten()

        # Histogram of hues (0-179 in OpenCV)
        hist, _ = np.histogram(hues, bins=180, range=(0, 180))

        # Smooth the histogram so a single dominant peak wins
        hist_smooth = np.convolve(hist, np.ones(7) / 7, mode='same')
        dominant_hue = int(np.argmax(hist_smooth))

        # Build mask: pixels within ±hue_window of dominant hue (wraps around 0/180)
        lo = (dominant_hue - self.hue_window) % 180
        hi = (dominant_hue + self.hue_window) % 180

        if lo <= hi:
            mask = cv2.inRange(hsv_crop,
                               np.array([lo,  sat_thresh, 50]), # Increased V min to 50
                               np.array([hi,  255,        255]))
        else:                            # wraps around (e.g. red 170-10)
            m1 = cv2.inRange(hsv_crop,
                             np.array([lo,  sat_thresh, 50]),
                             np.array([179, 255,        255]))
            m2 = cv2.inRange(hsv_crop,
                             np.array([0,   sat_thresh, 50]),
                             np.array([hi,  255,        255]))
            mask = cv2.bitwise_or(m1, m2)

        # Morphological clean-up: slightly larger kernel to remove small noise dots
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    # ------------------------------------------------------------------
    def analyze(self, frame: np.ndarray, bbox: tuple, class_label: str = ""):
        """
        Analyze a single detection and return oriented box info.

        Args:
            frame:       Full BGR frame.
            bbox:        (x1, y1, x2, y2) from YOLO.
            class_label: Unused — kept for API compatibility.

        Returns dict:
            'valid'             – True if a good contour was found
            'center'            – (cx, cy) in full-frame pixels
            'gripper_angle'     – degrees the gripper should rotate (0=horizontal)
            'aspect_ratio'      – fitted-box width/height from colour contour (>1=elongated)
            'bbox_aspect_ratio' – quick AR from raw YOLO box dimensions
            'box_points'        – 4 corners of the rotated box (full-frame coords)
        """
        x1, y1, x2, y2 = bbox
        h_frame, w_frame = frame.shape[:2]

        # ── Quick aspect ratio from the raw YOLO box ──────────────────────────
        box_w  = max(x2 - x1, 1)
        box_h  = max(y2 - y1, 1)
        bbox_ar = round(max(box_w, box_h) / min(box_w, box_h), 2)

        # ── Crop with padding ─────────────────────────────────────────────────
        cx1  = max(0, x1 - self.padding)
        cy1  = max(0, y1 - self.padding)
        cx2  = min(w_frame - 1, x2 + self.padding)
        cy2  = min(h_frame - 1, y2 + self.padding)
        crop = frame[cy1:cy2, cx1:cx2]

        if crop.size == 0:
            return {'valid': False, 'bbox_aspect_ratio': bbox_ar}

        crop_area = max((cx2 - cx1) * (cy2 - cy1), 1)

        # ── Build colour mask automatically ───────────────────────────────────
        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = self._dominant_hue_mask(hsv)

        # Fallback: if too few pixels matched, use Canny edges
        if cv2.countNonZero(mask) < 0.05 * crop_area:
            gray   = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            blur   = cv2.GaussianBlur(gray, (5, 5), 0)
            edges  = cv2.Canny(blur, 30, 100)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask   = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # ── Largest contour ───────────────────────────────────────────────────
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {'valid': False, 'bbox_aspect_ratio': bbox_ar}

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 0.04 * crop_area:
            return {'valid': False, 'bbox_aspect_ratio': bbox_ar}

        # ── Minimum-area rotated rectangle ────────────────────────────────────
        rect = cv2.minAreaRect(largest)
        (rect_cx, rect_cy), (rw, rh), angle = rect

        # Translate back to full-frame coordinates
        full_cx = int(rect_cx + cx1)
        full_cy = int(rect_cy + cy1)

        # Normalise: width = longer side
        if rw < rh:
            angle += 90
            rw, rh = rh, rw

        aspect_ratio  = round(rw / rh, 2) if rh > 0 else 1.0
        gripper_angle = angle % 180

        box_pts_full = cv2.boxPoints(rect).astype(np.intp) + np.array([cx1, cy1])

        return {
            'valid':             True,
            'center':            (full_cx, full_cy),
            'gripper_angle':     gripper_angle,
            'aspect_ratio':      aspect_ratio,
            'bbox_aspect_ratio': bbox_ar,
            'box_points':        box_pts_full,
        }

class DetectionSmoother:
    """
    Smoothes detections across frames to reduce flickering.
    Persistence: Keeps an object for 'max_age' frames if missed.
    Stability: Only shows an object after 'min_hits' detections.
    """
    def __init__(self, max_age=10, min_hits=3, dist_threshold=50):
        self.max_age = max_age
        self.min_hits = min_hits
        self.dist_threshold = dist_threshold
        self.tracked_objects = [] # List of dicts: {'bbox', 'class_id', 'hits', 'age', 'confidence'}

    def update(self, detections):
        # 1. Increment age of all tracked objects
        for obj in self.tracked_objects:
            obj['age'] += 1

        # 2. Match detections to tracked objects
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
                # Update existing object
                obj = self.tracked_objects[best_match]
                obj['bbox'] = det['bbox']
                obj['confidence'] = det['confidence']
                obj['hits'] += 1
                obj['age'] = 0 # Reset age
                matched_indices.add(best_match)
            else:
                # Add new potential object
                self.tracked_objects.append({
                    'bbox': det['bbox'],
                    'class_id': det['class_id'],
                    'confidence': det['confidence'],
                    'hits': 1,
                    'age': 0
                })

        # 3. Filter objects: remove old ones
        self.tracked_objects = [obj for obj in self.tracked_objects if obj['age'] < self.max_age]

        # 4. Return stable objects
        stable_detections = []
        for obj in self.tracked_objects:
            # Persistence: Show if we have enough hits AND it's not too old
            if obj['hits'] >= self.min_hits and obj['age'] < self.max_age:
                stable_detections.append({
                    'bbox': obj['bbox'],
                    'class_id': obj['class_id'],
                    'confidence': obj['confidence'],
                    'age': obj['age'] # Useful for fading or indicating stale data
                })
        
        return stable_detections

    def _get_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _dist(self, p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5


class YOLODetector:
    """YOLO detector using ONNX Runtime"""

    
    def __init__(self, model_path, conf_threshold=0.8, iou_threshold=0.45):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Load ONNX model
        print(f"Loading model: {model_path}")
        self.session = ort.InferenceSession(str(model_path))
        
        # Get input details
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        
        print(f"Model loaded! Input size: {self.input_width}x{self.input_height}")
    
    def preprocess(self, image):
        """Preprocess image for YOLO"""
        # Resize to model input size
        input_img = cv2.resize(image, (self.input_width, self.input_height))
        
        # Convert BGR to RGB
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and transpose to CHW format
        input_img = input_img.astype(np.float32) / 255.0
        input_img = np.transpose(input_img, (2, 0, 1))
        
        # Add batch dimension
        input_img = np.expand_dims(input_img, axis=0)
        
        return input_img
    
    def postprocess(self, outputs, orig_shape):
        """Post-process YOLO outputs"""
        predictions = outputs[0]  # Shape: (1, 7, 3549) - transposed format
        
        detections = []
        orig_h, orig_w = orig_shape[:2]
        
        # Debug: Print shape on first run
        if not hasattr(self, '_debug_printed'):
            print(f"[DEBUG] Model output shape: {predictions.shape}")
            print(f"[DEBUG] Transposed format detected")
            self._debug_printed = True
        
        # Transpose from (1, 7, 3549) to (1, 3549, 7)
        predictions = np.transpose(predictions, (0, 2, 1))
        
        for pred in predictions[0]:
            # YOLOv8 format: [x_center, y_center, width, height, class0_conf, class1_conf, class2_conf]
            x_center, y_center, width, height = pred[:4]
            class_scores = pred[4:]  # All class confidences
            
            # Get best class
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])
            
            if confidence < self.conf_threshold:
                continue
            
            # Convert from model coordinates to pixel coordinates
            x1 = int((x_center - width / 2) * orig_w / self.input_width)
            y1 = int((y_center - height / 2) * orig_h / self.input_height)
            x2 = int((x_center + width / 2) * orig_w / self.input_width)
            y2 = int((y_center + height / 2) * orig_h / self.input_height)
            
            # Clamp to image bounds
            x1 = max(0, min(x1, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            x2 = max(0, min(x2, orig_w - 1))
            y2 = max(0, min(y2, orig_h - 1))
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue
            
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': confidence,
                'class_id': class_id
            })
        
        # Apply NMS (Non-Maximum Suppression)
        detections = self.apply_nms(detections)
        
        return detections
    
    def apply_nms(self, detections):
        """Apply Non-Maximum Suppression"""
        if len(detections) == 0:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while len(detections) > 0:
            best = detections[0]
            keep.append(best)
            detections = detections[1:]
            
            # Remove overlapping boxes
            detections = [
                det for det in detections
                if self.iou(best['bbox'], det['bbox']) < self.iou_threshold
            ]
        
        return keep
    
    def iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def detect(self, image):
        """Run detection on image"""
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # Postprocess
        detections = self.postprocess(outputs, image.shape)
        
        return detections

def create_camera_pipeline():
    """Create DepthAI pipeline for camera and depth"""
    pipeline = dai.Pipeline()
    pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_4)
    
    # RGB Camera
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(640, 480)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(30)
    
    # Stereo Depth
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
    
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)
    
    # Outputs
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)
    
    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)
    
    return pipeline

# Shared analyzer instance
_obb_analyzer = OrientedBoxAnalyzer(padding=8)


def draw_detections(frame, detections, depth_frame):
    """Draw bounding boxes, oriented boxes, and depth info on frame."""
    # Use a copy of the frame for OBB analysis to avoid interference from previous annotations
    analysis_frame = frame.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        confidence = det['confidence']
        class_id = det['class_id']

        # Get class label and color
        if class_id < len(CLASSES):
            label = CLASSES[class_id]
            color = COLORS.get(label, (255, 255, 255))
        else:
            label = f"Class {class_id}"
            color = (255, 255, 255)

        # ── Oriented bounding box analysis ──────────────────────────────────
        obb = _obb_analyzer.analyze(analysis_frame, (x1, y1, x2, y2), class_label=label)

        if obb['valid']:
            # Draw the tight rotated box in the object's colour
            cv2.drawContours(frame, [obb['box_points']], 0, color, 2)

            cx, cy = obb['center']
            angle  = obb['gripper_angle']
            ar_fit = obb['aspect_ratio']          # from colour-contour
            ar_box = obb['bbox_aspect_ratio']     # from raw YOLO box

            # Orientation arrow through the object centre (cyan)
            arrow_len = 45
            rad = np.deg2rad(angle)
            ax  = int(cx + arrow_len * np.cos(rad))
            ay  = int(cy + arrow_len * np.sin(rad))
            cv2.arrowedLine(frame, (cx, cy), (ax, ay), (0, 255, 255), 2, tipLength=0.3)

            # Show aspect ratio and gripper angle below the box
            orient_text = f"AR:{ar_fit:.1f}(box:{ar_box:.1f}) Ang:{angle:.0f}deg"
            cv2.putText(frame, orient_text, (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

            det['oriented_box'] = obb
        else:
            det['oriented_box'] = None

        # ── Depth at detection centre ────────────────────────────────────────
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        if (depth_frame is not None
                and 0 <= center_y < depth_frame.shape[0]
                and 0 <= center_x < depth_frame.shape[1]):
            depth_mm = depth_frame[center_y, center_x]
            depth_m = depth_mm / 1000.0
            depth_text = f"{depth_m:.2f}m"
        else:
            depth_text = "N/A"

        # ── YOLO axis-aligned box (thin, dashed appearance via alpha) ───────
        # Draw the original YOLO bbox more faintly so the OBB stands out
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # ── Label ────────────────────────────────────────────────────────────
        label_text = f"{label} {int(confidence * 100)}% | {depth_text}"
        (lw, lh), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - lh - 10), (x1 + lw, y1), color, -1)
        cv2.putText(frame, label_text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame

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
        cv2.putText(clone, "Drag to select ROI (Mat area). Press 'Space' to confirm.", 
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
            if key == ord(' '):
                break
        
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

def main():
    # Model path
    model_path = Path(r"C:\Users\myuwa\OneDrive - Middlesex University\Major Project\Datasets\runs\train\yolov8n\weights\best.onnx")
    
    if not model_path.exists():
        print("=" * 60)
        print("ERROR: ONNX model not found!")
        print("=" * 60)
        print(f"Expected path: {model_path.absolute()}")
        print("\nPlease run convert_model.py first to create the ONNX model.")
        print("=" * 60)
        return
    
    # Initialize detector
    detector = YOLODetector(model_path, conf_threshold=0.8)
    
    print("\nCreating camera pipeline...")
    pipeline = create_camera_pipeline()
    
    print("Connecting to device...")
    config = dai.Device.Config()
    config.board.usb.maxSpeed = dai.UsbSpeed.HIGH
    
    with dai.Device(config) as device:
        device.startPipeline(pipeline)
        print(f"Connected to: {device.getMxId()}")
        print(f"USB Speed: {device.getUsbSpeed().name}")
        
        # Get output queues
        q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)
        
        print("\n" + "=" * 60)
        print("Live Object Detection Started!")
        print("=" * 60)
        print("Detecting: Cubes, Cylinders, Arcs")
        print("Press 'q' to quit")
        print("  'c' : Calibrate ROI (Select Mat)")
        print("=" * 60)
        
        roi = None
        roi_selector = ROISelector("Object Detection - Cubes, Cylinders, Arcs")
        smoother = DetectionSmoother(max_age=15, min_hits=3)

        import time
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        
        while True:
            # Get frames
            in_rgb = q_rgb.get()
            in_depth = q_depth.get()
            
            frame = in_rgb.getCvFrame()
            depth_frame = in_depth.getFrame()
            
            display_frame = frame.copy()

            if roi:
                x, y, w, h = roi
                roi_frame = frame[y:y+h, x:x+w]
                roi_depth = depth_frame[y:y+h, x:x+w] if depth_frame is not None else None
                
                # Run detection on ROI
                raw_detections = detector.detect(roi_frame)
                
                # Adjust detection coordinates back to full frame
                for det in raw_detections:
                    x1, y1, x2, y2 = det['bbox']
                    det['bbox'] = (x1 + x, y1 + y, x2 + x, y2 + y)

                # Smooth detections
                detections = smoother.update(raw_detections)

                # Draw detections on display frame
                display_frame = draw_detections(display_frame, detections, depth_frame)
                
                # Draw ROI border
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
            else:
                # Run detection on full frame
                raw_detections = detector.detect(frame)
                detections = smoother.update(raw_detections)
                display_frame = draw_detections(display_frame, detections, depth_frame)

            # ── Print orientation info for robot integration ──────────────────
            for det in detections:
                obb = det.get('oriented_box')
                if obb and obb['valid']:
                    cls = CLASSES[det['class_id']] if det['class_id'] < len(CLASSES) else str(det['class_id'])
                    cx, cy = obb['center']
                    print(
                        f"[GRIPPER] {cls:10s} | center=({cx:4d},{cy:4d}) | "
                        f"gripper_angle={obb['gripper_angle']:5.1f}° | "
                        f"aspect_ratio={obb['aspect_ratio']:.2f}"
                    )

            # Calculate FPS
            fps_counter += 1
            if fps_counter >= 10:
                current_fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
            
            # Display info
            cv2.putText(frame, f"Objects: {len(detections)} | FPS: {current_fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow("Object Detection - Cubes, Cylinders", display_frame)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('c'):
                roi = roi_selector.select(frame)
                print(f"[INFO] ROI updated: {roi}")
    
    cv2.destroyAllWindows()
    print("\nDone!")

if __name__ == "__main__":
    main()
