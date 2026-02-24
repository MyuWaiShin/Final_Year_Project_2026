import cv2
import numpy as np
import time
import sys
import os
from pathlib import Path
import depthai as dai

# Import YOLODetector and other utilities from detect_objects.py
try:
    from detect_objects import YOLODetector, create_camera_pipeline, CLASSES, COLORS
except ImportError:
    print("Error: Could not import detect_objects.py. Ensure it is in the same directory.")
    sys.exit(1)

# --- Configuration ---
MODEL_PATH = Path(r"C:\Users\myuwa\OneDrive - Middlesex University\Major Project\Datasets\runs\train\yolov8n\weights\best.onnx")
DISPLACEMENT_THRESHOLD = 15.0  # Pixels: Below this is a "Complete Miss"
TOLERANCE_RADIUS = 50.0        # Pixels: Within this is a "Nudge" (near P1)
                               # Beyond this or no detection = Success (Moved/Removed)

class GraspVerifier:
    def __init__(self, detector):
        self.detector = detector
        self.p1 = None           # Pre-grasp centroid (x, y)
        self.p1_class = None     # Class of the object at P1
        self.last_status = "Waiting for P1..."
        self.last_p2 = None      # Position after nudge
        self.current_frame = None
        self.detections = []

    def set_p1(self, detections):
        """Sets P1 to the object closest to the center of the frame, or a specific selection logic."""
        if not detections:
            self.last_status = "Error: No objects detected for P1"
            return False
        
        # For simplicity, we'll pick the object closest to the center
        # In a real scenario, this might be the target sent by the robot controller
        h, w = self.current_frame.shape[:2]
        center_f = (w // 2, h // 2)
        
        # Find closest detection to center
        closest_det = min(detections, key=lambda d: self._dist(self._get_centroid(d['bbox']), center_f))
        
        self.p1 = self._get_centroid(closest_det['bbox'])
        self.p1_class = closest_det['class_id']
        self.last_status = f"P1 Set: {CLASSES[self.p1_class]} at {self.p1}"
        print(f"[VERIFIER] {self.last_status}")
        return True

    def verify_grasp(self, detections):
        """
        Verifies the grasp based on the distance between P1 and the nearest detection of the same class.
        """
        if self.p1 is None:
            self.last_status = "Error: P1 not set!"
            return

        # Find the detection OF THE SAME CLASS closest to P1
        relevant_detections = [d for d in detections if d['class_id'] == self.p1_class]
        
        if not relevant_detections:
            self.last_status = "SUCCESS: Object removed from workspace"
            self.p1 = None # Reset
            return

        # Find closest detection of same class to P1
        closest_det = min(relevant_detections, key=lambda d: self._dist(self._get_centroid(d['bbox']), self.p1))
        p2 = self._get_centroid(closest_det['bbox'])
        distance = self._dist(self.p1, p2)

        if distance < DISPLACEMENT_THRESHOLD:
            self.last_status = f"FAILED: Complete Miss (dist={distance:.1f}px)"
        elif distance < TOLERANCE_RADIUS:
            self.last_status = f"FAILED: Nudge (dist={distance:.1f}px). New P2 recorded."
            self.last_p2 = p2
        else:
            self.last_status = f"SUCCESS: Target moved significantly or removed (dist={distance:.1f}px)"
            self.p1 = None # Reset

        print(f"[VERIFIER] {self.last_status}")

    def _get_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _dist(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def draw_info(self, frame):
        # Draw status text
        color = (255, 255, 255)
        if "SUCCESS" in self.last_status:
            color = (0, 255, 0)
        elif "FAILED" in self.last_status:
            color = (0, 0, 255)
        elif "P1 Set" in self.last_status:
            color = (255, 255, 0)

        cv2.putText(frame, f"Status: {self.last_status}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw P1 if set
        if self.p1:
            cv2.circle(frame, self.p1, 5, (0, 255, 255), -1)
            cv2.circle(frame, self.p1, int(TOLERANCE_RADIUS), (0, 255, 255), 1)
            cv2.putText(frame, "P1", (self.p1[0] + 10, self.p1[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

def main():
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    # Initialize detector
    detector = YOLODetector(MODEL_PATH, conf_threshold=0.8)
    verifier = GraspVerifier(detector)

    print("\nCreating camera pipeline...")
    pipeline = create_camera_pipeline()
    
    config = dai.Device.Config()
    config.board.usb.maxSpeed = dai.UsbSpeed.HIGH
    
    with dai.Device(config) as device:
        device.startPipeline(pipeline)
        
        q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)
        
        print("\n" + "=" * 60)
        print("Grasp Verification Script (Method 1: P1 Position Tracking)")
        print("=" * 60)
        print("Controls:")
        print("  '1' : Capture P1 (Pre-grasp)")
        print("  '2' : Verify Grasp (Post-grasp)")
        print("  'c' : Calibrate ROI (Select Mat)")
        print("  'r' : Reset P1")
        print("  'q' : Quit")
        print("=" * 60)
        
        roi = None
        
        def select_roi(frame):
            r = cv2.selectROI("Grasp Verification - Method 1", frame, fromCenter=False)
            cv2.destroyWindow("ROI selector") # selectROI sometimes creates this
            return r if r[2] > 0 and r[3] > 0 else None

        while True:
            in_rgb = q_rgb.get()
            frame = in_rgb.getCvFrame()
            
            display_frame = frame.copy()
            
            if roi:
                rx, ry, rw, rh = roi
                roi_frame = frame[ry:ry+rh, rx:rx+rw]
                
                # Run detection on ROI
                detections = detector.detect(roi_frame)
                
                # Adjust coords
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    det['bbox'] = (x1 + rx, y1 + ry, x2 + rx, y2 + ry)
                
                cv2.rectangle(display_frame, (rx, ry), (rx+rw, ry+rh), (255, 255, 0), 2)
            else:
                detections = detector.detect(frame)
            
            verifier.current_frame = frame
            verifier.detections = detections

            # Draw detections
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                label = CLASSES[det['class_id']]
                color = COLORS.get(label, (255, 255, 255))
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Draw verifier info
            verifier.draw_info(display_frame)

            # Handle keys
            key = cv2.waitKey(1)
            if key == ord('1'):
                verifier.set_p1(detections)
            elif key == ord('2'):
                verifier.verify_grasp(detections)
            elif key == ord('c'):
                roi = select_roi(frame)
                print(f"[VERIFIER] ROI updated: {roi}")
            elif key == ord('r'):
                verifier.p1 = None
                verifier.last_status = "Waiting for P1..."
                print("[VERIFIER] Reset")
            elif key == ord('q'):
                break
            
            cv2.imshow("Grasp Verification - Method 1", display_frame)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
