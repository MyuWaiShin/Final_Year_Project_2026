import os
os.environ["DEPTHAI_BOOT_TIMEOUT"] = "30000"

import depthai as dai
import cv2
import numpy as np
from pathlib import Path
import onnxruntime as ort
import time

# Class labels for your trained model
CLASSES = ["cube", "cylinder"]

# Colors for each class (BGR format)
COLORS = {
    "cube": (0, 255, 0),      # Green
    "cylinder": (255, 0, 0),  # Blue
}

class RawYOLODetector:
    """YOLO detector using ONNX Runtime with minimal filtering to show raw output"""
    
    def __init__(self, model_path, conf_threshold=0.50): # Lowered confidence to show noise
        self.conf_threshold = conf_threshold
        
        # Load ONNX model
        print(f"Loading model: {model_path}")
        self.session = ort.InferenceSession(str(model_path))
        
        # Get input details
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        
    def preprocess(self, image):
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
        predictions = outputs[0]  # Shape: (1, 7, 3549) - transposed format
        detections = []
        orig_h, orig_w = orig_shape[:2]
        
        # Transpose from (1, 7, 3549) to (1, 3549, 7)
        predictions = np.transpose(predictions, (0, 2, 1))
        
        for pred in predictions[0]:
            x_center, y_center, width, height = pred[:4]
            class_scores = pred[4:]  
            
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])
            
            # Use raw low confidence threshold without NMS to show overlapping boxes
            if confidence < self.conf_threshold:
                continue
            
            x1 = int((x_center - width / 2) * orig_w / self.input_width)
            y1 = int((y_center - height / 2) * orig_h / self.input_height)
            x2 = int((x_center + width / 2) * orig_w / self.input_width)
            y2 = int((y_center + height / 2) * orig_h / self.input_height)
            
            x1 = max(0, min(x1, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            x2 = max(0, min(x2, orig_w - 1))
            y2 = max(0, min(y2, orig_h - 1))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': confidence,
                'class_id': class_id
            })
            
        # Intentionally NOT returning NMS here to show overlapping boxes!
        return detections

    def detect(self, image):
        input_tensor = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        return self.postprocess(outputs, image.shape)

def create_camera_pipeline():
    pipeline = dai.Pipeline()
    pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_4)
    
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(640, 480)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(30)
    
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)
    
    return pipeline

def main():
    model_path = Path(r"C:\Users\myuwa\OneDrive - Middlesex University\FYP Datasets\runs\train\yolov8n\weights\best.onnx")
    
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return
    
    # Lower confidence to show background noise (black dots) and overlapping boxes
    detector = RawYOLODetector(model_path, conf_threshold=0.50)
    
    pipeline = create_camera_pipeline()
    config = dai.Device.Config()
    config.board.usb.maxSpeed = dai.UsbSpeed.HIGH
    
    with dai.Device(config) as device:
        device.startPipeline(pipeline)
        q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        
        print("\n=== RAW Object Detection (For Blog Pictures) ===")
        print("This script has NO smoothing, NO NMS, and NO OBB.")
        print("It uses a very low 20% confidence threshold to show all background noise.")
        print("Press 's' to save a picture. Press 'q' to quit.")
        
        # Ensure a directory exists for saved pictures
        save_dir = Path("blog_pictures")
        save_dir.mkdir(exist_ok=True)
        img_counter = 1
        
        while True:
            in_rgb = q_rgb.get()
            frame = in_rgb.getCvFrame()
            
            # Get raw detections
            detections = detector.detect(frame)
            
            # Draw raw AABB bounding boxes
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                confidence = det['confidence']
                class_id = det['class_id']
                
                label = CLASSES[class_id] if class_id < len(CLASSES) else "unknown"
                color = COLORS.get(label, (255, 255, 255))
                
                # Draw standard rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw text
                text = f"{label} {int(confidence*100)}%"
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1-h-5), (x1+w, y1), color, -1)
                cv2.putText(frame, text, (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            
            cv2.putText(frame, f"RAW View | Press 's' to save", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("RAW YOLO Output", frame)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = int(time.time() * 1000)
                filename = save_dir / f"raw_detection_{timestamp}.png"
                cv2.imwrite(str(filename), frame)
                print(f"Saved: {filename}")
                
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
