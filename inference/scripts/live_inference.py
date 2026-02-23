"""
Live Inference Script for YOLO Object Detection
Captures live camera feed and performs real-time object detection using trained YOLO models.
"""

import cv2
from ultralytics import YOLO
import os
from pathlib import Path

# ============================================================================
# CONFIGURATION - Change this path to test different models
# ============================================================================
MODEL_PATH = "../../Models/yolo26n_100epochs/weights/best.pt"  # Relative to this script
# Alternative models you can try:
# MODEL_PATH = "../../Models/yolov8n_50epochs/weights/best.pt"
# MODEL_PATH = "../../Models/yolo26n_100epochs/weights/best.pt"
# MODEL_PATH = "../../Models/yolo26n_50epochs/weights/best.pt"
# ============================================================================

# Camera settings
CAMERA_INDEX = 0  # Default camera (0), change if using external camera
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for detections
WINDOW_NAME = "Live Object Detection"

def get_model_path():
    """Convert relative model path to absolute path."""
    script_dir = Path(__file__).parent
    model_path = (script_dir / MODEL_PATH).resolve()
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    return str(model_path)

def main():
    """Main function to run live inference."""
    print("=" * 60)
    print("YOLO Live Inference")
    print("=" * 60)
    
    # Load the model
    try:
        model_path = get_model_path()
        print(f"Loading model from: {model_path}")
        model = YOLO(model_path)
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Open camera
    print(f"\nOpening camera (index: {CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        print("✗ Error: Could not open camera")
        return
    
    print("✓ Camera opened successfully!")
    print(f"\nRunning inference with confidence threshold: {CONFIDENCE_THRESHOLD}")
    print("\nControls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save current frame")
    print("=" * 60)
    
    frame_count = 0
    
    try:
        while True:
            # Read frame from camera
            ret, frame = cap.read()
            
            if not ret:
                print("✗ Error: Failed to grab frame")
                break
            
            # Run inference
            results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
            
            # Annotate frame with detections
            annotated_frame = results[0].plot()
            
            # Display FPS and detection count
            detections = results[0].boxes
            num_detections = len(detections)
            
            # Add info overlay
            info_text = f"Detections: {num_detections} | Frame: {frame_count}"
            cv2.putText(annotated_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show the frame
            cv2.imshow(WINDOW_NAME, annotated_frame)
            
            frame_count += 1
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                # Save current frame
                save_path = f"detection_frame_{frame_count}.jpg"
                cv2.imwrite(save_path, annotated_frame)
                print(f"✓ Saved frame to: {save_path}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nTotal frames processed: {frame_count}")
        print("Done!")

if __name__ == "__main__":
    main()
