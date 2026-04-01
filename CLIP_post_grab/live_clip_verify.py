import cv2
import torch
import clip
from PIL import Image
import pickle
import sys
import os
import time

try:
    import depthai as dai
except ImportError:
    print("depthai is needed for live camera feed. pip install depthai")
    sys.exit(1)

def load_probe_classifier(pkl_path="clip_probe_v2.pk"):
    """Loads the trained Logistic Regression model."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, pkl_path)
    
    if not os.path.exists(full_path):
        print(f"Error: Could not find '{full_path}'. Run train_clip_probe.py first!")
        sys.exit(1)
        
    with open(full_path, "rb") as f:
        clf = pickle.load(f)
    return clf

def open_oak_camera():
    """Initializes the OAK-D camera."""
    pipeline = dai.Pipeline()
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setVideoSize(1920, 1080)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(30)

    # Control input for autofocus
    controlIn = pipeline.create(dai.node.XLinkIn)
    controlIn.setStreamName("control")
    controlIn.out.link(cam_rgb.inputControl)

    # Hardware MJPEG encoding (drastically reduces USB bandwidth to prevent XLink crashes)
    videoEnc = pipeline.create(dai.node.VideoEncoder)
    videoEnc.setDefaultProfilePreset(30, dai.VideoEncoderProperties.Profile.MJPEG)
    videoEnc.setQuality(90)
    cam_rgb.video.link(videoEnc.input)

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("mjpeg")
    videoEnc.bitstream.link(xout.input)

    config = dai.Device.Config()
    device = dai.Device(config)
    device.startPipeline(pipeline)

    q_mjpeg = device.getOutputQueue("mjpeg", maxSize=4, blocking=False)
    q_control = device.getInputQueue("control")
    
    # Continuous Auto-focus
    ctrl = dai.CameraControl()
    ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
    q_control.send(ctrl)

    return device, q_mjpeg

def main():
    print("Initializing Live CLIP Verification...")

    # 1. Load Custom Classifier
    clf = load_probe_classifier("clip_probe.pkl")

    # 2. Load CLIP Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP backbone on {device}...")
    model, preprocess = clip.load("ViT-B/32", device=device)

    # 3. Start Camera
    print("Starting OAK-D camera...")
    oak_device, q_mjpeg = open_oak_camera()

    print("\n" + "=" * 50)
    print("LIVE INFERENCE RUNNING")
    print("Press 'q' in the video window to quit.")
    print("=" * 50 + "\n")

    try:
        while True:
            # Grab encoded frame and decode using OpenCV
            in_encoded = q_mjpeg.get()
            frame = cv2.imdecode(in_encoded.getData(), cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # The exact crop math we used during training
            h, w = frame.shape[:2]
            crop_w = 1400
            crop_h = 600
            cx = w // 2
            cy = h - (crop_h // 2) - 10 
            
            x1 = max(0, cx - crop_w // 2)
            y1 = max(0, cy - crop_h // 2)
            x2 = min(w, x1 + crop_w)
            y2 = min(h, y1 + crop_h)
            
            # Extract the region of interest
            crop_frame = frame[y1:y2, x1:x2].copy()

            # Format for CLIP
            img_rgb_cv = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb_cv)
            image_input = preprocess(pil_image).unsqueeze(0).to(device)

            # Inference
            t0 = time.time()
            with torch.no_grad():
                feature = model.encode_image(image_input)
                feature = feature / feature.norm(dim=-1, keepdim=True)
                feature_np = feature.cpu().numpy()
            
            prediction = clf.predict(feature_np)[0]
            probabilities = clf.predict_proba(feature_np)[0]
            infer_time = (time.time() - t0) * 1000

            # Determine Label
            label_map = {0: "Empty", 1: "Holding"}
            predicted_label = label_map[prediction]
            confidence = probabilities[prediction]

            # If confidence is lower than 85%, mark as uncertain
            color = (0, 255, 0) # Green for confident hit
            if confidence < 0.85:
                predicted_label = f"Uncertain ({predicted_label})"
                color = (0, 165, 255) # Orange for uncertain

            # ====== DISPLAY ======
            # Show the full frame (scaled down) with the crop box and HUD
            display = cv2.resize(frame, (854, 480))
            scale_x, scale_y = 854/w, 480/h
            
            # Draw crop boundary
            cv2.rectangle(display, 
                          (int(x1*scale_x), int(y1*scale_y)), 
                          (int(x2*scale_x), int(y2*scale_y)), 
                          (255, 0, 0), 2)
            
            # HUD Text
            hud_text = f"Status: {predicted_label}"
            conf_text = f"Confidence: {confidence*100:.1f}%"
            time_text = f"Inference Time: {infer_time:.1f}ms"
            
            cv2.putText(display, hud_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            cv2.putText(display, conf_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(display, time_text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("Live CLIP Gripper Verification", display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nManually interrupted.")
    finally:
        oak_device.close()
        cv2.destroyAllWindows()
        print("Camera closed.")

if __name__ == "__main__":
    main()
