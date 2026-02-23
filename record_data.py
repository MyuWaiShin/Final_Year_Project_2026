import os
os.environ["DEPTHAI_BOOT_TIMEOUT"] = "30000"

import depthai as dai
import cv2
import time
from pathlib import Path
from datetime import datetime

def create_recording_pipeline():
    """Create DepthAI pipeline for recording RGB video"""
    pipeline = dai.Pipeline()
    pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_4)
    
    # RGB Camera - 1080p for high quality
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setVideoSize(1920, 1080)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(30)
    
    # Preview output (for live view - lower res for performance)
    cam_rgb.setPreviewSize(640, 480)
    xout_preview = pipeline.create(dai.node.XLinkOut)
    xout_preview.setStreamName("preview")
    cam_rgb.preview.link(xout_preview.input)
    
    # Video output (full resolution)
    xout_video = pipeline.create(dai.node.XLinkOut)
    xout_video.setStreamName("video")
    cam_rgb.video.link(xout_video.input)
    
    return pipeline

def record_session(output_dir, session_name, duration_seconds=20):
    """
    Record a data collection session
    
    Args:
        output_dir: Base directory for recordings
        session_name: Name for this session (e.g., "cubes_high_angle")
        duration_seconds: Recording duration
    """
    # Detect object type from session name and create subfolder
    session_lower = session_name.lower()
    if 'cube' in session_lower:
        object_folder = 'cubes'
    elif 'cylinder' in session_lower:
        object_folder = 'cylinders'
    elif 'arc' in session_lower:
        object_folder = 'arcs'
    elif 'mixed' in session_lower or 'all' in session_lower:
        object_folder = 'mixed'
    else:
        object_folder = 'other'
    
    # Create output directory with object subfolder
    output_path = Path(output_dir) / object_folder
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"{session_name}_{timestamp}.mp4"
    video_path = output_path / video_filename
    
    print("\n" + "=" * 70)
    print("🎥 OAK-D DATA COLLECTION")
    print("=" * 70)
    print(f"Session: {session_name}")
    print(f"Output: {video_path}")
    print(f"Duration: {duration_seconds}s")
    print("Resolution: 1920x1080 @ 30fps")
    print("=" * 70)
    
    # Create pipeline
    pipeline = create_recording_pipeline()
    
    # Connect to device
    print("\nConnecting to camera...")
    config = dai.Device.Config()
    config.board.usb.maxSpeed = dai.UsbSpeed.HIGH
    
    with dai.Device(config) as device:
        device.startPipeline(pipeline)
        print(f"✓ Connected: {device.getMxId()}")
        print(f"✓ USB Speed: {device.getUsbSpeed().name}")
        
        # Get queues
        q_preview = device.getOutputQueue("preview", maxSize=4, blocking=False)
        q_video = device.getOutputQueue("video", maxSize=30, blocking=False)
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(video_path), fourcc, 30, (1920, 1080))
        
        print("\n" + "🔴 RECORDING STARTED" + "\n")
        print("Press 'q' to stop early")
        print("-" * 70)
        
        start_time = time.time()
        frame_count = 0
        
        while True:
            elapsed = time.time() - start_time
            
            # Check if duration reached
            if elapsed >= duration_seconds:
                break
            
            # Get and save full resolution frame
            if q_video.has():
                video_frame = q_video.get()
                frame_full = video_frame.getCvFrame()
                video_writer.write(frame_full)
                frame_count += 1
            
            # Get preview for display
            if q_preview.has():
                preview = q_preview.get()
                frame_preview = preview.getCvFrame()
                
                # Add recording indicator
                remaining = int(duration_seconds - elapsed)
                cv2.circle(frame_preview, (30, 30), 10, (0, 0, 255), -1)
                cv2.putText(frame_preview, f"REC {remaining}s", (50, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame_preview, session_name, (10, 470), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow("Recording Preview (Press 'q' to stop)", frame_preview)
            
            # Check for early exit
            if cv2.waitKey(1) == ord('q'):
                print("\n⚠ Recording stopped by user")
                break
        
        # Cleanup
        video_writer.release()
        cv2.destroyAllWindows()
        
        elapsed = time.time() - start_time
        print("-" * 70)
        print(f"✓ Recording complete!")
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Frames: {frame_count}")
        print(f"  FPS: {frame_count/elapsed:.1f}")
        print(f"  Saved: {video_path.name}")
        print("=" * 70)
        
        return video_path

def main():
    """Interactive data collection tool"""
    print("\n" + "=" * 70)
    print("OAK-D DATA COLLECTION TOOL")
    print("=" * 70)
    print("\nThis tool helps you collect training data systematically.")
    print("\nRecommended session naming:")
    print("  - Format: <object>_<height>_<angle>")
    print("  - Examples:")
    print("      cubes_high_top")
    print("      cylinders_medium_angled")
    print("      mixed_low_side")
    print("      arcs_high_tilted")
    
    output_dir = "recordings"
    
    while True:
        print("\n" + "-" * 70)
        print("NEW RECORDING SESSION")
        print("-" * 70)
        
        # Get session name
        session_name = input("\nSession name (or 'q' to quit): ").strip()
        
        if session_name.lower() == 'q':
            print("\nExiting data collection tool.")
            break
        
        if not session_name:
            print("❌ Session name cannot be empty!")
            continue
        
        # Get duration
        duration_input = input("Duration in seconds (default: 20): ").strip()
        duration = int(duration_input) if duration_input.isdigit() else 20
        
        print("\n📋 Ready to record:")
        print(f"   Session: {session_name}")
        print(f"   Duration: {duration}s")
        print("\nPress ENTER when ready (or 'c' to cancel)...")
        
        ready = input().strip().lower()
        if ready == 'c':
            print("Cancelled.")
            continue
        
        # Record
        try:
            video_path = record_session(output_dir, session_name, duration)
            
            # Ask if user wants to continue
            print("\nRecord another session? (y/n): ", end='')
            again = input().strip().lower()
            if again != 'y':
                break
                
        except KeyboardInterrupt:
            print("\n\n⚠ Recording cancelled")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            
            print("\nTry again? (y/n): ", end='')
            retry = input().strip().lower()
            if retry != 'y':
                break
    
    print("\n" + "=" * 70)
    print("Data collection complete!")
    print(f"Videos saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Extract frames from videos")
    print("2. Label frames using your annotation tool")
    print("3. Split into train/val/test (80/10/10)")
    print("4. Train YOLOv8n model")
    print("=" * 70)

if __name__ == "__main__":
    main()
