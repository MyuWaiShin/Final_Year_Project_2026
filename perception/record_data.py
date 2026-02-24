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
    
    # RGB Camera
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setVideoSize(1920, 1080)  # Full HD
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(30)
    
    # Video Encoder (H.264)
    video_enc = pipeline.create(dai.node.VideoEncoder)
    video_enc.setDefaultProfilePreset(30, dai.VideoEncoderProperties.Profile.H264_MAIN)
    cam_rgb.video.link(video_enc.input)
    
    # Preview output (for live view)
    xout_preview = pipeline.create(dai.node.XLinkOut)
    xout_preview.setStreamName("preview")
    cam_rgb.preview.link(xout_preview.input)
    
    # Video output
    xout_video = pipeline.create(dai.node.XLinkOut)
    xout_video.setStreamName("video")
    video_enc.bitstream.link(xout_video.input)
    
    return pipeline

def record_video(output_dir, duration_seconds=20, session_name=None):
    """
    Record video from OAK-D camera
    
    Args:
        output_dir: Directory to save videos
        duration_seconds: How long to record (default: 20 seconds)
        session_name: Optional name for this recording session
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    if session_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"{session_name}_{timestamp}.mp4"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"recording_{timestamp}.mp4"
    
    video_path = output_path / video_filename
    
    print("=" * 60)
    print("OAK-D Data Collection - Video Recording")
    print("=" * 60)
    print(f"Output: {video_path}")
    print(f"Duration: {duration_seconds} seconds")
    print("=" * 60)
    
    # Create pipeline
    pipeline = create_recording_pipeline()
    
    # Connect to device
    print("\nConnecting to camera...")
    config = dai.Device.Config()
    config.board.usb.maxSpeed = dai.UsbSpeed.HIGH
    
    with dai.Device(config) as device:
        device.startPipeline(pipeline)
        print(f"✓ Connected to: {device.getMxId()}")
        
        # Get queues
        q_preview = device.getOutputQueue("preview", maxSize=4, blocking=False)
        q_video = device.getOutputQueue("video", maxSize=30, blocking=True)
        
        # Open video file for writing
        video_file = open(video_path, 'wb')
        
        print(f"\n🔴 RECORDING... ({duration_seconds}s)")
        print("Press 'q' to stop early")
        print("-" * 60)
        
        start_time = time.time()
        frame_count = 0
        
        while True:
            elapsed = time.time() - start_time
            
            # Check if duration reached
            if elapsed >= duration_seconds:
                break
            
            # Get preview frame for display
            if q_preview.has():
                preview = q_preview.get()
                frame = preview.getCvFrame()
                
                # Add recording indicator
                remaining = int(duration_seconds - elapsed)
                cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)  # Red dot
                cv2.putText(frame, f"REC {remaining}s", (50, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.imshow("Recording Preview", frame)
            
            # Get encoded video data
            if q_video.has():
                video_data = q_video.get()
                video_file.write(video_data.getData())
                frame_count += 1
            
            # Check for early exit
            if cv2.waitKey(1) == ord('q'):
                print("\n⚠ Recording stopped by user")
                break
        
        # Cleanup
        video_file.close()
        cv2.destroyAllWindows()
        
        elapsed = time.time() - start_time
        print("-" * 60)
        print(f"✓ Recording complete!")
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Frames: {frame_count}")
        print(f"  Saved to: {video_path}")
        print("=" * 60)

def main():
    """Main recording interface"""
    print("\n" + "=" * 60)
    print("OAK-D Data Collection Tool")
    print("=" * 60)
    
    # Configuration
    output_dir = "data/recordings"
    
    print("\nRecording Configuration:")
    print(f"  Output directory: {output_dir}")
    print(f"  Default duration: 20 seconds")
    print(f"  Resolution: 1920x1080 @ 30fps")
    print(f"  Format: H.264 MP4")
    
    # Get session name
    print("\n" + "-" * 60)
    session_name = input("Enter session name (e.g., 'cubes_high', 'mixed_low'): ").strip()
    
    if not session_name:
        session_name = None
        print("No session name provided, using timestamp only")
    
    # Get duration
    duration_input = input("Duration in seconds (default: 20): ").strip()
    duration = int(duration_input) if duration_input else 20
    
    print("\n" + "-" * 60)
    print("Ready to record!")
    print("Press ENTER to start recording...")
    input()
    
    # Start recording
    try:
        record_video(output_dir, duration, session_name)
    except KeyboardInterrupt:
        print("\n\n⚠ Recording cancelled by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nDone!")

if __name__ == "__main__":
    main()
