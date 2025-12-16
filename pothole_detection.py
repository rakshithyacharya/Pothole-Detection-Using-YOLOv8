import subprocess
import os
import cv2
import time
import sys
from ultralytics import YOLO

# FFmpeg Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
ffmpeg_path = os.path.join(current_dir, "tools", "ffmpeg.exe")
if not os.path.exists(ffmpeg_path): ffmpeg_path = "ffmpeg" 

os.makedirs('tools', exist_ok=True)
os.makedirs('results', exist_ok=True)

def detect_from_video(video_path, model):
    if not os.path.exists(video_path): return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    temp_output = "results/temp_output.avi"
    final_output = "results/processed.mp4" 
    
    writer = cv2.VideoWriter(temp_output, cv2.VideoWriter_fourcc(*'XVID'), fps_video, (width, height))

    print("\n" + "="*50)
    print(f" VIDEO PROCESSING STARTED (Total Frames: {total_frames})")
    print("="*50)

    frame_count = 0
    
    # Metrics Variables
    total_inference_time = 0
    total_confidence_sum = 0
    frames_with_detections = 0

    while True:
        ret, frame = cap.read()
        if not ret: break   
        
        # --- Measure Speed ---
        t0 = time.time()
        results = model(frame, verbose=False) # verbose=False to keep terminal clean
        t1 = time.time()
        
        # Calculate Frame metrics
        inference_time = (t1 - t0)
        total_inference_time += inference_time
        
        # Calculate Accuracy (Confidence)
        frame_conf = 0.0
        if len(results[0].boxes) > 0:
            frame_conf = float(results[0].boxes.conf[0]) * 100
            total_confidence_sum += frame_conf
            frames_with_detections += 1

        annotated_frame = results[0].plot()
        writer.write(annotated_frame)
        
        frame_count += 1
        
        # Print progress every 10 frames
        if frame_count % 10 == 0 or frame_count == total_frames:
            current_fps = 1 / inference_time if inference_time > 0 else 0
            progress = (frame_count / total_frames) * 100
            sys.stdout.write(f"\r[Processing] Frame {frame_count}/{total_frames} ({progress:.1f}%) | Speed: {current_fps:.1f} FPS | Conf: {frame_conf:.1f}%   ")
            sys.stdout.flush()

    cap.release()
    writer.release()
    print("\n") # New line after progress bar

    # --- CALCULATE FINAL AVERAGES ---
    avg_fps = frame_count / total_inference_time if total_inference_time > 0 else 0
    avg_conf = total_confidence_sum / frames_with_detections if frames_with_detections > 0 else 0

    # --- TERMINAL OUTPUT FOR VIDEO ---
    print("="*50)
    print(" VIDEO PROCESSING COMPLETE")
    print("="*50)
    print(f"  Average Processing Speed : {avg_fps:.2f} FPS")
    print(f" Average Detection Accuracy: {avg_conf:.2f} %")
    print(f" Frames with Potholes     : {frames_with_detections}/{frame_count}")
    print("="*50 + "\n")

    # FFmpeg Conversion
    print("Converting video for web...")
    ffmpeg_cmd = [
        ffmpeg_path, "-i", temp_output, "-c:v", "libx264", 
        "-preset", "fast", "-crf", "22", "-movflags", "+faststart", "-y", final_output
    ]    
    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        if os.path.exists(temp_output): os.remove(temp_output)
        return final_output
    except Exception as e:
        print(f"FFmpeg Error: {e}")
        return temp_output