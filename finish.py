import os
import subprocess
import shutil
from moviepy.editor import VideoFileClip, concatenate_videoclips

# CONFIGURATION
OUTPUT_DIR = "./outputs"
BIN_DIR = "./bin"
PROJECT_NAME = "swan_teaser"
TEMP_FRAMES = "./temp_frames"
TEMP_UPSCALED = "./temp_upscaled"

def finish_teaser():
    print("--- Resuming Finalization (Stitch -> Frames -> Upscale -> Reassemble -> Smooth) ---")
    
    # Locate the existing generated segments
    p1 = os.path.join(OUTPUT_DIR, "part1_rotation.mp4")
    p2 = os.path.join(OUTPUT_DIR, "part2_reveal.mp4")
    
    if not os.path.exists(p1) or not os.path.exists(p2):
        print(f"ERROR: Generated segments not found in {OUTPUT_DIR}.")
        return

    # 1. Stitch the segments together
    combined_path = os.path.join(OUTPUT_DIR, f"{PROJECT_NAME}_combined.mp4")
    print(f"Stitching into: {combined_path}")
    c1, c2 = VideoFileClip(p1), VideoFileClip(p2)
    concatenate_videoclips([c1, c2]).write_videofile(combined_path, codec='libx264', fps=15)
    
    # 2. Extract Frames using FFmpeg
    if os.path.exists(TEMP_FRAMES): shutil.rmtree(TEMP_FRAMES)
    if os.path.exists(TEMP_UPSCALED): shutil.rmtree(TEMP_UPSCALED)
    os.makedirs(TEMP_FRAMES)
    os.makedirs(TEMP_UPSCALED)
    
    print("Extracting video frames for upscaling...")
    subprocess.run([
        "ffmpeg", "-i", combined_path, 
        "-qscale:v", "2", 
        f"{TEMP_FRAMES}/frame_%04d.jpg"
    ], check=True)
    
    # 3. Upscale the Frame Directory
    # We point to the 'models' folder in bin/ to ensure it loads correctly
    print("Upscaling frames with Real-ESRGAN (This may take a minute)...")
    subprocess.run([
        f"{BIN_DIR}/realesrgan-ncnn-vulkan", 
        "-i", TEMP_FRAMES, 
        "-o", TEMP_UPSCALED, 
        "-s", "2", 
        "-n", "realesrgan-x4plus",
        "-m", f"{BIN_DIR}/models"
    ], check=True)
    
    # 4. Reassemble Upscaled Video
    # The upscaler outputs .png files by default
    upscaled_video = os.path.join(OUTPUT_DIR, f"{PROJECT_NAME}_upscaled.mp4")
    print(f"Reassembling upscaled frames into: {upscaled_video}")
    subprocess.run([
        "ffmpeg", "-y", "-framerate", "15", 
        "-i", f"{TEMP_UPSCALED}/frame_%04d.png", 
        "-c:v", "libx264", "-pix_fmt", "yuv420p", 
        upscaled_video
    ], check=True)
    
    # 5. Smooth the final output with RIFE
    final = os.path.join(OUTPUT_DIR, f"{PROJECT_NAME}_final.mp4")
    print(f"Smoothing final video (Interpolation) into: {final}")
    subprocess.run([
        f"{BIN_DIR}/rife/rife-ncnn-vulkan", 
        "-i", upscaled_video, 
        "-o", final, 
        "-m", f"{BIN_DIR}/rife/rife-v4.6"
    ], check=True)
    
    # Cleanup temporary files
    shutil.rmtree(TEMP_FRAMES)
    shutil.rmtree(TEMP_UPSCALED)
    
    print(f"\nSUCCESS! Your Blackwell-accelerated teaser is ready: {final}")

if __name__ == "__main__":
    finish_teaser()