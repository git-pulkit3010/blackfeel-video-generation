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
    print("--- Finalizing Video (Stitch -> Optimized CPU Upscale -> Smooth) ---")
    
    p1 = os.path.join(OUTPUT_DIR, "part1_rotation.mp4")
    p2 = os.path.join(OUTPUT_DIR, "part2_reveal.mp4")
    
    if not os.path.exists(p1) or not os.path.exists(p2):
        print(f"ERROR: Generated segments not found in {OUTPUT_DIR}.")
        return

    # 1. Stitch
    combined_path = os.path.join(OUTPUT_DIR, f"{PROJECT_NAME}_combined.mp4")
    c1, c2 = VideoFileClip(p1), VideoFileClip(p2)
    concatenate_videoclips([c1, c2]).write_videofile(combined_path, codec='libx264', fps=15)
    
    # 2. Extract Frames
    if os.path.exists(TEMP_FRAMES): shutil.rmtree(TEMP_FRAMES)
    if os.path.exists(TEMP_UPSCALED): shutil.rmtree(TEMP_UPSCALED)
    os.makedirs(TEMP_FRAMES)
    os.makedirs(TEMP_UPSCALED)
    
    print("Extracting frames...")
    subprocess.run(["ffmpeg", "-i", combined_path, "-qscale:v", "2", f"{TEMP_FRAMES}/frame_%04d.jpg"], check=True)
    
    # 3. Optimized CPU Upscale
    # We remove '-g 0' and add '-j 4:4:4' to use all CPU cores
    print("Upscaling (CPU Optimized)... This will take ~5 minutes.")
    subprocess.run([
        f"{BIN_DIR}/realesrgan-ncnn-vulkan", 
        "-i", TEMP_FRAMES, 
        "-o", TEMP_UPSCALED, 
        "-s", "2", 
        "-n", "realesrgan-x4plus",
        "-m", f"{BIN_DIR}/models",
        "-j", "4:4:4" 
    ], check=True)
    
    # 4. Reassemble
    upscaled_video = os.path.join(OUTPUT_DIR, f"{PROJECT_NAME}_upscaled.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-framerate", "15", 
        "-i", f"{TEMP_UPSCALED}/frame_%04d.png", 
        "-c:v", "libx264", "-pix_fmt", "yuv420p", 
        upscaled_video
    ], check=True)
    
    # 5. Smooth (Interpolate)
    final = os.path.join(OUTPUT_DIR, f"{PROJECT_NAME}_final.mp4")
    print(f"Smoothing into: {final}")
    subprocess.run([
        f"{BIN_DIR}/rife/rife-ncnn-vulkan", 
        "-i", upscaled_video, 
        "-o", final, 
        "-m", f"{BIN_DIR}/rife/rife-v4.6"
    ], check=True)
    
    shutil.rmtree(TEMP_FRAMES)
    shutil.rmtree(TEMP_UPSCALED)
    print(f"\nSUCCESS! Teaser ready: {final}")

if __name__ == "__main__":
    finish_teaser()