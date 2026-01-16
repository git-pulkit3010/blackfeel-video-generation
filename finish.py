import os
import subprocess
from moviepy.editor import VideoFileClip, concatenate_videoclips

# CONFIGURATION
OUTPUT_DIR = "./outputs"
BIN_DIR = "./bin"
PROJECT_NAME = "swan_teaser"

def finish_teaser():
    print("--- Resuming Finalization (Stitch -> Upscale -> Smooth) ---")
    
    # Locate the existing generated parts
    p1 = os.path.join(OUTPUT_DIR, "part1_rotation.mp4")
    p2 = os.path.join(OUTPUT_DIR, "part2_reveal.mp4")
    
    if not os.path.exists(p1) or not os.path.exists(p2):
        print("ERROR: Could not find part1_rotation.mp4 or part2_reveal.mp4 in ./outputs")
        return

    c1, c2 = VideoFileClip(p1), VideoFileClip(p2)
    
    # 1. Concatenate the segments
    combined_path = os.path.join(OUTPUT_DIR, f"{PROJECT_NAME}_combined.mp4")
    print(f"Stitching into: {combined_path}")
    concatenate_videoclips([c1, c2]).write_videofile(combined_path, codec='libx264', fps=15)
    
    # 2. Upscale (2x)
    # Removed '--face_enhance' and added '-n realesrgan-x4plus' for high-quality realism
    upscaled = os.path.join(OUTPUT_DIR, f"{PROJECT_NAME}_upscaled.mp4")
    print(f"Upscaling into: {upscaled}")
    subprocess.run([
        f"{BIN_DIR}/realesrgan-ncnn-vulkan", 
        "-i", combined_path, 
        "-o", upscaled, 
        "-s", "2", 
        "-n", "realesrgan-x4plus"
    ], check=True)
    
    # 3. Smooth (Interpolate)
    final = os.path.join(OUTPUT_DIR, f"{PROJECT_NAME}_final.mp4")
    print(f"Smoothing into: {final}")
    subprocess.run([
        f"{BIN_DIR}/rife/rife-ncnn-vulkan", 
        "-i", upscaled, 
        "-o", final, 
        "-m", f"{BIN_DIR}/rife/rife-v4.6"
    ], check=True)
    
    print(f"\nSUCCESS! Your teaser is ready at: {final}")

if __name__ == "__main__":
    finish_teaser()