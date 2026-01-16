import torch
import torch.nn.functional as F
import subprocess, os, shutil, types, ftfy
from PIL import Image
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from moviepy.editor import VideoFileClip, concatenate_videoclips

# --- SETUP & PATCHES ---
import diffusers.pipelines.wan.pipeline_wan_i2v as wan_module
wan_module.ftfy = ftfy

_orig_sdpa = F.scaled_dot_product_attention
def safe_sdpa(*args, **kwargs):
    kwargs.pop('enable_gqa', None)
    return _orig_sdpa(*args, **kwargs)
F.scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention = safe_sdpa

# --- CONFIGURATION ---
MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
REF_IMAGE = "tshirt_final_20260116_154007.png" 
OUTPUT_DIR = "./outputs"
BIN_DIR = "./bin"
TEMP_FRAMES = "./temp_frames"
TEMP_UPSCALED = "./temp_upscaled"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_segment(prompt, image_path, filename):
    print(f"\n--- Generating: {filename} ---")
    
    # Load in FP16
    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16
    )

    # CRITICAL FIX: Use Sequential CPU Offload for 80GB VRAM
    # This is more memory-efficient than pipe.to("cuda")
    pipe.enable_sequential_cpu_offload()
    
    ref_image = load_image(image_path)
    
    # Generate video
    output = pipe(
        prompt=prompt, 
        image=ref_image, 
        num_frames=81, 
        num_inference_steps=30
    ).frames[0]
    
    path = os.path.join(OUTPUT_DIR, f"{filename}.mp4")
    export_to_video(output, path, fps=15)
    
    # Force deep cleanup
    del pipe
    torch.cuda.empty_cache()
    return path

def post_process(p1, p2, project_name):
    print("\n--- Finalizing (Stitch -> GPU Upscale -> Smooth) ---")
    combined = os.path.join(OUTPUT_DIR, f"{project_name}_combined.mp4")
    concatenate_videoclips([VideoFileClip(p1), VideoFileClip(p2)]).write_videofile(combined, codec='libx264', fps=15)
    
    # Setup temp dirs
    for d in [TEMP_FRAMES, TEMP_UPSCALED]:
        if os.path.exists(d): shutil.rmtree(d)
        os.makedirs(d)

    print("GPU Upscaling frames...")
    subprocess.run(["ffmpeg", "-i", combined, "-qscale:v", "2", f"{TEMP_FRAMES}/f_%04d.jpg"], check=True)
    # GPU acceleration for A100
    subprocess.run([f"{BIN_DIR}/realesrgan-ncnn-vulkan", "-i", TEMP_FRAMES, "-o", TEMP_UPSCALED, "-s", "2", "-n", "realesrgan-x4plus", "-m", f"{BIN_DIR}/models", "-g", "0"], check=True)
    
    upscaled_vid = os.path.join(OUTPUT_DIR, f"{project_name}_upscaled.mp4")
    subprocess.run(["ffmpeg", "-y", "-framerate", "15", "-i", f"{TEMP_UPSCALED}/f_%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", upscaled_vid], check=True)
    
    final = os.path.join(OUTPUT_DIR, f"{project_name}_final.mp4")
    print("GPU Smoothing...")
    subprocess.run([f"{BIN_DIR}/rife/rife-ncnn-vulkan", "-i", upscaled_vid, "-o", final, "-m", f"{BIN_DIR}/rife/rife-v4.6", "-g", "0"], check=True)
    
    # Cleanup
    shutil.rmtree(TEMP_FRAMES)
    shutil.rmtree(TEMP_UPSCALED)
    return final

if __name__ == "__main__":
    P1 = "A high-quality black t-shirt with a white swan logo inside a black circle. Centered under a spotlight, rotating 360 degrees. Realistic fabric texture."
    P2 = "The white swan logo on the chest pops out, enlarges, and floats toward the camera as a 3D object. Glowing white 'COMING SOON' text appears below."

    # Part 1: Rotation
    v1 = generate_segment(P1, REF_IMAGE, "rotation")
    # Part 2: Reveal
    v2 = generate_segment(P2, REF_IMAGE, "reveal")
    
    # Final production
    result = post_process(v1, v2, "swan_teaser")
    print(f"\nSUCCESS! Teaser ready: {result}")