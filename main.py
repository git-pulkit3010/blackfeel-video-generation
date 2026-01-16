import torch
import torch.nn.functional as F
import subprocess, os, shutil, ftfy
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from moviepy.editor import VideoFileClip, concatenate_videoclips

# --- PRODUCTION TOGGLE ---
# Use "DRAFT" to iterate fast (3-5 mins). Switch to "PRODUCTION" for the final master.
MODE = "DRAFT" 

# --- CONFIGURATION ---
MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
REF_IMAGE = "tshirt_final_20260116_154007.png" 
OUTPUT_DIR = "./outputs"
BIN_DIR = "./bin"
TEMP_FRAMES = "./temp_frames"
TEMP_UPSCALED = "./temp_upscaled"

# Balanced settings for speed vs brand quality
if MODE == "DRAFT":
    RES_W, RES_H = 832, 480
    STEPS = 15  # Fast iteration
else:
    RES_W, RES_H = 1280, 720
    STEPS = 30  # High-quality master

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Compatibility Patch for PyTorch/Diffusers environments
import diffusers.pipelines.wan.pipeline_wan_i2v as wan_module
wan_module.ftfy = ftfy
_orig_sdpa = F.scaled_dot_product_attention
def safe_sdpa(*args, **kwargs):
    kwargs.pop('enable_gqa', None)
    return _orig_sdpa(*args, **kwargs)
F.scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention = safe_sdpa

def generate_segment(prompt, image_path, filename):
    print(f"\n--- Generating ({MODE}): {filename} ---")
    
    # Load in FP16 for paramount quality (no heavy 4-bit quantization)
    pipe = WanImageToVideoPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    
    # Memory management for A100 80GB
    pipe.enable_model_cpu_offload() 

    ref_image = load_image(image_path)
    output = pipe(
        prompt=prompt, 
        image=ref_image, 
        width=RES_W, height=RES_H,
        num_frames=81, 
        num_inference_steps=STEPS,
        guidance_scale=5.0
    ).frames[0]
    
    path = os.path.join(OUTPUT_DIR, f"{filename}.mp4")
    export_to_video(output, path, fps=15)
    
    del pipe
    torch.cuda.empty_cache()
    return path

def finalize_production(p1, p2, project_name):
    print("\n--- GPU Post-Production (Upscale + Smooth) ---")
    combined = os.path.join(OUTPUT_DIR, f"{project_name}_combined.mp4")
    
    # Stitch the segments
    clip1 = VideoFileClip(p1)
    clip2 = VideoFileClip(p2)
    concatenate_videoclips([clip1, clip2]).write_videofile(combined, codec='libx264', fps=15)
    
    for d in [TEMP_FRAMES, TEMP_UPSCALED]:
        if os.path.exists(d): shutil.rmtree(d)
        os.makedirs(d)

    # 1. Extract frames for high-fidelity processing
    subprocess.run(["ffmpeg", "-i", combined, "-qscale:v", "2", f"{TEMP_FRAMES}/f_%04d.jpg"], check=True)
    
    # 2. GPU Upscale (A100 accelerated)
    subprocess.run([f"{BIN_DIR}/realesrgan-ncnn-vulkan", "-i", TEMP_FRAMES, "-o", TEMP_UPSCALED, "-s", "2", "-n", "realesrgan-x4plus", "-m", f"{BIN_DIR}/models", "-g", "0"], check=True)
    
    # 3. Reassemble
    upscaled_vid = os.path.join(OUTPUT_DIR, f"{project_name}_upscaled.mp4")
    subprocess.run(["ffmpeg", "-y", "-framerate", "15", "-i", f"{TEMP_UPSCALED}/f_%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", upscaled_vid], check=True)
    
    # 4. GPU Smooth (Frame Interpolation to 30fps)
    final = os.path.join(OUTPUT_DIR, f"{project_name}_final.mp4")
    subprocess.run([f"{BIN_DIR}/rife/rife-ncnn-vulkan", "-i", upscaled_vid, "-o", final, "-m", f"{BIN_DIR}/rife/rife-v4.6", "-g", "0"], check=True)
    
    shutil.rmtree(TEMP_FRAMES); shutil.rmtree(TEMP_UPSCALED)
    return final

if __name__ == "__main__":
    # Prompt optimized for the BlkcFeel aesthetic
    P1 = "A premium black t-shirt with a white swan logo inside a minimalist circle. The shirt is under a sharp cinematic spotlight and rotates 360 degrees. High-fidelity fabric texture."
    P2 = "The white swan logo on the black fabric pops out and moves toward the camera as a 3D glass-like object. Dramatic reveal of glowing white text 'COMING SOON' appearing below."

    part1 = generate_segment(P1, REF_IMAGE, "part1_rotation")
    part2 = generate_segment(P2, REF_IMAGE, "part2_reveal")
    
    final_path = finalize_production(part1, part2, "blackfeel_teaser")
    print(f"\nSUCCESS! Teaser ready: {final_path}")