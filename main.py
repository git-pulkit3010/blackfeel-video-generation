import torch
import torch.nn.functional as F
import subprocess, os, shutil, ftfy
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from moviepy.editor import VideoFileClip, concatenate_videoclips

# --- PRODUCTION TOGGLE ---
MODE = "DRAFT"  # Change to "PRODUCTION" for final high-res render

# --- CONFIGURATION ---
MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
REF_IMAGE = "tshirt_final_20260116_154007.png" 
OUTPUT_DIR = "./outputs"
BIN_DIR = "./bin"
TEMP_FRAMES = "./temp_frames"
TEMP_UPSCALED = "./temp_upscaled"

if MODE == "DRAFT":
    RES_W, RES_H = 832, 480
    STEPS = 15
    TEACACHE_THRESHOLD = 0.15
else:
    RES_W, RES_H = 1280, 720
    STEPS = 35
    TEACACHE_THRESHOLD = 0.05

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Compatibility Patches
import diffusers.pipelines.wan.pipeline_wan_i2v as wan_module
wan_module.ftfy = ftfy
_orig_sdpa = F.scaled_dot_product_attention
def safe_sdpa(*args, **kwargs):
    kwargs.pop('enable_gqa', None)
    return _orig_sdpa(*args, **kwargs)
F.scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention = safe_sdpa

def generate_segment(prompt, image_path, filename):
    print(f"\n--- Generating ({MODE}): {filename} ---")
    
    # Load in FP16 for paramount quality
    pipe = WanImageToVideoPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload() # Safely manages 80GB VRAM
    
    # TeaCache Implementation: Skips redundant math steps for 2x speedup
    if hasattr(pipe, "transformer"):
        pipe.transformer.set_cache_threshold(TEACACHE_THRESHOLD)

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

def finalize_video(p1, p2, project_name):
    print("\n--- GPU Post-Production (Upscale + Smooth) ---")
    combined = os.path.join(OUTPUT_DIR, f"{project_name}_combined.mp4")
    concatenate_videoclips([VideoFileClip(p1), VideoFileClip(p2)]).write_videofile(combined, codec='libx264', fps=15)
    
    for d in [TEMP_FRAMES, TEMP_UPSCALED]:
        if os.path.exists(d): shutil.rmtree(d)
        os.makedirs(d)

    # 1. Extract
    subprocess.run(["ffmpeg", "-i", combined, "-qscale:v", "2", f"{TEMP_FRAMES}/f_%04d.jpg"], check=True)
    
    # 2. GPU Upscale (Uses A100 -g 0)
    subprocess.run([f"{BIN_DIR}/realesrgan-ncnn-vulkan", "-i", TEMP_FRAMES, "-o", TEMP_UPSCALED, "-s", "2", "-n", "realesrgan-x4plus", "-m", f"{BIN_DIR}/models", "-g", "0"], check=True)
    
    # 3. Reassemble
    upscaled_vid = os.path.join(OUTPUT_DIR, f"{project_name}_upscaled.mp4")
    subprocess.run(["ffmpeg", "-y", "-framerate", "15", "-i", f"{TEMP_UPSCALED}/f_%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", upscaled_vid], check=True)
    
    # 4. GPU Smooth
    final = os.path.join(OUTPUT_DIR, f"{project_name}_final.mp4")
    subprocess.run([f"{BIN_DIR}/rife/rife-ncnn-vulkan", "-i", upscaled_vid, "-o", final, "-m", f"{BIN_DIR}/rife/rife-v4.6", "-g", "0"], check=True)
    
    shutil.rmtree(TEMP_FRAMES); shutil.rmtree(TEMP_UPSCALED)
    return final

if __name__ == "__main__":
    # Prompt optimized for the BlackFeel aesthetic
    P1 = "A high-quality black t-shirt with a white swan logo inside a black circle. The t-shirt is centered under a sharp cinematic spotlight. The t-shirt rotates slowly 360 degrees. Realistic fabric texture."
    P2 = "The white swan logo pops out from the fabric and floats forward toward the camera, becoming an enlarged 3D glowing object. The words 'COMING SOON' appear in bold white glowing text below it."

    part1 = generate_segment(P1, REF_IMAGE, "rotation")
    part2 = generate_segment(P2, REF_IMAGE, "reveal")
    
    final_path = finalize_video(part1, part2, "swan_teaser")
    print(f"\nSUCCESS! Teaser ready: {final_path}")