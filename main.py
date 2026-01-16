import torch
import torch.nn.functional as F
import subprocess, os, shutil, ftfy
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from moviepy.editor import VideoFileClip, concatenate_videoclips

# --- SETUP ---
MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
REF_IMAGE = "tshirt_final_20260116_154007.png" 
OUTPUT_DIR = "./outputs"

# --- CORE FIXES ---
import diffusers.pipelines.wan.pipeline_wan_i2v as wan_module
wan_module.ftfy = ftfy
_orig_sdpa = F.scaled_dot_product_attention
def safe_sdpa(*args, **kwargs):
    kwargs.pop('enable_gqa', None)
    return _orig_sdpa(*args, **kwargs)
F.scaled_dot_product_attention = safe_sdpa

def generate_teaser():
    print("\n--- Generating High-End Reveal (14B @ 40 Steps) ---")
    
    # We load directly to CUDA. With 80GB VRAM, 14B fits if we don't go 720p.
    # Staying at 480p prevents the 'hallucination' and stays FAST.
    pipe = WanImageToVideoPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("cuda")
    
    ref_image = load_image(REF_IMAGE)
    
    # NEW PROMPT: Focuses on lighting and logo stability over complex rotation
    prompt = (
        "Cinematic close-up of a premium black t-shirt. The white swan logo in the center "
        "begins to glow intensely with a cool white light. The camera zooms in slowly and "
        "smoothly onto the logo. High-quality fabric textures, 4k, professional lighting."
    )
    
    output = pipe(
        prompt=prompt, 
        image=ref_image, 
        width=832, height=480, # 480p is the 'sweet spot' for 14B stability
        num_frames=81, 
        num_inference_steps=40, # High steps = No more 'AI mush'
        guidance_scale=5.0
    ).frames[0]
    
    final_path = os.path.join(OUTPUT_DIR, "blackfeel_luxury_teaser.mp4")
    export_to_video(output, final_path, fps=15)
    
    # --- Phase 2: Instant 2k Upscale (No Smoothing) ---
    # We are skipping the 'minterpolate' filter because it caused the smears
    print("--- Upscaling to 1440p (Lanczos) ---")
    master_path = os.path.join(OUTPUT_DIR, "blkcfeel_master.mp4")
    subprocess.run([
        'ffmpeg', '-y', '-i', final_path,
        '-vf', 'scale=2560:1440:flags=lanczos',
        '-c:v', 'libx264', '-crf', '18', master_path
    ], check=True)
    
    print(f"\nSUCCESS! High-end teaser ready: {master_path}")

if __name__ == "__main__":
    generate_teaser()