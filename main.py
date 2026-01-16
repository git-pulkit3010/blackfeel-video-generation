import torch
import torch.nn.functional as F
import subprocess
import os
from PIL import Image
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from moviepy.editor import VideoFileClip, concatenate_videoclips
import ftfy

# --- MINIMAL PATCHES (Only for very new/dev features) ---
# Inject ftfy for text cleaning
import diffusers.pipelines.wan.pipeline_wan_i2v as wan_module
wan_module.ftfy = ftfy

# Fix scaled_dot_product_attention for Wan compatibility
_orig_sdpa = F.scaled_dot_product_attention
def safe_sdpa(q, k, v, m=None, d=0.0, c=False, **kwargs):
    kwargs.pop('enable_gqa', None)
    return _orig_sdpa(q, k, v, m, d, c, **kwargs)
F.scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention = safe_sdpa

# --- CONFIGURATION ---
MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
REF_IMAGE = "tshirt_final_20260116_154007.png"
OUTPUT_DIR = "./outputs"
REAL_ESRGAN_BIN = "./bin/realesrgan-ncnn-vulkan"
RIFE_BIN = "./bin/rife/rife-ncnn-vulkan"
RIFE_MODEL_DIR = "./bin/rife/rife-v4.6"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_segment(prompt, image_path, filename):
    print(f"\n--- Generating: {filename} ---")
    # On Blackwell 6000 PRO, we don't need CPU offload. 
    # We load everything into VRAM for maximum speed.
    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16
    ).to("cuda")
    
    ref_image = load_image(image_path)
    output = pipe(
        prompt=prompt, 
        image=ref_image, 
        num_frames=81, 
        guidance_scale=5.0, 
        num_inference_steps=30
    ).frames[0]
    
    path = os.path.join(OUTPUT_DIR, f"{filename}.mp4")
    export_to_video(output, path, fps=15)
    
    # Clean up between segments
    del pipe
    torch.cuda.empty_cache()
    return path

def finish_production(p1, p2, project_name):
    print("\n--- Finalizing Production ---")
    c1, c2 = VideoFileClip(p1), VideoFileClip(p2)
    
    # Concatenate base clips
    combined_path = os.path.join(OUTPUT_DIR, f"{project_name}_combined.mp4")
    concatenate_videoclips([c1, c2]).write_videofile(combined_path, codec='libx264', fps=15)
    
    # Post-Process: Upscale then Smooth
    upscaled = os.path.join(OUTPUT_DIR, f"{project_name}_upscaled.mp4")
    subprocess.run([REAL_ESRGAN_BIN, "-i", combined_path, "-o", upscaled, "-s", "2", "--face_enhance"], check=True)
    
    final = os.path.join(OUTPUT_DIR, f"{project_name}_final.mp4")
    subprocess.run([RIFE_BIN, "-i", upscaled, "-o", final, "-m", RIFE_MODEL_DIR], check=True)
    return final

if __name__ == "__main__":
    # Describe the Swan T-Shirt exactly as it appears
    PROMPT_1 = "A high-quality black t-shirt with a white swan logo in a black circle. The t-shirt is centered under a single cinematic spotlight. It slowly rotates 360 degrees. Hyper-realistic fabric."
    PROMPT_2 = "The white swan logo on the chest of the black t-shirt suddenly pops out and floats toward the camera, becoming a 3D glowing object. The words 'COMING SOON' appear in bold white neon text underneath it."

    # Execute workflow
    p1 = generate_segment(PROMPT_1, REF_IMAGE, "rotation")
    p2 = generate_segment(PROMPT_2, REF_IMAGE, "reveal")
    final_video = finish_production(p1, p2, "swan_teaser")
    
    print(f"\nSUCCESS! Teaser ready at: {final_video}")