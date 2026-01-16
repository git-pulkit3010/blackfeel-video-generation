import torch
import torch.nn.functional as F
import subprocess
import os
import types
from PIL import Image
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from moviepy.editor import VideoFileClip, concatenate_videoclips
import ftfy

# --- MINIMAL COMPATIBILITY PATCHES ---
# Required for Wan 2.1 in current Diffusers/PyTorch 2.8+ environments
import diffusers.pipelines.wan.pipeline_wan_i2v as wan_module
wan_module.ftfy = ftfy

_orig_sdpa = F.scaled_dot_product_attention
def safe_sdpa(q, k, v, m=None, d=0.0, c=False, **kwargs):
    kwargs.pop('enable_gqa', None)
    return _orig_sdpa(q, k, v, m, d, c, **kwargs)
F.scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention = safe_sdpa

# --- CONFIGURATION ---
MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
# USING THE FILENAME YOU PROVIDED
REF_IMAGE = "tshirt_final_20260116_154007.png" 
OUTPUT_DIR = "./outputs"
BIN_DIR = "./bin"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_segment(prompt, image_path, filename):
    print(f"\n--- Generating Segment: {filename} ---")
    # Loaded directly to CUDA for Blackwell speed (no offloading)
    pipe = WanImageToVideoPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("cuda")
    
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
    
    del pipe
    torch.cuda.empty_cache()
    return path

def finish_teaser(p1, p2, project_name):
    print("\n--- Finalizing Production (Stitch -> Upscale -> Smooth) ---")
    c1, c2 = VideoFileClip(p1), VideoFileClip(p2)
    
    # Extract frame for continuity check (internal)
    img = Image.fromarray(c1.get_frame(c1.duration - 0.01))
    img.save(os.path.join(OUTPUT_DIR, "transition_frame.jpg"))
    
    # 1. Concatenate
    combined_path = os.path.join(OUTPUT_DIR, f"{project_name}_combined.mp4")
    concatenate_videoclips([c1, c2]).write_videofile(combined_path, codec='libx264', fps=15)
    
    # 2. Upscale (2x for 1440p High Quality)
    upscaled = os.path.join(OUTPUT_DIR, f"{project_name}_upscaled.mp4")
    subprocess.run([f"{BIN_DIR}/realesrgan", "-i", combined_path, "-o", upscaled, "-s", "2", "--face_enhance"], check=True)
    
    # 3. Smooth (Interpolate to 30fps+)
    final = os.path.join(OUTPUT_DIR, f"{project_name}_final.mp4")
    subprocess.run([f"{BIN_DIR}/rife/rife-ncnn-vulkan", "-i", upscaled, "-o", final, "-m", f"{BIN_DIR}/rife/rife-v4.6"], check=True)
    
    return final

if __name__ == "__main__":
    # Part 1: Spotlight Rotation
    P1 = "A high-quality black t-shirt with a white swan logo inside a black circle. The t-shirt is centered under a sharp cinematic spotlight on a dark stage. The t-shirt rotates slowly 360 degrees. Realistic fabric folds."
    
    # Part 2: Logo Reveal
    P2 = "The white swan logo on the black t-shirt pops out from the fabric and floats forward toward the camera, becoming an enlarged 3D glowing object. The words 'COMING SOON' appear in bold white glowing text below it."

    if not os.path.exists(REF_IMAGE):
        print(f"ERROR: {REF_IMAGE} not found.")
    else:
        part1_vid = generate_segment(P1, REF_IMAGE, "part1_rotation")
        part2_vid = generate_segment(P2, REF_IMAGE, "part2_reveal")
        
        final_path = finish_teaser(part1_vid, part2_vid, "swan_teaser")
        print(f"\nSUCCESS! Teaser completed: {final_path}")