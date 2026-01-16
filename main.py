import torch
import torch.nn.functional as F
import subprocess, os, types, ftfy
from PIL import Image
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from moviepy.editor import VideoFileClip, concatenate_videoclips

# Inject ftfy for text cleaning required by the Wan pipeline
import diffusers.pipelines.wan.pipeline_wan_i2v as wan_module
wan_module.ftfy = ftfy

# Fix SDPA for Blackwell compatibility
_orig_sdpa = F.scaled_dot_product_attention
def safe_sdpa(q, k, v, m=None, d=0.0, c=False, **kwargs):
    kwargs.pop('enable_gqa', None)
    return _orig_sdpa(q, k, v, m, d, c, **kwargs)
F.scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention = safe_sdpa

# CONFIGURATION
MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
REF_IMAGE = "my_reference_photo.jpg"
BIN_DIR = "./bin"

def generate_segment(prompt, image_path, filename):
    print(f"\n--- Generating: {filename} ---")
    # Blackwell has 96GB VRAM; no need for CPU offloading hacks
    pipe = WanImageToVideoPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("cuda")
    
    ref_image = load_image(image_path)
    output = pipe(prompt=prompt, image=ref_image, num_frames=81, num_inference_steps=30).frames[0]
    
    path = f"./outputs/{filename}.mp4"
    os.makedirs("./outputs", exist_ok=True)
    export_to_video(output, path, fps=15)
    
    del pipe
    torch.cuda.empty_cache()
    return path

def process_final(p1, p2):
    print("\n--- Finalizing Production ---")
    c1, c2 = VideoFileClip(p1), VideoFileClip(p2)
    combined = "./outputs/combined.mp4"
    concatenate_videoclips([c1, c2]).write_videofile(combined, codec='libx264', fps=15)
    
    upscaled = "./outputs/upscaled.mp4"
    subprocess.run([f"{BIN_DIR}/realesrgan-ncnn-vulkan", "-i", combined, "-o", upscaled, "-s", "2", "--face_enhance"], check=True)
    
    final = "./outputs/swan_teaser_final.mp4"
    subprocess.run([f"{BIN_DIR}/rife/rife-ncnn-vulkan", "-i", upscaled, "-o", final, "-m", f"{BIN_DIR}/rife/rife-v4.6"], check=True)
    return final

if __name__ == "__main__":
    P1 = "A black t-shirt with a white swan logo inside a circle. The t-shirt is centered under a spotlight and rotates 360 degrees slowly. Cinematic lighting, dark background."
    P2 = "The white swan logo on the t-shirt pops out, enlarges, and moves toward the camera as a 3D object. The glowing white text 'COMING SOON' appears dramatically below it."

    part1 = generate_segment(P1, REF_IMAGE, "rotation")
    part2 = generate_segment(P2, REF_IMAGE, "reveal")
    print(f"DONE! Final Video: {process_final(part1, part2)}")