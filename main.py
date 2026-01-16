import torch
import subprocess
import os
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

# --- CONFIGURATION ---
# Wan 2.1 14B model requires ~30GB disk and 48GB VRAM
MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
OUTPUT_DIR = "./outputs"

# Updated paths based on your specific 'tree' output
REAL_ESRGAN_BIN = "./bin/realesrgan-ncnn-vulkan"
RIFE_BIN = "./bin/rife/rife-ncnn-vulkan"
RIFE_MODEL_DIR = "./bin/rife/rife-v4.6"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_base_video(prompt, image_path, filename):
    print(f"--- Step 1: Generating Base Video using {MODEL_ID} ---")
    
    # Load Model (Optimized for 48GB VRAM cards like A6000/A40)
    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16
    )
    pipe.to("cuda")
    
    # Load reference image
    ref_image = load_image(image_path)

    # Generate using native Wan 2.1 settings
    output = pipe(
        prompt=prompt,
        image=ref_image,
        num_frames=81,         
        guidance_scale=5.0,    
        num_inference_steps=30 
    ).frames[0]

    temp_path = os.path.join(OUTPUT_DIR, f"{filename}_base.mp4")
    # Exporting at 15fps as Wan native is typically 15-16fps
    export_to_video(output, temp_path, fps=15) 
    
    # Free up VRAM to prevent Out of Memory errors during upscaling
    del pipe
    torch.cuda.empty_cache()
    
    return temp_path

def upscale_video(input_path, filename):
    print("--- Step 2: Upscaling with Real-ESRGAN (2x + Face Fix) ---")
    output_path = os.path.join(OUTPUT_DIR, f"{filename}_upscaled.mp4")
    
    # Using standalone NCNN binary to avoid Python dependency conflicts
    cmd = [
        REAL_ESRGAN_BIN,
        "-i", input_path,
        "-o", output_path,
        "-s", "2", 
        "--face_enhance" 
    ]
    
    subprocess.run(cmd, check=True)
    return output_path

def smooth_video(input_path, filename):
    print("--- Step 3: Smoothing with RIFE (Interpolation) ---")
    output_path = os.path.join(OUTPUT_DIR, f"{filename}_final.mp4")
    
    # RIFE interpolation creates smoother motion quality
    cmd = [
        RIFE_BIN,
        "-i", input_path,
        "-o", output_path,
        "-m", RIFE_MODEL_DIR
    ]
    
    subprocess.run(cmd, check=True)
    return output_path

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # USER INPUTS
    PROMPT = "A cyberpunk detective smoking a cigarette, neon rain, highly detailed"
    ATTACHMENT = "my_reference_photo.jpg" # Ensure this file exists in your folder!
    PROJECT_NAME = "cyberpunk_scene_01"

    if not os.path.exists(ATTACHMENT):
        print(f"ERROR: Reference image '{ATTACHMENT}' not found.")
    else:
        # 1. Generate Base Video
        base_vid = generate_base_video(PROMPT, ATTACHMENT, PROJECT_NAME)
        
        # 2. Upscale (720p -> 1440p)
        upscaled_vid = upscale_video(base_vid, PROJECT_NAME)
        
        # 3. Smooth (Interpolation)
        final_vid = smooth_video(upscaled_vid, PROJECT_NAME)
        
        print(f"DONE! Final video saved at: {final_vid}")