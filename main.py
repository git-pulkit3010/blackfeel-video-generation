import torch
import subprocess
import os
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

# --- CONFIGURATION ---
MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
BIN_PATH = "./bin"  # Where we installed the tools in setup.sh
OUTPUT_DIR = "./outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_base_video(prompt, image_path, filename):
    print(f"--- Step 1: Generating Base Video using {MODEL_ID} ---")
    
    # Load Model (Optimized for 48GB VRAM)
    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16
    )
    pipe.to("cuda")
    
    # Enable offloading if you are on a 24GB card (Optional, slows it down)
    # pipe.enable_model_cpu_offload() 

    # Load User Attachment
    ref_image = load_image(image_path)

    # Generate
    output = pipe(
        prompt=prompt,
        image=ref_image,
        num_frames=81,         # Wan native length
        guidance_scale=5.0,    # How strictly to follow prompt
        num_inference_steps=30 
    ).frames[0]

    temp_path = os.path.join(OUTPUT_DIR, f"{filename}_base.mp4")
    export_to_video(output, temp_path, fps=15) # Wan native is usually 15-16fps
    
    # Free up VRAM for the next steps (Crucial!)
    del pipe
    torch.cuda.empty_cache()
    
    return temp_path

def upscale_video(input_path, filename):
    print("--- Step 2: Upscaling with Real-ESRGAN (4x + Face Fix) ---")
    output_path = os.path.join(OUTPUT_DIR, f"{filename}_upscaled.mp4")
    
    # We call the binary directly. 
    # -s 2: Scale 2x (720p -> 1440p). Use 4 for 4K.
    # --face_enhance: Magic switch for fixing faces.
    cmd = [
        f"{BIN_PATH}/realesrgan",
        "-i", input_path,
        "-o", output_path,
        "-s", "2", 
        "--face_enhance" 
    ]
    
    subprocess.run(cmd, check=True)
    return output_path

def smooth_video(input_path, filename):
    print("--- Step 3: Smoothing with RIFE (Interpolation) ---")
    # RIFE takes a folder of frames, but the NCNN binary can handle video files 
    # depending on the version. If the binary expects folders, we use ffmpeg.
    # PRO TIP: The simplest way with the binary is usually to just let it run.
    # However, to be safe via script, we will use the binary's default behavior.
    
    output_path = os.path.join(OUTPUT_DIR, f"{filename}_final.mp4")
    
    # usage: ./rife -i input.mp4 -o output.mp4
    cmd = [
        f"{BIN_PATH}/rife",
        "-i", input_path,
        "-o", output_path,
        "-m", "rife-v4.6" # Use the model included in the zip
    ]
    
    subprocess.run(cmd, check=True)
    return output_path

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # USER INPUTS
    PROMPT = "A cyberpunk detective smoking a cigarette, neon rain, highly detailed"
    ATTACHMENT = "my_reference_photo.jpg" 
    PROJECT_NAME = "cyberpunk_scene_01"

    # 1. Generate
    base_vid = generate_base_video(PROMPT, ATTACHMENT, PROJECT_NAME)
    
    # 2. Upscale
    upscaled_vid = upscale_video(base_vid, PROJECT_NAME)
    
    # 3. Smooth
    final_vid = smooth_video(upscaled_vid, PROJECT_NAME)
    
    print(f"DONE! Final video saved at: {final_vid}")