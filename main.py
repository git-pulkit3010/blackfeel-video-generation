# main.py -- compatibility shim + original pipeline code
# Fixes: AttributeError: module 'torch.utils._pytree' has no attribute 'register_pytree_node'
# Strategy: create aliases so both register_pytree_node and _register_pytree_node exist.

import importlib
import sys
import os

# -------------------------
# Compatibility shim (must run BEFORE importing transformers / diffusers)
# -------------------------
try:
    # Try to import the internal module directly
    _torch_pytree = importlib.import_module("torch.utils._pytree")
except Exception:
    # Fallback: try to access via torch.utils attribute if torch is already imported
    import torch as _torch_tmp
    _torch_pytree = getattr(_torch_tmp.utils, "_pytree", None)

if _torch_pytree is not None:
    # If only the underscored name exists, create the public alias
    if not hasattr(_torch_pytree, "register_pytree_node") and hasattr(_torch_pytree, "_register_pytree_node"):
        _torch_pytree.register_pytree_node = _torch_pytree._register_pytree_node

    # If only the public name exists, create the underscored alias (reverse case)
    if not hasattr(_torch_pytree, "_register_pytree_node") and hasattr(_torch_pytree, "register_pytree_node"):
        _torch_pytree._register_pytree_node = _torch_pytree.register_pytree_node

    # Put the patched module back where libraries expect it
    try:
        import torch as _torch
        if not hasattr(_torch.utils, "_pytree"):
            _torch.utils._pytree = _torch_pytree
        else:
            # replace with our patched reference for safety
            _torch.utils._pytree = _torch_pytree
    except Exception:
        # If torch isn't importable here, that's okay — import later will use sys.modules entry
        sys.modules["torch.utils._pytree"] = _torch_pytree
else:
    # If we couldn't locate torch.utils._pytree, keep going — import errors will surface normally
    pass

# -------------------------
# Now import libraries (safe to import transformers / diffusers)
# -------------------------
import torch
import subprocess
import types

# Keep XPU mock you had (useful on systems without XPU)
if not hasattr(torch, 'xpu'):
    class MockXPU:
        @staticmethod
        def is_available():
            return False
        
        @staticmethod  
        def device_count():
            return 0
        
        @staticmethod
        def empty_cache():
            pass
        
        @staticmethod
        def current_device():
            return 0
        
        @staticmethod
        def manual_seed(seed):
            print(f"Warning: torch.xpu.manual_seed({seed}) called on non-XPU system")
            pass
        
        @staticmethod
        def get_rng_state(device='xpu'):
            return torch.get_rng_state()
        
        @staticmethod
        def set_rng_state(new_state, device='xpu'):
            torch.set_rng_state(new_state)
    
    torch.xpu = MockXPU()

# Now import diffusers and utilities (your pipeline depends on this)
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

# --- CONFIGURATION ---
MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
OUTPUT_DIR = "./outputs"

# BINARY PATHS (Matches your setup exactly)
REAL_ESRGAN_BIN = "./bin/realesrgan-ncnn-vulkan"
RIFE_BIN = "./bin/rife/rife-ncnn-vulkan"
RIFE_MODEL_DIR = "./bin/rife/rife-v4.6"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_base_video(prompt, image_path, filename):
    print(f"--- Step 1: Generating Base Video using {MODEL_ID} ---")
    
    # Load Model (Optimized for 48GB VRAM)
    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16
    )
    pipe.to("cuda")
    
    # Load reference image
    ref_image = load_image(image_path)

    # Generate
    output = pipe(
        prompt=prompt,
        image=ref_image,
        num_frames=81,         
        guidance_scale=5.0,    
        num_inference_steps=30 
    ).frames[0]

    temp_path = os.path.join(OUTPUT_DIR, f"{filename}_base.mp4")
    export_to_video(output, temp_path, fps=15) 
    
    # Cleanup VRAM
    del pipe
    torch.cuda.empty_cache()
    
    return temp_path

def upscale_video(input_path, filename):
    print("--- Step 2: Upscaling with Real-ESRGAN (2x + Face Fix) ---")
    output_path = os.path.join(OUTPUT_DIR, f"{filename}_upscaled.mp4")
    
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
    PROMPT = "A cyberpunk detective smoking a cigarette, neon rain, highly detailed"
    ATTACHMENT = "my_reference_photo.jpg" 
    PROJECT_NAME = "cyberpunk_scene_01"

    if not os.path.exists(ATTACHMENT):
        print(f"ERROR: Reference image '{ATTACHMENT}' not found.")
    else:
        # 1. Generate
        base_vid = generate_base_video(PROMPT, ATTACHMENT, PROJECT_NAME)
        
        # 2. Upscale
        upscaled_vid = upscale_video(base_vid, PROJECT_NAME)
        
        # 3. Smooth
        final_vid = smooth_video(upscaled_vid, PROJECT_NAME)
        
        print(f"DONE! Final video saved at: {final_vid}")
