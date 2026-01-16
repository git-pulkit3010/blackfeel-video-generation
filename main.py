
import torch
import torch.nn.functional as F
import subprocess
import os
import types
import importlib
import sys

# ==============================================================================
#  THE COMPATIBILITY LAYER (Patches for PyTorch 2.1 on RunPod)
# ==============================================================================

# --- PATCH 1: Fix Transformers 4.57+ compatibility ---
try:
    import torch.utils._pytree as _pytree
    _orig_register = getattr(_pytree, "register_pytree_node", 
                     getattr(_pytree, "_register_pytree_node", None))
    def safe_register_pytree_node(typ, flatten_func, unflatten_func, serialized_type_name=None):
        return _orig_register(typ, flatten_func, unflatten_func)
    if _orig_register:
        _pytree.register_pytree_node = safe_register_pytree_node
        _pytree._register_pytree_node = safe_register_pytree_node
        print(">> Patched torch.utils._pytree.")
except Exception: pass

# --- PATCH 2: Mock missing RMSNorm (Required by Wan 2.1) ---
if not hasattr(torch.nn, 'RMSNorm'):
    class RMSNorm(torch.nn.Module):
        def __init__(self, dim, eps=1e-6, elementwise_affine=True):
            super().__init__()
            self.eps = eps
            self.weight = torch.nn.Parameter(torch.ones(dim)) if elementwise_affine else None
        def forward(self, x):
            norm_x = torch.mean(x**2, dim=-1, keepdim=True)
            x_normed = x * torch.rsqrt(norm_x + self.eps)
            if self.weight is not None:
                return x_normed * self.weight
            return x_normed
    torch.nn.RMSNorm = RMSNorm
    print(">> Patched torch.nn.RMSNorm.")

# --- PATCH 3: Mock DeviceMesh (Missing in 2.1) ---
if not hasattr(torch.distributed, 'device_mesh'):
    mock_mesh_module = types.SimpleNamespace()
    mock_mesh_module.DeviceMesh = type('MockDeviceMesh', (), {})
    torch.distributed.device_mesh = mock_mesh_module
    print(">> Patched torch.distributed.device_mesh.")

# --- PATCH 4: Robust XPU Mock ---
if not hasattr(torch, 'xpu'):
    class MockXPU:
        @staticmethod
        def is_available(): return False
        @staticmethod
        def device_count(): return 0
        @staticmethod
        def empty_cache(): pass
        @staticmethod
        def current_device(): return 0
        @staticmethod
        def manual_seed(seed): pass
        @staticmethod
        def seed(): return 0
        @staticmethod
        def get_rng_state(device='xpu'): return torch.ByteTensor([])
        @staticmethod
        def set_rng_state(new_state, device='xpu'): pass
    torch.xpu = MockXPU()
    print(">> Patched torch.xpu.")

# --- PATCH 6: Fix scaled_dot_product_attention (Drop 'enable_gqa') ---
# PyTorch 2.1 does not support 'enable_gqa'. We intercept the call and remove it.
_orig_sdpa = F.scaled_dot_product_attention
def safe_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
    # Drop unsupported arguments
    kwargs.pop('enable_gqa', None) 
    return _orig_sdpa(query, key, value, attn_mask, dropout_p, is_causal, **kwargs)
F.scaled_dot_product_attention = safe_sdpa
torch.nn.functional.scaled_dot_product_attention = safe_sdpa
print(">> Patched scaled_dot_product_attention.")

# ==============================================================================
#  MAIN PIPELINE CODE
# ==============================================================================
import ftfy 
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

# --- PATCH 5: Inject missing 'ftfy' into the pipeline module ---
try:
    import diffusers.pipelines.wan.pipeline_wan_i2v as wan_module
    wan_module.ftfy = ftfy
    print(">> Injected 'ftfy' into Wan pipeline module.")
except ImportError:
    pass 

MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
OUTPUT_DIR = "./outputs"
REAL_ESRGAN_BIN = "./bin/realesrgan-ncnn-vulkan"
RIFE_BIN = "./bin/rife/rife-ncnn-vulkan"
RIFE_MODEL_DIR = "./bin/rife/rife-v4.6"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_base_video(prompt, image_path, filename):
    print(f"--- Step 1: Generating Base Video using {MODEL_ID} ---")
    
    # 1. Load Model
    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16
    )
    
    # 2. Enable CPU Offloading (Critical for 48GB VRAM)
    pipe.enable_model_cpu_offload()
    
    ref_image = load_image(image_path)

    # 3. Generate
    output = pipe(
        prompt=prompt,
        image=ref_image,
        num_frames=81,         
        guidance_scale=5.0,    
        num_inference_steps=30 
    ).frames[0]

    temp_path = os.path.join(OUTPUT_DIR, f"{filename}_base.mp4")
    export_to_video(output, temp_path, fps=15) 
    
    # Clean up
    del pipe
    torch.cuda.empty_cache()
    
    return temp_path

def upscale_video(input_path, filename):
    print("--- Step 2: Upscaling ---")
    output_path = os.path.join(OUTPUT_DIR, f"{filename}_upscaled.mp4")
    subprocess.run([REAL_ESRGAN_BIN, "-i", input_path, "-o", output_path, "-s", "2", "--face_enhance"], check=True)
    return output_path

def smooth_video(input_path, filename):
    print("--- Step 3: Smoothing ---")
    output_path = os.path.join(OUTPUT_DIR, f"{filename}_final.mp4")
    subprocess.run([RIFE_BIN, "-i", input_path, "-o", output_path, "-m", RIFE_MODEL_DIR], check=True)
    return output_path

if __name__ == "__main__":
    PROMPT = "A cyberpunk detective smoking a cigarette, neon rain, highly detailed"
    ATTACHMENT = "my_reference_photo.jpg" 
    PROJECT_NAME = "cyberpunk_scene_01"

    if os.path.exists(ATTACHMENT):
        base_vid = generate_base_video(PROMPT, ATTACHMENT, PROJECT_NAME)
        upscaled_vid = upscale_video(base_vid, PROJECT_NAME)
        final_vid = smooth_video(upscaled_vid, PROJECT_NAME)
        print(f"DONE! Final video: {final_vid}")
    else:
        print(f"ERROR: {ATTACHMENT} not found.")
