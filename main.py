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

# ==============================================================================
#  COMPATIBILITY PATCHES (Kept for safety in PyTorch 2.1)
# ==============================================================================
try:
    import torch.utils._pytree as _pytree
    _orig_register = getattr(_pytree, "register_pytree_node", getattr(_pytree, "_register_pytree_node", None))
    def safe_register(typ, flat, unflat, serialized_type_name=None): return _orig_register(typ, flat, unflat)
    if _orig_register: _pytree.register_pytree_node = _pytree._register_pytree_node = safe_register
except Exception: pass

if not hasattr(torch.nn, 'RMSNorm'):
    class RMSNorm(torch.nn.Module):
        def __init__(self, dim, eps=1e-6, elementwise_affine=True):
            super().__init__()
            self.eps = eps
            self.weight = torch.nn.Parameter(torch.ones(dim)) if elementwise_affine else None
        def forward(self, x):
            norm_x = torch.mean(x**2, dim=-1, keepdim=True)
            return x * torch.rsqrt(norm_x + self.eps) * (self.weight if self.weight is not None else 1.0)
    torch.nn.RMSNorm = RMSNorm

if not hasattr(torch.distributed, 'device_mesh'):
    torch.distributed.device_mesh = types.SimpleNamespace(DeviceMesh=type('MockDeviceMesh', (), {}))

if not hasattr(torch, 'xpu'):
    torch.xpu = types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0, empty_cache=lambda: None, 
                                      current_device=lambda: 0, manual_seed=lambda s: None, seed=lambda: 0, 
                                      get_rng_state=lambda d='xpu': torch.ByteTensor([]), set_rng_state=lambda n, d='xpu': None)

_orig_sdpa = F.scaled_dot_product_attention
def safe_sdpa(q, k, v, m=None, d=0.0, c=False, **kwargs):
    kwargs.pop('enable_gqa', None)
    return _orig_sdpa(q, k, v, m, d, c, **kwargs)
F.scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention = safe_sdpa

# Inject ftfy
import diffusers.pipelines.wan.pipeline_wan_i2v as wan_module
wan_module.ftfy = ftfy

# ==============================================================================
#  PIPELINE CONFIGURATION
# ==============================================================================
MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
OUTPUT_DIR = "./outputs"
REAL_ESRGAN_BIN = "./bin/realesrgan-ncnn-vulkan"
RIFE_BIN = "./bin/rife/rife-ncnn-vulkan"
RIFE_MODEL_DIR = "./bin/rife/rife-v4.6"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_segment(prompt, image_path, filename):
    print(f"--- Generating Segment: {filename} ---")
    pipe = WanImageToVideoPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload() # Remove this line if you upgrade to H100/A100
    
    ref_image = load_image(image_path)
    output = pipe(prompt=prompt, image=ref_image, num_frames=81, guidance_scale=5.0, num_inference_steps=30).frames[0]
    
    path = os.path.join(OUTPUT_DIR, f"{filename}.mp4")
    export_to_video(output, path, fps=15)
    del pipe
    torch.cuda.empty_cache()
    return path

def finish_video(p1, p2, project_name):
    print("--- Finalizing Video (Concatenate -> Upscale -> Smooth) ---")
    c1, c2 = VideoFileClip(p1), VideoFileClip(p2)
    img = Image.fromarray(c1.get_frame(c1.duration - 0.01))
    img.save("last_frame.jpg")
    
    combined_path = os.path.join(OUTPUT_DIR, f"{project_name}_combined.mp4")
    concatenate_videoclips([c1, c2]).write_videofile(combined_path, codec='libx264', fps=15)
    
    upscaled = os.path.join(OUTPUT_DIR, f"{project_name}_upscaled.mp4")
    subprocess.run([REAL_ESRGAN_BIN, "-i", combined_path, "-o", upscaled, "-s", "2", "--face_enhance"], check=True)
    
    final = os.path.join(OUTPUT_DIR, f"{project_name}_final.mp4")
    subprocess.run([RIFE_BIN, "-i", upscaled, "-o", final, "-m", RIFE_MODEL_DIR], check=True)
    return final

if __name__ == "__main__":
    REF_IMAGE = "tshirt_final_20260116_154007.png"
    
    # PROMPT 1: 360 Rotation under spotlight
    P1 = "A black t-shirt featuring a white swan logo inside a black circle. The t-shirt is under a sharp cinematic spotlight on a dark stage. The t-shirt rotates slowly 360 degrees to show the front and back. High detail, 4k, realistic fabric."
    
    # PROMPT 2: Logo Pop & Coming Soon
    P2 = "The white swan logo on the black t-shirt suddenly detaches from the fabric and floats forward toward the camera. As it enlarges, the 3D glowing text 'COMING SOON' appears in bold white letters floating in the center of the screen."

    part1 = generate_segment(P1, REF_IMAGE, "part1_rotation")
    # In a real run, you'd use the last frame of part1 as the image for part2. 
    # For now, we use the REF_IMAGE to keep the script automated.
    part2 = generate_segment(P2, REF_IMAGE, "part2_reveal")
    
    final_teaser = finish_video(part1, part2, "swan_teaser")
    print(f"SUCCESS: {final_teaser}")