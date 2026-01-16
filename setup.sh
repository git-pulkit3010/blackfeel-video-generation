#!/bin/bash
# setup.sh - Run this in your RunPod terminal

echo "--- Installing Python Dependencies ---"
# We install the bleeding-edge Diffusers to ensure Wan 2.1 support
pip install --upgrade pip
pip install git+https://github.com/huggingface/diffusers.git
pip install transformers accelerate opencv-python imageio moviepy

echo "--- Downloading Standalone Upscalers (NCNN) ---"
# Create a 'bin' folder for our tools
mkdir -p bin
cd bin

# 1. Download Real-ESRGAN (Upscaler)
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip
unzip realesrgan-ncnn-vulkan-20220424-ubuntu.zip
mv realesrgan-ncnn-vulkan realesrgan
chmod +x realesrgan

# 2. Download RIFE (Frame Interpolation/Smoothing)
# We use the 'rife-ncnn-vulkan' implementation
wget https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-ubuntu.zip
unzip rife-ncnn-vulkan-20221029-ubuntu.zip
mv rife-ncnn-vulkan rife
chmod +x rife

echo "--- Setup Complete! ---"