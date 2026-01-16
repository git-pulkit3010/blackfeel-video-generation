#!/bin/bash
# setup.sh - Optimized for RunPod PyTorch 2.4.0 + RTX 6000 PRO

echo "--- Updating System and Installing Utilities ---"
apt-get update && apt-get install -y unzip wget git ffmpeg

echo "--- Installing Python Production Stack ---"
pip install --upgrade pip
# Install the specific Wan-enabled Diffusers branch
pip install git+https://github.com/huggingface/diffusers.git
# Install latest transformers and core deps
pip install transformers accelerate opencv-python imageio moviepy ftfy
# Downgrade NumPy to 1.x to maintain compatibility with PyTorch 2.4/OpenCV
pip install "numpy<2"

echo "--- Setting Up Standalone Binaries (NCNN) ---"
mkdir -p bin
cd bin

# 1. Setup Real-ESRGAN (Upscaler)
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip
unzip realesrgan-ncnn-vulkan-20220424-ubuntu.zip
# Cleanup zip and demo files
rm realesrgan-ncnn-vulkan-20220424-ubuntu.zip input.jpg input2.jpg onepiece_demo.mp4
chmod +x realesrgan-ncnn-vulkan

# 2. Setup RIFE (Frame Interpolation)
wget https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-ubuntu.zip
unzip rife-ncnn-vulkan-20221029-ubuntu.zip
mv rife-ncnn-vulkan-20221029-ubuntu rife
rm rife-ncnn-vulkan-20221029-ubuntu.zip
chmod +x rife/rife-ncnn-vulkan

cd ..

echo "--- Setup Complete! ---"
echo "You can now run: python main.py"