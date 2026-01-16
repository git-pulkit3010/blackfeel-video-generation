#!/bin/bash
# setup.sh - Optimized for RTX 6000 PRO (Blackwell)

echo "--- Installing System Utilities ---"
apt-get update && apt-get install -y unzip wget git ffmpeg

echo "--- Upgrading to Blackwell-Compatible PyTorch ---"
# Remove default versions that lack Blackwell kernels
pip uninstall -y torch torchvision torchaudio diffusers

# Install PyTorch nightly with CUDA 12.8 support for Blackwell sm_120
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

echo "--- Installing Production Libraries ---"
pip install git+https://github.com/huggingface/diffusers.git
pip install transformers accelerate opencv-python imageio moviepy ftfy "numpy<2"

echo "--- Setting Up Standalone Binaries ---"
mkdir -p bin && cd bin
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip
unzip realesrgan-ncnn-vulkan-20220424-ubuntu.zip
rm realesrgan-ncnn-vulkan-20220424-ubuntu.zip input.jpg input2.jpg onepiece_demo.mp4
chmod +x realesrgan-ncnn-vulkan

wget https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-ubuntu.zip
unzip rife-ncnn-vulkan-20221029-ubuntu.zip
mv rife-ncnn-vulkan-20221029-ubuntu rife
rm rife-ncnn-vulkan-20221029-ubuntu.zip
chmod +x rife/rife-ncnn-vulkan
cd ..

echo "--- Setup Complete! Run 'python main.py' next. ---"