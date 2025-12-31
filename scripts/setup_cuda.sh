#!/bin/bash
# =============================================================================
# Distillix CUDA Setup Script
# =============================================================================
# Sets up NVIDIA drivers and PyTorch for training on RTX 2080 Super
#
# Usage:
#   chmod +x scripts/setup_cuda.sh
#   ./scripts/setup_cuda.sh
#
# Copyright (c) 2025 Distillix. All Rights Reserved.
# =============================================================================

set -e

echo "========================================"
echo "Distillix CUDA Environment Setup"
echo "========================================"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (sudo ./scripts/setup_cuda.sh)"
    exit 1
fi

# Detect GPU
echo "Detecting GPU..."
if lspci | grep -i nvidia > /dev/null; then
    GPU_MODEL=$(lspci | grep -i nvidia | head -1 | cut -d':' -f3)
    echo "Found GPU: $GPU_MODEL"
else
    echo "No NVIDIA GPU detected!"
    echo "This script is for NVIDIA GPU setup only."
    exit 1
fi

# Check current driver status
echo ""
echo "Checking NVIDIA driver status..."
if nvidia-smi > /dev/null 2>&1; then
    echo "NVIDIA driver already installed:"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    SKIP_DRIVER=true
else
    echo "NVIDIA driver not installed or not working"
    SKIP_DRIVER=false
fi

# Install NVIDIA driver if needed
if [ "$SKIP_DRIVER" = false ]; then
    echo ""
    echo "Installing NVIDIA drivers..."
    
    # Update package list
    apt-get update
    
    # Install prerequisites
    apt-get install -y build-essential dkms
    
    # Add NVIDIA package repository
    apt-get install -y software-properties-common
    add-apt-repository -y ppa:graphics-drivers/ppa
    apt-get update
    
    # Install recommended driver
    # For RTX 2080 Super (Turing), driver 535+ is recommended
    apt-get install -y nvidia-driver-535
    
    echo ""
    echo "NVIDIA driver installed. A reboot is required."
    echo "After reboot, run this script again to continue setup."
    echo ""
    read -p "Reboot now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        reboot
    fi
    exit 0
fi

# Verify driver is working
echo ""
echo "Verifying NVIDIA driver..."
nvidia-smi

# Create Python virtual environment
echo ""
echo "Setting up Python environment..."

# Install Python venv if not available
apt-get install -y python3-venv python3-pip

# Create virtual environment
VENV_DIR="/root/distillix/.venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "Created virtual environment at $VENV_DIR"
else
    echo "Virtual environment already exists at $VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA support
echo ""
echo "Installing PyTorch with CUDA support..."

# Detect CUDA version from driver
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1,2)
echo "Detected CUDA capability: $CUDA_VERSION"

# Install PyTorch (using CUDA 12.1 wheels for compatibility)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch CUDA
echo ""
echo "Verifying PyTorch CUDA installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Install remaining dependencies
echo ""
echo "Installing project dependencies..."
pip install -r /root/distillix/requirements.txt

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "
import torch
import transformers
print('All core packages installed successfully!')
print(f'  torch: {torch.__version__}')
print(f'  transformers: {transformers.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
"

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To activate the environment:"
echo "  source /root/distillix/.venv/bin/activate"
echo ""
echo "To test the BitNet layer:"
echo "  python -m smelter.bitnet"
echo ""
echo "To start training:"
echo "  python -m smelter.train"
echo ""
