#!/bin/bash
# Pacific Yew Density Model - MINIMAL Environment Setup
# ======================================================
# This is a minimal installation that only installs absolutely required packages.
# Use this if you're running low on disk space.
#
# Usage: bash setup_yew_env_minimal.sh

set -e  # Exit on error

echo "======================================================================"
echo "Pacific Yew Density Model - MINIMAL Environment Setup"
echo "======================================================================"
echo ""
echo "This will create a minimal environment with only essential packages."
echo "Estimated space needed: ~3GB"
echo ""

# Environment name
ENV_NAME="yew_pytorch"

# Check available disk space
echo "Checking available disk space..."
AVAILABLE_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
echo "Available space: ${AVAILABLE_SPACE}GB"

if [ "$AVAILABLE_SPACE" -lt 5 ]; then
    echo "WARNING: Less than 5GB available. You may need to free up space."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

echo ""
echo "Found conda: $(which conda)"
echo "Conda version: $(conda --version)"
echo ""

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "WARNING: Environment '${ENV_NAME}' already exists!"
    read -p "Do you want to remove it and create a new one? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Exiting."
        exit 0
    fi
fi

# Clean conda cache to free space
echo "Cleaning conda cache to free space..."
conda clean --all -y

echo ""
echo "Creating conda environment with Python 3.10..."
conda create -n ${ENV_NAME} python=3.10 -y

echo ""
echo "Installing packages (this may take 10-15 minutes)..."
echo ""

# Install everything via pip to save space
echo "Installing PyTorch with CUDA 11.8..."
conda run -n ${ENV_NAME} pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "Installing scientific computing packages..."
conda run -n ${ENV_NAME} pip install --no-cache-dir \
    numpy \
    pandas \
    scipy \
    matplotlib \
    seaborn \
    scikit-learn \
    pillow \
    tqdm

echo ""
echo "======================================================================"
echo "Installation Complete!"
echo "======================================================================"
echo ""
echo "Minimal environment '${ENV_NAME}' has been created successfully."
echo ""
echo "Installed packages:"
echo "  - PyTorch 2.1.0 with CUDA 11.8"
echo "  - NumPy, Pandas, SciPy"
echo "  - Matplotlib, Seaborn"
echo "  - Scikit-learn"
echo "  - Pillow (image processing)"
echo ""
echo "To activate the environment:"
echo "    conda activate ${ENV_NAME}"
echo ""
echo "To test PyTorch and CUDA:"
echo "    conda activate ${ENV_NAME}"
echo "    python test_environment.py"
echo ""
echo "To run the yew density model:"
echo "    conda activate ${ENV_NAME}"
echo "    python yew_density_model.py"
echo ""
echo "To install additional packages later:"
echo "    conda activate ${ENV_NAME}"
echo "    pip install <package-name>"
echo ""
echo "======================================================================"
