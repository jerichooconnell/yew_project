#!/bin/bash
# Pacific Yew Density Model - Conda Environment Setup Script
# ===========================================================
# This script creates a conda environment with all required packages
# for running the PyTorch-based yew density prediction model.
#
# System Requirements:
# - CUDA 11.8 or 12.x (GPU support)
# - Linux x86_64
# - ~5GB disk space for environment
#
# Usage: bash setup_yew_env.sh

set -e  # Exit on error

echo "======================================================================"
echo "Pacific Yew Density Model - Environment Setup"
echo "======================================================================"
echo ""

# Environment name
ENV_NAME="yew_pytorch"

# Check available disk space
echo "Checking available disk space..."
AVAILABLE_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
echo "Available space: ${AVAILABLE_SPACE}GB"

if [ "$AVAILABLE_SPACE" -lt 10 ]; then
    echo "WARNING: Less than 10GB available. Installation may fail."
    echo "Consider freeing up space or using the minimal install option."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

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
        echo "Exiting. Please use a different environment name or remove the existing one manually."
        exit 0
    fi
fi

echo "Creating new conda environment: ${ENV_NAME}"
echo "Python version: 3.10"
echo ""

# Create environment with Python 3.10
conda create -n ${ENV_NAME} python=3.10 -y

echo ""
echo "Environment created. Installing packages..."
echo ""

# Activate environment and install packages via pip (more space-efficient)
# Install PyTorch with CUDA 11.8 support using pip
echo "Installing PyTorch with CUDA 11.8 support (via pip - more space efficient)..."
conda run -n ${ENV_NAME} pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Install core scientific computing packages (minimal versions)
echo ""
echo "Installing core scientific packages..."
conda run -n ${ENV_NAME} pip install --no-cache-dir \
    numpy \
    pandas \
    scipy \
    matplotlib \
    seaborn \
    scikit-learn \
    pillow

# Install essential packages only (minimal install to save space)
echo ""
echo "Installing essential packages..."
conda run -n ${ENV_NAME} pip install --no-cache-dir \
    tqdm

echo ""
echo "Basic installation complete!"
echo ""
read -p "Install optional packages? (Jupyter, geospatial tools) [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Install Jupyter
    echo ""
    echo "Installing Jupyter..."
    conda run -n ${ENV_NAME} pip install --no-cache-dir \
        jupyter \
        jupyterlab \
        ipykernel \
        ipywidgets
    
    # Register kernel
    echo "Registering kernel with Jupyter..."
    conda run -n ${ENV_NAME} python -m ipykernel install --user --name ${ENV_NAME} --display-name "Python (yew_pytorch)"
    
    # Install geospatial packages
    echo ""
    read -p "Install geospatial packages (Earth Engine, etc.)? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Installing geospatial packages..."
        conda run -n ${ENV_NAME} pip install --no-cache-dir \
            earthengine-api \
            geopandas \
            rasterio
    fi
    
    # Install ML tools
    echo ""
    read -p "Install ML tracking tools (tensorboard, wandb)? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Installing ML tools..."
        conda run -n ${ENV_NAME} pip install --no-cache-dir \
            tensorboard \
            wandb
    fi
fi

# Skip kernel registration if Jupyter not installed
if conda run -n ${ENV_NAME} python -c "import jupyter" 2>/dev/null; then
    echo ""
    echo "Jupyter found, skipping additional kernel registration..."
else
    echo ""
    echo "Jupyter not installed, skipping kernel registration..."
fi

echo ""
echo "======================================================================"
echo "Installation Complete!"
echo "======================================================================"
echo ""
echo "Environment '${ENV_NAME}' has been created successfully."
echo ""
echo "To activate the environment, run:"
echo "    conda activate ${ENV_NAME}"
echo ""
echo "To test your installation, run:"
echo "    conda activate ${ENV_NAME}"
echo "    python -c 'import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"CUDA version: {torch.version.cuda}\")'"
echo ""
echo "To run the yew density model:"
echo "    conda activate ${ENV_NAME}"
echo "    python yew_density_model.py"
echo ""
echo "To use in Jupyter:"
echo "    conda activate ${ENV_NAME}"
echo "    jupyter lab"
echo "    # Then select kernel: Python (yew_pytorch)"
echo ""
echo "======================================================================"
