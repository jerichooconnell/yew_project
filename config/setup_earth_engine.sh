#!/bin/bash
# Install Earth Engine API and authenticate

echo "======================================================================"
echo "Earth Engine API Installation"
echo "======================================================================"

# Activate conda environment
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "yew_pytorch" ]; then
    echo "Activating yew_pytorch environment..."
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate yew_pytorch
fi

echo ""
echo "Installing earthengine-api..."
pip install earthengine-api --no-cache-dir

echo ""
echo "Installing geemap (optional, for visualization)..."
read -p "Install geemap? (y/n) [default: y]: " install_geemap
install_geemap=${install_geemap:-y}

if [ "$install_geemap" = "y" ]; then
    pip install geemap --no-cache-dir
    echo "✓ geemap installed"
else
    echo "Skipping geemap installation"
fi

echo ""
echo "======================================================================"
echo "Authentication Setup"
echo "======================================================================"
echo ""
echo "You need to authenticate with Google Earth Engine."
echo "This will:"
echo "  1. Open a browser window"
echo "  2. Ask you to sign in with your Google account"
echo "  3. Generate an authorization code"
echo "  4. You paste the code back here"
echo ""
read -p "Ready to authenticate? (y/n) [default: y]: " do_auth
do_auth=${do_auth:-y}

if [ "$do_auth" = "y" ]; then
    echo ""
    echo "Starting authentication..."
    earthengine authenticate
    
    echo ""
    echo "Testing Earth Engine connection..."
    python -c "import ee; ee.Initialize(); print('✓ Earth Engine initialized successfully!')"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "======================================================================"
        echo "✓ Earth Engine setup complete!"
        echo "======================================================================"
        echo ""
        echo "You can now run:"
        echo "  python scripts/preprocessing/extract_ee_imagery.py"
        echo ""
    else
        echo ""
        echo "⚠ Authentication may have failed. Try running manually:"
        echo "  earthengine authenticate"
        echo ""
    fi
else
    echo ""
    echo "Skipping authentication. You can authenticate later with:"
    echo "  earthengine authenticate"
    echo ""
fi

echo "Done!"
