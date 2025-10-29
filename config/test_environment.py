#!/usr/bin/env python3
"""
Environment Test Script for Yew Density Model
==============================================
Tests that all required packages are installed and working correctly.

Run: python test_environment.py
"""

import sys
import subprocess


def test_import(package_name, import_name=None):
    """Test if a package can be imported."""
    if import_name is None:
        import_name = package_name

    try:
        __import__(import_name)
        print(f"✓ {package_name}")
        return True
    except ImportError as e:
        print(f"✗ {package_name} - {str(e)}")
        return False


def get_version(package_name, import_name=None):
    """Get package version."""
    if import_name is None:
        import_name = package_name

    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        return version
    except:
        return 'unknown'


def main():
    print("="*70)
    print("Pacific Yew Density Model - Environment Test")
    print("="*70)
    print()

    # Test Python version
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print()

    # Core packages
    print("Testing Core Packages:")
    print("-"*50)
    core_packages = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('scipy', 'scipy'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('sklearn', 'scikit-learn'),
        ('PIL', 'Pillow'),
    ]

    core_results = []
    for import_name, package_name in core_packages:
        result = test_import(package_name, import_name)
        core_results.append(result)

    print()

    # PyTorch
    print("Testing PyTorch:")
    print("-"*50)
    torch_ok = test_import('torch', 'torch')

    if torch_ok:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  cuDNN version: {torch.backends.cudnn.version()}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(
                    f"    Total memory: {props.total_memory / 1024**3:.2f} GB")
                print(f"    Compute capability: {props.major}.{props.minor}")
        else:
            print("  WARNING: CUDA not available. Running on CPU only.")

    print()

    # Deep Learning packages
    print("Testing Deep Learning Packages:")
    print("-"*50)
    dl_packages = [
        ('torchvision', 'torchvision'),
        ('torchaudio', 'torchaudio'),
    ]

    dl_results = []
    for import_name, package_name in dl_packages:
        result = test_import(package_name, import_name)
        dl_results.append(result)

    print()

    # Optional packages (won't fail if missing)
    print("Testing Optional Packages:")
    print("-"*50)
    optional_packages = [
        ('ee', 'earthengine-api'),
        ('geopandas', 'geopandas'),
        ('rasterio', 'rasterio'),
        ('tensorboard', 'tensorboard'),
        ('torchmetrics', 'torchmetrics'),
        ('albumentations', 'albumentations'),
    ]

    for import_name, package_name in optional_packages:
        test_import(package_name, import_name)

    print()

    # Test a simple PyTorch operation
    print("Testing PyTorch Operations:")
    print("-"*50)

    if torch_ok:
        try:
            import torch
            import torch.nn as nn

            # Create a simple tensor
            x = torch.randn(2, 3)
            print(f"✓ Created tensor: shape={x.shape}, dtype={x.dtype}")

            # Test GPU if available
            if torch.cuda.is_available():
                x_gpu = x.cuda()
                print(f"✓ Moved tensor to GPU: device={x_gpu.device}")

                # Simple operation on GPU
                y = x_gpu * 2 + 1
                print(f"✓ GPU computation successful")

            # Create a simple model
            model = nn.Sequential(
                nn.Linear(3, 10),
                nn.ReLU(),
                nn.Linear(10, 1)
            )

            if torch.cuda.is_available():
                model = model.cuda()

            # Forward pass
            with torch.no_grad():
                output = model(x_gpu if torch.cuda.is_available() else x)

            print(
                f"✓ Model forward pass successful: output shape={output.shape}")

        except Exception as e:
            print(f"✗ PyTorch operation failed: {str(e)}")

    print()

    # Summary
    print("="*70)
    print("Summary:")
    print("-"*50)

    all_core_ok = all(core_results)
    all_dl_ok = all(dl_results)

    if all_core_ok and all_dl_ok and torch_ok:
        print("✓ All essential packages are installed correctly!")
        print()
        if torch.cuda.is_available():
            print("✓ GPU support is enabled and working!")
        else:
            print("⚠ Warning: GPU not available, will use CPU (slower)")
        print()
        print("You can now run: python yew_density_model.py")
        return 0
    else:
        print("✗ Some packages are missing or not working correctly.")
        print("Please review the output above and install missing packages.")
        return 1

    print("="*70)


if __name__ == "__main__":
    sys.exit(main())
