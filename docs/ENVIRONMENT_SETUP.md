# Pacific Yew Density Model - Environment Setup

## ⚠️ DISK SPACE ERROR? READ THIS FIRST!

If you got: `[Errno 28] No space left on device`

**Solution:** Run the cleanup script first:
```bash
bash cleanup_conda.sh
```

Then choose between:
- **Minimal Install** (~3GB): `bash setup_yew_env_minimal.sh`
- **Full Install** (~10GB): `bash setup_yew_env.sh`

See [Troubleshooting Disk Space](#troubleshooting-disk-space) below for details.

---

## Quick Start

You have **three options** to set up the environment:

### Option 1: Minimal Installation (RECOMMENDED if < 10GB free)

```bash
cd ~/yew_project
bash setup_yew_env_minimal.sh
```

This installs only essentials via pip (~3GB total):
- PyTorch 2.1.0 with CUDA 11.8
- NumPy, Pandas, SciPy, Matplotlib, Seaborn, Scikit-learn
- Perfect for running the model

Later, add packages as needed:
```bash
conda activate yew_pytorch
pip install jupyterlab  # If you need Jupyter
pip install earthengine-api  # If you need Earth Engine
```

### Option 2: Full Installation (if >= 10GB free)

```bash
cd ~/yew_project
bash setup_yew_env.sh
```

This script will:
- Create a new conda environment named `yew_pytorch`
- Install PyTorch with CUDA 11.8 support (via pip, more space-efficient)
- Prompt you to select optional packages interactively
- Clean conda cache to save space

### Option 3: Using the YAML file (NOT RECOMMENDED - uses more space)

```bash
cd ~/yew_project
conda env create -f yew_pytorch_env.yml
```

## Activation

After installation, activate the environment:

```bash
conda activate yew_pytorch
```

## Testing

Test that everything is installed correctly:

```bash
python test_environment.py
```

This will check:
- ✓ All core packages (numpy, pandas, scipy, matplotlib, etc.)
- ✓ PyTorch and CUDA availability
- ✓ GPU detection and capabilities
- ✓ Simple PyTorch operations

## Running the Model

Once the environment is set up and tested:

```bash
conda activate yew_pytorch
python yew_density_model.py
```

## GPU Information

Your system has:
- **GPU**: NVIDIA GeForce RTX 4050 Laptop GPU (Compute Capability 8.9)
- **CUDA Driver**: 12.2
- **CUDA Runtime**: 11.8
- **Total Memory**: ~6GB

The environment is configured to use CUDA 11.8, which matches your CUDA Runtime version.

## Package Summary

### Core Scientific Computing
- numpy 1.26.4
- pandas 2.2.1
- scipy 1.11.4
- matplotlib 3.8.0
- seaborn 0.13.2
- scikit-learn 1.3.0
- scikit-image 0.22.0

### Deep Learning
- PyTorch 2.0+ with CUDA 11.8
- torchvision
- torchaudio
- tensorboard
- torchmetrics
- albumentations

### Geospatial (for Earth Engine integration)
- earthengine-api
- geopandas
- rasterio
- shapely
- pyproj
- folium

### Development Tools
- jupyter, jupyterlab
- ipykernel
- wandb, mlflow (experiment tracking)

## Troubleshooting

### Troubleshooting Disk Space

#### Check Available Space
```bash
df -h .  # Check space in current directory
```

**Space Requirements:**
- Minimal install: 3GB minimum
- Full install: 10GB recommended
- Conda cache can grow to 5-10GB over time

#### Free Up Space (BEFORE Installing)

1. **Run the cleanup script:**
   ```bash
   bash cleanup_conda.sh
   ```
   This removes conda cache, typically freeing 2-5GB.

2. **Remove old conda environments:**
   ```bash
   conda env list  # See all environments
   conda env remove -n <old-env-name>  # Remove ones you don't need
   ```

3. **Clear pip cache:**
   ```bash
   pip cache purge
   ```

4. **Find what's using space:**
   ```bash
   du -h --max-depth=1 ~/anaconda3 | sort -hr | head -10
   ```

5. **Nuclear option (clean everything):**
   ```bash
   conda clean --all -y
   pip cache purge
   ```

#### Installation Strategy by Available Space

| Free Space | Recommended Action |
|------------|-------------------|
| < 3GB | Free up space first with cleanup_conda.sh |
| 3-10GB | Use **minimal install**: `bash setup_yew_env_minimal.sh` |
| 10GB+ | Use **full install**: `bash setup_yew_env.sh` |

### CUDA not detected

If `test_environment.py` shows CUDA is not available:

1. Check NVIDIA driver:
   ```bash
   nvidia-smi
   ```

2. Verify CUDA installation:
   ```bash
   nvcc --version
   ```

3. Reinstall PyTorch with correct CUDA version:
   ```bash
   conda activate yew_pytorch
   pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir
   ```

### Out of memory errors

If you get GPU out of memory errors:

1. Reduce batch size in `yew_density_model.py`:
   ```python
   batch_size=16  # or even 8
   ```

2. Use gradient accumulation
3. Enable mixed precision training

### Import errors

If packages are missing:

```bash
conda activate yew_pytorch
pip install <missing-package>
```

## Using in Jupyter

The environment is automatically registered as a Jupyter kernel:

```bash
conda activate yew_pytorch
jupyter lab
```

Then select kernel: **Python (yew_pytorch)**

## Environment Management

### Update environment
```bash
conda env update -f yew_pytorch_env.yml --prune
```

### Export your environment
```bash
conda activate yew_pytorch
conda env export > my_yew_env.yml
```

### Remove environment
```bash
conda deactivate
conda env remove -n yew_pytorch
```

## Next Steps

1. ✅ Set up environment
2. ✅ Test installation
3. 🔄 Extract Earth Engine imagery (see `extract_ee_imagery.py` - coming soon)
4. 🔄 Train model with actual satellite data
5. 🔄 Make predictions on new sites

## Support

For issues:
- Check PyTorch docs: https://pytorch.org/get-started/locally/
- Check CUDA compatibility: https://pytorch.org/get-started/previous-versions/
- Verify GPU: `nvidia-smi` and `python test_environment.py`
