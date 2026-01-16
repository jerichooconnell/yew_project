#!/bin/bash
# Train model with global normalization that preserves reflectance scale
#
# Usage: bash scripts/training/train_with_global_norm.sh

set -e  # Exit on error

cd /home/jericho/yew_project

echo "================================================================"
echo "Training ResNet with Global Normalization"
echo "This preserves relative reflectance values between images"
echo "================================================================"
echo ""

# Step 1: Calculate global statistics from training data
echo "Step 1: Calculating global statistics..."
python scripts/preprocessing/calculate_global_stats.py \
    --train-csv data/processed/train_split_filtered.csv \
    --image-dir data/ee_imagery/image_patches_64x64 \
    --output data/processed/global_normalization_stats.json

echo ""
echo "================================================================"
echo "Step 2: Training model with global normalization..."
echo "================================================================"
echo ""

# Step 2: Train with global normalization
python scripts/training/train_cnn.py \
    --use-filtered \
    --use-global-norm \
    --global-stats data/processed/global_normalization_stats.json \
    --architecture resnet18 \
    --epochs 50 \
    --batch-size 16 \
    --lr 0.001 \
    --weight-decay 0.0001 \
    --early-stopping 15

echo ""
echo "================================================================"
echo "Training complete!"
echo "================================================================"
echo ""
echo "Model saved to: models/checkpoints/"
echo "Results saved to: results/training/"
echo ""
echo "Next steps:"
echo "  1. Compare performance with per-image normalization model"
echo "  2. Update prediction scripts to use global normalization"
echo "  3. Generate new predictions on southern Vancouver Island"
