#!/bin/bash
# Complete pipeline for training yew detection model with Google Satellite Embeddings
#
# This script:
# 1. Extracts 64-channel embedding patches from Google Earth Engine
# 2. Trains a ResNet model on the embeddings
#
# Usage: ./scripts/training/train_with_embeddings.sh

set -e  # Exit on error

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate yew_pytorch

echo "=================================="
echo "YEW DETECTION WITH EMBEDDINGS"
echo "=================================="
echo ""

# Configuration
YEAR=2024
PATCH_SIZE=64
ARCHITECTURE="resnet18"
EPOCHS=50
BATCH_SIZE=16
LEARNING_RATE=0.001

# Paths
TRAIN_CSV="data/processed/train_split_filtered.csv"
VAL_CSV="data/processed/val_split_filtered.csv"
EMBEDDING_DIR="data/ee_imagery/embedding_patches_64x64"

echo "Step 1: Extract Google Satellite Embedding patches"
echo "---------------------------------------------------"
echo "This will download 64-dimensional embedding vectors for each location"
echo "Year: $YEAR"
echo "Patch size: ${PATCH_SIZE}x${PATCH_SIZE} pixels"
echo ""

python scripts/preprocessing/extract_embedding_patches.py \
    --metadata "$TRAIN_CSV" \
    --output "$EMBEDDING_DIR" \
    --year "$YEAR" \
    --patch-size "$PATCH_SIZE" \
    --resume

echo ""
echo "Extracting validation patches..."
python scripts/preprocessing/extract_embedding_patches.py \
    --metadata "$VAL_CSV" \
    --output "$EMBEDDING_DIR" \
    --year "$YEAR" \
    --patch-size "$PATCH_SIZE" \
    --resume

echo ""
echo "Step 2: Train ResNet on embeddings"
echo "-----------------------------------"
echo "Architecture: $ARCHITECTURE"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo ""

python scripts/training/train_embedding_model.py \
    --train-csv "$TRAIN_CSV" \
    --val-csv "$VAL_CSV" \
    --image-dir "$EMBEDDING_DIR" \
    --architecture "$ARCHITECTURE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE"

echo ""
echo "=================================="
echo "PIPELINE COMPLETE!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Check results in results/training/"
echo "2. Compare with spectral band model"
echo "3. Run predictions using the embedding model"
echo ""
