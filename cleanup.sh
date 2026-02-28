#!/usr/bin/env bash
# =============================================================================
# cleanup.sh — Remove deprecated scripts from the working copy
#
# Run this from the MASTER branch (not this archive branch) to delete all
# files that are no longer part of the active pipeline.
#
# Usage:
#   git checkout master
#   bash cleanup.sh              # dry run (shows what would be deleted)
#   bash cleanup.sh --confirm    # actually deletes the files
#
# See DEPRECATED_FILES.md in this branch for an explanation of what each file
# did before it was retired.
# =============================================================================

set -euo pipefail

DRY_RUN=true
if [[ "${1:-}" == "--confirm" ]]; then
    DRY_RUN=false
fi

RED='\033[0;31m'
YLW='\033[1;33m'
GRN='\033[0;32m'
NC='\033[0m'

if $DRY_RUN; then
    echo -e "${YLW}DRY RUN — no files will be deleted. Pass --confirm to actually delete.${NC}"
else
    echo -e "${RED}LIVE RUN — files will be permanently removed from the working copy!${NC}"
    read -p "Are you sure? (yes/no): " confirm
    if [[ "$confirm" != "yes" ]]; then
        echo "Aborted."
        exit 0
    fi
fi

echo ""
echo "============================================================"
echo " Deprecated script cleanup"
echo "============================================================"

DEPRECATED_FILES=(
    # ── Prediction — superseded ──────────────────────────────────────────────
    "scripts/prediction/classify_bbox_embeddings.py"
    "scripts/prediction/classify_every_pixel_in_bbox.py"
    "scripts/prediction/classify_large_area.py"
    "scripts/prediction/classify_large_area_export.py"
    "scripts/prediction/classify_large_area_geotiff.py"
    "scripts/prediction/classify_tiled_area.py"
    "scripts/prediction/composite_and_overlay.py"
    "scripts/prediction/evaluate_center_pixel_classifier.py"
    "scripts/prediction/overlay_from_predictions.py"
    "scripts/prediction/predict_center_pixel_map.py"
    "scripts/prediction/predict_grid_region.py"
    "scripts/prediction/predict_yew_probability_grid.py"
    "scripts/prediction/sample_cwh_yew_population.py"

    # ── Training — superseded ────────────────────────────────────────────────
    "scripts/training/dataset.py"
    "scripts/training/model.py"
    "scripts/training/simple_embedding_model.py"
    "scripts/training/train_cnn.py"
    "scripts/training/train_yew_cnn.py"
    "scripts/training/train_reduced_model.py"
    "scripts/training/retrain_with_annotations.py"
    "scripts/training/train_xgboost_baseline.py"
    "scripts/training/train_xgboost_cwh_engineered.py"
    "scripts/training/train_xgboost_engineered.py"
    "scripts/training/train_xgboost_enhanced.py"
    "scripts/training/tune_svm_hyperparameters.py"
    "scripts/training/monitor_training.py"
    "scripts/training/yew_density_model.py"

    # ── Preprocessing — superseded ───────────────────────────────────────────
    "scripts/preprocessing/extract_ee_imagery.py"
    "scripts/preprocessing/extract_ee_image_patches.py"
    "scripts/preprocessing/extract_inat_yew_images.py"
    "scripts/preprocessing/extract_no_yew_images.py"
    "scripts/preprocessing/extract_yew_images_only.py"
    "scripts/preprocessing/extract_yew_images_parallel.py"
    "scripts/preprocessing/filter_and_extract_dataset.py"
    "scripts/preprocessing/convert_npy_to_png.py"
    "scripts/preprocessing/rebuild_metadata.py"
    "scripts/preprocessing/remove_california_samples.py"
    "scripts/preprocessing/add_no_yew_coordinates.py"
    "scripts/preprocessing/extract_more_yew_bc_wa.py"

    # ── Analysis — superseded ────────────────────────────────────────────────
    "scripts/analysis/analyze_feature_importance.py"
    "scripts/analysis/analyze_resnet_feature_importance.py"
    "scripts/analysis/analyze_pacific_yew.py"
    "scripts/analysis/analyze_pacific_yew_bc_sample.py"
    "scripts/analysis/analyze_yew_correlations.py"
    "scripts/analysis/classify_center_pixels.py"
    "scripts/analysis/variance_reduction.py"

    # ── Visualization — superseded ───────────────────────────────────────────
    "scripts/visualization/create_annotation_tool.py"
    "scripts/visualization/create_interactive_viewer.py"
    "scripts/visualization/create_south_vi_interactive.py"
    "scripts/visualization/detailed_yew_histograms.py"
    "scripts/visualization/display_ee_patches.py"
    "scripts/visualization/download_ee_images.py"
    "scripts/visualization/map_plot_locations.py"
    "scripts/visualization/review_inat_images.py"
    "scripts/visualization/visualize_ee_image_patches.py"

    # ── Root level ────────────────────────────────────────────────────────────
    "scripts/geotess_ex.py"

    # ── Notebooks ────────────────────────────────────────────────────────────
    "analyze_center_pixels.ipynb"
    "yew_data_exploration.ipynb"
)

DELETED=0
SKIPPED=0
NOT_FOUND=0

for f in "${DEPRECATED_FILES[@]}"; do
    if [[ -f "$f" ]]; then
        if $DRY_RUN; then
            echo -e "  ${YLW}WOULD DELETE${NC}  $f"
        else
            rm "$f"
            echo -e "  ${RED}DELETED${NC}  $f"
        fi
        DELETED=$((DELETED + 1))
    else
        echo -e "  ${GRN}NOT FOUND${NC} (already gone)  $f"
        NOT_FOUND=$((NOT_FOUND + 1))
    fi
done

echo ""
echo "============================================================"
if $DRY_RUN; then
    echo -e " ${YLW}DRY RUN complete.${NC}"
    echo "   Would delete : $DELETED files"
    echo "   Already gone : $NOT_FOUND files"
    echo ""
    echo " Run with --confirm to actually delete."
else
    echo -e " ${GRN}Cleanup complete.${NC}"
    echo "   Deleted      : $DELETED files"
    echo "   Already gone : $NOT_FOUND files"
    echo ""
    echo " You may want to commit the removal:"
    echo "   git add -A && git commit -m 'Remove deprecated scripts'"
    echo "   git push origin master"
fi
echo "============================================================"
