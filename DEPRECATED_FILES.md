# Deprecated Scripts — Reference Archive

This branch contains documentation of scripts that are no longer part of the active pipeline.
Run `cleanup.sh` from the **main** (or master) branch to delete these files from your working copy.

> **Why keep this branch?** So the git history remains intact and this document provides context
> on what each script did before it was retired.

---

## Migration Summary

The project evolved through three distinct phases:

| Phase | Approach | Status |
|---|---|---|
| **Phase 1** | CNN image classification (ResNet on PNG/npy patches) | Retired |
| **Phase 2** | XGBoost / SVM on hand-engineered spectral features | Retired |
| **Phase 3** | MLP on Google satellite 64-d embeddings (current) | **Active** |

---

## Deprecated Files by Category

### Prediction — superseded scripts

| File | What it did | Replaced by |
|---|---|---|
| `scripts/prediction/classify_bbox_embeddings.py` | Classified all GEE embedding pixels within a lon/lat bounding box; no polygon clipping; slow for large areas | `classify_tiled_gpu.py` |
| `scripts/prediction/classify_every_pixel_in_bbox.py` | Attempted to classify every 10 m pixel in a bbox by exhaustive GEE sampling; hit compute limits quickly | `classify_tiled_gpu.py` |
| `scripts/prediction/classify_large_area.py` | First BC-wide classifier; used coarse rectangular grid sampling with no CWH boundary; produced inflated estimates | `resample_cwh_100k.py` |
| `scripts/prediction/classify_large_area_export.py` | Tried exporting GEE assets to Drive then classifying locally; abandoned due to quota/delay issues | `resample_cwh_100k.py` |
| `scripts/prediction/classify_large_area_geotiff.py` | GeoTIFF export variant of the above; produced large rasters that were hard to handle | `resample_cwh_100k.py` |
| `scripts/prediction/classify_tiled_area.py` | CPU-only tiled classifier; predecessor to `classify_tiled_gpu.py`; no batching, slow | `classify_tiled_gpu.py` |
| `scripts/prediction/composite_and_overlay.py` | Composed RGB+probability overlay images from raw per-tile npy files using PIL; output format not used downstream | `create_cwh_kmz.py` |
| `scripts/prediction/evaluate_center_pixel_classifier.py` | Computed precision/recall for the Phase 1 CNN center-pixel approach; only relevant to the CNN model | — |
| `scripts/prediction/overlay_from_predictions.py` | Generated KML/HTML overlays from the old per-tile prediction CSV format (embed_0..embed_63); format changed | `create_cwh_kmz.py` |
| `scripts/prediction/predict_center_pixel_map.py` | Used Phase 1 CNN to predict the central pixel of each GEE patch; very slow, low spatial resolution | `classify_tiled_gpu.py` |
| `scripts/prediction/predict_grid_region.py` | Regular lat/lon grid sampler that classified each grid cell via CNN; superseded by embedding-based approach | `classify_tiled_gpu.py` |
| `scripts/prediction/predict_yew_probability_grid.py` | Another grid prediction variant with matplotlib visualization; early prototype | `generate_png_map.py` |
| `scripts/prediction/sample_cwh_yew_population.py` | Original CWH-wide population sampler; had two bugs: (1) WFS failure fell back to bounding-box rectangle ~7× too large; (2) no point-in-polygon filter. Produced inflated ~1.1M ha estimate | `resample_cwh_100k.py` |

---

### Training — superseded scripts

| File | What it did | Replaced by |
|---|---|---|
| `scripts/training/dataset.py` | PyTorch Dataset class for loading PNG/npy image patches (Phase 1 CNN) | `scripts/training/dataset_embedding.py` |
| `scripts/training/model.py` | ResNet-18 fine-tuning model for Phase 1 CNN classification | `scripts/training/model_embedding.py` |
| `scripts/training/simple_embedding_model.py` | Early MLP prototype without BatchNorm1d; caused `RuntimeError` when loading current model weights | `classify_tiled_gpu.py` (YewMLP definition) |
| `scripts/training/train_cnn.py` | Phase 1 CNN training loop; full ResNet fine-tune on iNat image patches | `classify_tiled_gpu.py` |
| `scripts/training/train_yew_cnn.py` | Alternative Phase 1 CNN trainer with augmentation; same outcome | `classify_tiled_gpu.py` |
| `scripts/training/train_reduced_model.py` | Experiment with fewer hidden layers and PCA-reduced features | `classify_tiled_gpu.py` |
| `scripts/training/retrain_with_annotations.py` | Script to add manual annotations to the CNN training set; superseded by `--annotations` flag | `classify_tiled_gpu.py` |
| `scripts/training/train_xgboost_baseline.py` | Phase 2 XGBoost on raw embedding dimensions; AUC ~0.94 | `classify_tiled_gpu.py` (YewMLP AUC 0.998) |
| `scripts/training/train_xgboost_cwh_engineered.py` | XGBoost with CWH-specific engineered features (NDVI, elevation quartiles, etc.) | `classify_tiled_gpu.py` |
| `scripts/training/train_xgboost_engineered.py` | XGBoost with general-purpose engineered spectral features | `classify_tiled_gpu.py` |
| `scripts/training/train_xgboost_enhanced.py` | XGBoost with SHAP-guided feature selection; marginal improvement over baseline | `classify_tiled_gpu.py` |
| `scripts/training/tune_svm_hyperparameters.py` | GridSearchCV over RBF-SVM; best AUC ~0.91, much slower than XGBoost | `classify_tiled_gpu.py` |
| `scripts/training/monitor_training.py` | TensorBoard callback wrapper for Phase 1 CNN; not used with MLP training | — |
| `scripts/training/yew_density_model.py` | Gaussian KDE density surface from iNat occurrence records; superseded by classifier-based approach | — |

---

### Preprocessing — superseded scripts

| File | What it did | Replaced by |
|---|---|---|
| `scripts/preprocessing/extract_ee_imagery.py` | Sequential (one-at-a-time) GEE image extractor; very slow for large batches | `extract_ee_imagery_fast.py`, `extract_ee_batch_chunks.py` |
| `scripts/preprocessing/extract_ee_image_patches.py` | Downloaded RGB+NIR image patches as npy arrays for CNN training | `extract_ee_batch_chunks.py` (downloads embeddings, not patches) |
| `scripts/preprocessing/extract_inat_yew_images.py` | Downloaded iNaturalist yew observation images as PNGs via GBIF API | Not needed — pipeline uses GEE embeddings, not raw images |
| `scripts/preprocessing/extract_no_yew_images.py` | Downloaded background (non-yew) images for CNN Phase 1 | Not needed |
| `scripts/preprocessing/extract_yew_images_only.py` | Filtered the full iNat download to confirmed yew observations | Not needed |
| `scripts/preprocessing/extract_yew_images_parallel.py` | Parallel (multiprocessing) version of the above | Not needed |
| `scripts/preprocessing/filter_and_extract_dataset.py` | Old end-to-end pipeline: filter iNat CSV → download images → build train/val splits | `create_balanced_dataset.py` |
| `scripts/preprocessing/convert_npy_to_png.py` | Batch-converted npy image tiles to PNG for manual review in a browser | Not needed |
| `scripts/preprocessing/rebuild_metadata.py` | One-time fix for broken `metadata.json` files from early tile downloads | Not needed |
| `scripts/preprocessing/remove_california_samples.py` | One-time geographic filter removing California iNat observations from training set | Handled by coordinate filter in `create_balanced_dataset.py` |
| `scripts/preprocessing/add_no_yew_coordinates.py` | One-time injection of manually collected non-yew GPS coordinates into negative CSV | Superseded by `generate_cwh_negatives.py` and `extract_faib_negatives.py` |
| `scripts/preprocessing/extract_more_yew_bc_wa.py` | One-time GEE extraction of extra iNat yew sites in BC and Washington State | Completed; results in `data/processed/` |

---

### Analysis — superseded scripts

| File | What it did | Replaced by |
|---|---|---|
| `scripts/analysis/analyze_feature_importance.py` | SHAP feature importance analysis for Phase 2 XGBoost model | — |
| `scripts/analysis/analyze_resnet_feature_importance.py` | GradCAM / activation maps for Phase 1 ResNet | — |
| `scripts/analysis/analyze_pacific_yew.py` | Initial EDA on raw iNaturalist observations CSV (histograms, maps) | `yew_data_exploration.ipynb` |
| `scripts/analysis/analyze_pacific_yew_bc_sample.py` | Same EDA restricted to BC + WA observations | `yew_data_exploration.ipynb` |
| `scripts/analysis/analyze_yew_correlations.py` | Pearson/Spearman correlation between spectral embedding dimensions and yew presence | — |
| `scripts/analysis/classify_center_pixels.py` | Ran Phase 1 CNN on center pixels of GEE patches; produced per-site probability CSVs in old format | `classify_cwh_spots.py` |
| `scripts/analysis/variance_reduction.py` | Stratified vs simple random sampling comparison for variance reduction; decided simple random adequate | — |

---

### Visualization — superseded scripts

| File | What it did | Replaced by |
|---|---|---|
| `scripts/visualization/create_annotation_tool.py` | Built a browser-based labelling tool (Flask + Leaflet) for manual yew/non-yew annotation | Annotations complete; tool no longer needed |
| `scripts/visualization/create_interactive_viewer.py` | Bokeh-based tile viewer for Phase 1 npy image outputs | Not needed |
| `scripts/visualization/create_south_vi_interactive.py` | South Vancouver Island specific Bokeh probability map from Phase 2 CSV predictions | `classify_cwh_spots.py` → HTML maps |
| `scripts/visualization/detailed_yew_histograms.py` | Matplotlib histograms of iNat occurrence attributes (month, elevation, province) | `yew_data_exploration.ipynb` |
| `scripts/visualization/display_ee_patches.py` | ASCII/image display of GEE image patches in terminal | Not needed |
| `scripts/visualization/download_ee_images.py` | Batch-downloaded GEE RGB thumbnails as PNGs for visual checking | Not needed |
| `scripts/visualization/map_plot_locations.py` | Static matplotlib dot map of iNat observation locations | `generate_png_map.py` |
| `scripts/visualization/review_inat_images.py` | Tkinter image review tool for manually labelling downloaded iNat images | Annotations complete |
| `scripts/visualization/visualize_ee_image_patches.py` | Grid display of GEE image patches with NDVI/RGB channels | Not needed |

---

### Root-level scripts

| File | What it did | Replaced by |
|---|---|---|
| `scripts/geotess_ex.py` | GeoTIFF read/write example code used during Phase 2 raster experiments | Not needed |

---

### Notebooks

| File | What it did | Replaced by |
|---|---|---|
| `analyze_center_pixels.ipynb` | Phase 1 analysis of CNN center-pixel classification outputs | `classify_cwh_spots.py` + spot comparison HTML maps |
| `yew_data_exploration.ipynb` | Initial EDA notebook for the raw iNat CSV | Still useful for reference, but not part of active pipeline |

---

## Active Pipeline Files (not deprecated)

For reference, these are the files currently in use:

```
scripts/prediction/
├── classify_cwh_spots.py          # 15-area CWH spot maps with VRI overlay
├── classify_tiled_gpu.py          # South VI tile classification + MLP training
├── resample_cwh_100k.py           # CWH zone 100k population sample
├── sample_coastal_region.py       # Coastal BC region 100k population sample

scripts/preprocessing/
├── apply_forestry_mask.py         # VRI suppression factor raster
├── build_cwh_boundary_from_vri.py # CWH MultiPolygon from VRI GDB
├── build_vri_coverage_boundary.py # VRI geographic coverage extent
├── calculate_global_stats.py      # Embedding mean/std for normalisation
├── check_dataset_status.py        # Dataset inventory utility
├── create_balanced_dataset.py     # Merge positives + negatives + annotations
├── create_filtered_splits.py      # Train/val splits with geographic filter
├── create_positives_only_splits.py # Positives-only splits for contrastive learning
├── deduplicate_sites.py           # Remove spatial duplicates from iNat sites
├── extract_alpine_negatives.py    # Mine hard negatives from alpine tiles
├── extract_cwh_logged_negatives.py # Mine negatives from logged CWH stands
├── extract_ee_batch_chunks.py     # Batch GEE embedding extraction
├── extract_ee_imagery_fast.py     # Fast single-batch GEE extraction
├── extract_embedding_patches.py   # Extract embedding patches for tile cache
├── extract_faib_negatives.py      # FAIB forestry negative embeddings
├── extract_logged_negatives.py    # Logged site negative embeddings
├── generate_cwh_negatives.py      # CWH zone random negative embeddings
├── generate_logged_negatives.py   # Logged site negative embeddings (v2)
├── test_ee_integration.py         # GEE connection / auth check utility

scripts/training/
├── analyze_cwh_predictions.py     # Post-hoc analysis of CWH predictions
├── create_final_summary.py        # Generate training run summary tables
├── dataset_embedding.py           # PyTorch Dataset for 64-d embeddings
├── model_embedding.py             # YewMLP architecture definition
├── train_embedding_model.py       # MLP training with embedding features
├── train_with_ee_data.py          # MLP training with GEE embeddings
├── train_yew_model_with_ee.py     # MLP training (alternate entry point)

scripts/visualization/
├── create_cwh_kmz.py              # KMZ ground overlay from sample predictions
├── create_interactive_map.py      # Folium/Leaflet interactive prediction map
├── create_kmz.py                  # Generic KMZ generator
├── generate_png_map.py            # Static PNG map from sample predictions
├── review_all_images.py           # Image review tool (active annotations)
├── visualize_misclassifications.py # Misclassification case analysis

scripts/analysis/
├── analyze_review_results.py      # Analyse manual review annotation results
├── analyze_bc_sample.py           # BC-wide sample statistics
```
