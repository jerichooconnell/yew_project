# Classifier Performance Comparison

## Overview
Comparing two approaches for yew classification using 64-channel satellite embeddings:
1. **predict_center_pixel_map.py** - Uses StandardScaler normalization
2. **classify_every_pixel_in_bbox.py** - No scaling (preserves relative tile scale)

---

## Method 1: Center Pixel with StandardScaler
**Script:** `predict_center_pixel_map.py`

### Configuration
- Features: Center pixel from (64, 64, 64) embeddings → 64 features
- Preprocessing: StandardScaler (z-score normalization)
- Model: SVM with RBF kernel, balanced class weights
- Training: Separate train/val evaluation

### Performance Metrics

#### Training Set (1,952 samples)
- **Accuracy:** 95.95%
- **Precision:** 95.00%
- **Recall:** 95.57%
- **F1 Score:** 95.28%
- **ROC-AUC:** 99.25%

Confusion Matrix:
```
           Predicted
           Non-Yew  Yew
Actual
Non-Yew    1075     42
Yew          37    798
```

#### Validation Set (487 samples)
- **Accuracy:** 91.58%
- **Precision:** 92.39%
- **Recall:** 87.50%
- **F1 Score:** 89.88%
- **ROC-AUC:** 97.80%

Confusion Matrix:
```
           Predicted
           Non-Yew  Yew
Actual
Non-Yew     264     15
Yew          26    182
```

---

## Method 2: Pixel Classification without Scaling
**Script:** `classify_every_pixel_in_bbox.py`

### Configuration
- Features: Center pixel from (64, 64, 64) embeddings → 64 features
- Preprocessing: None (only inf/nan handling)
- Model: SVM with RBF kernel, balanced class weights
- Training: Combined train+val for final model

### Performance Metrics

#### Training Set (1,784 samples)
- **Accuracy:** 74.05%
- **Precision:** 78.58%
- **Recall:** 61.15%
- **F1 Score:** 68.78%
- **ROC-AUC:** 81.20%

Confusion Matrix:
```
           Predicted
           Non-Yew  Yew
Actual
Non-Yew     811    139
Yew         324    510
```

#### Validation Set (447 samples)
- **Accuracy:** 69.80%
- **Precision:** 70.93%
- **Recall:** 58.94%
- **F1 Score:** 64.38%
- **ROC-AUC:** 75.67%

Confusion Matrix:
```
           Predicted
           Non-Yew  Yew
Actual
Non-Yew     190     50
Yew          85    122
```

---

## Key Findings

### 1. StandardScaler Impact
**Massive performance improvement with StandardScaler:**
- Validation accuracy: 91.58% vs 69.80% (+21.78 pp)
- Validation F1 score: 89.88% vs 64.38% (+25.50 pp)
- Validation ROC-AUC: 97.80% vs 75.67% (+22.13 pp)

### 2. Scaling Benefits
StandardScaler normalizes features to have:
- Mean = 0
- Standard deviation = 1

This is crucial for RBF kernel SVM because:
- RBF kernel uses distance-based similarity
- Features with different scales can dominate the distance calculation
- Normalization ensures all 64 embedding channels contribute equally

### 3. Sample Differences
Note the different sample counts:
- Method 1: 1,952 train, 487 val (2,439 total)
- Method 2: 1,784 train, 447 val (2,231 total)

This suggests ~208 samples were excluded in Method 2 due to missing embeddings.

### 4. Trade-offs

**Method 1 (WITH StandardScaler):**
- ✅ Much higher accuracy (91.58% val)
- ✅ Better generalization
- ✅ Lower false positive rate (3.1% vs 17.9%)
- ⚠️ Requires consistent scaling for predictions

**Method 2 (NO scaling):**
- ❌ Lower accuracy (69.80% val)
- ❌ Higher false positive rate
- ❌ More missed yew locations (recall 58.94% vs 87.50%)
- ✅ Simpler preprocessing
- ⚠️ May preserve some tile-specific information

---

## Recommendations

1. **Use StandardScaler** for this classification task
   - The performance improvement is substantial (+22 pp accuracy)
   - RBF kernel benefits greatly from normalized features

2. **Address the sample discrepancy**
   - Investigate why Method 2 lost ~208 samples
   - Ensure embedding paths are correctly resolved

3. **For spatial predictions:**
   - Use Method 1 approach (predict_center_pixel_map.py)
   - Apply the same StandardScaler to all prediction locations
   - Higher accuracy means fewer false positives in spatial maps

4. **Consider hyperparameter tuning:**
   - Both models use default SVM parameters
   - Grid search on C, gamma could improve performance further

---

## Conclusion

**StandardScaler is essential** for this task. The RBF kernel SVM performs dramatically better with normalized features, achieving 91.58% validation accuracy vs 69.80% without scaling.

For production use, `predict_center_pixel_map.py` with StandardScaler should be the preferred approach.
