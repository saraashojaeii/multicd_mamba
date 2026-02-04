# Eval_SCD Metrics Integration

## Overview

Added the official Semantic Change Detection (SCD) evaluation metrics from `Eval_SCD.py` to the test script. These metrics are specifically designed for evaluating semantic change detection models and are commonly used in SCD benchmarks.

---

## New Metrics Added

### 1. **Mean IoU** (`scd_mean_iou`)
- **Definition**: Average of IoU for unchanged and changed pixels
- **Formula**: `(IoU_unchanged + IoU_changed) / 2`
- **Range**: 0.0 to 1.0 (reported as percentage)
- **Interpretation**: Overall binary change detection performance

### 2. **SeK** (`scd_sek`)
- **Definition**: Semantic change Kappa score
- **Formula**: `(kappa_n0 * exp(IoU_fg)) / e`
  - `kappa_n0`: Cohen's Kappa without unchanged pixels
  - `IoU_fg`: IoU for changed (foreground) pixels
- **Range**: 0.0 to 1.0 (reported as percentage)
- **Interpretation**: Semantic change quality weighted by change detection accuracy

### 3. **Score** (`scd_score`)
- **Definition**: Combined metric for SCD evaluation
- **Formula**: `0.3 * Mean_IoU + 0.7 * SeK`
- **Range**: 0.0 to 1.0 (reported as percentage)
- **Interpretation**: Overall SCD performance (emphasizes semantic accuracy)

### 4. **SC_Precision** (`sc_precision`)
- **Definition**: Semantic Change Precision
- **Formula**: `SC_TP / change_pred_sum`
  - `SC_TP`: Correctly predicted semantic changes (diagonal of foreground confusion matrix)
  - `change_pred_sum`: Total predicted changed pixels
- **Range**: 0.0 to 1.0 (reported as percentage)
- **Interpretation**: Of all predicted changes, how many have correct semantics?

### 5. **SC_Recall** (`sc_recall`)
- **Definition**: Semantic Change Recall
- **Formula**: `SC_TP / change_label_sum`
  - `change_label_sum`: Total GT changed pixels
- **Range**: 0.0 to 1.0 (reported as percentage)
- **Interpretation**: Of all GT changes, how many did we predict with correct semantics?

### 6. **F_scd** (`f_scd`)
- **Definition**: Harmonic mean of SC_Precision and SC_Recall
- **Formula**: `hmean([SC_Precision, SC_Recall])`
- **Range**: 0.0 to 1.0 (reported as percentage)
- **Interpretation**: Balanced semantic change detection performance

---

## How It Works

### Confusion Matrix Construction

The metrics are computed from the semantic segmentation confusion matrix:

```python
# Semantic confusion matrix (n_classes × n_classes)
seg_cm = seg_metric.confusion_matrix

# Extract foreground classes (exclude class 0 = unchanged background)
hist_fg = seg_cm[1:, 1:]

# Build binary change confusion matrix
c2hist[0][0] = seg_cm[0][0]                    # TN: unchanged background
c2hist[0][1] = seg_cm.sum(1)[0] - seg_cm[0][0] # FP: pred changed, gt unchanged
c2hist[1][0] = seg_cm.sum(0)[0] - seg_cm[0][0] # FN: pred unchanged, gt changed
c2hist[1][1] = hist_fg.sum()                   # TP: both changed
```

### Key Assumptions

1. **Class 0 = Unchanged**: Pixels that didn't change between T1 and T2
2. **Classes 1-N = Changed**: Pixels that changed to different semantic classes
3. **Foreground = Changed**: Only foreground classes (1-N) represent actual changes

---

## Output Examples

### Console Output

```
============================================================
Test Results:
  Precision: 0.8234
  Recall:    0.7891
  F1-Score:  0.8059
  IoU:       0.6745
  Accuracy:  0.9123
  SeK:       0.6234
============================================================

Eval_SCD Metrics (Semantic Change Detection):
============================================================
  Mean IoU:      67.450%
  SeK:           62.340%
  Score:         63.873% (0.3*IoU + 0.7*SeK)
  SC_Precision:  78.230%
  SC_Recall:     74.560%
  F_scd:         76.340%
============================================================
```

### W&B Metrics

```python
{
  'test/scd_mean_iou': 0.6745,
  'test/scd_sek': 0.6234,
  'test/scd_score': 0.6387,
  'test/sc_precision': 0.7823,
  'test/sc_recall': 0.7456,
  'test/f_scd': 0.7634
}
```

---

## Comparison with Existing Metrics

### Binary Change Detection Metrics
- `test/precision_binary_change`
- `test/recall_binary_change`
- `test/f1_binary_change`
- `test/iou_binary_change`

**Focus**: Did the model detect change vs no-change?

### Semantic Segmentation Metrics
- `test/precision_semantic_masks`
- `test/recall_semantic_masks`
- `test/f1_semantic_masks`
- `test/iou_semantic_masks`

**Focus**: Are the semantic class predictions correct?

### Eval_SCD Metrics (NEW)
- `test/scd_mean_iou`
- `test/scd_sek`
- `test/scd_score`
- `test/sc_precision`
- `test/sc_recall`
- `test/f_scd`

**Focus**: Are the semantic changes detected correctly?

---

## Interpretation Guide

### Good Performance

```
Mean IoU:      > 70%  (Good binary change detection)
SeK:           > 60%  (Good semantic change quality)
Score:         > 65%  (Good overall SCD performance)
SC_Precision:  > 75%  (Most predicted changes are semantically correct)
SC_Recall:     > 70%  (Most GT changes are detected with correct semantics)
F_scd:         > 72%  (Balanced semantic change performance)
```

### What to Watch

1. **High Mean IoU, Low SeK**
   - Good at detecting change locations
   - Poor at predicting correct semantic classes
   - **Fix**: Improve segmentation head, use better class weights

2. **High SC_Precision, Low SC_Recall**
   - Conservative: only predicts changes when very confident
   - Misses many actual changes
   - **Fix**: Lower change detection threshold, increase change loss weight

3. **High SC_Recall, Low SC_Precision**
   - Aggressive: predicts too many changes
   - Many false positive changes
   - **Fix**: Increase change detection threshold, add regularization

4. **Low Score**
   - Overall poor SCD performance
   - **Fix**: Check all components (change detection + segmentation)

---

## Relationship to Original Eval_SCD.py

### What's the Same
- ✅ Exact same formulas for all metrics
- ✅ Same confusion matrix construction logic
- ✅ Same assumptions (class 0 = unchanged)

### What's Different
- ✅ Integrated into test script (no separate evaluation needed)
- ✅ Computed on-the-fly during testing
- ✅ Logged to W&B automatically
- ✅ Works with existing confusion matrix tracking

### Migration from Eval_SCD.py

If you were using `Eval_SCD.py` separately:

**Before:**
```bash
# 1. Run inference
python test_change.py -c config.json

# 2. Run separate evaluation
python Eval_SCD.py  # Edit paths manually
```

**After:**
```bash
# Everything in one step
python test_change.py -c config.json
# Eval_SCD metrics computed and logged automatically
```

---

## Files Modified

1. ✅ **test_change.py**
   - Added Eval_SCD metric computation (lines 339-391)
   - Added console logging (lines 461-471)
   - Added W&B logging (lines 563-570)

---

## Summary

✅ **Added 6 official SCD metrics** from Eval_SCD.py:
- Mean IoU, SeK, Score
- SC_Precision, SC_Recall, F_scd

✅ **Fully integrated** into test script:
- Computed automatically during testing
- Logged to console and W&B
- No separate evaluation script needed

✅ **Compatible** with existing metrics:
- Works alongside binary change and semantic segmentation metrics
- Uses same confusion matrix infrastructure
- No breaking changes

**Use these metrics to evaluate your semantic change detection model performance!**
