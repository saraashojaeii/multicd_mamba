# Comprehensive Metrics Update

## Overview

Added comprehensive metrics tracking for training, validation, and testing to monitor performance on:
1. **Changed pixels only** (semantic IoU/F1)
2. **Per-class performance** (building, nvg_surf, water, playground)
3. **Top transitions** (nvg_surf→building, low_veg↔nvg_surf, low_veg→building)

---

## New Metrics Logged

### 1. Semantic Metrics on Changed Pixels Only

**Why**: Overall semantic metrics include unchanged pixels (which dominate). Changed-pixel metrics show how well the model predicts semantic classes in regions that actually changed.

**Metrics**:
- `changed_pixels_iou`: Mean IoU computed only on changed pixels
- `changed_pixels_f1`: Mean F1 computed only on changed pixels
- `changed_pixels_acc`: Accuracy computed only on changed pixels

**Logged in**:
- `train/changed_pixels_*`
- `val/changed_pixels_*`
- `test/changed_pixels_*`

**Expected values**:
- Should be **lower** than overall metrics (changed regions are harder)
- Good performance: IoU > 0.5, F1 > 0.6
- Watch this improve with transition-aware weighting

---

### 2. Per-Class IoU and F1

**Why**: Track performance on key classes, especially rare ones (water, playground) and common ones (building, nvg_surf).

**Classes tracked**:
- **Class 0**: low_veg
- **Class 1**: nvg_surf
- **Class 3**: water (rare)
- **Class 4**: building
- **Class 5**: playground (rare)

**Metrics**:
- `class_building_iou` / `class_building_f1`
- `class_nvg_surf_iou` / `class_nvg_surf_f1`
- `class_water_iou` / `class_water_f1`
- `class_playground_iou` / `class_playground_f1`

**Logged in**:
- `train/class_*`
- `val/class_*`
- `test/class_*`

**Expected values**:
- Common classes (building, nvg_surf): F1 > 0.7
- Rare classes (water, playground): F1 > 0.3 (should improve with rebalancing)

---

### 3. Top Transition Performance

**Why**: Track how well the model predicts specific semantic transitions, especially the dominant ones.

**Transitions tracked**:
1. **nvg_surf→building** (1→4): Dominant transition (~32% in your GT)
2. **low_veg→nvg_surf** (0→1): Common bidirectional transition
3. **nvg_surf→low_veg** (1→0): Reverse of above
4. **low_veg→building** (0→4): Another common transition

**Metrics per transition**:
- `transition_X_to_Y_acc`: Accuracy for this specific transition (both T1 and T2 correct)
- `transition_X_to_Y_count`: Number of pixels with this transition in GT
- `transition_X_to_Y_correct_t1`: Accuracy for T1 prediction only
- `transition_X_to_Y_correct_t2`: Accuracy for T2 prediction only

**Logged in**:
- `train/transition_*`
- `val/transition_*`
- `test/transition_*`

**Expected values**:
- Dominant transitions (nvg_surf→building): accuracy > 0.7
- Less common transitions: accuracy > 0.5
- Watch rare transitions improve with transition-aware weighting

---

## Implementation

### New Functions in `core/metrics.py`

1. **`compute_semantic_metrics_on_changed()`**
   - Computes IoU, F1, accuracy only on pixels where GT changed
   - Returns per-class metrics as well

2. **`compute_per_class_metrics()`**
   - Computes IoU, F1, precision, recall for each class
   - Uses all pixels (T1 and T2 combined)

3. **`compute_transition_metrics()`**
   - Computes accuracy for specific transitions
   - Tracks T1, T2, and both correct

### Integration

**Training/Validation** (`train_seg_cd.py`):
- Computes metrics from last batch (representative sample)
- Logs to W&B every epoch

**Testing** (`test_change.py`):
- Accumulates predictions across all test batches
- Computes metrics on full test set
- Logs to W&B and saves to JSON

---

## W&B Dashboard Organization

### Training Metrics

**Change Detection**:
- `train/epoch_change_f1`
- `train/epoch_change_iou`
- `train/epoch_change_acc`

**Semantic Segmentation (Overall)**:
- `train/epoch_mF1_seg`
- `train/epoch_mIoU_seg`
- `train/epoch_OA_seg`

**Semantic on Changed Pixels**:
- `train/changed_pixels_iou`
- `train/changed_pixels_f1`
- `train/changed_pixels_acc`

**Per-Class Performance**:
- `train/class_building_f1`
- `train/class_nvg_surf_f1`
- `train/class_water_f1`
- `train/class_playground_f1`
- (+ IoU versions)

**Top Transitions**:
- `train/transition_1_to_4_acc` (nvg_surf→building)
- `train/transition_0_to_1_acc` (low_veg→nvg_surf)
- `train/transition_1_to_0_acc` (nvg_surf→low_veg)
- `train/transition_0_to_4_acc` (low_veg→building)
- (+ count, correct_t1, correct_t2 for each)

### Validation Metrics

Same structure as training, with `val/` prefix.

### Test Metrics

Same structure as training/validation, with `test/` prefix.
Plus additional test-specific metrics:
- `test/precision_binary_change`
- `test/recall_binary_change`
- `test/f1_binary_change`
- `test/iou_binary_change`
- `test/accuracy_binary_change`
- `test/sek_binary_change`
- `test/precision_semantic_masks`
- `test/recall_semantic_masks`
- `test/f1_semantic_masks`
- `test/iou_semantic_masks`
- `test/accuracy_semantic_masks`
- `test/sek_semantic_masks`
- `test/change_pixel_ratio`

---

## How to Use

### 1. Monitor Training Progress

**Key metrics to watch**:
- `val/changed_pixels_f1`: Should improve steadily
- `val/class_water_f1`: Should improve with rebalancing (target > 0.3)
- `val/class_playground_f1`: Should improve with rebalancing (target > 0.3)
- `val/transition_1_to_4_acc`: Dominant transition (should be high, ~0.7+)

**Create W&B charts**:
```python
# In W&B dashboard, create a custom chart:
# X-axis: epoch
# Y-axis: Multiple lines
#   - val/changed_pixels_f1
#   - val/class_water_f1
#   - val/class_playground_f1
```

### 2. Compare Experiments

**Baseline vs Rebalanced**:
```
Metric                          | Baseline | Rebalanced | Gain
--------------------------------|----------|------------|------
val/changed_pixels_f1           | 0.45     | 0.62       | +0.17
val/class_water_f1              | 0.12     | 0.45       | +0.33
val/class_playground_f1         | 0.08     | 0.38       | +0.30
val/transition_1_to_4_acc       | 0.68     | 0.75       | +0.07
```

### 3. Diagnose Issues

**If changed_pixels_f1 is low**:
- Increase `boost` parameter (try 7.0 or 10.0)
- Increase `oversample_factor` (try 3.0)

**If class_water_f1 or class_playground_f1 is low**:
- Switch to focal loss (`seg_loss: "focal_ce_dice"`)
- Increase `oversample_factor` for rare classes
- Check if rare classes appear in training data

**If transition_1_to_4_acc is low**:
- Check transition weights are being applied
- Verify transition matrix computation

**If transition_X_to_Y_count is 0**:
- This transition doesn't appear in the current batch/dataset
- Normal for rare transitions in training (batch-level metrics)
- Should have counts in test metrics (full dataset)

---

## Example Output

### Console (Training)

```
[Train] ep 10 loss 0.45231 | seg mF1 0.6234 | chg F1 0.7123
  Changed pixels F1: 0.5234
  Building F1: 0.7234, Water F1: 0.2134, Playground F1: 0.1823
  Transition nvg_surf→building: 0.6834 (count: 1234)
```

### W&B Dashboard

**Training curves** (epoch vs metric):
- Overall mF1: 0.62 → 0.75 (improving)
- Changed pixels F1: 0.52 → 0.68 (improving faster with rebalancing)
- Water F1: 0.21 → 0.45 (significant improvement)
- Playground F1: 0.18 → 0.38 (significant improvement)

**Transition accuracy**:
- nvg_surf→building: 0.68 → 0.75 (dominant, should be high)
- low_veg→nvg_surf: 0.54 → 0.62 (improving)
- nvg_surf→low_veg: 0.51 → 0.59 (improving)
- low_veg→building: 0.48 → 0.58 (improving)

### Test Results (JSON)

```json
{
  "test/changed_pixels_iou": 0.5234,
  "test/changed_pixels_f1": 0.6823,
  "test/class_building_f1": 0.7534,
  "test/class_water_f1": 0.4512,
  "test/class_playground_f1": 0.3823,
  "test/transition_1_to_4_acc": 0.7534,
  "test/transition_1_to_4_count": 45678,
  "test/transition_0_to_1_acc": 0.6234,
  "test/transition_0_to_1_count": 23456
}
```

---

## Files Modified

1. ✅ **core/metrics.py**
   - Added `compute_semantic_metrics_on_changed()`
   - Added `compute_per_class_metrics()`
   - Added `compute_transition_metrics()`

2. ✅ **train_seg_cd.py**
   - Import new metric functions
   - Compute changed-pixel metrics from last batch
   - Compute per-class metrics
   - Compute top transition metrics
   - Log all to W&B

3. ✅ **test_change.py**
   - Import new metric functions
   - Accumulate predictions across all batches
   - Compute changed-pixel metrics on full test set
   - Compute per-class metrics
   - Compute top transition metrics
   - Log all to W&B

---

## Summary

✅ **Implemented**:
1. Semantic metrics on changed pixels only (IoU, F1, accuracy)
2. Per-class metrics for key classes (building, nvg_surf, water, playground)
3. Top transition tracking (nvg_surf→building, low_veg↔nvg_surf, low_veg→building)
4. Comprehensive W&B logging for train/val/test

✅ **Benefits**:
- Better visibility into model performance on changed regions
- Track rare class improvements from rebalancing
- Monitor specific transition performance
- Diagnose issues more effectively

✅ **Ready to use**:
- All metrics logged automatically during training/val/test
- No configuration changes needed
- Works with existing rebalancing features

**Start training and monitor**: All new metrics will appear in W&B automatically!
