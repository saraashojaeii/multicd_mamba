# W&B Logging Guide for CDMamba Models

## Overview
Both training scripts (`train_seg_cd.py` and `train_change.py`) now have comprehensive W&B logging for metrics and visualizations.

---

## 1. train_seg_cd.py (Segmentation + Change Detection)

### **Training Metrics** (logged every epoch)
```python
'train/epoch_loss'           # Average training loss
'train/epoch_mF1_seg'        # Mean F1 for segmentation
'train/epoch_mIoU_seg'       # Mean IoU for segmentation
'train/epoch_change_f1'      # F1 for change detection
'train/epoch_change_iou'     # IoU for change detection
'epoch'                      # Current epoch number
```

### **Validation Metrics** (logged every epoch)
```python
# Segmentation metrics
'val/epoch_loss'             # Average validation loss
'val/epoch_mF1'              # Mean F1 across all classes
'val/epoch_mIoU'             # Mean IoU across all classes
'val/epoch_OA'               # Overall accuracy

# Change detection metrics
'val/epoch_change_prec'      # Change detection precision
'val/epoch_change_rec'       # Change detection recall
'val/epoch_change_f1'        # Change detection F1-score
'val/epoch_change_iou'       # Change detection IoU
'val/epoch_change_acc'       # Change detection accuracy

# Per-class metrics (if available)
'val/class_0_f1'             # F1 for class 0 (background)
'val/class_1_f1'             # F1 for class 1 (building)
'val/class_2_f1'             # F1 for class 2 (road)
# ... etc for each class

'val/class_0_iou'            # IoU for class 0
'val/class_1_iou'            # IoU for class 1
# ... etc for each class

'epoch'                      # Current epoch number
```

### **Best Model Tracking**
```python
'best_val_mF1'               # Best validation mF1 achieved
'best_model_epoch'           # Epoch when best model was saved
```

### **Visualizations** (first batch of train/val per epoch)

**Training:**
- `train/input_T1` - T1 input image
- `train/input_T2` - T2 input image
- `train/gt_seg_t1` - Ground truth segmentation T1 (color-coded)
- `train/gt_seg_t2` - Ground truth segmentation T2 (color-coded)
- `train/pred_seg_t1` - Predicted segmentation T1 (color-coded)
- `train/pred_seg_t2` - Predicted segmentation T2 (color-coded)
- `train/pred_seg_t1_prob` - Max probability map T1
- `train/pred_seg_t2_prob` - Max probability map T2
- `train/gt_change` - Ground truth change mask
- `train/pred_change_prob` - Predicted change probability
- `train/pred_change_mask` - Predicted binary change mask

**Validation:**
- Same as training, but with `val/` prefix

---

## 2. train_change.py (Change Detection Only)

### **Training Metrics** (logged every 500 steps + epoch summary)

**Per-step (every 500 steps):**
```python
'train/loss'                 # Current loss
'train/change_prec'          # Change detection precision
'train/change_rec'           # Change detection recall
'train/change_f1'            # Change detection F1-score
'train/change_iou'           # Change detection IoU
'train/lr'                   # Current learning rate
```

**Epoch summary:**
```python
'train/epoch_loss'           # Average epoch loss
'train/epoch_prec'           # Average epoch precision
'train/epoch_rec'            # Average epoch recall
'train/epoch_f1'             # Average epoch F1-score
'train/epoch_iou'            # Average epoch IoU
'epoch'                      # Current epoch number
```

### **Validation Metrics** (logged every epoch)
```python
'val/loss'                   # Average validation loss
'val/change_prec'            # Change detection precision
'val/change_rec'             # Change detection recall
'val/change_f1'              # Change detection F1-score
'val/change_iou'             # Change detection IoU
'epoch'                      # Current epoch number
```

### **Visualizations** (first batch of train/val per epoch)

**Training:**
- `train/input_T1` - T1 input image
- `train/input_T2` - T2 input image
- `train/gt_change` - Ground truth change mask
- `train/pred_change_prob` - Predicted change probability
- `train/pred_change_mask` - Predicted binary change mask

**Validation:**
- Same as training, but with `val/` prefix

---

## W&B Dashboard Recommendations

### **Key Charts to Create:**

1. **Loss Tracking**
   - Line chart: `train/epoch_loss` vs `val/epoch_loss`

2. **Segmentation Performance** (train_seg_cd.py only)
   - Line chart: `train/epoch_mF1_seg` vs `val/epoch_mF1`
   - Line chart: `train/epoch_mIoU_seg` vs `val/epoch_mIoU`
   - Bar chart: Per-class F1 scores (`val/class_*_f1`)

3. **Change Detection Performance**
   - Line chart: F1, Precision, Recall over epochs
   - Line chart: IoU over epochs
   - Comparison: `train/epoch_change_f1` vs `val/epoch_change_f1`

4. **Image Gallery**
   - Create panels for:
     - Input images (T1, T2)
     - Segmentation predictions (if applicable)
     - Change detection predictions
     - Ground truth comparisons

---

## Example W&B Run Configuration

```python
wandb.init(
    project="BuildingCD_mamba_based",
    name="SECOND-train-CDMamba-Seg-CD_v1",
    config={
        "model": "cdmamba_seg_cd",
        "dataset": "SECOND",
        "batch_size": 4,
        "learning_rate": 1e-4,
        "epochs": 200,
        "loss": "multi_class_cd",
        # ... other config parameters
    }
)
```

---

## Verification Checklist

After starting training, verify in W&B:

### ✅ Metrics Tab
- [ ] Training loss decreasing
- [ ] Validation metrics updating every epoch
- [ ] Per-class metrics appearing (if applicable)
- [ ] No NaN or Inf values

### ✅ Media Tab
- [ ] Input images visible (T1, T2)
- [ ] Ground truth masks visible
- [ ] Prediction masks visible
- [ ] Images updating every epoch

### ✅ System Tab
- [ ] GPU utilization tracked
- [ ] Memory usage tracked
- [ ] Training time per epoch

---

## Troubleshooting

### Metrics not logging?
1. Check console for: `✓ Logged to W&B` messages
2. Verify W&B initialization: `✓ W&B initialized: project=...`
3. Check internet connection (W&B needs to sync)
4. Verify `wandb` project name in config file

### Images not appearing?
1. Check if `log_first_batch_to_wandb()` is being called
2. Verify batch has correct keys: `'A'`, `'B'`, `'L1'`, `'L2'`
3. Check image normalization in `_norm()` function

### Per-class metrics missing?
1. Verify `ConfuseMatrixMeter.get_scores()` returns `'f1_per_class'` and `'iou_per_class'`
2. Check number of classes matches dataset

---

## Summary

**train_seg_cd.py logs:**
- ✅ Training loss + seg metrics + change metrics
- ✅ Validation loss + seg metrics + change metrics  
- ✅ Per-class F1 and IoU
- ✅ Best model tracking
- ✅ Full image visualizations (seg + change)

**train_change.py logs:**
- ✅ Training loss + change metrics (per-step + per-epoch)
- ✅ Validation loss + change metrics
- ✅ Change detection visualizations only

Both scripts provide comprehensive monitoring for training progress! 🎉
