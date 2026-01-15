# train_change.py - Comprehensive Verification Checklist

## ✅ All Checks Passed

### 1. **Imports** ✅
- [x] All necessary imports present
- [x] `torch`, `numpy`, `wandb` imported
- [x] `Data`, `Model`, `core.utils` imported
- [x] `parse_cfg`, `setup_logger`, `dict2str`, `dict_to_nonedict` imported
- [x] `get_scheduler`, `save_network` from `misc.torchutils`
- [x] Loss functions imported from `models.loss`

### 2. **Helper Functions** ✅
- [x] `set_all_seeds()` - Sets random seeds for reproducibility
- [x] `make_stamped_dirs()` - Creates timestamped directories
- [x] `derive_change_bin()` - Derives binary change from seg labels
- [x] `safe_to_numpy_uint8()` - Converts tensors to numpy
- [x] `log_first_batch_to_wandb()` - Logs images to W&B (NEW)

### 3. **Argument Parser** ✅
- [x] `--config` (required) - Path to JSON config
- [x] `--phase` (train/test) - Training or testing phase
- [x] `--dataset` - Dataset name (default: SECOND)
- [x] `--tag` - Experiment tag
- [x] `--seed` - Random seed
- [x] `--gpu_ids` - GPU device IDs
- [x] `--resume_path` - Path to checkpoint for resuming
- [x] `--wandb_project` - W&B project name
- [x] `--max_train_batches` - Limit training batches
- [x] `--max_val_batches` - Limit validation batches
- [x] `--max_test_batches` - Limit test batches
- [x] `--change_threshold` - Threshold for binary change (default: 0.5)

### 4. **Configuration & Setup** ✅
- [x] Config parsed with `parse_cfg(args)` (passes full args object)
- [x] GPU setup with `CUDA_VISIBLE_DEVICES`
- [x] Seeds set for reproducibility
- [x] Experiment folder created with timestamp
- [x] Directories stamped correctly
- [x] Logger setup for train/test phases
- [x] W&B initialization (if enabled)

### 5. **Data Loading** ✅
- [x] Uses `Data.create_scd_dataset()` (NOT create_cd_dataset)
  - Returns `'L1'`, `'L2'` keys (required for change-only model)
- [x] Creates train/val/test datasets
- [x] Seed worker function for reproducibility
- [x] Dataloaders created with proper parameters
- [x] Batch size, num_workers from config

### 6. **Model** ✅
- [x] Model created with `Model.create_CD_model(opt)`
- [x] Model moved to device
- [x] Optional torchinfo summary
- [x] Resume from checkpoint supported
- [x] Handles both 'model' key and direct state_dict

### 7. **Loss Function** ✅
- [x] Supports 'ce' (CrossEntropyLoss)
- [x] Supports 'dice' (DiceLoss)
- [x] Supports 'cedice' (CEDiceLoss)
- [x] Raises error for unsupported loss types

### 8. **Optimizer & Scheduler** ✅
- [x] Supports Adam, AdamW, SGD
- [x] Learning rate from config
- [x] Scheduler created with `get_scheduler()`
- [x] Mixed precision with `torch.cuda.amp.GradScaler`
- [x] Gradient accumulation supported

### 9. **Training Loop** ✅
- [x] Iterates through epochs
- [x] Model set to train mode
- [x] Loads batches with `'A'`, `'B'`, `'L1'`, `'L2'` keys
- [x] Forward pass: `change_pred = cd_model(im1, im2)`
- [x] Derives change_bin from seg labels
- [x] Computes loss correctly
- [x] Gradient accumulation implemented
- [x] Gradient clipping (max_norm=0.5)
- [x] Metrics computation (TP, FP, FN, TN)
- [x] **Image logging to W&B (first batch)** ✅
- [x] Periodic logging to console
- [x] W&B metric logging

### 10. **Validation Loop** ✅
- [x] Runs every `val_freq` epochs
- [x] Model set to eval mode
- [x] No gradient computation
- [x] Loss computed on validation set
- [x] Metrics: Precision, Recall, F1, IoU
- [x] **Image logging to W&B (first batch)** ✅
- [x] W&B metric logging
- [x] Best model saving with correct signature:
  - `save_network(opt, epoch, cd_model, optimizer, is_best_model=True)`
- [x] Epoch checkpoint saving (every 10 epochs)

### 11. **Test Loop** ✅
- [x] Model set to eval mode
- [x] No gradient computation
- [x] Loads test data correctly
- [x] Forward pass: `chg = cd_model(ti1, ti2)`
- [x] Handles both 2-class and sigmoid outputs
- [x] Computes final metrics
- [x] Logs results to console

### 12. **Image Logging to W&B** ✅
**Training:**
- [x] `train/input_T1` - T1 input image
- [x] `train/input_T2` - T2 input image
- [x] `train/gt_change` - Ground truth change mask
- [x] `train/pred_change_prob` - Predicted change probability
- [x] `train/pred_change_mask` - Predicted binary change mask

**Validation:**
- [x] `val/input_T1` - T1 input image
- [x] `val/input_T2` - T2 input image
- [x] `val/gt_change` - Ground truth change mask
- [x] `val/pred_change_prob` - Predicted change probability
- [x] `val/pred_change_mask` - Predicted binary change mask

### 13. **Error Handling** ✅
- [x] Try-except for torchinfo summary
- [x] Handles missing checkpoint keys
- [x] Validates loss type
- [x] Validates optimizer type

### 14. **Cleanup** ✅
- [x] W&B finish called at end
- [x] Proper logging throughout

---

## Key Differences from train_seg_cd.py

| Feature | train_seg_cd.py | train_change.py |
|---------|-----------------|-----------------|
| **Dataset** | `create_cd_dataset()` | `create_scd_dataset()` ✅ |
| **Model output** | `(seg1, seg2, change)` | `change` only ✅ |
| **Loss** | Multi-task or separate | Single change loss ✅ |
| **Metrics** | Seg + Change | Change only ✅ |
| **Image logging** | Seg + Change | Change only ✅ |
| **Save function** | Custom | `save_network(opt, epoch, model, optimizer, is_best)` ✅ |

---

## Fixed Issues

1. ✅ **parse_cfg** - Now passes full `args` object instead of just config path
2. ✅ **setup_logger** - Uses correct signature with `root`, `phase`, `level`, `screen`
3. ✅ **Dataset** - Uses `create_scd_dataset()` to get `L1`, `L2` keys
4. ✅ **Mamba SSM** - Fixed by using `ConvMamba` from `mamba_customer` module
5. ✅ **save_network** - Uses correct signature: `(opt, epoch, cd_model, optimizer, is_best_model)`
6. ✅ **Image logging** - Added `log_first_batch_to_wandb()` function

---

## Model Architecture (CDMamba_change)

```
Input: T1, T2 [B, 3, H, W]
  ↓
Encoder (SRCM blocks) - processes T1 and T2 separately
  ↓
Cross-Temporal Fusion at each scale
  concat(F1, F2, |F2-F1|, F2-F1) → conv → refine
  ↓
Bottleneck Context (dilated convs + global pooling)
  ↓
Fused Bottleneck
  ↓
Change Decoder (SRCM blocks + skip connections)
  ↓
Change Head (1×1 conv)
  ↓
Output: Change logits [B, 2, H, W]
```

---

## Expected Training Output

```
[Epoch 1/200] Step 500/557 | Loss: 0.4716 | Change - Prec: 0.4651, Rec: 0.3487, F1: 0.3986, IoU: 0.2489

[Epoch 1/200] Train Summary:
  Loss: 0.4634
  Change - Prec: 0.4821, Rec: 0.3509, F1: 0.4062, IoU: 0.2549

[Epoch 1/200] Validation Summary:
  Loss: 0.4045
  Change - Prec: 0.7250, Rec: 0.3579, F1: 0.4792, IoU: 0.3151

✓ Saved best model (F1: 0.4792)
```

---

## W&B Dashboard

**Metrics tracked:**
- `train/loss`, `train/change_prec`, `train/change_rec`, `train/change_f1`, `train/change_iou`, `train/lr`
- `val/loss`, `val/change_prec`, `val/change_rec`, `val/change_f1`, `val/change_iou`

**Images logged (first batch per epoch):**
- Training: T1, T2, GT change, predicted prob, predicted mask
- Validation: T1, T2, GT change, predicted prob, predicted mask

---

## Config Requirements

```json
{
  "model": {
    "name": "cdmamba_change",
    "loss": "ce",
    "n_classes": 2,
    "in_channels": 3,
    "init_filters": 16,
    "blocks_down": [1, 2, 2, 4],
    "blocks_up": [1, 1, 1]
  },
  "datasets": {
    "train": { "datasetroot": "...", "resolution": 512, "batch_size": 4 },
    "val": { "datasetroot": "...", "resolution": 512, "batch_size": 4 },
    "test": { "datasetroot": "...", "resolution": 512, "batch_size": 4 }
  },
  "train": {
    "n_epoch": 200,
    "optimizer": { "type": "adam", "lr": 1e-4 },
    "sheduler": { "lr_policy": "linear" }
  }
}
```

---

## Summary

✅ **All systems operational!**

The `train_change.py` script is fully functional and ready for training:
- Correct dataset loading (SCDDataset with L1/L2)
- Proper model architecture (CDMamba_change with ConvMamba)
- Correct loss computation (single change loss)
- Image logging to W&B (train + val)
- Proper checkpoint saving
- Complete metrics tracking

**No issues found. Ready to train!** 🚀
