# Consistency Loss Usage Guide

## Overview

The `SemanticChangeConsistencyLoss` ensures alignment between semantic segmentation changes and binary change detection predictions. This addresses the architectural flaw where semantic changes might not correspond to detected binary changes.

## Architecture Improvements Implemented

### 1. Multi-Scale Interaction Blocks
- **Before**: Interaction only at bottleneck
- **After**: Interaction at every encoder scale
- **Benefit**: Better multi-scale feature alignment

### 2. Joint Feature Learning
- **Before**: Post-hoc fusion (change features derived from semantic features)
- **After**: Bidirectional influence (change features modulate semantic features during encoding)
- **Benefit**: Change features actively influence semantic encoding, not just passive fusion

### 3. Skip Connection Alignment
- **Before**: Semantic decoders use temporal-specific skips, change decoder uses fused skips
- **After**: All three decoders use aligned skip connections from jointly-learned features
- **Benefit**: Consistent multi-scale representations across all paths

### 4. Consistency Loss
- **Before**: No explicit constraint between semantic changes and binary change predictions
- **After**: MSE loss between semantic change map and binary change probability
- **Benefit**: Ensures predictions are internally consistent

## Usage in Training

### Option 1: Use CombinedLoss (Recommended)

```python
from models.loss import CombinedLoss

# Initialize combined loss
loss_fn = CombinedLoss(
    num_classes=7,
    seg_ce_weight=1.0,
    seg_dice_weight=1.0,
    change_ce_weight=1.0,
    change_dice_weight=1.0,
    consistency_weight=0.5,  # Weight for consistency loss
    ignore_index=255,
    use_soft_consistency=True,  # Use soft probabilities (recommended)
)

# In training loop
seg_logits_t1, seg_logits_t2, change_logits = model(img1, img2)
total_loss, loss_dict = loss_fn(
    seg_logits_t1, seg_logits_t2, change_logits,
    seg_gt_t1, seg_gt_t2, change_gt
)

# Log individual components
wandb.log({
    'train/total_loss': loss_dict['total_loss'],
    'train/seg_loss': loss_dict['seg_loss'],
    'train/change_loss': loss_dict['change_loss'],
    'train/consistency_loss': loss_dict['consistency_loss'],
})
```

### Option 2: Use SemanticChangeConsistencyLoss Standalone

```python
from models.loss import SemanticChangeConsistencyLoss

# Initialize consistency loss
consistency_loss_fn = SemanticChangeConsistencyLoss(
    ignore_index=255,
    loss_weight=0.5,
    use_soft_labels=True,  # Use soft semantic probabilities
    temperature=1.0,
)

# In training loop (combine with your existing losses)
seg_loss = your_seg_loss_fn(seg_logits_t1, seg_gt_t1) + your_seg_loss_fn(seg_logits_t2, seg_gt_t2)
change_loss = your_change_loss_fn(change_logits, change_gt)
consistency_loss = consistency_loss_fn(
    seg_logits_t1, seg_logits_t2, change_logits,
    seg_gt_t1, seg_gt_t2
)

total_loss = seg_loss + change_loss + consistency_loss
```

## Configuration in JSON Config

Add to your training config:

```json
{
  "model": {
    "loss": "combined_consistency",
    "loss_weights": {
      "seg_ce": 1.0,
      "seg_dice": 1.0,
      "change_ce": 1.0,
      "change_dice": 1.0,
      "consistency": 0.5
    },
    "consistency_config": {
      "use_soft_labels": true,
      "temperature": 1.0
    }
  }
}
```

## How It Works

### Soft Consistency (Recommended)
1. Compute semantic probability distributions: `P_t1 = softmax(logits_t1)`, `P_t2 = softmax(logits_t2)`
2. Compute semantic change map: `semantic_change = ||P_t2 - P_t1||_1 / 2` (normalized to [0,1])
3. Compute binary change probability: `change_prob = softmax(change_logits)[1]`
4. Consistency loss: `MSE(change_prob, semantic_change)`

### Hard Consistency
1. Get hard predictions: `pred_t1 = argmax(logits_t1)`, `pred_t2 = argmax(logits_t2)`
2. Binary semantic change: `semantic_change = (pred_t1 != pred_t2)`
3. Binary change probability: `change_prob = softmax(change_logits)[1]`
4. Consistency loss: `MSE(change_prob, semantic_change)`

## Benefits

1. **Prevents Inconsistencies**: Model cannot predict semantic changes without binary change detection
2. **Improves Generalization**: Forces model to learn coherent representations
3. **Better Calibration**: Change probabilities better reflect actual semantic changes
4. **Handles Class Imbalance**: Soft consistency is more robust to rare classes

## Hyperparameter Tuning

- **consistency_weight**: Start with 0.5, increase if you see inconsistencies
- **use_soft_labels**: `True` for better gradients, `False` for stricter enforcement
- **temperature**: Higher values (>1) soften the semantic distributions, lower values (<1) sharpen them

## Expected Results

With consistency loss, you should see:
- Higher alignment between semantic change metrics and binary change metrics
- Reduced false positives in change detection when semantics don't change
- Better performance on rare class transitions
