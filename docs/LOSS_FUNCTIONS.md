# Segmentation Loss Functions for Rare Class Handling

## Overview

This document describes the per-pixel segmentation losses implemented for handling rare classes in semantic change detection. All losses return `[B,H,W]` per-pixel loss maps that can be weighted by change masks and transition weights.

---

## Loss Functions

### 1. WeightedCEDicePerPixel (Recommended)

**Formula**: `L = λ_ce * CE_weighted + λ_dice * Dice`

**Components**:
- **Weighted Cross-Entropy**: Uses class weights computed from GT frequencies
- **Per-Pixel Dice**: Computes Dice coefficient for each pixel across classes

**When to use**:
- Default choice for most datasets
- Good balance between class weighting and spatial consistency
- Handles moderate class imbalance well

**Configuration**:
```json
{
  "model": {
    "extended_triplet": {
      "seg_loss": "weighted_ce_dice"
    }
  }
}
```

**Advantages**:
- ✅ Stable training
- ✅ Handles rare classes via class weights
- ✅ Dice term improves boundary quality
- ✅ Works well with change-aware boosting

**Disadvantages**:
- ⚠️ May struggle with extreme imbalance (e.g., <0.1% class frequency)

---

### 2. FocalCEDicePerPixel (For Extreme Imbalance)

**Formula**: `L = λ_focal * Focal + λ_dice * Dice`

**Components**:
- **Focal Loss**: `FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)`
  - Downweights easy examples (high confidence)
  - Focuses on hard examples (low confidence)
  - `γ` controls focusing strength (default: 2.0)
- **Per-Pixel Dice**: Same as above

**When to use**:
- Extreme class imbalance (e.g., water/playground <0.5%)
- When WeightedCEDicePerPixel doesn't improve rare class performance
- When you want the model to focus more on hard examples

**Configuration**:
```json
{
  "model": {
    "extended_triplet": {
      "seg_loss": "focal_ce_dice",
      "focal_gamma": 2.0
    }
  }
}
```

**Advantages**:
- ✅ Better for extreme imbalance
- ✅ Automatically focuses on hard examples
- ✅ Can improve rare class recall significantly

**Disadvantages**:
- ⚠️ May be less stable (requires careful tuning)
- ⚠️ Can overfit to rare classes if γ is too high

---

## Class Weight Computation

Class weights are computed from training data using **median frequency weighting**:

```python
freq[c] = count[c] / total_pixels
median_freq = median(freq)
weight[c] = median_freq / freq[c]
```

**Example** (from SECOND dataset):
- nvg_surf (common): 35% → weight ≈ 0.5
- water (rare): 0.5% → weight ≈ 35.0
- playground (very rare): 0.1% → weight ≈ 175.0

Weights are automatically computed during training initialization from the first 200 batches.

---

## Integration with Change-Aware Weighting

All per-pixel losses work seamlessly with change-aware and transition-aware weighting:

```python
# 1. Compute per-pixel loss
loss_map = seg_loss_fn(logits, target)  # [B,H,W]

# 2. Apply change-aware boost
w = 1.0 + boost * change_mask  # [B,H,W]

# 3. Apply transition weights (for changed pixels)
w *= W_transition[y1, y2]  # [B,H,W]

# 4. Weighted average
L = (loss_map * w * valid_mask).sum() / (w * valid_mask).sum()
```

This ensures:
- Changed pixels get more weight than unchanged
- Rare transitions get more weight than common transitions
- Rare classes get more weight via class weights in the loss

---

## Comparison

| Feature | WeightedCEDicePerPixel | FocalCEDicePerPixel |
|---------|------------------------|---------------------|
| **Stability** | High | Medium |
| **Rare class handling** | Good | Excellent |
| **Hard example focus** | No | Yes (via γ) |
| **Training speed** | Fast | Slightly slower |
| **Hyperparameters** | Few | More (γ, α) |
| **Recommended for** | Most cases | Extreme imbalance |

---

## Hyperparameter Tuning

### WeightedCEDicePerPixel

**λ_ce and λ_dice** (default: 0.5 each):
- Increase λ_ce if boundaries are blurry → try 0.6 CE, 0.4 Dice
- Increase λ_dice if rare classes are fragmented → try 0.4 CE, 0.6 Dice

**Class weights**:
- Automatically computed from training data
- If rare classes still perform poorly, check if weights are being applied correctly

### FocalCEDicePerPixel

**γ (gamma)** (default: 2.0):
- Higher γ → more focus on hard examples
- γ = 0: equivalent to weighted CE
- γ = 1: moderate focusing
- γ = 2: standard focal loss
- γ = 3-5: aggressive focusing (use with caution)

**Tuning guide**:
- Start with γ = 2.0
- If rare class recall is still low → increase to 2.5 or 3.0
- If training becomes unstable → decrease to 1.5 or 1.0
- Monitor validation loss carefully

**λ_focal and λ_dice** (default: 0.5 each):
- Same guidance as WeightedCEDicePerPixel

---

## Usage Examples

### Example 1: Default (Weighted CE + Dice)

```json
{
  "model": {
    "loss": "extended_triplet",
    "extended_triplet": {
      "seg_loss": "weighted_ce_dice",
      "lambda_seg": 1.5,
      "boost": 5.0
    }
  }
}
```

**Expected behavior**:
- Stable training
- Good performance on common and moderately rare classes
- Improved boundary quality from Dice term

### Example 2: Focal Loss for Extreme Imbalance

```json
{
  "model": {
    "loss": "extended_triplet",
    "extended_triplet": {
      "seg_loss": "focal_ce_dice",
      "focal_gamma": 2.5,
      "lambda_seg": 1.5,
      "boost": 5.0
    }
  }
}
```

**Expected behavior**:
- Better rare class recall (water, playground)
- May require more epochs to converge
- Monitor for overfitting to rare classes

---

## Monitoring

### During Training

**Console logs**:
```
Using WeightedCEDicePerPixel (recommended for rare classes)
Class weights: tensor([0.5, 0.8, 1.2, 35.0, 1.5, 175.0])
```

**W&B metrics to watch**:
- `train/epoch_mF1_seg`: Overall segmentation quality
- `val/class_3_f1`: Water performance
- `val/class_5_f1`: Playground performance
- `train/epoch_loss`: Should decrease smoothly

### Signs of Good Performance

✅ **Weighted CE + Dice**:
- Validation mF1 improves steadily
- Rare class F1 > 0.3 after 20-30 epochs
- Loss decreases smoothly

✅ **Focal CE + Dice**:
- Rare class F1 improves faster
- May see more fluctuation in loss
- Better rare class recall than weighted CE

### Signs of Problems

⚠️ **Loss explodes or becomes NaN**:
- Solution: Reduce γ (if using focal) or reduce boost
- Check class weights aren't too extreme

⚠️ **Rare classes still perform poorly**:
- Solution: Try focal loss with higher γ
- Increase oversample_factor in balanced sampler
- Check if rare classes appear in training data

⚠️ **Model overfits to rare classes**:
- Solution: Reduce γ or reduce oversample_factor
- Add more regularization (dropout, weight decay)

---

## Implementation Details

### Per-Pixel Dice Computation

For each pixel, we compute Dice across classes:

```python
# For pixel (b, h, w):
probs = softmax(logits[b, :, h, w])  # [C]
target_oh = one_hot(target[b, h, w])  # [C]

intersection = (probs * target_oh).sum()
union = probs.sum() + target_oh.sum()
dice = (2 * intersection + smooth) / (union + smooth)
loss = 1 - dice
```

This differs from global Dice (which averages over all pixels first), providing finer-grained gradients.

### Focal Loss Computation

```python
# For each pixel:
p_t = prob[target_class]  # Probability of correct class
focal_weight = (1 - p_t) ** gamma
if alpha is not None:
    focal_weight *= alpha[target_class]
loss = -focal_weight * log(p_t)
```

The `(1 - p_t)^γ` term downweights easy examples (high p_t) and focuses on hard examples (low p_t).

---

## Comparison with Original Losses

### Before (Simple CE or CE+Dice)

```python
# Scalar loss, no per-pixel weighting
loss = CE(logits, target).mean()  # [1]
# or
loss = 0.5 * CE(logits, target) + 0.5 * Dice(logits, target)  # [1]
```

**Problems**:
- Unchanged pixels dominate the loss
- Rare classes get insufficient gradient
- No way to boost changed pixels

### After (Per-Pixel Weighted CE+Dice)

```python
# Per-pixel loss with change-aware weighting
loss_map = WeightedCE(logits, target)  # [B,H,W]
loss_map += Dice(logits, target)  # [B,H,W]
w = 1 + boost * change_mask  # [B,H,W]
w *= W_transition[y1, y2]  # [B,H,W]
loss = (loss_map * w).sum() / w.sum()  # [1]
```

**Benefits**:
- ✅ Changed pixels get 6x weight (boost=5.0)
- ✅ Rare transitions get up to 10x weight
- ✅ Rare classes get high weight via class weights
- ✅ Combined effect: rare class in rare transition in changed region gets ~600x weight!

---

## References

- **Focal Loss**: Lin et al., 2017 - [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- **Dice Loss**: Milletari et al., 2016 - [V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation](https://arxiv.org/abs/1606.04797)
- **Class Balancing**: Cui et al., 2019 - [Class-Balanced Loss Based on Effective Number of Samples](https://arxiv.org/abs/1901.05555)

---

## Summary

✅ **Use WeightedCEDicePerPixel** (default):
- Stable and effective for most cases
- Good balance of class weighting and spatial consistency

✅ **Use FocalCEDicePerPixel** when:
- Rare classes still perform poorly with weighted CE
- Extreme class imbalance (<0.5% frequency)
- You want automatic hard example mining

✅ **Both losses**:
- Return per-pixel maps for change-aware weighting
- Use class weights computed from training data
- Work seamlessly with transition-aware weighting
- Provide strong gradients for rare classes
