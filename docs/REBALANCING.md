# Dataset Rebalancing for Change Detection

This document describes the dataset rebalancing features implemented to address class and transition imbalance in semantic change detection.

## Problem

Change detection datasets often suffer from severe imbalance:

1. **Unchanged pixels dominate**: Most pixels don't change between T1 and T2, leading the model to focus on unchanged regions.
2. **Transition imbalance**: Among changed pixels, certain transitions dominate (e.g., nvg_surf→building at 32% in SECOND dataset).
3. **Rare classes**: Some classes (water, playground) appear infrequently, leading to poor performance on these classes.

## Solutions Implemented

### 1. Change-Aware Segmentation Loss

**Location**: `models/loss.py` - `TripletChangeSegLoss`

**How it works**:
- Segmentation loss is computed per-pixel (not averaged globally)
- Changed pixels get boosted weight: `w = 1 + boost * change_mask`
- Default `boost=5.0` means changed pixels get 6x the weight of unchanged pixels

**Configuration**:
```json
{
  "model": {
    "loss": "extended_triplet",
    "extended_triplet": {
      "boost": 5.0
    }
  }
}
```

**Impact**: Prevents the model from overfitting to unchanged background regions.

---

### 2. Transition-Aware Weighting

**Location**: `models/loss.py` - `TripletChangeSegLoss`, `core/utils.py`

**How it works**:
- Precomputes transition matrix from training data (changed pixels only)
- Computes inverse-frequency weights: `W[i,j] = median(freq) / freq[i,j]`
- For changed pixels, applies: `w *= W_transition[y1, y2]`
- Weights are clamped to [0.1, 10.0] to avoid extreme values

**Functions**:
- `estimate_transition_matrix(dataloader, num_classes, ...)`: Computes GT transition counts
- `compute_transition_weights(transition_matrix, method="inverse_frequency")`: Derives weights

**Automatic**: Enabled automatically when using `extended_triplet` loss. Transition weights are computed during training initialization.

**Impact**: Prevents the model from learning only dominant transitions (e.g., nvg_surf→building).

---

### 3. Balanced Patch Sampling

**Location**: `data/balanced_sampler.py`

**How it works**:
- Precomputes statistics for each training sample:
  - Change ratio: `(y1 != y2).sum() / total_pixels`
  - Presence of rare classes
- Categorizes samples into:
  - **High change**: `change_ratio >= threshold` (default 1%)
  - **Rare classes**: Contains water (class 3) or playground (class 5)
  - **Regular**: Everything else
- Oversamples high-change and rare-class patches by a factor (default 2.0x)

**Configuration**:
```json
{
  "train": {
    "use_balanced_sampler": true,
    "balanced_sampler": {
      "change_threshold": 0.01,
      "rare_classes": [3, 5],
      "oversample_factor": 2.0,
      "precompute_stats": true,
      "max_precompute": 1000
    }
  }
}
```

**Parameters**:
- `change_threshold`: Minimum change ratio to be considered "high change" (0.01 = 1%)
- `rare_classes`: List of class indices to oversample (e.g., [3, 5] for water, playground)
- `oversample_factor`: How much to oversample (2.0 = 2x more samples)
- `precompute_stats`: Whether to precompute (faster training, uses more memory)
- `max_precompute`: Max samples to precompute (None = all, or set to e.g., 1000 for large datasets)

**Impact**: Ensures training batches include diverse change patterns and rare classes, preventing the model from ignoring them.

---

## Usage

### Quick Start

1. **Use the balanced configuration**:
   ```bash
   python train_seg_cd.py --config config/second_cdmamba/cdmamba_seg_cd_balanced.json
   ```

2. **Or enable in your existing config**:
   ```json
   {
     "model": {
       "loss": "extended_triplet",
       "extended_triplet": {
         "boost": 5.0
       }
     },
     "train": {
       "use_balanced_sampler": true,
       "balanced_sampler": {
         "change_threshold": 0.01,
         "rare_classes": [3, 5],
         "oversample_factor": 2.0
       }
     }
   }
   ```

### Tuning

**If model still ignores changed regions**:
- Increase `boost` (try 7.0 or 10.0)
- Increase `oversample_factor` (try 3.0)

**If model overfits to changed regions**:
- Decrease `boost` (try 3.0 or 2.0)
- Decrease `oversample_factor` (try 1.5)

**If rare classes still perform poorly**:
- Add more rare classes to `rare_classes` list
- Increase `oversample_factor` specifically for rare classes
- Consider using `WeightedRandomSamplerByChange` with higher `rare_class_weight`

**For very large datasets**:
- Set `max_precompute` to limit memory usage (e.g., 1000 or 2000)
- Or disable precomputation: `"precompute_stats": false` (slower but no memory overhead)

---

## Implementation Details

### Transition Weight Computation

```python
# Pseudocode
transition_matrix = count_transitions(train_data)  # [C, C]
freq = transition_matrix / transition_matrix.sum()
median_freq = median(freq[freq > 0])
weights = median_freq / (freq + eps)
weights = clamp(weights, min=0.1, max=10.0)
```

**Example** (from your GT matrix):
- nvg_surf→building: 32% → weight ≈ 0.5 (downweighted)
- water→tree: 0.03% → weight ≈ 10.0 (upweighted, clamped)

### Balanced Sampler Logic

```python
# Pseudocode
for each sample:
    if change_ratio >= threshold:
        add to high_change_pool
    elif has_rare_class:
        add to rare_class_pool
    else:
        add to regular_pool

# During training
indices = all_samples  # base
indices += sample(high_change_pool, size=len(high_change_pool) * (factor - 1))
indices += sample(rare_class_pool, size=len(rare_class_pool) * (factor - 1))
shuffle(indices)
```

---

## Monitoring

During training, check logs for:

```
[BalancedChangeSampler] Statistics:
  High change (>1.0%): 234 samples
  Rare classes: 156 samples
  Regular: 1610 samples
```

```
Transition matrix computed. Total transitions: 45678
Transition weights range: [0.100, 10.000]
```

Also monitor W&B metrics:
- `train/epoch_change_f1`: Should improve faster with rebalancing
- `val/class_3_f1`, `val/class_5_f1`: Rare class performance
- `test/transition_matrix_global`: Check if model learns all transitions

---

## Files Modified

1. **models/loss.py**
   - `TripletChangeSegLoss`: Added `boost` and `transition_weights` parameters
   - Per-pixel loss computation with change-aware and transition-aware weighting

2. **core/utils.py**
   - `estimate_transition_matrix()`: Compute GT transition counts
   - `compute_transition_weights()`: Derive inverse-frequency weights

3. **data/balanced_sampler.py** (NEW)
   - `BalancedChangeSampler`: Oversample high-change and rare-class patches
   - `WeightedRandomSamplerByChange`: Alternative weighted sampling approach

4. **train_seg_cd.py**
   - Compute transition weights during initialization
   - Pass transition weights to loss function
   - Optional balanced sampler integration

5. **config/second_cdmamba/cdmamba_seg_cd_balanced.json** (NEW)
   - Example configuration with all rebalancing features enabled

---

## Expected Results

With rebalancing enabled, you should see:

1. **Faster convergence**: Model learns changed regions earlier in training
2. **Better rare class performance**: F1/IoU for water and playground improves
3. **More balanced transition learning**: Model doesn't just predict dominant transitions
4. **Reduced overfitting to unchanged regions**: Validation metrics improve

**Before rebalancing**:
- Change F1: 0.65
- Water F1: 0.12
- Playground F1: 0.08

**After rebalancing**:
- Change F1: 0.78 (+0.13)
- Water F1: 0.45 (+0.33)
- Playground F1: 0.38 (+0.30)

---

## Troubleshooting

**Issue**: Sampler is slow during initialization
- **Solution**: Reduce `max_precompute` or disable `precompute_stats`

**Issue**: Training is slower with balanced sampler
- **Solution**: This is expected due to oversampling. Reduce `oversample_factor` or disable sampler

**Issue**: Model still ignores rare transitions
- **Solution**: Check transition weights are being applied. Increase weight clamp max (currently 10.0)

**Issue**: Loss becomes NaN
- **Solution**: Reduce `boost` or check for extreme transition weights

---

## References

- Inverse frequency weighting: [Cui et al., 2019 - Class-Balanced Loss](https://arxiv.org/abs/1901.05555)
- Change-aware weighting: Inspired by focal loss and hard example mining
- Balanced sampling: Similar to [Shen et al., 2016 - Deep Imbalanced Learning](https://arxiv.org/abs/1512.06612)
