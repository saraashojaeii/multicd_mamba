# Training Sanity Check Output

When you start training, you should see these configuration summaries to verify your config is properly wired:

## Example Output (Full Model - All Features Enabled)

```
================================================================================
Model Architecture Configuration:
--------------------------------------------------------------------------------
  Change-Guided Gating:   ENABLED (α=1.00, β=0.20, mode=additive)
  Interaction Block:      ENABLED (heads=4, mamba=False)
  Change Head:            ENABLED
================================================================================

Total params: 12,345,678 | Trainable: 12,345,678 (100.00%)

================================================================================
TripletChangeSegLoss Configuration Summary:
--------------------------------------------------------------------------------
  Seg Masking Mode:       changed_only (min_changed=10)
  Pseudo-labeling:        ENABLED (λ=0.100, tau=0.90)
  Unch Conf Gating:       ENABLED (tau=0.90, method=max_prob)
  KL Warmup:              5 epochs (0 → 0.100)
  Changed-only Superv:    ENABLED
  Loss Weights:           λ_seg=1.50, λ_cd=1.00, λ_unch=0.10, λ_ch=0.20
================================================================================
```

## Example Output (Baseline - All Features Disabled)

```
================================================================================
Model Architecture Configuration:
--------------------------------------------------------------------------------
  Change-Guided Gating:   DISABLED (α=1.00, β=0.20, mode=additive)
  Interaction Block:      DISABLED (heads=4, mamba=False)
  Change Head:            ENABLED
================================================================================

Total params: 11,234,567 | Trainable: 11,234,567 (100.00%)

================================================================================
TripletChangeSegLoss Configuration Summary:
--------------------------------------------------------------------------------
  Seg Masking Mode:       full (min_changed=10)
  Pseudo-labeling:        DISABLED (λ=0.000, tau=0.90)
  Unch Conf Gating:       DISABLED (tau=0.90, method=max_prob)
  KL Warmup:              0 epochs (0 → 0.100)
  Changed-only Superv:    DISABLED
  Loss Weights:           λ_seg=1.50, λ_cd=1.00, λ_unch=0.10, λ_ch=0.20
================================================================================
```

## What to Check

### Model Architecture
- **Change-Guided Gating**: Should be ENABLED for full model, DISABLED for baseline
- **Interaction Block**: Should be ENABLED for full model, DISABLED for baseline
- **Change Head**: Should always be ENABLED

### Loss Configuration
- **Seg Masking Mode**: 
  - `changed_only` = supervise only changed pixels (full model)
  - `full` = supervise all pixels (baseline)
  - `mixed` = supervise changed strongly, unchanged weakly
  
- **Pseudo-labeling**: 
  - ENABLED with λ > 0 for full model
  - DISABLED with λ = 0 for baseline
  
- **Unch Conf Gating**: 
  - ENABLED for full model (filters low-confidence unchanged pixels from KL loss)
  - DISABLED for baseline
  
- **KL Warmup**: 
  - > 0 epochs for full model (gradually ramp up unchanged KL loss)
  - 0 epochs for baseline (use full strength from start)
  
- **Changed-only Superv**: 
  - ENABLED for full model
  - DISABLED for baseline

## Ablation Study Checklist

When running ablation experiments, verify:

1. ✅ Baseline shows all new features DISABLED
2. ✅ Each ablation step shows only the intended feature ENABLED
3. ✅ Full model shows all features ENABLED
4. ✅ Loss weights match your config file
5. ✅ Model architecture matches your config file

## Common Issues

**If you see unexpected values:**
1. Check your JSON config file for typos
2. Verify the config path is correct
3. Check for conflicting ablation switches
4. Ensure `enable_*` flags match the feature settings

**If features aren't working:**
1. Verify the sanity check shows them as ENABLED
2. Check lambda values are > 0 for loss components
3. Verify ablation switches are set correctly
