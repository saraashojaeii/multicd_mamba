# Precomputing Balanced Sampler Statistics

## Problem

The `BalancedChangeSampler` needs to analyze every training sample to determine:
- Which samples have high change ratios
- Which samples contain rare classes

This can take **several minutes** at the start of every training run.

## Solution

Precompute the statistics **once** and save them to a file. Then load them instantly during training.

---

## Step 1: Precompute Statistics

Run the precomputation script once:

```bash
python precompute_sampler_stats.py \
    --config config/second_cdmamba/cdmamba_seg_cd_balanced.json \
    --phase train
```

**Output:**
```
Loading config from: config/second_cdmamba/cdmamba_seg_cd_balanced.json

Sampler configuration:
  Change threshold: 0.01
  Rare classes: [3, 5]
  Max precompute: 1000

Creating train dataset...
Dataset size: 5000

Precomputing statistics for 1000/5000 samples...
100%|████████████████████████████████████| 1000/1000 [02:15<00:00,  7.38it/s]

Statistics:
  High change samples: 234 (23.4%)
  Rare class samples: 156 (15.6%)
  Regular samples: 610 (61.0%)

Saving statistics to: config/second_cdmamba/sampler_stats/train_sampler_stats.json

✅ Done! Statistics saved to: config/second_cdmamba/sampler_stats/train_sampler_stats.json

To use these statistics, add to your config:
  "balanced_sampler": {
    "precompute_stats": false,
    "stats_file": "config/second_cdmamba/sampler_stats/train_sampler_stats.json"
  }
```

This takes **~2-3 minutes once**, instead of every training run.

---

## Step 2: Update Your Config

Add the `stats_file` path to your config:

```json
{
  "train": {
    "use_balanced_sampler": true,
    "balanced_sampler": {
      "change_threshold": 0.01,
      "rare_classes": [3, 5],
      "oversample_factor": 2.0,
      "precompute_stats": false,
      "stats_file": "config/second_cdmamba/sampler_stats/train_sampler_stats.json"
    }
  }
}
```

**Important**: Set `"precompute_stats": false` to avoid recomputing!

---

## Step 3: Train

Start training as usual:

```bash
python train_seg_cd.py -c config/second_cdmamba/cdmamba_seg_cd_balanced.json
```

**Output:**
```
[BalancedChangeSampler] Loading precomputed stats from: config/second_cdmamba/sampler_stats/train_sampler_stats.json
  Loaded statistics:
    High change: 234 samples
    Rare classes: 156 samples
    Regular: 610 samples
Using BalancedChangeSampler with change_threshold=0.01
```

Training starts **immediately** (no precomputation delay)!

---

## Advanced Options

### Custom Output Path

```bash
python precompute_sampler_stats.py \
    --config config/second_cdmamba/cdmamba_seg_cd_balanced.json \
    --output /path/to/my_stats.json
```

### Different Dataset Phase

```bash
# Precompute for validation set
python precompute_sampler_stats.py \
    --config config/second_cdmamba/cdmamba_seg_cd_balanced.json \
    --phase val
```

### Limit Number of Samples

Edit your config to analyze fewer samples (faster precomputation):

```json
"balanced_sampler": {
  "max_precompute": 500  // Only analyze first 500 samples
}
```

---

## When to Recompute

You need to recompute statistics if:
- ✅ Your dataset changes (new samples added/removed)
- ✅ You change `change_threshold` or `rare_classes` in config
- ✅ You change `max_precompute` to analyze more samples

You **don't** need to recompute if:
- ❌ You change learning rate, batch size, or other training params
- ❌ You change model architecture
- ❌ You change loss weights

---

## File Format

The stats file is a JSON with this structure:

```json
{
  "high_change_indices": [12, 45, 67, ...],
  "rare_class_indices": [23, 56, 89, ...],
  "regular_indices": [0, 1, 2, ...],
  "change_ratios": {
    "0": 0.0023,
    "1": 0.0156,
    ...
  },
  "class_presence": {
    "0": [0, 1, 2],
    "1": [0, 1, 3, 4],
    ...
  },
  "num_samples": 5000,
  "max_precomputed": 1000,
  "change_threshold": 0.01,
  "rare_classes": [3, 5]
}
```

---

## Troubleshooting

### "Stats file has X samples, dataset has Y"

Your dataset size changed. Recompute the statistics:

```bash
python precompute_sampler_stats.py --config your_config.json
```

### "FileNotFoundError: stats file not found"

Check the path in your config. Use absolute path or relative to project root:

```json
"stats_file": "/absolute/path/to/stats.json"
// or
"stats_file": "config/second_cdmamba/sampler_stats/train_sampler_stats.json"
```

### Still slow?

Make sure you set `"precompute_stats": false` in your config when using `stats_file`.

---

## Summary

**Without precomputed stats:**
```
Training startup: 5-10 minutes (precomputing every time)
```

**With precomputed stats:**
```
One-time precomputation: 2-3 minutes
Training startup: <5 seconds (instant loading)
```

**Total time saved per training run: ~5-10 minutes!**
