# Test Results Location

## Where Are Test Results Saved?

When you run `test_change.py`, all results are saved to:

```
{path_cd.result}/{experiment_folder}/test/
```

Based on your config:
```
/root/home/pvc/01_change_detection/Building_changedetection_job/experiments/results/{experiment_name}_{dataset}_{timestamp}/test/
```

---

## Files Saved

### 1. **`test_metrics.json`** (NEW - Comprehensive Metrics)

**Location**: `{test_result_path}/test_metrics.json`

Contains **all** test metrics in a structured JSON format:

```json
{
  "binary_change_detection": {
    "precision": 0.8234,
    "recall": 0.7891,
    "f1": 0.8059,
    "iou": 0.6745,
    "accuracy": 0.9123,
    "sek": 0.6234
  },
  "semantic_segmentation": {
    "precision_macro": 0.7823,
    "recall_macro": 0.7456,
    "f1": 0.7634,
    "iou": 0.6234,
    "accuracy": 0.8567,
    "sek": 0.5678
  },
  "eval_scd_metrics": {
    "mean_iou": 0.6745,
    "sek": 0.6234,
    "score": 0.6387,
    "sc_precision": 0.7823,
    "sc_recall": 0.7456,
    "f_scd": 0.7634
  },
  "changed_pixels_only": {
    "iou": 0.5234,
    "f1": 0.6789,
    "accuracy": 0.7123
  },
  "per_class_metrics": {
    "iou_per_class": [0.65, 0.72, 0.58, 0.43, 0.81, 0.39],
    "f1_per_class": [0.78, 0.84, 0.73, 0.60, 0.89, 0.56],
    "precision_per_class": [...],
    "recall_per_class": [...]
  },
  "transition_metrics": {
    "1_to_4": {
      "accuracy": 0.8234,
      "count": 1234,
      "correct_t1": 1000,
      "correct_t2": 980
    },
    ...
  },
  "transition_matrix": {
    "matrix_counts": [[...], [...]],
    "matrix_percentages_global": [[...], [...]],
    "matrix_percentages_row": [[...], [...]],
    "class_names": ["low_veg", "nvg_surf", "tree", "water", "building", "playground"],
    "total_pixels": 12345678,
    "change_pixel_ratio": 0.1234
  }
}
```

**This is the main file you want to check for all metrics!**

---

### 2. **`transition_matrix.json`** (Backward Compatibility)

**Location**: `{test_result_path}/transition_matrix.json`

Contains only the transition matrix (subset of `test_metrics.json`):

```json
{
  "matrix_counts": [[12345, 3456, ...], ...],
  "matrix_percentages_global": [[0.12, 0.03, ...], ...],
  "matrix_percentages_row": [[50.2, 14.1, ...], ...],
  "class_names": ["low_veg", "nvg_surf", "tree", "water", "building", "playground"],
  "total_pixels": 12345678,
  "change_pixel_ratio": 0.1234
}
```

---

### 3. **Visualization Images** (First 10 Batches)

**Location**: `{test_result_path}/img_*.png`

- `img_A_{batch_id}.png` - Input image T1
- `img_B_{batch_id}.png` - Input image T2
- `img_gt_cm_{batch_id}.png` - Ground truth change mask
- `img_pred_cm_{batch_id}.png` - Predicted change mask

Only saved for the first 10 batches to avoid filling up disk space.

---

### 4. **Console Log**

**Location**: `{path_cd.log}/{experiment_folder}/test_*.log`

Contains all console output including:
- Detailed metrics
- Transition matrix tables
- Progress information

---

### 5. **Weights & Biases (if enabled)**

If you're using W&B, all metrics are also logged to your W&B project:

**Project**: `BuildingCD_mamba_based` (from your config)

**Metrics logged**:
- `test/precision_binary_change`
- `test/recall_binary_change`
- `test/f1_binary_change`
- `test/iou_binary_change`
- `test/sek_binary_change`
- `test/scd_mean_iou`
- `test/scd_sek`
- `test/scd_score`
- `test/sc_precision`
- `test/sc_recall`
- `test/f_scd`
- And many more...

Plus heatmap visualizations of the transition matrix.

---

## Example: Finding Your Results

### Step 1: Run Test

```bash
python test_change.py \
    -c config/second_cdmamba/cdmamba_seg_cd_balanced.json \
    --weights path/to/checkpoint.pth
```

### Step 2: Check Console Output

Look for this line in the console:
```
Results saved to /root/home/pvc/.../experiments/results/cdmamba_seg_cd_balanced_SECOND_20260204_095412/test/
```

### Step 3: Navigate to Results

```bash
cd /root/home/pvc/01_change_detection/Building_changedetection_job/experiments/results/cdmamba_seg_cd_balanced_SECOND_20260204_095412/test/
```

### Step 4: View Metrics

```bash
# View all metrics
cat test_metrics.json | jq '.'

# View specific metric
cat test_metrics.json | jq '.eval_scd_metrics'

# View binary change detection metrics
cat test_metrics.json | jq '.binary_change_detection'
```

---

## Quick Access to Key Metrics

### Binary Change Detection Performance

```bash
cat test_metrics.json | jq '.binary_change_detection'
```

Output:
```json
{
  "precision": 0.8234,
  "recall": 0.7891,
  "f1": 0.8059,
  "iou": 0.6745,
  "accuracy": 0.9123,
  "sek": 0.6234
}
```

### Eval_SCD Metrics (Official SCD Benchmark)

```bash
cat test_metrics.json | jq '.eval_scd_metrics'
```

Output:
```json
{
  "mean_iou": 0.6745,
  "sek": 0.6234,
  "score": 0.6387,
  "sc_precision": 0.7823,
  "sc_recall": 0.7456,
  "f_scd": 0.7634
}
```

### Per-Class Performance

```bash
cat test_metrics.json | jq '.per_class_metrics'
```

### Transition Matrix

```bash
cat test_metrics.json | jq '.transition_matrix'
```

---

## Summary

**Main metrics file**: `test_metrics.json` in the test results directory

**Path structure**:
```
{config.path_cd.result}/
└── {experiment_name}_{dataset}_{timestamp}/
    └── test/
        ├── test_metrics.json          ← All metrics here!
        ├── transition_matrix.json     ← Transition matrix only
        ├── img_A_0.png               ← Visualizations
        ├── img_B_0.png
        ├── img_gt_cm_0.png
        └── img_pred_cm_0.png
```

**To find your results**:
1. Look for the console message: `Results saved to ...`
2. Navigate to that directory
3. Open `test_metrics.json` for all metrics

**To view in terminal**:
```bash
cat {test_result_path}/test_metrics.json | jq '.'
```
