# Colormap Configuration Guide

## Overview
This guide explains how to configure dataset-specific colormaps to fix visualization issues in Weights & Biases (wandb) logging.

## Problem
The Landsat dataset was appearing all white in wandb visualizations because it was using the hardcoded SECOND dataset colormap, which has 7 classes, while Landsat only has 5 classes.

## Solution
Colormaps are now configurable in the JSON config files for each dataset.

## Configuration

### 1. SECOND Dataset (7 classes)
**File:** `config/second_cdmamba/cdmamba_seg_cd_balanced.json`

```json
{
  "colormap": [[255, 255, 255], [0, 0, 255], [128, 128, 128], [0, 128, 0], [0, 255, 0], [128, 0, 0], [255, 0, 0]],
  ...
}
```

**Class mapping:**
- Class 0: White (255, 255, 255)
- Class 1: Blue (0, 0, 255)
- Class 2: Gray (128, 128, 128)
- Class 3: Dark Green (0, 128, 0)
- Class 4: Bright Green (0, 255, 0)
- Class 5: Dark Red (128, 0, 0)
- Class 6: Bright Red (255, 0, 0)

### 2. Landsat Dataset (5 classes)
**File:** `config/landsat_cdmamba/cdmamba_seg_cd_balanced.json`

```json
{
  "colormap": [[255, 255, 255], [0, 0, 255], [128, 128, 128], [0, 128, 0], [0, 255, 0]],
  ...
}
```

**Class mapping:**
- Class 0: White (255, 255, 255)
- Class 1: Blue (0, 0, 255)
- Class 2: Gray (128, 128, 128)
- Class 3: Dark Green (0, 128, 0)
- Class 4: Bright Green (0, 255, 0)

## How to Customize

To change the colormap for your dataset:

1. Open your dataset's config JSON file
2. Add or modify the `"colormap"` field at the top level
3. Provide an array of RGB color triplets, one for each class
4. Each color is specified as `[R, G, B]` where values range from 0-255

Example:
```json
{
  "name": "my_dataset",
  "colormap": [
    [255, 0, 0],    // Class 0: Red
    [0, 255, 0],    // Class 1: Green
    [0, 0, 255],    // Class 2: Blue
    [255, 255, 0]   // Class 3: Yellow
  ],
  ...
}
```

## Implementation Details

### Files Modified:

1. **`data/colormap.py`**
   - Added `landsat_colormap` definition
   - Keeps both `second_colormap` and `landsat_colormap` for reference

2. **`core/utils.py`**
   - Updated `create_color_mask()` function to accept optional `colormap` parameter
   - Falls back to SECOND colormap if none provided (backward compatible)

3. **`train_seg_cd.py`**
   - Extracts `colormap` from config
   - Passes colormap to `log_first_batch_to_wandb()` function
   - Updates all visualization calls to use dataset-specific colormap

4. **Config files:**
   - `config/second_cdmamba/cdmamba_seg_cd_balanced.json`
   - `config/landsat_cdmamba/cdmamba_seg_cd_balanced.json`

## Testing

After making changes:
1. Run training with your config
2. Check wandb dashboard for segmentation mask visualizations
3. Verify that colors are distinct and match your expectations
4. Ensure all classes are visible (not all white or all black)

## Notes

- The colormap must have at least as many colors as your `n_classes` value
- Colors should be distinct enough to differentiate classes visually
- Avoid using black `[0, 0, 0]` for class 0 as it may be confused with background
- The first color (index 0) typically represents the background or unchanged class
