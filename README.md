# CDMamba: Multi-class Change Detection

Remote sensing image multi-class change detection using Mamba architecture.

## Installation

**Requirements:** Python 3.10, PyTorch 2.0+, CUDA 11.7+

```bash
conda create -n cd_mamba python=3.10
conda activate cd_mamba
pip install -r requirements.txt
```

```shell
pip uninstall opencv-python opencv-python-headless opencv-contrib-python -y
pip install opencv-python-headless==4.8.0.76
```
**Note:** For `mamba-ssm` installation help, see [Vim repository](https://github.com/hustvl/Vim)

## Dataset Structure

Organize your dataset as follows:

```
${DATASET_ROOT}/
├── A/          # T1 images
├── B/          # T2 images
├── label/      # Ground truth labels
└── list/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

## Training

### Precompute Sampler Statistics (Optional but Recommended)

If using balanced sampling (`use_balanced_sampler: true`), precompute change ratios and rare class statistics:

```bash
python data/balanced_sampler.py \
  --dataset_name SECOND \
  --dataset_root /path/to/SECOND/train \
  --phase train \
  --output_file config/second_cdmamba/sampler_stats/train_sampler_stats.json \
  --change_threshold 0.01 \
  --rare_classes 3 5
```

**For Landsat dataset:**
```bash
python data/balanced_sampler.py \
  --dataset_name Landsat \
  --dataset_root /path/to/Landsat/train \
  --phase train \
  --output_file config/landsat_cdmamba/sampler_stats/train_sampler_stats.json \
  --change_threshold 0.01 \
  --rare_classes 3 5
```

**Arguments:**
- `--dataset_name`: Dataset name (SECOND, Landsat, etc.)
- `--dataset_root`: Path to dataset split (train/val/test)
- `--phase`: Dataset phase (train, val, test)
- `--output_file`: Where to save precomputed stats JSON
- `--change_threshold`: Minimum change ratio to consider (default: 0.01)
- `--rare_classes`: Space-separated list of rare class IDs to track

This precomputation speeds up training startup and ensures consistent sampling across runs.

### Multi-class Change Detection

```bash
python train_seg_cd.py \
  --config config/second_cdmamba/cdmamba_seg_cd_balanced.json \
  --phase train \
  --dataset SECOND \
  --tag exp1 \
  --seed 123
```

**Key Arguments:**
- `--config`: Path to JSON config file (contains dataset paths, batch size, epochs, model parameters)
- `--phase`: `train` or `test`
- `--tag`: Custom experiment name (optional)
- `--max_train_batches`: Limit training batches for quick tests (0 = no limit)

Results are saved to directories specified in the config file under `path_cd`.

## Where outputs go

- Logs, tensorboard/WANDB: under `path_cd.log/<exp_folder>`.
- Checkpoints: `path_cd.checkpoint/<exp_folder>/best_net.pth` and epoch checkpoints (via `misc.torchutils.save_network`).
- Results: `path_cd.result/<exp_folder>/test`.