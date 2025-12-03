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

### Multi-class Change Detection

```bash
python train_seg_cd.py \
  --config config/second_cdmamba/cdmamba_seg_cd_multiclass.json \
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
