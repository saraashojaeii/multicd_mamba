# Tech Stack

## Overview
CDMamba is a PyTorch-based research framework for multi-class semantic change detection in bi-temporal remote sensing images. The stack is chosen for research velocity (rich ecosystem, W&B integration, config-driven experiments) with enough production infrastructure to support reproducible publication.

---

## Core Framework

| Layer | Technology | Version | Role |
|---|---|---|---|
| Language | Python | 3.10+ | Primary language |
| Deep Learning | PyTorch | 2.0+ | Model definition, training loop, autograd |
| Sequence Models | mamba-ssm | latest | Mamba state-space blocks (S6 / S4) for linear-complexity token mixing |
| Vision Utils | torchvision | 0.13+ | Image transforms, normalization |
| Model Zoo | timm | 0.9.16+ | Pretrained backbone access (ResNet, Swin, ViT variants) |
| Seg Toolkit | mmsegmentation | 1.2.2+ | Segmentation head components, evaluation utilities |
| Medical Imaging | MONAI | 1.3.0+ | Specialized augmentation and loss utilities (repurposed for remote sensing) |

---

## Model Architecture

### Encoder: Dual-Stream Siamese Mamba
- Two weight-sharing encoders process T1 (pre-change) and T2 (post-change) images independently
- Each encoder: 4-stage pyramid with `SRCMLayer` (Mamba token mixer) + `SRCMBlock` (residual)
- Feature channels: `[C, 2C, 4C, 8C]` across stages
- **Mamba blocks:** `ConvMamba` — bidirectional Mamba v2, grouped channel processing, gated residual: `σ(gate) × mamba_out + (1−σ(gate)) × input`
- **Positional encoding:** Coordinate-conditioned (normalized grid → 2-layer MLP → C dims); resolution-agnostic

### Bottleneck: ContextBlock2D
- Parallel dilated convolutions (dilation 1, 2) for local multi-scale context
- Global average pooling branch for scene-level context
- Fusion via learned weighted sum

### Cross-Temporal Interaction (Optional)
- `L_GF_Mamba`: Interleaved local-global feature fusion between T1 and T2 streams
- Controlled by `use_interaction_block` config flag

### Decoder: SRCMDecoder
- 3-stage upsampling with skip connections from encoder pyramid
- Shared decoder for both T1 and T2 feature reconstruction

### Output Heads
- **Segmentation Head T1:** `[1024→512→256→num_classes]` conv layers → full-resolution semantic map
- **Segmentation Head T2:** Same structure, separate weights
- **Change Head:** `1×1` conv on fused T1/T2 features → 1-channel binary change logit
- **Change-Guided Gating (Optional):** Change score gates decoder features (`use_change_gating`)

---

## Loss Functions

All defined in `models/loss.py` (1326 lines). Primary loss for publication experiments:

### TripletChangeSegLoss
Jointly optimizes segmentation quality, change detection accuracy, and semantic consistency:

```
L_total = λ_seg  · L_seg          (weighted CE + Dice on T1 & T2)
        + λ_cd   · L_change        (BCE + Dice on binary change mask)
        + λ_unch · L_kl            (symmetric KL on unchanged pixels)
        + λ_ch   · L_diversity     (cosine margin on changed pixels)
        + λ_cpl  · L_coupling      (L1 between change head and semantic change)
        + λ_ps   · L_pseudo        (self-supervision on high-confidence unchanged)
```

**Key ablation switches:** `enable_unch_conf_gating`, `enable_pseudo_labeling`, `enable_kl_warmup`, `enable_changed_only_supervision`

### Supporting Losses
- `CombinedLoss`: Lightweight CE + Dice + consistency (fast prototyping)
- `MultiClassCDLoss`: Weighted sum of per-head losses
- `PerPixelDiceLoss`: Spatial-weight-aware Dice
- `UnchangedSymmetricKLLoss`, `ChangedDiversityCosineMarginLoss`, `CouplingChangeSemanticLoss` (individual components, used for ablation)

---

## Training Infrastructure

### Optimizer & Scheduling
- **Optimizer:** AdamW (`lr=5e-5`, `weight_decay=1e-4`, β=(0.9, 0.999))
- **LR Scheduler:** CosineAnnealingLR (`eta_min=5e-7` for SECOND, `1e-6` for Landsat)
- **Gradient Clipping:** `max_norm=0.5`
- **Gradient Accumulation:** `grad_accum=2` (simulates larger effective batch size)
- **Mixed Precision:** `torch.cuda.amp` with automatic loss scaling

### Data Handling
- **Balanced Sampler:** `BalancedChangeSampler` — precomputes change ratios, oversamples changed scenes (factor 2.0), tracks rare classes
- **Class Weights:** Inverse-frequency estimation per epoch
- **Transition Matrix:** Per-epoch estimation of class transition frequencies → inverse-frequency weights for rare transitions
- **Augmentations:** Random flip, rotation, crop, color jitter; normalization to [-1, 1]

### Experiment Tracking
- **W&B (Weights & Biases):** Full metric logging — train/val/test mF1, mIoU, OA, change F1, changed-region IoU, per-class IoU, transition accuracy
- **Checkpointing:** Best model (by val mF1) + periodic epoch saves; full resume support
- **Seeding:** NumPy, PyTorch, CUDA seeds fixed for reproducibility

---

## Datasets

### SECOND (Primary Benchmark)
- Bi-temporal aerial imagery, 512×512 patches
- 7 classes: background, impervious surface, low vegetation, trees, water, playground, building
- Characteristics: Highly imbalanced (changed pixels ~3–10% of total)
- Split: standard train/val/test

### Landsat
- Bi-temporal satellite imagery, 416×416 patches
- 5 classes: background, low vegetation, urban, building, water
- Multi-sensor, lower spatial resolution than SECOND

### HICD
- High-resolution building change detection
- Config: `config/HICD_cdmamba/`

---

## Evaluation Metrics

| Metric | Implementation | Primary Use |
|---|---|---|
| mF1 (seg) | `core/metrics.py` | Best model selection |
| mIoU (seg) | `core/metrics.py` | SOTA comparison |
| SCD F1 / S_ek | `core/Eval_SCD.py` | Composite semantic+change quality |
| Change F1 / IoU | `misc/metric_tools.py` | Binary change detection |
| Changed-region IoU | Custom mask-gated IoU | Localization reliability |
| Per-class IoU/F1 | Confusion matrix diagonal | Fine-grained class analysis |
| Transition accuracy | Transition matrix tracking | Semantic transition correctness |

---

## Baseline Comparisons

Implemented in `models/` for fair comparison under identical training conditions:

| Model | Type | Reference |
|---|---|---|
| BiFA | Transformer (spatial-spectral attention) | Siamese bi-temporal |
| PaFormer | Transformer (pyramid attention) | Multi-scale change |
| DARNet | CNN (diverse receptive field) | Attention-based |
| IFNET | CNN (deep information fusion) | Multi-scale |
| DMINet | CNN (dilated multi-scale) | Lightweight |
| MSCANet | CNN (multi-scale channel attention) | Efficient |
| FC-EF / FC-Siam-Conc / FC-Siam-Diff | CNN Siamese | Early-fusion baselines |
| SNUNet | CNN (dense skip connections) | Dense prediction |

---

## Development Environment

```
OS:       Linux (CUDA server) / macOS (development)
Python:   3.10
CUDA:     11.7+
GPU:      Minimum 16GB VRAM (A100 / V100 recommended for 512×512, batch 2 + grad accum)
Storage:  ~50GB for SECOND + Landsat datasets + checkpoints
```

### Key Dependencies (from requirement.txt)
```
torch>=1.12.0
torchvision>=0.13.0
mamba-ssm
monai>=1.3.0
einops>=0.7.0
kornia>=0.7.0
timm>=0.9.16
transformers>=4.38.2
mmsegmentation>=1.2.2
opencv-python-headless>=4.8.0.76
scipy, numpy, pandas, matplotlib
accelerate>=0.20.0
wandb
```

---

## Configuration System
All experiments are driven by JSON config files in `config/`:
- `config/second_cdmamba/cdmamba_seg_cd_balanced.json` — SECOND training config
- `config/landsat_cdmamba/cdmamba_seg_cd_balanced.json` — Landsat training config
- `config/HICD_cdmamba/hicd_cdmamba.json` — HICD config

Configs control: model hyperparameters, loss weights, optimizer, scheduler, sampler, augmentation, dataset paths, W&B project/run naming.

---

## Design Decisions & Rationale

**Why Mamba over Transformers?**
Mamba's selective state-space model (S6) provides O(n) complexity in sequence length vs. O(n²) for self-attention. At 512×512 resolution, each feature map has 16K–262K tokens — making full attention computationally prohibitive. Mamba enables global context without the quadratic cost.

**Why joint segmentation + change detection?**
Training the two tasks jointly forces the model to learn semantically consistent representations: the change head learns *where* to look; the segmentation heads learn *what* the region contains. The coupling loss explicitly enforces this co-learning.

**Why TripletChangeSegLoss vs. simple CE?**
Simple CE treats T1 and T2 independently and cannot enforce temporal consistency. The triplet-style loss adds:
- Unchanged-KL: penalizes drift on stable regions
- Changed-diversity: encourages semantic dissimilarity on actually-changed regions
- Coupling: aligns change head signal with semantic-level change evidence
These terms directly target the two core failure modes: false change predictions and wrong class assignments.

**Why balanced sampling?**
SECOND has <10% changed pixels per image. Without oversampling, the model sees overwhelmingly unchanged scenes and learns a strong unchanged prior — suppressing sensitivity to actual change. The balanced sampler enforces ~50% changed scenes per batch and double-samples images containing rare classes (water, playground).
