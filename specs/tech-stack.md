# Tech Stack

**Last updated:** August 25, 2026

## Overview
CDMamba is a PyTorch-based research framework for multi-class semantic change detection in bi-temporal remote sensing images. The stack is chosen for research velocity (rich ecosystem, W&B integration, config-driven experiments) with enough infrastructure to support reproducible publication.

---

## Core Framework

| Layer | Technology | Version | Role |
|---|---|---|---|
| Language | Python | 3.10+ | Primary language |
| Deep Learning | PyTorch | 2.0+ | Model definition, training loop, autograd |
| Sequence Models | mamba-ssm | latest | Mamba state-space blocks (S6 / S4) |
| Vision Utils | torchvision | 0.13+ | Image transforms, normalization |
| Model Zoo | timm | 0.9.16+ | Pretrained backbones (ResNet, Swin, ViT) |
| Seg Toolkit | mmsegmentation | 1.2.2+ | Segmentation head components |
| Morphology | kornia | 0.7.0+ | Dilation-erosion boundary ring |
| DT Losses | MONAI | 1.3.0+ | HausdorffDTLoss |

---

## Model Architecture

### Encoder: Dual-Stream Siamese Mamba
- Two weight-sharing encoders process T1 and T2 independently
- 4-stage pyramid: `SRCMLayer` (Mamba token mixer) + `SRCMBlock` (residual)
- Channels `[C, 2C, 4C, 8C]`
- **Mamba blocks:** `ConvMamba` — bidirectional Mamba v2, grouped channel processing, gated residual `σ(gate)·mamba_out + (1−σ(gate))·input`
- **PE:** coordinate-conditioned (normalized grid → 2-layer MLP → C dims), resolution-agnostic

### Bottleneck: ContextBlock2D
Parallel dilated convs (dilation 1, 2) + global average pooling branch, fused by learned weighted sum.

### Cross-Temporal Interaction (Optional)
`L_GF_Mamba` — interleaved local-global fusion between streams. Flag: `use_interaction_block`.

### Decoder: SRCMDecoder
3-stage upsampling, skip connections from the encoder pyramid, shared across T1/T2.

### Output Heads
- **Segmentation T1 / T2:** `[1024→512→256→n_classes]`, separate weights
- **Change head:** 1×1 conv on fused features → 1-channel binary logit
- **Change-guided gating (optional):** change score gates decoder features (`use_change_gating`)

---

## Loss Functions

All in `models/loss.py`. Primary loss for publication experiments:

### TripletChangeSegLoss
```
L_total = λ_seg  · L_seg          (weighted CE + Dice on T1 & T2)
        + λ_cd   · L_change       (BCE + Dice on binary change mask)
        + λ_unch · L_kl           (symmetric KL on unchanged pixels)
        + λ_ch   · L_diversity    (cosine margin on changed pixels)
        + λ_cpl  · L_coupling     (L1 between change head and semantic change)
        + λ_ps   · L_pseudo       (self-supervision on high-confidence unchanged)
        + λ_bnd  · L_laplacian    (thin-edge boundary)
        + λ_morph· L_morph        (ChangeHeadMorphBoundaryLoss, ~7px kornia ring)
        + λ_hd   · L_hausdorff    (ChangeHeadHausdorffLoss, MONAI DT)
```

**Ablation switches:** `enable_unch_conf_gating`, `enable_pseudo_labeling`, `enable_kl_warmup`, `enable_changed_only_supervision`, `enable_boundary_warmup` (+ `boundary_warmup_epochs`).

**Supporting losses:** `CombinedLoss`, `MultiClassCDLoss`, `PerPixelDiceLoss`, and the individual components (`UnchangedSymmetricKLLoss`, `ChangedDiversityCosineMarginLoss`, `CouplingChangeSemanticLoss`) exposed for leave-one-out ablation.

---

## Evaluation Metrics

> **This section is normative.** It was rewritten in August 2026 after the previous headline result turned out to be a masked diagnostic. Read `mission.md` § Evaluation Protocol before touching metric code.

### Standard SCD protocol (the only benchmark numbers)

Confusion matrix accumulated over **T1 and T2 predictions concatenated, across all pixels**, with class 0 = "no change."

| Metric | Formula | Implementation |
|---|---|---|
| `Sek` | `kappa_n0 · exp(IoU_fg) / e` | `cm2score()` on unmasked `test_metric` |
| `F_scd` | harmonic mean of SC_Precision, SC_Recall | same |
| `IoU_mean` | mean of 2×2 collapsed unchanged/changed IoU | same |
| `Score` | `0.3·IoU_mean + 0.7·Sek` | to be added in Phase M1 |
| `mF1`, `mIoU` (seg) | per-class F1 / IoU over all pixels | `cm2score()` |
| Binary change F1 / IoU | change head vs. GT change mask | `test_seg_cd.py` TP/FP/FN counters |

`kappa_n0` uses `hist` with `hist[0][0]` zeroed, so the dominant correctly-predicted-unchanged cell does not inflate kappa. The `exp(IoU_fg)/e` factor damps the score by how well change is localized — which is precisely what the masked variants destroy.

**Reference implementation:** `core/Eval_SCD.py`. Currently dead code (broken import, hardcoded paths) — repaired in Phase M0 and used as the cross-validation oracle. The live pipeline must match it.

### Diagnostics (never benchmark numbers)

| Metric | What it does | Why it is not Sek |
|---|---|---|
| `Sek@GT-mask` | `test_metric_gt_masked` — sets **both** pred and GT to `ignore_index` where `gt_change == 0` | Hands the model the GT change mask; row 0 of the histogram empties, so `IoU_fg → 1`, `exp(IoU_fg)/e → 1`, and Sek collapses to plain kappa over changed pixels |
| `Sek@pred-mask` | `test_metric_pred_masked` — masks by predicted change | Excludes all false negatives from scoring |
| Changed-region IoU | mask-gated IoU on GT-changed pixels | Oracle localization |
| Per-class IoU / F1 | confusion diagonal | Fine-grained analysis only |
| Transition accuracy | transition matrix tracking | Fine-grained analysis only |

The `Sek` vs. `Sek@GT-mask` gap decomposes error into localization vs. classification. That decomposition is worth a paragraph in the paper's analysis section — it is a legitimate contribution, just not a benchmark row.

---

## Training Infrastructure

### Optimizer & Scheduling
- AdamW, `lr=5e-5`, `weight_decay=1e-4`, β=(0.9, 0.999)
- CosineAnnealingLR, `eta_min=5e-7` (SECOND) / `1e-6` (Landsat)
- Gradient clipping `max_norm=0.5`; accumulation `grad_accum=2`
- Mixed precision via `torch.cuda.amp`

### Data Handling
- `BalancedChangeSampler` — precomputed change ratios, oversamples changed scenes (factor 2.0), tracks rare classes
- Inverse-frequency class weights, re-estimated per epoch
- Per-epoch transition-frequency matrix → rare-transition weights
- Augmentation: random flip, rotation, crop, color jitter; normalization to [−1, 1]

### Experiment Tracking
- W&B project `scd-balanced-mamba`; runs stamped dataset/tag/seed
- Checkpoints: best by val mF1 + periodic; full resume
- NumPy / PyTorch / CUDA seeds fixed

---

## Datasets

### SECOND (primary)
- Bi-temporal aerial, 512×512
- `n_classes: 7`; `second_colormap` = white, blue, gray, dark green, green, dark red, red
- Changed pixels ~3–10% of total
- Standard train/val/test split

### Landsat-SCD
- Bi-temporal satellite, 416×416
- `n_classes: 5`; 0 no-change (white), 1 farmland (green), 2 desert (orange), 3 building (pink), 4 water (blue)
- Lower spatial resolution, multi-sensor

### HICD
- High-resolution building change; `config/HICD_cdmamba/`
- **Stretch goal** — first item cut if the schedule slips

### Label pipeline
`SCDDataset` loads `label1`/`label2` as RGB, maps to class ids via `rgb_mask_to_class(·, colormap)`, resizes with NEAREST, derives `change_bin = (lab1 != lab2) & valid`, then clamps to `[0, num_classes-1]`.

⚠️ The clamp reads `label_transform.get('num_classes', 6)` while configs declare `n_classes`. If that key does not reach `label_transform`, SECOND class 6 is silently folded into class 5. Phase M3 adds a pre-clamp assertion.

---

## Baselines

### SCD baselines — required for the comparison table
These predict semantic maps at both epochs and can be scored on Sek / F_scd.

| Model | Status | Priority |
|---|---|---|
| Bi-SRNet | ❌ not implemented | 1 |
| SSCD-l | ❌ not implemented | 2 |
| SCanNet | ❌ not implemented | 3 |
| HRSCD-str4 | ❌ not implemented | 4 |
| ChangeMask | ❌ not implemented | 5 (if time) |
| TED | ❌ not implemented | 6 (if time) |

### Binary CD baselines — already implemented
BiFA, PaFormer, DARNet, IFNET, DMINet, MSCANet, ACABFNet, P2V, FC-EF, FC-Siam-Conc, FC-Siam-Diff.

These are **binary** change detection and produce no semantic maps, so they cannot be compared on Sek, F_scd, or mIoU. Use them only for the binary-change-F1 row, and say so explicitly in the paper. Presenting them as the SCD comparison set is the second-highest reviewer risk after the metric issue.

---

## Development Environment

```
OS:       Linux (CUDA server) / macOS (development)
Python:   3.10
CUDA:     11.7+
GPU:      ≥16GB VRAM (A100 / V100 for 512×512, batch 2 + grad accum)
Storage:  ~50GB datasets + checkpoints
```

### Key dependencies (`requirement.txt`)
```
torch>=1.12.0            torchvision>=0.13.0     mamba-ssm
monai>=1.3.0             kornia>=0.7.0           einops>=0.7.0
timm>=0.9.16             transformers>=4.38.2    mmsegmentation>=1.2.2
opencv-python-headless>=4.8.0.76
scipy, numpy, pandas, matplotlib, accelerate>=0.20.0, wandb
```

Note: `README.md` says `pip install -r requirements.txt` but the file is `requirement.txt` (singular). Fix one or the other.

---

## Configuration System

JSON configs in `config/`, controlling model hyperparameters, loss weights, optimizer, scheduler, sampler, augmentation, dataset paths, W&B naming.

⚠️ `config/second_cdmamba/exp_E5_final_seed123.json` and `exp_E5_final_seed999.json` are **invalid JSON** — stray unterminated `"` near line 82 inside `extended_triplet`. Repaired in Phase M3.

**Add a validity check to the run script:**
```bash
for f in config/*/*.json; do
  python -c "import json;json.load(open('$f'))" || echo "BROKEN: $f"
done
```
Configs are the interface to every experiment; a malformed one discovered at launch time costs a GPU slot, and discovered during final runs costs a deadline.

---

## Known Defects

| # | Location | Issue | Fixed in |
|---|---|---|---|
| 1 | `test_seg_cd.py:356–368`, `466–476` | GT-masked diagnostic reported as Sek | M1 |
| 2 | `config/second_cdmamba/exp_E5_final_seed{123,999}.json` | Invalid JSON | M3 |
| 3 | `data/CDDataset.py` | `num_classes` / `n_classes` key mismatch → silent label folding | M3 |
| 4 | `core/Eval_SCD.py` | Broken import (`utils.utils`), hardcoded paths | M0 |
| 5 | `README.md` | `requirements.txt` vs. `requirement.txt` | anytime |

---

## Design Decisions & Rationale

**Why Mamba over Transformers?**
Linear O(n) complexity vs. O(n²) self-attention. At 512×512 each feature map carries 16K–262K tokens, making full attention prohibitive. **This claim is currently asserted, not measured** — ablation A7 (window-attention encoder at matched parameters) is the test. If A7 does not show an advantage, soften the claim rather than defend it; "competitive at lower cost" is a defensible result, and reviewers punish overclaiming harder than modest framing.

**Why joint segmentation + change detection?**
Joint training forces semantically consistent representations: the change head learns *where*, the segmentation heads learn *what*. The coupling loss enforces this explicitly — and is the component most likely to be doing nothing, which ablation L3 will settle.

**Why TripletChangeSegLoss over plain CE?**
CE treats T1 and T2 independently and cannot enforce temporal consistency. The added terms target the two failure modes directly: unchanged-KL penalizes drift on stable regions, changed-diversity encourages dissimilarity where change occurred, coupling aligns the change signal with semantic evidence.

**Why balanced sampling?**
SECOND has <10% changed pixels per image. Without oversampling the model learns a strong unchanged prior and loses sensitivity to real change. The sampler enforces ~50% changed scenes per batch and double-samples rare classes (water, playground).

**Why the evaluation protocol is written down normatively.**
SCD papers vary in how they compute Sek, which epochs they accumulate over, and whether they mask. That variance is exactly how a 20-point discrepancy went unnoticed here. Stating the protocol explicitly in the paper is a credibility asset, not boilerplate.
