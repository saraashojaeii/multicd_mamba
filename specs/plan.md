# Training Plan — Boundary Improvement Experiments

## Context

The model achieves strong semantic class assignment (competitive Sek on Landsat: 81.18 vs SOTA 60.53)
but produces imprecise change-mask boundaries. This plan systematically addresses that failure
through boundary loss ablations, then consolidates the best config into final paper-run seeds.

**Baseline results (already run):**
- Landsat: OA=88.04, F1=69.21, mIoU=61.42, Sek=81.18

Run all experiments on the server. Commands use the absolute path to `train_seg_cd.py` — adjust
if your working directory differs.

---

## What changed in the code

Two new loss classes were added to `models/loss.py`:

- `ChangeHeadMorphBoundaryLoss` — kornia dilation-erosion ring (~7-pixel boundary), up-weighted BCE
- `ChangeHeadHausdorffLoss` — MONAI HausdorffDTLoss on 2-channel change prediction

Both are wired into `TripletChangeSegLoss` via `lambda_morph_boundary` and `lambda_hausdorff`.
`train_seg_cd.py` now reads these from config. A `enable_boundary_warmup` flag holds them
at zero for the first N epochs so the model stabilises before boundary pressure is applied.

---

## Experiment Sequence

---

### E0 — Confirm Baseline (SECOND)

Run to obtain SECOND numbers matching the Landsat baseline run.
Required before comparing E1-E3 gains.

**Config:** `config/second_cdmamba/cdmamba_seg_cd_balanced.json`

```bash
python train_seg_cd.py \
  --config config/second_cdmamba/cdmamba_seg_cd_balanced.json \
  --phase train \
  --dataset SECOND \
  --tag E0_baseline \
  --seed 42
```

**What to record:** OA, mF1, mIoU, Sek, change F1, changed-region IoU

---

### E1 — Higher Laplacian Boundary Weight (SECOND)

Increases `lambda_boundary` from 0.5 → 1.0.
Tests whether the existing thin-edge boundary loss is under-weighted.

**Config:** `config/second_cdmamba/exp_E1_boundary_high.json`

```bash
python train_seg_cd.py \
  --config config/second_cdmamba/exp_E1_boundary_high.json \
  --phase train \
  --dataset SECOND \
  --tag E1_boundary_high \
  --seed 42
```

**What to record:** Same metrics as E0. Compare change F1 and changed-region IoU against E0.

**Decision criterion:** If change F1 improves > 0.5pp over E0 → laplacian weight was the bottleneck.
If no improvement → thin-ring supervision is insufficient; proceed to E2.

---

### E2 — Morphological Boundary Ring (SECOND)

Adds `lambda_morph_boundary=0.5` (kornia dilation-erosion, ~7-pixel ring).
Provides a thicker boundary region for stronger spatial gradient signal.
Both laplacian and morph boundary are warmed up together after epoch 20.

**Config:** `config/second_cdmamba/exp_E2_morph_boundary.json`

```bash
python train_seg_cd.py \
  --config config/second_cdmamba/exp_E2_morph_boundary.json \
  --phase train \
  --dataset SECOND \
  --tag E2_morph_boundary \
  --seed 42
```

**What to record:** Same metrics + W&B `morph_boundary` loss curve.
Look for whether change-mask edges become sharper in visualisations (run `test_seg_cd.py` on val).

**Decision criterion:** If changed-region IoU improves > 1pp vs E0 → morph boundary is the winner.

---

### E3 — Hausdorff Distance-Transform Loss (SECOND)

Adds `lambda_hausdorff=0.1` (MONAI HausdorffDTLoss), warmed up from epoch 20.
Most principled boundary loss — penalises spatial distance to true boundary, not just pixel overlap.
Note: slightly slower per step due to distance transform computation.

**Config:** `config/second_cdmamba/exp_E3_hausdorff.json`

```bash
python train_seg_cd.py \
  --config config/second_cdmamba/exp_E3_hausdorff.json \
  --phase train \
  --dataset SECOND \
  --tag E3_hausdorff \
  --seed 42
```

**What to record:** Same metrics + `hausdorff` loss curve + per-epoch training time (check if slowdown is acceptable).

**Decision criterion:** Best of E1/E2/E3 by changed-region IoU becomes the "winning config" for E4-E6.

---

### E4 — Best Boundary Config on Landsat

Apply the winning config from E1-E3 to Landsat.
Config is pre-set to E2 (morph boundary) — update `lambda_*` values if E1 or E3 won.

**Config:** `config/landsat_cdmamba/exp_E4_best_boundary.json`

```bash
python train_seg_cd.py \
  --config config/landsat_cdmamba/exp_E4_best_boundary.json \
  --phase train \
  --dataset Landsat \
  --tag E4_best_boundary \
  --seed 42
```

**What to record:** Full metric table. Compare OA, F1, mIoU, Sek against Landsat baseline.
Target: OA > 90, change F1 > 75, Sek ≥ 81.18 (must not regress).

---

### E5 — Final Paper Runs: SECOND (3 seeds)

Run the best config 3 times with different seeds to get mean ± std for the paper.

**Before running:** Open `exp_E5_final_seed*.json` and confirm the `lambda_boundary`,
`lambda_morph_boundary`, `lambda_hausdorff` values match the winner from E1-E3.

**Config files:**
- `config/second_cdmamba/exp_E5_final_seed42.json`
- `config/second_cdmamba/exp_E5_final_seed123.json`
- `config/second_cdmamba/exp_E5_final_seed999.json`

```bash
# Seed 42
python train_seg_cd.py \
  --config config/second_cdmamba/exp_E5_final_seed42.json \
  --phase train \
  --dataset SECOND \
  --tag E5_final \
  --seed 42

# Seed 123
python train_seg_cd.py \
  --config config/second_cdmamba/exp_E5_final_seed123.json \
  --phase train \
  --dataset SECOND \
  --tag E5_final \
  --seed 123

# Seed 999
python train_seg_cd.py \
  --config config/second_cdmamba/exp_E5_final_seed999.json \
  --phase train \
  --dataset SECOND \
  --tag E5_final \
  --seed 999
```

**What to record:** Report mean ± std across 3 seeds for all metrics in the paper table.

---

### E6 — Final Paper Runs: Landsat (3 seeds)

Same as E5 but for Landsat.

**Config files:**
- `config/landsat_cdmamba/exp_E6_final_seed42.json`
- `config/landsat_cdmamba/exp_E6_final_seed123.json`
- `config/landsat_cdmamba/exp_E6_final_seed999.json`

```bash
# Seed 42
python train_seg_cd.py \
  --config config/landsat_cdmamba/exp_E6_final_seed42.json \
  --phase train \
  --dataset Landsat \
  --tag E6_final \
  --seed 42

# Seed 123
python train_seg_cd.py \
  --config config/landsat_cdmamba/exp_E6_final_seed123.json \
  --phase train \
  --dataset Landsat \
  --tag E6_final \
  --seed 123

# Seed 999
python train_seg_cd.py \
  --config config/landsat_cdmamba/exp_E6_final_seed999.json \
  --phase train \
  --dataset Landsat \
  --tag E6_final \
  --seed 999
```

---

## Evaluation (test split)

After each training run, evaluate on the test split using:

```bash
python train_seg_cd.py \
  --config <same_config_used_for_training> \
  --phase test
```

Or use `test_seg_cd.py` if you prefer a standalone eval script.

---

## Summary Table (fill in as you run)

| Exp | Dataset | lambda_boundary | lambda_morph | lambda_hd | OA | F1 | mIoU | Sek | Changed IoU |
|-----|---------|----------------|-------------|-----------|-----|-----|------|-----|-------------|
| E0  | SECOND  | 0.5 | 0.0 | 0.0 | | | | | |
| E1  | SECOND  | 1.0 | 0.0 | 0.0 | | | | | |
| E2  | SECOND  | 0.5 | 0.5 | 0.0 | | | | | |
| E3  | SECOND  | 0.5 | 0.0 | 0.1 | | | | | |
| E4  | Landsat | 0.5 | 0.5 | 0.0 | | | | | |
| E5s42 | SECOND | best | best | best | | | | | |
| E5s123 | SECOND | best | best | best | | | | | |
| E5s999 | SECOND | best | best | best | | | | | |
| E6s42 | Landsat | best | best | best | | | | | |
| E6s123 | Landsat | best | best | best | | | | | |
| E6s999 | Landsat | best | best | best | | | | | |

**Baseline (already run):**
| — | Landsat | 0.5 (implicit) | 0.0 | 0.0 | 88.04 | 69.21 | 61.42 | 81.18 | — |

---

## Decision Tree

```
Run E0 (SECOND baseline)
  └─ Run E1 (laplacian ×2)
       ├─ change F1 improves > 0.5pp? → Use E1 config, skip E2/E3
       └─ No improvement → Run E2 (morph ring)
            ├─ changed-region IoU improves > 1pp? → Use E2 config
            └─ Marginal → Run E3 (Hausdorff)
                 └─ Compare E2 vs E3; pick winner by changed-region IoU
                      └─ Apply winner to Landsat (E4)
                           └─ Run E5 + E6 (3 seeds each) for paper table
```

---

## Notes

- All experiments use seed 42 for the ablation phase (E0-E4), then 3 seeds (42, 123, 999) for final runs.
- W&B run names are automatically stamped with dataset/tag/seed — filter by project `scd-balanced-mamba`.
- If E3 (Hausdorff) causes > 20% slowdown in training time, drop it and use E2 instead.
- Do not run E5/E6 until E1-E4 are complete and the winning config is confirmed.
- The `_comment` fields in the E5/E6 configs remind you to fill in the winning boundary lambdas.
