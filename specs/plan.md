# Experiment Plan — CVPR 2027 Cycle

**Last updated:** August 25, 2026
**Supersedes:** the previous boundary-only plan, whose baseline row (`Sek=81.18`) was a GT-masked diagnostic, not standard-protocol Sek.

---

## Context

The previous plan diagnosed "strong semantic class assignment, imprecise change-mask boundaries" from a Landsat run reporting `Sek=81.18` against `SOTA 60.53`. That figure came from `test_metric_gt_masked`, which masks **both** prediction and ground truth outside the GT change mask — handing the model the ground-truth change mask for free and degenerating the Sek formula (see `mission.md`, Evaluation Protocol).

The diagnosis may still be right. It is currently unsupported. **Phase M re-establishes the ground truth about where the model stands; every later phase is conditional on it.**

Run everything on the server. Commands assume repository root as working directory.

---

## Phase M — Metric Correction & Re-Baseline
*No training. Gate for everything downstream.*

### M0 — Repair the reference implementation

`core/Eval_SCD.py` currently imports `utils.utils` (does not exist; the repo has `core/utils.py`) and hardcodes inference/GT paths. Fix the import, add argparse for the four directories and `num_class`, leave the metric math **byte-identical**.

```bash
python core/Eval_SCD.py \
  --infer_dir1 <pred>/im1 --infer_dir2 <pred>/im2 \
  --label_dir1 <gt>/label1 --label_dir2 <gt>/label2 \
  --num_class 5
```

### M1 — Surface the correct metrics

In `test_seg_cd.py`:
- Promote `test_scores['SCD_Sek']`, `['Fscd']`, `['SCD_IoU_mean']` from the **unmasked** `test_metric` into the summary block, and add `Score = 0.3·IoU_mean + 0.7·Sek`
- Rename in logs, `test_metrics.txt`, and W&B keys:
  - `test_sek_gt_masked` → `sek_at_gt_mask`
  - `test_sek_pred_masked` → `sek_at_pred_mask`
- Add a one-line comment at each masked accumulator stating it is a diagnostic and why

`train_seg_cd.py:1048` logs `test/epoch_sek` from the unmasked accumulator — verify this and leave it alone if correct.

### M2 — Cross-validate (the gate)

```bash
# 1. dump predictions
python test_seg_cd.py --config config/landsat_cdmamba/cdmamba_seg_cd_balanced.json \
  --save_images results/landsat_baseline_recheck

# 2. run the repaired reference on the same PNGs
python core/Eval_SCD.py --infer_dir1 ... --num_class 5
```

**Pass criterion:** live `Sek`, `F_scd`, `IoU_mean` match `Eval_SCD.py` to within 1e-6.
**If they disagree**, stop and reconcile — a mismatch here invalidates every number in the paper. Likely culprits: label-value handling in `__fast_hist` (masks on GT only), the `+eps` terms in `cm2score` that `Eval_SCD.py` lacks, or the PNG round-trip.

### M3 — Config and label hygiene

```bash
# JSON validity across all configs
for f in config/*/*.json; do python -c "import json,sys;json.load(open('$f'))" \
  || echo "BROKEN: $f"; done
```

Known broken: `exp_E5_final_seed123.json`, `exp_E5_final_seed999.json` — stray unterminated `"` near line 82, inside the `extended_triplet` block. Repair by copying the block from `exp_E5_final_seed42.json` and changing only the seed.

Then add to `SCDDataset.__getitem__`, **before** the clamp:

```python
uniq = set(np.unique(lab1)) | set(np.unique(lab2))
assert uniq <= set(range(num_classes)) | {255}, \
    f"{img_name}: labels {uniq - (set(range(num_classes)) | {255})} outside [0,{num_classes-1}]"
```

Run over the full SECOND and Landsat train splits. SECOND declares `n_classes: 7` while the dataset defaults to `num_classes=6` when the key is absent from `label_transform` — if the assertion fires on class 6, that default has been silently folding it into class 5.

### M4 — Re-baseline

```bash
python test_seg_cd.py --config config/landsat_cdmamba/cdmamba_seg_cd_balanced.json \
  --save_images results/landsat_baseline_standard
```

Record into the table below. Also record per-epoch wall-clock and peak GPU memory — the Phase E/F budget depends on it.

| Run | Dataset | OA | mF1 | mIoU | **Sek** | **F_scd** | Score | Chg F1 | Sek@GT-mask | s/epoch |
|---|---|---|---|---|---|---|---|---|---|---|
| Landsat baseline | Landsat | 88.04 | | 61.42 | **?** | | | 69.21 | 81.18 | |

---

## Decision Gate

Compare the real `Sek` against published Landsat-SCD SOTA, and inspect the gap between `Sek` and `Sek@GT-mask`:

```
Sek competitive?
├─ YES, and Sek@GT-mask ≫ Sek
│     → localization is the bottleneck. Run Phase E (boundary) as written.
├─ YES, and the gap is small
│     → both heads are balanced. Skip Phase E; spend the budget on Phase A + Phase L.
└─ NO (far below SOTA)
      → classification is the bottleneck. Boundary losses are the wrong lever.
        Skip Phase E entirely. Go to Phase A + Phase L, and reconsider the
        problem framing (see roadmap Deferred Work).
```

Record the decision and the number that drove it in this file before proceeding.

---

## Phase B — Baselines
*Longest lead time. Start immediately after M2 passes, in parallel with everything else.*

### B0 — SECOND baseline for CDMamba

There are currently no SECOND numbers.

```bash
python train_seg_cd.py --config config/second_cdmamba/cdmamba_seg_cd_balanced.json \
  --phase train --dataset SECOND --tag B0_second_baseline --seed 42
```

### B1 — SCD baselines

The existing baseline set (BiFA, DARNet, IFNET, DMINet, MSCANet, PaFormer, FC-Siam-*) is **binary change detection** and cannot be compared on Sek or F_scd. Implement and train, in priority order:

1. **Bi-SRNet** — the standard SECOND reference point
2. **SSCD-l** — same family, cheaper
3. **SCanNet** — stronger, more recent
4. **HRSCD-str4** — the classic decomposition baseline
5. ChangeMask *(if time)*
6. TED *(if time)*

Identical splits, preprocessing, augmentation, epoch budget, and evaluation code. Where official numbers exist on the standard split, report **both** reproduced and published, and note discrepancies explicitly.

### B2 — Efficiency profile

Params, FLOPs at 512×512, images/sec, peak memory — for CDMamba and every baseline. This is the evidence for the linear-complexity claim in the core hypothesis. If it does not hold up, soften the claim in the paper rather than defending it.

---

## Phase A — Architecture Ablations
*SECOND, seed 42, single run each. Not 3 seeds — that budget goes to the finals.*

| ID | Change | Flag / knob |
|---|---|---|
| A1 | Change-guided decoder gating off | `use_change_gating: false` |
| A2 | Cross-temporal interaction off | `use_interaction_block: false` |
| A3 | Bottleneck: global attention instead of dilated conv | `ContextBlock2D` swap |
| A4 | Fixed sinusoidal PE instead of coordinate-conditioned | PE module swap |
| A5 | Unidirectional Mamba scan | `ConvMamba` config |
| A6 | Deep supervision on change head at decoder stages 1–2 | new aux loss |
| A7 | **Window-attention encoder at matched params** | backbone swap |

A7 is the core-hypothesis test — prioritize it over A3–A6 if the schedule compresses.

Report Δ mF1, Δ Sek, Δ changed-region IoU, Δ params, Δ FLOPs against B0.

---

## Phase L — Loss Ablations
*SECOND, seed 42. Leave-one-out from the best Phase A config.*

| ID | Removed / changed |
|---|---|
| L1 | `λ_unch = 0` (no unchanged-KL) |
| L2 | `λ_ch = 0` (no changed-diversity) |
| L3 | `λ_cpl = 0` (no coupling) — **open question #2** |
| L4 | `λ_ps = 0` (no pseudo-labeling) |
| L5 | `enable_unch_conf_gating: false` |
| L6 | Transition-weighted CE for rare pairs |
| L7 | Focal instead of CE on imbalanced classes |
| L8 | KL warmup: cosine instead of linear |

L3 is the one most likely to produce a negative result. Report it either way — a component that does not help, honestly reported, costs less credibility than one that quietly does nothing.

---

## Phase E — Boundary Losses
*Conditional. Run only if the Decision Gate points at localization.*

| ID | λ_boundary | λ_morph | λ_hd | Config |
|---|---|---|---|---|
| E1 | 1.0 | 0.0 | 0.0 | `second_cdmamba/exp_E1_boundary_high.json` |
| E2 | 0.5 | 0.5 | 0.0 | `second_cdmamba/exp_E2_morph_boundary.json` |
| E3 | 0.5 | 0.0 | 0.1 | `second_cdmamba/exp_E3_hausdorff.json` |

```bash
python train_seg_cd.py --config config/second_cdmamba/exp_E2_morph_boundary.json \
  --phase train --dataset SECOND --tag E2_morph_boundary --seed 42
```

All warm up from epoch 20 (`enable_boundary_warmup`, `boundary_warmup_epochs: 20`).

**Selection criterion:** best **standard-protocol Sek**, with binary change F1 as tiebreak. Not `Sek@GT-mask` — that metric is blind to exactly what these losses target.

If E3 costs >20% per-epoch time, drop it. Distance transforms are expensive and the budget is tight.

---

## Phase F — Final Runs
*Hard launch deadline: **Oct 30, 2026.** If the config is not settled, freeze what exists and run it.*

Before launching, confirm every `exp_F*` config carries the winning values from Phases A/L/E, and that all six parse as valid JSON.

```bash
# SECOND, 3 seeds
for s in 42 123 999; do
  python train_seg_cd.py --config config/second_cdmamba/exp_F_final_seed$s.json \
    --phase train --dataset SECOND --tag F_final --seed $s
done

# Landsat, 3 seeds
for s in 42 123 999; do
  python train_seg_cd.py --config config/landsat_cdmamba/exp_F_final_seed$s.json \
    --phase train --dataset Landsat --tag F_final --seed $s
done
```

Report mean ± std across seeds for every metric in the paper table.

---

## Results Table

Fill in as runs complete. **All Sek/F_scd columns are standard protocol** — all pixels, both epochs, class 0 = no change.

| Exp | Dataset | Seed | OA | mF1 | mIoU | Sek | F_scd | Score | Chg F1 | Chg IoU | Sek@GT-mask | s/epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M4 re-baseline | Landsat | — | 88.04 | | 61.42 | | | | 69.21 | | 81.18 | |
| B0 | SECOND | 42 | | | | | | | | | | |
| B1 Bi-SRNet | SECOND | 42 | | | | | | | | | | |
| B1 SSCD-l | SECOND | 42 | | | | | | | | | | |
| B1 SCanNet | SECOND | 42 | | | | | | | | | | |
| B1 HRSCD-str4 | SECOND | 42 | | | | | | | | | | |
| A1–A7 | SECOND | 42 | | | | | | | | | | |
| L1–L8 | SECOND | 42 | | | | | | | | | | |
| E1–E3 | SECOND | 42 | | | | | | | | | | |
| F SECOND | SECOND | 42/123/999 | | | | | | | | | | |
| F Landsat | Landsat | 42/123/999 | | | | | | | | | | |

---

## Notes

- W&B project `scd-balanced-mamba`; runs stamped with dataset/tag/seed.
- Ablations use seed 42 only. Three seeds are reserved for Phase F.
- After M1, **never quote a masked metric without its suffix.** The single highest-risk failure mode for this paper is a reviewer noticing that the headline number was computed with the ground-truth change mask supplied to the model.
- If the calendar slips: cut HICD first, then A3–A6, then B1 items 5–6. Protect Phase M, Phase B items 1–4, and Phase F.
