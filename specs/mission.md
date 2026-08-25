# Mission

## Project Name
**CDMamba** — Multi-Class Semantic Change Detection via Mamba State-Space Models

## Problem Statement
Remote sensing change detection must answer not just *where* change happened, but *what* changed and *into what*. Existing methods either (a) detect binary change without semantic understanding, or (b) perform per-epoch segmentation without jointly reasoning about change. The result is poor localization precision in changed regions and incorrect class assignments — the two most critical failure modes for operational Earth observation.

## Research Goal
Develop a joint semantic segmentation and change detection model that:

1. **Outperforms SCD SOTA** on established multi-class change detection benchmarks (SECOND, Landsat-SCD) on the standard SCD protocol — mIoU, Sek, and F_scd — measured over **all pixels**, not over ground-truth-changed pixels only.
2. **Produces reliable spatial predictions** — high precision/recall on changed pixel localization, not just overall accuracy.
3. **Assigns correct semantic classes** in changed regions — discriminating fine-grained transitions (e.g., low-vegetation → building vs. low-vegetation → impervious surface).

## Target Venue
**Primary: CVPR 2027.**
Projected paper deadline **Nov 13, 2026** (abstract Nov 6, 2026), per deadline trackers. As of July 2026 the CVF had not published an official CVPR 2027 call for papers — `cvpr.thecvf.com/Conferences/2027/CallForPapers` was still 404. **Verify the official date before locking the schedule; the roadmap assumes Nov 13 and has ~2 weeks of slack.**

Secondary: ICCV 2027 (if CVPR misses), IEEE TGRS (journal fallback, no fixed deadline).

## Core Hypothesis
A dual-stream Siamese encoder built on **Mamba state-space blocks** (linear-complexity sequence modeling) can capture long-range spatial context more efficiently than Transformers, while a carefully designed **triplet-style loss** (segmentation + change head + semantic coupling + unchanged-KL consistency) enforces co-learning that prevents the model from treating change detection and segmentation as independent tasks.

---

## Evaluation Protocol (binding)

This section exists because the previous spec's headline result was produced under a non-standard protocol. **All numbers reported in the paper must follow the definitions below.** Any deviation must be labeled as a diagnostic, not a benchmark result.

### The standard SCD protocol
1. The confusion matrix `hist` is accumulated over **T1 and T2 predictions concatenated**, across **all pixels** of the test set.
2. Class 0 means **"no change."** In SECOND and Landsat-SCD ground truth, unchanged pixels carry label 0 in *both* semantic maps; changed pixels carry their land-cover class in each.
3. Metrics:
   - `hist_n0 = hist` with `hist[0][0]` zeroed → `kappa_n0`
   - `c2hist` is the 2×2 collapse of `hist` into unchanged/changed → `IoU_fg`, `IoU_mean`
   - **`Sek = kappa_n0 · exp(IoU_fg) / e`**
   - **`F_scd`** = harmonic mean of `SC_Precision = diag(hist[1:,1:]).sum() / change_pred_sum` and `SC_Recall = diag(hist[1:,1:]).sum() / change_label_sum`
   - **`Score = 0.3·IoU_mean + 0.7·Sek`**

The reference implementation is `core/Eval_SCD.py`. In the live pipeline this corresponds to the **unmasked** `test_metric` accumulator in `test_seg_cd.py` (which already updates on both `p1/y1` and `p2/y2`).

### What is NOT a benchmark number
- **`test_metric_gt_masked`** sets both prediction *and* ground truth to `ignore_index` outside the GT change mask. This hands the model the ground-truth change mask for free (localization is not scored) and causes Sek to degenerate: with row 0 of the histogram empty, `c2hist[0][0] ≈ 0` and `IoU_fg → 1`, so `exp(IoU_fg)/e → 1` and Sek collapses to plain kappa over changed pixels. This is why the old Landsat figure of 81.18 sat 20 points above published SOTA. **Report as `Sek@GT-mask`, a diagnostic for semantic quality under oracle localization. Never as Sek.**
- **`test_metric_pred_masked`** masks by the predicted change mask, which excludes all false negatives from scoring. **Report as `Sek@pred-mask`, a diagnostic. Never as Sek.**

Both diagnostics are genuinely useful — the gap between `Sek` and `Sek@GT-mask` decomposes error into localization vs. classification, and that decomposition belongs in the analysis section. They just cannot appear in the comparison table.

---

## Success Criteria

### Publication-blocking
- **Correctness:** All reported metrics computed under the standard protocol above; live pipeline output verified to match `core/Eval_SCD.py` on the same predictions to within floating-point tolerance.
- **SCD baselines:** At least four semantic change detection baselines (not binary CD models) trained and evaluated under identical splits and preprocessing.
- **Reproducibility:** Config-driven, seed-controlled, checkpoint-resumable; final results as mean ± std over 3 seeds.
- **Ablation:** Each architectural and loss component measured for its contribution, with at least one negative or null result reported honestly.

### Target numbers
These are stated against the standard protocol and are **provisional until the P0 re-evaluation establishes where the model actually stands.** Published SECOND SCD numbers cluster far lower than the old spec's targets implied — Sek in particular is typically in the low tens, not the 60s or 80s.

| Metric | SECOND | Landsat-SCD |
|---|---|---|
| mIoU (SCD, standard) | beat best reproduced baseline | beat best reproduced baseline |
| Sek (standard) | beat best reproduced baseline | beat best reproduced baseline |
| F_scd | beat best reproduced baseline | beat best reproduced baseline |
| Binary change F1 | ≥ best reproduced baseline | ≥ best reproduced baseline |

**Rewrite this table with absolute numbers at the end of Phase 0**, once (a) the true standing of the current checkpoints is known and (b) baselines have been reproduced on the identical split. Targets set against literature numbers from a different split or protocol are not meaningful.

### Diagnostic targets (analysis section, not the comparison table)
- `Sek@GT-mask` high with `Sek` low → localization is the bottleneck; boundary work is justified.
- `Sek@GT-mask` and `Sek` both low → classification is the bottleneck; boundary work is not the right lever.

---

## Non-Goals
- Real-time inference (latency is not a primary constraint)
- Video change detection (focus is bi-temporal satellite imagery)
- Unsupervised or weakly supervised change detection
- Panoptic segmentation or instance-level outputs
- Graph-based post-processing / region-level refinement (evaluated and deferred; see `roadmap.md` Deferred Work)
