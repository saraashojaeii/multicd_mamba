# Mission

## Project Name
**CDMamba** — Multi-Class Semantic Change Detection via Mamba State-Space Models

## Problem Statement
Remote sensing change detection must answer not just *where* change happened, but *what* changed and *into what*. Existing methods either (a) detect binary change without semantic understanding, or (b) perform per-epoch segmentation without jointly reasoning about change. The result is poor localization precision in changed regions and incorrect class assignments — the two most critical failure modes for operational Earth observation.

## Research Goal
Develop a joint semantic segmentation and change detection model that:

1. **Outperforms SOTA** on established multi-class change detection benchmarks (SECOND, Landsat, HICD) on mIoU, mF1, and semantic change detection score (SCD F1 / S_ek).
2. **Produces reliable spatial predictions** — high precision/recall on changed pixel localization, not just overall accuracy.
3. **Assigns correct semantic classes** in changed regions — discriminating fine-grained transitions (e.g., low-vegetation → building vs. low-vegetation → impervious surface).

## Target Venue
Primary: **WACV 2026 / CVPR 2026**
Secondary: ICCV 2025 (if timeline permits), IEEE TGRS (journal fallback)

## Core Hypothesis
A dual-stream Siamese encoder built on **Mamba state-space blocks** (linear-complexity sequence modeling) can capture long-range spatial context more efficiently than Transformers, while a carefully designed **triplet-style loss** (segmentation + change head + semantic coupling + unchanged-KL consistency) enforces co-learning that prevents the model from treating change detection and segmentation as independent tasks.

## Success Criteria
- **SECOND dataset:** mF1 > 55%, SCD F1 > 0.65 (exceeds published SOTA as of 2024)
- **Landsat dataset:** mIoU > 60%, change F1 > 0.80
- **Changed-region segmentation:** IoU on GT-changed pixels > 50% (reliable localization, not inflated by unchanged majority)
- **Reproducibility:** Config-driven, seed-controlled, checkpoint-resumable training; results reproducible within ±0.5% across 3 seeds
- **Publication:** Accepted at WACV 2026 or CVPR 2026 with ablation study validating each architectural and loss component

## Non-Goals
- Real-time inference (latency is not a primary constraint)
- Video change detection (focus is bi-temporal satellite imagery)
- Unsupervised or weakly supervised change detection
- Panoptic segmentation or instance-level outputs
