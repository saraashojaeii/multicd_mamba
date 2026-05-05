# Roadmap

## Current State (as of May 2026)
The core architecture (`CDMamba_seg_cd`) and training pipeline are implemented and functional:
- Dual-stream Siamese Mamba encoder-decoder with change and segmentation heads
- Comprehensive loss suite (TripletChangeSegLoss with KL consistency, coupling, diversity terms)
- Balanced sampler for class-imbalanced datasets (SECOND, Landsat, HICD)
- W&B experiment tracking, mixed precision, gradient accumulation, checkpoint resume
- Baseline comparisons: BiFA, DARNet, IFNET, DMINet, MSCANet, PaFormer

**Gap:** No published benchmark results; no ablation study; no paper draft.

---

## Phase 1 — Establish Baseline & Diagnose Failures
*Goal: Know exactly where the model succeeds and fails before touching architecture.*

- [ ] Run full training on SECOND (3 seeds, 200 epochs) and collect: mF1, mIoU, SCD F1, S_ek, per-class IoU, changed-region IoU
- [ ] Run all baseline comparisons (BiFA, DARNet, IFNET, etc.) under identical conditions
- [ ] Build confusion matrix analysis — identify the class transitions the model gets wrong most
- [ ] Visualize predictions on val set: segment-level overlays, change mask overlays, side-by-side with GT
- [ ] Profile GPU memory and throughput (FLOPs, parameter count, training time per epoch)
- [ ] Repeat on Landsat and HICD datasets

**Deliverable:** Benchmark table + error analysis document

---

## Phase 2 — Architecture Improvements for Changed-Region Reliability
*Goal: Improve localization precision and recall in changed areas.*

- [ ] **Change-Guided Decoder Attention:** Enable and tune `use_change_gating` — verify it improves changed-region IoU without hurting unchanged accuracy
- [ ] **Cross-Temporal Interaction Block:** Validate the dual-stream interaction (L_GF_Mamba) — ablate with/without to confirm contribution
- [ ] **Bottleneck Context Module:** Compare dilated-conv context vs. transformer-style global attention at the bottleneck
- [ ] **Multi-Scale Change Supervision:** Add auxiliary change supervision at intermediate decoder stages (deep supervision)
- [ ] **Coordinate-Conditioned PE:** Verify it outperforms fixed sinusoidal PE on high-resolution inputs
- [ ] **Mamba Block Tuning:** Experiment with bidirectional vs. unidirectional scan; group size G sensitivity

**Deliverable:** Ablation table for architecture components (±contribution to mF1 and changed-IoU)

---

## Phase 3 — Loss Function Refinement
*Goal: Correct class assignment in changed regions; reduce false positives.*

- [ ] **Hyperparameter sweep on TripletChangeSegLoss:** λ_seg, λ_cd, λ_unch, λ_ch, λ_cpl, λ_pseudo — use W&B sweep
- [ ] **KL Warmup Schedule:** Tune warmup length for unchanged-KL term; test linear vs. cosine ramp
- [ ] **Transition-Weighted Loss:** Enable inverse-frequency transition matrix weighting for rare class pairs (e.g., water→building)
- [ ] **Focal Loss Variant:** Replace CE with focal loss for heavily imbalanced classes (playground, water)
- [ ] **Pseudo-Label Confidence Threshold:** Tune the unchanged pixel confidence gate (max_prob threshold)
- [ ] **Consistency Loss Strength:** Tune coupling between semantic change and change head predictions
- [ ] **Ablation:** Report model performance with each loss component removed

**Deliverable:** Loss ablation table; final best loss configuration per dataset

---

## Phase 4 — Data & Sampling Improvements
*Goal: Ensure the model sees representative changed/rare-class examples.*

- [ ] **Balanced Sampler Tuning:** Sweep `oversample_factor` (1.0–4.0) and `batch_balance_ratio` (0.3–0.7)
- [ ] **Augmentation Expansion:** Add: elastic distortion, Gaussian noise, random erasing, multi-scale random crop
- [ ] **Hard Negative Mining:** Identify and oversample images where the model historically fails
- [ ] **Test-Time Augmentation (TTA):** Horizontal/vertical flip + multi-scale ensemble at inference
- [ ] **Cross-Dataset Generalization:** Train on SECOND, evaluate zero-shot on Landsat; quantify domain gap

**Deliverable:** Sampling and augmentation ablation; TTA results table

---

## Phase 5 — SOTA Comparison & Paper Preparation
*Goal: Demonstrate clear improvement over SOTA; write publication-ready paper.*

- [ ] Reproduce published SOTA numbers for BiFA, PaFormer, DARNet on SECOND and Landsat (same splits, same preprocessing)
- [ ] Final model training: 3 seeds, best config, report mean ± std
- [ ] Produce publication-quality figures:
  - Qualitative comparisons (CDMamba vs. SOTA on difficult scenes)
  - Feature map visualizations (Mamba token activations)
  - Confusion matrices for semantic transitions
  - Architecture diagram (encoder, decoder, loss flow)
- [ ] Write paper: Introduction, Related Work, Method, Experiments, Conclusion
- [ ] Internal review + rebuttal preparation
- [ ] Submit to WACV 2026 (submission deadline typically August)

**Deliverable:** Camera-ready paper submission

---

## Phase 6 — Post-Submission
- [ ] Release code and pretrained weights (model zoo)
- [ ] Write reproducibility README with exact commands for each benchmark
- [ ] Submit journal extension to IEEE TGRS (expanded experiments, additional datasets)
- [ ] Explore efficient inference: quantization, ONNX export, TensorRT

---

## Key Metrics to Track (per experiment)

| Metric | Meaning | Target (SECOND) |
|---|---|---|
| mF1 (seg) | Mean F1 across semantic classes | > 55% |
| mIoU (seg) | Mean IoU across semantic classes | > 50% |
| SCD F1 / S_ek | Semantic change detection composite | > 0.65 |
| Change F1 | Binary change detection F1 | > 0.80 |
| Changed-region IoU | IoU computed only on GT changed pixels | > 50% |
| Per-class IoU (building) | Most common change target class | > 60% |
| Per-class IoU (water) | Rare class | > 45% |

---

## Open Research Questions
1. Does Mamba's linear-complexity scan provide a real advantage over window-attention Transformers at 512×512 resolution?
2. Is the coupling loss helping or hurting — does aligning change head with semantic change improve or constrain both?
3. What is the right granularity for pseudo-labeling — pixel-level or region-level unchanged supervision?
4. Can a single model trained on SECOND generalize to Landsat without fine-tuning (cross-resolution, cross-sensor)?
