# Roadmap

**Last updated:** August 25, 2026
**Target:** CVPR 2027 — projected deadline Nov 13, 2026 (~11.5 weeks out). Verify against the official CFP when it appears.

---

## Current State (August 2026)

**Implemented and functional:**
- Dual-stream Siamese Mamba encoder-decoder (`CDMamba_seg_cd`) with change and segmentation heads
- `TripletChangeSegLoss` with KL consistency, coupling, diversity, pseudo-labeling terms
- Boundary losses: Laplacian, `ChangeHeadMorphBoundaryLoss` (kornia dilation-erosion ring), `ChangeHeadHausdorffLoss` (MONAI DT)
- Balanced sampler with precomputed change-ratio statistics
- W&B tracking, AMP, gradient accumulation, checkpoint resume
- Binary CD baselines: BiFA, DARNet, IFNET, DMINet, MSCANet, PaFormer, FC-EF, FC-Siam-{Conc,Diff}, ACABFNet, P2V

**Runs completed:**
- Landsat baseline: OA=88.04, F1=69.21, mIoU=61.42, `Sek@GT-mask`=81.18
  *(the last figure was previously reported as "Sek" and compared against SOTA 60.53 — that comparison is invalid; see `mission.md`)*

**Known defects (fix before any further runs):**
1. `test_seg_cd.py` promotes the GT-masked diagnostic to headline status; the standard-protocol Sek from the unmasked `test_metric` is computed but not surfaced in the summary block.
2. `config/second_cdmamba/exp_E5_final_seed123.json` and `exp_E5_final_seed999.json` are **invalid JSON** — stray unterminated `"` around line 82. Two of three final SECOND runs will crash at launch.
3. `data/CDDataset.py` reads `label_transform.get('num_classes', 6)` and clamps labels to `num_classes-1`, but configs declare `n_classes` (SECOND: 7, Landsat: 5). If the key does not reach `label_transform`, SECOND class 6 is silently folded into class 5.
4. `core/Eval_SCD.py` is dead code — hardcoded paths, imports a non-existent `utils.utils`. It is nonetheless the canonical reference; repair it rather than delete it.

**Gaps:** No SCD baselines. No SECOND results at all. No ablation. No paper draft.

---

## Phase 0 — Correctness (Week 1: Aug 25 – Sep 4)
*Nothing downstream is meaningful until this is done. No new training runs start in this phase.*

- [ ] Repair `core/Eval_SCD.py`: fix the import, parameterize the paths, keep the metric math untouched
- [ ] Surface standard-protocol `Sek`, `F_scd`, `IoU_mean`, `Score` from the unmasked `test_metric` in the `test_seg_cd.py` summary; relabel the masked variants `Sek@GT-mask` / `Sek@pred-mask` in logs, W&B keys, and `test_metrics.txt`
- [ ] **Cross-validate:** dump predictions to PNG, run repaired `Eval_SCD.py` on them, confirm live pipeline matches to floating-point tolerance. This is the gate — do not proceed until it passes
- [ ] Fix the two malformed JSON configs; add a CI-style `python -c "import json; json.load(...)"` check over `config/**/*.json`
- [ ] Add an assertion in `SCDDataset.__getitem__` that unique label values ⊆ `[0, n_classes-1] ∪ {255}` **before** clamping; run over the full SECOND and Landsat train splits to confirm no silent folding
- [ ] Re-evaluate the existing Landsat checkpoint under the standard protocol → **the real baseline number**
- [ ] Measure and record per-epoch wall-clock time and GPU memory for CDMamba on both datasets

**Deliverable:** A corrected metrics module, a validated harness, and one honest number.

**Decision gate.** Compare the real Landsat Sek against published Landsat-SCD SOTA:
- **Competitive** → proceed as planned.
- **Far below** → the bottleneck is classification, not boundaries. Skip Phase 3's boundary work, escalate to architecture (Phase 2) and loss (Phase 3a), and consider the reframing under Deferred Work.
- Either way, use the `Sek` vs. `Sek@GT-mask` gap to decide where the error actually lives before choosing a lever.

**GPU budget check.** With measured epoch time, back-solve total GPU-hours for the run list in Phases 1–4. If it does not fit the calendar, cut the HICD dataset first, then reduce ablation runs to SECOND only.

---

## Phase 1 — Benchmarks & Baselines (Weeks 2–4: Sep 5 – Sep 25)
*Runs in parallel with Phase 2; these are the longest-lead items.*

- [ ] **SECOND baseline run** — CDMamba, 3 seeds, standard protocol. There are currently no SECOND numbers at all
- [ ] **Implement and train SCD baselines** — the current baseline set is binary CD and cannot be compared on Sek/F_scd. Minimum viable set:
  - [ ] Bi-SRNet
  - [ ] SSCD-l
  - [ ] SCanNet
  - [ ] HRSCD-str4
  - [ ] ChangeMask *(if time permits)*
  - [ ] TED *(if time permits)*
- [ ] Where official weights or numbers exist on the standard split, cite them **and** reproduce — report both, and note any discrepancy rather than hiding it
- [ ] Confusion matrix analysis: which transitions fail most on each dataset
- [ ] Qualitative visualization sweep on val: side-by-side T1/T2/GT/pred, change mask overlays
- [ ] Profile parameter count, FLOPs, throughput for CDMamba and all baselines (needed for the efficiency claim in the core hypothesis)

**Deliverable:** Benchmark table with real SCD baselines; error analysis document.

**Note on the efficiency claim.** The mission asserts Mamba beats Transformers on long-range context at linear cost. That claim needs a direct measurement — CDMamba vs. a window-attention variant at matched parameter count — or it should be softened in the paper. The FLOPs profile above is the input to that decision.

---

## Phase 2 — Architecture Ablations (Weeks 3–6: Sep 12 – Oct 9)

- [ ] **Change-guided decoder gating** (`use_change_gating`) — on/off, measured on changed-region IoU and unchanged accuracy
- [ ] **Cross-temporal interaction block** (`L_GF_Mamba`, `use_interaction_block`) — on/off
- [ ] **Bottleneck context module** — dilated-conv `ContextBlock2D` vs. global attention
- [ ] **Coordinate-conditioned PE** vs. fixed sinusoidal
- [ ] **Mamba scan direction** — bidirectional vs. unidirectional; group size G sensitivity
- [ ] **Multi-scale deep supervision** on the change head at intermediate decoder stages
- [ ] **Mamba vs. attention at matched params** — the core-hypothesis test

**Deliverable:** Architecture ablation table (Δ mF1, Δ Sek, Δ changed-region IoU, Δ params/FLOPs).

**Scope control:** seven ablations × 2 datasets × 3 seeds does not fit. Run ablations on **SECOND, single seed (42)**; only the final configuration gets 3 seeds on both datasets.

---

## Phase 3 — Loss Refinement (Weeks 5–7: Sep 26 – Oct 16)

### 3a. Component ablation (always run)
- [ ] Leave-one-out over `λ_seg, λ_cd, λ_unch, λ_ch, λ_cpl, λ_ps` — each term removed, everything else fixed
- [ ] Resolve open question #2: is coupling helping or constraining?
- [ ] Transition-weighted loss (inverse-frequency transition matrix) for rare pairs
- [ ] Focal variant for heavily imbalanced classes (water, playground)

### 3b. Boundary losses (conditional on the Phase 0 gate)
*Run only if `Sek@GT-mask` ≫ `Sek`, i.e. localization is confirmed as the bottleneck.*
- [ ] E1: Laplacian weight 0.5 → 1.0
- [ ] E2: morphological ring, `λ_morph=0.5`
- [ ] E3: Hausdorff DT, `λ_hd=0.1`
- [ ] KL warmup schedule: length, linear vs. cosine
- [ ] Boundary warmup epoch sensitivity

**Deliverable:** Loss ablation table; final per-dataset loss configuration.

---

## Phase 4 — Data & Final Runs (Weeks 7–9: Oct 17 – Oct 30)

- [ ] Balanced sampler: `oversample_factor` ∈ [1.0, 4.0], `batch_balance_ratio` ∈ [0.3, 0.7] — coarse sweep only
- [ ] Test-time augmentation: flip + multi-scale ensemble
- [ ] Cross-dataset generalization: train SECOND → zero-shot Landsat (open question #4)
- [ ] **Final runs: SECOND, 3 seeds, best config**
- [ ] **Final runs: Landsat, 3 seeds, best config**
- [ ] HICD *(stretch — cut first if the schedule slips)*

**Deliverable:** Final results table, mean ± std, standard protocol.

**Hard stop:** final runs must be launched by **Oct 30** to leave two weeks for writing. If the best config is not settled by then, freeze whatever is current and run it.

---

## Phase 5 — Paper (Weeks 9–11.5: Oct 31 – Nov 13)

- [ ] Figures: architecture diagram, qualitative comparisons on hard scenes, transition confusion matrices, Mamba activation maps
- [ ] Write: Method → Experiments → Related Work → Intro → Abstract (in that order)
- [ ] **Protocol statement in Experiments** — state the exact Sek/F_scd definitions used and that all numbers are all-pixel. Given how much SCD reporting varies across papers, this is a credibility asset, not boilerplate
- [ ] Error-decomposition analysis using the `Sek` vs. `Sek@GT-mask` gap — this is a genuinely interesting piece of analysis and few SCD papers report it
- [ ] Supplementary: full per-class tables, extra qualitatives, reproduction details
- [ ] Internal review with Dr. Bunyak Ersoy — **schedule for ~Nov 4**, before the abstract deadline
- [ ] Abstract registration **Nov 6**; paper **Nov 13**

---

## Deferred Work
*Considered and explicitly out of scope for this submission.*

- **Graph/region-level refinement.** A GNN over superpixel or connected-component nodes for object-level transition consistency. Rejected for this cycle: it sits downstream of the dense model, the obvious baseline (post-classification comparison) is strong, and the schedule has no room. Revisit for the TGRS extension.
- **Misaligned / cross-sensor bi-temporal pairs.** The setting where dense siamese differencing has nothing to subtract and region-level correspondence becomes load-bearing rather than decorative. This is the more defensible home for a graph method and a stronger novelty story than competing on aligned same-sensor benchmarks — but it is a different paper, not a section of this one.

---

## Key Metrics to Track

| Metric | Definition | Role |
|---|---|---|
| Sek | Standard protocol, all pixels | **Headline** |
| F_scd | Standard protocol, all pixels | **Headline** |
| IoU_mean / Score | Standard protocol | **Headline** |
| mF1, mIoU (seg) | All pixels, both epochs | Model selection |
| Binary change F1 / IoU | Change head vs. GT change | Localization |
| Sek@GT-mask | Changed pixels only, GT mask | Diagnostic — classification under oracle localization |
| Sek@pred-mask | Predicted-changed pixels only | Diagnostic — excludes FN |
| Per-class IoU | Confusion diagonal | Rare-class analysis (water, playground) |
| Params / FLOPs / img·s⁻¹ | Profiler | Efficiency claim |

---

## Open Research Questions
1. Does Mamba's linear-complexity scan provide a real advantage over window-attention Transformers at 512×512? *(Phase 2, matched-parameter test)*
2. Is the coupling loss helping or constraining? *(Phase 3a leave-one-out)*
3. Right granularity for pseudo-labeling — pixel or region level? *(Phase 3a)*
4. Does SECOND → Landsat transfer without fine-tuning? *(Phase 4)*
5. Where does the error actually live — localization or classification? *(Phase 0 gate; determines whether Phase 3b runs at all)*
