# BirdCLEF+ 2026 — Post-Mortem & Forward Recipe

## Final standing
**Public LB: 0.950** (Y0 = Anthony's ensemble fork)
**Top public LB: 0.966** (Nikita Babych, custom-trained HGNetV2 + pseudo-labeling)
**Gap: 0.016**

## Why we plateaued at 0.950 (high-confidence root cause)

All 11 of our 0.950 submissions descend from the SAME upstream pipeline:
- Perch v2 ONNX backbone
- yukiZ Bird26.REPRODUCE pretrained heads
- Same Pantanal-fine-tuned ProtoSSM/ResSSM

Only the post-hoc transforms differed (TAX_SMOOTHING, PCEN, hierarchical taxonomy, MAX-blends, top-K preserve, etc.).

### The mathematical proof

Macro-AUC = mean over 234 classes of AUC_c.
AUC_c = Pr(score(positive_i) > score(negative_j)) — depends ONLY on within-column rank order.

**Any monotone per-column transform g_c(s) preserves every pairwise inequality** → AUC unchanged → score pinned at 0.950 exactly.

| Variant | Transform | Class | AUC effect |
|---------|-----------|-------|------------|
| Z10 (temperature) | sigmoid(logit/T) | monotone | 0 → 0.950 |
| Z11 (5% MAX-blend top-1) | y = max(0.95s + 0.05·t, s) | near-monotone | 0 → 0.950 |
| Z12 (top-K preserve + 0.95× below median) | piecewise linear, both slopes > 0 | monotone | 0 → 0.950 |
| Z13 (intra-file triangular avg) | row convolution | **NOT monotone** | -0.002 → 0.948 |

Only non-monotone ops (cross-row Z13) can move AUC — but they regress because they leak rank between positives and dense negatives.

**Conclusion:** Postprocessing of a saturated ensemble output is mathematically incapable of beating the baseline AUC. Movement requires a different ranking source — i.e., a different model.

## Concrete techniques top teams used (from 0.950 → 0.967)

| Technique | Lift | Notes |
|-----------|------|-------|
| **20-second SED chunks** (vs 5s) | **+0.025–0.030** | Single biggest architecture choice. 5s → 20s in BirdCLEF 2025 = 0.842 → 0.872. |
| **Pseudo-label iter 1** on train_soundscapes | **+0.026** | MixUp ratio=1.0, power transform prob^(1/0.60). |
| Pseudo-label iters 2–4 | +0.015 to +0.032 | Diminishing returns by iter 5. |
| **NEW 2026 `train_soundscapes_labels.csv`** | **+0.010–0.020** | Expert-annotated subset of test-distribution audio. Unique 2026 lever. |
| Backbone diversity (b0/b3/b4_ns + nfnet_l0 + regnety + efficientnetv2_s) | +0.005–0.010 | 6–13 model ensemble blends across iterations. |
| Separate Amphibia/Insecta model + 17k extra XC | +0.002–0.003 | Column-replace only, don't blend. |
| Stochastic Depth (drop_path=0.15) in PL stages | +0.005/model | Don't use in stage 1. |
| Framewise overlap-averaging TTA | +0.002–0.003 | Sliding-window across neighboring 20s chunks. |
| [0.1, 0.2, 0.4, 0.2, 0.1] smoothing kernel | +0.001–0.002 | Apply in calibrated probability space AFTER ensemble. |
| Delta-shift TTA (offsets 0/2.5/5/7.5s) | +0.001–0.002 | |
| Silero VAD to remove human speech | small | Cleans train_audio. |

## 8-Week Action Recipe (top-30 target, ~$50k prize zone)

**Total budget: 160–220 hours over 8 weeks**

### Week 1 — Foundation
**Goal:** Lock in 20-second SED pipeline + single-model baseline.
- Fork Kaggle GPU notebook, build dataset class: 20s chunks, sr=32000, n_mels=224, n_fft=2048, hop=1252, fmin=20, fmax=16000, log(mel+1e-6), 3×224×512 mel.
- BirdCLEF-2021-4th SED head (clipwise + framewise attention pooling) on `tf_efficientnet_b0.ns_jft_in1k`.
- Loss: 0.5·CE_clip + 0.5·CE_frame on multi-hot targets with secondary_labels=1.
- Train: 15 epochs, AdamW(lr=5e-4, wd=1e-4), CosineAnnealingWarmRestarts T_0=5, BS=64, MixUp p=0.5 with max-of-labels combining.
- Silero VAD pass on train_audio to drop human speech.
- **Expected:** single-model 0.92–0.94 (lower abs but different ranking → higher ceiling).

### Week 2 — Train-soundscapes-labels (the NEW 2026 unlock)
**Goal:** Wire in the expert-annotated subset (+0.010–0.020).
- Parse `train_soundscapes_labels.csv` → 5s-segment multi-hot.
- Second dataloader; mix 1:1 with train_audio for last 5 epochs of stage 1.
- Cache mel spectrograms for all train_soundscapes once on disk.
- **Expected:** 0.94–0.95 matching shared-backbone ceiling but with DIFFERENT ranking.

### Week 3 — Noisy Student Iteration 1
**Goal:** First pseudo-label round (+0.026).
- Generate PLs on unlabeled train_soundscapes; framewise SED + clipwise max-prob.
- Power transform prob^(1/0.60); WeightedRandomSampler weights = sum max-prob per soundscape.
- Train iter-1 student: `tf_efficientnet_b0` from scratch, 25 epochs, drop_path=0.15, MixUp PL ratio=1.0, blending weight 0.5.
- **Expected:** +0.015 to +0.025 over week 2.

### Week 4 — Iteration 2 + backbone diversity
- Re-generate PLs from iter-1 (averaged with iter-0 at 0.7/0.3); power transform prob^(1/0.58).
- Train `tf_efficientnet_b3.ns_jft_in1k` as iter-2.
- Add framewise overlap-averaging TTA (sliding window across neighboring 20s chunks).
- Submit iter-1 + iter-2 ensemble mean.

### Week 5 — Iteration 3 + 4-model ensemble (top-30 entry)
- Power transform prob^(1/0.55).
- Train `eca_nfnet_l0` + `tf_efficientnetv2_s.in21k_ft_in1k` as iter-3.
- 4-model ensemble: b0 (iter-2) + b3 (iter-2) + nfnet_l0 (iter-3) + efv2_s (iter-3).
- Apply [0.1, 0.2, 0.4, 0.2, 0.1] smoothing across adjacent 5s rows in PROBABILITY space after ensemble.
- **Target LB: 0.955–0.960** ← Crosses the 0.950 plateau.

### Week 6 — Iteration 4 + Amphibia/Insecta specialist
- Train iter-4 students: `tf_efficientnet_b4.ns_jft_in1k` + `regnety_016`, drop_path=0.15.
- Scrape ~17k extra XenoCanto for Amphibia + Insecta (700 species, 1+ per species).
- Train standalone `efficientnet_b0`, BS=128, 40 epochs, only Amphibia/Insecta classes.
- **Column-replace** (not blend) Amphibia/Insecta predictions into ensemble submission.
- Add delta-shift TTA at offsets {0, 2.5, 5, 7.5}s.
- **Target LB: 0.960–0.964.**

### Week 7 — CPU runtime optimization
- Convert all checkpoints to OpenVINO IR (FP16).
- ThreadPoolExecutor: read audio → compute mel ONCE → fan out to all OpenVINO sessions in parallel.
- Cache mels per test file in /tmp.
- **Verify end-to-end submission runtime <85min on Kaggle CPU.**

### Week 8 — Ensemble expansion + final selection
- Grow to 6–13 model ensemble blending ACROSS iterations (not just last).
- Sweep ensemble weights on 10% held-out train_soundscapes_labels.
- Two final submissions: (A) best-AUC 8-model ensemble, (B) most-diverse 13-model for shake-up insurance.
- **Target public LB: 0.964–0.967. Private LB realistic range: 0.957–0.967 (top-30 floor → top-3 ceiling).**

## Minimum Viable Path (if budget collapses to ~60–80 hours)

Execute only weeks 1, 2, 3, 5:
1. 20s SED on efficientnet_b0 with train_audio (week 1)
2. Add train_soundscapes_labels 1:1 mix (week 2)
3. ONE pseudo-label iteration with PL mixup 1.0 + drop_path 0.15 (week 3)
4. Train ONE more backbone (eca_nfnet_l0) on iter-1 PLs; 2-model ensemble + smoothing kernel (compressed week 5)

**The two non-negotiables: (a) 20s chunks, (b) at least one PL round.**
**Expected: 0.955–0.960 (top-50 → top-30 floor).**

## Optimistic Path (if all 8 weeks execute clean + stretch)

Add weeks 9–10:
- Week 9: train HGNetV2 backbone on iter-4 PLs + train_soundscapes_labels (Babych's architecture choice).
- Week 10: 2-pass inference with class co-occurrence priors derived from train_soundscapes_labels — **this is the ONLY postproc that can move AUC** because it uses cross-column information.
- **Target public LB: 0.967–0.970. Private LB: top-3 territory.**

## Critical bottleneck
**Kaggle GPU quota = 30h/week.** Iterations 1–4 across 6 backbones ≈ 80–120 GPU-hours total → must queue back-to-back across 4–5 weeks. Colab Pro overflow recommended for parallelism.

## Why this competition's outcome was structurally locked
Started inference tweaks on Day 78. Top teams started training Day 1. Custom models take 4–8 weeks to train; the 0.950 → 0.967 gap = ~4–6 GPU-weeks of training work that we couldn't compress.

**The path to a prize is start training on Day 1, not Day 78.**
