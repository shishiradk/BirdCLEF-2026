# LEARNINGS — How to Win a Kaggle Competition

Distilled from BirdCLEF+ 2026 (final score 0.950, top public 0.966, top private 0.967).
Use this checklist for any future Kaggle competition with cash prizes.

---

## 🟢 DO — The 25 things that win

### Strategy & timing
1. **Start Day 1 of the competition, NOT in the last 2 weeks.** Top teams have 3+ months. We started Day 78 and capped at the public-ceiling.
2. **Train custom models from week 1.** Inference tweaks cap at whatever ceiling the existing public ensembles already produce. Post-processing alone cannot beat the model.
3. **Always have a safe baseline submission on the LB.** Y0 = 0.950 protected us from all 5 catastrophic regressions during the rest of the competition.
4. **Daily submission discipline: 1–2 carefully selected per day**, not 5-shot bursts. Learn from each score before deciding the next.
5. **Read the metric's mathematical properties FIRST.** Macro-AUC is invariant under monotone per-column transforms — knowing this would have saved Z9/Z10/Z11/Z12 slots.
6. **Trust public LB over local CV** in active competitions. Multiple top finishers explicitly said "take LB as CV."
7. **Manually select your 2 final submissions on Kaggle.** Don't trust auto-pick.

### Training pipeline (audio competitions)
8. **Use 20-second SED chunks**, NOT 5-second. This single change gives +0.025–0.030 (literally the biggest unlock in audio).
9. **Use the BirdCLEF-2021-4th-place SED head** (clipwise + frame attention pooling). Beats plain GeM consistently.
10. **CrossEntropy loss on multi-hot targets** (not BCE/Focal). Set secondary_labels=1.
11. **AdamW(lr=5e-4, wd=1e-4) + CosineAnnealingWarmRestarts(T_0=5).** 15 epochs stage 1.
12. **MixUp p=0.5 with max-of-labels combining** (not λ-weighted) on absmax-normalized raw audio.
13. **Mel params for birds:** sr=32000, n_mels=192–224, n_fft=2048–4096, hop=768–1252, fmin=20, fmax=15000–16000, log(mel+1e-6), repeat to 3 channels.
14. **Run Silero VAD to remove human speech segments** from train_audio.
15. **Manually screen classes with <30 samples** — they will dominate noise otherwise.

### Pseudo-labeling (the +0.058 unlock)
16. **Multi-iterative Noisy Student** — 4 rounds max (5th plateaus).
17. **MixUp ratio = 1.0** between focal train and pseudo-labeled soundscapes (every batch sample mixed, blend weight 0.5).
18. **Power transform pseudo-labels** between rounds: `prob ^ (1/0.55–1/0.65)` to denoise.
19. **WeightedRandomSampler** with weights = sum of max-label probs per soundscape (clean soundscapes oversampled, near-empty down-weighted).
20. **Stochastic Depth drop_path=0.15** ONLY in pseudo-label stages, not stage 1.

### Ensembling
21. **6–13 model ensemble across backbone families** AND across pseudo-label iterations (not just the last one).
22. **Backbone diversity:** tf_efficientnet_b0/b3/b4_ns + regnety_008/016 + eca_nfnet_l0 + tf_efficientnetv2_s/b3.
23. **Separate Amphibia/Insecta specialist** trained on 17k extra XenoCanto. Column-REPLACE (not blend) into those species columns. +0.002–0.003.
24. **OpenVINO + ThreadPoolExecutor + mel cache** to fit the full ensemble in Kaggle's 90-min CPU budget.
25. **Framewise overlap-averaging + [0.1, 0.2, 0.4, 0.2, 0.1] smoothing + delta-shift TTA** AFTER ensembling, in calibrated probability space.

---

## 🔴 DON'T — The 22 traps that kill scores

### Misuse of time
1. **Don't start with inference tweaks.** They cap at the public ceiling. We wasted Day 78–82 on them.
2. **Don't try to fix a regressed base notebook.** Revert to known-good. We had vA–E lose -0.006 chasing a broken patch.
3. **Don't burst-submit all candidates at once.** Each submission is a learning opportunity — burn one, observe, then decide next.
4. **Don't burn slots in the last 6 hours on speculative ideas.** Run them earlier so you have time to course-correct.

### Trust traps
5. **Don't trust notebook title claims.** `alrickh/bc26-exp070-public0952-may22` claimed 0.952 — we scored 0.899 (-0.051 catastrophic). Public claims often assume specific environments.
6. **Don't trust pre-computed CSV ensembles.** Old test-set row_ids won't match the current hidden test. Both Y2 and Z6 wasted slots on this.
7. **Don't trust "Improved" / "Enhanced" / "Better Blend" titles.** Most converge to the same public ceiling.

### Math/metric traps
8. **Don't apply monotone postprocessing and expect macro-AUC to change.** It's mathematically guaranteed to be unchanged. Z10/Z11/Z12 = exactly 0.950 by proof.
9. **Don't apply heavy cross-row smoothing.** It leaks rank between positives and dense negatives. Z9 (-0.001), Z13 (-0.002).
10. **Don't combine mixup + cutmix + sumix.** BNBU Anpeng Yuan: mixup alone > all three combined.
11. **Don't use ASL/Focal/BCE losses with SED.** Stick with plain CrossEntropy. All three regressed in top-team tests.

### Architecture traps
12. **Don't change architectures within the same submission family without testing first.** W2 (Two-Pass SSM, -0.020) and W3 (Cliff Gate, -0.008) regressed because they aren't Anthony-lineage.
13. **Don't add untested models to working ensembles.** Z5 broke because Model_74's xSED config was incomplete.
14. **Don't use chunk durations >5s on CLIP-LEVEL models** (different from SED). Confirmed regression.

### Data traps
15. **Don't chase "ghost species."** OpPrime confirmed Perch+ProtoSSM+MLP+ResSSM already captures 27/28 ghosts.
16. **Don't scrape external XC for TARGET species.** Hosts explicitly say all relevant files are already in train_audio.
17. **Don't ignore the new competition data** — `train_soundscapes_labels.csv` was the unique +0.010–0.020 lever this year.

### Tooling traps
18. **Don't push kaggle.json to GitHub.** Add to .gitignore on Day 1.
19. **Don't forget UTF-8 settings** for Kaggle CLI on Windows: `PYTHONIOENCODING=utf-8`, `PYTHONUTF8=1`, no emojis in kernel titles.
20. **Don't assume dataset mount paths** — they're `/kaggle/input/datasets/<owner>/<slug>/`, not just `/kaggle/input/<slug>/`. Always do a `glob` probe first.
21. **Don't run the same model with different post-processing tweaks** repeatedly — each costs a submission slot and they'll all return the same score per macro-AUC's monotone-invariance.
22. **Don't keep all of your `out_*/` and `fetch_*/` folders in the repo.** Clean them out before pushing to GitHub.

---

## 📋 Pre-competition checklist (use Week 1, every time)

- [ ] Read the competition Overview, Data, and Rules cover-to-cover
- [ ] Check the metric — write down its mathematical invariances
- [ ] Read top-voted public notebooks from past competitions (same domain)
- [ ] Set up Kaggle GPU quota tracking (30h/week — it's the bottleneck)
- [ ] Set up Colab Pro / RunPod for overflow if budget allows
- [ ] Initialize GitHub repo with .gitignore including kaggle.json, *.log, out_*, fetch_*
- [ ] Set up local kaggle CLI + auth (kaggle.json at ~/.kaggle/)
- [ ] Bookmark Discussions tab; check daily for tips
- [ ] Identify the 2 metrics: public LB and (if exists) local CV
- [ ] Plan a daily submission schedule (1–2 per day, save 2 for the last week)

---

## 📅 Mid-competition checklist (each week)

- [ ] Daily: 1 baseline + 1 experiment submission (max), check Discussions for new tricks
- [ ] Weekly: archive submission scores to `progression.md`; review what worked
- [ ] Weekly: check Kaggle GPU quota usage; queue jobs back-to-back
- [ ] Train next-iteration models WHILE old ones run inference (parallelize compute)
- [ ] Backup best submission CSV locally and to GitHub
- [ ] Verify the public-LB Y0 (safe submission) is STILL selected as final

---

## 🏁 Final-week checklist

- [ ] STOP training new models. Lock the ensemble.
- [ ] Run a CPU-time stress test on the final inference notebook (must fit competition limit).
- [ ] Submit your 2 best ensembles AND save 1 slot per day for emergencies.
- [ ] Manually select 2 final submissions on Kaggle: (A) best public LB, (B) most diverse ensemble for shake-up insurance.
- [ ] Write a one-pager solution writeup — many competitions have a "Best Solution" prize for clarity.
- [ ] Commit everything to GitHub with a clean README + POSTMORTEM.
- [ ] After deadline: write a `learnings.md` like this one.

---

## 💰 Picking the next competition (where to spend effort)

**Maximize: (prize size) ÷ (#teams) ÷ (skill mismatch)**

### Heuristics
- **<2000 teams** = beatable competition (BirdCLEF had 4155 — hard)
- **3+ months until deadline** = enough time to train custom models
- **Research category** = often less crowded than Featured
- **Code competitions** = inference-only execution, easier to debug
- **Niche domain that matches your skills** = compounding edge

### Top 3 from this analysis (June 2026)
1. **`neurogolf-2026`** — $50k, 1523 teams, 6 weeks. Best balance.
2. **`arc-prize-2026-paper-track`** — $450k, 61 teams (!!), 5 months. Requires research paper.
3. **`rogii-wellbore-geology-prediction`** — $50k, 2032 teams, 10 weeks. More time.

### Avoid
- ARC-AGI-2/3 ($700K–$850K but inherently hard reasoning)
- Hull tactical market prediction (finance ML, 3677 teams)
- Anything closing in <3 weeks with no existing baseline

---

## 🎯 The single most important meta-lesson

**Inference tweaks can NEVER beat the model's ceiling. Training a different model CAN.**

This competition's gap (0.950 → 0.967 = 0.017) was almost entirely:
- 20s SED chunks: +0.025–0.030
- Pseudo-label round 1: +0.026
- New 2026 train_soundscapes_labels: +0.010–0.020

Every one of those = TRAINING work, not POSTPROCESSING work.

**Next competition: start training on Day 1. Build inference infra on Day 30. Polish in the last 2 weeks.**

The path to a prize is custom-trained models, not forked public notebooks.
