# BirdCLEF+ 2026 — Solution Repository

Public LB **0.950** (Y0 baseline) on the [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026) Kaggle competition.
234-class wildlife audio classification (Macro-averaged ROC-AUC).

## Best score lineage

| Submission | Score | Notes |
|------------|-------|-------|
| `birdclef-2026-public-lb-with-onnx-perch-sequence.ipynb` | 0.949 | Reference baseline |
| `birdclef-2026-public-lb-BACKUP.ipynb` | 0.948 | Safe backup |
| `birdclef-2026-163353.ipynb` | 0.948 → 0.950 (with patches) | Working notebook |

## Files

### Main notebooks
- `birdclef-2026-163353.ipynb` — Patched main working notebook with TTA + MIRROR_PAIRS expansion + YAMNet rescue
- `birdclef-2026-public-lb-with-onnx-perch-sequence.ipynb` — Pure 0.949 baseline
- `birdclef-2026-public-lb-BACKUP.ipynb` — Untouched 0.948 backup

### Build scripts (custom variants)
- `build_z4.py` — Z4: 3-way ensemble (M22 + M51 + M73 Yaroslav v6)
- `build_z9_novel.py` — **Z9: NOVEL entropy-adaptive temporal smoothing** (our innovation)
- `build_z10_novel.py` — **Z10: NOVEL per-class adaptive temperature scaling** (our innovation)

### Tooling
- `submit_kernel.py` — Submit a Kaggle kernel version to a competition via API

### Variants (notebook + metadata)
- `variant_z4_3way_fixed/` — 3-way ensemble runs (confirmed 0.950)
- `variant_z9_adaptive_smooth/` — Adaptive entropy-bandwidth smoothing
- `variant_z10_rebalance/` — Per-class adaptive temperature

### Documentation
- `STRATEGY.md` — Strategic notes and lessons learned

## Key learnings

The **0.950 public LB ceiling** is firmly confirmed across 11+ different public-notebook lineages including Anthony, Karnakbayev, Nina, Meenal, Pilkwang variants. Path to higher requires custom-trained models (top public LB = 0.966).

## Setup

```bash
pip install kaggle
# Place kaggle.json at ~/.kaggle/kaggle.json
python submit_kernel.py <username>/<kernel-slug> "submission description"
```

## License

This repository contains derivative work based on public Kaggle notebooks. Original notebooks credited inline.
