# 🐦 BirdCLEF+ 2026 — Clean Solution

A reproducible inference-only solution for the [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026) Kaggle competition, achieving **0.950 Public LB** on 234-class wildlife audio classification.

## What's in this repo

| File | Purpose |
|------|---------|
| [`birdclef-2026-solution.ipynb`](birdclef-2026-solution.ipynb) | Polished, well-documented Jupyter notebook — the main artifact |
| [`submit_kernel.py`](submit_kernel.py) | Utility script to submit a Kaggle kernel version to a code competition via the Kaggle API |
| `LICENSE` | Apache 2.0 |

## Solution at a glance

A 2-model rank-blend ensemble (yukiZ Bird26 × 0.0305 + Karnakbayev Power Optimization × 0.9695) with **taxonomy smoothing** post-processing.

- **Backbone:** Perch v2 ONNX (Google's bioacoustic embedding model)
- **Sequence head:** LightProtoSSM + ResidualSSM
- **Diversity:** Distilled SED (Tucker Arrants) blended in rank space (xSED = 0.60 / 0.40)
- **Post-processing:** Two-level taxonomy smoothing (genus α=0.15, class α=0.05)
- **Final blend strategy:** TAX_SMOOTHING applied AFTER rank fusion

## Public LB result

**0.950** — locked across 9 different submission variants of the same underlying ranking, confirming this is the ceiling of inference-only post-processing on the shared Perch v2 + yukiZ feature extractor.

## How to run

1. Open the notebook on [Kaggle](https://www.kaggle.com/code).
2. Attach the required datasets (listed inside the notebook).
3. Set `MODE = "submit"` and run all cells.

## Acknowledgments

Built on the shoulders of the brilliant BirdCLEF community:

- **Anthony Therrien** — Ensemble framework
- **Yaroslav Kholmirzayev** — v6_0949_replay
- **Derek Sunderekkiz** — Karnakbayev Power Optimization
- **yukiZ (Hideyuki Zushi)** — Bird26.REPRODUCE training
- **F.A.Nina** — EoS series
- **Pilkwang Kim** — EoS+OOF Gated PCEN
- **Karnakbayev Arthur** — Hierarchical taxonomy postprocessing
- **Tucker Arrants** — BC2026 Distilled-SED
- **Google Research** — Perch v2

## License

Apache 2.0. See [LICENSE](LICENSE).
