# BirdCLEF+ 2026 — Strategic Plan for Day 2 (UTC May 28)

## Current best: Y0 = 0.950 (Anthony's ensemble, locked in on LB)

## Ready-to-submit kernels (already ran successfully, just queued for submission):

| Variant | Source | Expected | Evidence |
|---------|--------|----------|----------|
| **Y2** | Pilkwang EoS8+BirdNET+PCEN | 0.951 likely | Research +0.001 |
| **Y5** | Karnakbayev hierarchical tax | 0.950–0.951 | Karnakbayev's own follow-up |
| **Y1** | Anthony m2-heavy fork | 0.949–0.951 | weight-tweak speculation |
| **Y3** | Raunak V9 (newest) | unknown | 67 votes fast = community signal |
| **Y4** | Yaroslav v221 tax | 0.949–0.950 | latest Yaroslav |

## Day 2 strategic submission order (5 slots, learning between each):

1. **Submit Y2 first** (highest evidence) → wait for score
2. Based on Y2:
   - If Y2 ≥ 0.951 → submit Y5 (combine winning signals) 
   - If Y2 = 0.950 → submit Y5 (try different angle)
   - If Y2 < 0.950 → revert thinking, submit Y3 (try Raunak)
3. Submit best remaining candidate
4. Reserve slot for a CUSTOM combination based on what we learn
5. Reserve slot for safety / quick win

## Custom combinations to build if needed (Day 2 build queue):

- **Y6 = Y0 + Y2's BirdNET postproc grafted** (clean composition)
- **Y7 = Y0 + custom MIRROR_PAIRS tightening** (test sonotype grouping)
- **Y8 = Y0 with Model_22 weight sweep** (0.025, 0.035, 0.04, 0.05)

## What to AVOID

- ❌ Submitting 5 variants in a burst without learning → wasted slots
- ❌ Custom postproc on islet base (proven neutral)
- ❌ Pre-computed CSV ensembles (row_ids don't match hidden test)
- ❌ Over-aggressive parameter changes (vA-E lesson: -0.006)

## Top public LB target: 0.963 (Yannan Chen)
Gap from us: 0.013 = realistic 6–8 iterations of +0.001-0.002 each
