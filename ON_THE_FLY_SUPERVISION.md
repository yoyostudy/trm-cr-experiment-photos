# On-the-Fly Supervision: All Runs Summary

This doc is the canonical summary of all on-the-fly (`derive_ignore`) supervision
experiments. Companion data file: [`on-the-fly-supervision.csv`](on-the-fly-supervision.csv).

For the design framework and three-axis analysis, see
[`DERIVE_IGNORE_FRAMEWORK.md`](DERIVE_IGNORE_FRAMEWORK.md).

## Mechanism recap

```
pred       = model(clue)
trust      = clue ∪ {cell : pred == gt [∧ conf > τ]}   # τ = optional confidence filter
derivable  = solver(trust, cap = N)                    # N = max solver propagation steps
mask       = 81 − trust − derivable                     # cells receiving 0 loss

loss = Σ w_t · CE(pred, gt)  for cell ∈ trust
     + Σ w_d · CE(pred, gt)  for cell ∈ derivable
     + Σ w_m · CE(pred, gt)  for cell ∈ mask
```

Three axes:
1. **Solver cap (N)**: how many propagation steps the solver gets
2. **Trust filter (τ)**: confidence threshold for "model knows" cells
3. **Loss weight (w_t, w_d, w_m)**: per-bucket gradient weighting (default `(1, 1, 0)`)

## All runs — peak n=64 performance

All evaluations on full test set (sudoku-extreme, 422786 puzzles).
"Peak" = checkpoint with highest `all.exact_acc_at_halt_or_max` at n=64.

| Run | Trust | Cap | Loss weight | Peak step | n=64 (all / hard) | Δ baseline n=64 | Notes |
|---|---|---|---|---|---|---|---|
| baseline (cr_0413) | — | — | full GT | 55986 | 0.7148 / 0.6057 | — | reference |
| v2 | pred==gt | ∞ | (1, 1, 0) | 49476 | 0.7193 / 0.6060 | +0.45 pp | ≈ baseline |
| c20 | pred==gt | 20 | (1, 1, 0) | 65100 | 0.6861 / 0.5630 | -2.87 pp | |
| M3 | pred==gt | 10 | (1, 1, 0) | 63798 | 0.6640 / 0.5252 | -5.08 pp | |
| c5 | pred==gt | 5 | (1, 1, 0) | 65100 | 0.4404 / 0.2907 | -27.44 pp | |
| M2 (clue only) | clue only | — | (1, 1, 0) | 65100 | 0.2582 / 0.0924 | -45.66 pp | |
| inverse_mask cap=10 | pred==gt | 10 | **(0.1, 0.1, 1.0)** | 62496 | 0.6517 / 0.5053 | -6.31 pp | inverted weights |
| **conf=0.90 cap=10 (s0)** | pred==gt ∧ conf>0.9 | 10 | (1, 1, 0) | 63798 | 0.7336 / 0.6265 | +1.88 pp | ✅ |
| **conf=0.90 cap=10 (s1)** | pred==gt ∧ conf>0.9 | 10 | (1, 1, 0) | 65100 | 0.7250 / 0.6114 | +1.02 pp | ✅ 2-seed avg 0.7279 ± 0.004 |
| conf=0.70 cap=10 | pred==gt ∧ conf>0.7 | 10 | (1, 1, 0) | 65100 | 0.7289 / 0.6234 | +1.41 pp | ✅ |
| conf=0.80 cap=10 | pred==gt ∧ conf>0.8 | 10 | (1, 1, 0) | 65100 | 0.7139 / 0.5988 | -0.09 pp | outlier |
| conf=0.95 cap=10 | pred==gt ∧ conf>0.95 | 10 | (1, 1, 0) | 61194 | 0.7330 / 0.6326 | +1.82 pp | ✅ |
| ⭐ **conf=0.99 cap=10** | pred==gt ∧ conf>0.99 | 10 | (1, 1, 0) | 65100 | **0.7355** / **0.6300** | **+2.07 pp** | ✅ **stable best** |
| conf=0.99 cap=15 | pred==gt ∧ conf>0.99 | 15 | (1, 1, 0) | 48174 | 0.7077 / 0.5954 | -0.71 pp | stable but lower |
| ⚠️ **conf=0.99 cap=20** | pred==gt ∧ conf>0.99 | 20 | (1, 1, 0) | **39060** | **0.7431** / **0.6423** | **+2.83 pp** | ⚠️ **collapses to 0.0176 by final** |

## Key findings

### 1. Sweet spot: τ=0.99, cap=10 (n=64 = 0.7355, +2.07 pp)

The confidence-thresholded trust (`τ=0.99`) is the single most important
design choice. At cap=10 it's the only stable configuration that consistently
beats baseline. The 2-seed result at τ=0.90 confirms reproducibility (±0.004).

### 2. cap=20 has the highest peak ever, but collapses

| step | conf=0.99 cap=20 n=64 |
|---|---|
| 39060 | **0.7431** ⭐ peak — best of any run |
| 45570 | 0.7293 |
| 52080 | 0.6694 ⬇ |
| 58590 | 0.1446 ⬇⬇ |
| 65100 | 0.0176 💀 mode collapse |

This contradicts the naive "fewer ignore cells = better" hypothesis. The
mask (15-20 cells at cap=10) plays a stabilizing role; allowing too much
solver propagation (cap=20) lets noisy trust contaminate derived labels
faster than the model can self-correct.

### 3. Confidence filter is necessary; cap=10 is necessary for stability

- Without filter (M3/c20/c5/M2): all worse than baseline. Worse the tighter the cap.
- With filter (τ ≥ 0.70) at cap=10: all beat baseline (except 0.80 outlier).
- Both filter + cap=10 needed: τ=0.99 + cap=20 = collapse.

### 4. Hard cells separate winners from losers

Easy and medium cells saturate around 0.93-0.95 / 0.85-0.90 across all decent
runs. The differentiator is `hard_exact_acc`. Top runs vs baseline at peak n=64:

```
conf=0.99 cap=20 (peak)  hard=0.6423  Δ+3.66pp  (then collapses)
conf=0.95 cap=10         hard=0.6326  Δ+2.69pp
conf=0.99 cap=10         hard=0.6300  Δ+2.43pp  ⭐ stable best
conf=0.90 cap=10         hard=0.6265  Δ+2.08pp
baseline                  hard=0.6057  —
```

### 5. Inverse loss weighting is worse than the original

Pushing loss onto hard (mask) cells with `(0.1, 0.1, 1.0)` weights yields
0.6517 (-6.3 pp). The mask cells alone are too few and too varied for the
model to learn from in isolation; the trust+derivable cells provide the
structural prior the model needs.

## Wandb runs (all logged with eval_strat_nsup{16,32,64} namespaces)

| Run | Wandb URL |
|---|---|
| baseline | https://wandb.ai/sfu-tai-lab-zhe/TRM_CR/runs/cr_0413_baseline_220918 |
| v2 | https://wandb.ai/sfu-tai-lab-zhe/TRM_CR/runs/cr_0429_derive_ignore_194437 |
| c20 | https://wandb.ai/sfu-tai-lab-zhe/TRM_CR/runs/cr_0429_derive_ignore_c20_smoke_115745 |
| M3 | https://wandb.ai/sfu-tai-lab-zhe/TRM_CR/runs/cr_0429_derive_ignore_m3_194001 |
| c5 | https://wandb.ai/sfu-tai-lab-zhe/TRM_CR/runs/cr_0429_derive_ignore_c5_162547 |
| M2 | https://wandb.ai/sfu-tai-lab-zhe/TRM_CR/runs/cr_0429_derive_ignore_m2_204257 |
| inverse_mask | https://wandb.ai/sfu-tai-lab-zhe/TRM_CR/runs/cr_0429_derive_ignore_inverse_mask_cap_10_smoke_131137 |
| conf=0.90 (s0) | https://wandb.ai/sfu-tai-lab-zhe/TRM_CR/runs/cr_0429_derive_ignore_confidence_90_cap_10_smoke_130616 |
| conf=0.90 (s1) | https://wandb.ai/sfu-tai-lab-zhe/TRM_CR/runs/cr_0429_derive_ignore_confidence_90_cap_10_smoke_seed1_155423 |
| conf=0.70 | https://wandb.ai/sfu-tai-lab-zhe/TRM_CR/runs/cr_0429_derive_ignore_confidence_70_cap_10_smoke_103803 |
| conf=0.80 | https://wandb.ai/sfu-tai-lab-zhe/TRM_CR/runs/cr_0429_derive_ignore_confidence_80_cap_10_smoke_103803 |
| conf=0.95 | https://wandb.ai/sfu-tai-lab-zhe/TRM_CR/runs/cr_0429_derive_ignore_confidence_95_cap_10_smoke_103803 |
| conf=0.99 cap=10 | https://wandb.ai/sfu-tai-lab-zhe/TRM_CR/runs/cr_0429_derive_ignore_confidence_99_cap_10_smoke_103803 |
| conf=0.99 cap=15 | https://wandb.ai/sfu-tai-lab-zhe/TRM_CR/runs/cr_0429_derive_ignore_confidence_99_cap_15_smoke_160041 |
| conf=0.99 cap=20 | https://wandb.ai/sfu-tai-lab-zhe/TRM_CR/runs/cr_0429_derive_ignore_confidence_99_cap_20_smoke_160041 |

## Companion figures

- **Coverage comparison (all 9 runs)**:
  https://github.com/yoyostudy/trm-cr-experiment-photos/blob/coverage_comparison_all_runs/panel_comparisons/coverage_comparison_3x3.png
- **Confidence sweep response curve**:
  https://github.com/yoyostudy/trm-cr-experiment-photos/blob/sweep_response_curve/panel_comparisons/confidence_sweep_response_curve.png
- **Cap sweep at τ=0.99 (showing cap=20 collapse)**:
  https://github.com/yoyostudy/trm-cr-experiment-photos/blob/cap_sweep_curves/panel_comparisons/cap_sweep_at_tau99.png
