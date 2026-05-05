# Experiment summary: `cr_0429_derive_ignore_v2_154835`

_Generated: 2026-05-05 14:57:10_

## Config

- arch: H_cycles=**1** L_cycles=**2** L_layers=2 halt_max_steps=16
- loss: `losses@DeriveIgnoreACTLossHead`
- batch_size: 768
- epochs: 20000
- data: ['data/sudoku-extreme-1k-aug-1000']

- run state: **finished**
- total runtime: 19.2 h
- final step: 65178

## Final in-training eval (`all.*`)

- step: **59892**
- cell_acc: **0.8545**
- exact_acc: **0.4768**
- baseline_220918 @ same step (interp):  cell=0.8657  exact=0.5998
- **Δcell = -0.0111**   **Δexact = -0.1230**

- best exact_acc on this run: **0.6506** @ step 46872
- baseline_220918 best exact_acc: 0.5998 @ step 59892

## Snapshot eval (44 ckpts evaluated, nsup={16,32,64})

Latest checkpoint:

_step 9114_  (baseline ref: `stage0_step_9114`)

| nsup | cov | exact@halt | exact@halt_or_max | exact@max || base cov | base exact@halt | base exact@halt_or_max | base exact@max | Δexact@halt_or_max |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 0.1734 | 0.7750 | 0.1355 | 0.1518 | 0.2044 | 0.8365 | 0.1717 | 0.1882 | -0.0362 |
| 32 | 0.2139 | 0.7096 | 0.1519 | 0.1634 | 0.2335 | 0.7942 | 0.1856 | 0.2056 | -0.0336 |
| 64 | 0.2397 | 0.6591 | 0.1580 | 0.1696 | 0.2498 | 0.7635 | 0.1908 | 0.2133 | -0.0327 |

## Wandb run

- https://wandb.ai/sfu-tai-lab-zhe/TRM_CR/runs/cr_0429_derive_ignore_v2_154835

