# Experiment summary: `cr_0429_derive_ignore_194437`

_Generated: 2026-04-30 10:05:05_

## Config

- arch: H_cycles=**1** L_cycles=**2** L_layers=2 halt_max_steps=16
- loss: `losses@DeriveIgnoreACTLossHead`
- batch_size: 768
- epochs: 30000
- data: ['data/sudoku-extreme-1k-aug-1000']

- run state: **finished**
- total runtime: 6.8 h
- final step: 39388

## Final in-training eval (`all.*`)

- step: **39060**
- cell_acc: **0.8724**
- exact_acc: **0.6349**
- baseline_220918 @ same step (interp):  cell=0.8414  exact=0.2576
- **Δcell = +0.0310**   **Δexact = +0.3773**

- best exact_acc on this run: **0.6391** @ step 37758
- baseline_220918 best exact_acc: 0.5998 @ step 59892

## Snapshot eval (30 ckpts evaluated, nsup={16,32,64})

Latest checkpoint:

_step 9114_  (baseline ref: `stage0_step_9114`)

| nsup | cov | exact@halt | exact@halt_or_max | exact@max || base cov | base exact@halt | base exact@halt_or_max | base exact@max | Δexact@halt_or_max |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 0.1734 | 0.7730 | 0.1352 | 0.1518 | 0.2044 | 0.8365 | 0.1717 | 0.1882 | -0.0365 |
| 32 | 0.2136 | 0.7090 | 0.1516 | 0.1635 | 0.2335 | 0.7942 | 0.1856 | 0.2056 | -0.0339 |
| 64 | 0.2396 | 0.6585 | 0.1578 | 0.1698 | 0.2498 | 0.7635 | 0.1908 | 0.2133 | -0.0329 |

## Wandb run

- https://wandb.ai/sfu-tai-lab-zhe/TRM_CR/runs/cr_0429_derive_ignore_194437

