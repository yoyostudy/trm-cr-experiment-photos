# Experiment summary: `cr_0429_derive_ignore_v2_154835`

_Generated: 2026-05-05 14:50:46_

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

## Wandb run

- https://wandb.ai/sfu-tai-lab-zhe/TRM_CR/runs/cr_0429_derive_ignore_v2_154835

