# Experiment summary: `exp_0414_bt_descendents_015622`

_Generated: 2026-05-05 15:23:59_

## Config

- arch: H_cycles=**1** L_cycles=**2** L_layers=2 halt_max_steps=16
- loss: `losses@ACTLossHead`
- batch_size: 768
- epochs: 30000
- data: ['data/sudoku-extreme-1k-aug-1000']

- run state: **finished**
- total runtime: 8.7 h
- final step: 65200

## Final in-training eval (`all.*`)

- step: **59892**
- cell_acc: **0.8622**
- exact_acc: **0.4580**
- baseline_220918 @ same step (interp):  cell=0.8657  exact=0.5998
- **Δcell = -0.0034**   **Δexact = -0.1418**

- best exact_acc on this run: **0.4595** @ step 57288
- baseline_220918 best exact_acc: 0.5998 @ step 59892

## Wandb run

- https://wandb.ai/sfu-tai-lab-zhe/TRM_CR/runs/exp_0414_bt_descendents_015622

