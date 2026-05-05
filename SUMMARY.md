# Experiment summary: `cr_0429_derive_ignore_m2_204257`

_Generated: 2026-05-05 15:51:23_

## Config

- arch: H_cycles=**1** L_cycles=**2** L_layers=2 halt_max_steps=16
- loss: `losses@DeriveIgnoreACTLossHead`
  - auxiliary_gt_loss_weight = **0**
- batch_size: 768
- epochs: 20000
- data: ['data/sudoku-extreme-1k-aug-1000']

- run state: **finished**
- total runtime: 18.1 h
- final step: 65168

## Final in-training eval (`all.*`)

- step: **59892**
- cell_acc: **0.6925**
- exact_acc: **0.2387**
- baseline_220918 @ same step (interp):  cell=0.8657  exact=0.5998
- **Δcell = -0.1731**   **Δexact = -0.3611**

- best exact_acc on this run: **0.2387** @ step 59892
- baseline_220918 best exact_acc: 0.5998 @ step 59892

## Wandb run

- https://wandb.ai/sfu-tai-lab-zhe/TRM_CR/runs/cr_0429_derive_ignore_m2_204257

