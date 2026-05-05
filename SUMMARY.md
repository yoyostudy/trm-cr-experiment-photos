# Experiment summary: `exp_0419_cr_baseline_004937`

_Generated: 2026-05-05 15:24:07_

## Config

- arch: H_cycles=**2** L_cycles=**4** L_layers=2 halt_max_steps=16
- loss: `losses@ACTLossHead`
- batch_size: 768
- epochs: 30000
- data: ['data/sudoku-extreme-1k-aug-1000']

- run state: **finished**
- total runtime: 15.1 h
- final step: 65148

## Final in-training eval (`all.*`)

- step: **59892**
- cell_acc: **0.8843**
- exact_acc: **0.6382**
- baseline_220918 @ same step (interp):  cell=0.8657  exact=0.5998
- **Δcell = +0.0186**   **Δexact = +0.0384**

- best exact_acc on this run: **0.6550** @ step 55986
- baseline_220918 best exact_acc: 0.5998 @ step 59892

## Wandb run

- https://wandb.ai/sfu-tai-lab-zhe/TRM_CR/runs/exp_0419_cr_baseline_004937

