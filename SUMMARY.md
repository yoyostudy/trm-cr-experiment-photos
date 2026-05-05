# Experiment summary: `exp_0421_cr_trace_mix_gaussian_192914`

_Generated: 2026-05-05 15:24:22_

## Config

- arch: H_cycles=**1** L_cycles=**2** L_layers=2 halt_max_steps=16
- loss: `losses@ACTLossHead`
- batch_size: 768
- epochs: 10000
- data: ['data/sudoku-extreme-1k-aug-1000']

- run state: **finished**
- total runtime: 9.3 h
- final step: 65148

## Final in-training eval (`all.*`)

- step: **59892**
- cell_acc: **0.8436**
- exact_acc: **0.5194**
- baseline_220918 @ same step (interp):  cell=0.8657  exact=0.5998
- **Δcell = -0.0220**   **Δexact = -0.0804**

- best exact_acc on this run: **0.5301** @ step 57288
- baseline_220918 best exact_acc: 0.5998 @ step 59892

## Wandb run

- https://wandb.ai/sfu-tai-lab-zhe/TRM_CR/runs/exp_0421_cr_trace_mix_gaussian_192914

