# Experiment summary: `exp_0413_cr_trace_mix_gaussian_210715`

_Generated: 2026-05-05 15:23:44_

## Config

- arch: H_cycles=**1** L_cycles=**2** L_layers=2 halt_max_steps=16
- loss: `losses@ACTLossHead`
- batch_size: 768
- epochs: 30000
- data: ['data/sudoku-extreme-1k-aug-1000']

- run state: **finished**
- total runtime: 5.6 h
- final step: 65250

## Final in-training eval (`all.*`)

- step: **59892**
- cell_acc: **0.8609**
- exact_acc: **0.5914**
- baseline_220918 @ same step (interp):  cell=0.8657  exact=0.5998
- **Δcell = -0.0047**   **Δexact = -0.0085**

- best exact_acc on this run: **0.5914** @ step 59892
- baseline_220918 best exact_acc: 0.5998 @ step 59892

## Wandb run

- https://wandb.ai/sfu-tai-lab-zhe/TRM_CR/runs/exp_0413_cr_trace_mix_gaussian_210715

