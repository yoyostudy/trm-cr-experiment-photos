# Experiment summary: `cr_0429_derive_ignore_v2_154835`

_Generated: 2026-05-05 15:23:36_

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

Latest checkpoint (numerically last step):

_step 65100_  (baseline ref: `stage2_step_65100`)

| nsup | cov | exact@halt | exact@halt_or_max | exact@max || base cov | base exact@halt | base exact@halt_or_max | base exact@max | Δexact@halt_or_max |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 0.5566 | 0.8569 | 0.4773 | 0.4485 | 0.6195 | 0.9154 | 0.5676 | 0.4976 | -0.0903 |
| 32 | 0.6186 | 0.8631 | 0.5341 | 0.3686 | 0.7003 | 0.9184 | 0.6434 | 0.6214 | -0.1093 |
| 64 | 0.6559 | 0.8643 | 0.5669 | 0.1841 | 0.7583 | 0.9177 | 0.6959 | 0.6753 | -0.1290 |

## Wandb run

- https://wandb.ai/sfu-tai-lab-zhe/TRM_CR/runs/cr_0429_derive_ignore_v2_154835

