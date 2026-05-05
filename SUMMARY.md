# Experiment summary: `cr_0429_derive_ignore_194437`

_Generated: 2026-05-05 15:23:19_

## Config

- arch: H_cycles=**1** L_cycles=**2** L_layers=2 halt_max_steps=16
- loss: `losses@DeriveIgnoreACTLossHead`
- batch_size: 768
- epochs: 20000
- data: ['data/sudoku-extreme-1k-aug-1000']

- run state: **finished**
- total runtime: 11.3 h
- final step: 65909

## Final in-training eval (`all.*`)

- step: **59892**
- cell_acc: **0.8541**
- exact_acc: **0.3602**
- baseline_220918 @ same step (interp):  cell=0.8657  exact=0.5998
- **Δcell = -0.0116**   **Δexact = -0.2396**

- best exact_acc on this run: **0.6510** @ step 46872
- baseline_220918 best exact_acc: 0.5998 @ step 59892

## Snapshot eval (50 ckpts evaluated, nsup={16,32,64})

Latest checkpoint (numerically last step):

_step 65100_  (baseline ref: `stage2_step_65100`)

| nsup | cov | exact@halt | exact@halt_or_max | exact@max || base cov | base exact@halt | base exact@halt_or_max | base exact@max | Δexact@halt_or_max |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 0.5853 | 0.8783 | 0.5144 | 0.1769 | 0.6195 | 0.9154 | 0.5676 | 0.4976 | -0.0531 |
| 32 | 0.6592 | 0.8839 | 0.5828 | 0.0414 | 0.7003 | 0.9184 | 0.6434 | 0.6214 | -0.0605 |
| 64 | 0.7098 | 0.8844 | 0.6278 | 0.0108 | 0.7583 | 0.9177 | 0.6959 | 0.6753 | -0.0681 |

## Wandb run

- https://wandb.ai/sfu-tai-lab-zhe/TRM_CR/runs/cr_0429_derive_ignore_194437

