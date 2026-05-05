# Experiment summary: `cr_0429_derive_ignore_m3_194001`

_Generated: 2026-05-05 16:16:22_

## Config

- arch: H_cycles=**1** L_cycles=**2** L_layers=2 halt_max_steps=16
- loss: `losses@DeriveIgnoreACTLossHead`
  - auxiliary_gt_loss_weight = **0**
- batch_size: 768
- epochs: 20000
- data: ['data/sudoku-extreme-1k-aug-1000']

- run state: **finished**
- total runtime: 16.7 h
- final step: 63903

## Final in-training eval (`all.*`)

- step: **59892**
- cell_acc: **0.8555**
- exact_acc: **0.5948**
- baseline_220918 @ same step (interp):  cell=0.8657  exact=0.5998
- **Δcell = -0.0101**   **Δexact = -0.0050**

- best exact_acc on this run: **0.5948** @ step 59892
- baseline_220918 best exact_acc: 0.5998 @ step 59892

## Snapshot eval (36 ckpts evaluated, nsup={16,32,64})

Latest checkpoint (numerically last step):

_step 63798_  (baseline ref: `stage2_step_63798`)

| nsup | cov | exact@halt | exact@halt_or_max | exact@max || base cov | base exact@halt | base exact@halt_or_max | base exact@max | Δexact@halt_or_max |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 0.6098 | 0.9431 | 0.5758 | 0.5401 | 0.6266 | 0.9212 | 0.5776 | 0.5697 | -0.0018 |
| 32 | 0.6706 | 0.9416 | 0.6316 | 0.5610 | 0.7084 | 0.9234 | 0.6543 | 0.5721 | -0.0227 |
| 64 | 0.7088 | 0.9367 | 0.6640 | 0.4752 | 0.7641 | 0.9226 | 0.7050 | 0.5484 | -0.0409 |

## Wandb run

- https://wandb.ai/sfu-tai-lab-zhe/TRM_CR/runs/cr_0429_derive_ignore_m3_194001

