# Per-Step Supervision (Intra-ACT)

## 1. Background — what supervision currently exists

TRM unrolls **up to `halt_max_steps = 16` ACT iterations** per forward pass:

```
step 1:  carry_0 → model → carry_1, pred_1, q_halt_1
step 2:  carry_1 → model → carry_2, pred_2, q_halt_2
...
step 16: carry_15 → model → carry_16, pred_16, q_halt_16
                                          ↑
                       sample halts whenever q_halt_k > 0
```

**Each iteration is independently forward+backward'd** in the training loop:

```python
# pretrain.py train loop (simplified)
carry = model.initial_carry(batch)
while True:
    carry, loss, metrics, preds, all_finish = model(carry=carry, batch=batch)
    loss.backward()
    optimizer.step()
    if all_finish: break
```

So gradient already flows at each ACT step — but every step uses the **same
loss target** (full GT, with derive_ignore mask if applicable). The model's
intermediate predictions `pred_1, …, pred_15` are supervised with the *same*
signal as the final `pred_16`.

This is implicit per-step supervision, but without exploiting the *step
structure* of reasoning. The improvements below add explicit step-aware
training signals.

---

## 2. Improvement Directions (intra-ACT)

### A. Discount-weighted anytime supervision

**Idea.** Weight later ACT steps higher so the loss prioritizes the "final
answer" while still giving every step a non-zero gradient.

```python
loss = Σ_k  γ^(16-k) * CE(pred_k, gt_k)
        k=1..16

γ ∈ (0, 1)   e.g. γ = 0.9 → step 1 weight ≈ 0.21, step 16 weight = 1.0
```

**Properties.**
- Cheap: 1-line code change, no new labels.
- Already implicitly happens (each step calls `loss.backward()`); this just
  rebalances how much each contributes.
- Can be combined with derive_ignore mask or its inverse.

**Risk.** Effect may be small — gradient magnitudes already vary across steps
because of the carry detachment between iterations.

---

### B. Difficulty-curriculum across steps

**Idea.** Earlier ACT steps supervise *easy* cells (solver-derivable in few
propagations). Later steps supervise *hard* cells.

```
ACT step k supervises cells with solver_depth ≤ k
                                       ┃
        depth = number of naked/hidden-single propagations
                needed to derive the cell from clues alone
```

**Concrete.**
1. For each puzzle, run the full solver with depth annotation, label every
   cell with `solver_depth ∈ {1, 2, …}`.
2. In the loss, build a per-step mask:
   ```python
   step_mask_k = (solver_depth ≤ k) & valid
   loss_k = CE(pred_k, gt) on step_mask_k
   ```
3. Sum: `loss = Σ_k loss_k`.

**Why this might help.** Forces the model to learn a *layered* reasoning
strategy — step 1 must crack only single-propagation cells; step 16 must
crack the deeply-buried cells. This directly encodes the recursive structure
of TRM into the loss function.

**Cost.** Need to label cells with `solver_depth` (one-time pre-processing),
modify the loss head to keep per-step masks. Medium implementation effort.

---

### C. Auxiliary heads at each step

**Idea.** Add a small auxiliary prediction head on the carry at each step, with
a side objective that complements the main GT loss.

```python
for k in range(16):
    pred_k    = main_head(carry_k)         # primary: predict GT cells
    aux_k     = aux_head(carry_k)          # auxiliary: e.g., predict
                                           #   - difficulty score per cell
                                           #   - "remaining steps to halt"
                                           #   - solver-step-of-derivation
    loss_k = CE(pred_k, gt) + λ * AuxLoss(aux_k, aux_target_k)
```

**Variants of aux target.**
- `aux_target_k = (solver_depth == k)` — predict which cells will be derivable
  *at this exact* solver step.
- `aux_target_k = remaining_solver_depth` — predict how much reasoning remains.
- `aux_target_k = uncertainty_per_cell` — predict the model's own confidence
  (self-distillation).

**Properties.** Decouples reasoning from prediction — the model learns
*about* the problem alongside *solving* it. Common in self-supervised /
multi-task learning.

---

### D. Solver-trajectory chain-of-thought

**Idea.** Tie ACT step `k` directly to *solver step `k`*. The supervision at
step `k` is "the cells the solver derives in its `k`-th propagation".

```
Solver run from clue alone:
  prop 1 → fills cells {a, b, c}        (3 naked singles)
  prop 2 → fills cells {d, e}           (2 hidden singles)
  prop 3 → fills cells {f, g, h, i}
  ...
  prop K → puzzle solved (or stuck)

ACT step k of TRM:
  target_k = cells filled at solver prop k
            ↑
   Each ACT step incrementally adds cells, mirroring solver chain
```

**Concrete loss.**
```python
target_k = solver_trace[k]                # cells filled at solver step k
mask_k = make_mask(target_k, gt)
loss_k = CE(pred_k, gt) on cells in target_k
loss = Σ_k loss_k
```

This is essentially **chain-of-thought training adapted to sudoku**. TRM's
ACT iterations become an *explicit* simulation of solver propagation.

**Why this is the most promising "per-step" direction.**
- Aligns the inductive bias of TRM (recurrent reasoning) with the structure
  of the problem (sequential rule application).
- Provides a *graduated* curriculum within every forward pass: easy cells
  early, hard cells late.
- Avoids the systematic "hard cells get no gradient" failure of `derive_ignore`.

**Cost.** Highest among the four:
1. Build dataset with solver trajectory per puzzle (one-time pre-processing).
2. Modify ACT loop / loss head to use per-step targets.
3. Likely needs different hyperparameters (lr schedule, halt behavior).
1–2 days of implementation + debugging.

---

### E. Anytime halt regularization (already in TRM)

For completeness — TRM already trains a `q_halt` head that learns
*when* to stop. This is implicit per-step supervision on the halt decision.

```python
# Existing:
loss_halt = BCE(q_halt_k, is_correct_at_step_k)
```

This could be extended (e.g., pose halt as continuous "remaining-difficulty"
regression) but the existing version is reasonable.

---

## 3. Comparison & Priority

| Direction | Cost | Likely Effect | Combinable with derive_ignore |
|---|---|---|---|
| A. Discount weights | 1 line | small (±1-2 pp) | ✓ |
| B. Difficulty curriculum | medium | medium (+2-5 pp?) | ✓ |
| C. Auxiliary heads | medium | unclear | ✓ |
| D. Solver-trajectory CoT | high | **could be big (+5-10 pp?)** | replaces derive_ignore |
| E. Halt reg (existing) | done | small | already in |

**Recommended order.**
1. Run the current `c20 / confidence_90_cap_10 / inverse_mask_cap_10`
   ablations to completion (today).
2. If none beat baseline, commit to **direction D** (solver-trajectory CoT) —
   it has the highest theoretical upside and re-frames the entire approach.
3. Direction A as a free-cheap safety knob layered on whatever wins.

---

## 4. Connection to the existing framework

Per-step supervision is a fourth dimension orthogonal to the
`(trust × solver-cap × loss-weight)` axes in
[`DERIVE_IGNORE_FRAMEWORK.md`](DERIVE_IGNORE_FRAMEWORK.md):

```
Original framework axes:
  Trust definition   →   what counts as "model knows"?
  Solver cap         →   how aggressively to derive easy cells
  Loss weight        →   where to place gradient

NEW: Step axis:
  Per-step supervision design    →   how to use intra-ACT structure
```

Combining axes opens a large design space, e.g. *"inverse mask + cap=10 +
solver-trajectory CoT"*. But each new axis multiplies experiment count —
proceed only if early results justify.

---

## 5. Why this matters for the paper

If `derive_ignore` and its variants all fail to beat baseline (current
trajectory looks this way), the paper still has two options:

1. **Negative result paper.** "We tried 6 variants of solver-guided
   supervision; none beat baseline. Here's why: it systematically removes
   gradient from the cells most needing supervision."

2. **Pivot to per-step / chain-of-thought.** Use derive_ignore failure as
   motivation; introduce solver-trajectory CoT (direction D) as the
   *positive* contribution.

Option 2 is much stronger paper material if it works.
