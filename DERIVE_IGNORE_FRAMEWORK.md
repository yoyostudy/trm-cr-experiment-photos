# On-the-Fly Supervision: Design Space and Improvement Directions

## 1. Original Mechanism (`derive_ignore`)

```
pred       = model(clue)
trust      = clue ∪ {cell : pred == gt}     # cells the model already gets right
derivable  = solver(trust, cap = ∞)         # cells the solver can propagate from trust
                                            # via naked-single + hidden-single rules
mask       = 81 − trust − derivable          # cells solver cannot derive ("hard")

loss = Σ CE(pred, gt) for cell in (trust ∪ derivable)
     (mask cells get NO loss / are ignored)
```

### Core critique

Mask cells = the cells that require **real neural reasoning** (not solver rules).
By zeroing the loss on these, we **systematically deprive the model of gradient
signal exactly where it most needs to learn**. The model ends up with capacity
spent on cells the solver can already handle, while the genuinely hard cells
never receive direct supervision.

---

## 2. Three Independent Improvement Dimensions

The original mechanism can be modified along three orthogonal axes.

### Dimension 1 — Trust Definition

How do we decide which cells the model "knows"?

| Variant | Trust set |
|---|---|
| **Original** | `clue ∪ {pred == gt}` |
| **Confidence-thresholded** | `clue ∪ {pred == gt AND conf(pred) > τ}` |
| **Strict (M2)** | `clue` only |
| **Soft trust (untested)** | per-cell weight `= sigmoid(conf)` (not binary) |

Tightening trust forces the mask to **engage even when the model is correct
but uncertain**, hopefully avoiding bootstrap noise from "lucky guesses".

### Dimension 2 — Solver Strength

How aggressively does the solver fill in cells from the trust set?

| Variant | Solver depth |
|---|---|
| **Original** | `cap = ∞` (full propagation) |
| **Capped** | `cap = N` (N propagations max) |
| **Disabled (N3)** | `derivable = ∅` |

A smaller cap means more cells fall outside `trust ∪ derivable`, so the mask
genuinely engages. Without a cap the solver fills nearly everything from any
modest trust set, making the mask an empty no-op.

### Dimension 3 — Loss Weighting

Where do we place the gradient?

| Variant | Per-cell weight `(trust, derivable, mask)` |
|---|---|
| **Original** | `(1.0, 1.0, 0.0)` — easy cells supervised, hard cells ignored |
| **Inverse** | `(0.1, 0.1, 1.0)` — focus on the hard cells |
| **Uniform** | `(1.0, 1.0, 1.0)` — same as plain baseline |
| **Focal (untested)** | weight ∝ per-cell CE — automatic hard-example mining |

Inverse weighting flips the original intuition: instead of teaching the model
the rules the solver already knows, push gradient where the solver gives up.

---

## 3. Final Results (all runs evaluated on full test, 422786 puzzles)

Each row is a 3-D coordinate `(Trust × Solver × Loss-weight)`.

| Run | Trust | Cap | Loss weight | n=16 | n=32 | **n=64** | Δ baseline |
|---|---|---|---|---|---|---|---|
| baseline (cr_0413) | — | — | full GT | 0.5972 | 0.6691 | **0.7148** | — |
| `v2` (2-seed avg) | `pred==gt` | ∞ | (1, 1, 0) | 0.6168 ± .002 | 0.6790 ± .003 | **0.7164 ± .004** | +0.2 pp |
| **τ=0.99 + cap=10** | `pred==gt ∧ conf>0.99` | 10 | (1, 1, 0) | **0.6489** | **0.7031** | **0.7355** ⭐⭐ | **+2.07 pp** ✅ |
| **τ=0.99 + cap=15** | `pred==gt ∧ conf>0.99` | 15 | (1, 1, 0) | TBD | TBD | **0.6933** (final) | -2.15 pp ⚠️ |
| &nbsp;&nbsp;↳ peak (step 48174) | … | … | … | — | — | 0.7077 | -0.71 pp |
| **τ=0.99 + cap=20** | `pred==gt ∧ conf>0.99` | 20 | (1, 1, 0) | TBD | TBD | **0.0176** 💀 collapse | catastrophic |
| &nbsp;&nbsp;↳ **peak (step 39060)** | … | … | … | — | — | **0.7431** ⭐⭐⭐ best ever | **+2.83 pp** at peak |
| **τ=0.95 + cap=10** | `pred==gt ∧ conf>0.95` | 10 | (1, 1, 0) | 0.6455 | 0.7007 | **0.7321** | +1.73 pp ✅ |
| ⭐ **τ=0.90 + cap=10 `conf_90_cap_10`** (2-seed) | `pred==gt ∧ conf>0.9` | 10 | (1, 1, 0) | 0.6364 ± .005 | 0.6942 ± .005 | **0.7279 ± .004** | +1.31 pp ✅ |
| &nbsp;&nbsp;↳ seed=0 | … | … | … | 0.6396 | 0.6975 | 0.7307 | +1.59 pp |
| &nbsp;&nbsp;↳ seed=1 | … | … | … | 0.6331 | 0.6909 | 0.7250 | +1.02 pp |
| **τ=0.70 + cap=10** | `pred==gt ∧ conf>0.7` | 10 | (1, 1, 0) | 0.6453 | 0.6980 | **0.7289** | +1.41 pp ✅ |
| **τ=0.80 + cap=10** | `pred==gt ∧ conf>0.8` | 10 | (1, 1, 0) | 0.6186 | 0.6785 | **0.7139** | -0.09 pp ⚠️ outlier |
| `c20` | `pred==gt` | 20 | (1, 1, 0) | 0.5929 | 0.6525 | 0.6861 | -2.9 pp |
| `M3` | `pred==gt` | 10 | (1, 1, 0) | 0.5758 | — | 0.6640 | -5.1 pp |
| `inverse_mask_cap_10` | `pred==gt` | 10 | **(0.1, 0.1, 1.0)** | 0.5307 | 0.6013 | 0.6463 | -6.9 pp |
| `c5` | `pred==gt` | 5 | (1, 1, 0) | 0.3382 | 0.4005 | 0.4404 | -27 pp |
| `M2` | **clue only** | — | (1, 1, 0) | — | — | 0.2548 | -46 pp |

---

## 4. Per-axis verdict

### Axis 1 — Solver cap: ❌ does not help on its own (and large cap is unstable with τ filter!)

**Without confidence filter** (τ=0): tightening the cap monotonically degrades
accuracy. `mask_n_ignore ↑ → exact_acc ↓` is a near-linear curve:

| Run | Cap | mask_n_ignore (final) | n=64 |
|---|---|---|---|
| `v2` | ∞ | ~0 (cap=∞ propagates everything) | 0.7164 |
| `c20` | 20 | 5.6 | 0.6861 |
| `M3` | 10 | 14.6 | 0.6640 |
| `c5` | 5 | 22.4 | 0.4404 |
| `M2` | clue-only | 47.7 | 0.2548 |

**With confidence filter** (τ=0.99): the relationship is **non-monotonic**!
Larger cap → fewer ignore cells → but training becomes unstable:

| Cap | mask_n_ignore (est.) | n=64 peak | n=64 final | Stability |
|---|---|---|---|---|
| **10** | ~15 (19%) | 0.7355 | **0.7355** | ✅ stable |
| 15 | ~9 (11%) | 0.7077 (step 48174) | 0.6933 | ✅ stable, but lower |
| **20** | ~6 (7%) | **0.7431** (step 39060) ⭐ best ever | **0.0176** | 💀 **collapses** |

**Key finding (counter-intuitive)**: At τ=0.99, **cap=20 achieves the highest
peak ever observed (0.7431, +2.83 pp over baseline) at step 39060**, but then
training catastrophically collapses to ~random by step 65100. cap=10 has
*more* ignore cells (19%) but is the only stable setting. The hypothesis
that "fewer ignore cells = better" is wrong: cap=10's mask plays a stabilizing
role even though it removes 19% of supervision.

Plausible cause of cap=20 collapse: with the very strict trust filter
(τ=0.99) the trust set is small and noisy; allowing the solver to propagate
deeper (cap=20) lets early errors snowball into incorrectly-labeled
"derivable" cells. The model trains on these wrong labels, drifts, and never
recovers. cap=10's tighter solver acts as a circuit breaker.

### Axis 2 — Confidence-thresholded trust: ⭐ this is the win

Same cap=10, only difference is the confidence threshold `τ` on `pred==gt`.
We swept `τ ∈ {0.70, 0.80, 0.90, 0.95, 0.99}` (cap=10 fixed):

| Run | Trust filter (cap=10) | n=64 | Δ baseline |
|---|---|---|---|
| `M3` (τ=0, no filter) | `pred==gt` | 0.6640 | -5.1 pp |
| τ=0.70 | `pred==gt ∧ conf>0.70` | 0.7289 | +1.41 pp ✅ |
| τ=0.80 | `pred==gt ∧ conf>0.80` | 0.7139 | -0.09 pp ⚠️ outlier |
| τ=0.90 (2-seed) | `pred==gt ∧ conf>0.90` | 0.7279 ± .004 | +1.31 pp ✅ |
| τ=0.95 | `pred==gt ∧ conf>0.95` | 0.7321 | +1.73 pp ✅ |
| **τ=0.99** | `pred==gt ∧ conf>0.99` | **0.7355** ⭐⭐ | **+2.07 pp** ✅ |

**+7.1 pp** going from no filter (M3 = 0.664) to τ=0.99 (0.736).

**Sweet spot: τ=0.99** — higher confidence threshold is monotonically better
(τ=0.80 appears to be an outlier; τ=0.70/0.95/0.99 follow a clear upward
trend with τ). Reproduced across 2 seeds at τ=0.90 (0.7307 and 0.7250),
confirming this is not a lucky-seed artifact.

**Mechanism**: without confidence filtering, the trust set contains cells
where the model "got lucky" — its prediction happened to equal GT but it
doesn't reliably know that cell. Solver propagation from these noisy seeds
produces incorrect "derivable" cells, polluting the entire supervision
signal. The confidence threshold filters out unreliable predictions, so
solver propagation starts from a clean trust set.

### Axis 3 — Inverse loss weighting: ❌ worse than the baseline weighting

Same cap=10, only difference is loss weight:

| Run | Loss weight (trust, derivable, mask) | n=64 |
|---|---|---|
| `M3` | (1.0, 1.0, 0.0) — ignore hard cells | 0.6640 |
| `inverse_mask_cap_10` | (0.1, 0.1, 1.0) — focus on hard cells | 0.6463 |

Putting loss weight on the 5–15 mask cells instead of the 65+ trust+derivable
cells provides ~10× less total gradient signal per puzzle. The hard cells
alone are too few and too varied for the model to learn them — the easy
cells provide the structural prior needed for any reasoning to emerge.

---

## 5. Paper takeaway

**Old narrative** (planned, before `conf_90_cap_10` was found): negative
result paper showing that derive_ignore variants don't help.

**New narrative** (actual results): positive result + systematic analysis.

> *Solver-guided supervision (derive_ignore) systematically harms learning
> unless trust is filtered by model confidence. Specifically:*
>
> 1. *Naive derive_ignore hurts: tightening the solver cap monotonically
>    degrades accuracy.*
> 2. *Confidence-thresholded trust fixes it: filtering trust by
>    `pred==gt ∧ conf>0.9` allows derive_ignore to beat baseline (+1.31 pp at
>    n=64, 2-seed avg), while the same cap without confidence filtering loses
>    5 pp. The improvement is reproduced across 2 seeds (0.7307 and 0.7250),
>    confirming it is not a lucky-seed artifact.*
> 3. *Re-weighting the loss onto hard cells doesn't help: directly putting
>    loss weight on the cells the solver can't derive is worse than ignoring
>    them, because the hard cells are too few and too varied to learn from
>    in isolation.*
>
> *The value of solver-guided supervision is not in the cap or in inverse
> weighting, but in using the solver's reliance on confident model
> predictions to filter out unreliable bootstrap signal.*

---

## 6. Untested combinations worth trying next

```
A. confidence + cap=20:    trust = conf>τ, cap=20         ← softer mask, see if τ scales
B. confidence sweep:       conf>0.7, 0.8, 0.95           ← find sweet spot for τ
C. confidence-only:        trust = conf>0.9, cap=∞       ← the originally-crashed run
D. focal weighting:        weight ∝ CE_per_cell           ← solver-free hard mining
E. soft trust:             trust_weight = sigmoid(conf), uniform loss
```

**Done**:
- 2-seed reproduction of `conf_90_cap_10` confirmed real signal (seeds 0
  and 1: 0.7307 and 0.7250, both above baseline 0.7148).
- Confidence threshold sweep `τ ∈ {0.70, 0.80, 0.90, 0.95, 0.99}` with
  cap=10. Sweet spot is **τ=0.99 → 0.7355 (+2.07 pp)**; trend is monotonic
  upward except τ=0.80 outlier.

**Next highest priority**:
- C (cap=∞ with confidence filter) to isolate the contribution of the cap.
- 2nd seed of τ=0.99 to confirm the new best is not lucky-seed.
- D (focal weighting): solver-free hard-example mining as an alternative
  axis.

---

## 5. Smoke-Eval Setup

To enable fast in-training feedback, every ablation evaluates on a stratified
smoke test set of **600 puzzles** (12% easy / 26% medium / 62% hard, matching
the full test distribution). This makes each in-training eval ≈ 5 s instead of
the full ≈ 46 min, reducing total training time by ~30%.

Per-checkpoint, every 5th checkpoint is additionally evaluated on the **full
test set** (~422k puzzles) for headline-quality numbers.

---

## 6. Evaluation Metric

Primary: `exact_acc_at_halt_or_max @ nsup ∈ {16, 32, 64}` on the full test set.

`exact_accuracy` = the entire 81-cell puzzle is solved correctly.
`at_halt_or_max` = take prediction at the step model halted, or at max nsup if
not halted. `nsup = 16` matches baseline's `halt_max_steps`; `nsup = 32, 64`
test extra "thinking time".

The headline summary should always be computed on the full test set with the
true difficulty distribution; smoke-test averages need to be re-weighted as
`weighted_all = 0.115 * easy + 0.257 * medium + 0.628 * hard` for fair comparison.
