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

## 3. Currently-Running Ablations

Each row is a 3-D coordinate `(Trust × Solver × Loss-weight)`.

| Run | Trust | Solver cap | Loss weight | Hypothesis tested |
|---|---|---|---|---|
| `c20` | `pred==gt` | 20 | original | mild mask = regularization? |
| `confidence_90_cap_10` | `pred==gt ∧ conf > 0.9` | 10 | original | confidence-filtered trust + real mask = self-paced curriculum? |
| `inverse_mask_cap_10` | `pred==gt` | 10 | inverse `(0.1, 0.1, 1.0)` | flip the gradient onto hard cells |

### Reference (already completed)

| Run | Trust | Cap | Loss weight | n=64 final |
|---|---|---|---|---|
| baseline (cr_0413) | (no derive_ignore) | — | full GT | **0.7148** |
| `v2` / `194437` | `pred==gt` | ∞ | original | 0.7164 ± 0.004 (≈ baseline) |
| `M3` = `c=10` | `pred==gt` | 10 | original | 0.6640 (−5 pp) |
| `c=5` | `pred==gt` | 5 | original | 0.4404 (−27 pp) |
| `M2` | clues only | 0 | original | 0.2548 (−46 pp) |

**Pattern:** with the original loss weighting, accuracy degrades monotonically
as the mask strengthens (`mask plateau` ↑). The "any non-trivial mask hurts"
finding motivates the improvements above.

---

## 4. Promising Untested Combinations

```
A. focal:              weight ∝ CE_per_cell                      ← solver-free hard mining
B. inverse + conf:     trust = conf>0.9, cap=10, weight=(0.1,0.1,1)  ← double filter
C. soft trust:         trust_weight = sigmoid(conf), uniform loss  ← removes binary mask
D. no-solver inverse:  cap=0, weight = (0,0,1)                  ← all-hard supervision
E. uniform / baseline: weight = (1,1,1)                          ← sanity check
```

If the current 3 ablations don't beat baseline, A and C are the next promising
directions — both decouple the loss design from the solver entirely.

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
