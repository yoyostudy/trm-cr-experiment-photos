# Solver-Trajectory Chain-of-Thought — Implementation Plan

This doc is the concrete implementation plan for direction **D** in
[`PER_STEP_SUPERVISION.md`](https://github.com/yoyostudy/trm-cr-experiment-photos/blob/main/PER_STEP_SUPERVISION.md):
align each ACT iteration with the corresponding solver propagation step.

> tl;dr — instead of supervising every ACT step on the same final GT, supervise
> ACT step *k* only on cells the solver derives at solver-step *k*. TRM's
> recurrent reasoning is forced to mirror solver's chain.

Working directory: `/local-scratch/localhome/zwa204/tmp_0506/TinyRecursiveModels`
(code copy of `tmp_0429`, `data/` symlinked to share with `tmp_0429`).

---

## Status (2026-05-06)

- ✅ **Phase 1 script written** at `tmp_0506/TinyRecursiveModels/build_solver_trajectory.py`
- ✅ **Sanity-tested on 100 puzzles** of sudoku-extreme test set
- ⏸ Phases 2/3/4 pending GPU availability (current ablations finish ~17:00–19:00 today)

### Sanity-test findings (100 puzzles, sudoku-extreme test)

```
max solver step:                         18    (some cells need 18 propagations)
puzzles fully solvable (naked+hidden):   16 / 100  (16%)
avg unsolved cells per puzzle:           ~40-50 / 81  (need backtracking)
runtime:                                 0.2 s for 100 puzzles
                                         → ≈ 13 min for full 422k test set on 1 CPU
                                         → ≈ 1 min with 16 workers
```

### Key implications for the design questions

1. **Cells with `solver_step == -1` are the majority in hard puzzles**, not a
   rare edge case. ~50% of cells in sudoku-extreme are *not* solvable by
   single-propagation alone. The choice of how to supervise them is the most
   consequential decision.

2. **Max solver step (18) ≥ halt_max_steps (16)**. Some cells naturally need
   >16 propagations. Truncation will collapse the deepest few buckets into
   ACT step 16.

3. **Naked+hidden single is the bottleneck.** Most sudoku-extreme puzzles
   require backtracking, which the `solver_step` curriculum can't capture.
   This argues for a *fallback*: at the final ACT step, supervise the `-1`
   cells with full GT.

### Updated default decisions (post-sanity-test)

| Question | Default |
|---|---|
| Cumulative vs strict mode | **cumulative** (stable bootstrapping; many cells supervised early) |
| Cells with `solver_step == -1` | **option b — supervise at the LAST ACT step (k=16) with full GT** |
| Truncate `solver_step > 16` | **truncate to 16** (only a tail of cells; minimal info loss) |
| Discount γ | start with **1.0 (uniform)**; try 0.9 only if uniform underperforms |

---

## Phase 1 — Dataset preprocessing  (1–2 h)

For each sudoku puzzle, run the solver and record **which solver step each cell
is filled at**.

```python
# build_solver_trajectory.py — pseudocode
for puzzle in dataset:
    grid = puzzle.clues
    solver_step = np.full(81, -1, dtype=np.int8)   # -1 = solver gave up
    
    for cell in clue_cells:
        solver_step[cell] = 0                        # given at "step 0"
    
    step = 1
    while solver_can_propagate(grid):
        newly_filled = naked_single(grid) | hidden_single(grid)
        for cell in newly_filled:
            solver_step[cell] = step
        apply(newly_filled, grid)
        step += 1
    
    np.save(f"all__solver_step.npy[idx]", solver_step)
```

Output: same dataset directory plus a new file
`<dataset>/test/all__solver_step.npy` (and same in `train/`).

### Sanity stats to print after build

- Distribution of max-step `K` (`max(solver_step) per puzzle`).
- Fraction of cells with `solver_step == -1` (require backtracking).
- Histogram of solver_step values across all cells.

Used to decide:
- whether `K > 16` is common (decides truncation strategy)
- how many `-1` cells exist (decides if we can simply ignore them)

---

## Phase 2 — Loss head implementation  (2–3 h)

### New class `SolverTrajectoryACTLossHead`

```python
class SolverTrajectoryACTLossHead(nn.Module):
    """
    Aligns ACT iteration k with solver step k.

    Modes:
      - 'cumulative': supervise cells with solver_step <= k
      - 'strict':     supervise only cells with solver_step == k
      - 'discounted': cumulative with weight = gamma^(K-k) per step
    """
    def __init__(self, model, loss_type, mode='cumulative',
                 gamma=0.9, max_solver_step=16, **kwargs):
        ...
    
    def forward(self, **kwargs):
        # carry holds an act_step_idx counter (incremented externally)
        carry, outputs = self.model(**kwargs)
        labels      = carry.current_data['labels']
        solver_step = carry.current_data['solver_step']    # NEW field
        act_k       = carry.act_step_idx                   # NEW field
        
        if self.mode == 'cumulative':
            mask_k = (solver_step <= act_k) & (labels != IGNORE_LABEL_ID)
        elif self.mode == 'strict':
            mask_k = (solver_step == act_k) & (labels != IGNORE_LABEL_ID)
        # cells with solver_step == -1: not supervised at any step
        
        target = labels.clone()
        target[~mask_k] = IGNORE_LABEL_ID
        
        lm_loss = (loss_fn(outputs['logits'], target,
                            ignore_index=IGNORE_LABEL_ID, valid_mask=mask_k)
                   / mask_k.sum(-1).clamp_min(1).unsqueeze(-1)).sum()
        
        if self.mode == 'discounted':
            lm_loss = lm_loss * (self.gamma ** (self.max_solver_step - act_k))
        
        return ..., lm_loss, ...
```

### Required plumbing changes

1. **`puzzle_dataset.py`** — load `all__solver_step.npy` and put `solver_step`
   into each batch dict alongside `inputs`, `labels`, `difficulty`.
2. **`models/trm.py` (or wherever `initial_carry` lives)** — store an
   `act_step_idx` counter in the carry, increment it each iteration.
3. **`pretrain.py`** — pass through `solver_step` in batch dict (already
   pops `difficulty` similarly).

### Yaml configs

```yaml
# config/arch/trm_solver_traj_cot.yaml
loss:
  name: losses@SolverTrajectoryACTLossHead
  loss_type: stablemax_cross_entropy
  mode: cumulative           # or 'strict' or 'discounted'
  gamma: 0.9
  max_solver_step: 16
halt_max_steps: 16
H_cycles: 1
L_cycles: 2
...

# cr_0506_solver_traj_cot_smoke.yaml
name: cr_0506_solver_traj_cot_smoke
arch: trm_solver_traj_cot
data_path_test: data/sudoku-extreme-1k-aug-1000-smoke
stages:
  - data_path: data/sudoku-extreme-1k-aug-1000
    nsup: 16
    num_epochs: 50000
    batch_size: 768
    eval_interval: 1000
```

---

## Phase 3 — Training & debug  (1 day)

1. Smoke-test on a tiny puzzle subset locally (10 puzzles, ensure loss flows
   and decreases).
2. Launch full training on srv03 GPU 2 once `inverse_mask_cap_10` finishes
   (≈ 17:00 today). Expected total: 7 h training + 0.5 h smoke evals.
3. Evaluate the 3 modes (`cumulative` / `strict` / `discounted`) — pick best.

---

## Phase 4 — Comparison & paper material

After training completes, compare to:

| Reference | Expected n=64 |
|---|---|
| baseline (cr_0413) | 0.7148 |
| exp_0419 (H=2/L=4) | 0.7750 |
| derive_ignore best (v2) | 0.7164 |
| derive_ignore family (worst) | 0.2548 |
| **`solver_traj_cot` (this run)** | **TBD** |

### Expected outcomes

| | n=64 | Probability |
|---|---|---|
| **A** — beats baseline by +3 to +8 pp | 0.74–0.79 | 30% |
| **B** — matches baseline | 0.69–0.73 | 40% |
| **C** — slightly worse (overfits early-step easy cells) | 0.65–0.69 | 30% |

If outcome A: this becomes the paper's positive contribution. The story
becomes "*derive_ignore was the wrong way to use solver supervision; using
the solver's reasoning trajectory as a per-step curriculum is the right way.*"

If outcome B/C: still publishable as a careful negative-result paper showing
that several principled ideas don't help, with the framework as the
contribution.

---

## Open design questions to settle in Phase 1 stats

1. **Cumulative vs strict supervision?**
   - cumulative: more stable, more cells supervised early.
   - strict: stricter curriculum, but later ACT steps see very few cells.
   
   Default to **cumulative**; fall back to discounted if cumulative is too
   biased toward easy cells.

2. **What happens when `K > halt_max_steps` (16)?**
   - Truncate: cap `solver_step` at 16 → cells with depth > 16 share the last
     bucket.
   - Stretch ACT: allow `halt_max_steps = K_max` for hard puzzles. Costlier.
   
   Default: **truncate at 16**.

3. **Cells with `solver_step == -1` (require backtracking)?**
   - Option a: `IGNORE` at every ACT step (no supervision ever).
   - Option b: supervise at the *final* step only (`solver_step := 16` for them).
   - Option c: supervise at solver_step == K_max + 1.
   
   Default: **option b** — assign these to the last ACT step so the model
   still sees the GT, just only at deepest reasoning.

4. **Discount factor `gamma` (if using discounted mode)?**
   - 1.0 = uniform (= cumulative)
   - 0.9 = mild later-bias
   - 0.5 = strong later-bias
   
   Default: start with **0.9**, try 0.5 if 0.9 doesn't separate from baseline.

---

## Risk / failure modes

1. **Solver runs forever**: most sudoku puzzles solvable in <30 propagations,
   but degenerate cases exist. Cap solver iterations at 100.
2. **All-`-1` puzzles**: very hard puzzles where naked+hidden single can't
   start. These get fully `IGNORE` under option a. If option b, only step-16
   supervision. Tradeoff between the two.
3. **`act_step_idx` plumbing**: the cleanest implementation requires the
   carry to track `act_step_idx`, which means modifying TRM's model code,
   not just the loss head. Cleaner architecture; more invasive.
4. **Compute cost**: Phase 1 preprocessing on 1M puzzles × 30 propagations
   each ≈ minutes (CPU-bound, simple loops). Negligible.

---

## File layout

```
tmp_0506/TinyRecursiveModels/
  build_solver_trajectory.py            # Phase 1 preprocessing
  models/
    losses.py                           # add SolverTrajectoryACTLossHead
    trm.py                              # add act_step_idx to carry (if needed)
  config/arch/trm_solver_traj_cot.yaml  # arch with new loss
  cr_0506_solver_traj_cot_smoke.yaml    # top-level run config
  data/                                 # symlink to tmp_0429/data
  SOLVER_TRAJECTORY_COT_PLAN.md         # this doc
```
