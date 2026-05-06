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
- 🔁 **Decision: skip Phase A, go directly to Phase C** — naked+hidden-single solver
  only solves 16% of puzzles, leaving 84% with `solver_step == -1`. This trains
  the model on easy puzzles only. To learn hard reasoning we need a stronger
  solver that traces backtracking. See "Phase C strategy" below.
- ⏸ Phases 2/3/4 pending Phase C solver implementation

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

---

# Phase C strategy — Teach the model backtracking

The naked+hidden-single solver is too weak to provide useful supervision for
84% of puzzles. The point of CoT supervision is to teach the model **harder**
reasoning, not just the rules it could pick up implicitly. So Phase C extends
Phase 1 with a stronger solver that handles backtracking.

There are three distinct ways to encode "backtracking" as supervision; we
recommend **C1** as the most tractable.

## C1 — Per-cell technique → ACT step (recommended)

Use [`tmp_0413/utils/sudoku_scorer.py`](https://github.com/yoyostudy/trm-cr-experiment-photos/blob/main/SOLVER_TRAJECTORY_COT_PLAN.md)
(naked single → hidden single → naked pair → hidden pair → naked triple →
backtrack), but instrument it to record, **per cell**, the technique that
first determined it. Then map technique → ACT step:

| Technique | ACT step bucket |
|---|---|
| `clue` (given) | 0 |
| `naked_single` | 1 |
| `hidden_single` | 2 |
| `naked_pair` | 3 |
| `hidden_pair` | 4 |
| `naked_triple` | 5 |
| (reserved for X-wing / locked / etc. if added) | 6–15 |
| `backtrack` (any depth) | 16 |

**Loss head behaviour:** identical to Phase A's `SolverTrajectoryACTLossHead`
in cumulative mode. ACT step `k` supervises cells with `technique_step ≤ k`.
The "fallback at step 16" is now natural — backtracking-derived cells are
exactly those that need the deepest reasoning, so they fall at the last ACT
iteration.

**Pros**
- Same loss-head architecture as Phase A; minimal new code.
- No `solver_step == -1` cells (backtracking always finishes).
- ACT step 16 is the explicit "use backtracking" bucket → semantic clarity.
- Same compute and training time as Phase A.

**Cons**
- Doesn't expose the *internal structure* of the backtracking trace; the
  model only knows "some cells need backtracking", not how to do it.
- Backtracking cells all share step 16 — no graduated curriculum within
  search-required cells.

**Cost.** 1 day (1–2 h to extend the scorer, then same as Phase A).

## C2 — Wrong-subtree data augmentation (already explored)

Already implemented as `exp_0414_bt_descendents` in the cr_0413 line. For each
hard puzzle, find the canonical backtracking tree and generate examples of
"states reached after a wrong guess, where the model must recognise it must
backtrack". Train on `clue ∪ wrong_subtree_examples`.

**Result so far:** `exp_0414_bt_descendents` reached 0.7222 vs baseline 0.7148
— **+0.7 pp improvement, modest**.

**Pros**
- Already implemented (`tmp_0413/utils/sudoku_with_wrong_child.py`).
- No architectural changes.

**Cons**
- Marginal (+0.7 pp).
- Doesn't combine with Phase A/C1 supervision (it's a data approach, not a
  loss approach — but we could in principle stack).

## C3 — Explicit trace prediction (most ambitious)

Each ACT step predicts one trace element: `(action, cell, value)` where
`action ∈ {guess, propagate, contradict, undo}`. The supervision target at
step `k` is the `k`-th element of the canonical backtracking trace.

```
target_k = trace[k]   # e.g. ("guess", cell=27, value=4)
loss_k = CE(action_k, trace[k].action) + CE(cell_k, trace[k].cell)
       + CE(value_k, trace[k].value)
```

**Pros**
- Most LLM-CoT-like. Model truly learns reasoning trajectory.
- Explicit "undo" supervision — model sees the failed branches.

**Cons**
- Variable trace length (5 → 1000+ steps). Need to compress to
  `halt_max_steps = 16` somehow.
- Requires architecture changes (new prediction heads for action / cell / value).
- 5+ days to implement and debug.
- High risk of failure: very different from baseline TRM training.

## C4 — Reverse-trace / diffusion-style supervision (recommended)

Construct the trace by walking **backwards** from the GT solution, dropping
one filled cell at a time until only clues remain. Reverse the sequence:
the model is supervised on incrementally-filled-in states, like a denoising
diffusion model.

```python
def construct_recall_trace(puzzle_clues, gt):
    state = gt.copy()                    # solved state
    trace = [state.copy()]
    while state != puzzle_clues:
        cell = random_choice(non_clue_cells_filled_in_state)
        state[cell] = unfilled
        trace.append(state.copy())
    return list(reversed(trace))         # forward: clues → ... → solution
```

### Hybrid with C1: deterministic-easy + random-hard ordering

A pure random reverse-trace gives no semantic ordering — the model could
just be learning "fill in K random cells per ACT step". To fix this, combine
with C1's technique-based curriculum:

```python
def hybrid_trace(puzzle, gt, strong_solver):
    techniques = strong_solver.run(puzzle)              # per-cell technique
    cell_to_step = {}
    
    # Easy cells: deterministic ACT step from technique difficulty
    EASY_TECH_TO_STEP = {
        'clue': 0, 'naked_single': 1, 'hidden_single': 2,
        'naked_pair': 3, 'hidden_pair': 4, 'naked_triple': 5,
    }
    for cell, tech in techniques.items():
        if tech in EASY_TECH_TO_STEP:
            cell_to_step[cell] = EASY_TECH_TO_STEP[tech]
    
    # Hard cells (backtrack-derived or solver-failed):
    # randomly distribute across late ACT steps 6-16
    hard_cells = [c for c in range(81) if c not in cell_to_step]
    random.shuffle(hard_cells)                          # ← key change
    n_hard = len(hard_cells)
    n_late_steps = 16 - 6 + 1
    bucket_size = max(1, math.ceil(n_hard / n_late_steps))
    for i, cell in enumerate(hard_cells):
        cell_to_step[cell] = 6 + i // bucket_size
    
    return cell_to_step                                 # 81 entries
```

### Variable-length normalization

Different puzzles have different empty-cell counts (25–56). To make all
puzzles use exactly 16 ACT steps, scale the trace position:

```
cell_to_step_normalized = ceil(position_in_trace * 16 / num_empty_cells)
```

This ensures every puzzle's trace fills [step 1, step 16] proportionally.

### Pros vs C1 / C2 / C3

| | C1 | C2 | C3 | **C4 (this)** |
|---|---|---|---|---|
| Every cell has a step | ✗ (backtrack all step 16) | n/a | partial | **✓** |
| Semantic ordering | ✓ | n/a | ✓ (trace) | partial (easy cells fixed, hard cells random) |
| Architecture change | none | none | yes | **none** |
| Implementation cost | 1 day | done | 5+ days | **1–2 days** |
| Per-epoch data augmentation | none | none | none | **✓ (random shuffle)** |
| Loss-head compatibility | reuse Phase A | none | new heads | **reuse Phase A** |

### Pros (C4 specific)

- Every cell appears in the trace; no "step 16 dump" of backtrack cells.
- Deterministic for easy cells (model learns rules cleanly), random for hard
  cells (model learns to fill in arbitrary order — robust to ordering).
- Different shuffle each epoch ⇒ implicit data augmentation.
- Like diffusion: model learns to denoise increasingly-sparse states; this
  is a well-established training paradigm.

### Cons

- Random ordering of hard cells provides *less* explicit reasoning structure
  than C3's full trace prediction.
- Two hyperparameters (shuffle seed, normalization). Less reproducible
  than C1.

## Recommended progression (revised)

```
1. Phase C4 (1–2 days)
   - extend sudoku_scorer.py with per-cell technique tracking
   - implement hybrid_trace (deterministic-easy + random-shuffled-hard)
   - normalize trace length to 16 ACT steps
   - train one run on cr_0506_solver_traj_cot_c4_smoke
   - compare to baseline + derive_ignore family

2. If C4 ≥ baseline: claim victory.
   If C4 < baseline:
     - try C1 (deterministic only) to isolate whether random shuffle hurts
     - try C3 (full trace) if a deeper change seems needed
   - C2 (wrong-subtree augmentation) is a cheap orthogonal stack-on.
```

C4 is the recommended starting point because it (a) covers every cell,
(b) keeps the same architecture, (c) provides natural data augmentation,
and (d) is computationally identical to baseline training.
