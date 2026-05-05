# Combined: `cr_0429_derive_ignore (2-seed avg)` (n=2 seeds)

Two independent runs of the same configuration (Plan C derive_ignore, default solver).
`cr_0429_derive_ignore.yaml` and `cr_0429_derive_ignore_v2.yaml` differ only in `name`
and `num_epochs` — the actual loss / arch / batch_size are identical.

## Member runs
- `cr_0429_derive_ignore_194437` (seed_1)
- `cr_0429_derive_ignore_v2_154835` (seed_2)

## Best exact_acc @ halt_or_max (mean ± std)

| nsup | seed_1 | seed_2 | mean ± std |
|---|---|---|---|
| 16 | 0.6181 | 0.6154 | **0.6168 ± 0.0018** |
| 32 | 0.6811 | 0.6769 | **0.6790 ± 0.0030** |
| 64 | 0.7193 | 0.7135 | **0.7164 ± 0.0041** |

## Training accuracy (last-1000 avg)

- seed_1: 0.8956,  seed_2: 0.9129
- mean ± std: **0.9042 ± 0.0122**

## Plots (`training_dynamics/`)

Line = mean across 2 seeds; shaded band = ± 1 std interpolated on common step grid.
