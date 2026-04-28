# trm-cr-experiment-photos

Experiment tracking — one branch per run, panels rendered into a shared Google Sheet.

## Workflow

1. **Branch per experiment** — `branch_name = run.name`
2. **Pull from W&B** → matplotlib → `training_dynamics/<panel>.png` on the branch
3. **Index in `experiments.csv` on `main`** → Google Sheet auto-syncs and renders images via `=IMAGE(URL)`
