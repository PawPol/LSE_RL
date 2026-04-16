# LSE_RL — Selective Temporal Credit Assignment via Weighted-LSE Bellman Operators

Research repository for the NeurIPS submission *Selective Temporal Credit Assignment*
and its empirical program built on top of [MushroomRL](https://mushroomrl.readthedocs.io/).

Remote: [`github.com/PawPol/LSE_RL`](https://github.com/PawPol/LSE_RL)

---

## Repository layout

```
LSE_RL/
├── paper/              # LaTeX source, bib, compiled PDF, revision notes
├── docs/
│   └── specs/          # Phase I / II / III experimental specifications
├── tasks/              # Active todo list and cumulative lessons
├── src/lse_rl/         # Our research package (algorithms, envs, experiments, analysis, utils)
├── mushroom-rl-dev/    # Vendored MushroomRL fork — the RL framework we build on
├── experiments/        # Experiment entry points and configs (hydra/yaml)
│   └── configs/
├── results/            # Run outputs
│   ├── raw/            # untouched per-run artifacts (.json, .npz, logs)
│   └── processed/      # aggregated tables, calibration data
├── figures/            # Paper-ready figures
├── notebooks/          # Analysis notebooks
├── scripts/            # One-off utilities, launchers, sweep drivers
├── tests/              # Unit tests for `lse_rl`
├── .claude/
│   └── agents/         # Agent specifications (to be filled in)
├── .github/
│   └── workflows/      # CI (to be added)
├── AGENTS.md           # Agent roster and protocols (stub — pending spec)
├── CLAUDE.md           # Working instructions for Claude in this repo
├── README.md
├── .gitignore
└── pyproject.toml      # `lse_rl` package config (editable install)
```

## Paper

The active manuscript, bibliography, compiled PDF, and revision notes live in
[`paper/`](paper/). The core contribution is an operator-level change to the
Bellman recursion (a weighted log-sum-exp operator), contrasted at the operator
level against adjacent methods (risk-sensitive RL, sparse-reward shaping,
non-stationarity, representation learning, data regularization).

## Code

Our research code lives in [`src/lse_rl/`](src/lse_rl/) and is installed as an
editable Python package. It imports from the vendored
[`mushroom-rl-dev/`](mushroom-rl-dev/) rather than pip's `mushroom-rl`, so all
framework edits (if any) are reproducible from a single commit.

Per the Phase specs, **prefer adding new modules/classes in `src/lse_rl/` over
editing stable MushroomRL code** unless the edit is clearly justified.

## Experiment phases

Design documents in [`docs/specs/`](docs/specs/):

- `phase_I_classical_beta0_experiments.md` — classical (β=0) baselines, harness,
  calibration-ready logging.
- `phase_II_stress_test_beta0_experiments.md` — stress tests of the β=0 harness.
- `phase_III_safe_weighted_lse_experiments.md` — weighted-LSE operator
  experiments with safety constraints.

## Working conventions

See [`CLAUDE.md`](CLAUDE.md) for the full protocol. Highlights:

- Plan-first: write to `tasks/todo.md` before non-trivial changes.
- Capture corrections in `tasks/lessons.md`.
- Verification before done: tests, smoke runs, metric sanity checks.
- One bundled PR over many churny splits unless the user says otherwise.

## Status

Project structure initialized 2026-04-16. Agent specifications pending.
