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
│   ├── agents/         # Role subagent specs (planner, env-builder, ...)
│   └── commands/lse/   # Project slash commands (/lse:plan-phase, /lse:verify, ...)
├── .codex/
│   └── config.toml     # Pinned Codex model + reasoning effort for reviews
├── .github/
│   └── workflows/      # CI (to be added)
├── AGENTS.md           # Orchestration protocol (authoritative)
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

## Orchestration

Claude Code is the main orchestrator. Role-specialized subagents in
[`.claude/agents/`](.claude/agents/) do the implementation work;
Codex (via the [codex-plugin-cc](https://github.com/openai/codex-plugin-cc))
acts as read-only reviewer and second code tester at phase boundaries.

See [`AGENTS.md`](AGENTS.md) for the authoritative protocol and
[`docs/workflow.md`](docs/workflow.md) for the end-to-end lifecycle
diagram.

### Project slash commands (`/lse:*`)

| Command                            | Purpose |
|------------------------------------|---------|
| `/lse:plan-phase <I\|II\|III>`     | Spawn `planner` on a phase spec; decompose into `tasks/todo.md`. |
| `/lse:implement <task-id>`         | Route a tagged task to its role subagent per `AGENTS.md §4`. |
| `/lse:verify [--full]`             | Run the `verifier` subagent (tests + diffs + schema checks). |
| `/lse:review <I\|II\|III>`         | Phase-boundary Codex gate: `/codex:review` + `/codex:adversarial-review` in background, then `review-triage`. |
| `/lse:status`                      | Read-only status: todo progress + active Codex jobs + quarantined runs. |

### Codex plugin setup (once per machine)

```
/plugin marketplace add openai/codex-plugin-cc
/plugin install codex@openai-codex
/reload-plugins
/codex:setup
```

The repo-scoped Codex model + reasoning effort are pinned in
[`.codex/config.toml`](.codex/config.toml).

### Model policy

All Claude subagents use `claude-opus-4-6` (set in frontmatter). Codex
uses the strongest available GPT-5 variant at
`model_reasoning_effort = "high"`. Upgrade happens via a single-file
PR on the pins.

## Status

- 2026-04-16: Project structure initialized.
- 2026-04-16: Orchestration authored — 10 role subagents, 5 `/lse:*`
  slash commands, Codex gate at phase boundaries.
