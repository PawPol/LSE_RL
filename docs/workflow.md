# docs/workflow.md — End-to-end workflow for `LSE_RL`

This document is prose + diagrams for the orchestration defined in
`AGENTS.md`. It is non-normative — `AGENTS.md` is authoritative in any
conflict.

## 1. Lifecycle of a phase

```
   spec (docs/specs/phase_P_*.md)
        │
        ▼
  /lse:plan-phase P
        │
        ▼  (planner writes tasks/todo.md; user approves)
        │
        ▼
  loop over tasks in tasks/todo.md:
      /lse:implement <task>
          │
          ▼  (role subagent, possibly in worktree)
          │
          ▼
      /lse:verify            ◄─── FAIL ──► new todo entry, loop
          │ PASS
          ▼
      tick task in todo.md
  end loop
        │
        ▼
  /lse:verify --full          (final local gate; paired overheads, slow tests)
        │ PASS
        ▼
  commit to phase-P/closing
        │
        ▼
  /lse:review P               (Codex review + adversarial-review, background)
        │
        ▼  review-triage populates tasks/todo.md
        │
        ▼  BLOCKERs ? → route through /lse:implement, re-review
        │
        ▼  BLOCKER == 0
        ▼
  user merges phase-P/closing → main
        │
        ▼
  tasks/todo.md "Review" section added; tasks/lessons.md appended
```

## 2. Codex plugin setup

### 2.1 Installation (once per machine)

```
/plugin marketplace add openai/codex-plugin-cc
/plugin install codex@openai-codex
/reload-plugins
/codex:setup
```

### 2.2 Repository-scoped config

Create `.codex/config.toml` at repo root (this is committed; it does
not contain secrets):

```toml
# .codex/config.toml
# Pinned Codex configuration for LSE_RL reviews.

# Strongest available Codex model for code review.
# Update this value in a single PR when OpenAI releases a stronger one.
model = "gpt-5.5"
model_reasoning_effort = "xhigh"

# Scope: operator math, safety machinery, calibration pipeline, and the
# full test suite are the reviewer's primary targets.
# Everything under mushroom-rl-dev/ is read-only-for-Codex context;
# we never ask Codex to edit the vendored fork.
```

### 2.3 Command cheat-sheet

| Command                         | Blocking | Writes code | Used for |
|---------------------------------|----------|-------------|----------|
| `/codex:review`                 | optional | No          | Structural / correctness pass |
| `/codex:adversarial-review`     | optional | No          | Steerable challenge of a specific claim |
| `/codex:rescue`                 | optional | Yes         | Delegated focused fix, user-invoked only |
| `/codex:status`                 | Yes      | No          | List running + recent jobs |
| `/codex:result <session-id>`    | Yes      | No          | Retrieve final output |
| `/codex:cancel <session-id>`    | Yes      | No          | Cancel a background job |

All review commands support `--background`, `--base <ref>`, `--model
<name>`, `--effort <level>`. We use `--background` by default for
phase-boundary reviews since they touch many files and the plugin
README calls out multi-file reviews as slow.

## 3. Worktree layout

Branches:

- `main` — always green; phases merge here.
- `phase-<P>/closing` — the integration branch for closing phase `P`.
- `phase-<P>/<role>/<slug>` — short-lived worktree branches spawned
  by `/lse:implement --worktree`.

A typical parallel burst in Phase II:

```
phase-ii/closing
├── phase-ii/env-builder/chain-jackpot         (worktree)
├── phase-ii/env-builder/chain-catastrophe     (worktree)
├── phase-ii/env-builder/grid-regime-shift     (worktree)
└── phase-ii/test-author/stress-tasks          (worktree)
```

The orchestrator merges completed worktree branches into
`phase-ii/closing` sequentially, running `/lse:verify` after each
merge, before starting the next merge.

## 4. Result directory contract

```
results/
├── raw/
│   └── <experiment>/
│       └── <run_id>/
│           ├── config.json          # input config, seed, env, git_sha
│           ├── run.json             # timing, host, schema_version
│           ├── metrics.npz          # per-episode arrays, schema_version
│           ├── transitions.npz      # per-step arrays, schema_version
│           ├── calibration_stats.npz
│           ├── timings.json
│           └── stdout.log
├── processed/
│   └── <experiment>/
│       ├── aggregated.parquet
│       ├── tables/P1-A.csv, P1-A.tex, ...
│       └── codex_reviews/phase_<P>/{review, adversarial}.md
```

## 5. Model policy recap

- All Claude subagents: `claude-opus-4-6`, set in each subagent file's
  frontmatter.
- Codex: `model = "gpt-5.5"` with `model_reasoning_effort = "xhigh"`,
  pinned in `.codex/config.toml`.
- Upgrade policy: when Anthropic or OpenAI ships a stronger model,
  update the pins in a single-file PR and run `/lse:verify --full` +
  `/lse:review` on the bump before merging.

## 6. When the workflow fails

If any of these happen, STOP:

1. A subagent returns with `FAIL` and its report lacks a reproducible
   failing command.
2. `/lse:verify` keeps failing on the same check after two fix
   attempts.
3. A Codex BLOCKER cannot be resolved without touching
   `mushroom-rl-dev/` beyond what `tasks/lessons.md` has already
   justified.
4. A spec section appears to contradict observed code behavior.

In all four cases, re-plan with the user. Do not push through.

## 7. Overnight autonomous mode (`/lse:overnight`)

For long-running phase pipelines (Phase IV-A → B → C), the orchestrator
supports fully autonomous execution via `/lse:overnight`.

### 7.1 Key differences from interactive mode

| Aspect                | Interactive              | Overnight                      |
|-----------------------|--------------------------|--------------------------------|
| Plan approval         | User approves            | Auto-approved, logged          |
| Open questions        | Surfaced to user         | Conservative default, logged   |
| Failure handling      | User decides             | Retry once, then failure budget|
| Phase gates           | User reviews             | Automated gate checks          |
| Codex reviews         | User reads triage        | Auto-triage, auto-fix BLOCKERs|
| Merge                 | User merges              | Local branches only, no push   |

### 7.2 Safety mechanisms

1. **Failure budget**: 3 task-level failures per sub-phase. Budget
   exhaustion stops the pipeline.
2. **Gate-or-stop**: Each sub-phase must pass its exit gate before the
   next begins. Failed gates stop the pipeline.
3. **Checkpoint persistence**: State written to
   `tasks/overnight_checkpoint.json` after every action. Crash recovery
   via `--resume`.
4. **Full logging**: Every action, decision, and auto-resolution logged
   to `tasks/overnight_log.md` with timestamps.
5. **No remote push**: All work stays on local branches.
6. **Additive only**: Phase I/II/III outputs are never modified.

### 7.3 Checkpoint state machine

```
scripts/overnight/checkpoint.py  — state tracker (init, get, update,
                                   task-done, task-fail, gate, finish)
scripts/overnight/check_gate.py  — artifact-based gate verification
                                   (IV-A activation, IV-B translation,
                                   IV-C completion)
```

### 7.4 Usage

```
# Full pipeline from IV-A:
/lse:overnight

# Resume after crash:
/lse:overnight --resume

# Start from specific phase (if earlier phases complete):
/lse:overnight --phase IV-B

# Fast iteration without Codex (development only):
/lse:overnight --skip-review

# Plan-only mode:
/lse:overnight --dry-run
```

### 7.5 Gate definitions

| Sub-phase | Gate name          | Key conditions |
|-----------|--------------------|----------------|
| IV-A      | Activation gate    | `selected_tasks.json` non-empty, `mean_abs_u >= 5e-3` and `frac_active >= 10%` on ≥1 family, all audit artifacts present |
| IV-B      | Translation gate   | All matched comparisons complete, diagnostic sweep done, 5-step analysis pipeline done, nulls reported |
| IV-C      | Completion gate    | All 3 estimator variants tested, scheduler + geometry-DP comparisons done, all 7 ablation types run, attribution analysis done |
