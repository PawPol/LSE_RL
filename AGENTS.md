# AGENTS.md — Orchestration protocol for `LSE_RL`

This document is authoritative. Every agent (human or LLM) operating in this
repository follows it. Subagent specs live under `.claude/agents/`; custom
slash commands live under `.claude/commands/`. The end-to-end workflow
diagram is in `docs/workflow.md`.

---

## 1. Architecture

```
          ┌────────────────────────────────────────────────────────────┐
          │                MAIN ORCHESTRATOR (Claude Code)              │
          │  - Reads specs, maintains tasks/todo.md, sequences work    │
          │  - Dispatches to role subagents (one task per subagent)    │
          │  - Invokes Codex gates at phase boundaries                 │
          │  - Enforces verification before "done"                     │
          └────────────────────────────────────────────────────────────┘
                    │                      │                        │
        ┌───────────┼───────────┐   ┌──────┴──────┐          ┌──────┴──────┐
        ▼           ▼           ▼   ▼             ▼          ▼             ▼
   planner   env-builder   algo-impl   operator-theorist   calibration  experiment
                                                           -engineer     -runner
        ▼           ▼           ▼             ▼             ▼             ▼
   test-author   plotter-analyst   verifier   review-triage   (Codex via /codex:*)
```

All Claude subagents run `claude-opus-4-6`. Codex runs its strongest
available model at `model_reasoning_effort = "high"` — see `docs/workflow.md`
for the `.codex/config.toml` stanza.

## 2. Subagent roster

Each subagent has a single, narrow scope. Tool allowlists are tight — a
plotter cannot run pytest, a verifier cannot edit code. Full specs in
`.claude/agents/<name>.md`.

| Subagent              | Scope                                                       | Edits code | Runs tests | Typical phase |
|-----------------------|-------------------------------------------------------------|-----------|-----------|---------------|
| `planner`             | Spec → `tasks/todo.md` decomposition                        | No        | No        | All           |
| `env-builder`         | Environments, wrappers, stress families, activation tasks   | Yes       | No        | I, II, IV-A   |
| `algo-implementer`    | Classical DP/TD (I) + safe DP/TD (III) + stabilized RL (IV-C) | Yes     | No        | I, III, IV-C  |
| `operator-theorist`   | Safe LSE operator, certification math, `logaddexp`, geometry| Yes       | No        | III, IV-A     |
| `calibration-engineer`| Calibration emission (I/II) + schedule build (III) + v3 natural-shift calibration (IV-A) + state-dependent schedulers (IV-C) | Yes | No | I, II, III, IV-A, IV-C |
| `experiment-runner`   | Configs, run dispatch, result aggregation                   | Yes       | No        | All           |
| `test-author`         | Unit, integration, smoke tests per spec                     | Yes       | No        | All           |
| `plotter-analyst`     | Figures + paper tables + analysis notebooks                 | Yes       | No        | All           |
| `verifier`            | Runs tests, diffs behavior, metric sanity checks (read-only)| No        | Yes       | All (final gate) |
| `review-triage`       | Consumes Codex output → actionable tasks in `todo.md`       | No        | No        | Phase boundaries |

## 3. Main-orchestrator invariants

1. **Plan before dispatch.** Every phase starts with the `planner` subagent
   writing a checklist to `tasks/todo.md`. Implementation subagents are
   not invoked until the plan is approved by the user.
2. **One task per subagent.** Do not batch unrelated work into a single
   subagent call. Narrow scope → cleaner diffs → faster verification.
3. **Parallelize in worktrees.** Independent tasks (e.g. Phase II stress
   families, Phase III ablation sweeps) are dispatched with
   `isolation: "worktree"`. A single shared tree is used only for
   strictly sequential work.
4. **Verification gate before done.** `verifier` must pass
   (tests + diffs + metric sanity) before any todo item is ticked.
5. **Codex gate at phase boundaries.** `/codex:review --background` and
   `/codex:adversarial-review --background` are mandatory before closing
   a phase. The orchestrator polls with `/codex:status`, pulls with
   `/codex:result`, and triages through `review-triage`.
6. **Spec is load-bearing.** If a spec in `docs/specs/` conflicts with
   observed behavior, STOP and ask the user. Do not silently resolve.
7. **Record corrections.** Every user correction or self-found mistake
   appends a lesson to `tasks/lessons.md` (template in that file).

## 4. Dispatch decision table

Given a task in `tasks/todo.md` tagged `[X]`, route as follows:

| Tag prefix                  | Role subagent          |
|-----------------------------|------------------------|
| `[infra]`                   | `planner` or `experiment-runner` (split by sub-task) |
| `[env]`, `[stress-design]`  | `env-builder`          |
| `[algo]`, `[algo-integration]` | `algo-implementer`  |
| `[operator]`, `[safety]`    | `operator-theorist`    |
| `[calibration]`, `[calibration-prep]` | `calibration-engineer` |
| `[logging]`                 | `experiment-runner` (callbacks, schemas) or `algo-implementer` (instrumentation fields) |
| `[test]`                    | `test-author`          |
| `[plot]`, `[analysis]`      | `plotter-analyst`      |
| `[ablation]`                | `experiment-runner` (+ `plotter-analyst` for readout) |
| `[geometry]`                | `operator-theorist` (natural-shift, activation metrics) |
| `[activation]`              | `operator-theorist` (search pipeline) + `env-builder` (task families) |
| `[counterfactual]`          | `experiment-runner` (replay dispatch) + `operator-theorist` (target math) |
| `[translation]`             | `experiment-runner` (run matrix) + `plotter-analyst` (analysis) |
| `[stabilization]`           | `algo-implementer` (SafeDoubleQ, SafeTargetQ, SafeTargetExpectedSARSA) |
| `[scheduler]`               | `calibration-engineer` (state-dependent, smoothing, backoff) |
| `[priority-dp]`             | `algo-implementer` (geometry-prioritized async DP) |
| `[audit]`                   | `operator-theorist` (Phase III compatibility audit) |
| `[spec-read]`               | `planner`              |

## 5. Codex integration

Codex-plugin-cc is the read-only reviewer and the second code tester.

### 5.1. Install (run once in Claude Code)

```
/plugin marketplace add openai/codex-plugin-cc
/plugin install codex@openai-codex
/reload-plugins
/codex:setup
```

Also create `.codex/config.toml` at repo root (see `docs/workflow.md`) to
pin the strongest Codex model with high reasoning effort.

### 5.2. Phase-boundary protocol

When closing phase `P` (= I, II, or III):

1. `verifier` passes locally (tests + smoke runs + metric sanity).
2. Commit WIP branch `phase-P/closing`.
3. `/codex:review --base main --background` — structural review against main.
4. `/codex:adversarial-review --base main --background "challenge the
   <phase-specific focus>"`. Focus strings:
   - Phase I: "challenge the finite-horizon DP and calibration-logging
     correctness; flag any silent math or schema drift from the spec."
   - Phase II: "challenge whether stress families actually isolate the
     weakness we claim, and whether event logging supports the Phase III
     calibration contract."
   - Phase III: "challenge operator correctness, certification-box
     invariance, β=0 collapse, and numerical stability of logaddexp."
   - Phase IV-A: "challenge Phase III backward compatibility, natural-shift
     geometry correctness, activation-search ex-ante purity (no safe-return
     leakage), counterfactual replay isolation, and matched-control
     completeness per docs/specs/phase_IV_A_activation_audit_and_counterfactual.md."
   - Phase IV-B: "challenge whether activation diagnostics actually translate
     into outcome improvements, whether matched-control pairing is valid,
     whether diagnostic-strength sweep is monotone, and whether null results
     are reported honestly per docs/specs/phase_IV_B_translation_experiments.md."
   - Phase IV-C: "challenge estimator-stabilization correctness (dual tables,
     target networks), state-dependent scheduler freeze discipline, geometry
     priority scoring, ablation isolation, and whether gains are attributable
     to the claimed mechanism per
     docs/specs/phase_IV_C_advanced_stabilization_and_geometry_ablations.md."
5. Poll with `/codex:status`. When both jobs complete, `/codex:result`
   each and pipe into `review-triage`.
6. `review-triage` writes actionable entries to `tasks/todo.md`, grouped
   by severity (BLOCKER / MAJOR / MINOR / NIT). BLOCKERs must be
   resolved before the phase is closed.
7. Re-run `verifier`. Merge the closing branch only after
   BLOCKER == ∅ and `verifier` is green.

### 5.3. Ad-hoc Codex use

Outside phase boundaries, Codex is called on-demand whenever a change
touches any of:

- Operator math (`src/lse_rl/algorithms/*/safe_*.py`)
- Certification / clipping (`B̂_t`, `κ_t`, `β_cap`)
- Calibration pipeline (`experiments/weighted_lse_dp/calibration/`)
- Logging schemas (`results/**` schema headers)

Prefer `/codex:review` (read-only). Use `/codex:rescue` only when we
want Codex to take over a concrete, well-specified fix — never to
delegate design.

## 6. Worktree discipline

- Worktrees are created at dispatch time via `isolation: "worktree"`.
- Branch naming: `phase-<P>/<role>/<short-slug>`, e.g.
  `phase-iii/operator-theorist/safe-target-closed-form`,
  `phase-iv-a/operator-theorist/natural-shift-geometry`,
  `phase-iv-b/experiment-runner/translation-rl`,
  `phase-iv-c/algo-implementer/safe-double-q`.
- The orchestrator is responsible for merging worktree branches back
  into the current working branch. Verifier runs on the merged branch,
  not on the worktree.
- If a worktree produces no changes (agent concluded no edit needed),
  the orchestrator records that finding in `tasks/todo.md` and discards
  the worktree.

## 7. Handoff contract

Every subagent, on completion, returns a structured report:

```
## Summary
- What was done (1-3 bullets)

## Artifacts
- paths created / modified

## Verification evidence
- tests run, key outputs, or "verifier needed" if out of scope

## Open questions
- anything that blocks the next step
```

The orchestrator merges this into `tasks/todo.md` and, where
appropriate, promotes "Open questions" to user-visible clarification
prompts.

## 8. Overnight autonomous mode

For unattended Phase IV execution, `/lse:overnight` wraps the full
plan → implement → verify → review → gate lifecycle. See
`docs/workflow.md §7` for details. Key additions:

- Checkpoint state machine: `scripts/overnight/checkpoint.py`
- Gate checks: `scripts/overnight/check_gate.py`
- Auto-resolution of open questions (conservative defaults, logged)
- Failure budget: 3 per sub-phase, then stop
- Sequential gates: IV-A activation → IV-B translation → IV-C completion

The overnight command follows all invariants in §3 with these
overrides: plan auto-approval (§3.1), automated BLOCKER routing
(§5.2 step 6), and gate-based phase transitions (replacing user
merge approval).

## 9. When this document changes

AGENTS.md, subagent specs, and slash commands are themselves
spec-level artifacts. Changes require:

1. A clear motivating incident (usually a lesson in `tasks/lessons.md`).
2. A pass through `/codex:adversarial-review` on the diff before merging.
