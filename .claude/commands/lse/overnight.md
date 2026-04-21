---
description: "Autonomous overnight run of Phase IV (A → B → C). Plans, implements, verifies, reviews, and gates each sub-phase without human intervention. Usage: /lse:overnight [--resume] [--phase IV-A|IV-B|IV-C] [--skip-review] [--dry-run]"
argument-hint: "[--resume] [--phase IV-A|IV-B|IV-C] [--skip-review] [--dry-run]"
---

# /lse:overnight

Autonomous end-to-end orchestration of Phase IV. The user starts this
command and walks away. It runs through plan → implement → verify →
review → gate for each sub-phase in sequence: IV-A → IV-B → IV-C.

Arguments: `$ARGUMENTS`
- `--resume`: Resume from the last checkpoint in `tasks/overnight_checkpoint.json`.
- `--phase IV-A|IV-B|IV-C`: Start from a specific sub-phase (default: IV-A).
- `--skip-review`: Skip Codex review gates (for faster iteration; NOT for final runs).
- `--dry-run`: Plan all phases but do not dispatch implementation agents.

---

## 0. Overnight invariants

1. **No human questions.** Do not use AskUserQuestion. If the planner
   produces Open questions, log them to `tasks/overnight_log.md` and
   resolve with the most conservative default (safest, spec-compliant
   choice). Document every auto-resolution.
2. **Checkpoint after every significant action.** Write state to
   `tasks/overnight_checkpoint.json` after each task completion,
   verification, review, and gate check. If the session crashes, `--resume`
   picks up from the last checkpoint.
3. **Failure budget per phase.** Each sub-phase gets a failure budget of
   3 task-level failures. A "failure" is: subagent returns incomplete/error,
   OR verification fails after one retry. On budget exhaustion, stop the
   phase, log the state, and do NOT proceed to the next phase.
4. **Gate-or-stop.** Each sub-phase has an exit gate. If the gate fails,
   stop and write a detailed report. Do NOT proceed to the next phase.
5. **No spec deviation.** If a spec ambiguity cannot be resolved
   conservatively, stop and log. Do not guess.
6. **Additive only.** Never overwrite Phase I/II/III outputs.
7. **Full logging.** Every action, decision, and outcome is appended to
   `tasks/overnight_log.md` with timestamps.

---

## 1. Initialization

1. Read `tasks/lessons.md` — lessons take precedence over all heuristics.
2. Read `tasks/overnight_checkpoint.json` if `--resume` was passed.
   Otherwise initialize a fresh checkpoint:
   ```json
   {
     "started_at": "<ISO timestamp>",
     "current_phase": "IV-A",
     "phase_status": {"IV-A": "pending", "IV-B": "pending", "IV-C": "pending"},
     "task_queue": [],
     "completed_tasks": [],
     "failed_tasks": [],
     "failure_count": 0,
     "failure_budget": 3,
     "gate_results": {},
     "codex_sessions": [],
     "last_checkpoint": "<ISO timestamp>"
   }
   ```
3. Write header to `tasks/overnight_log.md`:
   ```
   # Overnight Run — Phase IV
   Started: <timestamp>
   Arguments: $ARGUMENTS
   ---
   ```
4. Create git branch `phase-iv/overnight` from current HEAD if not
   already on it.

---

## 2. Phase pipeline (repeat for IV-A, IV-B, IV-C)

For each sub-phase `P` in sequence:

### 2.1 Plan

1. Read the spec: `docs/specs/phase_IV_{A,B,C}_*.md`.
2. Spawn the `planner` subagent with the spec, current `tasks/todo.md`,
   and `tasks/lessons.md`. Include this instruction in the brief:
   > "This is an autonomous overnight run. Do NOT produce Open questions
   > that require human input. If a spec detail is ambiguous, choose the
   > most conservative interpretation and note your choice in the plan.
   > Tag every task for dispatch."
3. The planner writes tasks to `tasks/todo.md`.
4. **Auto-approve the plan.** Log: `"[AUTO-APPROVE] Phase {P} plan: {N} tasks, {M} groups"`.
5. Parse the task list into an ordered queue respecting dependencies.
   Store in checkpoint `task_queue`.

### 2.2 Implement + Verify loop

For each task `T` in the queue:

1. **Pre-check**: If `T` is blocked by an incomplete task, skip and
   re-queue at the end. If re-queued more than twice, count as failure.
2. **Dispatch**: Map tag → role via `AGENTS.md §4`. Compose the brief
   per `/lse:implement` protocol. Spawn the subagent.
   - Use `isolation: "worktree"` for tasks in parallelizable groups.
   - For parallelizable groups: dispatch ALL tasks in the group
     simultaneously (multiple Agent tool calls in one message), then
     collect results.
3. **Subagent returns**:
   - If report says "complete" with artifacts → proceed to verify.
   - If report says "incomplete" or has errors → log failure, increment
     `failure_count`, continue to next task.
   - If report has Open questions that can be resolved conservatively →
     resolve inline, re-dispatch with resolution. Counts as the retry.
4. **Verify**: Spawn `verifier` subagent.
   - `PASS` → mark task complete in `tasks/todo.md` and checkpoint.
   - `FAIL` → **one retry**: re-dispatch the implementation subagent
     with the failure report as context. Re-verify.
     - Second `PASS` → mark complete.
     - Second `FAIL` → log failure, increment `failure_count`.
5. **Budget check**: If `failure_count >= failure_budget`, stop the
   phase. Write `phase_status[P] = "failed"` to checkpoint. STOP.
6. **Checkpoint**: Update `tasks/overnight_checkpoint.json`.

### 2.3 Full verification

After all tasks complete:

1. Run `/lse:verify --full` (spawns verifier with slow tests + paired
   overhead ratios).
2. On `FAIL`: attempt to fix via one round of targeted `/lse:implement`
   on each failing check. Re-verify once. If still failing, stop.
3. On `PASS`: commit to `phase-iv-{a,b,c}/closing`.

### 2.4 Codex review (unless `--skip-review`)

1. Run `/codex:review --base main --background`. Capture session id.
2. Run `/codex:adversarial-review --base main --background "<focus>"`.
   Use the focus string from `.claude/commands/lse/review.md` for phase `P`.
3. Poll `/codex:status` with 60-second intervals. Log each poll.
4. On completion: `/codex:result <id>` for each. Save to
   `results/processed/codex_reviews/phase_IV_{A,B,C}/`.
5. Spawn `review-triage` subagent.
6. **Automated triage handling**:
   - `BLOCKER`: Route through `/lse:implement` with the BLOCKER
     description. Verify. If fix fails, stop the phase.
   - `MAJOR`: Route through `/lse:implement`. Verify. If fix fails,
     log but do NOT stop (MAJORs don't gate).
   - `MINOR` / `NIT`: Log only. Do not fix during overnight run.
7. If any BLOCKERs were found and fixed, re-run Codex review ONCE.
   If new BLOCKERs appear on re-review, stop.

### 2.5 Gate check

Each sub-phase has a specific gate:

**IV-A activation gate** (spec §13):
Run `scripts/overnight/check_gate.py --phase IV-A` which verifies:
- Phase III compatibility tests pass.
- Activation search has frozen a suite in `selected_tasks.json`.
- Counterfactual replay shows certified activation:
  - `mean_abs_u >= 5e-3` on at least one family.
  - `frac(|u| >= 5e-3) >= 10%` on at least one family.
- All mandatory artifacts exist.

**IV-B translation gate** (spec §14):
Run `scripts/overnight/check_gate.py --phase IV-B` which verifies:
- Activation gate conditions still hold.
- All matched comparisons complete (classical, safe-zero, safe-nonlinear).
- Diagnostic-strength sweep complete.
- Translation analysis 5-step pipeline complete.
- Honest nulls reported where activation didn't translate.

**IV-C completion gate** (spec §14):
Run `scripts/overnight/check_gate.py --phase IV-C` which verifies:
- All advanced estimator variants tested.
- State-dependent scheduler comparison complete.
- Geometry-priority DP comparison complete.
- All 7 ablation types run.
- Attribution analysis complete (which mechanism drove gains).

**Gate result handling:**
- `PASS` → set `phase_status[P] = "complete"`, proceed to next phase.
- `FAIL` → set `phase_status[P] = "gate_failed"`, log details, STOP.
  Do NOT proceed to the next sub-phase.

---

## 3. Completion

After all three sub-phases (or on any stop):

1. Write final summary to `tasks/overnight_log.md`:
   ```
   ## Final Report
   Completed: <timestamp>
   Duration: <hours>

   Phase IV-A: <status> (<N> tasks, <M> failures)
   Phase IV-B: <status> (<N> tasks, <M> failures)
   Phase IV-C: <status> (<N> tasks, <M> failures)

   Gate results:
     IV-A activation: <PASS/FAIL> — <detail>
     IV-B translation: <PASS/FAIL> — <detail>
     IV-C completion: <PASS/FAIL> — <detail>

   Codex reviews: <count> rounds, <BLOCKER/MAJOR/MINOR counts>

   Auto-resolved questions: <count>
     1. <question> → <resolution>
     ...

   Unresolved failures:
     1. <task> — <error>
     ...

   Artifacts produced:
     - results/weighted_lse_dp/phase4/...
     - figures/phase4/...
   ```
2. Write final checkpoint with `"finished_at"` timestamp.
3. If all three phases passed their gates, create a summary commit on
   `phase-iv/overnight`.

---

## 4. Parallelization strategy

Within each sub-phase, dispatch independent tasks in parallel:

- **IV-A parallel groups**: (1) audit + negative-control replay, (2) task
  families (each env-builder call independent), (3) geometry modules
  (independent of task families), (4) tests (after implementations).
- **IV-B parallel groups**: (1) DP experiments across task families, (2)
  RL experiments across task families, (3) diagnostic sweep (after main
  experiments), (4) analysis + figures (after aggregation).
- **IV-C parallel groups**: (1) SafeDoubleQ + SafeTargetQ + SafeTargetExpSARSA
  (independent algorithms), (2) state-dependent schedulers, (3) geometry-
  priority DP, (4) ablations (after baselines), (5) analysis + figures.

Use `isolation: "worktree"` for all parallel dispatches. Merge worktrees
sequentially with verification between merges.

---

## 5. Recovery protocol (`--resume`)

1. Read `tasks/overnight_checkpoint.json`.
2. Determine the last completed action.
3. Resume from the next action. Specifically:
   - If mid-implementation: re-dispatch the current task (idempotent stubs).
   - If mid-verification: re-run verification.
   - If mid-review: re-check `/codex:status` for pending jobs.
   - If at a gate: re-run the gate check.
4. Failure counts carry over from the checkpoint.

---

## 6. Non-negotiables

- This command NEVER pushes to remote. All work is local branches.
- This command NEVER modifies Phase I/II/III result files.
- This command NEVER skips verification (even with `--skip-review`).
- This command writes to `tasks/overnight_log.md` before every
  significant action (pre-log) and after (post-log with outcome).
- If the session is about to hit context limits, checkpoint and stop
  cleanly rather than producing degraded output.
