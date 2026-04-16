# tasks/todo.md — Active task list

This file is the single source of truth for the active plan. Update
status in real time. Add a short review section at the end of each
completed task.

---

## Active task: Project structure initialization (2026-04-16)

- [x] Decide on repository layout (vendored MushroomRL, `lse_rl` package,
      specs under `docs/specs/`, git init with PawPol/LSE_RL remote).
- [x] Create top-level directory skeleton.
- [x] Move paper artifacts into `paper/`.
- [x] Move phase specs into `docs/specs/`.
- [x] Write `README.md`, `CLAUDE.md`, `AGENTS.md` (stub),
      `tasks/todo.md`, `tasks/lessons.md`, `.gitignore`, `pyproject.toml`,
      and package `__init__.py` stubs.
- [x] `git init`, add `origin = github.com/PawPol/LSE_RL.git`, stage
      tracked files (NO commit yet — user will review).
- [x] Verify final structure; confirm `mushroom-rl-dev/` is untouched.

### Review (2026-04-16)

- Structure matches the answers from the onboarding question set.
- `mushroom-rl-dev/` left byte-identical (no files added or removed).
- Paper + specs relocated without content changes (only `git mv`-equivalent).
- Git initialized locally on branch `main` with remote `origin` pointing at
  `github.com/PawPol/LSE_RL.git`. Nothing committed or pushed — awaiting
  user sign-off and the agents spec before the first commit.

---

## Active task: Orchestration setup (2026-04-16)

- [x] Parallel recon via Explore subagents: summarize Phase I/II/III
      specs + fetch codex-plugin-cc docs.
- [x] Collect architectural decisions from user: function-aligned
      roster, phase-boundary Codex gate, worktree isolation,
      opus-everywhere.
- [x] Write `AGENTS.md` as authoritative orchestration protocol.
- [x] Write 10 role subagents under `.claude/agents/` (all
      `claude-opus-4-6`): planner, env-builder, algo-implementer,
      operator-theorist, calibration-engineer, experiment-runner,
      test-author, plotter-analyst, verifier, review-triage.
- [x] Write 5 project slash commands under `.claude/commands/lse/`:
      plan-phase, implement, verify, review, status.
- [x] Write `.codex/config.toml` (model = gpt-5.4, reasoning high).
- [x] Write `docs/workflow.md` (lifecycle diagram + Codex integration).
- [x] Update `README.md` with orchestration + Codex setup sections.
- [x] Verify frontmatter validity, cross-reference consistency, path
      existence.

### Review (2026-04-16)

- Roster is function-aligned (10 agents), not phase-aligned.
  Justification: the same role (e.g. `env-builder`) does similar work
  across phases, so knowledge reuse happens inside the agent rather
  than across phase-agents. One task per subagent is preserved via
  `/lse:implement`.
- Every agent frontmatter validated: `name`, `description`, `tools`,
  `model: claude-opus-4-6` all present.
- All 10 agent names referenced in `AGENTS.md` map 1:1 to files under
  `.claude/agents/`.
- Slash commands namespaced correctly as `.claude/commands/lse/<cmd>.md`
  (subdirectory-based namespacing).
- Codex gate is MANDATORY at phase boundaries but not per-commit —
  matches the plugin README's warning that multi-file reviews are slow
  and the review gate can drain usage quickly.
- `mushroom-rl-dev/` remains byte-identical; no edits introduced by
  this task.

---

## Next up (pending user input)

- Review the orchestration artifacts. Adjustments welcome before the
  first commit.
- When approved: `/lse:plan-phase I` to decompose Phase I into
  `tasks/todo.md` entries via the `planner` subagent.
