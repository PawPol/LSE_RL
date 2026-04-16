# CLAUDE.md — Working instructions for Claude in `LSE_RL`

These instructions apply to every session in this repository. They are the
local contract between the user (an ML researcher working on NeurIPS/ICML/ICLR
submissions) and any Claude agent operating on the repo.

---

## 1. User profile

- ML researcher with advanced math background (measure theory, optimization,
  stochastic approximation, RL theory).
- Expects technical rigor: consistent notation, explicit assumptions, proof
  sketches when relevant, primary citations.
- Code must be idiomatic Python with type hints, shape annotations on tensor
  operations, reproducibility defaults (seeds, config files), and efficient
  vectorized implementations.

## 2. Workflow orchestration

### Plan mode default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural
  decisions).
- Write the active plan to `tasks/todo.md` as checkable items before acting.
- If something goes sideways, STOP and re-plan immediately — do not push
  through a broken architecture.
- Use plan mode for verification steps, not just building.

### Subagent strategy
- Use subagents liberally to keep the main context clean.
- Offload research, exploration, and parallel analysis to subagents.
- One task per subagent for focused execution.

### Self-improvement loop
- After ANY correction from the user: append to `tasks/lessons.md` using the
  lesson template (pattern + prevention rule + source incident).
- Review `tasks/lessons.md` at session start.
- Ruthlessly iterate until the same mistake stops recurring.

### Verification before done
- Never mark a task complete without proving it works: tests, smoke runs,
  metric sanity checks, diffs against baseline behavior.
- Ask: "Would a staff engineer approve this?"
- For high-stakes work, spawn a verification subagent.

### Elegance (balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant
  solution."
- Skip this for obvious fixes — don't over-engineer.

### Autonomous bug fixing
- Bug reports are actionable, not questions. Point at logs/errors/failing
  tests and resolve them without hand-holding.

## 3. Clarification protocol

- Ask follow-up questions when key details are missing.
- Do NOT assume defaults or silently fill gaps for architectural or
  methodological decisions (operator forms, reward scales, seed counts,
  calibration targets, etc.).
- Simple, obvious choices do not require a question.

## 4. Repository-specific rules

### Package boundaries
- Research code lives in `src/lse_rl/`. Framework code lives in
  `mushroom-rl-dev/`.
- **Prefer adding new modules/classes in `src/lse_rl/` over editing stable
  MushroomRL code** unless the edit is clearly justified and isolated.
- When a MushroomRL edit is unavoidable, record it in `tasks/lessons.md` with
  the justification.

### Phase specs are load-bearing
- `docs/specs/phase_{I,II,III}_*.md` are the implementation contracts for the
  empirical program. Treat them as specs, not suggestions.
- If a spec conflicts with observed code behavior, STOP and re-plan with the
  user before resolving the conflict.

### Reproducibility defaults
- Every experiment entry point takes `--seed` and `--config` (yaml/hydra).
- Raw run artifacts go to `results/raw/<experiment>/<run_id>/` as `.json` +
  `.npz` with a schema header.
- Aggregations go to `results/processed/`.
- Figures in `figures/` are regeneratable from scripts in `scripts/` or
  notebooks in `notebooks/`.

### Logging contract
- Reuse `mushroom_rl.core.logger.Logger` where it helps.
- In addition, every run emits a structured `run.json` (config, env, seed,
  timing, git SHA) and a `metrics.npz` (per-episode arrays). Phase I spec is
  authoritative.

## 5. File handling

- Do not expose internal sandbox paths (`/sessions/...`) to the user. Refer to
  files as "the paper notes" / "the Phase I spec" / "your folder" as
  appropriate.
- Final deliverables go into the user's selected folder (repo root), not the
  scratchpad.

## 6. Memory

- Persistent memory lives at the session-level memory system described in
  the system prompt — NOT in this repo.
- Repo-level persistent notes go in `tasks/lessons.md` (corrections) or
  `docs/` (design notes).

## 7. Agent protocol

`AGENTS.md` defines the roster of specialized agents and their invocation
protocol. **This file is currently a stub** — the user and Claude will
co-author it in a subsequent session. Do not invent agents until then.
