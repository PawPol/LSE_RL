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

## Next up (pending user input)

- Co-author `AGENTS.md` and the individual agent specs under
  `.claude/agents/`.
- First real implementation task is Phase I per
  `docs/specs/phase_I_classical_beta0_experiments.md`. Do NOT start until
  the agents spec is in place.
