---
description: Dispatch a tagged task from tasks/todo.md to the correct role subagent, using the dispatch table in AGENTS.md §4. Usage: /lse:implement <task-id-or-line-number> [--worktree]
argument-hint: <task-id-or-line-number> [--worktree]
---

# /lse:implement

Arguments: `$ARGUMENTS` — a task identifier (line number or slug) from
`tasks/todo.md`. Optional `--worktree` flag forces `isolation:
"worktree"` on the spawned subagent.

## Protocol

1. Parse `tasks/todo.md`, locate the task line. Read its tag (e.g.
   `[operator]`, `[env]`).
2. Map tag → role subagent via the table in `AGENTS.md §4`. If
   ambiguous, stop and ask the user.
3. Compose a brief for the subagent containing:
   - the exact task line,
   - the cited spec reference (file + section),
   - the handoff contract from `AGENTS.md §7`,
   - any lessons from `tasks/lessons.md` tagged with the same role,
   - any upstream artifacts it depends on (e.g. an `operator-theorist`
     mixin that `algo-implementer` must import).
4. Spawn via Agent tool. Use `isolation: "worktree"` if:
   - `--worktree` was passed, OR
   - the task's "Parallelizable groups" annotation places it in a group
     being executed in parallel with other work.
5. On return, merge the subagent's report into `tasks/todo.md`. Do NOT
   mark the task complete here — completion requires `/lse:verify` to
   pass.
6. If the report contains Open questions, surface them to the user
   before proceeding to the next task.

## Failure handling

- If the subagent returns with an incomplete or failed status, STOP and
  re-plan. Do not auto-retry. Follow workflow principle #1.
