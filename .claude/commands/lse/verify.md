---
description: Run the verifier subagent on the current branch. Blocks until PASS/FAIL is returned. Usage: /lse:verify [--full]
argument-hint: [--full]
---

# /lse:verify

Arguments: `$ARGUMENTS` — optional `--full` flag to include
`@pytest.mark.slow` tests and paired overhead-ratio measurements.

## Protocol

1. Spawn the `verifier` subagent via the Agent tool (`subagent_type:
   "verifier"`).
2. Brief it with:
   - the list of files changed since `main` (`git diff --name-only main...HEAD`),
   - the `--full` flag if present,
   - the current phase (inferred from the active section of
     `tasks/todo.md`).
3. Wait for the report. The first line of "Summary" is `PASS` or
   `FAIL: <reason>`.
4. On `PASS`: the caller (usually `/lse:implement` follow-up or the
   user) may tick the relevant items in `tasks/todo.md`.
5. On `FAIL`: the failing checks are surfaced. Do NOT attempt fixes
   from this command; route back through `/lse:implement` with a new
   todo entry citing the failing check.

## Non-negotiables

- This command never writes code.
- This command never modifies `tasks/todo.md`. The orchestrator does
  that after reading the verdict.
- Verifier evidence (command output, diffs) is preserved in the chat
  so the user can audit what was actually checked.
