---
description: Print a compact status view. Combines tasks/todo.md progress, active Codex jobs (/codex:status), and any quarantined failed runs under results/raw/*/_failed/. Usage: /lse:status
---

# /lse:status

Produce a single, read-only status panel so we always know where the
project stands.

## Protocol

1. Read `tasks/todo.md`. Compute per-phase counts: `completed /
   in_progress / pending / blocked`.
2. Run `/codex:status` and capture running / recent jobs with their
   session ids, start times, and targets (`--base`).
3. `ls results/raw/*/_failed/ 2>/dev/null` — report any quarantined
   runs.
4. `git status --short` and `git diff --stat main...HEAD` to summarize
   the working-tree delta.
5. Print:

```
== LSE_RL status ==

Phase I:    <counts>
Phase II:   <counts>
Phase III:  <counts>

Active Codex jobs:
  - <session-id>  <command>  started <t>   base=<ref>
  - ...

Quarantined runs:
  - results/raw/<experiment>/_failed/<run_id>  (reason: <line 1 of traceback>)

Branch delta vs main:
  <git diff --stat tail>

Open questions awaiting user:
  - ... (pulled from the most recent planner/review-triage reports)
```

This command is pure read-only — it NEVER writes to disk or spawns
implementation work.
