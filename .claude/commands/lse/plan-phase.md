---
description: Spawn the planner subagent on a given phase spec and produce tasks/todo.md entries. Usage: /lse:plan-phase I | II | III | IV-A | IV-B | IV-C
argument-hint: <I | II | III | IV-A | IV-B | IV-C>
---

# /lse:plan-phase

Arguments: `$ARGUMENTS` — one of `I`, `II`, `III`, `IV-A`, `IV-B`, `IV-C`
(or a spec path).

## Protocol

1. Read `docs/specs/phase_$ARGUMENTS_*.md`. If no such file exists, stop
   and ask the user.
2. Read `tasks/todo.md` and `tasks/lessons.md` — lessons take precedence
   over planning heuristics.
3. Spawn the `planner` subagent via the Agent tool with
   `subagent_type: "planner"`, passing:
   - the spec path,
   - the current todo state,
   - instruction to follow `.claude/agents/planner.md`.
4. On return, present the planner's checklist to the user for
   approval BEFORE dispatching any implementation agent. Surface all
   "Open questions" from the handoff report.

## Output format to the user

```
Phase $ARGUMENTS plan drafted. <N> tasks, <M> parallelizable groups,
<K> open questions requiring your input before we start.

Open questions:
  1. ...
  2. ...

Reply "approve" to proceed, or edit tasks/todo.md and reply "approve".
```

Do NOT dispatch implementation agents from this command.
