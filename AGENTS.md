# AGENTS.md — Agent roster and protocols

**Status: STUB. Do not invent agents until this file is co-authored with the user.**

This document will define:

1. **Agent roster** — the named specialized agents used in this repo
   (e.g. a researcher agent, an experiment-runner agent, a reviewer agent, a
   theory/proof agent, a calibration agent).
2. **Invocation protocol** — when to spawn each, how to brief them, what
   artifacts they return, and how their output is reconciled with the main
   context.
3. **Tool allowlists** per agent.
4. **Handoff format** — the canonical structure for passing plans,
   intermediate results, and verification artifacts between agents.
5. **Termination / verification criteria** — how the main agent decides a
   sub-agent's output is acceptable before integrating it.

Companion specs for each agent will live in `.claude/agents/<agent_name>.md`.

Until this file is filled in, Claude should use the generic subagent types
available in its environment (e.g. `Plan`, `Explore`, `general-purpose`) per
the workflow rules in `CLAUDE.md`, and should NOT invent domain-specific
agents speculatively.
