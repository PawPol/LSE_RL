---
description: Phase-boundary Codex gate. Runs /codex:review and /codex:adversarial-review in the background, polls, aggregates, and triages. Usage: /lse:review <I | II | III> [--base <ref>]
argument-hint: <I | II | III> [--base <ref>]
---

# /lse:review

Arguments: `$ARGUMENTS` — phase (`I`, `II`, or `III`) and optional
`--base <ref>` (default `main`).

## Preconditions

1. `/lse:verify --full` must have returned `PASS` on the closing branch.
   If not, abort and instruct the user to run `/lse:verify --full` first.
2. The closing branch must be named `phase-<P>/closing` per
   `AGENTS.md §6`. If not, abort and ask the user to rename/checkout.

## Protocol

1. Run `/codex:review --base $BASE --background`. Capture session id.
2. Pick the adversarial focus string by phase:
   - `I` — "challenge finite-horizon DP correctness and the
     calibration-logging schema; flag any silent math or schema drift
     from docs/specs/phase_I_classical_beta0_experiments.md."
   - `II` — "challenge whether stress families actually isolate the
     classical weakness and whether event logging is sufficient to
     drive Phase III schedule calibration per
     docs/specs/phase_II_stress_test_beta0_experiments.md."
   - `III` — "challenge operator correctness (g_t^safe closed form,
     responsibility, local derivative), certification-box invariance
     (kappa_t, B_hat_t, beta_cap), β=0 collapse, and logaddexp
     numerical stability, per
     docs/specs/phase_III_safe_weighted_lse_experiments.md."
3. Run `/codex:adversarial-review --base $BASE --background "<focus>"`.
   Capture session id.
4. Poll `/codex:status` until both jobs complete. Do NOT busy-wait in
   tight loops; use the subagent reporting mechanism or Read
   transcripts.
5. `/codex:result <session-id>` for each job. Save both to
   `results/processed/codex_reviews/phase_<P>/{review, adversarial}.md`.
6. Spawn `review-triage` subagent with both review files as input.
7. Present the triage summary (BLOCKER / MAJOR / MINOR / NIT /
   DISPUTE counts) to the user.

## Exit criteria

- If BLOCKER count > 0: do NOT close the phase. Route BLOCKERs through
  `/lse:implement` and re-run `/lse:review` afterward.
- If BLOCKER count == 0: phase can be closed. User still approves the
  merge manually.

## Non-negotiables

- This command never silently auto-merges. All merges are user-driven.
- This command never calls `/codex:rescue`. Rescue is an explicit,
  scoped tool the user invokes when they want Codex to write a fix.
