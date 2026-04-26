Go fully autonomous.

I authorize the overnight run to proceed through implementation, review, experiments, analysis, and final paper-results update, not only Stage A. Do not stop at the Stage A gate unless there is a genuine technical failure, unsafe inconsistency, or failed regression that cannot be repaired.

Answers:

1. Stage gating during overnight

Choose modified option (b): auto-promote based on quantitative evidence.

Do not hard-stop at Stage A. Run Stage A first, compute the promotion diagnostics, then automatically promote the strongest environments if they clear the predefined bar.

Promotion bar:

- Adaptive-β improves AUC over vanilla on paired seeds in at least one environment.
- No increase in catastrophic episodes relative to vanilla beyond noise.
- No divergence for clipped adaptive-β.
- Mechanism metrics are directionally consistent on non-bandit environments:
  - improved alignment rate, or
  - meaningful effective-discount modulation, or
  - faster post-shift recovery.

If no environment clears the bar, stop after Stage A and write the failure/negative-result report.

If at least one environment clears the bar, proceed to Stage B.

Stage C should run only for the strongest headline environment(s), not everything.

2. Stage B compute budget

Use the following budget:

- Stage A:
  - 1,000 episodes
  - 3 seeds
  - 4 environments
  - core methods only

- Stage B:
  - promote strongest 1–2 environments
  - 10,000 episodes
  - 10 seeds
  - full main-method comparison
  - limited ablations only

- Stage C:
  - only if Stage B shows stable headline-quality results
  - strongest 1 environment, or at most 2
  - 20 seeds
  - 10,000 episodes
  - final paper-quality figures and tables

Stop after approximately 8–10 hours if unfinished, but preserve all completed results and write a status memo.

3. Codex gate cadence

Use the aggressive-but-bounded review plan:

- M1 operator refactor:
  - verifier
  - `/codex:review`
  - `/codex:adversarial-review`
  - triage and fix BLOCKER/MAJOR issues

- M2 and M3:
  - verifier-only unless something touches stable infrastructure

- End of Stage A:
  - verifier

- End of Stage B:
  - `/codex:review`
  - triage

- End of Stage C / final paper update:
  - `/codex:review`
  - `/codex:adversarial-review`
  - final verifier

Use Codex agents aggressively when needed, but do not waste calls on trivial formatting-only edits.

4. MushroomRL-dev refactor blast radius

Confirmed.

Use strict rollback discipline.

Before and after the refactor:

- Run the relevant existing pytest suite.
- Run numerical equivalence tests for `SafeWeightedCommon.compute_safe_target`.
- Test `compute_effective_discount` and `compute_rho` if touched.
- Compare outputs over a fixed grid of `(β, γ, r, v)`.

If equivalence fails and cannot be repaired cleanly, roll back the refactor and stop with a memo. Do not continue with a broken operator.

5. Branch and commit policy

Choose option (a).

Create one branch:

`phase-VII-overnight-2026-04-26`

Use commits per milestone/task. Do not push and do not open a PR unless explicitly requested later.

No direct commits to main.

6. Failure handling

Choose option (c).

Retry up to 2 times for transient or local implementation failures. If still failing, stop, preserve logs, write a status memo, and do not continue downstream on a corrupted state.

For expected no-clip divergence, do not stop. Log and count it as an experimental outcome.

7. Self-play in Stage A

Confirmed.

Do not implement or run self-play RPS in Stage A.

However, since I am authorizing a fully autonomous overnight run, self-play may be implemented and run in Stage B only if Stage A produces a stable adaptive-β signal and Stage B has enough time budget remaining. Treat it as a secondary stress test, not a headline result.

8. Data volume

Confirmed.

Use full transition logs for Stage A. Do not stratify or silently drop logs.

For Stage B/C, keep full episode-level logs and mechanism summaries. Full transition logs are preferred, but if disk becomes excessive, compress to parquet and document the retention policy.

9. Fully autonomous spec edits

Use this rule:

- For editorial inconsistencies or obvious missing details: make the call, document it in `tasks/lessons.md`, and continue.
- For architectural ambiguities that could invalidate Phase III–VI compatibility, stop and write a memo.
- For experiment-selection decisions after Stage A/B: proceed autonomously using the quantitative promotion criteria above.

10. End-of-run deliverable

I want all of the following:

- `tasks/phase_VII_overnight_2026-04-26.md`
  - full task ledger
  - commits
  - tests
  - Codex review outcomes
  - failures/retries
  - final status

- `results/adaptive_beta/stage_A_summary.md`

- Stage B and Stage C summaries if reached:
  - `results/adaptive_beta/stage_B_summary.md`
  - `results/adaptive_beta/stage_C_summary.md`

- All regenerable figures and processed tables under:
  - `results/adaptive_beta/processed/`
  - `results/adaptive_beta/figures/`
  - `results/adaptive_beta/tables/`

- Final recommendation memo:
  - `results/adaptive_beta/final_recommendation.md`

- Final paper update:
  - update the paper’s experiments/results section only if Stage B or C produces strong enough evidence.
  - If results are weak or mixed, do not force them into the main paper; instead produce appendix/supplement text and a clear recommendation.

Additional instruction:

Run the full pipeline down to the final paper-results update if the data supports it. The goal is not merely to implement the suite, but to produce paper-ready evidence if it exists.

Final autonomous plan:

- Branch: `phase-VII-overnight-2026-04-26`
- M1: shared operator kernel extraction + SafeWeightedCommon refactor + equivalence tests + Codex review/adversarial review.
- M2: MushroomRL-compatible environments, excluding self-play from Stage A.
- M3: adaptive-β agent, schedules, logging, runner, reproducibility tests.
- Stage A: 4 envs × core methods × 3 seeds × 1k episodes.
- Auto-gate: promote strongest 1–2 environments if criteria pass.
- Stage B: 10k episodes × 10 seeds on promoted environments.
- Stage C: 20 seeds on the strongest headline environment(s), only if Stage B is strong.
- Analysis: generate figures, tables, mechanism diagnostics, and summaries.
- Paper update: revise the experiments/results section if justified; otherwise prepare appendix/supplement material and final recommendation.
- Final Codex review + adversarial review + verifier audit.

Proceed autonomously.