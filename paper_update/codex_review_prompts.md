# Codex review prompts for Phase VII-B

Source: internal review pass on `phase-VII-B-strategic-2026-04-26`,
2026-04-26. The user authorized full autonomous run; the assistant
cannot launch `/codex:review` or `/codex:adversarial-review` directly
(those require user invocation). The two prompts below are prepared
for the user to run after the BLOCKER list in `tasks/todo.md` is
resolved.

**IMPORTANT precondition.** All three BLOCKERs in
`tasks/todo.md § Phase VII-B Review Triage (2026-04-26)` must be
resolved before triggering Codex. In particular: the working tree
must be committed onto `phase-VII-B-strategic-2026-04-26` so the
`--base` diff against `phase-VII-overnight-2026-04-26` is non-empty.
At the time this file was written, both branches point to the same
commit (`656e7cbf`) and `git diff phase-VII-overnight-2026-04-26..HEAD --stat`
produces no output, which would yield an empty Codex review.

---

## 1. Structural review

```text
/codex:review --base phase-VII-overnight-2026-04-26 --background
```

When prompted for a focus / context string, paste the block below
verbatim:

> Review the Phase VII-B strategic-learning extension on branch
> `phase-VII-B-strategic-2026-04-26`. Implementation surface to
> review:
>
> - `experiments/adaptive_beta/strategic_games/`
>   (`matrix_game.py`, `history.py`, `registry.py`,
>   `run_strategic.py`, `metrics.py`, `logging.py`,
>   `adversaries/*`, `games/*`, `configs/*`, `analysis/*`).
> - `tests/adaptive_beta/strategic_games/*` (12 test modules).
> - `scripts/figures/phase_VII_B/stage_B2_{dev,main}_analysis.py`.
> - `results/adaptive_beta/strategic/stage_B2_dev_summary.md`,
>   `stage_B2_main_summary.md`,
>   `paper_update/no_update_recommendation.md`.
>
> The phase implements an endogenous-opponent extension to the
> Phase VII-A adaptive-β work. Stage B2-Dev (180 cells, 3 seeds)
> produced 3 promoted (game, adversary) cells; Stage B2-Main
> (100 cells, 10 seeds × 10k episodes) ran on shapley × {FM-RM,
> HypTest} only; verdict is NO UPDATE because fixed_negative
> ties or beats adaptive_beta on shapley × HypTest auc_full
> (paired-bootstrap CI excludes 0).
>
> Spec authority for this review:
>
> - Phase-specific spec: `tasks/phase_VII_B_strategic_learning_coding_agent_spec.md`.
> - Parent operator spec: `docs/specs/phase_VII_adaptive_beta.md`.
> - Workflow rules: `AGENTS.md` §§4–5, `CLAUDE.md` §4.
>
> Specifically check (one finding per item, severity-tagged
> BLOCKER/MAJOR/MINOR/NIT):
>
> 1. **Operator non-duplication (spec §2.1).** Confirm
>    `experiments/adaptive_beta/strategic_games/` does not
>    re-implement the TAB / safe weighted log-sum-exp operator
>    `g_{β,γ}(r, v)`, `ρ_{β,γ}`, or `d_{β,γ}` outside
>    `src/lse_rl/operator/tab_operator.py`. The only `np.logaddexp`
>    call in the strategic_games tree must be in
>    `adversaries/smoothed_fictitious_play.py`, and only for the
>    opponent-policy softmax — NOT for any Bellman target.
>
> 2. **Same-code-path invariant (parent spec §16.2).** All five
>    methods (`vanilla`, `fixed_positive`, `fixed_negative`,
>    `adaptive_beta`, `adaptive_sign_only`) flow through the
>    Phase VII-A `AdaptiveBetaQAgent.step()`; only the schedule
>    object differs. `run_strategic.run_one_cell` calls
>    `AdaptiveBetaQAgent` and `build_schedule` from the
>    Phase VII-A modules — confirm no shadow update path.
>
> 3. **Endogenous-adversary purity (spec §18).** The three
>    Dev-stage promoted adversaries — `finite_memory_best_response`,
>    `finite_memory_regret_matching`, `hypothesis_testing` — must
>    update their state from `GameHistory` only (not from a
>    scripted clock or a wall-clock counter). `scripted_phase`
>    is permitted only as the regression-fixture target for
>    `tests/adaptive_beta/strategic_games/test_regression_rps.py`.
>
> 4. **Manifest accounting (spec §13).** Every (game, adversary,
>    method, seed) cell in Dev (180) and Main (100) must have a
>    manifest entry with `status ∈ {completed, failed, skipped}`.
>    Note the known issue: Dev manifest contains 181 records
>    because of a stale 5-episode smoke duplicate; analyst
>    deduped via `started_at` in the analysis script but the
>    raw manifest is still polluted. Flag the runner contract.
>
> 5. **Schema parity with Phase VII-A.** `episodes_strategic.csv`
>    is written in parallel to the legacy `episodes.csv`; both
>    come from the same per-episode loop. The strategic schema
>    columns (spec §13: `opponent_policy_entropy`,
>    `policy_total_variation`, `support_shift`, `model_rejected`,
>    `search_phase`, `memory_m`, `inertia_lambda`, `temperature`,
>    `tau`) must be present and non-NaN in completed runs.
>
> 6. **Paired-seed integrity (parent spec §8.4 + §9).** All
>    methods at fixed `seed_id` must share the same
>    `common_env_seed` (= `10000 + seed_id`). Agent seeds are
>    offset by `stable_hash(method) % 1000` (SHA-256 of the
>    method name, NOT Python's randomised builtin). The
>    `seed_id`-keyed paired bootstrap in
>    `scripts/figures/phase_VII_B/stage_B2_main_analysis.py`
>    relies on this; confirm the two paths cannot drift.
>
> 7. **Reproducibility test coverage (spec §12.3).**
>    `tests/adaptive_beta/strategic_games/test_reproducibility.py`
>    asserts byte-identical `metrics.npz` across two dispatches
>    of the same config + seed; assert that the test isn't
>    weakened by an env- or RNG-leakage shortcut.
>
> 8. **Failure handling (spec §19, parent §2.6).** No-clip
>    divergence is a first-class outcome (`divergence_event`
>    flag set, run completes); confirm `run_one_cell` records
>    `status="failed"` AND a traceback log on every exception
>    path, never silently skips.
>
> Output: severity-tagged file:line citations. BLOCKER must
> include a concrete acceptance criterion. The phase-specific
> branch is at HEAD `656e7cbf` (same commit as
> `phase-VII-overnight-2026-04-26` until BLOCKER-3 is fixed);
> review the working-tree diff once it is committed.
```

---

## 2. Adversarial review

```text
/codex:adversarial-review --base phase-VII-overnight-2026-04-26 --background
```

When prompted for the focus string, paste the block below verbatim
(this is the Phase VII-B–specific adaptation of the parent
`docs/specs/phase_VII_adaptive_beta.md` §14.1 focus, calibrated to
VII-B's actual risks — no-update verdict honesty, single-game
generalization limits, fixed_negative-dominance interpretation):

> challenge the Phase VII-B strategic-learning verdict. Specifically:
> (1) ARE the three "endogenous" adversaries
>     (`finite_memory_best_response`,
>     `finite_memory_regret_matching`, `hypothesis_testing`)
>     genuinely endogenous, or do any of them reduce to a scripted
>     clock once you account for the agent being ε-greedy with a
>     deterministic ε-decay schedule? Inspect each adversary's
>     `act` and `observe` for any branch that reads a global
>     time counter rather than the `GameHistory`. The
>     hypothesis-testing adversary's `search_phase` budget is a
>     local counter — confirm it is not a clock-equivalent.
> (2) IS the `fixed_negative ≥ adaptive_beta` finding on
>     shapley × HypTest robust to seed-pairing leakage? The Stage
>     B2-Main paired bootstrap pairs by `seed_id` (which sets
>     `common_env_seed`) — but `agent_seed = base + stable_hash(method) % 1000`
>     means different methods see different ε-greedy action
>     traces. Verify that paired-bootstrap CIs would not flip
>     direction if you re-paired by `(common_env_seed, method)`
>     instead of `seed_id`. Construct a counter-example or
>     confirm robustness.
> (3) IS the no-update verdict honest about the
>     `adaptive_beta_vs_fixed_negative` AUC_full diff of
>     -374.9 [-708.2, -45.1] on HypTest? The point estimate
>     is real and CI excludes zero, but only on AUC_full (cumulative,
>     n=10k episodes). On the spec-§15 PRIMARY endpoint
>     (`auc_first_2k`, sample efficiency) the diff is -63.8
>     [-175.4, +46.9] — CI overlaps zero. Pick the sharper
>     framing: is the no-update verdict driven by
>     **dominance** or by **failure to clear the §17 ≥ 2
>     settings bar with a strong CI**? The memo conflates the
>     two; force a separation.
> (4) STAGE B2-DEV is n=3 seeds; the Cartesian-product filter
>     dropped (matching_pennies, FM-RM) which DID clear the
>     strict CI gate (z=+2.57) so it could match the Cartesian
>     shape with the (HypTest, mp) cell that failed. This is a
>     SELECTION-ON-COVARIATES move that is not in the spec.
>     Challenge whether dropping a strictly-promoted cell to
>     maintain a Cartesian product is consistent with spec §11.1
>     "promoted settings" — or whether it constitutes a covert
>     gating artifact.
> (5) THE STRATEGIC RPS regression (`adaptive_beta` UNDER-performs
>     vanilla on `strategic_rps × {FM-BR, FM-RM, HypTest}` at n=3,
>     Δ AUC ∈ {-744, -415, -703}) is mentioned as an "Open
>     Follow-up" in the Main memo but NOT escalated. Spec §17
>     says "no paper update if gains are only in trivial settings"
>     — but the existing Phase VII-A appendix on RPS is silent
>     about whether the gain transfers to endogenous opponents.
>     The strategic_rps regression is evidence the Phase VII-A
>     RPS narrative may be ARTIFACTUAL OF THE SCRIPTED OPPONENT.
>     Demand that the next reviewer either reproduce the
>     regression at n=10 or explicitly hedge the existing paper
>     §VII-A appendix.
> (6) MECHANISM METRICS (`align ≈ 0.86`, `d_eff ≈ 0.48 < γ=0.95`)
>     are SUPPORTIVE on shapley × HypTest. But `fixed_negative`
>     also runs at `align ≈ 0.90`, `d_eff ≈ 0.55` (per Strategic-
>     Metric Table in `stage_B2_main_summary.md` §5). I.e., the
>     mechanism is identical between adaptive_beta and
>     fixed_negative on this game family. The "mechanism is
>     active" framing is true but does NOT distinguish adaptive
>     from a one-line constant. Force a separation: report
>     `mechanism_metric_delta(adaptive_beta − fixed_negative)`
>     and treat near-zero as null.
> (7) MANIFEST ACCOUNTING. Stage B2-Dev raw manifest contains
>     181 records (180 expected) — one stale `n_episodes=5`
>     smoke duplicate at `(matching_pennies,
>     finite_memory_best_response, vanilla, seed=0)`. The
>     analyst's analysis script deduplicates by `started_at`
>     and proceeds; the raw manifest is still polluted on disk.
>     Spec §2.3 ("no silent run drops") is satisfied
>     literally (the duplicate is an extra, not a drop), but
>     the runner contract is violated and any future audit
>     that joins via the manifest will silently double-count.
> (8) NO operator-math duplication. Verifier's audit
>     (`tests/adaptive_beta/strategic_games/test_operator_shared_kernel.py`)
>     confirms `np.logaddexp` only appears in
>     `smoothed_fictitious_play.py`, and that no Bellman-target
>     code path is shadowed. Spot-check that finding by reading
>     `run_strategic.py:686-845` — the Bellman update is
>     delegated to `AdaptiveBetaQAgent.step` which imports from
>     `tab_operator`. Confirm no shortcut.
> (9) SINGLE-GAME stage B2-Main scope. The promoted matrix
>     (shapley × {FMRM, HypTest}) is below the spec §17
>     "≥ 2 strategic settings" main-update bar AND below the
>     appendix-only "one strong + others mixed" bar. The
>     no-update verdict is therefore correct on a different
>     ground (scope, not dominance). Force the memo to make
>     that explicit.
> (10) HONESTY of weak/mixed reporting. Spec §18 forbids
>      "burying" weak results. Audit the no-update memo and
>      Stage B2-Main summary §8 for: (a) explicit naming of
>      the strategic_rps regression with numbers, (b) explicit
>      naming of the matching_pennies × FMRM Dev-promotion-then-
>      drop, (c) explicit naming of fixed_negative ≥
>      adaptive_beta on HypTest. (a)/(c) are present; (b) is
>      missing.
>
> Output: severity-tagged BLOCKER/MAJOR/MINOR/NIT with file:line
> citations. Each BLOCKER must include a concrete acceptance
> criterion. Apply Y2K-conservative pessimism: any line that
> could plausibly be a bug or a misleading framing IS one until
> proven otherwise.
```

---

## Bookkeeping

After both Codex jobs return:

1. `/codex:status` until both complete.
2. `/codex:result <session-id>` for each.
3. Pipe both outputs through the `review-triage` subagent (this
   subagent — `.claude/agents/review-triage.md`) to produce
   severity-rolled BLOCKER/MAJOR/MINOR/NIT entries appended to
   `tasks/todo.md § Phase VII-B Review Triage (2026-04-26)` —
   merge with the internal review entries already there.
4. BLOCKER == ∅ before merging the VII-B branch back to
   `phase-VII-overnight-2026-04-26` (NOT main; VII-B's no-update
   verdict means main does not change in this phase).
