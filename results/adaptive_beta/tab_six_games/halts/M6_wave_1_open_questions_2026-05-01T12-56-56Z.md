# HALT 5 — M6 wave 1 scaffolding done; 4 OQs blocking wave 1.5/wave 2 dispatch

- **Halt UTC**: 2026-05-01T12:56:56Z
- **Milestone**: M6 (post-M2-close, post-M5-close)
- **Wave 1 status**: COMPLETE.
  - 5 new files (1722 LoC): runner + 2 configs + smoke test + `__init__.py`.
  - Targeted smoke: 1 PASS in 1.37 s.
  - Full suite: **1694 PASS + 2 SKIP + 0 FAIL** in 35.61 s.
- **HEAD at halt**: `5c15687c6b95adb4b817cc32f4c327b782eba05b`
  (M2 ledger close; wave 1 scaffolding is uncommitted in working
  tree pending OQ resolution).
- **Why halting now**: wave 1.5 (AC-Trap pre-sweep) is the headline
  single-cell experiment for the paper's payoff-dominance claim
  (per spec §10.2 / patch §5.2: "the strongest single-cell result
  for the paper"). The wiring of `AC-Trap`'s adversary is
  underspecified in the spec and the runner's guess
  (`finite_memory_best_response` + `inertia_lambda=0.9`) is plausible
  but unverified. Per CLAUDE.md §3 (no silent gap-filling on
  methodological choices) and the v5b lesson on patch directive scope
  (methodological decisions = design = halt; significance thresholds
  = instrumentation = auto-fix), I am halting before any AC-Trap
  dispatch.

## Wave 1 deliverables (uncommitted; ready to ship pending OQ resolution)

| File | LoC |
| --- | ---: |
| `experiments/adaptive_beta/tab_six_games/runners/__init__.py` | 15 |
| `experiments/adaptive_beta/tab_six_games/runners/run_phase_VIII_stage1_beta_sweep.py` | 916 |
| `experiments/adaptive_beta/tab_six_games/configs/dev.yaml` | 253 |
| `experiments/adaptive_beta/tab_six_games/configs/stage1_beta_sweep.yaml` | 320 |
| `tests/adaptive_beta/tab_six_games/test_runner_smoke.py` | 218 |

The runner already integrates with `Phase8RunRoster` (manifests.py),
`make_run_dir(base="results/adaptive_beta/tab_six_games")`,
canonical `metrics.npz` schema (per-episode arrays for `return`,
`bellman_residual`, `beta_used`, `beta_raw`, `alignment_rate`,
`effective_discount_mean`, `q_abs_max`, plus subcase-specific
extras), and the v5b headline-metric routing
(`bellman_residual_beta` AUC for advance-only delayed_chain;
return-AUC for everything else). It also writes
`headline_metric` and `t11_guard ∈ {gap_based, cohens_d}` into each
run's `run.json` so wave 6 bug-hunt can dispatch the appropriate
detector per cell.

## Open questions (blocking wave 1.5 / wave 2)

### OQ3 [BLOCKING wave 1.5] — AC-Trap adversary wiring is unspecified

The spec §5.4 lists `AC-Trap` as a `[DONE]` subcase, but the
asymmetric_coordination game's `build()` does not register a
`subcase=` parameter; `AC-Trap` is encoded entirely by the
`(adversary, adversary_params)` tuple paired with the game.

The spec does not enumerate this tuple. Phase VII-B's
`stage_B2_dev.yaml` did not run asymmetric_coordination at all
(only matching_pennies, strategic_rps, shapley, rules_of_road), so
there is no Phase VII-B precedent.

The runner agent picked `finite_memory_best_response` with
`memory_m=20`, `inertia_lambda=0.9`, `temperature=0.2`, which is
plausible (high-inertia trapping behaviour matches the spec text
"miscoordination traps; pathwise dynamics") but **is a
methodological choice that affects the AC-Trap pre-sweep result**.

Since the user-quoted predicted ordering for wave 1.5 is
`AUC(+1) > AUC(0) > AUC(-1)` with Cohen's d > 0.5 vs vanilla, the
adversary choice will materially affect whether that prediction
holds. Wrong wiring → false halt or false success on the headline
single-cell claim.

**Question**: confirm or replace the AC-Trap adversary wiring.
Options:
- **(α)** Confirm `finite_memory_best_response(memory_m=20,
  inertia_lambda=0.9, temperature=0.2)` — the runner's choice.
- **(β)** Use `finite_memory_best_response(memory_m=20,
  inertia_lambda=0.5, temperature=0.2)` — Phase VII-B's canonical
  inertia setting (lower stickiness; less obvious trap).
- **(γ)** Use a different adversary (e.g.
  `finite_memory_regret_matching` — myopic regret matching is the
  classic trap-inducer for stag-hunt).
- **(δ)** Provide a different (adversary, params) tuple.

### OQ4 [BLOCKING wave 2] — `stationary` adversary `probs` for RR / SO / PG

Several Stage A subcases (`RR-StationaryConvention`,
`SO-Coordination`, `PG-StationaryFair`) use a `stationary`
adversary. The spec does not enumerate the `probs` distribution.

Runner agent's choices:
- RR (2 actions): `probs=[0.7, 0.3]` (right-biased convention,
  matching the "RR-StationaryConvention" naming intent).
- SO (m actions, varies): uniform.
- PG (3 actions canonical): uniform `[1/3, 1/3, 1/3]`.

These are plausible but again methodological. The RR right-bias of
0.7 is potentially load-bearing for the `RR-StationaryConvention`
cell's role in §22.5's mechanism-degenerate diagnostic.

**Question**: confirm or override the `probs` choices above.

### OQ1 [BLOCKING wave 4] — Stage 1 main run-count gap (1,820 vs spec 4,340)

Spec §10.2 advertises ~4,340 runs as
`1,260 (matrix) + 280 (RR-Sparse) + 2,800 (delayed_chain)`.
Under the natural `(subcase × method × seed)` dispatch
interpretation, the main count is `26 cells × 7 methods × 10 seeds
= 1,820`. The 2,800 delayed_chain figure looks like
`280 × 10 paired-seed comparisons` — i.e. an aggregator-side
accounting multiplier, not actual dispatched runs. The runner
encodes 1,820 with an inline comment.

**Question**: confirm 1,820 dispatched runs is the correct
interpretation, or specify the missing ~2,520 runs.

### OQ2 [non-blocking; wave 7 concern] — aggregator schema parity

The aggregator's `PHASE_VIII_EXPECTED_COLUMNS` does not include
`beta_raw`, `beta_used`, `effective_discount_mean`, `goal_reaches`.
Aggregator currently emits a soft `schema_drift` note and processes
the run anyway, but the long-CSV won't have those columns. Wave 6
T1-T10 detectors need them.

**Resolution path**: extend `PHASE_VIII_EXPECTED_COLUMNS` and/or
rename runner output keys to spec §7.4 wording verbatim
(`beta_used` → `beta_deployed`, `effective_discount_mean` →
`mean_d_eff`). Defer to wave 7 fix-up unless user wants to address
now.

## Orchestrator recommendation

For OQ3 (AC-Trap): **(γ)** — `finite_memory_regret_matching` with
`memory_m=20`. Regret matching is the classical "myopic trap"
adversary for stag-hunt; this is the cleanest stress on the
"fixed positive TAB selects payoff-dominant equilibria where
vanilla Q-learning is risk-dominated" claim. Rationale:
finite-memory BR with high inertia would ALSO trap, but it traps
on the dynamics rather than on the payoff structure; regret
matching traps on the payoff structure directly, which is the
phenomenon spec §10.2 wants to demonstrate.

For OQ4 (stationary probs): default to **(α)** the runner's
choices; researcher can override if Stage A diagnostics reveal
the stationary distribution is suboptimal for the test.

For OQ1: confirm 1,820 dispatched runs (the runner's
interpretation).

For OQ2: defer to wave 7 fix-up.

**Decision required from user** before:
1. wave 1.5 dispatch (AC-Trap pre-sweep — needs OQ3),
2. wave 2 dispatch (Stage A dev pass — needs OQ4),
3. wave 4 dispatch (Stage 1 main pass — needs OQ1).

Wave 1 deliverables can be committed *now* if the user confirms
OQ3+OQ4+OQ1 (configs need no edits in that case) or *after* the
user's edits land.
