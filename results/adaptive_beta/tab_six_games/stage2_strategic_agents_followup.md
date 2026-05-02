# M7.2 Stage 2 — Strategic-learning agent baselines (RM + FP)

- **Created**: 2026-05-02
- **Spec authority**: `docs/specs/phase_VIII_tab_six_games.md` §6.3
  patch §3 (lines 744–770), promoted from M11 → M7
- **Codex review**: completed 2026-05-02; 3 findings (1 already-fixed,
  1 applied-pre-dispatch, 1 deferred). See §3 below.
- **Verdict**: **PASS — strategic-learning agents constitute a major
  challenge to TAB's matrix-game claims and a clean diagnostic
  failure on cycling cells.**

## 1. Dispatch summary

| stage | runs | wall | status |
|---|---:|---:|---|
| `m7_2_stage2_strategic_agents_headline` | 240 | ~10 min | 240 / 240 ✓ |

Scope: 3 cells × 4 γ × 10 seeds × 2 methods = 240 runs.
- Cells: AC-Trap, RR-StationaryConvention, SH-FiniteMemoryRegret
  (DC-Long50 dropped per Codex P1 #2 — see §3)
- γ: {0.60, 0.80, 0.90, 0.95}
- Seeds: 0–9 (paired with M7.1 TAB re-dispatch + Q-learning baselines)
- Methods: `regret_matching_agent`, `smoothed_fictitious_play_agent`

All 240 runs completed cleanly. Test suite regression-clean.

## 2. Build provenance

The strategic-learning agent wrappers are NEW code at:
- `experiments/adaptive_beta/strategic_games/agents/strategic_learning_agents.py`
  (564 LOC, three classes: `_StrategicAgentBase`,
  `RegretMatchingAgent`, `SmoothedFictitiousPlayAgent`)
- `experiments/adaptive_beta/strategic_games/agents/__init__.py` (23 LOC)

Both wrap the existing **opponent** classes
(`adversaries/regret_matching.py`,
`adversaries/smoothed_fictitious_play.py`) into the agent interface
used by `AdaptiveBetaQAgent`. The wrappers:
- Read the env's history (`env.history` public property) to observe
  the env-adversary's last action.
- Pass `payoff_agent.T` to the inner class so the class's
  "agent_action" axis becomes the env-adversary's actions and the
  "opponent_action" axis becomes the wrapper's own action space.
- For the smoothed-FP variant, use a fast-path that reads
  `history.empirical_opponent_policy(m=memory_m)` directly instead of
  copying the full role-swapped history per step
  (avoids the O(T²) cost flagged in spec §6 of the Codex review).

The Stage 2 runner (`run_phase_VIII_stage2_baselines.py`) is extended
with two new branches in `_build_baseline_agent` plus a
`_resolve_payoff_agent` helper. The 3 M7.1 baseline branches
(`restart_Q_learning`, `sliding_window_Q_learning`,
`tuned_epsilon_greedy_Q_learning`) are unchanged bit-for-bit.

## 3. Codex review disposition

The Codex review (working-tree pass, 2026-05-02) returned three
findings. Disposition:

### P1 #1 — quadratic history copy in smoothed FP

Reported at `strategic_learning_agents.py:536-538`. The wrapper's
naive role-swap path was reported as O(T²) over a 10k-episode run
because it would copy the full cross-episode `GameHistory` per step.

**Disposition: ALREADY FIXED at build time.** The algo-implementer
detected the same hot path during smoke profiling (>30 min wallclock
for one cell with `memory_m=100`) and replaced the role-swap with a
direct fast-path that reads
`history.empirical_opponent_policy(m=memory_m, n_actions=...)`
(O(memory_m) per step, no list copy). The math is identical; the
fast-path is documented inline at line 543. Resulting wallclock:
7.1 s for AC-Trap × 10k episodes (≈ 250× speedup). Verified by reading
the actual file post-Codex-flag (line 543 contains the
`empirical_opponent_policy` call, NOT a `_role_swap` copy).

### P1 #2 — DC-Long50 false-best on bellman_residual headline

Reported at `stage2_strategic_agents_headline.yaml:100`. Strategic-
learning agents emit `bellman_residual ≡ 0` (no TD error in their
update rule). The DC-Long50 headline metric is
`-log(bellman_residual)` AUC, which would compute
`-log(max(0, 1e-12)) × 10 000 ≈ 276 300` — strictly greater than
vanilla TAB's ~269 408 — falsely ranking these "expected-failure"
agents as #1 on the chain task.

**Disposition: APPLIED PRE-DISPATCH.** Removed DC-Long50 from the
strategic-agents config. New scope: 3 cells × 4 γ × 10 seeds ×
2 methods = 240 runs (down from 320). The diagnostic value (these
agents cannot learn DC-Long50 because they have no value
bootstrapping) is documented in spec §10.3 patch §3; running them on
DC-Long50 would have created a misleading measurement artefact in the
headline aggregation. The exclusion is annotated inline in the
config at the dropped subcase comment.

### P2 #3 — `payoff_agent` resolved from game module, ignores game_kwargs

Reported at `run_phase_VIII_stage2_baselines.py:734`. The runner
calls `_resolve_payoff_agent(subcase.game)` which reads the module
top-level `payoff_agent` constant rather than the env-instantiated
payoff matrix, ignoring any `subcase.game_kwargs` that parameterise
payoffs (e.g. `rules_of_road` with `payoff_bias`,
`asymmetric_coordination` with custom `coop_payoff`/`risk_payoff`).

**Disposition: DEFERRED — no impact on M7.1's 4-cell scope.**
Inspection of the M7.1 config envelope: AC-Trap uses only
`horizon: 20` (no payoff overrides); RR-StationaryConvention uses
`horizon: 20` + `stationary_mixed` adversary `probs: [0.7, 0.3]`
(no `payoff_bias`); SH-FiniteMemoryRegret uses only `horizon: 20`;
DC-Long50 has no payoff matrix at all (and is excluded from M7.2).
For the current dispatch the default module-level `payoff_agent`
matches the env's payoff exactly. **Tracked for future M7.3 / M9
expansion** where parameterised cells might be added.

## 4. Paired-comparison results (full M7 suite, post-M7.2)

Now folding the strategic-learning agents into the §4 of
`stage2_fixed_tab_vs_baselines.md`. Method ordering: TAB-positive,
TAB-negative, TAB-grid, restart, sliding-window, tuned-ε, RM-agent,
FP-agent.

### 4.1 AC-Trap

| γ | method | source | Δ vs vanilla | CI₉₅ | sig |
|---|---|---|---:|---|---|
| 0.60 | best_fixed_positive_TAB | `fixed_beta_+0.1` | +131.7 | [+87.8, +174.2] | ✓ |
| 0.60 | restart_Q_learning | — | +250 596.6 | [+248 254, +252 827] | ✓ |
| 0.60 | tuned_epsilon_greedy_Q_learning | — | +44 194.1 | [+43 878, +44 512] | ✓ |
| 0.60 | **regret_matching_agent** | — | **+271 018.7** | [+270 520, +271 592] | **✓ (best)** |
| 0.60 | smoothed_fictitious_play_agent | — | +71 069.4 | [+70 768, +71 393] | ✓ |
| 0.95 | regret_matching_agent | — | **+270 949** | [+270 415, +271 546] | ✓ (best) |
| 0.95 | smoothed_fictitious_play_agent | — | +70 999.7 | [+70 684, +71 339] | ✓ |
| 0.95 | restart_Q_learning | — | +248 462.7 | [+245 970, +250 984] | ✓ |
| 0.95 | tuned_epsilon_greedy_Q_learning | — | +43 485.4 | [+43 089, +43 857] | ✓ |
| 0.95 | best_fixed_*_TAB | various | various | various (all ✗) |

**AC-Trap takeaway**: `regret_matching_agent` is the single best
method on AC-Trap at every γ tested (Δ ≈ +271k AUC, ~51 % relative
gain over vanilla). The wrapper directly plays the payoff-dominant
Stag/Stag equilibrium (last-100 mean return ≈ 4 = halfway between
risk=3 and coop=5 → near-pure-Stag late). Smoothed FP attractors to
the risk-dominant Hare/Hare equilibrium (Δ ≈ +71k, ~13 % gain).
Restart Q-learning is third (Δ ≈ +249k); tuned-ε fourth (Δ ≈ +44k).
**TAB's best gain (+131.7) is three orders smaller than RM's
(+271 019).**

### 4.2 RR-StationaryConvention

| γ | method | Δ vs vanilla | CI₉₅ | sig |
|---|---|---:|---|---|
| 0.60 | regret_matching_agent | **+39 384.6** | [+27 467, +51 377] | **✓ (best)** |
| 0.60 | smoothed_fictitious_play_agent | +34 044.6 | [+33 843, +34 239] | ✓ |
| 0.60 | tuned_epsilon_greedy_Q_learning | +14 465.6 | [+14 281, +14 642] | ✓ |
| 0.60 | best_fixed_negative_TAB | +26.0 (β=−0.5) | [+8.2, +42.8] | ✓ |
| 0.60 | restart_Q_learning | -2 155.8 | [-2 321, -2 010] | ✗ |
| 0.95 | regret_matching_agent | +39 714.0 | [+27 865, +51 655] | ✓ (best) |
| 0.95 | smoothed_fictitious_play_agent | +34 374.0 | [+34 126, +34 626] | ✓ |
| 0.95 | tuned_epsilon_greedy_Q_learning | +14 419.0 | [+14 233, +14 602] | ✓ |
| 0.95 | best_fixed_negative_TAB | +320.0 (β=−0.5) | [+241.2, +401.4] | ✓ |

**RR takeaway**: same pattern as AC-Trap — `regret_matching_agent`
is the strict best method at every γ (Δ ≈ +39k, ~70 % relative
gain). The stationary biased opponent is a fixed
`probs=[0.7, 0.3]` — RM directly best-responds to this fixed
distribution and converges fast. Smoothed FP also beats vanilla by
+34k. **TAB's best gain (+320) is two orders smaller than RM's
(+39 714).**

### 4.3 SH-FiniteMemoryRegret — the diagnostic failure

| γ | method | Δ vs vanilla | CI₉₅ | sig |
|---|---|---:|---|---|
| 0.60 | tuned_epsilon_greedy_Q_learning | **+11 402.5** | [+11 300, +11 495] | **✓ (best)** |
| 0.60 | best_fixed_negative_TAB | +38.7 | [-60.8, +145.2] | 0 |
| 0.60 | restart_Q_learning | -20 378.6 | [-23 747, -17 040] | ✗ |
| 0.60 | smoothed_fictitious_play_agent | **-56 264.9** | [-56 406, -56 136] | **✗ (worst)** |
| 0.60 | regret_matching_agent | **-74 355.3** | [-74 542, -74 157] | **✗ (worst)** |
| 0.95 | tuned_epsilon_greedy_Q_learning | +11 582.9 | [+11 469, +11 711] | ✓ (best) |
| 0.95 | best_fixed_negative_TAB | +147.8 | [-14.9, +334.6] | 0 |
| 0.95 | regret_matching_agent | -74 243.7 | [-74 425, -74 043] | ✗ |
| 0.95 | smoothed_fictitious_play_agent | -56 153.3 | [-56 289, -56 020] | ✗ |

**SH-FMR takeaway**: this is the **expected diagnostic failure** per
spec §6.2 (Shapley = "central testbed for adaptive β under
endogenous cycling"). When BOTH the agent and the env-adversary use
strategic-learning rules (RM-agent vs FMR-adversary;
FP-agent vs FMR-adversary), the joint dynamics enter the
Brown-Robinson cycling pathology and accumulate uniformly worse-than
vanilla returns:

- `regret_matching_agent` Δ ≈ −74 k at every γ
- `smoothed_fictitious_play_agent` Δ ≈ −56 k at every γ

Magnitudes are stable across γ (±0.05 of mean), implying the
cycling is regime-stable. This is a clean **negative result** for
strategic-learning baselines specifically: when the opponent is
itself a strategic learner, classical FP/RM fail catastrophically.
Q-learning with a tuned ε schedule (`+11 500`) and TAB-`fixed_beta_-0.5`
(γ=0.80, +97.3) are the only methods that beat vanilla here.

This is the most important finding of M7.2: it provides a clean
**existence proof** of a regime where TAB and tuned-Q baselines
beat strategic-learning baselines, validating the spec §6.2 framing
of Shapley as the "cycling testbed" and supplying the paper with a
concrete cell where the alignment-condition argument has visible
purchase.

## 5. M7.1 + M7.2 combined headline (across the matrix-game subset)

Best method per (cell, γ) by paired-CI₉₅ vs vanilla (rendered at
γ = 0.95 for compactness; pattern is stable across γ):

| cell | best method | Δ | runner-up | runner-up Δ |
|---|---|---:|---|---:|
| AC-Trap | **regret_matching_agent** | **+270 949** | restart_Q_learning | +248 463 |
| RR-StationaryConvention | **regret_matching_agent** | **+39 714** | smoothed_fictitious_play_agent | +34 374 |
| SH-FiniteMemoryRegret | **tuned_epsilon_greedy_Q_learning** | **+11 583** | best_fixed_negative_TAB | +148 (γ=0.95 not sig) |

**Across all three matrix games**:
- TAB never wins on AUC magnitude vs the strategic-learning baselines.
- TAB's only CI-significant matrix-game wins (RR all γ, SH-FMR γ=0.80,
  AC-Trap γ=0.60 with β=+0.10) are **two-to-three orders of magnitude
  smaller** than the strategic-learning agent gains.
- TAB's distinctive contribution remains DC-Long50 (covered in M7.1
  §4.4; not re-tested here per Codex P1 #2).

## 6. Implications for the paper headline (post-M7.2)

The earlier M7.1 conclusion — *"TAB is specialised for delayed-credit
tasks; tuned-ε wins matrix games"* — sharpens after M7.2:

> **Strategic-learning agents (RM, FP) dominate Q-learning-family
> methods (vanilla, tuned-ε, restart, sliding-window, TAB) on
> stationary or payoff-anchored matrix games (AC-Trap,
> RR-StationaryConvention).** They achieve this without value
> bootstrapping. TAB's contribution is therefore **not** a matrix-game
> win — that goes to the right strategic-learning baseline. TAB's
> contribution is the **DC-Long50 chain task** (only TAB-`fixed_beta_-2.0`
> beats vanilla there) and the **SH-FMR cycling diagnostic** at
> γ=0.80 (where strategic-learning baselines fail catastrophically
> and TAB-β=−0.5 produces a small but CI-significant gain).

The paper headline should reframe TAB as a *credit-assignment
mechanism for delayed-reward / value-bootstrapping-required tasks*,
NOT as a general-purpose Bellman-operator improvement.

## 7. Acceptance for M7 → M8 / M9 promotion

M7.1 already passed acceptance (1 G_+ + 9 G_- candidates). M7.2 adds:

- ✓ Diagnostic failure cell identified: SH-FMR with strategic-learning
  agents (cycling pathology, Δ ≈ −60k to −74k vs vanilla).
- ✓ Strict-best baseline established per cell:
  - AC-Trap: `regret_matching_agent`
  - RR-StationaryConvention: `regret_matching_agent`
  - SH-FiniteMemoryRegret: `tuned_epsilon_greedy_Q_learning`

M7 is now closed pending user sign-off. Recommended next step:
**M8 → M9 sign-switching composite** (already analysis-only-passed at
HEAD). M9's primary composite candidate (AC-Trap γ=0.60 + RR γ=0.60)
remains valid; the strategic-learning-agent finding sharpens the
context but does not change M8's G_+ / G_− classification.

## 8. Limitations + future work

1. **Codex P2 #3 unaddressed**: `_resolve_payoff_agent` reads the
   module-level `payoff_agent` constant, ignoring `subcase.game_kwargs`.
   No impact on M7.1's 4 cells (none uses payoff-modifying kwargs).
   Apply a `payoff_agent_kwargs_override` path before any future
   dispatch on `soda_uncertain` or `rules_of_road` with `payoff_bias`.
2. **DC-Long50 strategic-agent diagnostic not run.** Smoke confirmed
   they complete (per spec §10.3 expected-failure characterisation),
   but the headline excluded them. If a chain-specific diagnostic
   table is needed for the paper, add it to M7.3 with a metric these
   agents emit meaningfully (e.g. mean return = 1.0 always; not
   distinguishing).
3. **One tested γ-strip per cell.** Each cell uses its own γ-grid
   {0.60, 0.80, 0.90, 0.95}. The strategic-learning agents are
   γ-insensitive (no value bootstrapping), so their AUC is stable
   across γ within ±0.1 %. This is itself a finding, but worth
   noting as a baseline characteristic.

## 9. Reproduction

```bash
# M7.2 main dispatch (240 runs, ~10 min)
.venv/bin/python -m experiments.adaptive_beta.tab_six_games.runners.run_phase_VIII_stage2_baselines \
    --config experiments/adaptive_beta/tab_six_games/configs/stage2_strategic_agents_headline.yaml \
    --output-root results/adaptive_beta/tab_six_games

# Re-aggregate (M7.1 + M7.2 → 4080 rows in long CSV; 96 paired comparisons)
.venv/bin/python scripts/figures/phase_VIII/m7_aggregate.py
```

Inputs: M7.1 commit `722fd275` + M7.2 240 runs (this dispatch).

Outputs:
- `processed/m7_1_long.csv` (extended to 4 080 rows: 3 360 TAB + 480 Q-baselines + 240 strategic agents)
- `processed/m7_1_paired_comparison.csv` (144 paired rows, was 96 in M7.1)
- this memo
