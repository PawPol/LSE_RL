# AC-Trap pre-sweep Codex review

- Review UTC: 2026-05-01T16:52:53Z
- Repo/branch: `LSE_RL`, `phase-VIII-tab-six-games-2026-04-30`
- HEAD under review: `1d88e769`

## 1. **Verdict**: GENUINE FINDING

This is not a BLOCKER/MAJOR/MINOR/NIT implementation bug in the checked source surfaces. The spec predicted `AUC(+1) > AUC(0) > AUC(-1)` with Cohen's `d > 0.5` for AC-Trap (`docs/specs/phase_VIII_tab_six_games.md:1175-1195`), and the spec's contradicted branch explicitly allows a GENUINE FINDING disposition (`docs/specs/phase_VIII_tab_six_games.md:1197-1208`). The five-condition ablation reverses or fails the positive-β claim in every condition (`results/adaptive_beta/tab_six_games/pre_sweep_AC_Trap_ablation.md:61-69`), including the longer 1000-episode run where `d(+1,0) = -23.380` (`results/adaptive_beta/tab_six_games/pre_sweep_AC_Trap_ablation.md:67`).

Confirmed evidence: the operator, game, agent, schedule, and listed adversaries are internally consistent with the implemented experiment. The reversal is explained by the operator's bootstrap-alignment dynamics: positive β is beneficial only when `β * (r - v_next) > 0`; it is destabilizing when learned bootstraps make `v_next > r`.

## 2. **(a) Destabilization hypothesis**: yes

Yes. The hypothesis is theoretically sound: alignment controls convergence to the current bootstrap target, not to the payoff-dominant equilibrium label. The implemented kernel is

```text
g_{β,γ}(r,v) = (1 + γ) / β * [logaddexp(β r, β v + log γ) - log(1+γ)]
```

with the classical branch `g = r + γ v` only at `β = 0` / `|β| <= 1e-8` (`src/lse_rl/operator/tab_operator.py:52-63`). The effective discount is `(1+γ) * (1 - rho)` (`src/lse_rl/operator/tab_operator.py:77-83`), where `rho` depends on `β * (r - v) - log γ` (`src/lse_rl/operator/tab_operator.py:66-74`).

Asymptotic forms from the implemented `logaddexp`:

- `β -> +∞`: `g_{β,γ}(r,v) -> (1+γ) * max(r, v)`.
- `β -> -∞`: `g_{β,γ}(r,v) -> (1+γ) * min(r, v)`.
- Ties give `(1+γ) * r = (1+γ) * v` in either limit.

Therefore, when `v_next > r`, positive β makes `d_eff -> 1+γ`, which is greater than 1 for the AC-Trap runs with `γ = 0.95` (`experiments/adaptive_beta/tab_six_games/configs/pre_sweep_AC_Trap.yaml:34`). Negative β makes `d_eff -> 0` in the same regime. This is exactly the observed pattern.

Raw alignment evidence:

- Baseline fixed `+1`, seed 0: `results/adaptive_beta/tab_six_games/raw/VIII/pre_sweep_AC_Trap/asymmetric_coordination/AC-Trap/fixed_beta_+1/seed_0/metrics.npz::alignment_rate[:20].mean() = 0.575`, `alignment_rate[100:120].mean() = 0.0475`, `alignment_rate[-20:].mean() = 0.0500`; `q_abs_max[-1] = 81678.24276147355`; `effective_discount_mean[-20:].mean() = 1.8389121675043754`.
- Baseline fixed `+1`, all seeds: `alignment_rate[-20:].mean()` over seed 0/1/2 is `0.0500`; `q_abs_max[-1]` mean is `85918.471` from the three `fixed_beta_+1/seed_{0,1,2}/metrics.npz` files.
- A2 long fixed `+1`, seed 0: `results/adaptive_beta/tab_six_games/raw/VIII/pre_sweep_AC_Trap_A2_long/asymmetric_coordination/AC-Trap/fixed_beta_+1/seed_0/metrics.npz::alignment_rate[:20].mean() = 0.5500`, `alignment_rate[-20:].mean() = 0.0525`, `q_abs_max.max() = 1085189.1328948075`, and `divergence_event.sum() = 311`.

This is overshoot relative to the finite-horizon payoff scale. The game pays at most `5` per step (`experiments/adaptive_beta/strategic_games/games/asymmetric_coordination.py:67-88`), the pre-sweep horizon is `20` (`experiments/adaptive_beta/tab_six_games/configs/pre_sweep_AC_Trap.yaml:43-50`), and `γ = 0.95` (`experiments/adaptive_beta/tab_six_games/configs/pre_sweep_AC_Trap.yaml:34`), so even the discounted all-Stag payoff bound is about `5 * (1 - 0.95^20) / (1 - 0.95) = 64.1514`. Fixed `+1` reaches `q_abs_max` from thousands to over one million in the raw metrics cited above.

## 3. **(b) Rescue regimes**

1. Very-optimistic init + very-low ε: not a robust rescue. Existing optimistic-init evidence already fails under the configured ε schedule: A1 uses `q_init = 5.0` (`experiments/adaptive_beta/tab_six_games/configs/pre_sweep_AC_Trap_A1_q5.yaml:18-25`) and still gives `mean(+1) = 8139.67 < mean(0) = 10363.17 < mean(-1) = 10446.83` (`results/adaptive_beta/tab_six_games/pre_sweep_AC_Trap_ablation.md:63-69`). A very-low-ε variant could path-lock to Stag because action selection tie-breaks greedy ties to the lowest action (`experiments/adaptive_beta/agents.py:146-156`) and action `0` is Stag (`experiments/adaptive_beta/strategic_games/games/asymmetric_coordination.py:11-14`). That would be a narrow tie-break/exploration artifact, not evidence that `+β` selects payoff dominance robustly. No raw `metrics.npz` exists in the supplied AC-Trap tree for a very-low-ε optimistic variant; the supplied raw tree contains only the five pre-sweep conditions listed in `results/adaptive_beta/tab_six_games/pre_sweep_AC_Trap_ablation.md:61-69`.

2. Opponent mixes uniformly: no rescue. A4 uses `stationary_mixed` with `probs: [0.5, 0.5]` (`experiments/adaptive_beta/tab_six_games/configs/pre_sweep_AC_Trap_A4_uniform.yaml:28-36`) and reports `mean(+1) = 10991.00 < mean(0) = 11264.00 < mean(-1) = 11574.17` (`results/adaptive_beta/tab_six_games/pre_sweep_AC_Trap_ablation.md:63-69`). This also matches the one-step payoff geometry: under a uniform opponent, Stag has expected immediate payoff `0.5*5 + 0.5*0 = 2.5`, while Hare has payoff `3` regardless of opponent action (`experiments/adaptive_beta/strategic_games/games/asymmetric_coordination.py:20-28`, `experiments/adaptive_beta/strategic_games/games/asymmetric_coordination.py:80-88`).

3. Very small β, e.g. `0.1`: not an evidenced rescue; at best it narrows the harm by making the operator close to classical Q-learning. The small-β expansion of the implemented kernel is `g = r + γv + β * γ/(2(1+γ)) * (r-v)^2 + O(β^2)`, derived from the implemented `logaddexp` formula (`src/lse_rl/operator/tab_operator.py:52-63`). Thus `β=0.1` mainly adds a small optimism term; it does not change the sign of the alignment condition, and if `v_next > r`, positive β still moves `d_eff` above γ per `effective_discount` (`src/lse_rl/operator/tab_operator.py:77-83`). No supplied AC-Trap raw `metrics.npz` exists for `fixed_beta_+0.1`; the spec only schedules supplementary `±0.1` figure-grid values for a future figures-only sub-pass, not for this AC-Trap pre-sweep (`docs/specs/phase_VIII_tab_six_games.md:1210-1230`).

4. Very-low γ: not a robust paper-claim rescue. Lower γ suppresses `v_next`, so the β mechanism vanishes toward an immediate-reward learner rather than selecting payoff-dominant equilibrium structure. The agent accepts `0 <= gamma < 1` (`experiments/adaptive_beta/agents.py:92-93`), and the operator's nonzero-β branch is bootstrap-driven through `log γ` and `v` (`src/lse_rl/operator/tab_operator.py:59-63`). As `γ -> 0`, the bootstrap term disappears and the AC-Trap immediate payoffs still favor Hare under the uniform opponent as above. No supplied AC-Trap raw `metrics.npz` exists for a low-γ condition.

Overall: I found no regime in the supplied evidence where `+β` robustly helps AC-Trap. The only plausible positive-return variant is a very narrow low-exploration/tie-break path-locking setup, which would not support the paper claim that fixed-positive TAB selects payoff-dominant equilibria.

## 4. **(c) Alignment diagnostic**: yes

Yes. The diagnostic correctly flags AC-Trap as outside the positive-β regime. The agent computes strict alignment as `signed = beta * (reward - v_next)` and `aligned = signed > 0.0` (`experiments/adaptive_beta/agents.py:350-353`), aggregates it as `alignment_rate` (`experiments/adaptive_beta/agents.py:246-252`), and returns both strict `alignment_rate` and non-strict `frac_positive_signed_alignment` (`experiments/adaptive_beta/agents.py:277-288`). The runner persists `alignment_rate` into `metrics.npz` (`experiments/adaptive_beta/tab_six_games/runners/run_phase_VIII_stage1_beta_sweep.py:639-645`, `experiments/adaptive_beta/tab_six_games/runners/run_phase_VIII_stage1_beta_sweep.py:656-664`).

Confirmed raw data: fixed `+1` falls below 0.5 over training in every supplied condition. Baseline fixed `+1` seed 0 has `alignment_rate[:20].mean() = 0.575` but `alignment_rate[-20:].mean() = 0.0500` in `results/adaptive_beta/tab_six_games/raw/VIII/pre_sweep_AC_Trap/asymmetric_coordination/AC-Trap/fixed_beta_+1/seed_0/metrics.npz`. A2 long fixed `+1` seed 0 has `alignment_rate[:20].mean() = 0.5500` and `alignment_rate[-20:].mean() = 0.0525` in `results/adaptive_beta/tab_six_games/raw/VIII/pre_sweep_AC_Trap_A2_long/asymmetric_coordination/AC-Trap/fixed_beta_+1/seed_0/metrics.npz`. A4 uniform fixed `+1` seed 0 has `alignment_rate[:20].mean() = 0.7550` and `alignment_rate[-20:].mean() = 0.0475` in `results/adaptive_beta/tab_six_games/raw/VIII/pre_sweep_AC_Trap_A4_uniform/asymmetric_coordination/AC-Trap/fixed_beta_+1/seed_0/metrics.npz`.

Non-causal logging note: these pre-sweep `metrics.npz` files do not contain a persisted `frac_positive_signed_alignment` key; the runner only writes `alignment_rate` and `effective_discount_mean` among the alignment/discount diagnostics (`experiments/adaptive_beta/tab_six_games/runners/run_phase_VIII_stage1_beta_sweep.py:136-146`, `experiments/adaptive_beta/tab_six_games/runners/run_phase_VIII_stage1_beta_sweep.py:656-670`). This is not a blind spot for this adjudication because the strict `alignment_rate` field is present and flags the failure strongly.

## 5. **(d) Implementation bugs**

- `asymmetric_coordination.py`: CLEAN. The payoff matrix is rows = agent action, columns = opponent action, with Stag/Stag = `5`, Stag/Hare = `0`, Hare/Stag = `3`, Hare/Hare = `3` (`experiments/adaptive_beta/strategic_games/games/asymmetric_coordination.py:72-92`). `MatrixGameEnv` realizes rewards as `payoff_agent[agent_action, opponent_action]` and `payoff_opponent[agent_action, opponent_action]` (`experiments/adaptive_beta/strategic_games/matrix_game.py:383-387`). The source sets `metadata["canonical_sign"] = "+"` and `env.env_canonical_sign = "+"` (`experiments/adaptive_beta/strategic_games/games/asymmetric_coordination.py:158-193`), matching the current spec's pre-registered claim (`docs/specs/phase_VIII_tab_six_games.md:369-375`). No source diff.

- `finite_memory_regret_matching.py`: CLEAN. It validates the payoff matrix is 2-D, validates `memory_m >= 1`, infers/checks `n_actions == payoff_opponent.shape[1]`, samples with `rng.choice(self.n_actions, p=policy)`, and computes regrets over the sliding window as `payoff_opponent[a, :] - payoff_opponent[a, b]` (`experiments/adaptive_beta/strategic_games/adversaries/finite_memory_regret_matching.py:59-89`, `experiments/adaptive_beta/strategic_games/adversaries/finite_memory_regret_matching.py:103-124`, `experiments/adaptive_beta/strategic_games/adversaries/finite_memory_regret_matching.py:126-138`). No off-by-one or probability-vector bug found. No source diff.

- `inertia.py`: CLEAN for sampling/validation. It validates `inertia_lambda` in `[0,1]`, inherits `n_actions` validation, samples with `rng.integers(0, self.n_actions)` using NumPy's exclusive upper bound, and repeats the previous action with probability `inertia_lambda` (`experiments/adaptive_beta/strategic_games/adversaries/base.py:82-87`, `experiments/adaptive_beta/strategic_games/adversaries/inertia.py:80-96`, `experiments/adaptive_beta/strategic_games/adversaries/inertia.py:113-146`). The class reset clears `_last_action` (`experiments/adaptive_beta/strategic_games/adversaries/inertia.py:102-111`); the matrix-game runner intentionally preserves rolling adversary state after episode 0 (`experiments/adaptive_beta/strategic_games/matrix_game.py:327-334`). That convention affects A3 interpretation but is not a causal bug for the baseline/A1/A2/A4 reversal. No source diff.

- `stationary.py`: CLEAN. It validates `probs` is 1-D, nonnegative, finite-sum-to-one within `1e-9`, validates `n_actions == len(probs)`, renormalizes a copy, and samples via `rng.choice(self.n_actions, p=self._probs)` (`experiments/adaptive_beta/strategic_games/adversaries/stationary.py:37-63`, `experiments/adaptive_beta/strategic_games/adversaries/stationary.py:72-79`). No source diff.

- `AdaptiveBetaQAgent._step_update`: CLEAN. On nonterminal transitions it sets `v_next = max_a Q[next_state, a]`, not `Q[state, action]`; terminal transitions use `0.0` (`experiments/adaptive_beta/agents.py:323-330`). It passes `g(beta, gamma, reward, v_next)` (`experiments/adaptive_beta/agents.py:331-333`) and then updates exactly the selected cell `Q[state, action]` (`experiments/adaptive_beta/agents.py:347-348`). No source diff.

- `FixedBetaSchedule` and sign plumbing: CLEAN. The runner maps `fixed_beta_+x` to `METHOD_FIXED_POSITIVE` with `beta0=x` and `fixed_beta_-x` to `METHOD_FIXED_NEGATIVE` with `beta0=x` (`experiments/adaptive_beta/tab_six_games/runners/run_phase_VIII_stage1_beta_sweep.py:42-53`, `experiments/adaptive_beta/tab_six_games/runners/run_phase_VIII_stage1_beta_sweep.py:184-238`). `FixedBetaSchedule(+1)` returns `+beta0`, and `FixedBetaSchedule(-1)` returns `-beta0` for every episode (`experiments/adaptive_beta/schedules.py:364-385`, `experiments/adaptive_beta/schedules.py:922-927`). Confirmed raw fields: `results/adaptive_beta/tab_six_games/raw/VIII/pre_sweep_AC_Trap/asymmetric_coordination/AC-Trap/fixed_beta_+1/seed_0/metrics.npz::beta_used.min() = beta_used.max() = 1.0`; `results/adaptive_beta/tab_six_games/raw/VIII/pre_sweep_AC_Trap/asymmetric_coordination/AC-Trap/fixed_beta_-1/seed_0/metrics.npz::beta_used.min() = beta_used.max() = -1.0`. No source diff.

## 6. **Recommended action**

Treat AC-Trap as a genuine falsification cell for the naive payoff-dominance claim, not as a failed implementation. Proceed with M6 only after amending the narrative so AC-Trap is not used as a positive-control G+ cell.

Recommended v7 amendment for `docs/specs/phase_VIII_tab_six_games.md` §5.4:

```text
### 5.4 Asymmetric Coordination — `asymmetric_coordination.py` [DONE]

Actions = {A, B} (stag-hunt-style). Canonical pre-registration sign: +.
Subcases: `AC-FictitiousPlay`, `AC-SmoothedBR`, `AC-Inertia`, `AC-Trap`.

Role: miscoordination traps; pathwise dynamics; falsifiability cell for
the claim that optimistic TAB selects payoff-dominant equilibria.

AC-Trap is NOT a positive-control G+ cell after the 2026-05-01
pre-sweep. Across q-init, horizon, regret-matching, inertia, and uniform
opponent ablations, fixed +β fails to beat vanilla and often strongly
underperforms. The mechanism diagnosis is that TAB sign must be judged
by bootstrap alignment β * (r - v_next), not by the equilibrium payoff
label. Once learned Q makes v_next exceed the realized reward, +β drives
d_eff above γ and can exceed 1, destabilizing Q even in a stag-hunt
payoff-dominant environment.

Use AC-Trap as a negative/falsifiability subsection and as an alignment
diagnostic demonstration: positive β is outside-regime when
alignment_rate drops below 0.5 over training.
```

Recommended new negative-result subsection, replacing/renumbering current §13.5 as needed:

```text
### 13.5 Negative-result honesty: AC-Trap falsifiability cell

The AC-Trap pre-sweep is reported even when it contradicts the original
payoff-dominance prediction. A valid report includes:

1. AUC and paired-seed effect sizes for β ∈ {-1, 0, +1}.
2. Alignment-rate traces showing whether fixed +β remains in-regime.
3. q_abs_max / divergence diagnostics when d_eff rises above 1.
4. A statement that TAB sign is governed by β * (r - v_next), not by
the payoff-dominant equilibrium label.

Success for the broader Phase VIII program does not require AC-Trap to
be positive. It requires that AC-Trap's negative result be correctly
predicted by the alignment diagnostic and not hidden by post-hoc cell
selection.
```

No source diff is recommended for this reversal. A separate schema cleanup may persist `frac_positive_signed_alignment` in Phase VIII `metrics.npz`, but the present verdict does not depend on that field because `alignment_rate` already flags AC-Trap correctly.
