# Codex Adversarial Review — Phase I (branch diff against main)

**Job**: review-mo28i5mw-uxfs9z  
**Completed**: 2026-04-17  
**Base**: main  
**Branch**: phase-I/closing  
**Focus**: "challenge finite-horizon DP correctness and the calibration-logging schema; flag any silent math or schema drift from docs/specs/phase_I_classical_beta0_experiments.md."

## Verdict

**needs-attention** — No-ship. The branch breaks the finite-horizon contract in the time wrapper, inflates RL success metrics to the point of being non-informative, logs incorrect terminal calibration values, and the RL gamma' ablation does not actually change the training dynamics.

## Findings

### [critical] Time-augmented wrapper ends episodes one step early relative to the advertised horizon
**File**: `mushroom-rl-dev/mushroom_rl/environments/time_augmented_env.py:266-277`  
**Confidence**: 0.98

`DiscreteTimeAugmentedEnv.step()` forces `absorbing=True` when `t_next >= horizon - 1`. With `Core` also terminating when `episode_steps >= env.info.horizon`, this wrapper stops episodes after only `horizon - 1` actions, while the DP code and spec treat a horizon `H` as decision stages `0..H-1` with terminal row `V[H]=0`. That silently shifts the RL task to a shorter problem than the DP planners and corrupts stage-indexed comparisons and calibration by dropping the last decision stage.

**Recommendation**: Terminate only after the final decision stage has been executed, i.e. align wrapper termination with `Core`'s horizon accounting so an environment with horizon `H` yields `H` actions and a terminal successor at stage `H`.

### [high] Evaluation treats any episode termination as success, so timeout terminations are counted as solved episodes
**File**: `experiments/weighted_lse_dp/common/callbacks.py:443-452`  
**Confidence**: 0.97

`RLEvaluator.evaluate()` sets `episode_success = True` on any `absorbing` transition. In this branch the time-augmentation wrapper uses `absorbing` to signal horizon exhaustion as well as true task termination, so every evaluation rollout that simply times out is counted as a success. The resulting `success_rate`, `steps_to_threshold`, and paper curves can look perfect even when the policy never reaches the goal.

**Recommendation**: Separate task success from episode termination. Derive success from an explicit environment/info signal or task-specific goal predicate, and do not map generic timeout/horizon termination to `success=True`.

### [high] Terminal transitions log a non-zero continuation value instead of the terminal zero value
**File**: `experiments/weighted_lse_dp/common/callbacks.py:112-126`  
**Confidence**: 0.96

`TransitionLogger` always computes `v_next_beta0` as `max_a Q[next_aug_id, a]`, even when `absorbing`/`last` is true. Under the finite-horizon contract, terminal continuations should be zero; here terminal samples inherit a bootstrap value from the clamped next augmented state instead. That directly corrupts `v_next_beta0`, `margin_beta0`, `td_target_beta0`, and `td_error_beta0` in `transitions.npz`, so the calibration lake is wrong exactly on the boundary where the horizon matters most.

**Recommendation**: When `absorbing` or `last` is true, write `v_next_beta0 = 0.0` and recompute the derived terminal quantities from that value instead of reading from the Q-table.

### [high] The RL gamma' ablation only changes logged metadata and evaluation discounting; training still runs at the original gamma
**File**: `experiments/weighted_lse_dp/runners/run_phase1_rl.py:251-306`  
**Confidence**: 0.95

`run_phase1_rl.py` overwrites the local `gamma` variable when `--gamma-prime` is passed, but the environment is already created by the factory with its built-in gamma and the agent is built from `mdp_rl.info`, so the learning updates still use the task's original discount. The override only reaches the resolved config, the transition logger, and the evaluator's return computation. That makes the advertised fixed-discount ablation invalid: the run directory says gamma' while the learned policy/Q-values came from gamma.

**Recommendation**: Apply `gamma_prime` before constructing the RL environment/agent, or patch `mdp_rl.info.gamma` (and any derived evaluator/env copies) before training so both updates and logged metadata use the same effective discount.

## Next Steps (from Codex)

1. Fix the time-wrapper horizon semantics first; the current RL/DP comparison is not on the same finite-horizon problem.
2. Recompute RL evaluation with an explicit success predicate instead of using `absorbing` as a proxy.
3. Regenerate transition/calibration artifacts after zeroing terminal continuations.
4. Make the RL gamma' override affect the actual MDP/agent before rerunning any ablation results.
