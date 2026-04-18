# Codex Review — Phase I (branch diff against main)

**Job**: review-mo28fhl3-bbtc3z  
**Completed**: 2026-04-17  
**Base**: main  
**Branch**: phase-I/closing  

## Summary

The new Phase I RL pipeline miscomputes key evaluation and calibration quantities, and the advertised RL gamma' ablation does not actually vary the training discount. These issues materially affect the reported experiment outputs.

## Findings

### [P1] Do not treat every terminal rollout as a successful episode
**File**: `experiments/weighted_lse_dp/common/callbacks.py:451-452`

On the time-augmented environments used by `run_phase1_rl.py`, `step()` forces `absorbing=True` at the horizon even when the agent never reaches the task goal. Because `evaluate()` marks `episode_success = True` for any absorbing transition, horizon expirations are counted as successes, so `success_rate` (and therefore `steps_to_threshold`) becomes artificially high or even 1.0 on every checkpoint for tasks that simply time out.

### [P1] Evaluate RL checkpoints without epsilon-greedy exploration
**File**: `experiments/weighted_lse_dp/common/callbacks.py:443-445`

`RLEvaluator` claims to run greedy rollouts, but it calls `self._agent.draw_action(...)` on the training agent, whose policy was constructed as `EpsGreedy(epsilon=0.1)`. That means every evaluation checkpoint is still exploring 10% of the time, so the reported learning curves and final returns are systematically noisier and lower than the policy the paper intends to measure.

### [P1] Propagate gamma' into the RL environment before building the agent
**File**: `experiments/weighted_lse_dp/runners/run_phase1_rl.py:255-284`

The ablation runner passes `gamma_prime` into `run_single()`, but this function only rewrites the local `gamma` variable used for logging and evaluation. The actual environment and `mdp_rl.info` are still created from the base task config, so `QLearning`/`ExpectedSARSA` train with the original discount (0.99) for every ablation run. As a result, the RL half of the gamma' sweep does not actually change the trained objective, while the emitted metadata says it did.

### [P2] Zero out bootstrapped value on terminal transitions in calibration logs
**File**: `experiments/weighted_lse_dp/common/callbacks.py:109-115`

`TransitionLogger` always records `v_next_beta0` as `max_a Q(next_state,a)`, even when the sampled transition is terminal (`absorbing`/`last`). For goal-reaching or horizon-ending transitions, the bootstrap term should be 0; otherwise `margin_beta0`, `td_target_beta0`, and the aggregated calibration statistics are biased by a value estimate from a state where no further return should be accrued.
