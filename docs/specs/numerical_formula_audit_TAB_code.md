# Numerical Stability and Formula Audit for Phases I–III

## Overall conclusion

The core scalar TAB/safe-TAB operator is mostly numerically careful:
- `np.logaddexp` is used for the weighted-LSE target,
- `scipy.special.expit` is used for the sigmoid / responsibility,
- the KL form is implemented consistently with the target,
- no overflow/underflow/non-finite outputs were observed in direct stress tests over wide finite ranges.

However, there are several confirmed issues. One is a **major mathematical bug** in the safe DP planners; a few others are real numerical or statistical bugs; and several more are edge-case weaknesses.

## Confirmed major formula bug

### 1. Safe DP planners implement `g(r̄, E[V'])` instead of `E[g(r̄, V')]`

The latest TAB paper defines the finite-horizon Bellman equations as

\[
Q_t(s,a)=\mathbb{E}_{s'\sim P(\cdot\mid s,a)}\bigl[\bar g_t(\hat r(s,a), V_{t+1}(s'))\bigr].
\]

But the current safe DP implementations compute the nonlinear target **after** averaging the next-state value:

\[
Q_t(s,a)=g_t^{\rm safe}(\bar r(s,a), \mathbb{E}[V_{t+1}(s')\mid s,a]).
\]

This is mathematically different whenever `beta != 0` and the transition is stochastic.

Confirmed locations:
- `mushroom_rl/algorithms/value/dp/safe_weighted_value_iteration.py`
- `mushroom_rl/algorithms/value/dp/safe_weighted_policy_evaluation.py`
- `mushroom_rl/algorithms/value/dp/safe_weighted_modified_policy_iteration.py`
- `mushroom_rl/algorithms/value/dp/safe_weighted_async_value_iteration.py`
- `SafeWeightedPolicyIteration` inherits this through `SafeWeightedPolicyEvaluation`.

A direct numerical witness with `gamma=0.99`, `beta=0.5`, `r=0`, and next values `{-10,+10}` equally likely:
- `g(r, E[V']) = g(0, 0) = 0`
- `E[g(r, V')] \approx 7.218`

So this is not a small approximation error; it can be large.

Implication: safe DP tables, safe DP calibration that depends on them, and DP-side theory/experiment comparisons on stochastic tasks are not faithful to the current paper.

## Confirmed numerical issues

### 2. Tiny-beta safeguard has an off-by-threshold problem

In `safe_weighted_common.py`, the code treats `|beta| < 1e-8` as classical, but **not** `|beta| == 1e-8`.

So exactly at the threshold the code still evaluates the log-sum-exp formula, where cancellation against the classical branch can be noticeable for large value scales.

Observed absolute error of `compute_safe_target(r,v,t)` relative to `r + gamma v` at `beta = 1e-8`, `gamma = 0.99`, `r = scale`, `v = -scale`:
- scale = `1`: error `2.0e-08`
- scale = `100`: error `9.95e-05`
- scale = `1000`: error `9.95e-03`
- scale = `10000`: error `9.95e-01`

This is easy to fix with `<=` instead of `<`, or with a series expansion around `beta = 0`.

### 3. Tiny-beta paths are internally inconsistent

For `|beta| < 1e-8`, `compute_safe_target()` returns the classical target and sets the effective discount to exactly `gamma`.

But `compute_effective_discount()` and `compute_rho()` do **not** use the same tiny-beta shortcut. So for the same `(r, v, t)` the target path and the explicit effective-discount path can differ slightly.

The mismatch is small, but it is real and unnecessary.

### 4. `metrics.aggregate(..., axis=1)` computes malformed bootstrap CIs

`experiments/weighted_lse_dp/common/metrics.py` handles the default `axis=0` case correctly, but the bootstrap branch is wrong for `axis=1`.

A direct check with a `(3,4)` array gives:
- `mean.shape == (3,)`
- but `ci_low.shape == (100,)` and `ci_high.shape == (100,)`

So the helper is mathematically wrong away from the default aggregation axis.

### 5. Tail-risk helper crashes on empty inputs

`TailRiskLogger.compute()` calls `np.nanpercentile` before validating `n > 0`.
On empty arrays this raises instead of returning a controlled result.

This is an edge-case numerical bug rather than a core algorithmic one.

## Confirmed calibration/statistical formula bug

### 6. The Phase III schedule builder uses the wrong positive-margin statistic

The schedule builder treats `aligned_positive_mean_mean` as a proxy for a representative positive margin and then defines

\[
\text{raw}_t = \text{aligned\_positive\_mean}[t]\,\sqrt{\text{aligned\_margin\_freq}[t]}.
\]

But in the Phase II calibration code,

\[
\text{aligned\_positive\_mean}[t] = \mathbb{E}[\max(m_t,0)]
= \Pr(m_t>0)\,\mathbb{E}[m_t \mid m_t>0].
\]

So the implemented informativeness score is actually proportional to

\[
\mathbb{E}[m_t \mid m_t>0] \cdot \Pr(m_t>0)^{3/2},
\]

which suppresses rare informative stages by an extra factor of `Pr(m_t>0)`.

That is a strong candidate explanation for the near-zero `beta_cap` / near-classical Phase III behavior.

## Formulas that appear correct

The following parts looked mathematically consistent and numerically sound in direct checks:

### Scalar safe target and responsibility
In `safe_weighted_common.py`:
- `compute_safe_target`
- `compute_rho`
- `compute_effective_discount`
- `compute_kl_term`

The target matches the variational / KL form to machine precision:

\[
g_{\beta,\gamma}(r,v)
=(1+\gamma)\left[\rho r + (1-\rho)v - \beta^{-1}\mathrm{KL}(\rho\|p_0)\right]
\]

with `rho` equal to the implemented logistic responsibility.

### Stable primitives
The code uses the right stable primitives in the dangerous places:
- `np.logaddexp` instead of `log(exp(a)+exp(b))`
- `expit` instead of `1 / (1 + exp(-x))`

A direct stress sweep over large finite ranges produced no non-finite outputs.

### Inverse design of raw beta
`compute_raw_beta()` in the calibration utilities appears algebraically correct for the intended positive-margin / chosen-sign regime. When the resulting beta is plugged back into the implemented effective-discount formula, it reproduces the requested target discount to machine precision for positive-sign tests.

## Lower-severity semantic issues

### 7. Adaptation recovery thresholds are awkward on negative-return tasks
`AdaptationMetricsLogger` defines recovery as reaching a percentage of the post-change optimum. If the optimum is negative, these thresholds can become semantically odd or unattainable.

This is not a crash, but it can make adaptation metrics hard to interpret.

### 8. The reward averaging convention is only safe under the paper's current reward model
The DP code precomputes `r_bar = E[R | s,a]` and feeds that into the nonlinear target. That is consistent with the **current paper's** state-action reward notation `\hat r(s,a)`. But if you later generalize the theory to truly transition-dependent stochastic rewards inside the nonlinear target, that implementation would need to change to an expectation of the nonlinear target over the joint next-state / reward kernel.

## Priority order for fixes

1. Fix the safe DP Bellman operator to compute the expectation of the nonlinear target, not the nonlinear target of the expectation.
2. Fix the schedule-builder margin/informativeness statistics.
3. Separate train and eval environments and repair the previously reported experimental-pipeline bugs.
4. Harden the tiny-beta branch (`<=`, or series expansion).
5. Fix `metrics.aggregate` for nonzero aggregation axes.
6. Add guards for empty-input tail-risk and negative-return adaptation semantics.

## Practical takeaway

The core scalar TAB operator is **not** the main numerical weak point. The bigger problems are:
- a DP/operator mismatch with the paper,
- calibration statistics that overly shrink the deployed nonlinearity,
- and a few experimental/statistical helper bugs.

So the right next step is not “more floating-point tuning.” It is fixing the Bellman operator and the schedule/calibration pipeline first.
