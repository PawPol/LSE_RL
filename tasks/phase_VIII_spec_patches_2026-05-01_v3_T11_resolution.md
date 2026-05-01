# Phase VIII Spec Patch v3 — T11 Halt Resolution (delayed_chain metric)

**Status:** AUTHORITATIVE micro-amendment to
`docs/specs/phase_VIII_tab_six_games.md` §5.7 (the §11 fold-in from
v2 patch). Resolves the T11 halt at commit `09e7a262`.
**Source:** Researcher decision in response to halt memo
`results/adaptive_beta/tab_six_games/halts/delayed_chain_smoke_prediction_failure_2026-05-01T02-43-53Z.md`.
**Authority:** Same level as v2 patch in the authority chain.
**Lifecycle:** Apply via the same fold-in protocol as v1/v2; archive
to `tasks/archive/` after fold-in.

---

## 0. Decision: option (a) with refinement

The advance-only subcases (DC-Short10, DC-Medium20, DC-Long50) test
**operator contraction speed** via Q-convergence rate. AUC is the
wrong metric on these subcases because Discrete(1) action space
makes the policy invariant to β.

DC-Branching20 retains AUC as its metric because Discrete(2) action
space (advance vs branch_wrong) makes the policy β-dependent.

This refinement turns out to be cleaner than v2: each chain type
tests a distinct slice of the TAB story. The advance-only chain is a
**contraction-speed demonstrator** (operator mechanism); the branching
chain is a **temporal-credit + exploration demonstrator** (policy
quality under deferred reward).

Options (b) and (c) from the halt memo were rejected:

- **(b) add no-op action**: rejected because no-op vs advance is an
  artificial decision that distorts the long-horizon credit-assignment
  story. The natural metric for the operator is contraction; adding a
  fake action to keep AUC as the metric is a band-aid.
- **(c) scope P-Sign to DC-Branching20 only**: rejected because it
  forfeits the P-Scaling test (effect size grows with chain length L),
  which is the load-bearing claim for the paper title.

---

## 1. Apply protocol

After reading the halt memo and this file, the orchestrator MUST:

1. Open
   `/Users/liq/Documents/Claude/Projects/LSE_RL/docs/specs/phase_VIII_tab_six_games.md`
   §5.7 (which carries `<!-- patch-2026-05-01 §11 -->` markers from
   the v2 fold-in).
2. Apply the spec text edits in §2 below, marking each with inline
   `<!-- patch-2026-05-01-v3 -->` comment for provenance.
3. Open
   `/Users/liq/Documents/Claude/Projects/LSE_RL/experiments/adaptive_beta/tab_six_games/metrics.py`
   (created in M5) and add the `q_convergence_rate` metric per §3
   below.
4. Open
   `/Users/liq/Documents/Claude/Projects/LSE_RL/tests/adaptive_beta/strategic_games/test_delayed_chain_smoke_prediction.py`
   (created in M2 reopen) and rewrite the smoke test per §4 below.
5. Re-run the targeted smoke test under `.venv/bin/python` to confirm
   the new metric makes P-Sign testable on advance-only chains:
       .venv/bin/python -m pytest tests/adaptive_beta/strategic_games/test_delayed_chain_smoke_prediction.py -v
   Expect PASS.
6. Re-run full suite to confirm no regression:
       .venv/bin/python -m pytest tests/ -x -q
   Expect 1684 PASS + 2 SKIP + 0 FAIL (the previous FAIL is the smoke
   test; replacing the metric flips it to PASS).
7. Bump spec §23 changelog with:
       2026-05-01 v3 — T11 halt resolution: advance-only delayed_chain
       subcases switch from AUC to q_convergence_rate metric per
       tasks/phase_VIII_spec_patches_2026-05-01_v3_T11_resolution.md.
8. Append `tasks/lessons.md`:
       Lesson: when defining falsifiable predictions for Phase VIII
       cells, verify the metric axis is sensitive to the controlled
       variable for that cell's structure. AUC is invariant to β on
       Discrete(1) action spaces because the policy is forced;
       Q-convergence rate (or Bellman residual decay) is the natural
       β-sensitive axis for forced-policy MDPs. Source: T11 halt at
       commit 09e7a262, delayed_chain DC-Long50 smoke prediction
       failure 2026-05-01.
9. Commit:
       git add docs/specs/phase_VIII_tab_six_games.md
               experiments/adaptive_beta/tab_six_games/metrics.py
               tests/adaptive_beta/strategic_games/test_delayed_chain_smoke_prediction.py
               tasks/lessons.md
               tasks/phase_VIII_spec_patches_2026-05-01_v3_T11_resolution.md
       git commit -m "phase-VIII(M2 reopen): resolve T11 halt — switch advance-only delayed_chain to q_convergence_rate metric"
10. Update checkpoint with new spec_patch_2026_05_01_v3_sha256
    field (extend schema 1.2.0 → 1.3.0); record the SHA of this file.
11. Resume the autonomous run from M2 reopen close per the resumption
    plan in `session_progress_2026-05-01.md`. M4 reopen → M6 waves
    per session 3 original plan, with the metric fix applied.

If any step 1–9 cannot be applied cleanly, HALT with
`patch_v3_apply_failed` memo and STOP.

---

## 2. Spec text edits to §5.7

In `/Users/liq/Documents/Claude/Projects/LSE_RL/docs/specs/phase_VIII_tab_six_games.md`
§5.7 (the delayed_chain section folded in from v2 patch §11), apply
these edits:

### 2.1 Replace the §11.3 "Theoretical prediction" block

Replace the entire P-Sign / P-Scaling / P-Branch / P-VII-Parity
table with:

```text
**Falsifiable predictions:**

P-Contract:    On advance-only subcases (DC-Short10, DC-Medium20,
               DC-Long50), Q-convergence rate is monotonically
               ordered:
                   q_convergence_rate(+β) > q_convergence_rate(0) > q_convergence_rate(-β)
               where q_convergence_rate is the per-episode rate at
               which ||Q_e - Q*||_∞ decays toward 0, and Q* is the
               analytical optimum (Q*(s, advance) = γ^(L-s) for
               s ∈ [0, L-1]; Q*(L, ·) = 0). Effect: tightening of
               the alignment-condition contraction d_β,γ ≤ γ on
               positive-β-side; loosening on negative-β-side.

P-Scaling:     |q_convergence_rate(+β) - q_convergence_rate(0)|
               grows monotonically with chain length L:
                   DC-Short10 < DC-Medium20 < DC-Long50
               Direct test of the long-horizon temporal-credit
               assignment claim from the paper title.

P-AUC-Branch:  On DC-Branching20 (Discrete(2) action space), AUC
               is the natural metric:
                   AUC(+β) > AUC(0) > AUC(-β)
               Effect operates through value-driven exploration: +β
               concentrates value mass on the advance-arm faster,
               accelerating the resolution of the explore/exploit
               tradeoff between advance and branch_wrong.

P-VII-Parity:  q_convergence_rate(0) on DC-Long50 within
               paired-bootstrap 95% CI of Phase VII-A delayed_chain
               reference at the matched L=50 setting (cross-validation
               with the prior phase). Note: Phase VII-A reported AUC
               on a chain MDP that may have used Discrete(2) — verify
               metric comparability before claiming parity; if
               Phase VII-A used a different metric, P-VII-Parity is
               ABSENT (mark as N/A in the M6 summary, not a halt).
```

### 2.2 Update §11.2 specification

Append to the existing §11.2 specification block:

```text
Action-space note (added per v3 amendment):

  DC-Short10, DC-Medium20, DC-Long50:  Discrete(1) — single "advance"
    action. Policy is forced; β affects Q-value convergence speed but
    NOT episode return. Tested via q_convergence_rate metric, NOT AUC.

  DC-Branching20:  Discrete(2) — "advance" or "branch_wrong". Policy
    is β-dependent through the Q-value comparison. Tested via AUC
    metric (standard).
```

### 2.3 Update §11.7 trigger T11

Replace the existing T11 definition with:

```text
T11 — paper-critical prediction failure on delayed_chain (REVISED v3):

      For advance-only subcases (DC-Short10, DC-Medium20, DC-Long50):
          q_convergence_rate(+β) ≤ q_convergence_rate(0) on any
          advance-only subcase fires T11.

      For DC-Branching20:
          AUC(+β) ≤ AUC(0) on DC-Branching20 fires T11.

      T11 retains its paper-critical halt semantics: on fire, HALT
      for human review (NOT auto-fix), per addendum §6 BLOCKER
      semantics. The Codex bug-hunt review prompt is extended to
      include: "this contradicts the contraction-speed prediction
      for positive expected advantage on optimistically-initialized
      delayed-reward chains. Investigate (a) Q-init not sufficiently
      optimistic relative to V*, (b) episode horizon binding before
      terminal reward propagates back, (c) ε-greedy schedule
      preventing convergence within the horizon, (d) implementation
      off-by-one in chain transition, (e) implementation off-by-one
      in q_convergence_rate metric, (f) Q* analytical formula bug,
      OR (g) the prediction itself is theoretically misguided."
```

---

## 3. New metric: q_convergence_rate

Add to
`/Users/liq/Documents/Claude/Projects/LSE_RL/experiments/adaptive_beta/tab_six_games/metrics.py`:

```python
def q_convergence_rate(
    q_history: np.ndarray,           # shape (E, S, A), per-episode Q snapshots
    q_star: np.ndarray,              # shape (S, A), analytical optimum
    eps: float = 1e-8,
    norm: str = "linf",              # "linf" or "l2"
) -> np.ndarray:
    """Per-episode Q-convergence rate.

    Definition:
        rate_e = log(||Q_e - Q*|| + eps) - log(||Q_{e+1} - Q*|| + eps)

    Returns shape (E-1,) — one rate value per inter-episode step.

    The cumulative AUC of rate_e over episodes IS the headline metric
    for advance-only delayed_chain subcases. Use np.trapz on the
    cumulative rate sequence for a single-number summary; or report
    rate_e at fixed checkpoints (e.g., e=100, 1000, full horizon) for
    distributional reporting.

    Note: uses np.log directly with eps floor (NOT log1p; lessons.md
    #27 — expm1/log1p underflow on negative tails).
    """
    if q_history.ndim != 3:
        raise ValueError(f"q_history must be (E, S, A), got {q_history.shape}")
    if q_star.ndim != 2:
        raise ValueError(f"q_star must be (S, A), got {q_star.shape}")
    if q_history.shape[1:] != q_star.shape:
        raise ValueError(
            f"q_history (S, A) {q_history.shape[1:]} != q_star {q_star.shape}"
        )
    diffs = q_history - q_star[None, :, :]   # broadcast (E, S, A)
    if norm == "linf":
        residuals = np.abs(diffs).reshape(diffs.shape[0], -1).max(axis=1)
    elif norm == "l2":
        residuals = np.sqrt(np.sum(diffs ** 2, axis=(1, 2)))
    else:
        raise ValueError(f"unknown norm: {norm!r}")
    log_res = np.log(residuals + eps)
    rate = log_res[:-1] - log_res[1:]        # shape (E-1,)
    return rate.astype(np.float64)


def q_star_delayed_chain(L: int, gamma: float) -> np.ndarray:
    """Analytical Q* for advance-only delayed_chain of length L.

    Q*(s, advance) = γ^(L-s) for s ∈ [0, L-1]
    Q*(L, ·)       = 0  (terminal)

    Returns shape (L+1, 1) for Discrete(1) advance-only chain.
    """
    q = np.zeros((L + 1, 1), dtype=np.float64)
    for s in range(L):
        q[s, 0] = gamma ** (L - s)
    return q
```

Add corresponding tests to
`tests/adaptive_beta/tab_six_games/test_phase_VIII_metrics.py`:

```python
def test_q_convergence_rate_shape():
    rng = np.random.default_rng(0)
    q_hist = rng.normal(size=(100, 11, 1))
    q_star = q_star_delayed_chain(L=10, gamma=0.95)
    rate = q_convergence_rate(q_hist, q_star)
    assert rate.shape == (99,)
    assert rate.dtype == np.float64

def test_q_convergence_rate_monotone_under_perfect_decay():
    # Q approaches Q* exponentially fast → rate is positive and
    # roughly constant
    L, gamma = 10, 0.95
    q_star = q_star_delayed_chain(L, gamma)
    q_hist = q_star[None, :, :] + np.exp(-np.arange(100))[:, None, None]
    rate = q_convergence_rate(q_hist, q_star)
    assert np.all(rate > 0)
    # rate should be ~1 per step (since residual decays as e^-e)
    assert np.abs(rate.mean() - 1.0) < 0.1

def test_q_star_delayed_chain_geometric():
    L, gamma = 5, 0.9
    q_star = q_star_delayed_chain(L, gamma)
    # Q*(s=0, advance) = gamma^5
    assert np.isclose(q_star[0, 0], gamma ** L)
    # Q*(s=L, ·) = 0
    assert np.isclose(q_star[L, 0], 0.0)

def test_q_convergence_rate_eps_floor_safety():
    # Q exactly equal to Q* should not produce -inf or NaN
    L, gamma = 10, 0.95
    q_star = q_star_delayed_chain(L, gamma)
    q_hist = np.broadcast_to(q_star[None, :, :], (10, L+1, 1)).copy()
    rate = q_convergence_rate(q_hist, q_star)
    assert np.all(np.isfinite(rate))
    assert not np.any(np.isnan(rate))
```

---

## 4. Smoke test rewrite

Replace
`/Users/liq/Documents/Claude/Projects/LSE_RL/tests/adaptive_beta/strategic_games/test_delayed_chain_smoke_prediction.py`
content:

```python
"""Smoke test for delayed_chain advance-only Q-convergence prediction.

Per spec §5.7 P-Contract:
    On DC-Long50 with optimistic Q-init,
        q_convergence_rate(+β) > q_convergence_rate(0) > q_convergence_rate(-β)
    paired-bootstrap 95% CI strictly ordered.

Failure of this smoke test is a T11 trigger and HALTS the run for
human review per addendum §6.
"""
import numpy as np
import pytest

from experiments.adaptive_beta.agents import AdaptiveBetaQAgent, linear_epsilon_schedule
from experiments.adaptive_beta.schedules import (
    ZeroBetaSchedule, FixedBetaSchedule,
)
from experiments.adaptive_beta.strategic_games.matrix_game import MatrixGameEnv
from experiments.adaptive_beta.strategic_games.registry import (
    make_game, make_adversary,
)
from experiments.adaptive_beta.tab_six_games.metrics import (
    q_convergence_rate, q_star_delayed_chain,
)

# Notes on shapes used below:
#   q_history shape: (E, S, A) where E = num episodes, S = chain
#   length+1, A = 1 for advance-only.

@pytest.mark.smoke
def test_smoke_DC_Long50_q_convergence_ordering() -> None:
    """P-Contract on DC-Long50: AUC(rate(+1)) > AUC(rate(0)) > AUC(rate(-1))."""
    L = 50
    gamma = 0.95
    n_episodes = 1000
    n_seeds = 3
    eps = 1e-8

    q_star = q_star_delayed_chain(L=L, gamma=gamma)        # shape (L+1, 1)

    schedules = {
        "minus": FixedBetaSchedule(beta0=-1.0),
        "zero":  ZeroBetaSchedule(),
        "plus":  FixedBetaSchedule(beta0=+1.0),
    }
    auc_by_method = {k: [] for k in schedules}

    for seed in range(n_seeds):
        for name, schedule in schedules.items():
            game = make_game("delayed_chain", subcase="DC-Long50")
            opp = make_adversary("passive", seed=seed)
            env = MatrixGameEnv(game=game, adversary=opp, horizon=L, seed=seed)

            agent = AdaptiveBetaQAgent(
                n_states=L + 1,
                n_actions=1,
                gamma=gamma,
                learning_rate=0.1,
                epsilon_schedule=linear_epsilon_schedule(
                    start=0.0, end=0.0, n_steps=n_episodes,
                ),                                          # ε=0; forced policy anyway
                beta_schedule=schedule,
                rng=np.random.default_rng(seed),
                q_init=1.0,                                 # optimistic init
            )
            q_hist = np.zeros(
                (n_episodes, L + 1, 1), dtype=np.float64,
            )                                                # shape (E, S, A)
            for e in range(n_episodes):
                agent.begin_episode(e)
                state, _ = env.reset()
                done = False
                while not done:
                    action = agent.select_action(state)
                    next_state, reward, done, _ = env.step(action)
                    agent.step(state, action, reward, next_state, done)
                    state = next_state
                agent.end_episode()
                q_hist[e] = agent._Q                        # tabular Q snapshot

            rate = q_convergence_rate(
                q_hist, q_star, eps=eps, norm="linf",
            )                                                # shape (E-1,)
            auc = float(np.trapz(rate))                     # cumulative log-reduction
            auc_by_method[name].append(auc)

    auc_minus = np.array(auc_by_method["minus"])
    auc_zero  = np.array(auc_by_method["zero"])
    auc_plus  = np.array(auc_by_method["plus"])

    assert auc_plus.mean() > auc_zero.mean(), (
        f"P-Contract violated: AUC(rate(+1))={auc_plus.mean():.3f} "
        f"≤ AUC(rate(0))={auc_zero.mean():.3f}"
    )
    assert auc_zero.mean() > auc_minus.mean(), (
        f"P-Contract violated: AUC(rate(0))={auc_zero.mean():.3f} "
        f"≤ AUC(rate(-1))={auc_minus.mean():.3f}"
    )
    # Cohen's d > 0.3 on both gaps
    pooled_std = np.std(np.concatenate([auc_minus, auc_zero, auc_plus]))
    d_plus_zero = (auc_plus.mean() - auc_zero.mean()) / (pooled_std + 1e-12)
    d_zero_minus = (auc_zero.mean() - auc_minus.mean()) / (pooled_std + 1e-12)
    assert d_plus_zero > 0.3, f"Cohen's d (plus vs zero) = {d_plus_zero:.3f} ≤ 0.3"
    assert d_zero_minus > 0.3, f"Cohen's d (zero vs minus) = {d_zero_minus:.3f} ≤ 0.3"
```

If after applying the resolution this test STILL fails, that is a
genuine T11 fire (now with a testable metric) — HALT and surface
again. Do NOT auto-fix beyond the v3 amendment.

---

## 5. M6 sweep configuration update

The M6 wave 1 runner config (`stage1_beta_sweep.yaml`) must record
the metric per-subcase so the aggregator and plotter know which
y-axis to use:

```yaml
delayed_chain:
  subcases:
    DC-Short10:
      action_space: discrete_1
      headline_metric: q_convergence_rate
      L: 10
    DC-Medium20:
      action_space: discrete_1
      headline_metric: q_convergence_rate
      L: 20
    DC-Long50:
      action_space: discrete_1
      headline_metric: q_convergence_rate
      L: 50
    DC-Branching20:
      action_space: discrete_2
      headline_metric: AUC
      L: 20
```

The aggregator at
`experiments/adaptive_beta/tab_six_games/analysis/aggregate.py`
must read `headline_metric` per subcase and produce a separate
result column per metric type. Tables in
`results/adaptive_beta/tab_six_games/tables/` segregate q_convergence
results from AUC results to avoid apples-to-oranges comparisons.

---

## 6. Changelog

- 2026-05-01 v3 (T11 resolution): advance-only delayed_chain
  subcases switch headline metric from AUC to q_convergence_rate.
  DC-Branching20 retains AUC. P-Sign rewritten as P-Contract +
  P-Scaling + P-AUC-Branch + P-VII-Parity. T11 trigger semantics
  preserved (still paper-critical, still HALTS on fire). Adds
  q_convergence_rate + q_star_delayed_chain helpers to metrics.py.
  Rewrites smoke test. Adds aggregator metric-routing per subcase.
  Resolves halt at commit 09e7a262.
