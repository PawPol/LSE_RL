"""Sign-prediction smoke for the Phase VIII §5.7 delayed-reward chain.

Spec authority: ``docs/specs/phase_VIII_tab_six_games.md`` §5.7
(``<!-- patch-2026-05-01 §11 -->``); upstream patch
``tasks/phase_VIII_spec_patches_2026-05-01.md`` §11.4 (smoke spec) and
§11.7 (T11 trigger semantics).

Falsifiable prediction tested
-----------------------------
``P-Sign``: on ``DC-Long50`` with optimistic Q-init,

    AUC(+1) > AUC(0) > AUC(-1)

with at least Cohen's d > 0.3 on each gap. ``AUC`` here is the sum of
per-episode returns over a 1k-episode budget (3 seeds).

Test policy on failure
----------------------
**Failure does NOT auto-fix.** Per patch §11.4: "smoke-prediction
failures need researcher attention rather than implementer auto-fix".
A failure is a candidate ``T11`` bug-hunt trigger (patch §11.7) — it
contradicts the alignment-condition prediction for positive expected
advantage on delayed-reward chains and warrants human review. The test
is marked ``@pytest.mark.smoke`` so it can be selectively deselected
under tight CI budgets while remaining part of milestone-close sweeps.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import pytest

from experiments.adaptive_beta.agents import (
    AdaptiveBetaQAgent,
    linear_epsilon_schedule,
)
from experiments.adaptive_beta.schedules import (
    FixedBetaSchedule,
    ZeroBetaSchedule,
)
from experiments.adaptive_beta.strategic_games.adversaries.passive import (
    PassiveOpponent,
)
from experiments.adaptive_beta.strategic_games.games.delayed_chain import (
    SUBCASE_LONG50,
    build,
)


# ---------------------------------------------------------------------------
# Smoke configuration (patch §11.4)
# ---------------------------------------------------------------------------
N_SEEDS: int = 3
N_EPISODES: int = 1_000
GAMMA: float = 0.95
LR: float = 0.5             # tabular-chain default; large LR speeds credit propagation
EPSILON_DECAY_EPS: int = 500

# Optimistic Q-init: roughly 1/(1-γ) so backward propagation from the +1
# terminal differentiates +β / 0 / -β cleanly within the budget.
OPTIMISTIC_Q_INIT: float = 1.0 / (1.0 - GAMMA)   # = 20.0 at γ=0.95

# Cohen's d threshold per patch §11.4. Note: this is a smoke-level
# threshold; the production statistical comparison in M6 uses paired-
# bootstrap CIs (spec §12).
MIN_COHENS_D: float = 0.3


def _build_schedule(beta: float):
    """Construct a fixed-β schedule (vanilla for β==0)."""
    if beta == 0.0:
        return ZeroBetaSchedule(hyperparams=None)
    sign = +1 if beta > 0 else -1
    return FixedBetaSchedule(sign, hyperparams={"beta0": abs(beta)})


def _train_one_seed(
    *,
    beta: float,
    seed: int,
    subcase: str,
    n_episodes: int,
) -> float:
    """Run one (β, seed) training trajectory; return AUC = Σ per-episode return.

    AUC is the simple cumulative return; for a tabular chain with a
    sparse +1 terminal it equals the count of "successful" episodes
    minus penalties (the trap delivers -1 on DC-Branching only; on
    DC-Long50 the AUC is exactly the success count).
    """
    adv = PassiveOpponent(n_actions=1)
    env = build(subcase=subcase, adversary=adv, seed=seed, gamma=GAMMA)

    agent = AdaptiveBetaQAgent(
        n_states=env.info.observation_space.size[0],
        n_actions=env.info.action_space.size[0],
        gamma=GAMMA,
        learning_rate=LR,
        epsilon_schedule=linear_epsilon_schedule(
            start=1.0, end=0.05, decay_episodes=EPSILON_DECAY_EPS
        ),
        beta_schedule=_build_schedule(beta),
        rng=np.random.default_rng(seed),
        env_canonical_sign=env.env_canonical_sign,
        q_init=OPTIMISTIC_Q_INIT,
    )

    auc: float = 0.0
    for e in range(n_episodes):
        state, _ = env.reset()
        s_int = int(np.asarray(state).flat[0])
        agent.begin_episode(e)
        episode_return: float = 0.0
        done = False
        while not done:
            a = agent.select_action(s_int, e)
            ns, r, done, _ = env.step(np.asarray([a], dtype=np.int64))
            ns_int = int(np.asarray(ns).flat[0])
            agent.step(
                state=s_int,
                action=a,
                reward=float(r),
                next_state=ns_int,
                absorbing=bool(done),
                episode_index=e,
            )
            s_int = ns_int
            episode_return += float(r)
        agent.end_episode(e)
        auc += episode_return

    return auc


def _cohens_d(a: List[float], b: List[float]) -> float:
    """Cohen's d for two independent samples with pooled std.

    d = (mean_a - mean_b) / sqrt((var_a + var_b) / 2)
    Returns ``+inf`` (or ``-inf``) when both samples have zero variance
    and means differ; ``0.0`` when both samples are identical.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    mean_diff = float(a_arr.mean() - b_arr.mean())
    pooled = math.sqrt((float(a_arr.var(ddof=0)) + float(b_arr.var(ddof=0))) / 2.0)
    if pooled <= 0.0:
        if mean_diff > 0:
            return float("inf")
        if mean_diff < 0:
            return float("-inf")
        return 0.0
    return mean_diff / pooled


def _train_grid(
    betas: Tuple[float, ...],
    *,
    subcase: str,
    n_seeds: int = N_SEEDS,
    n_episodes: int = N_EPISODES,
) -> dict:
    """Run β × seeds, return ``{β: [auc_per_seed, ...]}``."""
    out: dict = {}
    for beta in betas:
        per_seed: List[float] = []
        for s in range(n_seeds):
            per_seed.append(
                _train_one_seed(
                    beta=beta,
                    seed=s,
                    subcase=subcase,
                    n_episodes=n_episodes,
                )
            )
        out[beta] = per_seed
    return out


# ---------------------------------------------------------------------------
# 1. test_smoke_DC_Long50_AUC_ordering
# ---------------------------------------------------------------------------
@pytest.mark.smoke
def test_smoke_DC_Long50_AUC_ordering() -> None:
    """``AUC(+1) > AUC(0) > AUC(-1)`` on DC-Long50 with Cohen's d > 0.3.

    Failure of this test is a candidate ``T11`` bug-hunt trigger
    (patch §11.7). Do NOT auto-fix — flag for researcher attention.
    """
    grid = _train_grid(
        (-1.0, 0.0, +1.0),
        subcase=SUBCASE_LONG50,
        n_seeds=N_SEEDS,
        n_episodes=N_EPISODES,
    )

    auc_neg = grid[-1.0]
    auc_zero = grid[0.0]
    auc_pos = grid[+1.0]

    mean_neg = float(np.mean(auc_neg))
    mean_zero = float(np.mean(auc_zero))
    mean_pos = float(np.mean(auc_pos))

    # ----- Sign-ordering (mean-level) -------------------------------------
    assert mean_pos > mean_zero, (
        f"AUC(+1)={mean_pos:.3f} <= AUC(0)={mean_zero:.3f}; "
        f"P-Sign violated on DC-Long50. T11 trigger candidate "
        f"(patch §11.7); do NOT auto-fix. per-seed: "
        f"+1={auc_pos}, 0={auc_zero}, -1={auc_neg}"
    )
    assert mean_zero > mean_neg, (
        f"AUC(0)={mean_zero:.3f} <= AUC(-1)={mean_neg:.3f}; "
        f"P-Sign violated on DC-Long50. T11 trigger candidate "
        f"(patch §11.7); do NOT auto-fix. per-seed: "
        f"+1={auc_pos}, 0={auc_zero}, -1={auc_neg}"
    )

    # ----- Cohen's d on each gap ------------------------------------------
    d_pos_zero = _cohens_d(auc_pos, auc_zero)
    d_zero_neg = _cohens_d(auc_zero, auc_neg)

    assert d_pos_zero > MIN_COHENS_D, (
        f"Cohen's d (+1 vs 0) = {d_pos_zero:.3f} <= {MIN_COHENS_D}; "
        f"effect size below smoke threshold. T11 trigger candidate. "
        f"per-seed: +1={auc_pos}, 0={auc_zero}"
    )
    assert d_zero_neg > MIN_COHENS_D, (
        f"Cohen's d (0 vs -1) = {d_zero_neg:.3f} <= {MIN_COHENS_D}; "
        f"effect size below smoke threshold. T11 trigger candidate. "
        f"per-seed: 0={auc_zero}, -1={auc_neg}"
    )
