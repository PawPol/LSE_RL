"""Smoke test for delayed_chain advance-only Bellman residual decay.

Per spec §5.7 P-Contract (v5 per HALT 3 resolution: switch metric from
classical-Q* convergence (v3/v4) to β-specific Bellman residual
``||T_β Q − Q||_∞``; v5b per HALT 4 resolution: replace Cohen's d
guard with a relative-gap + absolute-floor pair appropriate for the
fully-deterministic DC-Long50 testbed):

    On DC-Long50 with optimistic Q-init,

        AUC(-log R_{β=-1}) > AUC(-log R_{β=0}) > AUC(-log R_{β=+1})

    Each inter-method gap clears an absolute floor of 100, AND the
    smaller gap is at least 10% of the larger gap.

Why v5
------
The v3/v4 metric ``q_convergence_rate(Q, Q*_classical)`` was biased
against β≠0: each TAB schedule has its OWN fixed point Q*_β (because
``g_{β,γ}(0,v) → (1+γ)·v`` as ``β→+∞`` rather than ``γ·v``), so a
residual against the classical Q* saturates at ``|Q*_β − Q*_0|`` and
appears non-converging even when the agent IS converging — to a
different fixed point. The β-specific Bellman residual
``||T_β Q − Q||_∞`` goes to zero at Q*_β regardless of β, which is the
mathematically clean test of "did this schedule converge at all" and
"how fast".

P-Contract (v5) prediction. Under optimistic Q-init
``Q_0 = 1/(1-γ) ≈ 20`` and reward ``r=0`` along the chain, the
alignment-condition residue ``β·(r-v)`` is:

    β = -1, v ≈ 20  →  β·(r-v) = -1·(0-20) = +20  (aligned)
    β =  0          →  classical (always)
    β = +1          →  β·(r-v) = +1·(0-20) = -20  (NOT aligned)

So β=-1 tightens contraction toward Q*_{-1}; β=+1 violates alignment
and amplifies the optimistic bootstrap.

Failure of this smoke test is a T11 trigger and HALTS the run for
human review per addendum §6 BLOCKER semantics. Do NOT auto-fix beyond
the v5 amendment itself; if it still fails, fall back to option (ζ)
(drop delayed_chain from M6 entirely).
"""

from __future__ import annotations

from typing import Callable, Iterable, Tuple

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
    build as build_delayed_chain,
)
from experiments.adaptive_beta.tab_six_games.metrics import (
    auc_neg_log_residual,
    bellman_residual_beta,
)


def _epsilon_schedule_zero():
    """ε ≡ 0 schedule. Policy is forced anyway (Discrete(1)) so ε is
    irrelevant; we explicitly pin it to 0 for determinism.

    ``linear_epsilon_schedule`` requires ``end <= start <= 1`` and
    ``decay_episodes > 0``; ``start=0, end=0, decay_episodes=1``
    satisfies that and yields a constant-zero ε.
    """
    return linear_epsilon_schedule(start=0.0, end=0.0, decay_episodes=1)


def _make_advance_only_transition(
    L: int,
) -> Callable[[int, int], Iterable[Tuple[float, float, int]]]:
    """Deterministic transition function for the advance-only chain.

    ``transition(s, a) -> Iterable[(prob, r, s')]`` matches the contract
    consumed by ``bellman_residual_beta``:

      * for ``s ∈ [0, L-1]``: ``s → s+1`` deterministically;
        reward ``+1`` iff ``s+1 == L``, else ``0``.
      * for ``s == L`` (terminal): self-loop with reward ``0``.

    Single-action chain so the action argument is irrelevant.
    """

    def transition(s: int, _a: int) -> Iterable[Tuple[float, float, int]]:
        if s >= L:
            yield (1.0, 0.0, L)
            return
        next_s = s + 1
        reward = 1.0 if next_s == L else 0.0
        yield (1.0, reward, next_s)

    return transition


@pytest.mark.smoke
def test_smoke_DC_Long50_bellman_residual_decay() -> None:
    """P-Contract (v5/v5b) on DC-Long50:
    AUC(-log R_{β=-1}) > AUC(-log R_{β=0}) > AUC(-log R_{β=+1}).

    Guard (v5b): each gap ≥ 100 absolute, AND the smaller gap is
    at least 10% of the larger. Replaces the v5 Cohen's d ≥ 0.3
    test, which is degenerate on this fully-deterministic testbed.
    """
    L = 50
    gamma = 0.95
    n_episodes = 1000
    n_seeds = 3
    eps = 1e-8
    n_states = L + 1
    n_actions = 1

    env_transition = _make_advance_only_transition(L)
    beta_value_by_name = {"minus": -1.0, "zero": 0.0, "plus": +1.0}
    schedules_factory = {
        "minus": lambda: FixedBetaSchedule(-1, hyperparams={"beta0": 1.0}),
        "zero":  lambda: ZeroBetaSchedule(),
        "plus":  lambda: FixedBetaSchedule(+1, hyperparams={"beta0": 1.0}),
    }
    auc_by_method: dict[str, list[float]] = {k: [] for k in schedules_factory}

    for seed in range(n_seeds):
        for name, make_schedule in schedules_factory.items():
            beta_value = beta_value_by_name[name]
            adversary = PassiveOpponent(n_actions=n_actions, seed=seed)
            env = build_delayed_chain(
                subcase="DC-Long50",
                adversary=adversary,
                seed=seed,
            )

            agent = AdaptiveBetaQAgent(
                n_states=n_states,
                n_actions=n_actions,
                gamma=gamma,
                learning_rate=0.1,
                epsilon_schedule=_epsilon_schedule_zero(),
                beta_schedule=make_schedule(),
                rng=np.random.default_rng(seed),
                q_init=1.0 / (1.0 - gamma),  # = 20 (optimistic)
            )

            residuals = np.zeros(n_episodes, dtype=np.float64)

            for e in range(n_episodes):
                agent.begin_episode(e)
                state_arr, _ = env.reset()
                state = int(np.asarray(state_arr).flat[0])
                done = False
                while not done:
                    action = agent.select_action(state, e)
                    next_state_arr, reward, done, _ = env.step(
                        np.array([action])
                    )
                    next_state = int(np.asarray(next_state_arr).flat[0])
                    agent.step(
                        state=state,
                        action=action,
                        reward=float(reward),
                        next_state=next_state,
                        absorbing=bool(done),
                        episode_index=e,
                    )
                    state = next_state
                agent.end_episode(e)

                # Override terminal-state Q-value to 0 (boundary
                # condition): the agent never updates Q[L,:] because
                # state L is absorbing and the runner stops issuing
                # actions there, so it retains the optimistic init
                # (~20) which would otherwise leak into the
                # T_β Q(L-1, 0) target via max Q[L,:]. The true
                # terminal value is 0 by MDP convention.
                Q_corrected = agent.Q.copy()
                Q_corrected[L, :] = 0.0

                residuals[e] = bellman_residual_beta(
                    Q=Q_corrected,
                    beta=beta_value,
                    gamma=gamma,
                    env_transition=env_transition,
                    n_states=n_states,
                    n_actions=n_actions,
                )

            auc_by_method[name].append(
                auc_neg_log_residual(residuals, eps=eps)
            )

    auc_minus = np.array(auc_by_method["minus"])
    auc_zero = np.array(auc_by_method["zero"])
    auc_plus = np.array(auc_by_method["plus"])

    assert auc_minus.mean() > auc_zero.mean(), (
        f"P-Contract (v5) violated: AUC(-log R_-1)={auc_minus.mean():.4f} "
        f"<= AUC(-log R_0)={auc_zero.mean():.4f}"
    )
    assert auc_zero.mean() > auc_plus.mean(), (
        f"P-Contract (v5) violated: AUC(-log R_0)={auc_zero.mean():.4f} "
        f"<= AUC(-log R_+1)={auc_plus.mean():.4f}"
    )

    # The DC-Long50 advance-only chain is fully deterministic
    # (Discrete(1) action + ε=0 + PassiveOpponent + deterministic
    # transitions). Per-seed AUCs are bit-identical so Cohen's d
    # is degenerate (intra-method σ = 0). Use a relative-gap floor
    # instead: the smaller of the two inter-method gaps must be at
    # least 10% of the larger gap, AND each gap must clear an
    # absolute floor of 100. This guards against degenerate cases
    # where one method dominates by orders of magnitude and the
    # others are nearly tied — which would still pass mean-ordering
    # but would indicate the prediction is essentially binary
    # (alignment-violator vs everyone-else) rather than the
    # three-tier ordering the alignment condition predicts.
    gap_minus_zero = float(auc_minus.mean() - auc_zero.mean())
    gap_zero_plus = float(auc_zero.mean() - auc_plus.mean())
    gap_small = min(gap_minus_zero, gap_zero_plus)
    gap_large = max(gap_minus_zero, gap_zero_plus)
    assert gap_small >= 100.0, (
        f"v5b absolute-floor guard failed: smaller gap {gap_small:.1f} < 100"
    )
    assert gap_small >= 0.10 * gap_large, (
        f"v5b relative-gap guard failed: ratio "
        f"{gap_small / gap_large:.3f} < 0.10 "
        f"(gaps: minus->zero={gap_minus_zero:.1f}, "
        f"zero->plus={gap_zero_plus:.1f})"
    )
