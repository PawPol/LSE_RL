"""Realised-payoff-only regret adversary (spec §7.9 — OPTIONAL).

# OPTIONAL — minimal impl, defer to follow-up if not exercised by Dev

Spec §7.9 marks this adversary as optional for the first pass. We
ship a minimal implementation here so:

(a) the registry can declare a factory for it (no ``KeyError`` on
    config typos that reference §7.9), and
(b) Stage B2-Dev can opt in if a (game, adversary) cell is added.

Behaviour
---------
- Maintains an ``n_actions``-vector of action-value estimates ``Q_hat``,
  updated only from realised payoffs (no counterfactual reads):

      Q_hat[b_t] ← (1 − value_lr) * Q_hat[b_t] + value_lr * realised_payoff

- With probability ``epsilon_experiment``, plays a uniform-random
  action (forces exploration so all entries of ``Q_hat`` are visited).
- Otherwise, samples from the regret-matching policy on
  ``regret[j] = Q_hat[j] − Q_hat[b_last]`` against the most recent
  realised opponent action.

This is the spec §7.9 "realised payoff only, occasional
experimentation" rule, in its simplest form. A more aggressive
variant (e.g. stochastic-approximation regret with optimistic
initialisation) is left for the follow-up.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from experiments.adaptive_beta.strategic_games.adversaries.base import (
    StrategicAdversary,
)
from experiments.adaptive_beta.strategic_games.adversaries.regret_matching import (
    _regret_to_policy,
)
from experiments.adaptive_beta.strategic_games.history import GameHistory


class RealizedPayoffRegret(StrategicAdversary):
    """Realised-payoff-only regret adversary (minimal §7.9 impl).

    Parameters
    ----------
    n_actions
        Opponent action-space cardinality. Required (no payoff matrix
        is consumed by this adversary).
    epsilon_experiment
        Per-step uniform-experimentation probability. Spec grid:
        ``{0.01, 0.05, 0.10}``.
    value_lr
        Step size for the realised-payoff estimator. Spec grid:
        ``{0.01, 0.05, 0.10}``.
    memory_m
        Recorded for ``info()`` reporting consistency with the §5.2
        block; not used in the regret update directly. Spec grid:
        ``{20, 100}``.
    seed
        Integer seed.
    """

    adversary_type: str = "realized_payoff_regret"

    def __init__(
        self,
        n_actions: int,
        epsilon_experiment: float = 0.05,
        value_lr: float = 0.05,
        memory_m: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        if not (0.0 <= epsilon_experiment <= 1.0):
            raise ValueError(
                f"epsilon_experiment must lie in [0, 1], got {epsilon_experiment}"
            )
        if not (0.0 < value_lr <= 1.0):
            raise ValueError(
                f"value_lr must lie in (0, 1], got {value_lr}"
            )
        if memory_m is not None and memory_m <= 0:
            raise ValueError(
                f"memory_m must be >= 1 or None, got {memory_m}"
            )
        super().__init__(n_actions=n_actions, seed=seed)

        self._epsilon: float = float(epsilon_experiment)
        self._value_lr: float = float(value_lr)
        self._memory_m: Optional[int] = (
            None if memory_m is None else int(memory_m)
        )
        self._q_hat: np.ndarray = np.zeros(self.n_actions, dtype=np.float64)
        self._last_opponent_action: Optional[int] = None
        self._last_policy: np.ndarray = np.full(
            self.n_actions, 1.0 / self.n_actions, dtype=np.float64
        )

    # ------------------------------------------------------------------
    # ABC interface
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)
        self._q_hat.fill(0.0)
        self._last_opponent_action = None
        self._last_policy = np.full(
            self.n_actions, 1.0 / self.n_actions, dtype=np.float64
        )

    def act(
        self,
        history: GameHistory,
        agent_action: Optional[int] = None,
    ) -> int:
        if self._rng.random() < self._epsilon:
            policy = np.full(self.n_actions, 1.0 / self.n_actions, dtype=np.float64)
            chosen = int(self._rng.choice(self.n_actions, p=policy))
            self._last_policy = policy
            return chosen

        if self._last_opponent_action is None:
            policy = np.full(self.n_actions, 1.0 / self.n_actions, dtype=np.float64)
        else:
            baseline = float(self._q_hat[self._last_opponent_action])
            regret = self._q_hat - baseline
            policy = _regret_to_policy(regret)
        self._last_policy = policy
        return int(self._rng.choice(self.n_actions, p=policy))

    def observe(
        self,
        agent_action: int,
        opponent_action: int,
        agent_reward: float,
        opponent_reward: Optional[float] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        b = int(np.asarray(opponent_action).flat[0])
        if opponent_reward is None:
            # No realised payoff signal — skip the update. This makes
            # the adversary degenerate; logged for the caller to detect.
            return None
        r_opp = float(np.asarray(opponent_reward).flat[0])
        # Realised-payoff update only on the played row.
        self._q_hat[b] = (
            (1.0 - self._value_lr) * self._q_hat[b]
            + self._value_lr * r_opp
        )
        self._last_opponent_action = b

    def info(self) -> Dict[str, Any]:
        return self._build_info(
            phase="exploit",
            memory_m=self._memory_m,
            temperature=None,
            policy_entropy=self._entropy(self._last_policy),
            epsilon_experiment=self._epsilon,
            value_lr=self._value_lr,
        )
