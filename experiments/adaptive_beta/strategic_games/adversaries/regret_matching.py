"""Regret-matching adversary (spec §7.6).

Maintains cumulative regret per opponent action. At each step:

    positive_regret = max(regret, 0)
    if sum(positive_regret) > 0:
        policy = positive_regret / sum(positive_regret)
    else:
        policy = uniform

Two information modes are supported:

- ``mode='full_info'`` — counterfactual payoffs are read directly from
  the supplied ``payoff_opponent`` matrix:

      regret[j] += payoff_opponent[a_t, j] − payoff_opponent[a_t, b_t]

  where ``a_t`` is the agent's realised action and ``b_t`` is the
  opponent's realised action at time ``t``. This is the canonical
  regret-matching recursion (Hart & Mas-Colell, 2000).

- ``mode='realized_payoff'`` — the adversary does not observe the
  full payoff matrix counterfactuals. Instead, it maintains a running
  estimate of the per-action expected payoff against the agent's
  empirical play, and uses that estimate to compute regrets:

      Q_hat[j] = (1 - lr) * Q_hat[j] + lr * payoff_opponent[a_t, j]    (full update)
      regret[j] += Q_hat[j] − Q_hat[b_t]

  This is a softer approximation; the spec §7.9 separately defines a
  more aggressive realised-payoff regret variant with an
  experimentation rate (``RealizedPayoffRegret`` — stub in this
  dispatch).

Determinism: the only stochastic operation is sampling from the
regret-matching policy via ``self._rng``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from experiments.adaptive_beta.strategic_games.adversaries.base import (
    StrategicAdversary,
)
from experiments.adaptive_beta.strategic_games.history import GameHistory


_VALID_MODES = ("full_info", "realized_payoff")


def _regret_to_policy(regret: np.ndarray) -> np.ndarray:
    """Convert a regret vector into a regret-matching policy.

    ``regret`` shape: ``(n_actions,)``. Returns ``(n_actions,)``.
    Uniform fallback when ``Σ max(regret, 0) ≤ 0``.
    """
    pos = np.maximum(regret, 0.0)
    s = float(pos.sum())
    if s <= 0.0 or not np.isfinite(s):
        n = regret.shape[0]
        return np.full(n, 1.0 / n, dtype=np.float64)
    return pos / s


class RegretMatching(StrategicAdversary):
    """Cumulative regret matching with full-info or realised-payoff updates.

    Parameters
    ----------
    payoff_opponent
        Shape ``(n_agent_actions, n_opponent_actions)``. Required even
        in ``realized_payoff`` mode — the env hands realised opponent
        payoffs back via ``observe``, but we use the matrix to read
        counterfactual rows.
    mode
        ``'full_info'`` (default) or ``'realized_payoff'``.
    value_lr
        Step size for the running per-action payoff estimate when
        ``mode='realized_payoff'``. Ignored in ``full_info`` mode.
    n_actions
        Opponent action-space cardinality.
    seed
        Integer seed.
    """

    adversary_type: str = "regret_matching"

    def __init__(
        self,
        payoff_opponent: np.ndarray,
        mode: str = "full_info",
        value_lr: float = 0.05,
        n_actions: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        po = np.asarray(payoff_opponent, dtype=np.float64)
        if po.ndim != 2:
            raise ValueError(
                f"payoff_opponent must be 2-D, got shape {po.shape}"
            )
        if mode not in _VALID_MODES:
            raise ValueError(
                f"mode must be one of {_VALID_MODES}, got {mode!r}"
            )
        if not (0.0 < value_lr <= 1.0):
            raise ValueError(
                f"value_lr must lie in (0, 1], got {value_lr}"
            )

        self._payoff_opponent: np.ndarray = po
        self._n_agent_actions: int = int(po.shape[0])

        n = int(po.shape[1]) if n_actions is None else int(n_actions)
        if n != po.shape[1]:
            raise ValueError(
                f"n_actions={n} != payoff_opponent.shape[1]={po.shape[1]}"
            )
        super().__init__(n_actions=n, seed=seed)

        self._mode: str = mode
        self._value_lr: float = float(value_lr)
        self._regret: np.ndarray = np.zeros(self.n_actions, dtype=np.float64)
        self._q_hat: np.ndarray = np.zeros(self.n_actions, dtype=np.float64)
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
        self._regret.fill(0.0)
        self._q_hat.fill(0.0)
        self._last_policy = np.full(
            self.n_actions, 1.0 / self.n_actions, dtype=np.float64
        )

    def act(
        self,
        history: GameHistory,
        agent_action: Optional[int] = None,
    ) -> int:
        policy = _regret_to_policy(self._regret)
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
        a = int(np.asarray(agent_action).flat[0])
        b = int(np.asarray(opponent_action).flat[0])

        # Counterfactual opponent payoff vector against agent's a_t.
        cf = self._payoff_opponent[a, :]  # shape (n_opp_actions,)
        realised = float(cf[b])

        if self._mode == "full_info":
            # regret[j] += payoff_opponent[a, j] − payoff_opponent[a, b]
            self._regret += cf - realised
        else:  # realized_payoff
            # Update running per-action payoff estimate against this row.
            self._q_hat = (
                (1.0 - self._value_lr) * self._q_hat + self._value_lr * cf
            )
            # Regret update against the realised payoff under the same row.
            self._regret += self._q_hat - self._q_hat[b]

    def info(self) -> Dict[str, Any]:
        return self._build_info(
            phase=self._mode,
            policy_entropy=self._entropy(self._last_policy),
        )
