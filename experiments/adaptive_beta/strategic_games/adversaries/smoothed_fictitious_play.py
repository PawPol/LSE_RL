"""Smoothed (logit / quantal) fictitious play (spec §7.5).

The opponent best-responds to the agent's empirical action frequency
under a logit / quantal-response policy:

    π(j) ∝ exp(Q_br[j] / temperature)

with log-sum-exp stabilisation via ``np.logaddexp.reduce`` for
numerical robustness at large ``|Q_br|``.

This is the canonical adversary for cycle-game studies (Shapley, etc.):
non-degenerate ``temperature`` smooths the FP step and gives the
operator a Lipschitz best-response.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from experiments.adaptive_beta.strategic_games.adversaries.base import (
    StrategicAdversary,
)
from experiments.adaptive_beta.strategic_games.history import GameHistory


def _logsumexp_stable(logits: np.ndarray) -> float:
    """Stable log-sum-exp using ``numpy.logaddexp.reduce``.

    For a 1-D array, this is equivalent to
    ``np.log(np.sum(np.exp(logits)))`` but never overflows. Empty
    inputs raise ``ValueError`` (caller should validate shape).
    """
    if logits.size == 0:
        raise ValueError("_logsumexp_stable requires a non-empty input")
    return float(np.logaddexp.reduce(logits))


class SmoothedFictitiousPlay(StrategicAdversary):
    """Logit / quantal best response to the agent's empirical play.

    Parameters
    ----------
    payoff_opponent
        Opponent payoff matrix of shape ``(n_agent_actions, n_opponent_actions)``.
    temperature
        Logit temperature ``τ > 0``. Smaller ``τ`` → sharper best
        response; ``τ → ∞`` → uniform. Spec §7.5 grid: ``{0.05, 0.2, 1.0}``.
    memory_m
        Empirical-belief window. ``None`` (default) = unbounded
        fictitious play; integer ``> 0`` = finite memory.
    n_actions
        Opponent action-space cardinality.
    seed
        Integer seed.
    """

    adversary_type: str = "smoothed_fictitious_play"

    def __init__(
        self,
        payoff_opponent: np.ndarray,
        temperature: float,
        memory_m: Optional[int] = None,
        n_actions: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        po = np.asarray(payoff_opponent, dtype=np.float64)
        if po.ndim != 2:
            raise ValueError(
                f"payoff_opponent must be 2-D, got shape {po.shape}"
            )
        if temperature <= 0.0:
            raise ValueError(
                f"temperature must be > 0 for smoothed FP, got {temperature}"
            )
        if memory_m is not None and memory_m <= 0:
            raise ValueError(
                f"memory_m must be >= 1 or None, got {memory_m}"
            )

        self._payoff_opponent: np.ndarray = po
        self._n_agent_actions: int = int(po.shape[0])

        n = int(po.shape[1]) if n_actions is None else int(n_actions)
        if n != po.shape[1]:
            raise ValueError(
                f"n_actions={n} != payoff_opponent.shape[1]={po.shape[1]}"
            )
        super().__init__(n_actions=n, seed=seed)

        self._temperature: float = float(temperature)
        self._memory_m: Optional[int] = (
            None if memory_m is None else int(memory_m)
        )
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
        self._last_policy = np.full(
            self.n_actions, 1.0 / self.n_actions, dtype=np.float64
        )

    def _logit_policy(self, q_br: np.ndarray) -> np.ndarray:
        """Compute the logit / quantal-response policy with stable LSE.

        ``q_br`` shape: ``(n_actions,)``. Returns shape ``(n_actions,)``.
        """
        logits = q_br / self._temperature  # (n_actions,)
        # log π(j) = logits[j] − log Σ_k exp(logits[k])
        log_z = _logsumexp_stable(logits)
        log_pi = logits - log_z
        # Renormalise after exp for FP error robustness.
        pi = np.exp(log_pi)
        s = float(pi.sum())
        if s <= 0.0 or not np.isfinite(s):
            return np.full(self.n_actions, 1.0 / self.n_actions, dtype=np.float64)
        return pi / s

    def act(
        self,
        history: GameHistory,
        agent_action: Optional[int] = None,
    ) -> int:
        belief = history.empirical_agent_policy(
            m=self._memory_m, n_actions=self._n_agent_actions
        )  # (n_agent_actions,)
        q_br = belief @ self._payoff_opponent  # (n_opp_actions,)
        policy = self._logit_policy(q_br)
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
        return None

    def info(self) -> Dict[str, Any]:
        return self._build_info(
            phase=None,
            memory_m=self._memory_m,
            temperature=self._temperature,
            policy_entropy=self._entropy(self._last_policy),
        )
