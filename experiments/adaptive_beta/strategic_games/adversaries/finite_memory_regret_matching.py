"""Finite-memory regret-matching adversary (spec §7.7).

Same regret-matching recursion as ``RegretMatching`` (§7.6), but
regrets are recomputed from scratch over the **last ``m`` observed
joint actions** at every step. This drops cumulative weight on
ancient rounds and creates **endogenous support shifts** — the
discontinuity that makes finite-memory RM a stronger nonstationary
adversary than vanilla cumulative RM.

Implementation note
-------------------
Two natural designs exist:

(A) Sliding window: store the last ``m`` joint actions; recompute the
    regret vector from the window at each step. Cost: ``O(m·A)`` per
    step.
(B) Discounted regret (``regret ← γ·regret + Δ``) with effective
    horizon ``m``. Cost: ``O(A)`` per step.

Spec §7.7 prefers (A) ("regrets are computed over the last ``m``
observations only"), so that's what we implement. ``m`` is typically
small enough (≤ 100 in the spec grid) that this is negligible.
"""

from __future__ import annotations

from typing import Any, Deque, Dict, Optional, Tuple
from collections import deque

import numpy as np

from experiments.adaptive_beta.strategic_games.adversaries.base import (
    StrategicAdversary,
)
from experiments.adaptive_beta.strategic_games.adversaries.regret_matching import (
    _regret_to_policy,
)
from experiments.adaptive_beta.strategic_games.history import GameHistory


class FiniteMemoryRegretMatching(StrategicAdversary):
    """Regret matching over a sliding window of the last ``m`` joint actions.

    Parameters
    ----------
    payoff_opponent
        Shape ``(n_agent_actions, n_opponent_actions)``.
    memory_m
        Window length. Spec §7.7 uses small ``m`` (5–100) to create
        endogenous support shifts.
    n_actions
        Opponent action-space cardinality.
    seed
        Integer seed.
    """

    adversary_type: str = "finite_memory_regret_matching"

    def __init__(
        self,
        payoff_opponent: np.ndarray,
        memory_m: int,
        n_actions: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        po = np.asarray(payoff_opponent, dtype=np.float64)
        if po.ndim != 2:
            raise ValueError(
                f"payoff_opponent must be 2-D, got shape {po.shape}"
            )
        if memory_m <= 0:
            raise ValueError(f"memory_m must be >= 1, got {memory_m}")

        self._payoff_opponent: np.ndarray = po
        self._n_agent_actions: int = int(po.shape[0])

        n = int(po.shape[1]) if n_actions is None else int(n_actions)
        if n != po.shape[1]:
            raise ValueError(
                f"n_actions={n} != payoff_opponent.shape[1]={po.shape[1]}"
            )
        super().__init__(n_actions=n, seed=seed)

        self._memory_m: int = int(memory_m)
        # Sliding window of (agent_action, opponent_action) pairs.
        self._window: Deque[Tuple[int, int]] = deque(maxlen=self._memory_m)
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
        self._window.clear()
        self._last_policy = np.full(
            self.n_actions, 1.0 / self.n_actions, dtype=np.float64
        )

    def _window_regret(self) -> np.ndarray:
        """Recompute the regret vector from the current window.

        Output shape: ``(n_actions,)``.
        """
        regret = np.zeros(self.n_actions, dtype=np.float64)
        if len(self._window) == 0:
            return regret
        for a, b in self._window:
            cf = self._payoff_opponent[a, :]  # shape (n_actions,)
            regret += cf - cf[b]
        return regret

    def act(
        self,
        history: GameHistory,
        agent_action: Optional[int] = None,
    ) -> int:
        regret = self._window_regret()
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
        a = int(np.asarray(agent_action).flat[0])
        b = int(np.asarray(opponent_action).flat[0])
        # ``deque(maxlen=m)`` evicts the oldest pair automatically.
        self._window.append((a, b))

    def info(self) -> Dict[str, Any]:
        return self._build_info(
            phase=None,
            memory_m=self._memory_m,
            policy_entropy=self._entropy(self._last_policy),
        )
