"""Stationary mixed-strategy opponent (spec §7.1).

A baseline adversary used for sanity checks — plays from a fixed
probability distribution over actions, ignoring history.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from experiments.adaptive_beta.strategic_games.adversaries.base import (
    StrategicAdversary,
)
from experiments.adaptive_beta.strategic_games.history import GameHistory


class StationaryMixedOpponent(StrategicAdversary):
    """Plays each action ``a`` with fixed probability ``probs[a]``.

    Parameters
    ----------
    probs
        Probability vector of length ``n_actions``. Validated to be
        non-negative and to sum to 1 within ``1e-9``.
    n_actions
        Action-space cardinality. Inferred from ``len(probs)`` if not
        passed.
    seed
        Optional integer seed; deterministic stream of actions across
        identical configurations.
    """

    adversary_type: str = "stationary_mixed"

    def __init__(
        self,
        probs: Sequence[float],
        n_actions: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        probs_arr = np.asarray(probs, dtype=np.float64)
        if probs_arr.ndim != 1:
            raise ValueError(
                f"probs must be 1-D, got shape {probs_arr.shape}"
            )
        if np.any(probs_arr < 0.0):
            raise ValueError("probs must be non-negative")
        total = float(probs_arr.sum())
        if not np.isfinite(total) or abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"probs must sum to 1 within 1e-9, got {total}"
            )
        n = int(probs_arr.shape[0]) if n_actions is None else int(n_actions)
        if n != probs_arr.shape[0]:
            raise ValueError(
                f"n_actions={n} != len(probs)={probs_arr.shape[0]}"
            )
        super().__init__(n_actions=n, seed=seed)
        # Snapshot a renormalised copy to absorb tiny FP drift.
        self._probs: np.ndarray = probs_arr / probs_arr.sum()  # shape (n_actions,)

    # ------------------------------------------------------------------
    # ABC interface
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)

    def act(
        self,
        history: GameHistory,
        agent_action: Optional[int] = None,
    ) -> int:
        # Stationary: history and agent_action ignored.
        return int(self._rng.choice(self.n_actions, p=self._probs))

    def observe(
        self,
        agent_action: int,
        opponent_action: int,
        agent_reward: float,
        opponent_reward: Optional[float] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        # No internal state to update.
        return None

    def info(self) -> Dict[str, Any]:
        return self._build_info(
            phase="stationary",
            policy_entropy=self._entropy(self._probs),
        )
