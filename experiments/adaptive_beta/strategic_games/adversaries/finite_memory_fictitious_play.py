"""Finite-memory fictitious play (spec §7.4).

Classical fictitious play maintains a belief about the opponent's
mixed strategy as the empirical action frequency, then best-responds.
Here, **the adversary's "opponent" is the learning agent**, so the
belief is the agent's empirical action distribution over the last
``m`` rounds.

Distinction from ``FiniteMemoryBestResponse`` (§7.3): conceptually the
two are duals (FP best-responds to the agent's empirical play; FMBR is
also a best response to empirical play). The two differ in:

- FP can softly smooth via an explicit ``temperature > 0`` (kept here
  for parity with §7.4 "optional softmax"); FMBR adds an inertia
  knob (λ) on top.
- FP without smoothing is the canonical Brown / Robinson rule and is
  the textbook adversary in cycle-game studies (Shapley).

Implementation reuses the softmax / argmax helpers in
``finite_memory_best_response.py`` for numerical consistency.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from experiments.adaptive_beta.strategic_games.adversaries.base import (
    StrategicAdversary,
)
from experiments.adaptive_beta.strategic_games.adversaries.finite_memory_best_response import (
    _argmax_random,
    _softmax_stable,
)
from experiments.adaptive_beta.strategic_games.history import GameHistory


class FiniteMemoryFictitiousPlay(StrategicAdversary):
    """Best response to the agent's empirical action frequency.

    Parameters
    ----------
    payoff_opponent
        Opponent payoff matrix, shape ``(n_agent_actions, n_opponent_actions)``.
    memory_m
        Window length for the empirical-belief estimator. Set to a
        large value (e.g. 10**9) to recover unbounded fictitious play.
    temperature
        Optional softmax temperature for smoothed best response.
        ``0.0`` = pure argmax (canonical FP). ``> 0`` = quantal /
        smoothed FP (which is technically what §7.5 implements; we
        keep the knob here for the §7.4 "optional smoothing" clause).
    n_actions
        Opponent action-space cardinality.
    seed
        Integer seed.
    """

    adversary_type: str = "finite_memory_fictitious_play"

    def __init__(
        self,
        payoff_opponent: np.ndarray,
        memory_m: int,
        temperature: float = 0.0,
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
        if temperature < 0.0:
            raise ValueError(
                f"temperature must be >= 0, got {temperature}"
            )

        self._payoff_opponent: np.ndarray = po
        self._n_agent_actions: int = int(po.shape[0])

        n = int(po.shape[1]) if n_actions is None else int(n_actions)
        if n != po.shape[1]:
            raise ValueError(
                f"n_actions={n} != payoff_opponent.shape[1]={po.shape[1]}"
            )
        super().__init__(n_actions=n, seed=seed)

        self._memory_m: int = int(memory_m)
        self._temperature: float = float(temperature)
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

    def act(
        self,
        history: GameHistory,
        agent_action: Optional[int] = None,
    ) -> int:
        belief = history.empirical_agent_policy(
            m=self._memory_m, n_actions=self._n_agent_actions
        )  # (n_agent_actions,)
        q_br = belief @ self._payoff_opponent  # (n_opp_actions,)

        if self._temperature <= 0.0:
            chosen = _argmax_random(q_br, self._rng)
            policy = np.zeros(self.n_actions, dtype=np.float64)
            policy[chosen] = 1.0
        else:
            logits = q_br / self._temperature
            policy = _softmax_stable(logits)
            chosen = int(self._rng.choice(self.n_actions, p=policy))

        self._last_policy = policy
        return chosen

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
