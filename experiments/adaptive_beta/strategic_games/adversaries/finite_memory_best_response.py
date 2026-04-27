"""Finite-memory best-response adversary (spec §7.3).

At each step:
1. Estimate the agent's empirical action distribution over the last
   ``m`` rounds (from ``GameHistory``).
2. Compute the opponent's best response to that distribution under the
   provided opponent payoff matrix.
3. With probability ``inertia_lambda``, repeat the previous opponent
   action (lock-in / hysteresis).
4. Otherwise, with probability ``temperature``-controlled softmax,
   smooth the best response.

Payoff convention
-----------------
``payoff_opponent`` is a 2-D array of shape ``(n_agent_actions,
n_opponent_actions)``: ``payoff_opponent[i, j]`` is the opponent's
realised payoff when agent plays ``i`` and opponent plays ``j``.

The expected opponent payoff for opponent action ``j`` against the
agent's empirical distribution ``p_agent`` is::

    Q_br[j] = sum_i p_agent[i] * payoff_opponent[i, j]

The best response is ``argmax_j Q_br[j]`` (ties broken by
``self._rng``). When ``temperature > 0`` and not inertial, the
adversary samples from the softmax of ``Q_br / temperature`` (with
log-sum-exp stabilisation).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from experiments.adaptive_beta.strategic_games.adversaries.base import (
    StrategicAdversary,
)
from experiments.adaptive_beta.strategic_games.history import GameHistory


def _argmax_random(values: np.ndarray, rng: np.random.Generator) -> int:
    """Argmax with random tie-breaking via ``rng``.

    ``values`` is a 1-D array. Returns an integer index.
    """
    vmax = float(values.max())
    ties = np.flatnonzero(values >= vmax - 1e-12)
    if ties.size == 1:
        return int(ties[0])
    return int(rng.choice(ties))


def _softmax_stable(logits: np.ndarray) -> np.ndarray:
    """Softmax with log-sum-exp stabilization.

    Input: 1-D ``logits``. Output: 1-D probability vector summing to 1.
    """
    x = np.asarray(logits, dtype=np.float64)
    if x.size == 0:
        return x
    m = float(x.max())
    shifted = x - m
    expx = np.exp(shifted)
    z = float(expx.sum())
    if z <= 0.0 or not np.isfinite(z):
        # Degenerate: fall back to uniform.
        return np.full(x.shape, 1.0 / x.size, dtype=np.float64)
    return expx / z


class FiniteMemoryBestResponse(StrategicAdversary):
    """Finite-memory best response with optional inertia and softmax.

    Parameters
    ----------
    payoff_opponent
        Opponent payoff matrix, shape ``(n_agent_actions, n_opponent_actions)``.
    memory_m
        Number of recent agent actions used for the empirical estimate.
        Spec §7.3 grid: ``{5, 20, 100}``.
    inertia_lambda
        Probability ``λ ∈ [0, 1]`` of repeating the previous opponent
        action. ``0.0`` = no inertia. Spec grid: ``{0.0, 0.5, 0.9}``.
    temperature
        Softmax temperature. ``0.0`` = pure (random-tie-broken) argmax;
        ``> 0`` samples from the logit distribution. Spec grid:
        ``{0.05, 0.2, 1.0}``.
    n_actions
        Opponent action-space cardinality. Inferred from
        ``payoff_opponent.shape[1]`` if omitted.
    seed
        Integer seed.
    """

    adversary_type: str = "finite_memory_best_response"

    def __init__(
        self,
        payoff_opponent: np.ndarray,
        memory_m: int,
        inertia_lambda: float = 0.0,
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
        if not (0.0 <= inertia_lambda <= 1.0):
            raise ValueError(
                f"inertia_lambda must lie in [0, 1], got {inertia_lambda}"
            )
        if temperature < 0.0:
            raise ValueError(
                f"temperature must be >= 0, got {temperature}"
            )

        self._payoff_opponent: np.ndarray = po  # (n_agent_actions, n_opp_actions)
        self._n_agent_actions: int = int(po.shape[0])

        n = int(po.shape[1]) if n_actions is None else int(n_actions)
        if n != po.shape[1]:
            raise ValueError(
                f"n_actions={n} != payoff_opponent.shape[1]={po.shape[1]}"
            )
        super().__init__(n_actions=n, seed=seed)

        self._memory_m: int = int(memory_m)
        self._inertia_lambda: float = float(inertia_lambda)
        self._temperature: float = float(temperature)
        self._last_action: Optional[int] = None
        # Cache the most recent policy vector for entropy reporting in info().
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
        self._last_action = None
        self._last_policy = np.full(
            self.n_actions, 1.0 / self.n_actions, dtype=np.float64
        )

    def _compute_qbr(self, history: GameHistory) -> np.ndarray:
        """Expected opponent payoff per opponent action (shape ``(n_actions,)``)."""
        p_agent = history.empirical_agent_policy(
            m=self._memory_m, n_actions=self._n_agent_actions
        )
        # Q_br[j] = sum_i p_agent[i] * payoff_opponent[i, j]   shape (n_opp_actions,)
        return p_agent @ self._payoff_opponent

    def act(
        self,
        history: GameHistory,
        agent_action: Optional[int] = None,
    ) -> int:
        # Inertia first: repeat last action with probability λ.
        if self._last_action is not None and self._inertia_lambda > 0.0:
            if self._rng.random() < self._inertia_lambda:
                # Update cached policy for info(): degenerate at last_action.
                policy = np.zeros(self.n_actions, dtype=np.float64)
                policy[self._last_action] = 1.0
                self._last_policy = policy
                return int(self._last_action)

        q_br = self._compute_qbr(history)  # (n_actions,)

        if self._temperature <= 0.0:
            # Hard argmax with random tie-breaking.
            chosen = _argmax_random(q_br, self._rng)
            policy = np.zeros(self.n_actions, dtype=np.float64)
            policy[chosen] = 1.0
        else:
            logits = q_br / self._temperature
            policy = _softmax_stable(logits)
            chosen = int(self._rng.choice(self.n_actions, p=policy))

        self._last_policy = policy
        self._last_action = chosen
        return chosen

    def observe(
        self,
        agent_action: int,
        opponent_action: int,
        agent_reward: float,
        opponent_reward: Optional[float] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        # _last_action is already updated in ``act``; nothing to do here.
        # Kept as a no-op so the env can call ``observe`` uniformly.
        return None

    def info(self) -> Dict[str, Any]:
        return self._build_info(
            phase=None,
            memory_m=self._memory_m,
            inertia_lambda=self._inertia_lambda,
            temperature=self._temperature,
            policy_entropy=self._entropy(self._last_policy),
        )
