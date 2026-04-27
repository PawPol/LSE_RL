"""Shapley cycle game (spec §6.2).

A canonical cyclic 3x3 environment in which simple learning rules
(fictitious play, finite-memory FP, regret matching) tend to enter
non-convergent action cycles. This is the central testbed for adaptive
β under endogenous cycling (spec §6.2).

Construction
------------
The Shapley game lives in a family of 3x3 games whose unique mixed
Nash is uniform. We use the *binding* anti-coordination resolution
documented in the dispatch instructions:

    payoff_agent[i][j]    = 1  if j == (i - 1) mod 3 else 0
    payoff_opponent[i][j] = 1  if j == (i + 1) mod 3 else 0

In words: the agent prefers to play one step "ahead" of the opponent
(mod 3); the opponent prefers to play one step "ahead" of the agent.
This creates a payoff cycle 0 → 1 → 2 → 0 → ... when both players
greedily best-respond to the other's last action — exactly the
Brown-Robinson cycling pathology.

Action encoding
---------------
- 0 = action A
- 1 = action B
- 2 = action C

The action labels carry no semantic meaning beyond the cyclic
structure of the payoff matrix.

Properties
----------
- Non-zero-sum (general-sum). Empirical frequencies converge to
  ``(1/3, 1/3, 1/3)`` under fictitious play, but pathwise actions
  cycle.
- ``canonical_sign = None``: no "optimistic vs pessimistic" alignment
  direction is privileged.

References
----------
- L. S. Shapley, "Some topics in two-person games" (1964) — original
  3x3 cycling example (different payoffs but the same cycling
  structure).
- H. Peyton Young, *Strategic Learning and Its Limits*, Ch. 2.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from experiments.adaptive_beta.strategic_games.adversaries.base import (
    StrategicAdversary,
)
from experiments.adaptive_beta.strategic_games.matrix_game import (
    MatrixGameEnv,
    StateEncoder,
    make_default_state_encoder,
)
from experiments.adaptive_beta.strategic_games.registry import register_game


GAME_NAME: str = "shapley"
N_ACTIONS: int = 3


def _build_payoff_matrices() -> tuple[np.ndarray, np.ndarray]:
    """Return ``(payoff_agent, payoff_opponent)`` for the Shapley game.

    See module docstring for the cyclic construction.
    """
    pa = np.zeros((N_ACTIONS, N_ACTIONS), dtype=np.float64)
    po = np.zeros((N_ACTIONS, N_ACTIONS), dtype=np.float64)
    for i in range(N_ACTIONS):
        pa[i, (i - 1) % N_ACTIONS] = 1.0
        po[i, (i + 1) % N_ACTIONS] = 1.0
    return pa, po


payoff_agent, payoff_opponent = _build_payoff_matrices()


def build(
    adversary: StrategicAdversary,
    horizon: int = 20,
    seed: Optional[int] = None,
    state_encoder: Optional[StateEncoder] = None,
    **kwargs: Any,
) -> MatrixGameEnv:
    """Construct a Shapley-cycle ``MatrixGameEnv``.

    Parameters
    ----------
    adversary
        Pre-built ``StrategicAdversary`` (must be 3-action). Spec §6.2
        canonical adversaries: fictitious play, smoothed fictitious
        play, finite-memory fictitious play, regret matching.
    horizon
        Episode length. Default 20.
    seed
        Optional integer seed.
    state_encoder
        Optional override. ``None`` selects
        ``make_default_state_encoder(horizon, 3)``.
    **kwargs
        Forwarded to ``MatrixGameEnv``. Supported: ``gamma``,
        ``metadata`` (merged on top of the game-supplied metadata).
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if adversary.n_actions != N_ACTIONS:
        raise ValueError(
            f"adversary.n_actions={adversary.n_actions} != {N_ACTIONS} (shapley)"
        )

    if state_encoder is None:
        encoder, n_states = make_default_state_encoder(
            horizon=horizon, n_actions=N_ACTIONS
        )
    else:
        encoder = state_encoder
        n_states = int(kwargs.pop("n_states", 1))

    metadata: dict = {
        "canonical_sign": None,
        "mechanism_degenerate": horizon == 1,
        "action_labels": ("A", "B", "C"),
        "is_zero_sum": False,
    }
    user_meta = kwargs.pop("metadata", None)
    if user_meta:
        metadata.update(user_meta)

    gamma = float(kwargs.pop("gamma", 0.95))
    if kwargs:
        raise TypeError(
            f"build(shapley) got unexpected kwargs: {sorted(kwargs)}"
        )

    return MatrixGameEnv(
        payoff_agent=payoff_agent,
        payoff_opponent=payoff_opponent,
        adversary=adversary,
        horizon=horizon,
        state_encoder=encoder,
        n_states=n_states,
        seed=seed,
        game_name=GAME_NAME,
        metadata=metadata,
        gamma=gamma,
    )


register_game(GAME_NAME, build)
