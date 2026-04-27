"""Matching pennies (spec §6.1).

Two-action zero-sum game. The agent wins (+1) when its action matches
the opponent's, loses (-1) otherwise.

Action encoding
---------------
- 0 = Heads
- 1 = Tails

Payoff matrices (agent's perspective in ``payoff_agent``):

    payoff_agent    = [[+1, -1],
                       [-1, +1]]
    payoff_opponent = -payoff_agent     # zero-sum

Default horizon
---------------
The default horizon is 1. At ``H = 1`` the game is mechanism-degenerate
for adaptive-β analysis (no in-episode value-of-next signal), so the
factory tags ``metadata['mechanism_degenerate'] = True`` whenever
``horizon == 1``. The plotter inspects this flag and excludes the run
from mechanism panels (spec §22.3 precedent).

Canonical sign
--------------
``None`` — zero-sum games have no "optimistic vs pessimistic"
alignment direction (spec §22.3). ``wrong_sign`` /
``adaptive_magnitude_only`` schedules will raise from the schedule
factory when constructed against this game.
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
    default_single_state_encoder,
    make_default_state_encoder,
)
from experiments.adaptive_beta.strategic_games.registry import register_game


GAME_NAME: str = "matching_pennies"

payoff_agent: np.ndarray = np.array(
    [
        [+1.0, -1.0],
        [-1.0, +1.0],
    ],
    dtype=np.float64,
)

payoff_opponent: np.ndarray = -payoff_agent


def build(
    adversary: StrategicAdversary,
    horizon: int = 1,
    seed: Optional[int] = None,
    state_encoder: Optional[StateEncoder] = None,
    **kwargs: Any,
) -> MatrixGameEnv:
    """Construct a matching-pennies ``MatrixGameEnv``.

    Parameters
    ----------
    adversary
        Pre-built ``StrategicAdversary`` (must be 2-action). The caller
        owns the adversary's payoff-matrix configuration; the env only
        owns its lifecycle (reset / act / observe).
    horizon
        Episode length. Default 1. ``horizon == 1`` flips the
        ``mechanism_degenerate`` metadata flag.
    seed
        Optional integer seed propagated into the env's RNG and the
        adversary on first reset.
    state_encoder
        Optional override. ``None`` selects the default:
        single-dummy-state encoder when ``horizon == 1``, else the
        ``make_default_state_encoder(horizon, 2)`` encoder.
    **kwargs
        Forwarded to ``MatrixGameEnv``. Supported: ``gamma``,
        ``metadata`` (merged on top of the game-supplied metadata).

    Returns
    -------
    env : MatrixGameEnv
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if adversary.n_actions != payoff_agent.shape[1]:
        raise ValueError(
            f"adversary.n_actions={adversary.n_actions} != 2 (matching pennies)"
        )

    if state_encoder is None:
        if horizon == 1:
            encoder: StateEncoder = default_single_state_encoder
            n_states: int = 1
        else:
            encoder, n_states = make_default_state_encoder(
                horizon=horizon, n_actions=int(payoff_opponent.shape[1])
            )
    else:
        encoder = state_encoder
        n_states = int(kwargs.pop("n_states", 1))

    metadata: dict = {
        "canonical_sign": None,
        "mechanism_degenerate": horizon == 1,
        "action_labels": ("heads", "tails"),
        "is_zero_sum": True,
    }
    user_meta = kwargs.pop("metadata", None)
    if user_meta:
        metadata.update(user_meta)

    gamma = float(kwargs.pop("gamma", 0.95))
    if kwargs:
        raise TypeError(
            f"build(matching_pennies) got unexpected kwargs: {sorted(kwargs)}"
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


# Register at import time (spec §4 contract).
register_game(GAME_NAME, build)
