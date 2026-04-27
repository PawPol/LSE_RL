"""Strategic Rock-Paper-Scissors variant (spec §6.5, OQ-2 resolution).

This is the **strategic** RPS variant for Phase VII-B: a standard 3x3
zero-sum RPS game played against an *endogenous* adversary (finite-
memory regret matching, hypothesis testing, finite-memory best
response, ...). It is implemented as a ``MatrixGameEnv`` and is
distinct from the Phase VII-A scripted-phase RPS environment in
``experiments/adaptive_beta/envs/rps.py`` — that file remains
untouched and is used for the regression test that locks the Phase
VII-A claims.

Action encoding
---------------
- 0 = rock
- 1 = paper
- 2 = scissors

Payoff (agent's perspective):

    payoff_agent = [[ 0, -1, +1],
                    [+1,  0, -1],
                    [-1, +1,  0]]
    payoff_opponent = -payoff_agent

The standard "cyclic +1 / -1 / 0" RPS payoff is recovered: row ``i``
beats opponent action ``(i - 1) mod 3`` (e.g. paper beats rock).

Canonical sign
--------------
``None`` (zero-sum). ``wrong_sign`` and ``adaptive_magnitude_only``
schedules raise on construction (spec §22.3).

Adversary dispatch
------------------
The factory accepts an ``adversary_name`` string and dispatches via
``make_adversary(name, **adversary_kwargs)``. Spec §6.5 lists the
canonical adversaries: ``finite_memory_regret_matching``,
``hypothesis_testing``, ``finite_memory_best_response``,
``smoothed_fictitious_play``. Alternatively, the caller may pass an
already-constructed ``adversary`` instance and the factory will skip
dispatch.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from experiments.adaptive_beta.strategic_games.adversaries.base import (
    StrategicAdversary,
)
from experiments.adaptive_beta.strategic_games.matrix_game import (
    MatrixGameEnv,
    StateEncoder,
    make_default_state_encoder,
)
from experiments.adaptive_beta.strategic_games.registry import (
    make_adversary,
    register_game,
)


GAME_NAME: str = "strategic_rps"
N_ACTIONS: int = 3

payoff_agent: np.ndarray = np.array(
    [
        [0.0, -1.0, +1.0],
        [+1.0, 0.0, -1.0],
        [-1.0, +1.0, 0.0],
    ],
    dtype=np.float64,
)

payoff_opponent: np.ndarray = -payoff_agent


def build(
    adversary: Optional[StrategicAdversary] = None,
    horizon: int = 20,
    seed: Optional[int] = None,
    state_encoder: Optional[StateEncoder] = None,
    adversary_name: Optional[str] = None,
    adversary_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> MatrixGameEnv:
    """Construct a strategic-RPS ``MatrixGameEnv``.

    Parameters
    ----------
    adversary
        Optional pre-built ``StrategicAdversary``. If supplied,
        ``adversary_name`` and ``adversary_kwargs`` are ignored.
    horizon
        Episode length. Default 20 (spec §6.5).
    seed
        Optional integer seed.
    state_encoder
        Optional override. ``None`` uses
        ``make_default_state_encoder(horizon, 3)``.
    adversary_name
        Adversary registry key (e.g.
        ``"finite_memory_regret_matching"``,
        ``"hypothesis_testing"``,
        ``"finite_memory_best_response"``,
        ``"smoothed_fictitious_play"``). Required when ``adversary`` is
        ``None``.
    adversary_kwargs
        Keyword arguments forwarded to ``make_adversary``. Common
        fields: ``payoff_opponent`` (the env's opponent payoff matrix —
        injected automatically if absent), ``memory_m``,
        ``temperature``, ``test_window_s``, ``tolerance_tau``,
        ``search_len``, ``seed``, ``n_actions``.
    **kwargs
        Forwarded to ``MatrixGameEnv``. Supported: ``gamma``,
        ``metadata`` (merged on top of the game-supplied metadata).
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    if adversary is None:
        if adversary_name is None:
            raise ValueError(
                "build(strategic_rps) requires either ``adversary`` or "
                "``adversary_name`` to be provided"
            )
        adv_kw: Dict[str, Any] = (
            dict(adversary_kwargs) if adversary_kwargs else {}
        )
        # Inject the canonical opponent payoff matrix and action
        # cardinality if the caller did not pass them. The adversary
        # registry honours ``payoff_opponent`` and ``n_actions``.
        adv_kw.setdefault("payoff_opponent", payoff_opponent)
        adv_kw.setdefault("n_actions", N_ACTIONS)
        if seed is not None:
            adv_kw.setdefault("seed", int(seed))
        adversary = make_adversary(adversary_name, **adv_kw)

    if adversary.n_actions != N_ACTIONS:
        raise ValueError(
            f"adversary.n_actions={adversary.n_actions} != {N_ACTIONS} "
            f"(strategic_rps)"
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
        "action_labels": ("rock", "paper", "scissors"),
        "is_zero_sum": True,
        "adversary_name": adversary_name,
    }
    user_meta = kwargs.pop("metadata", None)
    if user_meta:
        metadata.update(user_meta)

    gamma = float(kwargs.pop("gamma", 0.95))
    if kwargs:
        raise TypeError(
            f"build(strategic_rps) got unexpected kwargs: {sorted(kwargs)}"
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
