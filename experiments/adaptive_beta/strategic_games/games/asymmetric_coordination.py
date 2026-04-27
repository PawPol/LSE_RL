"""Asymmetric coordination trap (spec §6.4).

A 2x2 stag-hunt-style coordination game with two strict equilibria:

- (Stag, Stag) at action ``(0, 0)`` — payoff-dominant (both get
  ``coop_payoff = 5``).
- (Hare, Hare) at action ``(1, 1)`` — risk-dominant (both get
  ``risk_payoff = 3``); also obtained by mismatched (Hare, Stag) play
  for the agent (it always gets ``risk_payoff`` when it plays Hare).

Action encoding
---------------
- 0 = Stag (cooperative / payoff-dominant)
- 1 = Hare (defect / risk-dominant)

Payoff matrices
---------------
With defaults ``coop_payoff = 5``, ``risk_payoff = 3``:

    payoff_agent    = [[+5, +0],
                       [+3, +3]]
    payoff_opponent = payoff_agent.T
                    = [[+5, +3],
                       [+0, +3]]

Best-response structure:
- If opponent plays Stag (col 0) → agent's best is Stag (5 > 3).
- If opponent plays Hare (col 1) → agent's best is Hare (3 > 0).

The (Stag, Stag) equilibrium dominates in payoff but is "risky": a
miscoordinated Stag costs the agent ``coop_payoff − 0`` worth of
expected reward if the opponent defects. The (Hare, Hare) equilibrium
is risk-dominant and is the typical attractor of myopic best-response
dynamics.

Canonical sign
--------------
``+1`` (encoded as ``"+"`` per ``schedules._canonical_sign_to_value``).
The optimistic-β direction is the one that propagates "value of
cooperation" forward — adaptive β should push positive when the agent
nudges into the payoff-dominant equilibrium, so the canonical sign is
``+`` (spec §22.3, ``delayed_chain`` precedent). With this tag,
``wrong_sign`` and ``adaptive_magnitude_only`` schedules are
*permitted* by the schedule factory.
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


GAME_NAME: str = "asymmetric_coordination"
N_ACTIONS: int = 2

# Default coefficients (spec §6.4 stag-hunt instantiation).
DEFAULT_COOP_PAYOFF: float = 5.0
DEFAULT_RISK_PAYOFF: float = 3.0


def _build_payoff_matrices(
    coop_payoff: float, risk_payoff: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(payoff_agent, payoff_opponent)`` for stag-hunt.

    Layout (rows = agent action, cols = opponent action; 0 = Stag,
    1 = Hare):

        payoff_agent = [[coop_payoff, 0          ],
                        [risk_payoff, risk_payoff]]
        payoff_opponent = payoff_agent.T
    """
    pa = np.array(
        [
            [float(coop_payoff), 0.0],
            [float(risk_payoff), float(risk_payoff)],
        ],
        dtype=np.float64,
    )
    po = pa.T.copy()
    return pa, po


# Module-level constants instantiated at the spec defaults.
payoff_agent, payoff_opponent = _build_payoff_matrices(
    DEFAULT_COOP_PAYOFF, DEFAULT_RISK_PAYOFF
)


def build(
    adversary: StrategicAdversary,
    horizon: int = 20,
    seed: Optional[int] = None,
    state_encoder: Optional[StateEncoder] = None,
    coop_payoff: float = DEFAULT_COOP_PAYOFF,
    risk_payoff: float = DEFAULT_RISK_PAYOFF,
    **kwargs: Any,
) -> MatrixGameEnv:
    """Construct an asymmetric-coordination ``MatrixGameEnv``.

    Parameters
    ----------
    adversary
        Pre-built ``StrategicAdversary`` (must be 2-action).
    horizon
        Episode length. Default 20.
    seed
        Optional integer seed.
    state_encoder
        Optional override. ``None`` uses
        ``make_default_state_encoder(horizon, 2)``.
    coop_payoff
        Reward at the payoff-dominant equilibrium (Stag, Stag).
        Default 5.
    risk_payoff
        Reward at the risk-dominant equilibrium (Hare, Hare). Also the
        reward whenever the agent plays Hare. Default 3. Must satisfy
        ``coop_payoff > risk_payoff > 0`` for the stag-hunt geometry to
        hold.
    **kwargs
        Forwarded to ``MatrixGameEnv``. Supported: ``gamma``,
        ``metadata`` (merged on top of the game-supplied metadata).
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if adversary.n_actions != N_ACTIONS:
        raise ValueError(
            f"adversary.n_actions={adversary.n_actions} != {N_ACTIONS} "
            f"(asymmetric_coordination)"
        )
    if not (coop_payoff > risk_payoff > 0):
        raise ValueError(
            "asymmetric_coordination requires coop_payoff > risk_payoff > 0; "
            f"got coop={coop_payoff}, risk={risk_payoff}"
        )

    pa, po = _build_payoff_matrices(coop_payoff, risk_payoff)

    if state_encoder is None:
        encoder, n_states = make_default_state_encoder(
            horizon=horizon, n_actions=N_ACTIONS
        )
    else:
        encoder = state_encoder
        n_states = int(kwargs.pop("n_states", 1))

    metadata: dict = {
        "canonical_sign": "+",
        "mechanism_degenerate": horizon == 1,
        "action_labels": ("stag", "hare"),
        "is_zero_sum": False,
        "coop_payoff": float(coop_payoff),
        "risk_payoff": float(risk_payoff),
    }
    user_meta = kwargs.pop("metadata", None)
    if user_meta:
        metadata.update(user_meta)

    gamma = float(kwargs.pop("gamma", 0.95))
    if kwargs:
        raise TypeError(
            f"build(asymmetric_coordination) got unexpected kwargs: "
            f"{sorted(kwargs)}"
        )

    env = MatrixGameEnv(
        payoff_agent=pa,
        payoff_opponent=po,
        adversary=adversary,
        horizon=horizon,
        state_encoder=encoder,
        n_states=n_states,
        seed=seed,
        game_name=GAME_NAME,
        metadata=metadata,
        gamma=gamma,
    )
    # Override the env-level attribute so downstream consumers
    # (schedules.build_schedule) read the correct sign even if they
    # don't peek into metadata.
    env.env_canonical_sign = "+"
    return env


register_game(GAME_NAME, build)
