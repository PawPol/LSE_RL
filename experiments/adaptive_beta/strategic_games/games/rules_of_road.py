"""Rules-of-the-road coordination game (spec §6.3).

A 2x2 pure-coordination game with two strict Nash equilibria
((Left, Left) and (Right, Right)). The factory exposes two variant
knobs documented in the spec:

- ``tremble_prob`` — probability that the adversary's chosen action is
  perturbed by an independent Bernoulli flip BEFORE it reaches the
  payoff matrix. Models small execution noise / "trembling-hand"
  perturbations. Spec §6.3 grid: ``{0.0, 0.05, 0.10}``.

- ``payoff_bias`` — additional reward added to the (Right, Right) cell
  in BOTH players' payoff matrices. ``payoff_bias > 0`` makes
  Right-Right the unique payoff-dominant equilibrium. Spec §6.3 grid:
  ``{0.0, 0.5}``.

Action encoding
---------------
- 0 = Left
- 1 = Right

Base payoff matrices (before bias):

    payoff_agent    = [[+1, -1],
                       [-1, +1]]
    payoff_opponent = +payoff_agent     # symmetric coordination

Canonical sign
--------------
``None``. Coordination payoffs are symmetric around the diagonal —
neither "optimistic" (push-toward-cooperation) nor "pessimistic"
(push-away-from-miscoordination) is privileged in our adaptive-β
mechanism (spec §22.3). ``wrong_sign`` / ``adaptive_magnitude_only``
schedules raise on construction against this game.

Tremble implementation
----------------------
The tremble is realised by wrapping the user-supplied adversary with a
``_TrembleAdversary`` decorator that draws an independent Bernoulli
flip from the env-level RNG (passed in via the seed). The decorator
preserves all of the underlying adversary's ``observe`` / ``info``
semantics — the inner adversary still observes the *post-tremble*
opponent action, which is the "physical" action played, matching the
Phase VII-B observability convention (spec §5.2 / §7).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from experiments.adaptive_beta.strategic_games.adversaries.base import (
    StrategicAdversary,
)
from experiments.adaptive_beta.strategic_games.history import GameHistory
from experiments.adaptive_beta.strategic_games.matrix_game import (
    MatrixGameEnv,
    StateEncoder,
    make_default_state_encoder,
)
from experiments.adaptive_beta.strategic_games.registry import register_game


GAME_NAME: str = "rules_of_road"
N_ACTIONS: int = 2

payoff_agent: np.ndarray = np.array(
    [
        [+1.0, -1.0],
        [-1.0, +1.0],
    ],
    dtype=np.float64,
)

payoff_opponent: np.ndarray = payoff_agent.copy()


class _TrembleAdversary(StrategicAdversary):
    """Wraps an adversary with a Bernoulli-flip on the chosen action.

    Used by the ``rules_of_road`` factory when ``tremble_prob > 0``.
    The wrapper:

    - Forwards ``reset`` to the inner adversary, with a derived
      sub-seed for the tremble RNG so that runs remain deterministic.
    - In ``act``, samples an inner action then flips it with
      probability ``tremble_prob``.
    - In ``observe``, forwards the realised (post-tremble) opponent
      action to the inner adversary.
    - Surfaces ``tremble_prob`` in ``info()`` for logging.
    """

    adversary_type: str = "tremble_wrapper"

    def __init__(
        self,
        inner: StrategicAdversary,
        tremble_prob: float,
        seed: Optional[int] = None,
    ) -> None:
        if not (0.0 <= tremble_prob <= 1.0):
            raise ValueError(
                f"tremble_prob must lie in [0, 1], got {tremble_prob}"
            )
        super().__init__(n_actions=inner.n_actions, seed=seed)
        self._inner: StrategicAdversary = inner
        self._tremble_prob: float = float(tremble_prob)
        # The wrapper's adversary_type is reported via info(), but the
        # logger usually wants the inner type — surface it as an extra
        # field so consumers can audit either.
        self._inner_type: str = inner.adversary_type

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._seed = int(seed)
        # Derive independent sub-seeds for the tremble RNG and the
        # inner adversary so that they don't interleave.
        if self._seed is None:
            self._rng = np.random.default_rng()
            self._inner.reset(seed=None)
        else:
            ss = np.random.SeedSequence(self._seed)
            tremble_seed_arr, inner_seed_arr = ss.spawn(2)
            self._rng = np.random.default_rng(tremble_seed_arr)
            inner_seed = int(
                np.random.default_rng(inner_seed_arr).integers(
                    0, 2**31 - 1
                )
            )
            self._inner.reset(seed=inner_seed)

    def act(
        self,
        history: GameHistory,
        agent_action: Optional[int] = None,
    ) -> int:
        a = int(self._inner.act(history, agent_action=agent_action))
        if self._tremble_prob > 0.0 and self._rng.random() < self._tremble_prob:
            # Bernoulli flip across the action set. For 2-action this is
            # the canonical "trembling hand" flip; for k > 2 we draw
            # uniformly from the OTHER k - 1 actions.
            if self.n_actions == 2:
                a = 1 - a
            else:
                others = [i for i in range(self.n_actions) if i != a]
                a = int(self._rng.choice(others))
        return a

    def observe(
        self,
        agent_action: int,
        opponent_action: int,
        agent_reward: float,
        opponent_reward: Optional[float] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Forward the realised (post-tremble) opponent action to the
        # inner adversary so its history-based estimators stay coherent.
        self._inner.observe(
            agent_action=agent_action,
            opponent_action=opponent_action,
            agent_reward=agent_reward,
            opponent_reward=opponent_reward,
            info=info,
        )

    def info(self) -> Dict[str, Any]:
        inner_info = dict(self._inner.info())
        # Augment with tremble-wrapper provenance.
        inner_info["tremble_prob"] = self._tremble_prob
        inner_info["inner_adversary_type"] = self._inner_type
        # Override the surface adversary_type so logs reflect the wrapper.
        inner_info["adversary_type"] = self.adversary_type
        return inner_info

    # Forward optional hooks so phase-clock adversaries keep working.
    def on_episode_end(self) -> None:
        hook = getattr(self._inner, "on_episode_end", None)
        if callable(hook):
            hook()


def build(
    adversary: StrategicAdversary,
    horizon: int = 20,
    seed: Optional[int] = None,
    state_encoder: Optional[StateEncoder] = None,
    tremble_prob: float = 0.0,
    payoff_bias: float = 0.0,
    **kwargs: Any,
) -> MatrixGameEnv:
    """Construct a rules-of-the-road ``MatrixGameEnv``.

    Parameters
    ----------
    adversary
        Pre-built ``StrategicAdversary`` (must be 2-action).
    horizon
        Episode length. Default 20.
    seed
        Optional integer seed (also used to seed the tremble RNG when
        ``tremble_prob > 0``).
    state_encoder
        Optional override. ``None`` uses
        ``make_default_state_encoder(horizon, 2)``.
    tremble_prob
        Probability of an independent action flip applied to the
        opponent's chosen action. Spec grid: ``{0.0, 0.05, 0.10}``.
    payoff_bias
        Additional payoff added to the (Right, Right) cell in BOTH
        players' matrices. Spec grid: ``{0.0, 0.5}``.
    **kwargs
        Forwarded to ``MatrixGameEnv``. Supported: ``gamma``,
        ``metadata`` (merged on top of the game-supplied metadata).
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if adversary.n_actions != N_ACTIONS:
        raise ValueError(
            f"adversary.n_actions={adversary.n_actions} != {N_ACTIONS} "
            f"(rules_of_road)"
        )
    if not (0.0 <= tremble_prob <= 1.0):
        raise ValueError(
            f"tremble_prob must lie in [0, 1], got {tremble_prob}"
        )

    # Apply the payoff bias (right-right cell) on a fresh copy so the
    # module-level constants stay unchanged.
    pa = payoff_agent.copy()
    po = payoff_opponent.copy()
    if payoff_bias != 0.0:
        pa[1, 1] += float(payoff_bias)
        po[1, 1] += float(payoff_bias)

    # Wrap with the tremble decorator iff requested.
    if tremble_prob > 0.0:
        wrapped: StrategicAdversary = _TrembleAdversary(
            inner=adversary,
            tremble_prob=tremble_prob,
            seed=seed,
        )
    else:
        wrapped = adversary

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
        "action_labels": ("left", "right"),
        "is_zero_sum": False,
        "tremble_prob": float(tremble_prob),
        "payoff_bias": float(payoff_bias),
    }
    user_meta = kwargs.pop("metadata", None)
    if user_meta:
        metadata.update(user_meta)

    gamma = float(kwargs.pop("gamma", 0.95))
    if kwargs:
        raise TypeError(
            f"build(rules_of_road) got unexpected kwargs: {sorted(kwargs)}"
        )

    return MatrixGameEnv(
        payoff_agent=pa,
        payoff_opponent=po,
        adversary=wrapped,
        horizon=horizon,
        state_encoder=encoder,
        n_states=n_states,
        seed=seed,
        game_name=GAME_NAME,
        metadata=metadata,
        gamma=gamma,
    )


register_game(GAME_NAME, build)
