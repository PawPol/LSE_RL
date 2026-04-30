"""Sticky-action ("inertia") opponent (Phase VIII spec §5.7).

Spec authority: ``docs/specs/phase_VIII_tab_six_games.md`` §5.7
(adversary catalog) — used by the Asymmetric-Coordination subcase
``AC-Inertia``.

Behaviour
---------
At each step the opponent either repeats its previous action with
probability ``inertia_lambda`` or samples a fresh uniform-random action
with probability ``(1 - inertia_lambda)``. The first action of an
episode (no previous action available) is uniform random.

This produces a positively-autocorrelated action stream whose
short-window empirical distribution drifts slowly relative to the
agent's update horizon — the regime that motivates β-adaptive operators
in the AC-Inertia subcase (the agent must distinguish a momentary
opponent commitment from genuine policy change).

Determinism
-----------
All sampling routes through ``self._rng``; ``reset(seed=...)`` re-seeds
both the generator and clears ``last_action``, so two adversaries
configured with the same ``(n_actions, inertia_lambda, seed)`` produce
byte-identical action streams under identical histories.

Spec compliance
---------------
- Inherits ``StrategicAdversary`` (spec §5.2 ABC).
- ``info()`` returns the full ``ADVERSARY_INFO_KEYS`` set with
  ``phase="inertia"`` and ``inertia_lambda`` populated; non-applicable
  fields (``memory_m``, ``temperature``, ``hypothesis_id``) are ``None``.
- ``last_action`` is exposed as an extra ``info`` field for diagnostic
  / oracle β use.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from experiments.adaptive_beta.strategic_games.adversaries.base import (
    StrategicAdversary,
)
from experiments.adaptive_beta.strategic_games.history import GameHistory


class InertiaOpponent(StrategicAdversary):
    """Sticky-action opponent with inertia parameter ``inertia_lambda``.

    At step ``t``:
        with probability ``inertia_lambda`` play ``last_action``,
        otherwise sample ``a ~ Uniform({0, ..., n_actions - 1})``.
    The first call (no ``last_action`` yet) always samples uniformly.

    Parameters
    ----------
    n_actions
        Action-space cardinality (``int``, ``>= 1``).
    inertia_lambda
        Probability of repeating the previous action. Must lie in
        ``[0, 1]`` (boundary inclusive: ``0.0`` reduces to a uniform
        adversary, ``1.0`` to a constant adversary that locks onto its
        first random draw forever).
    seed
        Optional integer seed. ``None`` produces a non-deterministic
        stream (default ``np.random.default_rng()``).

    Notes
    -----
    The "fresh uniform draw" branch is allowed to coincidentally match
    ``last_action`` — that is consistent with the spec text ("plays a
    random action") and yields the cleanest statistical interpretation
    (action distribution at stationarity equals the uniform).
    """

    adversary_type: str = "inertia"

    def __init__(
        self,
        n_actions: int,
        inertia_lambda: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        # Validate inertia_lambda BEFORE delegating to the parent (which
        # validates n_actions). We also reject NaN explicitly: the
        # ``not 0.0 <= x <= 1.0`` idiom returns True for NaN, but being
        # explicit makes the error message informative.
        lam = float(inertia_lambda)
        if not np.isfinite(lam) or lam < 0.0 or lam > 1.0:
            raise ValueError(
                f"inertia_lambda must lie in [0, 1], got {inertia_lambda!r}"
            )
        super().__init__(n_actions=n_actions, seed=seed)
        self._inertia_lambda: float = lam
        self._last_action: Optional[int] = None

    # ------------------------------------------------------------------
    # ABC interface (spec §5.2)
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> None:
        """Re-seed the RNG and clear ``last_action``.

        Identical ``(n_actions, inertia_lambda, seed)`` triples produce
        identical action streams under identical histories.
        """
        if seed is not None:
            self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)
        self._last_action = None

    def act(
        self,
        history: GameHistory,
        agent_action: Optional[int] = None,
    ) -> int:
        """Return the next opponent action.

        The ``history`` argument is ignored — sticky-action depends only
        on the adversary's own ``last_action``. The ``agent_action``
        argument is also ignored (production-mode adversary; no peeking).

        Implementation note (RNG stream order)
        --------------------------------------
        We draw a uniform candidate action *unconditionally* every step,
        then either return ``last_action`` (with probability
        ``inertia_lambda``) or the candidate (with probability
        ``1 - inertia_lambda``). This keeps the per-step RNG consumption
        constant (one ``integers`` + one ``random`` draw on every call
        after the first), which makes the action stream stable under
        small refactors and matches the M3 dispatch's calibrated smoke
        threshold. The marginal stationary distribution is identical to
        the alternative "draw uniform only when switching" variant; only
        the realised stream order differs.
        """
        # First action: pure uniform draw, no Bernoulli (no last_action
        # to stick to). Subsequent actions: candidate-then-decide.
        cand = int(self._rng.integers(0, self.n_actions))
        if self._last_action is None:
            a = cand
        else:
            stick = bool(self._rng.random() < self._inertia_lambda)
            a = int(self._last_action) if stick else cand
        self._last_action = a
        return a

    def observe(
        self,
        agent_action: int,
        opponent_action: int,
        agent_reward: float,
        opponent_reward: Optional[float] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """No-op: ``last_action`` is set inside ``act`` for determinism.

        We deliberately do NOT mirror ``opponent_action`` from the
        environment back into ``self._last_action`` here, because that
        would couple internal state to a side-channel that may have been
        rounded / coerced by the env. The contract is "previous action
        produced by this adversary", which is captured directly in
        ``act``. This also keeps ``observe`` a pure-spectator method
        (consistent with the ``stationary`` and ``scripted_phase``
        precedents).
        """
        return None

    def info(self) -> Dict[str, Any]:
        """Spec §5.2 metadata block.

        Adds ``last_action`` as an extra field (alongside the mandatory
        ``ADVERSARY_INFO_KEYS``) so loggers / oracle-β schedules can
        condition on the current sticky target.
        """
        return self._build_info(
            phase="inertia",
            inertia_lambda=self._inertia_lambda,
            last_action=self._last_action,
        )
