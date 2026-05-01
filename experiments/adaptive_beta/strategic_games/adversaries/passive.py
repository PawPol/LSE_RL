"""Passive (no-op) opponent for single-agent chain MDPs (Phase VIII §5.7).

Spec authority: ``docs/specs/phase_VIII_tab_six_games.md`` §5.7
(``<!-- patch-2026-05-01 §11 -->``); upstream patch
``tasks/phase_VIII_spec_patches_2026-05-01.md`` §11.4.

A :class:`PassiveOpponent` always plays the only legal opponent action
(``0``). It carries no rolling state and no RNG dependence — used by
:class:`~experiments.adaptive_beta.strategic_games.games.delayed_chain.DelayedChainGame`
to wedge a fundamentally single-agent chain MDP into the Phase VIII
two-player matrix-game framework without polluting the reward channel
or introducing nondeterminism.

Spec compliance
---------------
- Inherits :class:`StrategicAdversary` (spec §5.2 ABC).
- ``info()`` returns the full :data:`ADVERSARY_INFO_KEYS` set:
  ``adversary_type="passive"``, ``phase="stationary"``,
  ``memory_m=0``, ``inertia_lambda=0.0``, ``temperature=0.0``,
  ``model_rejected=False``, ``search_phase="none"``,
  ``hypothesis_id=None``, ``policy_entropy=0.0``.
- ``reset(seed)`` is a no-op (no RNG state to seed).
- ``act(history, agent_action=None)`` always returns ``0``.
- ``observe(...)`` is a no-op.

Note on ``search_phase``
------------------------
Spec §5.2 / :data:`ADVERSARY_INFO_KEYS` defines ``search_phase`` as a
boolean state-machine flag (``False`` for adversaries without a search
phase). The patch §11.4 contract specifies the literal string
``"none"`` for a passive opponent. We honour the patch contract — the
key is present, the value is documentation-friendly, and the existing
:meth:`StrategicAdversary._build_info` helper accepts arbitrary scalar
types since it only enforces key presence (see
``adversaries/base.py::_validate_info``). Downstream consumers that
type-check ``search_phase`` against ``bool`` should treat the
:class:`PassiveOpponent` as a special case (``"none"`` ↔ "no search
phase concept applies").
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from experiments.adaptive_beta.strategic_games.adversaries.base import (
    StrategicAdversary,
    _validate_info,
)
from experiments.adaptive_beta.strategic_games.history import GameHistory


class PassiveOpponent(StrategicAdversary):
    """No-op opponent: always plays action ``0``; no RNG dependence.

    Parameters
    ----------
    n_actions
        Action-space cardinality. Must be ``1`` — the passive opponent
        is only well-defined for single-action opponent slots
        (consistent with :class:`DelayedChainGame`'s contract). Anything
        other than ``1`` raises :class:`ValueError` to surface
        misconfiguration loudly.
    seed
        Accepted for interface parity but ignored (passive has no RNG
        state).
    """

    adversary_type: str = "passive"

    def __init__(
        self,
        n_actions: int = 1,
        seed: Optional[int] = None,
    ) -> None:
        n = int(n_actions)
        if n != 1:
            raise ValueError(
                f"PassiveOpponent requires n_actions == 1, got {n_actions}"
            )
        # Delegate to the base ABC constructor so the public ``n_actions``
        # attribute and the (unused) ``self._rng`` are established.
        super().__init__(n_actions=n, seed=seed)

    # ------------------------------------------------------------------
    # ABC interface (spec §5.2)
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> None:
        """No-op. ``seed`` is ignored — the opponent has no RNG state."""
        del seed

    def act(
        self,
        history: GameHistory,
        agent_action: Optional[int] = None,
    ) -> int:
        """Return ``0`` unconditionally."""
        del history, agent_action
        return 0

    def observe(
        self,
        agent_action: int,
        opponent_action: int,
        agent_reward: float,
        opponent_reward: Optional[float] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """No-op: passive opponent has no internal state to update."""
        del agent_action, opponent_action, agent_reward, opponent_reward, info

    def info(self) -> Dict[str, Any]:
        """Spec §5.2 / patch §11.4 metadata block.

        Bypasses :meth:`StrategicAdversary._build_info` because the patch
        spec requires ``search_phase="none"`` (string) which the helper
        coerces to ``bool``. We construct the dict directly and run it
        through :func:`_validate_info` to guarantee the required key
        set is present.
        """
        info_dict: Dict[str, Any] = {
            "adversary_type":  self.adversary_type,
            "phase":           "stationary",
            "memory_m":        0,
            "inertia_lambda":  0.0,
            "temperature":     0.0,
            "model_rejected":  False,
            "search_phase":    "none",      # patch §11.4 literal
            "hypothesis_id":   None,
            "policy_entropy":  0.0,
        }
        return _validate_info(info_dict)
