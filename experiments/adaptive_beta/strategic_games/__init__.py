"""Strategic-learning game and adversary subpackage for Phase VII-B.

Spec authority: ``tasks/phase_VII_B_strategic_learning_coding_agent_spec.md``
(§§4, 5.1–5.3, 7) and parent ``docs/specs/phase_VII_adaptive_beta.md``.

Branch identifier: ``phase-VII-B-strategic-2026-04-26``.

Reuse contract (spec §2.1, §22.1 of parent):
- All TAB / safe weighted log-sum-exp operator math is imported from
  ``src.lse_rl.operator.tab_operator``. This subpackage MUST NOT
  reimplement ``g_{β,γ}``, ``ρ_{β,γ}``, or ``d_{β,γ}``.
- The environment subclass uses the existing MushroomRL
  ``Environment`` ABC (Phase VII §22.2 lock).

DEFERRED modules (per user authorization 2026-04-26):
- ``self_play.py`` — Phase VII-B spec §7.10. Track for follow-up phase.
- ``games/soda_game.py`` — Phase VII-B spec §6.6. Hidden-type game.
- Stage B2-Stress (spec §11.3) is not in this dispatch's matrix.

This file exposes the public API of the subpackage. Game modules
(matching_pennies, shapley, rules_of_road, asymmetric_coordination,
strategic_rps) and config/runner glue are written in subsequent
dispatches.
"""

from __future__ import annotations

from experiments.adaptive_beta.strategic_games.history import GameHistory
from experiments.adaptive_beta.strategic_games.matrix_game import (
    MatrixGameEnv,
    StateEncoder,
)
from experiments.adaptive_beta.strategic_games.registry import (
    GAME_REGISTRY,
    ADVERSARY_REGISTRY,
    register_game,
    register_adversary,
    make_game,
    make_adversary,
)
from experiments.adaptive_beta.strategic_games.adversaries.base import (
    StrategicAdversary,
    ADVERSARY_INFO_KEYS,
)

# Import the games subpackage AFTER the registry is defined: each game
# module calls ``register_game(...)`` at import time, so this side-effect
# import is what populates ``GAME_REGISTRY``.
from experiments.adaptive_beta.strategic_games import games  # noqa: F401  E402

__all__ = [
    "GameHistory",
    "MatrixGameEnv",
    "StateEncoder",
    "GAME_REGISTRY",
    "ADVERSARY_REGISTRY",
    "register_game",
    "register_adversary",
    "make_game",
    "make_adversary",
    "StrategicAdversary",
    "ADVERSARY_INFO_KEYS",
]
