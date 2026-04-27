"""Phase VII-B game catalogue (spec §6).

Importing this subpackage registers all five Phase VII-B games into
``GAME_REGISTRY`` via the ``register_game`` calls at the bottom of
each game module.

Registered games:
- ``matching_pennies``         (§6.1, 2x2 zero-sum)
- ``shapley``                  (§6.2, 3x3 cyclic non-zero-sum)
- ``rules_of_road``            (§6.3, 2x2 coordination, tremble + bias variants)
- ``asymmetric_coordination``  (§6.4, stag-hunt, canonical sign = "+")
- ``strategic_rps``            (§6.5, 3x3 zero-sum vs endogenous opponents)

Deferred per user authorization 2026-04-26:
- ``soda_game`` (§6.6, hidden-type game).
"""

from __future__ import annotations

# Ordered imports are deliberate: each module calls ``register_game``
# at import time; the order here is alphabetical except ``strategic_rps``
# trails so its dispatch-via-registry log entry is the last one printed
# during diagnostic imports.
from experiments.adaptive_beta.strategic_games.games import (  # noqa: F401
    asymmetric_coordination,
    matching_pennies,
    rules_of_road,
    shapley,
    strategic_rps,
)

__all__ = [
    "asymmetric_coordination",
    "matching_pennies",
    "rules_of_road",
    "shapley",
    "strategic_rps",
]
