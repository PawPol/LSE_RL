"""Strategic-learning adversary suite (Phase VII-B).

Spec authority: ``tasks/phase_VII_B_strategic_learning_coding_agent_spec.md``
§§5.2, 7.1–7.10.

Public adversaries exposed here:
- ``StationaryMixedOpponent``         (§7.1)
- ``ScriptedPhaseOpponent``           (§7.2; regression-compat with
                                       ``experiments/adaptive_beta/envs/rps.py``)
- ``FiniteMemoryBestResponse``        (§7.3)
- ``FiniteMemoryFictitiousPlay``      (§7.4)
- ``SmoothedFictitiousPlay``          (§7.5)
- ``RegretMatching``                  (§7.6, full-info + realized-payoff)
- ``FiniteMemoryRegretMatching``      (§7.7)
- ``HypothesisTestingAdversary``      (§7.8, priority adversary)
- ``RealizedPayoffRegret``            (§7.9, OPTIONAL stub)

DEFERRED per user authorization 2026-04-26:
- ``self_play.py`` (spec §7.10) — implement only after Stage B2-Main
  consolidates. Tracked by todo VII-B-D1.

These are deliberately NOT in this dispatch's scope. ``games/soda_game.py``
(§6.6) is similarly deferred.
"""

from __future__ import annotations

from experiments.adaptive_beta.strategic_games.adversaries.base import (
    ADVERSARY_INFO_KEYS,
    StrategicAdversary,
)
from experiments.adaptive_beta.strategic_games.adversaries.stationary import (
    StationaryMixedOpponent,
)
from experiments.adaptive_beta.strategic_games.adversaries.scripted_phase import (
    ScriptedPhaseOpponent,
)
from experiments.adaptive_beta.strategic_games.adversaries.finite_memory_best_response import (
    FiniteMemoryBestResponse,
)
from experiments.adaptive_beta.strategic_games.adversaries.finite_memory_fictitious_play import (
    FiniteMemoryFictitiousPlay,
)
from experiments.adaptive_beta.strategic_games.adversaries.smoothed_fictitious_play import (
    SmoothedFictitiousPlay,
)
from experiments.adaptive_beta.strategic_games.adversaries.regret_matching import (
    RegretMatching,
)
from experiments.adaptive_beta.strategic_games.adversaries.finite_memory_regret_matching import (
    FiniteMemoryRegretMatching,
)
from experiments.adaptive_beta.strategic_games.adversaries.hypothesis_testing import (
    HypothesisTestingAdversary,
)
from experiments.adaptive_beta.strategic_games.adversaries.realized_payoff_regret import (
    RealizedPayoffRegret,
)
from experiments.adaptive_beta.strategic_games.adversaries.inertia import (
    InertiaOpponent,
)

__all__ = [
    "ADVERSARY_INFO_KEYS",
    "StrategicAdversary",
    "StationaryMixedOpponent",
    "ScriptedPhaseOpponent",
    "FiniteMemoryBestResponse",
    "FiniteMemoryFictitiousPlay",
    "SmoothedFictitiousPlay",
    "RegretMatching",
    "FiniteMemoryRegretMatching",
    "HypothesisTestingAdversary",
    "RealizedPayoffRegret",
    "InertiaOpponent",
]
