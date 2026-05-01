"""Game and adversary registries for Phase VII-B strategic learning.

Spec authority: ``tasks/phase_VII_B_strategic_learning_coding_agent_spec.md``
§4.

Two name-keyed registries are exposed:

- ``GAME_REGISTRY``      : ``game_name`` → factory returning a configured
                            ``MatrixGameEnv``.
- ``ADVERSARY_REGISTRY`` : ``adversary_name`` → factory returning a
                            configured ``StrategicAdversary``.

Game and adversary modules register themselves at import time via
``register_game(...)`` and ``register_adversary(...)``. Lookup of an
unknown pair raises a loud ``KeyError`` (spec §4 / §8 wrong_sign rules:
silent fallback is forbidden).

The adversary registry pre-populates entries for the adversaries
shipped in this dispatch (spec §7.1–§7.9). Game factories are
deliberately left empty here — the games (matching_pennies, shapley,
rules_of_road, asymmetric_coordination, strategic_rps) are written in a
subsequent dispatch and are responsible for calling
``register_game(...)`` themselves at import time.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from experiments.adaptive_beta.strategic_games.adversaries.base import (
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
from experiments.adaptive_beta.strategic_games.adversaries.convention_switching import (
    ConventionSwitchingOpponent,
)
from experiments.adaptive_beta.strategic_games.adversaries.sign_switching_regime import (
    SignSwitchingRegimeOpponent,
)
from experiments.adaptive_beta.strategic_games.adversaries.passive import (
    PassiveOpponent,
)


# Factory signature: ``factory(**kwargs) -> StrategicAdversary``  (or env).
AdversaryFactory = Callable[..., StrategicAdversary]
GameFactory = Callable[..., Any]   # MatrixGameEnv — typed Any to break import cycle.


# ---------------------------------------------------------------------------
# Adversary registry
# ---------------------------------------------------------------------------
ADVERSARY_REGISTRY: Dict[str, AdversaryFactory] = {
    "stationary":                       StationaryMixedOpponent,
    "stationary_mixed":                 StationaryMixedOpponent,
    "scripted_phase":                   ScriptedPhaseOpponent,
    "finite_memory_best_response":      FiniteMemoryBestResponse,
    "finite_memory_fictitious_play":    FiniteMemoryFictitiousPlay,
    "smoothed_fictitious_play":         SmoothedFictitiousPlay,
    "regret_matching":                  RegretMatching,
    "finite_memory_regret_matching":    FiniteMemoryRegretMatching,
    "hypothesis_testing":               HypothesisTestingAdversary,
    "realized_payoff_regret":           RealizedPayoffRegret,  # OPTIONAL §7.9
    "inertia":                          InertiaOpponent,                 # Phase VIII M3 §5.7
    "convention_switching":             ConventionSwitchingOpponent,     # Phase VIII M3 §5.7
    "sign_switching_regime":            SignSwitchingRegimeOpponent,     # Phase VIII M3 §5.7 / §10.5
    "passive":                          PassiveOpponent,                 # Phase VIII §5.7 patch §11.4
}


def register_adversary(
    name: str,
    factory: AdversaryFactory,
    *,
    overwrite: bool = False,
) -> None:
    """Register an adversary factory under ``name``.

    Parameters
    ----------
    name
        Registry key. Use snake_case for consistency.
    factory
        Callable returning a ``StrategicAdversary``. Typically the
        adversary class itself.
    overwrite
        If ``False`` (default), raises ``KeyError`` on duplicate
        names; if ``True``, replaces the existing entry. Use
        ``overwrite=True`` only in test fixtures.
    """
    if not overwrite and name in ADVERSARY_REGISTRY:
        raise KeyError(
            f"Adversary {name!r} already registered (pass overwrite=True "
            f"to replace)"
        )
    ADVERSARY_REGISTRY[name] = factory


def make_adversary(name: str, **kwargs: Any) -> StrategicAdversary:
    """Instantiate the registered adversary called ``name``.

    Raises ``KeyError`` if ``name`` is unknown — silent fallback is
    forbidden (spec §4).
    """
    if name not in ADVERSARY_REGISTRY:
        known = ", ".join(sorted(ADVERSARY_REGISTRY.keys()))
        raise KeyError(
            f"Unknown adversary {name!r}; registered names: [{known}]"
        )
    factory = ADVERSARY_REGISTRY[name]
    return factory(**kwargs)


# ---------------------------------------------------------------------------
# Game registry
# ---------------------------------------------------------------------------
# Game modules register here at import time.
GAME_REGISTRY: Dict[str, GameFactory] = {}


def register_game(
    name: str,
    factory: GameFactory,
    *,
    overwrite: bool = False,
) -> None:
    """Register a game factory under ``name``.

    The factory is expected to return a ``MatrixGameEnv``. It typically
    accepts ``adversary``, ``seed``, and game-specific knobs (tremble
    probability, payoff bias, etc.) as kwargs.
    """
    if not overwrite and name in GAME_REGISTRY:
        raise KeyError(
            f"Game {name!r} already registered (pass overwrite=True "
            f"to replace)"
        )
    GAME_REGISTRY[name] = factory


def make_game(name: str, **kwargs: Any) -> Any:
    """Instantiate the registered game called ``name``.

    Raises ``KeyError`` if ``name`` is unknown.
    """
    if name not in GAME_REGISTRY:
        known = (
            ", ".join(sorted(GAME_REGISTRY.keys())) or "(none yet)"
        )
        raise KeyError(
            f"Unknown game {name!r}; registered names: [{known}]"
        )
    factory = GAME_REGISTRY[name]
    return factory(**kwargs)


# ---------------------------------------------------------------------------
# Phase VIII M2 — auto-register Soda / Uncertain Game (spec §5.5) and
# Potential / Weakly-Acyclic Game (spec §5.6).
#
# The Phase VII games self-register via ``games/__init__.py`` import, but
# the Phase VIII modules live outside that package's import chain (per the
# M2 dispatch boundary: only ``registry.py`` may be edited to wire them in).
# Importing each module triggers its bottom-of-file ``register_game(...)``
# call, populating ``GAME_REGISTRY``. Placed at module bottom so
# ``register_game`` is fully defined when these imports execute.
from experiments.adaptive_beta.strategic_games.games import (  # noqa: E402,F401
    soda_uncertain,
)
from experiments.adaptive_beta.strategic_games.games import (  # noqa: E402,F401
    potential as _potential_game,
)


# ---------------------------------------------------------------------------
# Phase VIII M2 reopen — RR-Sparse subcase (patch §1, 2026-05-01).
#
# Registered as a separate factory key ``"rules_of_road_sparse"`` that
# delegates to the dense RoR ``build`` with ``sparse_terminal=True`` and
# the patch §1.2 default horizon ``H = 20`` (longer than dense subcases
# to stress credit assignment over multiple steps). The dense
# ``"rules_of_road"`` registration is unchanged.
from experiments.adaptive_beta.strategic_games.games import (  # noqa: E402
    rules_of_road as _rules_of_road,
)


def _build_rules_of_road_sparse(
    adversary: StrategicAdversary,
    horizon: int = 20,
    **kwargs: Any,
) -> Any:
    """Factory wrapper for ``rules_of_road`` with ``sparse_terminal=True``.

    Per patch §1.2: defaults to ``H = 20`` (longer than the dense RoR
    subcases). Passing ``sparse_terminal`` explicitly is rejected so
    the registry key is the single source of truth for the sparse vs
    dense distinction.
    """
    if "sparse_terminal" in kwargs:
        raise TypeError(
            "rules_of_road_sparse factory does not accept the "
            "'sparse_terminal' kwarg (it is implied by the registry key)"
        )
    return _rules_of_road.build(
        adversary=adversary,
        horizon=horizon,
        sparse_terminal=True,
        **kwargs,
    )


register_game("rules_of_road_sparse", _build_rules_of_road_sparse)


# ---------------------------------------------------------------------------
# Phase VIII M2 reopen — delayed_chain game (patch §11, 2026-05-01).
#
# Long-horizon delayed-reward chain with 4 subcases (DC-Short10,
# DC-Medium20, DC-Long50, DC-Branching20). Anchors the paper-title
# "Selective Temporal Credit Assignment" claim. Self-registers at
# import time via the bottom-of-module register_game call.
from experiments.adaptive_beta.strategic_games.games import (  # noqa: E402,F401
    delayed_chain as _delayed_chain_game,
)
