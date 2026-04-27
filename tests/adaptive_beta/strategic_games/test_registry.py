"""Tests for ``GAME_REGISTRY`` / ``ADVERSARY_REGISTRY`` (Phase VII-B spec §4).

Spec authority: ``tasks/phase_VII_B_strategic_learning_coding_agent_spec.md``
§4 ("Game and adversary registries").

Invariants guarded
------------------
- Unknown name lookup raises ``KeyError`` (silent fallback is forbidden).
- Double-register without ``overwrite=True`` raises ``KeyError``.
- ``GAME_REGISTRY`` lists exactly the 5 expected games after package import.
- ``ADVERSARY_REGISTRY`` lists every expected adversary name (10 keys
  covering 9 distinct classes + ``stationary_mixed`` alias).
"""

from __future__ import annotations

import pytest

from experiments.adaptive_beta.strategic_games import (
    ADVERSARY_REGISTRY,
    GAME_REGISTRY,
    make_adversary,
    make_game,
    register_adversary,
    register_game,
)
from experiments.adaptive_beta.strategic_games.adversaries.stationary import (
    StationaryMixedOpponent,
)


# ---------------------------------------------------------------------------
# Game registry
# ---------------------------------------------------------------------------
EXPECTED_GAMES = frozenset(
    {
        "matching_pennies",
        "shapley",
        "rules_of_road",
        "asymmetric_coordination",
        "strategic_rps",
    }
)


def test_game_registry_lists_exact_five_games() -> None:
    """`spec §4 / §6` — exactly the five Phase VII-B games must register at import."""
    assert set(GAME_REGISTRY.keys()) == EXPECTED_GAMES, (
        f"unexpected game registry keys: "
        f"missing={EXPECTED_GAMES - set(GAME_REGISTRY)}, "
        f"extra={set(GAME_REGISTRY) - EXPECTED_GAMES}"
    )


def test_make_game_unknown_raises_keyerror() -> None:
    """`spec §4` — unknown game name raises a loud ``KeyError`` (no silent fallback)."""
    with pytest.raises(KeyError, match="Unknown game"):
        make_game("not_a_real_game_xyz")


def test_register_game_double_register_raises() -> None:
    """`spec §4` — re-registering without ``overwrite=True`` is rejected."""

    def _factory(**_: object) -> object:
        return object()

    name = "__test_double_register_game__"
    register_game(name, _factory)
    try:
        with pytest.raises(KeyError, match="already registered"):
            register_game(name, _factory)
        # ``overwrite=True`` succeeds.
        register_game(name, _factory, overwrite=True)
    finally:
        GAME_REGISTRY.pop(name, None)


# ---------------------------------------------------------------------------
# Adversary registry
# ---------------------------------------------------------------------------
EXPECTED_ADVERSARIES = frozenset(
    {
        # Spec §7 list:
        "stationary",
        "stationary_mixed",
        "scripted_phase",
        "finite_memory_best_response",
        "finite_memory_fictitious_play",
        "smoothed_fictitious_play",
        "regret_matching",
        "finite_memory_regret_matching",
        "hypothesis_testing",
        "realized_payoff_regret",
    }
)


def test_adversary_registry_includes_all_expected_names() -> None:
    """`spec §7.1–§7.9` — every expected adversary name must be registered."""
    missing = EXPECTED_ADVERSARIES - set(ADVERSARY_REGISTRY.keys())
    assert not missing, f"adversary registry missing: {sorted(missing)}"


def test_make_adversary_unknown_raises_keyerror() -> None:
    """`spec §4` — unknown adversary name raises a loud ``KeyError``."""
    with pytest.raises(KeyError, match="Unknown adversary"):
        make_adversary("not_a_real_name")


def test_register_adversary_double_register_raises() -> None:
    """`spec §4` — re-registering without ``overwrite=True`` is rejected."""
    name = "__test_double_register_adv__"
    register_adversary(name, StationaryMixedOpponent)
    try:
        with pytest.raises(KeyError, match="already registered"):
            register_adversary(name, StationaryMixedOpponent)
        register_adversary(name, StationaryMixedOpponent, overwrite=True)
    finally:
        ADVERSARY_REGISTRY.pop(name, None)


def test_make_adversary_round_trip() -> None:
    """`spec §4` — ``make_adversary`` returns a properly-typed instance."""
    adv = make_adversary("stationary", probs=[0.5, 0.5], seed=0)
    assert isinstance(adv, StationaryMixedOpponent)
    assert adv.n_actions == 2


def test_invariant_unknown_lookup_does_not_silently_fall_back() -> None:
    """`spec §4` — silent fallback would bypass the canonical-sign rule and
    let a typo'd adversary name execute as if it were valid. The registry
    must always raise ``KeyError`` on miss; this is the tripwire.
    """
    # Even an extremely close typo should miss.
    with pytest.raises(KeyError):
        make_adversary("Hypothesis_Testing")  # capital H
