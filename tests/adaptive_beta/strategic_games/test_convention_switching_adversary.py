"""Tests for the Phase VIII M3 convention-switching adversary.

Spec authority: ``docs/specs/phase_VIII_tab_six_games.md`` M3 and
sec. 5.7. The adversary switches between the two Rules-of-the-Road
conventions at episode boundaries and exposes the convention as
``info()["regime"]`` for oracle-only diagnostics.
"""

from __future__ import annotations

from experiments.adaptive_beta.strategic_games.adversaries.base import (
    ADVERSARY_INFO_KEYS,
)
from experiments.adaptive_beta.strategic_games.adversaries.convention_switching import (
    ConventionSwitchingOpponent,
)
from experiments.adaptive_beta.strategic_games.history import GameHistory
from experiments.adaptive_beta.strategic_games.registry import (
    ADVERSARY_REGISTRY,
    make_adversary,
)


REGIMES = {"left", "right"}
EXPECTED_INFO_KEYS = ADVERSARY_INFO_KEYS | frozenset(
    {
        "regime",
        "mode",
        "switches_so_far",
        "convention_index",
        "episode_count",
        "switch_period_episodes",
        "switch_prob",
    }
)


def _episode_trajectory(
    adv: ConventionSwitchingOpponent,
    *,
    episodes: int,
) -> list[tuple[int, str, int]]:
    """Record action/regime/switch count before each episode boundary."""
    history = GameHistory()
    trajectory: list[tuple[int, str, int]] = []
    for _ in range(episodes):
        action = int(adv.act(history))
        info = adv.info()
        trajectory.append(
            (action, str(info["regime"]), int(info["switches_so_far"]))
        )
        adv.on_episode_end()
    return trajectory


def test_register_in_registry() -> None:
    """M3 registry contract: ``convention_switching`` is registered."""
    assert "convention_switching" in ADVERSARY_REGISTRY
    adv = make_adversary("convention_switching", seed=7)
    assert isinstance(adv, ConventionSwitchingOpponent)


def test_periodic_mode_switches_at_period() -> None:
    """Periodic mode flips at completed episodes 5, 10, and 15."""
    adv = ConventionSwitchingOpponent(
        mode="periodic",
        switch_period_episodes=5,
        initial_convention=0,
    )

    flip_episodes: list[int] = []
    for episode in range(1, 16):
        before = adv.info()["regime"]
        adv.on_episode_end()
        after = adv.info()["regime"]
        if after != before:
            flip_episodes.append(episode)

    assert flip_episodes == [5, 10, 15]
    assert adv.info()["switches_so_far"] == 3
    assert adv.info()["regime"] == "right"


def test_stochastic_mode_determinism() -> None:
    """Same stochastic seed yields an identical regime trajectory."""
    kwargs = {
        "mode": "stochastic",
        "switch_prob": 0.5,
        "initial_convention": 0,
        "seed": 20260430,
    }
    a = ConventionSwitchingOpponent(**kwargs)
    b = ConventionSwitchingOpponent(**kwargs)

    assert _episode_trajectory(a, episodes=100) == _episode_trajectory(
        b, episodes=100
    )


def test_stochastic_mode_eventual_switch() -> None:
    """With ``switch_prob=0.5``, a long seeded run visits both regimes."""
    adv = ConventionSwitchingOpponent(
        mode="stochastic",
        switch_prob=0.5,
        initial_convention=0,
        seed=12345,
    )

    regimes = {str(adv.info()["regime"])}
    for _ in range(200):
        adv.on_episode_end()
        regimes.add(str(adv.info()["regime"]))

    assert regimes == REGIMES


def test_constant_within_episode() -> None:
    """Repeated ``act()`` calls within one episode return one pure action."""
    adv = ConventionSwitchingOpponent(
        mode="stochastic",
        switch_prob=1.0,
        initial_convention=1,
        seed=0,
    )
    history = GameHistory()

    actions = [adv.act(history, agent_action=i % 2) for i in range(20)]

    assert actions == [1] * 20
    assert adv.info()["regime"] == "right"
    assert adv.info()["switches_so_far"] == 0


def test_info_regime_in_set() -> None:
    """``info()["regime"]`` is always one of the documented labels."""
    adv = ConventionSwitchingOpponent(
        mode="periodic",
        switch_period_episodes=1,
        initial_convention=0,
    )

    assert adv.info()["regime"] in REGIMES
    adv.on_episode_end()
    assert adv.info()["regime"] in REGIMES


def test_info_keys_schema() -> None:
    """``info()`` exposes the base adversary keys plus M3 regime fields."""
    adv = ConventionSwitchingOpponent(
        mode="periodic",
        switch_period_episodes=5,
        switch_prob=0.25,
        initial_convention=0,
    )
    info = adv.info()

    missing = EXPECTED_INFO_KEYS - set(info)
    assert not missing, f"ConventionSwitchingOpponent.info missing {missing}"
    assert info["adversary_type"] == "convention_switching"
    assert info["phase"] == "convention_switching"


def test_on_episode_end_hook_present() -> None:
    """The environment-facing episode-boundary hook is callable."""
    adv = ConventionSwitchingOpponent()

    assert callable(getattr(adv, "on_episode_end", None))
    adv.on_episode_end()
    assert adv.info()["episode_count"] == 1


def test_initial_convention() -> None:
    """``initial_convention`` selects the first exposed regime/action."""
    left = ConventionSwitchingOpponent(initial_convention=0)
    right = ConventionSwitchingOpponent(initial_convention=1)

    assert left.info()["regime"] == "left"
    assert left.act(GameHistory()) == 0
    assert right.info()["regime"] == "right"
    assert right.act(GameHistory()) == 1


def test_n_actions_default() -> None:
    """Default construction honors the two-action convention space."""
    adv = ConventionSwitchingOpponent()

    assert adv.n_actions == 2
    assert 0 <= adv.act(GameHistory()) < adv.n_actions
