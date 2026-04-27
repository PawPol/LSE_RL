"""Tests for the 5 Phase VII-B games (spec §6).

Spec authority: ``tasks/phase_VII_B_strategic_learning_coding_agent_spec.md``
§§6.1–6.5.

Invariants guarded
------------------
- Payoff matrices match spec literal values byte-for-byte.
- ``reset()`` and ``step(action)`` return the documented signatures.
- ``info["adversary_info"]`` covers ``ADVERSARY_INFO_KEYS``.
- Canonical-sign mapping per spec §22.3 / §6.4.
- Game-specific behaviour (matching_pennies horizon=1 mechanism flag,
  shapley cycle, rules_of_road tremble + payoff bias, asymmetric
  coordination guards, strategic_rps adversary dispatch).
- Schedule-build rules: ``wrong_sign`` rejects games with no canonical
  sign, accepts asymmetric_coordination.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pytest

from experiments.adaptive_beta.schedules import build_schedule
from experiments.adaptive_beta.strategic_games.adversaries.base import (
    ADVERSARY_INFO_KEYS,
)
from experiments.adaptive_beta.strategic_games.adversaries.stationary import (
    StationaryMixedOpponent,
)
from experiments.adaptive_beta.strategic_games.games import (
    asymmetric_coordination,
    matching_pennies,
    rules_of_road,
    shapley,
    strategic_rps,
)
from experiments.adaptive_beta.strategic_games.matrix_game import MatrixGameEnv


# ---------------------------------------------------------------------------
# Spec §6 byte-for-byte payoff matrices
# ---------------------------------------------------------------------------

def test_matching_pennies_payoffs_byte_match() -> None:
    """`spec §6.1` — matching pennies payoff matrices: [[+1, -1], [-1, +1]]."""
    expected_a = np.array([[+1.0, -1.0], [-1.0, +1.0]], dtype=np.float64)
    np.testing.assert_array_equal(matching_pennies.payoff_agent, expected_a)
    np.testing.assert_array_equal(matching_pennies.payoff_opponent, -expected_a)


def test_shapley_payoffs_cyclic_construction() -> None:
    """`spec §6.2` — payoff_agent[i, (i-1)%3] == 1, payoff_opponent[i, (i+1)%3] == 1,
    all other cells are 0.
    """
    pa = shapley.payoff_agent
    po = shapley.payoff_opponent
    expected_a = np.zeros((3, 3), dtype=np.float64)
    expected_o = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        expected_a[i, (i - 1) % 3] = 1.0
        expected_o[i, (i + 1) % 3] = 1.0
    np.testing.assert_array_equal(pa, expected_a)
    np.testing.assert_array_equal(po, expected_o)


def test_shapley_off_target_cells_are_zero() -> None:
    """`spec §6.2` — every cell that is NOT on the cyclic diagonal is 0."""
    pa = shapley.payoff_agent
    po = shapley.payoff_opponent
    for i in range(3):
        for j in range(3):
            if j != (i - 1) % 3:
                assert pa[i, j] == 0.0
            if j != (i + 1) % 3:
                assert po[i, j] == 0.0


def test_rules_of_road_payoffs_symmetric_coordination() -> None:
    """`spec §6.3` — base payoffs: [[+1, -1], [-1, +1]], opponent matrix == agent."""
    expected = np.array([[+1.0, -1.0], [-1.0, +1.0]], dtype=np.float64)
    np.testing.assert_array_equal(rules_of_road.payoff_agent, expected)
    np.testing.assert_array_equal(rules_of_road.payoff_opponent, expected)


def test_asymmetric_coordination_payoffs_stag_hunt() -> None:
    """`spec §6.4` — defaults: coop=5, risk=3 → [[5, 0], [3, 3]] / transpose."""
    expected_a = np.array([[5.0, 0.0], [3.0, 3.0]], dtype=np.float64)
    np.testing.assert_array_equal(asymmetric_coordination.payoff_agent, expected_a)
    np.testing.assert_array_equal(
        asymmetric_coordination.payoff_opponent, expected_a.T
    )


def test_strategic_rps_payoffs_cyclic() -> None:
    """`spec §6.5` — cyclic RPS payoff with 0 on diagonal."""
    expected_a = np.array(
        [[0.0, -1.0, +1.0], [+1.0, 0.0, -1.0], [-1.0, +1.0, 0.0]],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(strategic_rps.payoff_agent, expected_a)
    np.testing.assert_array_equal(strategic_rps.payoff_opponent, -expected_a)


# ---------------------------------------------------------------------------
# Game factories: reset + step contract; canonical sign metadata
# ---------------------------------------------------------------------------

GAMES_2A = [
    ("matching_pennies", matching_pennies),
    ("rules_of_road", rules_of_road),
    ("asymmetric_coordination", asymmetric_coordination),
]
GAMES_3A = [
    ("shapley", shapley),
    ("strategic_rps", strategic_rps),
]
EXPECTED_CANONICAL_SIGN = {
    "matching_pennies": None,
    "shapley": None,
    "rules_of_road": None,
    "strategic_rps": None,
    "asymmetric_coordination": "+",
}


def _build_env(name: str, mod) -> MatrixGameEnv:
    if name == "strategic_rps":
        return mod.build(
            adversary=StationaryMixedOpponent(probs=[1 / 3, 1 / 3, 1 / 3], seed=0),
            horizon=5, seed=0,
        )
    if name in {"matching_pennies", "rules_of_road"}:
        adv = StationaryMixedOpponent(probs=[0.5, 0.5], seed=0)
    elif name == "asymmetric_coordination":
        adv = StationaryMixedOpponent(probs=[0.5, 0.5], seed=0)
    else:
        adv = StationaryMixedOpponent(probs=[1 / 3, 1 / 3, 1 / 3], seed=0)
    return mod.build(adversary=adv, horizon=5, seed=0)


@pytest.mark.parametrize("name,mod", GAMES_2A + GAMES_3A)
def test_game_reset_returns_state_info(name: str, mod) -> None:
    """`spec §6 / §5.1` — every game's ``reset()`` returns ``(state, info)``."""
    env = _build_env(name, mod)
    state, info = env.reset()
    assert isinstance(state, np.ndarray)
    assert state.shape == (1,)
    assert state.dtype == np.int64
    assert isinstance(info, dict)


@pytest.mark.parametrize("name,mod", GAMES_2A + GAMES_3A)
def test_game_step_signature(name: str, mod) -> None:
    """`spec §6` — ``step(action)`` returns ``(next_state, reward, absorb, info)``."""
    env = _build_env(name, mod)
    env.reset()
    next_state, reward, absorbing, info = env.step(0)
    assert isinstance(next_state, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(absorbing, bool)
    assert isinstance(info, dict)
    # adversary_info covers the ADVERSARY_INFO_KEYS.
    adv_info = info["adversary_info"]
    missing = ADVERSARY_INFO_KEYS - set(adv_info.keys())
    assert not missing, f"{name} adversary_info missing keys: {sorted(missing)}"


@pytest.mark.parametrize(
    "name,mod", GAMES_2A + GAMES_3A
)
def test_game_canonical_sign_assignment(name: str, mod) -> None:
    """`spec §22.3 / §6.4` — canonical_sign is None for zero-sum / cyclic /
    coordination games and ``"+"`` for asymmetric_coordination.
    """
    env = _build_env(name, mod)
    sign = getattr(env, "env_canonical_sign", "MISSING")
    md = env.game_info()
    assert sign == EXPECTED_CANONICAL_SIGN[name], (
        f"{name}: env_canonical_sign={sign} != expected "
        f"{EXPECTED_CANONICAL_SIGN[name]}"
    )
    assert md["env_canonical_sign"] == EXPECTED_CANONICAL_SIGN[name]


# ---------------------------------------------------------------------------
# Matching pennies: horizon=1 mechanism degeneracy flag
# ---------------------------------------------------------------------------

def test_matching_pennies_horizon_1_mechanism_degenerate() -> None:
    """`spec §6.1` — at ``horizon=1``, the env reports ``mechanism_degenerate=True``."""
    env = matching_pennies.build(
        adversary=StationaryMixedOpponent(probs=[0.5, 0.5], seed=0),
        horizon=1, seed=0,
    )
    md = env.game_info()
    assert md["mechanism_degenerate"] is True


def test_matching_pennies_horizon_gt_1_mechanism_not_degenerate() -> None:
    """`spec §6.1` — at ``horizon>1``, ``mechanism_degenerate`` is False."""
    env = matching_pennies.build(
        adversary=StationaryMixedOpponent(probs=[0.5, 0.5], seed=0),
        horizon=10, seed=0,
    )
    md = env.game_info()
    assert md["mechanism_degenerate"] is False


# ---------------------------------------------------------------------------
# Rules of road: tremble + payoff bias variants
# ---------------------------------------------------------------------------

def test_rules_of_road_tremble_zero_passes_through() -> None:
    """`spec §6.3` — ``tremble_prob=0`` leaves opponent action unmodified."""
    inner = StationaryMixedOpponent(probs=[1.0, 0.0], seed=0)  # always 0
    env = rules_of_road.build(adversary=inner, horizon=20, seed=0, tremble_prob=0.0)
    env.reset()
    actions = []
    for _ in range(50):
        _, _, done, info = env.step(0)
        actions.append(info["opponent_action"])
        if done:
            env.reset()
    # All opponent actions should be 0 (no tremble).
    assert all(a == 0 for a in actions)


def test_rules_of_road_tremble_one_flips_deterministically() -> None:
    """`spec §6.3` — ``tremble_prob=1.0`` flips the opponent's action every step."""
    inner = StationaryMixedOpponent(probs=[1.0, 0.0], seed=0)  # always 0
    env = rules_of_road.build(adversary=inner, horizon=20, seed=0, tremble_prob=1.0)
    env.reset()
    actions = []
    for _ in range(50):
        _, _, done, info = env.step(0)
        actions.append(info["opponent_action"])
        if done:
            env.reset()
    # Tremble fires every step → action flipped to 1.
    assert all(a == 1 for a in actions)


def test_rules_of_road_payoff_bias_increments_right_right_cell() -> None:
    """`spec §6.3` — ``payoff_bias=0.5`` adds 0.5 to (R,R) cell in BOTH matrices."""
    inner = StationaryMixedOpponent(probs=[0.5, 0.5], seed=0)
    env = rules_of_road.build(
        adversary=inner, horizon=5, seed=0, payoff_bias=0.5,
    )
    pa = env._payoff_agent  # type: ignore[attr-defined]
    po = env._payoff_opponent  # type: ignore[attr-defined]
    # Base (1,1) is +1.0. After +0.5 bias, becomes 1.5 in both.
    np.testing.assert_allclose(pa[1, 1], 1.5)
    np.testing.assert_allclose(po[1, 1], 1.5)
    # Other cells unchanged.
    np.testing.assert_allclose(pa[0, 0], 1.0)
    np.testing.assert_allclose(pa[0, 1], -1.0)
    np.testing.assert_allclose(pa[1, 0], -1.0)


# ---------------------------------------------------------------------------
# Asymmetric coordination: parameter validation + canonical sign
# ---------------------------------------------------------------------------

def test_asymmetric_coordination_rejects_coop_le_risk() -> None:
    """`spec §6.4` — ``coop_payoff <= risk_payoff`` violates the stag-hunt geometry."""
    adv = StationaryMixedOpponent(probs=[0.5, 0.5], seed=0)
    with pytest.raises(ValueError, match="coop_payoff > risk_payoff"):
        asymmetric_coordination.build(
            adversary=adv, horizon=5, coop_payoff=3.0, risk_payoff=3.0,
        )


def test_asymmetric_coordination_rejects_non_positive_risk() -> None:
    """`spec §6.4` — ``risk_payoff <= 0`` violates the stag-hunt geometry."""
    adv = StationaryMixedOpponent(probs=[0.5, 0.5], seed=0)
    with pytest.raises(ValueError):
        asymmetric_coordination.build(
            adversary=adv, horizon=5, coop_payoff=5.0, risk_payoff=0.0,
        )


def test_asymmetric_coordination_canonical_sign_is_positive() -> None:
    """`spec §6.4` — ``env.env_canonical_sign == "+"``."""
    adv = StationaryMixedOpponent(probs=[0.5, 0.5], seed=0)
    env = asymmetric_coordination.build(adversary=adv, horizon=5, seed=0)
    assert env.env_canonical_sign == "+"


# ---------------------------------------------------------------------------
# Strategic RPS: adversary dispatch contract
# ---------------------------------------------------------------------------

def test_strategic_rps_requires_adversary_or_name() -> None:
    """`spec §6.5` — passing neither ``adversary`` nor ``adversary_name`` raises."""
    with pytest.raises(ValueError, match="adversary"):
        strategic_rps.build(horizon=5, seed=0)


def test_strategic_rps_dispatches_via_adversary_name() -> None:
    """`spec §6.5` — passing ``adversary_name="hypothesis_testing"`` builds."""
    env = strategic_rps.build(
        horizon=5,
        seed=0,
        adversary_name="hypothesis_testing",
        adversary_kwargs={
            "test_window_s": 50,
            "tolerance_tau": 0.05,
            "search_len": 10,
            "temperature": 0.2,
        },
    )
    assert env._adversary.adversary_type == "hypothesis_testing"  # type: ignore[attr-defined]
    assert env._adversary.n_actions == 3  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# wrong_sign schedule construction rules (spec §22.3)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,mod",
    [
        ("matching_pennies", matching_pennies),
        ("shapley", shapley),
        ("rules_of_road", rules_of_road),
        ("strategic_rps", strategic_rps),
    ],
)
def test_wrong_sign_rejected_for_canonical_sign_none_games(name: str, mod) -> None:
    """`spec §22.3` — ``wrong_sign`` schedule rejects games with no canonical sign.

    The error message is the literal `"not defined for env without canonical sign"`
    pinned by ``schedules._canonical_sign_to_value`` (line 122-123).
    """
    env = _build_env(name, mod)
    sign = getattr(env, "env_canonical_sign", None)
    with pytest.raises(
        ValueError, match="not defined for env without canonical sign"
    ):
        build_schedule("wrong_sign", sign, {"beta0": 1.0})


def test_wrong_sign_accepted_for_asymmetric_coordination() -> None:
    """`spec §22.3 / §6.4` — ``wrong_sign`` schedule accepts asymmetric_coordination."""
    adv = StationaryMixedOpponent(probs=[0.5, 0.5], seed=0)
    env = asymmetric_coordination.build(adversary=adv, horizon=5, seed=0)
    sched = build_schedule("wrong_sign", env.env_canonical_sign, {"beta0": 1.0})
    assert sched is not None


# ---------------------------------------------------------------------------
# Tripwire: matrix-shape regressions
# ---------------------------------------------------------------------------

def test_invariant_all_games_2d_payoff_matrices() -> None:
    """`spec §6` — every game module exposes module-level ``payoff_agent``
    and (optionally) ``payoff_opponent`` as 2-D float64 arrays. A regression
    that downcast to int or flattened the matrix would break the runner's
    payoff-row indexing.
    """
    for mod in (matching_pennies, shapley, rules_of_road,
                asymmetric_coordination, strategic_rps):
        pa = getattr(mod, "payoff_agent")
        po = getattr(mod, "payoff_opponent", None)
        assert pa.ndim == 2
        assert pa.dtype == np.float64
        if po is not None:
            assert po.ndim == 2
            assert po.dtype == np.float64
            assert po.shape == pa.shape
