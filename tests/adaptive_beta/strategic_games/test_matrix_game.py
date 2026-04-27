"""Tests for ``MatrixGameEnv`` / ``make_default_state_encoder``
(Phase VII-B spec §5.1).

Spec authority: ``tasks/phase_VII_B_strategic_learning_coding_agent_spec.md``
§5.1 (Matrix Game environment + state encoder).

Invariants guarded
------------------
- ``reset(seed=k)`` returns ``(state, info)`` with the right dtype/shape and
  the documented ``info`` keys.
- ``step(action)`` accepts ``int``, ``np.int64``, shape-(1,) array, 0-d array
  identically (lessons.md numpy-state-scalar pattern).
- Determinism: same seed + same actions → byte-identical reward stream.
- The MushroomRL ``env.info`` property still returns ``MDPInfo`` with
  ``.gamma`` / ``.observation_space`` / ``.action_space``.
- ``make_default_state_encoder`` cardinality and index formula.
"""

from __future__ import annotations

import numpy as np
import pytest

from mushroom_rl.core import MDPInfo

from experiments.adaptive_beta.strategic_games.adversaries.stationary import (
    StationaryMixedOpponent,
)
from experiments.adaptive_beta.strategic_games.history import GameHistory
from experiments.adaptive_beta.strategic_games.matrix_game import (
    MatrixGameEnv,
    default_single_state_encoder,
    make_default_state_encoder,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def matching_pennies_env() -> MatrixGameEnv:
    """Build a matching-pennies env from raw payoff matrices, horizon=1."""
    pa = np.array([[+1.0, -1.0], [-1.0, +1.0]], dtype=np.float64)
    adv = StationaryMixedOpponent(probs=[0.5, 0.5], seed=0)
    return MatrixGameEnv(
        payoff_agent=pa,
        payoff_opponent=-pa,
        adversary=adv,
        horizon=1,
        seed=42,
        game_name="matching_pennies",
        gamma=0.95,
    )


# ---------------------------------------------------------------------------
# reset / info contract
# ---------------------------------------------------------------------------

REQUIRED_INFO_KEYS = {
    "phase",
    "is_shift_step",
    "catastrophe",
    "terminal_success",
    "opponent_action",
    "agent_action",
    "adversary_info",
    "game_name",
    "episode_index",
}


def test_reset_returns_state_and_info(matching_pennies_env: MatrixGameEnv) -> None:
    """`spec §5.1` — ``reset`` returns ``(state, info)``; state shape (1,) int64."""
    state, info = matching_pennies_env.reset()
    assert isinstance(state, np.ndarray)
    assert state.dtype == np.int64
    assert state.shape == (1,)
    assert isinstance(info, dict)
    missing = REQUIRED_INFO_KEYS - set(info.keys())
    assert not missing, f"reset info missing keys: {sorted(missing)}"


def test_reset_info_has_documented_default_values(
    matching_pennies_env: MatrixGameEnv,
) -> None:
    """`spec §5.1` — at reset, agent_action / opponent_action are ``None`` and
    is_shift_step / catastrophe / terminal_success default to ``False``.
    """
    _, info = matching_pennies_env.reset()
    assert info["agent_action"] is None
    assert info["opponent_action"] is None
    assert info["is_shift_step"] is False
    assert info["catastrophe"] is False
    assert info["terminal_success"] is False
    assert info["game_name"] == "matching_pennies"


def test_env_info_property_returns_mdpinfo(
    matching_pennies_env: MatrixGameEnv,
) -> None:
    """`spec §5.1 module note` — ``env.info`` (property) returns ``MDPInfo``,
    not the spec §5.1 metadata dict (which lives at ``game_info()``).
    """
    info = matching_pennies_env.info
    assert isinstance(info, MDPInfo)
    assert pytest.approx(info.gamma) == 0.95
    assert info.observation_space is not None
    assert info.action_space is not None
    # Action space cardinality matches payoff-matrix rows.
    assert int(info.action_space.size[0]) == 2


def test_game_info_method_returns_metadata_dict(
    matching_pennies_env: MatrixGameEnv,
) -> None:
    """`spec §5.1` — the metadata-dict accessor lives at ``game_info()``."""
    matching_pennies_env.reset()
    md = matching_pennies_env.game_info()
    assert md["game_name"] == "matching_pennies"
    assert md["n_agent_actions"] == 2
    assert md["n_opponent_actions"] == 2
    assert md["horizon"] == 1
    assert pytest.approx(md["gamma"]) == 0.95


# ---------------------------------------------------------------------------
# step(action) accepts multiple scalar shapes (lessons.md pattern)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "make_action",
    [
        ("python_int", lambda: 1),
        ("numpy_int64", lambda: np.int64(1)),
        ("shape_1_array", lambda: np.array([1], dtype=np.int64)),
        ("zero_d_array", lambda: np.array(1, dtype=np.int64)),
    ],
    ids=lambda x: x[0] if isinstance(x, tuple) else str(x),
)
def test_step_accepts_action_in_multiple_shapes(make_action: tuple) -> None:
    """`lessons.md` / `spec §5.1` — step() accepts int / np.int64 / shape-(1,) /
    0-d array identically. Per-call rewards must agree across shapes when
    seeds and adversary state are equal.
    """
    _, factory = make_action
    pa = np.array([[+1.0, -1.0], [-1.0, +1.0]], dtype=np.float64)
    rewards = []
    for _ in range(4):
        adv = StationaryMixedOpponent(probs=[0.5, 0.5], seed=7)
        env = MatrixGameEnv(
            payoff_agent=pa,
            payoff_opponent=-pa,
            adversary=adv,
            horizon=1,
            seed=11,
            game_name="mp",
        )
        env.reset()
        _, r, absorbing, _ = env.step(factory())
        rewards.append(float(r))
        assert absorbing is True
    # All four shapes must yield byte-identical rewards.
    assert len(set(rewards)) == 1, f"reward differs across action shapes: {rewards}"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def _run_episode(seed: int, agent_actions: list[int]) -> np.ndarray:
    """Run one episode of fixed length and capture the reward stream."""
    pa = np.array([[+1.0, -1.0], [-1.0, +1.0]], dtype=np.float64)
    adv = StationaryMixedOpponent(probs=[0.5, 0.5], seed=seed)
    env = MatrixGameEnv(
        payoff_agent=pa,
        payoff_opponent=-pa,
        adversary=adv,
        horizon=len(agent_actions),
        seed=seed,
        game_name="mp",
    )
    env.reset()
    rs = []
    for a in agent_actions:
        _, r, done, _ = env.step(a)
        rs.append(float(r))
        if done:
            break
    return np.asarray(rs, dtype=np.float64)


def test_step_determinism_byte_identical() -> None:
    """`spec §5.1` — same seed + same actions ⇒ byte-identical reward stream."""
    actions = [0, 1, 0, 1, 0, 0, 1, 1]
    r1 = _run_episode(seed=123, agent_actions=actions)
    r2 = _run_episode(seed=123, agent_actions=actions)
    np.testing.assert_array_equal(r1, r2)


def test_step_action_out_of_range_raises(matching_pennies_env: MatrixGameEnv) -> None:
    """`spec §5.1` — agent actions outside ``[0, n_actions)`` are rejected."""
    matching_pennies_env.reset()
    with pytest.raises(ValueError, match="agent action"):
        matching_pennies_env.step(7)


# ---------------------------------------------------------------------------
# make_default_state_encoder
# ---------------------------------------------------------------------------

def test_default_state_encoder_n_states_formula() -> None:
    """`spec §5.1` — ``n_states == H * (A + 1)`` (slot 0 = "no prior action")."""
    encoder, n_states = make_default_state_encoder(horizon=20, n_actions=3)
    assert n_states == 20 * (3 + 1)
    assert callable(encoder)


def test_default_state_encoder_step_zero_no_prior_action() -> None:
    """`spec §5.1` — at ``step=0, last_opponent_action=None`` the encoder returns ``[0]``."""
    encoder, _ = make_default_state_encoder(horizon=5, n_actions=2)
    s = encoder({"step_in_episode": 0, "last_opponent_action": None})
    assert s.shape == (1,)
    assert int(s.flat[0]) == 0


@pytest.mark.parametrize("step_t,prev_action", [(1, 0), (2, 1), (4, 2)])
def test_default_state_encoder_index_formula(step_t: int, prev_action: int) -> None:
    """`spec §5.1` — encoder index ``= step * (A + 1) + (prev_opp + 1)``."""
    A = 3
    encoder, _ = make_default_state_encoder(horizon=5, n_actions=A)
    s = encoder({"step_in_episode": step_t, "last_opponent_action": prev_action})
    expected = step_t * (A + 1) + (prev_action + 1)
    assert int(s.flat[0]) == expected


def test_default_state_encoder_rejects_bad_horizon_or_actions() -> None:
    """`spec §5.1` — defensive validation on horizon / n_actions."""
    with pytest.raises(ValueError):
        make_default_state_encoder(horizon=0, n_actions=2)
    with pytest.raises(ValueError):
        make_default_state_encoder(horizon=5, n_actions=0)


def test_default_state_encoder_rejects_out_of_range_prev_action() -> None:
    """`spec §5.1` — out-of-range ``last_opponent_action`` is rejected loudly."""
    encoder, _ = make_default_state_encoder(horizon=5, n_actions=2)
    with pytest.raises(ValueError, match="out of range"):
        encoder({"step_in_episode": 0, "last_opponent_action": 9})


# ---------------------------------------------------------------------------
# Single-dummy-state encoder
# ---------------------------------------------------------------------------

def test_default_single_state_encoder_returns_zero() -> None:
    """`spec §5.1` — default single-dummy-state encoder always returns ``[0]``."""
    s = default_single_state_encoder({"step_in_episode": 99})
    assert s.shape == (1,) and int(s.flat[0]) == 0


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------

def test_matrix_game_rejects_mismatched_payoff_shapes() -> None:
    """`spec §5.1` — agent / opponent payoff matrices must share shape."""
    pa = np.zeros((2, 2))
    po = np.zeros((2, 3))
    adv = StationaryMixedOpponent(probs=[1.0, 0.0, 0.0], seed=0)
    with pytest.raises(ValueError, match="shape"):
        MatrixGameEnv(payoff_agent=pa, payoff_opponent=po, adversary=adv, horizon=1)


def test_matrix_game_rejects_adversary_action_count_mismatch() -> None:
    """`spec §5.1` — adversary's ``n_actions`` must match payoff shape."""
    pa = np.zeros((2, 2))
    adv = StationaryMixedOpponent(probs=[1.0, 0.0, 0.0], seed=0)  # 3-action
    with pytest.raises(ValueError, match="n_actions"):
        MatrixGameEnv(payoff_agent=pa, payoff_opponent=None, adversary=adv, horizon=1)


# ---------------------------------------------------------------------------
# Tripwire: environment-info dict must surface the keys the runner reads.
# ---------------------------------------------------------------------------

def test_invariant_step_info_includes_runner_required_keys() -> None:
    """`spec §5.1` — ``step()`` info dict must include the keys the strategic
    runner reads (``adversary_info``, ``agent_action``, ``opponent_action``,
    ``catastrophe``, ``terminal_success``, ``phase``, ``game_name``).
    Missing any of these would silently break ``run_strategic.run_one_cell``.
    """
    pa = np.array([[+1.0, -1.0], [-1.0, +1.0]], dtype=np.float64)
    adv = StationaryMixedOpponent(probs=[0.5, 0.5], seed=0)
    env = MatrixGameEnv(
        payoff_agent=pa, payoff_opponent=-pa, adversary=adv, horizon=2, seed=0,
    )
    env.reset()
    _, _, _, info = env.step(0)
    for key in (
        "adversary_info", "agent_action", "opponent_action",
        "catastrophe", "terminal_success", "phase", "game_name",
    ):
        assert key in info, f"step info missing required key {key!r}"
