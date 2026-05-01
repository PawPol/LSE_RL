"""Tests for the Rules-of-the-Road **sparse-terminal** subcase (RR-Sparse).

Spec authority: ``tasks/phase_VIII_spec_patches_2026-05-01.md`` §1
(folded into ``docs/specs/phase_VIII_tab_six_games.md`` §5.3, marked
``<!-- patch-2026-05-01 §1 -->``).

Invariants guarded
------------------
- ``"rules_of_road_sparse"`` is registered in ``GAME_REGISTRY``.
- For every non-terminal step ``t < H - 1`` the env returns
  ``reward == 0.0``.
- The terminal step pays ``+c`` (default ``1.0``) when the last
  played profile is coordinated and ``-m`` (default ``0.5``) when
  miscoordinated.
- The default horizon is ``H = 20`` (longer than dense RoR subcases,
  per patch §1.2).
- ``env.env_canonical_sign`` matches the dense RoR (no canonical-sign
  drift introduced by the sparse-terminal flag).
- Same seed ⇒ same trajectory (reset/step determinism).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pytest

from experiments.adaptive_beta.strategic_games.adversaries.stationary import (
    StationaryMixedOpponent,
)
from experiments.adaptive_beta.strategic_games.games import rules_of_road
from experiments.adaptive_beta.strategic_games.registry import (
    GAME_REGISTRY,
    make_game,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sparse_env(
    *,
    opp_probs: Tuple[float, float] = (1.0, 0.0),
    seed: int = 0,
    horizon: int | None = None,
):
    """Build a sparse-terminal RoR env with a deterministic stationary opponent.

    ``opp_probs=(1.0, 0.0)`` makes the opponent always play action 0 (Left),
    which gives the test full control over the coordinated/miscoordinated
    trajectory via the agent's action choice.
    """
    adv = StationaryMixedOpponent(probs=list(opp_probs), seed=seed)
    kwargs: dict = {"adversary": adv, "seed": seed}
    if horizon is not None:
        kwargs["horizon"] = horizon
    return make_game("rules_of_road_sparse", **kwargs)


def _roll_episode(env, agent_action: int) -> Tuple[List[float], List[bool]]:
    """Run one episode with a fixed agent action; return per-step rewards and dones."""
    rewards: List[float] = []
    dones: List[bool] = []
    env.reset()
    for _ in range(env._horizon):
        _, r, done, _ = env.step(np.array([agent_action]))
        rewards.append(float(r))
        dones.append(bool(done))
        if done:
            break
    return rewards, dones


# ---------------------------------------------------------------------------
# 1. Registry membership
# ---------------------------------------------------------------------------

def test_rr_sparse_in_game_registry() -> None:
    """`patch §1.4` — ``"rules_of_road_sparse"`` is a registered game key."""
    assert "rules_of_road_sparse" in GAME_REGISTRY


# ---------------------------------------------------------------------------
# 2. Per-step reward is zero on non-terminal steps
# ---------------------------------------------------------------------------

def test_rr_sparse_per_step_reward_zero() -> None:
    """`patch §1.2` — non-terminal steps have ``reward == 0`` regardless of profile."""
    # Coordinated trajectory (agent=0, opponent=0) — the dense env would
    # pay +1.0 every step; the sparse env must pay 0 except terminal.
    env = _make_sparse_env(opp_probs=(1.0, 0.0), seed=0)
    rewards, dones = _roll_episode(env, agent_action=0)
    assert len(rewards) == env._horizon
    # All non-terminal steps return reward = 0.
    for t, (r, done) in enumerate(zip(rewards[:-1], dones[:-1])):
        assert r == 0.0, f"non-terminal step t={t} returned r={r}, expected 0.0"
        assert done is False, f"non-terminal step t={t} reported done=True"

    # Same check on a miscoordinated trajectory (agent=1, opponent=0).
    env = _make_sparse_env(opp_probs=(1.0, 0.0), seed=0)
    rewards, dones = _roll_episode(env, agent_action=1)
    for t, (r, done) in enumerate(zip(rewards[:-1], dones[:-1])):
        assert r == 0.0, f"non-terminal step t={t} returned r={r}, expected 0.0"
        assert done is False


# ---------------------------------------------------------------------------
# 3. Terminal reward when the last profile is coordinated
# ---------------------------------------------------------------------------

def test_rr_sparse_terminal_reward_coordinated() -> None:
    """`patch §1.2` — coordinated terminal pays ``+c`` (default ``c=1.0``)."""
    # opp always plays 0; agent plays 0 → coordinated on every step,
    # including terminal.
    env = _make_sparse_env(opp_probs=(1.0, 0.0), seed=0)
    rewards, dones = _roll_episode(env, agent_action=0)
    # The episode must terminate at step H-1.
    assert dones[-1] is True
    # Terminal reward equals +c with default c = 1.0.
    assert rewards[-1] == pytest.approx(1.0)
    # Episode return equals exactly the terminal reward (other steps zero).
    assert sum(rewards) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 4. Terminal reward when the last profile is miscoordinated
# ---------------------------------------------------------------------------

def test_rr_sparse_terminal_reward_miscoordinated() -> None:
    """`patch §1.2` — miscoordinated terminal pays ``-m`` (default ``m=0.5``)."""
    # opp always plays 0; agent plays 1 → miscoordinated on every step.
    env = _make_sparse_env(opp_probs=(1.0, 0.0), seed=0)
    rewards, dones = _roll_episode(env, agent_action=1)
    assert dones[-1] is True
    # Terminal reward equals -m with default m = 0.5.
    assert rewards[-1] == pytest.approx(-0.5)
    # Episode return equals exactly the terminal reward.
    assert sum(rewards) == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
# 5. Default horizon is 20
# ---------------------------------------------------------------------------

def test_rr_sparse_horizon_default_20() -> None:
    """`patch §1.2` — default horizon ``H = 20`` for the sparse subcase."""
    env = _make_sparse_env(opp_probs=(0.5, 0.5), seed=0)
    assert env._horizon == 20
    md = env.game_info()
    assert md["horizon"] == 20


# ---------------------------------------------------------------------------
# 6. Canonical sign matches the dense RoR
# ---------------------------------------------------------------------------

def test_rr_sparse_canonical_sign_unchanged() -> None:
    """`patch §1` — the sparse flag must NOT mutate ``env_canonical_sign``.

    The dense RoR env publishes a particular ``env_canonical_sign``; the
    sparse subcase preserves it (the patch only changes the reward
    schedule, not the operator-canonical-sign metadata).
    """
    adv_dense = StationaryMixedOpponent(probs=[0.5, 0.5], seed=0)
    dense_env = rules_of_road.build(adversary=adv_dense, horizon=20, seed=0)
    sparse_env = _make_sparse_env(opp_probs=(0.5, 0.5), seed=0)

    assert sparse_env.env_canonical_sign == dense_env.env_canonical_sign
    assert (
        sparse_env.game_info()["env_canonical_sign"]
        == dense_env.game_info()["env_canonical_sign"]
    )
    # canonical_sign metadata key is also preserved.
    assert (
        sparse_env.game_info()["canonical_sign"]
        == dense_env.game_info()["canonical_sign"]
    )


# ---------------------------------------------------------------------------
# 7. Seed determinism: same seed ⇒ same trajectory
# ---------------------------------------------------------------------------

def test_rr_sparse_seed_determinism() -> None:
    """`patch §1` — same seed reproduces the same opponent + reward sequence."""
    # Use a non-degenerate stationary mixed opponent so the env's RNG
    # actually drives variability across seeds; same seed must reproduce
    # bit-identical trajectories.
    seed = 42

    def _trajectory(s: int) -> Tuple[List[int], List[float]]:
        env = _make_sparse_env(opp_probs=(0.5, 0.5), seed=s)
        opp_actions: List[int] = []
        rewards: List[float] = []
        env.reset()
        # Always-Left agent for reproducibility — stress only the
        # opponent + env RNG.
        for _ in range(env._horizon):
            _, r, done, info = env.step(np.array([0]))
            opp_actions.append(int(info["opponent_action"]))
            rewards.append(float(r))
            if done:
                break
        return opp_actions, rewards

    opp_a, r_a = _trajectory(seed)
    opp_b, r_b = _trajectory(seed)
    assert opp_a == opp_b, "opponent action sequence drifted across same-seed runs"
    assert r_a == r_b, "reward sequence drifted across same-seed runs"

    # And different seeds (here for at least one) produce different
    # opponent action sequences — guards against accidentally hard-coding
    # a fixed sequence.
    found_diff = False
    for other_seed in (0, 1, 2, 7, 99):
        opp_other, _ = _trajectory(other_seed)
        if opp_other != opp_a:
            found_diff = True
            break
    assert found_diff, (
        "opponent sequence identical across all tried seeds — RNG seeding "
        "may not be propagating to the adversary"
    )
