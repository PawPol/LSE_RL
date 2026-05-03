"""Phase VIII Stage 4 sign-switching composite smoke tests (M9).

Spec authority: ``docs/specs/phase_VIII_tab_six_games.md`` §10.5,
§13.4 (composite tests).

Coverage (six tests, in order):

1. Composite env constructs cleanly with both AC-Trap + RR components.
2. ξ flips at the configured dwell.
3. State and action spaces are compatible across the pair.
4. Smoke runner emits the expected metrics.npz schema (incl. composite
   columns).
5. Oracle β reads the correct β per regime (test against canned ξ
   trajectory).
6. Non-oracle methods do NOT have access to ``env.regime`` — regression
   guard that the runner only forwards regime info inside the oracle
   method branch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

import experiments.adaptive_beta.strategic_games as _sg  # noqa: F401
from experiments.adaptive_beta.tab_six_games.composites import (
    SignSwitchingComposite,
)
from experiments.adaptive_beta.tab_six_games.manifests import (
    Phase8RunRoster,
)
from experiments.adaptive_beta.tab_six_games.runners.run_phase_VIII_stage4_composite import (  # noqa: E501
    REQUIRED_METRICS,
    _build_component_env,
    _parse_composite,
    dispatch,
)


def _make_smoke_config(n_episodes: int = 80, dwell: int = 20) -> Dict[str, Any]:
    """Build a minimal in-memory Stage 4 composite config.

    Small dwell (20) and modest episode count (80) so the smoke ξ-flip
    test sees ≥ 3 flips and the runner finishes in < 2 s.
    """
    return {
        "stage": "stage4_composite_smoke",
        "phase": "VIII",
        "episodes": int(n_episodes),
        "seeds": [0],
        "methods": [
            "vanilla",
            "fixed_positive_TAB",
            "fixed_negative_TAB",
            "oracle_beta",
            "hand_adaptive_beta",
            "contraction_UCB_beta",
        ],
        "method_kwargs_per_method": {
            "fixed_positive_TAB": {"beta0": 0.10},
            "fixed_negative_TAB": {"beta0": 0.50},
            "oracle_beta": {
                "beta_g_plus": 0.10,
                "beta_g_minus": -0.50,
            },
        },
        "learning_rate": 0.1,
        "q_init": 0.0,
        "epsilon": {
            "start": 1.0,
            "end": 0.05,
            "decay_episodes": max(1, n_episodes // 2),
        },
        "composite": {
            "type": "sign_switching",
            "g_plus": {
                "game": "asymmetric_coordination",
                "game_kwargs": {"horizon": 20},
                "adversary": "finite_memory_regret_matching",
                "adversary_kwargs": {"memory_m": 20},
                "gamma": 0.60,
            },
            "g_minus": {
                "game": "rules_of_road",
                "game_kwargs": {"horizon": 20},
                "adversary": "stationary",
                "adversary_kwargs": {"probs": [0.7, 0.3]},
                "gamma": 0.60,
            },
            "dwell_grid": [int(dwell)],
        },
    }


# ---------------------------------------------------------------------------
# Test 1: composite env constructs cleanly
# ---------------------------------------------------------------------------
def test_composite_constructs_with_AC_RR_components() -> None:
    cfg = _make_smoke_config()
    spec = _parse_composite(cfg)
    env_plus = _build_component_env(spec.g_plus, seed=0)
    env_minus = _build_component_env(spec.g_minus, seed=42)
    composite = SignSwitchingComposite(
        env_g_plus=env_plus,
        env_g_minus=env_minus,
        dwell=20,
        seed=0,
    )

    assert composite.regime == "plus"
    assert composite.regime_int == +1
    assert composite.dwell == 20
    assert composite.switch_count == 0
    assert composite.regime_history == []  # nothing played yet
    # Both envs reachable.
    assert composite.env_g_plus is env_plus
    assert composite.env_g_minus is env_minus


# ---------------------------------------------------------------------------
# Test 2: ξ flips at the configured dwell
# ---------------------------------------------------------------------------
def test_regime_flips_every_dwell_episodes() -> None:
    cfg = _make_smoke_config()
    spec = _parse_composite(cfg)
    env_plus = _build_component_env(spec.g_plus, seed=0)
    env_minus = _build_component_env(spec.g_minus, seed=42)
    dwell = 5
    composite = SignSwitchingComposite(
        env_g_plus=env_plus,
        env_g_minus=env_minus,
        dwell=dwell,
        seed=0,
    )

    # Drive the env through 3 × dwell + 1 episodes; should observe 3
    # flips and the regime should be back to 'plus' afterwards.
    n_episodes = 3 * dwell + 1
    for _ in range(n_episodes):
        composite.reset()
        # Drive the episode to absorb. The matrix-game horizon is 20,
        # but for this clock test we don't care what the agent plays;
        # any legal action works.
        for _ in range(20):
            _, _, absorbing, _ = composite.step(np.array([0]))
            if absorbing:
                break

    assert composite.switch_count == 3
    # After 3 flips starting from "plus": plus → minus → plus → minus → plus.
    # 3 flips means we are at the regime BEFORE the 4th-cycle flip; given
    # the composite increments the dwell counter on absorbing and flips
    # only on the next reset, the public regime now is the regime under
    # which the LAST played episode sat. Episode 16 (index 15) ran under
    # whatever the regime was at episode start.
    # regime_history: episodes 0-4 = "plus" (1st block), 5-9 = "minus",
    # 10-14 = "plus", 15 = "minus" (post 3rd flip).
    expected_history = (
        ["plus"] * dwell
        + ["minus"] * dwell
        + ["plus"] * dwell
        + ["minus"] * 1
    )
    assert composite.regime_history == expected_history

    # After the initial loop we played 16 episodes:
    #   episodes 0-4 (plus, dwell=5)   -> flip happens at reset of episode 5
    #   episodes 5-9 (minus)           -> flip at reset of episode 10
    #   episodes 10-14 (plus)          -> flip at reset of episode 15
    #   episode 15 (minus)             -> 1 episode in current 'minus' regime
    # To trigger a 4th flip we need 4 more 'minus' episodes (so that
    # _episodes_in_current_regime hits 5) PLUS the next reset (which
    # is the one that fires the flip). 4 episodes + 1 trigger reset
    # = 5 additional reset() calls.
    for _ in range(4):
        composite.reset()
        for _ in range(20):
            _, _, absorbing, _ = composite.step(np.array([0]))
            if absorbing:
                break
    # After 4 more 'minus' episodes -> episodes_in_current_regime == 5,
    # but no reset has yet fired since the absorbing step of episode 19.
    # Trigger the 4th flip by calling reset() once more (entering
    # episode 20 in the post-flip 'plus' regime).
    composite.reset()
    assert composite.switch_count == 4
    assert composite.regime == "plus"


# ---------------------------------------------------------------------------
# Test 3: state and action spaces are compatible across the pair
# ---------------------------------------------------------------------------
def test_state_and_action_space_compatibility() -> None:
    cfg = _make_smoke_config()
    spec = _parse_composite(cfg)
    env_plus = _build_component_env(spec.g_plus, seed=0)
    env_minus = _build_component_env(spec.g_minus, seed=42)

    # Both envs use make_default_state_encoder(horizon=20, n_actions=2)
    # => n_states = horizon * (n_actions + 1) = 20 * 3 = 60.
    assert env_plus.info.observation_space.size[0] == env_minus.info.observation_space.size[0]
    assert env_plus.info.action_space.size[0] == env_minus.info.action_space.size[0]
    assert env_plus.info.observation_space.size[0] == 60
    assert env_plus.info.action_space.size[0] == 2

    # Composite construction succeeds.
    composite = SignSwitchingComposite(
        env_g_plus=env_plus,
        env_g_minus=env_minus,
        dwell=10,
        seed=0,
    )
    assert composite.info.observation_space.size[0] == 60
    assert composite.info.action_space.size[0] == 2
    assert composite.info.gamma == pytest.approx(0.60)


def test_state_space_mismatch_raises() -> None:
    """Defensive: incompatible observation spaces must raise."""
    cfg_plus = _make_smoke_config()["composite"]["g_plus"]
    cfg_minus = _make_smoke_config()["composite"]["g_minus"]
    # Override the G_- horizon so n_states differs.
    cfg_minus = dict(cfg_minus)
    cfg_minus["game_kwargs"] = {"horizon": 10}

    from experiments.adaptive_beta.tab_six_games.runners.run_phase_VIII_stage4_composite import (  # noqa: E501
        _parse_component,
    )
    spec_plus = _parse_component(cfg_plus, "g_plus")
    spec_minus = _parse_component(cfg_minus, "g_minus")
    env_plus = _build_component_env(spec_plus, seed=0)
    env_minus = _build_component_env(spec_minus, seed=42)

    with pytest.raises(ValueError, match="observation_space"):
        SignSwitchingComposite(
            env_g_plus=env_plus,
            env_g_minus=env_minus,
            dwell=10,
            seed=0,
        )


# ---------------------------------------------------------------------------
# Test 4: smoke runner emits expected metrics.npz schema (incl. composite cols)
# ---------------------------------------------------------------------------
def test_smoke_runner_emits_composite_metrics(tmp_path: Path) -> None:
    cfg = _make_smoke_config(n_episodes=60, dwell=20)
    cfg["seeds"] = [0]
    cfg["methods"] = ["vanilla", "oracle_beta"]  # smallest viable subset

    roster = dispatch(
        config=cfg,
        seed_override=None,
        output_root=tmp_path,
        config_path=None,
        fail_fast=True,
    )

    raw_root = tmp_path / "raw" / "VIII" / cfg["stage"]
    assert raw_root.exists()

    # Expect 2 (methods) × 1 (dwell) × 1 (seed) = 2 completed runs.
    completed = [r for r in roster.rows if r.status == "completed"]
    assert len(completed) == 2

    # Verify metrics.npz contents on each.
    for row in completed:
        npz_path = Path(row.result_path) / "metrics.npz"
        assert npz_path.exists(), f"missing metrics.npz for {row.run_id}"
        with np.load(npz_path, allow_pickle=False) as data:
            for key in REQUIRED_METRICS:
                assert key in data.files, (
                    f"required metric {key!r} missing from {npz_path}; "
                    f"got files = {sorted(data.files)}"
                )
            # Composite-specific columns explicitly checked.
            assert "regime_per_episode" in data.files
            assert "episodes_since_switch" in data.files
            assert "switch_count" in data.files
            assert "switch_event" in data.files
            # Sanity: regime_per_episode is +/-1 only.
            regime_arr = data["regime_per_episode"]
            assert regime_arr.shape == (60,)
            assert set(np.unique(regime_arr).tolist()).issubset({-1, 1})


# ---------------------------------------------------------------------------
# Test 5: Oracle β reads the correct β per regime (canned ξ trajectory)
# ---------------------------------------------------------------------------
def test_oracle_reads_correct_beta_per_regime(tmp_path: Path) -> None:
    cfg = _make_smoke_config(n_episodes=20, dwell=5)
    cfg["seeds"] = [0]
    cfg["methods"] = ["oracle_beta"]

    dispatch(
        config=cfg,
        seed_override=None,
        output_root=tmp_path,
        config_path=None,
        fail_fast=True,
    )

    # Find the oracle run dir.
    raw_root = tmp_path / "raw" / "VIII" / cfg["stage"]
    npz_paths = list(raw_root.rglob("metrics.npz"))
    assert len(npz_paths) == 1
    with np.load(npz_paths[0], allow_pickle=False) as data:
        regime = data["regime_per_episode"]
        beta_used = data["beta_used"]

    # Episode 0: schedule has not seen any regime yet -> β_0 = 0.0
    # (OracleBetaSchedule._initial_beta is 0.0 per spec §6.6).
    assert beta_used[0] == pytest.approx(0.0), (
        f"episode 0 oracle β must be 0 before first observation, got "
        f"{beta_used[0]}"
    )

    # For episodes >= 1, β_e is set FROM regime observed at the END of
    # episode (e-1). Composite contract: regime_per_episode[e] is the
    # regime active during episode e. Because the regime is constant
    # within an episode and only flips on episode boundaries (BEFORE
    # the next episode's first step), the regime active during episode
    # (e-1) at episode end equals regime_per_episode[e-1].
    for e in range(1, len(beta_used)):
        prev_regime = int(regime[e - 1])
        if prev_regime == +1:
            expected_beta = 0.10
        else:
            expected_beta = -0.50
        assert beta_used[e] == pytest.approx(expected_beta), (
            f"episode {e}: regime_per_episode[{e - 1}]={prev_regime} "
            f"should produce β={expected_beta}, got {beta_used[e]}"
        )


# ---------------------------------------------------------------------------
# Test 6: non-oracle methods do NOT have access to env.regime
#         (regression guard at the runner boundary)
# ---------------------------------------------------------------------------
def test_non_oracle_methods_do_not_receive_regime(tmp_path: Path) -> None:
    """Inspect the source of the runner's per-cell loop and verify the
    regime is forwarded to ``schedule.update_after_episode`` ONLY
    inside the oracle method branch.

    This is a static-source guard: it pins the runner-side discipline
    that the env exposes ``regime`` publicly but only the oracle
    method is allowed to read it (spec §10.5 / §6.6).
    """
    import experiments.adaptive_beta.tab_six_games.runners.run_phase_VIII_stage4_composite as runner_mod

    src = Path(runner_mod.__file__).read_text(encoding="utf-8")
    # The runner's episode_info construction must guard on
    # method == "oracle_beta". Pin the exact branch to make a
    # silent-regression accident impossible.
    assert 'if method == "oracle_beta":' in src, (
        "non-oracle regime gate missing from M9 runner; spec §10.5 "
        "requires episode_info=None for non-oracle methods."
    )

    # Stronger runtime check: drive the dispatcher with a non-oracle
    # method and verify env.regime is NEVER what the schedule sees.
    cfg = _make_smoke_config(n_episodes=20, dwell=5)
    cfg["seeds"] = [0]
    cfg["methods"] = ["fixed_positive_TAB", "hand_adaptive_beta", "vanilla"]

    # Tripwire: monkey-patch schedule.update_after_episode in each
    # build path to fail loudly if episode_info is non-None for
    # non-oracle methods.
    from experiments.adaptive_beta import schedules as schedules_mod

    original_update = schedules_mod._BaseSchedule.update_after_episode
    contraventions: list[str] = []

    def guarded_update(self, episode_index, rewards, v_next,
                       divergence_event=False, episode_info=None,
                       bellman_residual=None, episode_return=None):
        if (
            episode_info is not None
            and self.name != schedules_mod.METHOD_ORACLE_BETA
        ):
            contraventions.append(
                f"non-oracle schedule {self.name!r} received "
                f"episode_info={episode_info!r}"
            )
        return original_update(
            self, episode_index, rewards, v_next,
            divergence_event=divergence_event,
            episode_info=episode_info,
            bellman_residual=bellman_residual,
            episode_return=episode_return,
        )

    schedules_mod._BaseSchedule.update_after_episode = guarded_update
    try:
        dispatch(
            config=cfg,
            seed_override=None,
            output_root=tmp_path,
            config_path=None,
            fail_fast=True,
        )
    finally:
        schedules_mod._BaseSchedule.update_after_episode = original_update

    assert contraventions == [], (
        f"non-oracle methods leaked regime to their schedule: "
        f"{contraventions[:3]}"
    )
