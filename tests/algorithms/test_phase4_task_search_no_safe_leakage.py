"""
Phase IV-A §5, §9.5: Verify task selection uses no safe-performance data.

The activation-task search must rely only on classical diagnostics and
certification diagnostics. No safe-operator return data may leak into the
scoring function used to select tasks.
"""
from __future__ import annotations

import ast
import inspect
import pathlib
import types

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SEARCH_MODULE = (
    _REPO_ROOT
    / "experiments"
    / "weighted_lse_dp"
    / "geometry"
    / "task_activation_search.py"
)
_RUNNER_MODULE = (
    _REPO_ROOT
    / "experiments"
    / "weighted_lse_dp"
    / "runners"
    / "run_phase4_activation_search.py"
)

# Paths under which Phase IV safe return data would live.
_SAFE_RESULT_DIRS = [
    "phase4/dp",
    "phase4/rl",
    "phase4/safe",
    "phase4_rl",
    "phase4_dp",
]


def _source_of(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def _check_no_safe_path_reference(source: str, filename: str) -> None:
    """Assert that none of the safe-result path fragments appear in the source."""
    for frag in _SAFE_RESULT_DIRS:
        assert frag not in source, (
            f"{filename}: references safe result path '{frag}', "
            f"which must not appear in the activation-search pipeline."
        )


def test_search_uses_only_classical_diagnostics() -> None:
    """§5, §9.5: Task scoring inputs are classical diagnostic fields only.

    The scoring function ``compute_candidate_score`` operates on:
    - pilot_data: output of run_classical_pilot (classical random-policy rollout)
    - schedule: output of build_schedule_v3_from_pilot (closed-form calibration)
    - reward_bound: scalar from task config

    Verify by inspecting the function signature and source code that no
    safe-return quantities appear as inputs.
    """
    from experiments.weighted_lse_dp.geometry.task_activation_search import (
        compute_candidate_score,
    )

    sig = inspect.signature(compute_candidate_score)
    params = set(sig.parameters.keys())

    # The function must accept pilot_data, schedule, reward_bound, and
    # optionally weights. No safe-return fields should be parameters.
    assert "pilot_data" in params, "compute_candidate_score missing pilot_data"
    assert "schedule" in params, "compute_candidate_score missing schedule"
    assert "reward_bound" in params, "compute_candidate_score missing reward_bound"

    # Source must not reference safe return paths.
    source = _source_of(_SEARCH_MODULE)
    _check_no_safe_path_reference(source, "task_activation_search.py")


def test_search_uses_only_certification_diagnostics() -> None:
    """§5, §9.5: Task scoring inputs include only certification diagnostic fields.

    The only certification quantities used for scoring are those derivable
    from the v3 schedule (u_ref_used_t, U_safe_ref_t, trust_clip, etc.) —
    all computed ex ante from pilot margins and closed-form formulas.
    No safe-operator RL/DP returns appear.
    """
    source = _source_of(_SEARCH_MODULE)

    # The source must not import from Phase IV safe runner modules.
    FORBIDDEN_IMPORTS = [
        "run_phase4_dp",
        "run_phase4_rl",
        "aggregate_phase4",
        "from experiments.weighted_lse_dp.runners.run_phase4",
    ]
    for frag in FORBIDDEN_IMPORTS:
        assert frag not in source, (
            f"task_activation_search.py imports/references '{frag}', "
            f"which is a safe-return module."
        )

    # The scoring function must not reference safe_target, safe_return, or
    # safe_episode_return fields.
    FORBIDDEN_FIELDS = [
        "safe_target",
        "safe_return",
        "safe_episode_return",
        "J_safe",
        "episode_reward_safe",
    ]
    for field in FORBIDDEN_FIELDS:
        assert field not in source, (
            f"task_activation_search.py references '{field}', "
            f"which is a safe-return metric."
        )


def test_no_safe_return_in_scoring() -> None:
    """§9.5: Safe-operator return metrics absent from scoring; runner enforces this."""
    # Check the runner also has no safe result path references.
    if not _RUNNER_MODULE.exists():
        pytest.skip("run_phase4_activation_search.py not found")

    runner_source = _source_of(_RUNNER_MODULE)
    _check_no_safe_path_reference(runner_source, "run_phase4_activation_search.py")

    # Verify the runner does not import any safe RL/DP result loaders.
    FORBIDDEN = [
        "load_safe_results",
        "load_phase4_rl",
        "load_phase4_dp",
    ]
    for frag in FORBIDDEN:
        assert frag not in runner_source, (
            f"run_phase4_activation_search.py references '{frag}'"
        )

    # End-to-end: run score_all_candidates on a tiny grid and verify
    # the returned pilot_data has no safe-return keys.
    from experiments.weighted_lse_dp.geometry.task_activation_search import (
        score_all_candidates,
    )

    tiny_grid = [
        {
            "family": "chain_sparse_credit",
            "state_n": 10,
            "gamma": 0.97,
            "horizon": 5,
            "step_cost": 0.0,
            "reward_bound": 1.0,
        }
    ]
    results = score_all_candidates(
        search_grid=tiny_grid,
        seed=42,
        n_pilot_episodes=3,
    )
    assert len(results) == 1

    pilot = results[0].get("pilot_data") or {}
    SAFE_KEYS = {"safe_target", "safe_return", "J_safe", "safe_episode_rewards"}
    leaked = SAFE_KEYS & set(pilot.keys())
    assert not leaked, (
        f"pilot_data contains safe-return keys: {leaked}"
    )
