"""Phase VIII Stage 2 strategic-learning agent baselines smoke test (M7.2).

Drives the Stage-2 dispatcher end-to-end on the two new
strategic-learning agent baselines added in M7.2 (spec §6.3
patch-2026-05-01 §3):

* ``regret_matching_agent`` → :class:`RegretMatchingAgent`
* ``smoothed_fictitious_play_agent`` → :class:`SmoothedFictitiousPlayAgent`

Coverage matrix (per spec §10.3 expected-failure clause for DC-Long50):

* AC-Trap (matrix game) × {RM agent, FP agent} × 1 γ × 1 seed × 100 ep
  — must produce non-NaN, finite returns; AUC must be sensible.
* DC-Long50 (non-matrix game) × {RM agent, FP agent} × 1 γ × 1 seed ×
  100 ep — must complete without raising, even though the agent's
  value-bootstrapping is undefined; this is the documented diagnostic
  fallback (uniform-random action selection).

Validates run.json / metrics.npz schema for both methods, and the
ε == 0 / bellman_residual == 0 mandate from spec §6.3 patch §3.

Budget: < 30 seconds total on the dev box.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from experiments.adaptive_beta.tab_six_games.manifests import (
    Phase8RunRoster,
)
from experiments.adaptive_beta.tab_six_games.runners.run_phase_VIII_stage2_baselines import (  # noqa: E501
    REQUIRED_METRICS,
    SUPPORTED_BASELINE_METHODS,
    dispatch,
)


# Strategic-learning agent baselines added in M7.2 (subset of
# SUPPORTED_BASELINE_METHODS).
M7_2_METHODS = (
    "regret_matching_agent",
    "smoothed_fictitious_play_agent",
)

REQUIRED_RUN_JSON_KEYS = (
    "config",
    "env",
    "seed",
    "git_sha",
    "phase",
    "stage",
    "method",
    "game",
    "subcase",
    "baseline_method",
)

#: Operator-mechanism arrays that MUST NOT appear in a Stage-2
#: baseline metrics.npz (TAB-specific; spec §6.3 mandates strict
#: subset for baselines).
FORBIDDEN_METRICS = (
    "alignment_rate",
    "beta_used",
    "beta_raw",
    "effective_discount_mean",
)


def _ac_trap_subcase() -> Dict[str, Any]:
    """AC-Trap subcase block — matrix game, 2 actions."""
    return {
        "id": "AC-Trap",
        "game": "asymmetric_coordination",
        "game_kwargs": {"horizon": 5},   # short horizon for smoke speed
        "adversary": "finite_memory_regret_matching",
        "adversary_kwargs": {"memory_m": 10},
        "headline_metric": "cumulative_return_auc",
        "t11_guard": "cohens_d",
    }


def _dc_long50_subcase() -> Dict[str, Any]:
    """DC-Long50 subcase block — non-matrix game; expected-failure path."""
    return {
        "id": "DC-Long50",
        "game": "delayed_chain",
        "game_kwargs": {"subcase": "DC-Long50"},
        "adversary": "passive",
        "adversary_kwargs": {},
        "headline_metric": "bellman_residual_beta_auc",
        "t11_guard": "gap_based",
    }


def _make_smoke_config(
    *,
    methods: List[str],
    subcases: List[Dict[str, Any]],
    n_episodes: int = 100,
) -> Dict[str, Any]:
    """Build a minimal in-memory Stage 2 config for the new methods."""
    return {
        "stage": "stage2_baselines",
        "phase": "VIII",
        "episodes": int(n_episodes),
        "seeds": [0],
        "methods": list(methods),
        "method_kwargs_per_method": {
            "regret_matching_agent": {
                "mode": "full_info",
                "value_lr": 0.05,
            },
            "smoothed_fictitious_play_agent": {
                "temperature": 0.2,
                "memory_m": 50,
            },
        },
        "gamma": 0.95,
        "learning_rate": 0.1,
        "q_init": 0.0,
        "epsilon": {
            "start": 1.0,
            "end": 0.05,
            "decay_episodes": max(1, n_episodes // 2),
        },
        "subcases": list(subcases),
    }


@pytest.mark.smoke
def test_m7_2_methods_registered_in_supported_set() -> None:
    """Spec §6.3 patch §3: both new methods must appear in the dispatch
    surface so config typos for the new methods fail loudly the same
    way they do for M7.1 baselines."""
    for method in M7_2_METHODS:
        assert method in SUPPORTED_BASELINE_METHODS, (
            f"M7.2 method {method!r} missing from "
            f"SUPPORTED_BASELINE_METHODS={SUPPORTED_BASELINE_METHODS!r}"
        )


@pytest.mark.smoke
def test_m7_2_runner_smoke_writes_phase8_artifacts(tmp_path: Path) -> None:
    """End-to-end smoke for both M7.2 methods on AC-Trap (matrix game).

    Asserts:
    (a) run.json carries the Phase VIII Stage-2 required keys plus
        ``baseline_method == True``;
    (b) metrics.npz has the per-episode arrays mandated by spec §6.3,
        no operator-mechanism arrays;
    (c) ``epsilon == 0`` always (M7.2 mandate; spec §6.3 patch §3);
    (d) ``bellman_residual == 0`` always (no TD error for non-Q
        agents; same source);
    (e) ``return`` is finite and non-NaN for every episode (matrix-
        game cell).
    """
    n_episodes = 100
    config = _make_smoke_config(
        methods=list(M7_2_METHODS),
        subcases=[_ac_trap_subcase()],
        n_episodes=n_episodes,
    )
    output_root = tmp_path / "phase8_root"

    roster = dispatch(
        config=config,
        seed_override=None,
        output_root=output_root,
        config_path=None,
        fail_fast=True,
    )

    assert isinstance(roster, Phase8RunRoster), (
        f"dispatch() must return Phase8RunRoster, got {type(roster)!r}"
    )
    expected_runs = len(M7_2_METHODS) * 1 * 1  # 2 methods × 1 cell × 1 seed
    assert len(roster) == expected_runs, (
        f"roster must hold {expected_runs} rows, got {len(roster)}"
    )
    statuses = {row.status for row in roster.rows}
    assert statuses == {"completed"}, (
        f"every smoke row must complete; got status set {statuses!r}"
    )
    methods_seen = sorted(row.method for row in roster.rows)
    assert methods_seen == sorted(M7_2_METHODS), (
        f"roster method coverage mismatch: expected {sorted(M7_2_METHODS)}, "
        f"got {methods_seen}"
    )

    raw_root = output_root / "raw"
    for row in roster.rows:
        run_dir = Path(row.result_path)

        # (a) Path layout sanity.
        assert raw_root in run_dir.parents, run_dir
        assert "weighted_lse_dp" not in run_dir.as_posix(), run_dir

        # (a) run.json schema.
        run_json_path = run_dir / "run.json"
        assert run_json_path.exists(), run_json_path
        with open(run_json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for key in REQUIRED_RUN_JSON_KEYS:
            assert key in payload, (key, run_json_path)
        assert payload["phase"] == "VIII", payload
        assert payload["baseline_method"] is True, payload
        assert payload["method"] in M7_2_METHODS, payload
        assert payload["game"] == "asymmetric_coordination", payload
        assert payload["subcase"] == "AC-Trap", payload

        # (b) metrics.npz schema + (c)/(d)/(e) array invariants.
        metrics_path = run_dir / "metrics.npz"
        assert metrics_path.exists(), metrics_path
        with np.load(metrics_path, allow_pickle=False) as data:
            on_disk = set(data.keys())
            for name in REQUIRED_METRICS:
                assert name in on_disk, (name, sorted(on_disk))
                arr = data[name]
                assert arr.shape == (n_episodes,), (name, arr.shape)
            for name in FORBIDDEN_METRICS:
                assert name not in on_disk, (name, sorted(on_disk))

            # (c) ε == 0 always.
            eps_arr = np.asarray(data["epsilon"], dtype=np.float64)
            assert np.all(eps_arr == 0.0), (
                f"strategic-learning agent {payload['method']!r} must report "
                f"ε == 0 (spec §6.3 patch §3), got eps_arr={eps_arr[:5]}..."
            )

            # (d) bellman_residual == 0 always.
            br_arr = np.asarray(data["bellman_residual"], dtype=np.float64)
            assert np.all(br_arr == 0.0), (
                f"strategic-learning agent {payload['method']!r} must report "
                f"bellman_residual == 0 (no TD error); got {br_arr[:5]}..."
            )

            # (e) Returns are finite and non-NaN.
            ret_arr = np.asarray(data["return"], dtype=np.float64)
            assert np.all(np.isfinite(ret_arr)), (
                f"return contains non-finite values for "
                f"{payload['method']!r}: {ret_arr}"
            )

            # AC-Trap stag-hunt yields ε ∈ {0, 3, 5} per step; with
            # horizon 5 the per-episode return must lie in [0, 25].
            assert np.all(ret_arr >= 0.0) and np.all(ret_arr <= 5 * 5.0), (
                f"AC-Trap return out of [0, 25] for "
                f"{payload['method']!r}: min={ret_arr.min()}, "
                f"max={ret_arr.max()}"
            )

            # divergence_event must be 0 always for these baselines.
            div_arr = np.asarray(data["divergence_event"], dtype=np.uint8)
            assert np.all(div_arr == 0), (payload["method"], div_arr[:5])

            # nan_count == 0 always.
            nan_arr = np.asarray(data["nan_count"], dtype=np.int64)
            assert np.all(nan_arr == 0), (payload["method"], nan_arr[:5])


@pytest.mark.smoke
@pytest.mark.parametrize("method", list(M7_2_METHODS))
def test_m7_2_runner_smoke_per_method_codepath(
    tmp_path: Path, method: str
) -> None:
    """Per-method smoke for AC-Trap. Parametrising guarantees a single
    method's regression cannot mask the other."""
    n_episodes = 100
    config = _make_smoke_config(
        methods=[method],
        subcases=[_ac_trap_subcase()],
        n_episodes=n_episodes,
    )
    output_root = tmp_path / "phase8_root"

    roster = dispatch(
        config=config,
        seed_override=None,
        output_root=output_root,
        config_path=None,
        fail_fast=True,
    )

    assert len(roster) == 1, (method, len(roster))
    row = roster.rows[0]
    assert row.status == "completed", row.status
    assert row.method == method, (row.method, method)

    metrics_path = Path(row.result_path) / "metrics.npz"
    with np.load(metrics_path, allow_pickle=False) as data:
        ret_arr = np.asarray(data["return"], dtype=np.float64)
        # The matrix-game wrapper must produce a non-NaN AUC. We
        # assert the cumulative-return AUC is finite (the standard
        # headline metric on AC-Trap; see spec §10.1).
        cum_auc = float(np.trapezoid(ret_arr.cumsum()))
        assert np.isfinite(cum_auc), (method, cum_auc)


@pytest.mark.smoke
@pytest.mark.parametrize("method", list(M7_2_METHODS))
def test_m7_2_runner_dc_long50_fallback_completes(
    tmp_path: Path, method: str
) -> None:
    """DC-Long50 expected-failure fallback (spec §10.3).

    The wrapper has no payoff matrix on ``delayed_chain`` and falls
    back to a uniform-random policy. The runner must:

    - complete without raising (the cell is documented as expected to
      fail-to-learn, NOT to error);
    - emit metrics.npz with the schema-required keys;
    - produce a non-NaN AUC (random-policy AUC, but a number the
      aggregator can use).

    On DC-Long50 with ``L=50``, ``advance``-only (1-action) chain plus
    one wrapper agent action axis (still 1 since the chain has 1 agent
    action), the wrapper's uniform random over n_actions=1 always
    returns 0; reward is 0 until the goal at step 50. With horizon=50
    every episode reaches the goal and earns +1 (deterministic chain).
    """
    n_episodes = 100
    config = _make_smoke_config(
        methods=[method],
        subcases=[_dc_long50_subcase()],
        n_episodes=n_episodes,
    )
    output_root = tmp_path / "phase8_root"

    roster = dispatch(
        config=config,
        seed_override=None,
        output_root=output_root,
        config_path=None,
        fail_fast=True,
    )

    assert len(roster) == 1, (method, len(roster))
    row = roster.rows[0]
    assert row.status == "completed", (
        f"DC-Long50 fallback must complete, not error. "
        f"method={method!r}, status={row.status!r}, "
        f"failure_reason={getattr(row, 'failure_reason', None)!r}"
    )

    metrics_path = Path(row.result_path) / "metrics.npz"
    assert metrics_path.exists(), metrics_path
    with np.load(metrics_path, allow_pickle=False) as data:
        on_disk = set(data.keys())
        for name in REQUIRED_METRICS:
            assert name in on_disk, (method, name, sorted(on_disk))

        ret_arr = np.asarray(data["return"], dtype=np.float64)
        assert np.all(np.isfinite(ret_arr)), (method, ret_arr)
        # AUC of cumulative return is finite — the aggregator can
        # consume the result, even if the policy did not learn
        # anything.
        cum_auc = float(np.trapezoid(ret_arr.cumsum()))
        assert np.isfinite(cum_auc), (method, cum_auc)
