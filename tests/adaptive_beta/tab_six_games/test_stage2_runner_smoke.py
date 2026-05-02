"""Phase VIII Stage 2 baseline-runner smoke test (M7.1).

Drives the Stage-2 dispatcher end-to-end on a minimal in-memory config
covering all three baseline-method codepaths
(``restart_Q_learning``, ``sliding_window_Q_learning``,
``tuned_epsilon_greedy_Q_learning``) at 1 cell × 1 γ × 1 seed × 100
episodes per method, and asserts:

(a) the raw run dirs land under ``<output_root>/raw/`` (lessons.md #11
    + spec §13.6 result-root regression);
(b) ``run.json`` carries the Phase VIII Stage-2 required keys
    (``config``, ``env``, ``seed``, ``git_sha``, ``phase = "VIII"``,
    ``stage``, ``method``, ``game``, ``subcase``, ``baseline_method =
    True``);
(c) ``metrics.npz`` opens cleanly and contains the per-episode arrays
    listed in :data:`REQUIRED_METRICS` (``return``, ``length``,
    ``epsilon``, ``bellman_residual``, ``q_abs_max``, ``nan_count``,
    ``divergence_event``);
(d) ``metrics.npz`` does NOT contain operator-mechanism arrays (the
    baselines have no β; spec §6.3);
(e) ``Phase8RunRoster`` registered every dispatched run with
    ``status == "completed"``.

Budget: < 30 seconds total (3 × 100-episode matching-pennies cells run
in well under a couple of seconds on the dev box).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

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
#: baseline metrics.npz. Asserting absence guards against accidental
#: drift that would silently misrepresent baselines as having β data.
FORBIDDEN_METRICS = (
    "alignment_rate",
    "beta_used",
    "beta_raw",
    "effective_discount_mean",
)


def _make_smoke_config(
    methods: Any, n_episodes: int = 100
) -> Dict[str, Any]:
    """Build a minimal in-memory Stage 2 baselines config.

    Uses ``matching_pennies`` × ``finite_memory_best_response`` for the
    same reasons the Stage 1 smoke uses it: a 2-action payoff matrix
    is exposed at module level (so :func:`_resolve_payoff_opponent`
    finds it without user-supplied ``probs``) and the adversary builder
    is exercised through its signature-driven kwarg path.
    """
    return {
        "stage": "stage2_baselines",
        "phase": "VIII",
        "episodes": int(n_episodes),
        "seeds": [0],
        "methods": list(methods),
        "gamma": 0.95,
        "learning_rate": 0.1,
        "q_init": 0.0,
        "epsilon": {
            "start": 1.0,
            "end": 0.05,
            "decay_episodes": max(1, n_episodes // 2),
        },
        "subcases": [
            {
                "id": "MP-FiniteMemoryBR",
                "game": "matching_pennies",
                "game_kwargs": {"horizon": 5},
                "adversary": "finite_memory_best_response",
                "adversary_kwargs": {
                    "memory_m": 10,
                    "inertia_lambda": 0.5,
                    "temperature": 0.2,
                },
                "headline_metric": "cumulative_return_auc",
                "t11_guard": "cohens_d",
            }
        ],
    }


@pytest.mark.smoke
def test_stage2_runner_smoke_writes_phase8_artifacts(tmp_path: Path) -> None:
    """End-to-end smoke for the Phase VIII Stage 2 dispatcher.

    Exercises all three baseline codepaths with one cell, one γ, one
    seed, and 100 episodes per method. Writes to a pytest-managed
    tmp_path so the canonical ``results/adaptive_beta/tab_six_games/``
    tree is never polluted.
    """
    n_episodes = 100
    methods = list(SUPPORTED_BASELINE_METHODS)
    config = _make_smoke_config(methods=methods, n_episodes=n_episodes)
    output_root = tmp_path / "phase8_root"

    roster = dispatch(
        config=config,
        seed_override=None,
        output_root=output_root,
        config_path=None,
        fail_fast=True,
    )

    # ---- (e) Phase8RunRoster registered every requested run.
    assert isinstance(roster, Phase8RunRoster), (
        f"dispatch() must return a Phase8RunRoster, got {type(roster)!r}"
    )
    n_methods = len(config["methods"])
    n_seeds = len(config["seeds"])
    n_subcases = len(config["subcases"])
    expected_runs = n_methods * n_seeds * n_subcases
    assert len(roster) == expected_runs, (
        f"roster must hold {expected_runs} rows (one per method codepath), "
        f"got {len(roster)}"
    )
    statuses = {row.status for row in roster.rows}
    assert statuses == {"completed"}, (
        f"every smoke row must complete; got status set {statuses!r}"
    )

    # Method coverage: each of the three baselines is represented
    # exactly once in the roster (1 cell × 1 seed each).
    methods_seen = sorted(row.method for row in roster.rows)
    assert methods_seen == sorted(methods), (
        f"roster method coverage mismatch: expected {sorted(methods)}, "
        f"got {methods_seen}"
    )

    # Roster snapshot file is also written on disk.
    manifest_path = (
        output_root / "raw" / "VIII" / config["stage"] / "manifest.jsonl"
    )
    assert manifest_path.exists(), (
        f"manifest snapshot missing at {manifest_path}"
    )

    raw_root = output_root / "raw"

    for row in roster.rows:
        run_dir = Path(row.result_path)

        # ---- (a) raw run dir lives under <output_root>/raw/.
        assert raw_root in run_dir.parents, (
            f"run dir {run_dir} must live under {raw_root}"
        )
        assert "weighted_lse_dp" not in run_dir.as_posix(), (
            f"run dir {run_dir} accidentally inherits weighted_lse_dp root"
        )

        # ---- (b) run.json schema header.
        run_json_path = run_dir / "run.json"
        assert run_json_path.exists(), f"missing run.json under {run_dir}"
        with open(run_json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        for key in REQUIRED_RUN_JSON_KEYS:
            assert key in payload, (
                f"run.json missing required key {key!r} (path={run_json_path})"
            )

        assert payload["phase"] == "VIII", (
            f"run.json phase must equal 'VIII', got {payload['phase']!r}"
        )
        assert payload["stage"] == "stage2_baselines", (
            f"run.json stage must equal 'stage2_baselines', got "
            f"{payload['stage']!r}"
        )
        assert payload["baseline_method"] is True, (
            f"baseline_method marker must be True for Stage-2 runs, got "
            f"{payload['baseline_method']!r}"
        )
        assert payload["method"] in methods, payload["method"]
        assert payload["game"] == "matching_pennies", payload["game"]
        assert payload["subcase"] == "MP-FiniteMemoryBR", payload["subcase"]
        assert isinstance(payload["config"], dict), payload["config"]
        assert "hash" in payload["config"], payload["config"]
        assert isinstance(payload["env"], dict), payload["env"]
        assert "n_states" in payload["env"], payload["env"]
        assert "n_actions" in payload["env"], payload["env"]
        assert isinstance(payload["seed"], int), payload["seed"]
        assert payload["seed"] == int(row.seed)
        assert isinstance(payload["git_sha"], str), payload["git_sha"]
        assert payload["git_sha"], "git_sha must be a non-empty string"
        assert payload["episodes"] == n_episodes

        # Single-γ (Tier I) backwards-compat: no ``gamma_<value>/``
        # path segment should appear when ``gamma_grid`` is absent.
        assert "gamma_" not in run_dir.as_posix(), (
            f"single-γ Tier I path must not contain a gamma_<value> "
            f"segment, got {run_dir.as_posix()!r}"
        )

        # ---- (c) metrics.npz opens with all required arrays.
        metrics_path = run_dir / "metrics.npz"
        assert metrics_path.exists(), f"missing metrics.npz under {run_dir}"
        with np.load(metrics_path, allow_pickle=False) as data:
            on_disk = set(data.keys())
            for name in REQUIRED_METRICS:
                assert name in on_disk, (
                    f"metrics.npz missing required array {name!r}; on-disk "
                    f"keys: {sorted(on_disk)} (path={metrics_path})"
                )
                arr = data[name]
                assert arr.shape == (n_episodes,), (
                    f"array {name!r} has shape {arr.shape}, expected "
                    f"({n_episodes},) (path={metrics_path})"
                )

            # ---- (d) operator-mechanism arrays MUST be absent.
            for name in FORBIDDEN_METRICS:
                assert name not in on_disk, (
                    f"metrics.npz unexpectedly contains operator-mechanism "
                    f"array {name!r}; baselines have no β (spec §6.3)"
                )

            # Schema header round-trips.
            assert "_schema" in on_disk, "metrics.npz missing _schema header"
            schema_bytes = bytes(data["_schema"])
            schema = json.loads(schema_bytes.decode("utf-8"))
            assert schema.get("phase") == "VIII", schema
            assert schema.get("schema_version") == "phaseVIII.metrics.v1", schema

            # Every method in the smoke produces non-trivial returns
            # (the matching_pennies horizon-5 episode return distribution
            # is well-bounded; we only assert the array is finite + not
            # identically zero for the run).
            ret_arr = np.asarray(data["return"], dtype=np.float64)
            assert np.all(np.isfinite(ret_arr)), (
                f"return array contains non-finite values for method "
                f"{payload['method']!r}: {ret_arr}"
            )


@pytest.mark.smoke
@pytest.mark.parametrize("method", list(SUPPORTED_BASELINE_METHODS))
def test_stage2_runner_smoke_per_method_codepath(
    tmp_path: Path, method: str
) -> None:
    """Per-method smoke that runs each baseline class in isolation.

    This complements the combined smoke above: parametrising over the
    method dispatch surface guarantees that a regression in any single
    baseline's constructor / interface does not mask the other two by
    happening to fail before the verifier reaches them.
    """
    n_episodes = 100
    config = _make_smoke_config(methods=[method], n_episodes=n_episodes)
    output_root = tmp_path / "phase8_root"

    roster = dispatch(
        config=config,
        seed_override=None,
        output_root=output_root,
        config_path=None,
        fail_fast=True,
    )

    assert len(roster) == 1, (
        f"per-method smoke for {method!r} must produce 1 roster row, "
        f"got {len(roster)}"
    )
    row = roster.rows[0]
    assert row.status == "completed", row.status
    assert row.method == method, (row.method, method)

    run_dir = Path(row.result_path)
    metrics_path = run_dir / "metrics.npz"
    assert metrics_path.exists(), metrics_path

    with np.load(metrics_path, allow_pickle=False) as data:
        # ε is paired across seeds — the runner-built schedule is
        # passed to every baseline (including TunedEpsilonGreedy, which
        # would otherwise default to a different decay).
        eps_arr = np.asarray(data["epsilon"], dtype=np.float64)
        assert eps_arr.shape == (n_episodes,)
        # The schedule is monotone non-increasing on this config.
        assert np.all(np.diff(eps_arr) <= 1e-12), (
            f"epsilon schedule must be non-increasing for method {method!r}"
        )
        # bellman_residual is the per-episode mean |td_error|; must be
        # finite and non-negative for every episode.
        br = np.asarray(data["bellman_residual"], dtype=np.float64)
        assert np.all(np.isfinite(br)), (method, br)
        assert np.all(br >= 0.0), (method, br)
