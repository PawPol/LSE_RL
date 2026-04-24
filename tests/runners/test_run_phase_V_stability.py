"""Phase V WP4 -- safe-vs-raw stability runner tests.

Spec: ``docs/specs/phase_V_mechanism_experiments.md`` section 7 WP4 +
14.6 figure 4.

Coverage
--------
1. Safe stability on real shortlist -- every Family C safe sweep lands in
   ``{stable_linear, stable_nonlinear}``.  Failure = BLOCKER.
2. Classical stability -- ``stable_linear`` on every Family C task.
3. Raw stress reachable -- at least one Family C task has raw status in
   ``{expansive_bounded, expansive_unbounded, nan_guarded}``.
4. NaN guard -- synthetic ultra-large-beta MDP forces the guard to fire
   (or ``expansive_unbounded`` if +inf propagates without crossing to NaN).
5. Smoke end-to-end -- CLI produces parquet, figure, and manifest
   ``wp4_runs`` block without clobbering other manifest keys.
6/7. Classifier unit tests (decision table + status-set coverage).
8. Family C factory smoke (contract that the runner depends on).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_MUSHROOM = _REPO_ROOT / "mushroom-rl-dev"
if _MUSHROOM.exists() and str(_MUSHROOM) not in sys.path:
    sys.path.insert(0, str(_MUSHROOM))

from experiments.weighted_lse_dp.runners.run_phase_V_stability import (  # noqa: E402
    _PARQUET_COLUMNS,
    _STATUS_ALL,
    _STATUS_STABLE,
    classify_sweep,
    main,
    run_stability,
    sweep_operator,
)
from experiments.weighted_lse_dp.tasks.family_c_raw_stress import (  # noqa: E402
    family_c,
)

_SHORTLIST_PATH = _REPO_ROOT / "results" / "search" / "shortlist.csv"


# ---------------------------------------------------------------------------
# Shared fixture: run once on the live shortlist, reuse for 1/2/3/5.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def live_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    if not _SHORTLIST_PATH.is_file():
        pytest.skip(f"shortlist not found at {_SHORTLIST_PATH}")
    out_dir = tmp_path_factory.mktemp("wp4")
    fig_path = out_dir / "fig_safe_vs_raw_stability.pdf"
    result = run_stability(
        shortlist_path=_SHORTLIST_PATH,
        output_root=out_dir,
        figure_path=fig_path,
        seed=42,
        exact_argv=["pytest_live_run"],
    )
    df = pd.read_parquet(out_dir / "safe_vs_raw_stability.parquet")
    return {
        "result": result, "df": df,
        "output_root": out_dir, "figure_path": fig_path,
    }


# ---------------------------------------------------------------------------
# 1-3. Real-shortlist checks
# ---------------------------------------------------------------------------

def test_safe_stability_on_real_shortlist(live_run: dict[str, Any]) -> None:
    """BLOCKER: safe sweep must be stable_* on every Family C task."""
    safe_df = live_run["df"][live_run["df"]["operator"] == "safe"]
    assert len(safe_df) == 6, f"expected 6 Family C safe sweeps; got {len(safe_df)}"
    unstable = safe_df[~safe_df["status"].isin(_STATUS_STABLE)]
    assert unstable.empty, (
        "BLOCKER: Phase V safety story broken -- safe operator not stable on "
        f"{unstable[['task_id', 'status', 'local_deriv_max']].to_dict('records')}"
    )


def test_classical_stability_on_real_shortlist(live_run: dict[str, Any]) -> None:
    cl_df = live_run["df"][live_run["df"]["operator"] == "classical"]
    non_linear = cl_df[cl_df["status"] != "stable_linear"]
    assert non_linear.empty, (
        f"classical misclassified: "
        f"{non_linear[['task_id', 'status']].to_dict('records')}"
    )


def test_raw_stress_reachable(live_run: dict[str, Any]) -> None:
    """Family C must stress raw on at least one task."""
    raw_df = live_run["df"][live_run["df"]["operator"] == "raw"]
    stress_set = {"expansive_bounded", "expansive_unbounded", "nan_guarded"}
    stressed = raw_df[raw_df["status"].isin(stress_set)]
    assert not stressed.empty, (
        "Family C failed to stress raw: all statuses "
        f"{raw_df['status'].value_counts().to_dict()}"
    )


# ---------------------------------------------------------------------------
# 4. NaN guard on synthetic ultra-large-beta MDP
# ---------------------------------------------------------------------------

def _huge_beta_report(
    *, beta: float, R_fill: float, T: int, S: int = 3,
) -> "object":
    """Build a tiny overflow-prone MDP and run one raw sweep."""
    A = 2
    P = np.zeros((S, A, S), dtype=np.float64)
    R = np.zeros((S, A, S), dtype=np.float64)
    # s=0 -> s=1 -> ... -> s=S-1 absorbing, rewards all = R_fill.
    for s in range(S - 1):
        P[s, :, s + 1] = 1.0
        R[s, :, s + 1] = R_fill
    P[S - 1, :, S - 1] = 1.0

    gamma = 0.95
    kappa_t = np.full(T, 0.965, dtype=np.float64)
    weights = np.full((T, S, A, S), 1.0 / (T * S * A * S), dtype=np.float64)
    beta_t = np.full(T, beta, dtype=np.float64)
    R_max = max(abs(R_fill), 1.0)
    R_box_bound = (1.0 + gamma) / (1.0 - gamma) * R_max + 1.0
    return sweep_operator(
        operator="raw", beta_t=beta_t, kappa_t=kappa_t,
        P=P, R=R, gamma=gamma, T=T,
        weights=weights, R_box_bound=R_box_bound,
    )


def test_nan_guard_synthetic_large_beta() -> None:
    """beta=1e6 + R=1e308 overflows the raw target; sweep must not report
    a stable_* status."""
    rep = _huge_beta_report(beta=1e6, R_fill=1e308, T=5)
    assert rep.status in {"nan_guarded", "expansive_unbounded"}, (
        f"expected nan_guarded / expansive_unbounded; got {rep.status}"
    )
    if rep.status == "nan_guarded":
        assert not rep.sweep_completed


def test_nan_guard_via_infinite_V_next() -> None:
    """Smaller MDP exercising the isfinite guard via +inf propagation."""
    rep = _huge_beta_report(beta=1e6, R_fill=1e306, T=2, S=2)
    assert rep.status not in _STATUS_STABLE, (
        f"expected non-stable on ultra-large-beta sweep; got {rep.status}"
    )


# ---------------------------------------------------------------------------
# 5. Smoke end-to-end (CLI)
# ---------------------------------------------------------------------------

def test_smoke_cli(tmp_path: Path) -> None:
    if not _SHORTLIST_PATH.is_file():
        pytest.skip(f"shortlist not found at {_SHORTLIST_PATH}")

    out_dir = tmp_path / "wp4_cli"
    fig_path = tmp_path / "fig_safe_vs_raw_stability.pdf"
    argv = [
        "--shortlist", str(_SHORTLIST_PATH),
        "--output-root", str(out_dir),
        "--figure-path", str(fig_path),
        "--seed", "42",
    ]
    assert main(argv) == 0

    parquet_path = out_dir / "safe_vs_raw_stability.parquet"
    assert parquet_path.is_file()
    assert fig_path.is_file()

    df = pd.read_parquet(parquet_path)
    for col in _PARQUET_COLUMNS:
        assert col in df.columns, f"parquet missing column {col}"
    assert len(df) > 0 and len(df) % 3 == 0
    for op in ("classical", "safe", "raw"):
        assert (df["operator"] == op).sum() == len(df) // 3

    manifest_path = _REPO_ROOT / "results" / "summaries" / "experiment_manifest.json"
    assert manifest_path.is_file()
    with open(manifest_path) as f:
        m = json.load(f)
    assert "wp4_runs" in m, "WP4 manifest extension missing"
    wp4 = m["wp4_runs"]
    for key in (
        "runner", "exact_argv", "seed_list", "shortlist_path",
        "output_paths", "n_tasks", "per_task_summary", "git_sha",
    ):
        assert key in wp4, f"wp4_runs missing {key}"
    assert wp4["runner"] == "run_phase_V_stability"
    assert wp4["seed_list"] == [42]
    for entry in wp4["per_task_summary"]:
        for k in ("task_id", "classical_status", "safe_status",
                  "raw_status", "raw_local_deriv_p90", "raw_v_max_abs"):
            assert k in entry


# ---------------------------------------------------------------------------
# 6-7. Classifier unit tests
# ---------------------------------------------------------------------------

def test_classify_sweep_decision_table() -> None:
    base = dict(R_box_bound=100.0, kappa_t_max=0.965, gamma=0.95)
    assert classify_sweep(nan_aborted=False, v_max_abs=1.0,
                          local_deriv_max=0.95, **base) == "stable_linear"
    assert classify_sweep(nan_aborted=False, v_max_abs=1.0,
                          local_deriv_max=0.96, **base) == "stable_nonlinear"
    assert classify_sweep(nan_aborted=False, v_max_abs=1.0,
                          local_deriv_max=1.05, **base) == "expansive_bounded"
    assert classify_sweep(nan_aborted=False, v_max_abs=1000.0,
                          local_deriv_max=1.05, **base) == "expansive_unbounded"
    assert classify_sweep(nan_aborted=True, v_max_abs=1.0,
                          local_deriv_max=0.95, **base) == "nan_guarded"
    assert classify_sweep(nan_aborted=False, v_max_abs=float("inf"),
                          local_deriv_max=0.95, **base) == "expansive_unbounded"


def test_status_set_coverage() -> None:
    assert _STATUS_ALL == {
        "stable_linear", "stable_nonlinear",
        "expansive_bounded", "expansive_unbounded", "nan_guarded",
    }
    assert _STATUS_STABLE == {"stable_linear", "stable_nonlinear"}
    assert _STATUS_STABLE.issubset(_STATUS_ALL)


# ---------------------------------------------------------------------------
# 8. Family C factory smoke (contract runner relies on)
# ---------------------------------------------------------------------------

def test_family_c_builder_contract() -> None:
    mdp = family_c.build_mdp(
        lam=0.0,
        psi={"L_tail": 4, "R_penalty": 1.0, "gamma": 0.95,
             "beta_raw_multiplier": 4.0},
    )
    assert int(getattr(mdp, "initial_state", 0)) == 0
    assert float(mdp.info.gamma) == pytest.approx(0.95)
    assert int(mdp.info.horizon) == 4
