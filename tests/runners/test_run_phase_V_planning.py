"""Phase V WP3 -- tests for the limited-backup planning diagnostic runner.

Spec reference: ``docs/specs/phase_V_mechanism_experiments.md`` section 7
(WP3) and the orchestrator brief.

Tests:
  1. Analytic toy MDP: classical ``||V_k - V*||_inf`` decays monotonically
     to zero and hits within 1e-10 at ``k = horizon``.
  2. Policy-disagreement trajectory converges to zero at ``k >= T``.
  3. Shortlist filtering: runner processes only Family A rows (skips C).
  4. Smoke end-to-end: on the real shortlist with ``--k-max 3``, both
     parquet files are emitted with non-zero row counts and every column
     in the spec.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
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

from experiments.weighted_lse_dp.runners.run_phase_V_planning import (  # noqa: E402
    _METRIC_COLUMNS,
    _SCALAR_COLUMNS,
    limited_backup_sweep,
    load_family_a_shortlist,
    run_planning,
)


# ---------------------------------------------------------------------------
# Analytic toy MDP: 3-state chain with one reward at the tail
# ---------------------------------------------------------------------------

def _toy_chain_mdp(T: int = 3, gamma: float = 0.9, R_tail: float = 1.0) -> Any:
    """Deterministic 3-state chain ``s0 -> s1 -> s2 (absorbing)`` with one
    action.  Reward ``R_tail`` is paid on the final edge ``s_{T-1} -> s_T``.

    The analytic optimum is::

        V*[t, s=t] = gamma^(T - 1 - t) * R_tail   for t in [0, T-1]
        V*[T, s]   = 0

    Here ``s = t`` on the visited trajectory (deterministic chain), so
    ``V*[0, 0] = gamma^{T-1} R_tail``.  Under a limited backup budget
    ``k < T`` the terminal reward has only propagated back ``k`` stages:
    ``V_k[T - k, s] = gamma^{k - 1} R_tail``; ``V_k[t, :] = 0`` for
    ``t < T - k``.

    Returns a MushroomRL-style ``FiniteMDP`` (a ``SimpleNamespace`` with
    ``.p``, ``.r``, ``.info.gamma``, ``.info.horizon``, ``.initial_state``).
    """
    S = T + 1                               # states 0..T (state T = terminal)
    A = 1
    P = np.zeros((S, A, S), dtype=np.float64)
    R = np.zeros((S, A, S), dtype=np.float64)
    for s in range(T):
        P[s, 0, s + 1] = 1.0
    P[T, 0, T] = 1.0                        # absorbing self-loop
    R[T - 1, 0, T] = float(R_tail)

    info = SimpleNamespace(gamma=float(gamma), horizon=int(T))
    return SimpleNamespace(p=P, r=R, info=info, initial_state=0)


def _classical_only_schedule(T: int) -> dict[str, np.ndarray]:
    """beta = 0 schedule -> safe operator collapses to classical DP."""
    return {
        "beta_used_t": np.zeros(T, dtype=np.float64),
        "beta_cap_t": np.zeros(T, dtype=np.float64),
        "beta_raw_t": np.zeros(T, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# 1. Monotone decay on toy MDP
# ---------------------------------------------------------------------------

def test_classical_v_err_decays_to_zero_on_toy_chain():
    T = 5
    gamma = 0.9
    R_tail = 1.0
    mdp = _toy_chain_mdp(T=T, gamma=gamma, R_tail=R_tail)
    sched = _classical_only_schedule(T)
    k_values = list(range(1, T + 1))
    sweep = limited_backup_sweep(mdp, sched, k_values=k_values)

    V_star = sweep["V_cl_star"]
    # Expected closed form for the deterministic chain at full horizon:
    #   V*[t, s] = gamma^(T - 1 - s) * R_tail when t <= s < T,
    #   V*[t, s] = 0 otherwise (unreachable / past terminal).
    # The reward is paid on the final edge s_{T-1} -> s_T, so the present
    # value from (t, s) is gamma^(T - 1 - s) R_tail whenever the agent has
    # enough remaining stages (T - t) to cover the path (T - s).
    V_star_expected = np.zeros_like(V_star)
    for t in range(T):
        for s in range(t, T):
            V_star_expected[t, s] = (gamma ** (T - 1 - s)) * R_tail
    np.testing.assert_allclose(V_star, V_star_expected, atol=1e-12)

    # Sup-norm error per k at stage t=0 collapses monotonically.
    errs = np.zeros(T, dtype=np.float64)
    for i, k in enumerate(k_values):
        V_k = sweep["V_cl"][i]
        errs[i] = float(np.max(np.abs(V_k - V_star)))

    # Monotonic non-increasing in k.
    assert np.all(np.diff(errs) <= 1e-12), (
        f"||V_k - V*||_inf not monotone: {errs}"
    )
    # Error at k = T must hit zero within floating-point tolerance.
    assert errs[-1] < 1e-10, f"||V_T - V*||_inf = {errs[-1]!r} > 1e-10"
    # At k = 0 (conceptually) everything is zero; at k=1 only the terminal
    # reward is propagated one stage -- so V_1[T-1, T-1] = R_tail only.
    assert errs[0] > 0.0
    assert errs[T - 1] < 1e-10


# ---------------------------------------------------------------------------
# 2. Policy-disagreement trajectory converges at k >= T
# ---------------------------------------------------------------------------

def test_policy_disagreement_vanishes_at_full_horizon():
    """Safe vs classical greedy policies must agree under d_ref once both
    operators have converged (k >= T).  With beta = 0 the two operators are
    bit-identical, so this must hold at every k, but the spec-stated
    invariant is the k >= T convergence case."""
    T = 4
    mdp = _toy_chain_mdp(T=T, gamma=0.9, R_tail=2.0)
    sched = _classical_only_schedule(T)
    k_values = list(range(1, T + 1))
    sweep = limited_backup_sweep(mdp, sched, k_values=k_values)

    # At k = T the classical and safe Q-tables must coincide (beta = 0 is
    # the classical-recovery path).  Their argmax is thus identical.
    Q_cl_T = sweep["Q_cl"][-1]
    Q_safe_T = sweep["Q_safe"][-1]
    pi_cl = np.argmax(Q_cl_T, axis=-1)
    pi_safe = np.argmax(Q_safe_T, axis=-1)
    assert np.array_equal(pi_cl, pi_safe), (
        "safe vs classical argmax differ at k = T under beta = 0"
    )
    # And the full-horizon references also agree.
    np.testing.assert_allclose(
        sweep["Q_cl_star"], sweep["Q_safe_star"], atol=1e-12
    )


# ---------------------------------------------------------------------------
# 3. Shortlist filtering: only Family A rows are processed
# ---------------------------------------------------------------------------

def test_shortlist_filter_drops_family_c_rows(tmp_path: Path):
    """Synthetic shortlist mixes A and C; loader returns A-only."""
    shortlist_path = tmp_path / "shortlist.csv"
    df = pd.DataFrame({
        "family": ["A", "C", "A", "B", "C"],
        "psi_json": [
            json.dumps({"L": 4, "R": 1.0, "gamma": 0.95, "shape": "flat"}),
            json.dumps({"L_tail": 4, "R_penalty": 1.0,
                        "beta_raw_multiplier": 4.0, "gamma": 0.95}),
            json.dumps({"L": 6, "R": 2.0, "gamma": 0.95, "shape": "flat"}),
            json.dumps({"variant": "single_event", "L": 4, "gamma": 0.95,
                        "b": 0.2, "p": 0.1, "C": 2.0}),
            json.dumps({"L_tail": 8, "R_penalty": 2.0,
                        "beta_raw_multiplier": 4.0, "gamma": 0.95}),
        ],
        "lam": [0.1, 0.0, 0.2, 0.15, 0.0],
        "value_gap_norm": [0.005, 0.01, 0.006, 0.004, 0.01],
    })
    df.to_csv(shortlist_path, index=False)

    out = load_family_a_shortlist(shortlist_path)
    assert list(out["family"]) == ["A", "A"]
    assert len(out) == 2
    # Explicit override still works.
    out_ac = load_family_a_shortlist(shortlist_path, families=["A", "C"])
    assert set(out_ac["family"]) == {"A", "C"}


# ---------------------------------------------------------------------------
# 4. Smoke end-to-end on the real shortlist
# ---------------------------------------------------------------------------

_REAL_SHORTLIST = Path("/tmp/phaseV_with_E/shortlist.csv")


@pytest.mark.skipif(
    not _REAL_SHORTLIST.exists(),
    reason="real shortlist at /tmp/phaseV_with_E/shortlist.csv not present",
)
def test_smoke_end_to_end_on_real_shortlist(tmp_path: Path):
    """Run the pipeline with k_max=3 on the frozen shortlist and assert
    both parquet files are emitted with non-zero row counts and every
    column in the WP3 schema."""
    output_root = tmp_path / "planning"
    result = run_planning(
        _REAL_SHORTLIST,
        output_root=output_root,
        k_max=3,
        families=["A"],
        seed=42,
        exact_argv=["pytest", "--k-max", "3"],
        dry_run=False,
        emit_figures=False,
    )
    assert result["n_tasks"] >= 1
    metrics_path = output_root / "limited_backup_metrics.parquet"
    scalars_path = output_root / "limited_backup_scalars.parquet"
    assert metrics_path.exists()
    assert scalars_path.exists()
    m_df = pd.read_parquet(metrics_path)
    s_df = pd.read_parquet(scalars_path)
    assert len(m_df) > 0
    assert len(s_df) > 0
    for col in _METRIC_COLUMNS:
        assert col in m_df.columns, f"metrics parquet missing column {col!r}"
    for col in _SCALAR_COLUMNS:
        assert col in s_df.columns, f"scalars parquet missing column {col!r}"
    # At each k: exactly two rows per (task, t, operator) -> classical + safe.
    ops = set(m_df["operator"].unique())
    assert ops == {"classical", "safe"}
    # Every task in the metrics df must appear in scalars too.
    assert set(m_df["task_id"].unique()) == set(s_df["task_id"].unique())
    # k in [1, 3] only when --k-max=3.
    assert set(m_df["k"].unique()).issubset({1, 2, 3})
