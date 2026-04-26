"""Phase VII §22.1 / test 13.1.6: kernel equivalence pin.

Pins numerical agreement between
``src/lse_rl/operator/tab_operator`` (the single shared kernel) and
``mushroom_rl.algorithms.value.dp.safe_weighted_common.SafeWeightedCommon``
(the certified DP planner) over a fixed (β, γ, r, v) grid.

Backward-compatibility regression for the §22.1 refactor.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from lse_rl.operator import tab_operator as tab
from mushroom_rl.algorithms.value.dp.safe_weighted_common import (
    BetaSchedule,
    SafeWeightedCommon,
)


GAMMAS = [0.5, 0.9, 0.95, 0.99]
BETAS = [-2.0, -1.0, -1e-6, 0.0, 1e-12, 1e-6, 1.0, 2.0]
RVS = [-40.0, -10.0, -1.0, 0.0, 1.0, 10.0, 40.0]


def _single_stage_schedule(beta: float, gamma: float) -> BetaSchedule:
    cap = abs(beta) + 10.0
    schedule_dict = {
        "gamma": gamma,
        "sign": int(np.sign(beta)),
        "task_family": "test",
        "beta_raw_t": [beta],
        "beta_cap_t": [cap],
        "beta_used_t": [beta],
        "alpha_t": [0.0],
        "kappa_t": [gamma],
        "Bhat_t": [0.0, 0.0],
        # reward_bound omitted -> certification recurrence check is skipped.
    }
    return BetaSchedule(schedule_dict)


def _grid_tuples():
    return list(itertools.product(GAMMAS, BETAS, RVS, RVS))


@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("beta", BETAS)
def test_scalar_equivalence(beta: float, gamma: float) -> None:
    schedule = _single_stage_schedule(beta, gamma)
    swc = SafeWeightedCommon(schedule, gamma=gamma, n_base=1)

    for r, v in itertools.product(RVS, RVS):
        ref_target = swc.compute_safe_target(r, v, t=0)
        ref_rho = swc.compute_rho(r, v, t=0)
        ref_eff_d = swc.compute_effective_discount(r, v, t=0)

        kern_target = tab.g(beta, gamma, r, v)
        kern_rho = tab.rho(beta, gamma, r, v)
        kern_eff_d = tab.effective_discount(beta, gamma, r, v)

        if abs(beta) <= tab._EPS_BETA:
            # β = 0 / tiny-β classical-collapse path: bit-exact equality.
            assert kern_target == ref_target, (
                f"β=0 path target diverged: β={beta} γ={gamma} r={r} v={v} "
                f"kernel={kern_target} ref={ref_target}"
            )
            assert kern_rho == ref_rho
            # eff_d in scalar SWC is computed as (1+γ)*(1-1/(1+γ)); the kernel
            # short-circuits to γ. The two values are mathematically equal but
            # may differ by 1 ULP due to fp rounding; pin a tight tolerance.
            assert abs(kern_eff_d - ref_eff_d) <= 1e-15
        else:
            assert abs(kern_target - ref_target) <= 1e-12, (
                f"β={beta} γ={gamma} r={r} v={v} "
                f"kernel={kern_target} ref={ref_target}"
            )
            assert abs(kern_rho - ref_rho) <= 1e-12
            assert abs(kern_eff_d - ref_eff_d) <= 1e-12


@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("beta", BETAS)
def test_batch_matches_scalar(beta: float, gamma: float) -> None:
    r_grid, v_grid = np.meshgrid(np.array(RVS), np.array(RVS), indexing="ij")
    # r_grid, v_grid: (len(RVS), len(RVS))

    g_batch = tab.g_batch(beta, gamma, r_grid, v_grid)
    rho_batch = tab.rho_batch(beta, gamma, r_grid, v_grid)
    d_batch = tab.effective_discount_batch(beta, gamma, r_grid, v_grid)

    for i, r in enumerate(RVS):
        for j, v in enumerate(RVS):
            assert g_batch[i, j] == tab.g(beta, gamma, r, v)
            assert rho_batch[i, j] == tab.rho(beta, gamma, r, v)
            assert d_batch[i, j] == tab.effective_discount(beta, gamma, r, v)


@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("beta", BETAS)
def test_batch_equivalence_with_safe_weighted_common(
    beta: float, gamma: float
) -> None:
    schedule = _single_stage_schedule(beta, gamma)
    swc = SafeWeightedCommon(schedule, gamma=gamma, n_base=1)

    r_grid, v_grid = np.meshgrid(np.array(RVS), np.array(RVS), indexing="ij")
    ref_target = swc.compute_safe_target_batch(r_grid, v_grid, t=0)
    ref_rho = swc.compute_rho_batch(r_grid, v_grid, t=0)

    kern_target = tab.g_batch(beta, gamma, r_grid, v_grid)
    kern_rho = tab.rho_batch(beta, gamma, r_grid, v_grid)

    if abs(beta) <= tab._EPS_BETA:
        np.testing.assert_array_equal(kern_target, ref_target)
        np.testing.assert_array_equal(kern_rho, ref_rho)
    else:
        np.testing.assert_allclose(kern_target, ref_target, atol=1e-12, rtol=0)
        np.testing.assert_allclose(kern_rho, ref_rho, atol=1e-12, rtol=0)


def test_grid_coverage_count() -> None:
    # Documents the exact grid this test pins.
    assert len(GAMMAS) == 4
    assert len(BETAS) == 8
    assert len(RVS) == 7
    assert len(_grid_tuples()) == 4 * 8 * 7 * 7  # 1568 tuples


# ---------------------------------------------------------------------------
# Phase VII M1 / BLOCKER-1: broadcast-shape correctness at classical collapse
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("beta", [0.0, 1e-12, 1e-9, 1e-10])
def test_rho_batch_broadcasts_at_classical_collapse(beta: float) -> None:
    """Regression: rho_batch / effective_discount_batch must broadcast like
    g_batch, not preserve only r's shape (BLOCKER-1, 2026-04-26)."""
    gamma = 0.95
    r = np.array([[[1.0]], [[2.0]]])      # (2, 1, 1)
    v = np.array([[[1.0, 2.0, 3.0]]])     # (1, 1, 3)
    expected_shape = np.broadcast_shapes(r.shape, v.shape)  # (2, 1, 3)

    g = tab.g_batch(beta, gamma, r, v)
    rho = tab.rho_batch(beta, gamma, r, v)
    eff = tab.effective_discount_batch(beta, gamma, r, v)

    assert g.shape == expected_shape
    assert rho.shape == expected_shape
    assert eff.shape == expected_shape

    np.testing.assert_allclose(
        rho, np.full(expected_shape, 1.0 / (1.0 + gamma)), atol=1e-12
    )
    np.testing.assert_allclose(
        eff, np.full(expected_shape, gamma), atol=1e-12
    )


@pytest.mark.parametrize("beta", [0.0, 1e-9])
def test_effective_discount_batch_v_only_classical(beta: float) -> None:
    """v-only broadcast: r is scalar-like, v drives the output shape."""
    gamma = 0.9
    r = np.array(0.5)                    # 0-d
    v = np.array([[1.0, -1.0], [2.0, -2.0]])  # (2, 2)
    eff = tab.effective_discount_batch(beta, gamma, r, v)
    rho = tab.rho_batch(beta, gamma, r, v)
    assert eff.shape == (2, 2)
    assert rho.shape == (2, 2)
    np.testing.assert_allclose(eff, np.full((2, 2), gamma), atol=1e-12)
    np.testing.assert_allclose(rho, np.full((2, 2), 1.0 / (1.0 + gamma)),
                               atol=1e-12)


# ---------------------------------------------------------------------------
# Phase VII M1 / MAJOR-4 (b): compute_safe_target_ev_batch equivalence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("beta", [0.0, -1.5, 1.5])
def test_compute_safe_target_ev_batch_matches_kernel(beta: float) -> None:
    """3-D EV form equals per-(s,a,s') kernel evaluation averaged by P.

    Pins ``SafeWeightedCommon.compute_safe_target_ev_batch`` against a
    direct loop using ``tab_operator.g``.  Catches einsum / broadcasting
    drift after the §22.1 refactor (MAJOR-4 gap, 2026-04-26).
    """
    gamma = 0.9
    schedule = _single_stage_schedule(beta, gamma)
    swc = SafeWeightedCommon(schedule, gamma=gamma, n_base=1)

    # Tiny stochastic MDP: S=3, A=2.  Rows of P sum to 1.
    rng = np.random.default_rng(seed=20260426)
    S, A = 3, 2
    r_bar = rng.uniform(-1.5, 1.5, size=(S, A))         # (S, A)
    V_next = rng.uniform(-2.0, 2.0, size=S)             # (S',)
    p_unnorm = rng.uniform(0.05, 1.0, size=(S, A, S))
    p = p_unnorm / p_unnorm.sum(axis=-1, keepdims=True)  # (S, A, S')

    Q_kernel = swc.compute_safe_target_ev_batch(r_bar, V_next, p, t=0)
    assert Q_kernel.shape == (S, A)

    # Reference: explicit triple loop using the scalar kernel.
    Q_ref = np.zeros((S, A), dtype=np.float64)
    for s in range(S):
        for a in range(A):
            acc = 0.0
            for sp in range(S):
                acc += p[s, a, sp] * tab.g(beta, gamma, r_bar[s, a],
                                           V_next[sp])
            Q_ref[s, a] = acc

    np.testing.assert_allclose(Q_kernel, Q_ref, atol=1e-12, rtol=0)


@pytest.mark.parametrize("beta", [0.0, 1e-12, -2.0, 2.0])
def test_compute_safe_target_ev_batch_linear_collapse(beta: float) -> None:
    """At |β| ≤ ε the EV target equals r + γ·E[V']."""
    gamma = 0.95
    schedule = _single_stage_schedule(beta, gamma)
    swc = SafeWeightedCommon(schedule, gamma=gamma, n_base=1)

    rng = np.random.default_rng(seed=42)
    S, A = 4, 3
    r_bar = rng.uniform(-1.0, 1.0, size=(S, A))
    V_next = rng.uniform(-1.0, 1.0, size=S)
    p_unnorm = rng.uniform(0.1, 1.0, size=(S, A, S))
    p = p_unnorm / p_unnorm.sum(axis=-1, keepdims=True)

    Q = swc.compute_safe_target_ev_batch(r_bar, V_next, p, t=0)

    if abs(beta) <= tab._EPS_BETA:
        E_v = np.einsum("ijk,k->ij", p, V_next)
        np.testing.assert_allclose(Q, r_bar + gamma * E_v,
                                   atol=1e-14, rtol=0)


# ---------------------------------------------------------------------------
# Phase VII M1 / MAJOR-4 (c): instrumentation byte-equivalence
# ---------------------------------------------------------------------------

def _expected_instrumentation(beta: float, gamma: float, r: float,
                              v: float) -> dict:
    """Pre-refactor closed-form expectation for the last_* fields.

    Mirrors :meth:`SafeWeightedCommon.compute_safe_target` for the scalar
    code path: classical collapse for |β| ≤ ε, otherwise the centered/scaled
    weighted-LSE kernel.
    """
    one_plus_gamma = 1.0 + gamma
    margin = r - v
    if abs(beta) <= tab._EPS_BETA:
        target = r + gamma * v
        rho = 1.0 / one_plus_gamma
    else:
        log_gamma = np.log(gamma)
        log_1pg = np.log(one_plus_gamma)
        log_sum = np.logaddexp(beta * r, beta * v + log_gamma)
        target = (one_plus_gamma / beta) * (log_sum - log_1pg)
        from scipy.special import expit
        rho = float(expit(beta * (r - v) - log_gamma))
    eff_d = one_plus_gamma * (1.0 - rho)
    return {
        "target": target,
        "rho": rho,
        "eff_d": eff_d,
        "margin": margin,
        "beta_used": beta,
    }


# Small grid: 4 betas x 3 gammas x ~3 (r,v) tuples = 36 combos > 30.
_INSTR_BETAS = [-1.0, 0.0, 1e-12, 1.5]
_INSTR_GAMMAS = [0.5, 0.9, 0.99]
_INSTR_RV = [(-1.0, 0.5), (0.0, 0.0), (2.0, -1.5)]


@pytest.mark.parametrize("beta", _INSTR_BETAS)
@pytest.mark.parametrize("gamma", _INSTR_GAMMAS)
@pytest.mark.parametrize("r,v", _INSTR_RV)
def test_compute_safe_target_instrumentation_byte_equivalent(
    beta: float, gamma: float, r: float, v: float
) -> None:
    """Pin last_target / last_rho / last_effective_discount / last_margin /
    last_beta_used after the §22.1 refactor (MAJOR-4 gap, 2026-04-26)."""
    schedule = _single_stage_schedule(beta, gamma)
    swc = SafeWeightedCommon(schedule, gamma=gamma, n_base=1)

    out = swc.compute_safe_target(r, v, t=0)
    expected = _expected_instrumentation(beta, gamma, r, v)

    # Return value
    if abs(beta) <= tab._EPS_BETA:
        assert out == expected["target"]
    else:
        assert abs(out - expected["target"]) <= 1e-12

    # Instrumentation (scalar fields)
    assert swc.last_stage == 0
    assert swc.last_beta_used == expected["beta_used"]
    assert swc.last_margin == expected["margin"]

    if abs(beta) <= tab._EPS_BETA:
        # Classical-collapse path: bit-exact.
        assert swc.last_target == expected["target"]
        assert swc.last_rho == expected["rho"]
        assert swc.last_effective_discount == expected["eff_d"]
    else:
        assert abs(swc.last_target - expected["target"]) <= 1e-12
        assert abs(swc.last_rho - expected["rho"]) <= 1e-12
        assert abs(swc.last_effective_discount - expected["eff_d"]) <= 1e-12
