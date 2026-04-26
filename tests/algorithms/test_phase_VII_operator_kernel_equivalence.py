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
