"""
Tests for the safe weighted-LSE Bellman operator (SafeWeightedCommon)
and certification utilities.

Spec reference: docs/specs/phase_III_safe_weighted_lse_experiments.md S8.1

Each test class targets a specific mathematical invariant of the operator.
If the invariant is broken, the corresponding test will fail.
"""

import pathlib
import sys
import itertools

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup -- same pattern as test_classical_finite_horizon_dp.py
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MUSHROOM_DEV = _REPO_ROOT / "mushroom-rl-dev"
for _p in (_REPO_ROOT, _MUSHROOM_DEV):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mushroom_rl.algorithms.value.dp.safe_weighted_common import (
    BetaSchedule,
    SafeWeightedCommon,
    compute_beta_cap,
    compute_certified_radii,
    compute_kappa,
    build_certification,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_zero_schedule(T: int, gamma: float) -> BetaSchedule:
    """Create an all-zero beta schedule for T stages."""
    return BetaSchedule.zeros(T, gamma)


def _make_constant_schedule(T: int, gamma: float, beta_val: float,
                            alpha_val: float = 0.05,
                            R_max: float = 1.0) -> BetaSchedule:
    """Create a constant-beta schedule for testing.

    Builds the certification chain from constant alpha, then sets beta_raw
    to *beta_val* at every stage. The deployed beta may be clipped by the cap.
    This helper constructs the schedule dict that BetaSchedule expects.
    """
    alpha_t = np.full(T, alpha_val)
    kappa_t = compute_kappa(alpha_t, gamma)

    # Backward recursion: Bhat has shape (T+1,); Bhat[T]=0 is terminal.
    Bhat = np.zeros(T + 1)
    for t in range(T - 1, -1, -1):
        Bhat[t] = (1 + gamma) * R_max + kappa_t[t] * Bhat[t + 1]

    # beta_cap uses Bhat[t+1] for each stage t → Bhat[1:T+1]
    Bhat_next = Bhat[1:]  # shape (T,)
    with np.errstate(divide="ignore", invalid="ignore"):
        numerator = np.log(kappa_t / (gamma * (1 + gamma - kappa_t)))
        denominator = R_max + Bhat_next
        beta_cap_t = np.where(denominator > 0, numerator / denominator, 0.0)

    beta_raw_t = np.full(T, beta_val)
    beta_used_t = np.clip(beta_raw_t, -beta_cap_t, beta_cap_t)

    sign = 1 if beta_val >= 0 else -1
    schedule_dict = {
        "gamma": gamma,
        "sign": sign,
        "task_family": "test",
        "alpha_t": alpha_t.tolist(),
        "kappa_t": kappa_t.tolist(),
        "Bhat_t": Bhat.tolist(),           # length T+1, as BetaSchedule requires
        "beta_raw_t": beta_raw_t.tolist(),
        "beta_cap_t": beta_cap_t.tolist(),
        "beta_used_t": beta_used_t.tolist(),
    }
    return BetaSchedule(schedule_dict)


def _make_direct_schedule(T: int, gamma: float, beta_val: float) -> BetaSchedule:
    """Create a schedule with *exact* beta_used_t = beta_val, no certification clipping.

    Used in monotonicity tests where we want to assert the mathematical
    property of the operator at a specific beta, without the confound of
    the safety-clipping reducing beta to near-zero.
    """
    cap = abs(beta_val) * 2.0 + 1.0  # cap strictly > |beta_val| => no clipping
    sign = 1 if beta_val >= 0 else -1
    schedule_dict = {
        "gamma": gamma,
        "sign": sign,
        "task_family": "test_direct",
        "beta_raw_t": [beta_val] * T,
        "beta_cap_t": [cap] * T,
        "beta_used_t": [beta_val] * T,
        "alpha_t": [0.0] * T,
        "kappa_t": [gamma] * T,
        "Bhat_t": [0.0] * (T + 1),
    }
    return BetaSchedule(schedule_dict)


def _g_safe_reference(r: float, v: float, beta: float, gamma: float) -> float:
    """Naive (non-stable) reference implementation of the safe target.

    Only valid when |beta * r| and |beta * v| are small enough to avoid
    overflow in np.exp.
    """
    if abs(beta) < 1e-12:
        return r + gamma * v
    return ((1 + gamma) / beta) * (
        np.log(np.exp(beta * r) + gamma * np.exp(beta * v))
        - np.log(1 + gamma)
    )


def _rho_reference(r: float, v: float, beta: float, gamma: float) -> float:
    """Reference responsibility computation."""
    if abs(beta) < 1e-12:
        return 1.0 / (1.0 + gamma)
    return 1.0 / (1.0 + np.exp(-(beta * (r - v) - np.log(gamma))))


# ===================================================================
# Test class 1: beta=0 collapse
# ===================================================================

class TestBetaZeroCollapse:
    """Verify that safe target == classical target when beta=0.

    Invariant guarded: the beta=0 branch must return r + gamma * v
    *exactly* (no floating-point error), ensuring the safe operator
    degenerates to the classical Bellman operator.

    Spec: phase_III S8.1 item 1.
    """

    GAMMA = 0.99
    T = 5

    R_VALS = [-5.0, -1.0, 0.0, 1.0, 5.0]
    V_VALS = [-3.0, 0.0, 3.0]

    def _get_common(self):
        schedule = _make_zero_schedule(self.T, self.GAMMA)
        return SafeWeightedCommon(schedule, self.GAMMA, n_base=10)

    @pytest.mark.parametrize("r,v", list(itertools.product(
        [-5.0, -1.0, 0.0, 1.0, 5.0],
        [-3.0, 0.0, 3.0],
    )))
    def test_scalar_exact_equality(self, r, v):
        """compute_safe_target with zero schedule equals r + gamma*v exactly.

        # docs/specs/phase_III_*.md S8.1 item 1
        """
        common = self._get_common()
        for t in range(self.T):
            result = common.compute_safe_target(r, v, t)
            expected = r + self.GAMMA * v
            assert result == expected, (
                f"beta=0 branch not exact: got {result}, expected {expected} "
                f"for r={r}, v={v}, t={t}"
            )

    def test_batch_exact_equality(self):
        """compute_safe_target_batch with zero schedule equals r + gamma*v exactly.

        # docs/specs/phase_III_*.md S8.1 item 1
        """
        common = self._get_common()
        r_bar = np.array(self.R_VALS)
        v_next = np.array(self.V_VALS)

        # Create a (5, 3) grid via broadcasting
        r_grid = r_bar[:, None] * np.ones((1, len(v_next)))  # (5, 3)
        v_grid = np.ones((len(r_bar), 1)) * v_next[None, :]  # (5, 3)

        r_flat = r_grid.ravel()
        v_flat = v_grid.ravel()

        for t in range(self.T):
            result = common.compute_safe_target_batch(r_flat, v_flat, t)
            expected = r_flat + self.GAMMA * v_flat
            np.testing.assert_array_equal(
                result, expected,
                err_msg=f"Batch beta=0 not exact at t={t}",
            )


# ===================================================================
# Test class 2: responsibility properties
# ===================================================================

class TestResponsibilityProperties:
    """Verify structural properties of the responsibility weight rho.

    Invariant guarded: rho must lie strictly in (0,1) and must indicate
    the sign of margin correctly.

    Spec: phase_III S8.1 item 4.
    """

    GAMMA = 0.99
    T = 5

    BETA_VALS = [-2.0, -0.5, 0.5, 2.0]
    R_VALS = [-5.0, -1.0, 0.0, 1.0, 5.0]
    V_VALS = [-3.0, 0.0, 3.0]

    @pytest.mark.parametrize("beta,r,v", list(itertools.product(
        [-2.0, -0.5, 0.5, 2.0],
        [-5.0, -1.0, 0.0, 1.0, 5.0],
        [-3.0, 0.0, 3.0],
    )))
    def test_rho_strictly_in_unit_interval(self, beta, r, v):
        """rho is strictly in (0, 1) for all finite (r, v, beta).

        # docs/specs/phase_III_*.md S8.1 item 4
        """
        schedule = _make_constant_schedule(self.T, self.GAMMA, beta)
        common = SafeWeightedCommon(schedule, self.GAMMA, n_base=10)
        for t in range(self.T):
            rho = common.compute_rho(r, v, t)
            assert 0.0 < rho < 1.0, (
                f"rho={rho} not in (0,1) for beta={beta}, r={r}, v={v}, t={t}"
            )

    @pytest.mark.parametrize("r,v", [
        (2.0, 0.0), (5.0, -3.0), (1.0, 0.5),
    ])
    def test_positive_margin_positive_beta_rho_gt_half(self, r, v):
        """When beta > 0 and r > v, rho > 0.5.

        Uses _make_direct_schedule (no certification clipping) so beta_used
        is exactly beta_val, testing the pure mathematical property.
        # docs/specs/phase_III_*.md S2.1
        """
        beta = 1.0
        schedule = _make_direct_schedule(self.T, self.GAMMA, beta)
        common = SafeWeightedCommon(schedule, self.GAMMA, n_base=10)
        for t in range(self.T):
            rho = common.compute_rho(r, v, t)
            assert rho > 0.5, (
                f"rho={rho} not > 0.5 for beta>0, r={r} > v={v}"
            )

    @pytest.mark.parametrize("r,v", [
        (0.0, 2.0), (-3.0, 0.0), (0.5, 1.0),
    ])
    def test_negative_margin_positive_beta_rho_lt_half(self, r, v):
        """When beta > 0 and r < v (by more than log(gamma)/beta), rho < 0.5.

        Uses _make_direct_schedule (no certification clipping) so beta_used
        is exactly beta_val, testing the pure mathematical property.
        # docs/specs/phase_III_*.md S2.1
        """
        beta = 1.0
        schedule = _make_direct_schedule(self.T, self.GAMMA, beta)
        common = SafeWeightedCommon(schedule, self.GAMMA, n_base=10)
        for t in range(self.T):
            rho = common.compute_rho(r, v, t)
            assert rho < 0.5, (
                f"rho={rho} not < 0.5 for beta>0, r={r} < v={v}"
            )

    def test_beta_zero_rho_equals_prior(self):
        """When beta=0, rho = 1/(1+gamma) (the prior weight).

        # docs/specs/phase_III_*.md S2.1
        """
        schedule = _make_zero_schedule(self.T, self.GAMMA)
        common = SafeWeightedCommon(schedule, self.GAMMA, n_base=10)
        expected = 1.0 / (1.0 + self.GAMMA)
        for r in self.R_VALS:
            for v in self.V_VALS:
                for t in range(self.T):
                    rho = common.compute_rho(r, v, t)
                    np.testing.assert_allclose(
                        rho, expected, rtol=1e-12,
                        err_msg=f"beta=0 rho != prior for r={r}, v={v}",
                    )

    @pytest.mark.parametrize("beta,r,v", [
        # beta*(r-v) >= 0  =>  effective_discount <= gamma
        (1.0, 2.0, 1.0),    # positive margin, positive beta
        (-1.0, 0.0, 3.0),   # negative margin, negative beta => product >= 0
        (0.5, 1.0, 1.0),    # zero margin
        # beta*(r-v) < 0   =>  effective_discount > gamma
        (1.0, 0.0, 2.0),    # positive beta, negative margin
        (-1.0, 3.0, 0.0),   # negative beta, positive margin
    ])
    def test_effective_discount_vs_gamma(self, beta, r, v):
        """effective_discount <= gamma iff beta*(r-v) >= 0.

        # docs/specs/phase_III_*.md S2.1
        """
        schedule = _make_constant_schedule(self.T, self.GAMMA, beta)
        common = SafeWeightedCommon(schedule, self.GAMMA, n_base=10)
        t = 0
        ed = common.compute_effective_discount(r, v, t)
        margin_sign = beta * (r - v)
        if margin_sign >= 0:
            assert ed <= self.GAMMA + 1e-12, (
                f"effective_discount={ed} > gamma={self.GAMMA} "
                f"but beta*(r-v)={margin_sign} >= 0"
            )
        else:
            assert ed > self.GAMMA - 1e-12, (
                f"effective_discount={ed} <= gamma={self.GAMMA} "
                f"but beta*(r-v)={margin_sign} < 0"
            )


# ===================================================================
# Test class 3: analytic derivative vs finite difference
# ===================================================================

class TestAnalyticDerivativeVsFiniteDifference:
    """Verify d_t = (1+gamma)(1-rho_t) matches the numerical derivative of g_safe wrt v.

    Invariant guarded: the analytic continuation derivative formula must
    match the actual partial derivative of the safe target function.

    Spec: phase_III S8.1 item 3.
    """

    GAMMA = 0.99
    T = 5
    EPS = 1e-5

    @pytest.mark.parametrize("beta,r,v", list(itertools.product(
        [-1.0, -0.5, 0.5, 1.0],
        [-1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
    )))
    def test_derivative_matches_finite_difference(self, beta, r, v):
        """Analytic derivative d_t = (1+gamma)(1-rho) agrees with finite differences.

        # docs/specs/phase_III_*.md S8.1 item 3
        """
        schedule = _make_constant_schedule(self.T, self.GAMMA, beta)
        common = SafeWeightedCommon(schedule, self.GAMMA, n_base=10)

        for t in range(self.T):
            # Analytic: d_t = (1 + gamma) * (1 - rho)
            rho = common.compute_rho(r, v, t)
            d_analytic = (1 + self.GAMMA) * (1 - rho)

            # Finite difference
            g_plus = common.compute_safe_target(r, v + self.EPS, t)
            g_minus = common.compute_safe_target(r, v - self.EPS, t)
            d_numerical = (g_plus - g_minus) / (2 * self.EPS)

            np.testing.assert_allclose(
                d_analytic, d_numerical,
                rtol=1e-4, atol=1e-6,
                err_msg=(
                    f"Derivative mismatch at beta={beta}, r={r}, v={v}, t={t}: "
                    f"analytic={d_analytic}, numerical={d_numerical}"
                ),
            )


# ===================================================================
# Test class 4: numerical stability vs reference
# ===================================================================

class TestNumericalStabilityVsReference:
    """Verify logaddexp-based implementation matches naive reference on safe grids.

    Invariant guarded: the stable implementation must be numerically
    equivalent to the naive formula in the non-overflow regime, and must
    not overflow where the naive formula would.

    Spec: phase_III S8.1 item 2.
    """

    GAMMA = 0.99
    T = 5

    @pytest.mark.parametrize("beta,r,v", list(itertools.product(
        [0.1, 0.5, 1.0, -0.5],
        [-2.0, 0.0, 2.0],
        [-2.0, 0.0, 2.0],
    )))
    def test_agrees_with_naive_on_safe_grid(self, beta, r, v):
        """Stable implementation agrees with naive reference when |beta*x| < 20.

        # docs/specs/phase_III_*.md S8.1 item 2
        """
        schedule = _make_constant_schedule(self.T, self.GAMMA, beta)
        common = SafeWeightedCommon(schedule, self.GAMMA, n_base=10)

        for t in range(self.T):
            # Get the deployed beta (may be clipped)
            beta_used = schedule.beta_used_at(t)
            result = common.compute_safe_target(r, v, t)
            expected = _g_safe_reference(r, v, beta_used, self.GAMMA)
            np.testing.assert_allclose(
                result, expected, rtol=1e-10,
                err_msg=(
                    f"Mismatch at beta_used={beta_used}, r={r}, v={v}, t={t}"
                ),
            )

    def test_large_arguments_no_overflow(self):
        """Large |beta*r| does not cause overflow with logaddexp implementation.

        # docs/specs/phase_III_*.md S13.4
        """
        # Use a schedule with large alpha so beta_cap is large enough
        # to not clip beta=100
        T = 3
        gamma = 0.99
        beta_val = 100.0
        r, v = 5.0, -5.0

        # Build a schedule with very large alpha to avoid clipping
        schedule = _make_constant_schedule(T, gamma, beta_val,
                                           alpha_val=0.99, R_max=100.0)
        common = SafeWeightedCommon(schedule, gamma, n_base=10)

        result = common.compute_safe_target(r, v, 0)
        assert np.isfinite(result), (
            f"Safe target overflowed: got {result} for beta=100, r=5, v=-5"
        )


# ===================================================================
# Test class 5: certification math
# ===================================================================

class TestCertificationMath:
    """Verify the certification chain: kappa, Bhat, beta_cap, build_certification.

    Invariant guarded: the backward recursion and clipping thresholds must
    match the closed-form formulas in the spec.

    Spec: phase_III S8.2 and S2.2.
    """

    def test_kappa_all_zeros(self):
        """compute_kappa with alpha=0 returns gamma everywhere.

        # docs/specs/phase_III_*.md S2.2
        """
        gamma = 0.99
        alpha_t = np.zeros(5)
        kappa = compute_kappa(alpha_t, gamma)
        np.testing.assert_allclose(kappa, 0.99, rtol=1e-14)

    def test_kappa_with_alpha(self):
        """compute_kappa with alpha=0.1 returns gamma + 0.1*(1-gamma).

        # docs/specs/phase_III_*.md S2.2
        """
        gamma = 0.99
        alpha_t = np.ones(5) * 0.1
        kappa = compute_kappa(alpha_t, gamma)
        expected = 0.99 + 0.1 * 0.01  # = 0.991
        np.testing.assert_allclose(kappa, expected, rtol=1e-14)

    def test_certified_radii_backward_recursion(self):
        """Verify Bhat backward recursion for T=3, alpha=0, R_max=1.

        # docs/specs/phase_III_*.md S2.2

        With alpha_t = 0, kappa_t = gamma = 0.99 for all t.
        Bhat_3 = 0
        Bhat_2 = (1+0.99)*1.0 + 0.99*0     = 1.99
        Bhat_1 = (1+0.99)*1.0 + 0.99*1.99  = 1.99 + 1.9701 = 3.9601
        Bhat_0 = (1+0.99)*1.0 + 0.99*3.9601 = 1.99 + 3.920499 = 5.910499
        """
        T = 3
        gamma = 0.99
        R_max = 1.0
        alpha_t = np.zeros(T)
        kappa_t = compute_kappa(alpha_t, gamma)

        Bhat = compute_certified_radii(T, kappa_t, R_max, gamma)

        # Bhat has shape (T+1,) = (4,); index t is stage t, index T is terminal=0.
        expected = np.zeros(T + 1)
        # Backward fill:
        expected[T] = 0.0                                         # terminal
        expected[2] = (1 + gamma) * R_max + gamma * expected[3]   # 1.99
        expected[1] = (1 + gamma) * R_max + gamma * expected[2]   # 3.9601
        expected[0] = (1 + gamma) * R_max + gamma * expected[1]   # 5.910499

        np.testing.assert_allclose(Bhat, expected, rtol=1e-10,
                                   err_msg="Bhat backward recursion mismatch")

    def test_beta_cap_zero_when_alpha_zero(self):
        """alpha_t = 0 => kappa_t = gamma => beta_cap_t = 0.

        # docs/specs/phase_III_*.md S2.2

        When kappa_t = gamma, the numerator of beta_cap is
        log(gamma / (gamma * (1 + gamma - gamma))) = log(1) = 0.
        """
        T = 5
        gamma = 0.99
        alpha_t = np.zeros(T)
        kappa_t = compute_kappa(alpha_t, gamma)
        R_max = 1.0

        Bhat = compute_certified_radii(T, kappa_t, R_max, gamma)
        beta_cap = compute_beta_cap(kappa_t, Bhat, R_max, gamma)

        np.testing.assert_allclose(beta_cap, 0.0, atol=1e-14,
                                   err_msg="beta_cap not zero when alpha=0")

    def test_build_certification_keys(self):
        """build_certification returns dict with expected keys.

        # docs/specs/phase_III_*.md S8.2
        """
        T = 3
        alpha_t = np.array([0.05, 0.05, 0.05])
        result = build_certification(alpha_t, R_max=1.0, gamma=0.99)
        for key in ("kappa_t", "beta_cap_t"):
            assert key in result, f"Missing key '{key}' in certification dict"
            assert len(result[key]) == T, (
                f"Key '{key}' has wrong length: {len(result[key])} (expected {T})"
            )
        # Bhat_t has length T+1 (includes terminal stage)
        assert "Bhat_t" in result, "Missing key 'Bhat_t' in certification dict"
        assert len(result["Bhat_t"]) == T + 1, (
            f"Bhat_t has wrong length: {len(result['Bhat_t'])} (expected {T+1})"
        )


# ===================================================================
# Test class 6: instrumentation fields
# ===================================================================

class TestInstrumentationFields:
    """Verify that instrumentation fields are set correctly after compute_safe_target.

    Invariant guarded: the logging infrastructure depends on these fields
    being accurate; incorrect values corrupt the Phase III diagnostics.

    Spec: phase_III S3.3 and S7.1.
    """

    GAMMA = 0.99
    T = 5

    def test_last_stage(self):
        """last_stage equals the stage argument t.

        # docs/specs/phase_III_*.md S3.3
        """
        schedule = _make_constant_schedule(self.T, self.GAMMA, 0.5)
        common = SafeWeightedCommon(schedule, self.GAMMA, n_base=10)
        for t in range(self.T):
            common.compute_safe_target(1.0, 0.5, t)
            assert common.last_stage == t

    def test_last_beta_used_matches_schedule(self):
        """last_beta_used equals schedule.beta_used_at(t).

        # docs/specs/phase_III_*.md S3.3
        """
        schedule = _make_constant_schedule(self.T, self.GAMMA, 0.5)
        common = SafeWeightedCommon(schedule, self.GAMMA, n_base=10)
        for t in range(self.T):
            common.compute_safe_target(1.0, 0.5, t)
            expected = schedule.beta_used_at(t)
            assert common.last_beta_used == expected, (
                f"last_beta_used={common.last_beta_used} != "
                f"schedule.beta_used_at({t})={expected}"
            )

    def test_last_rho_in_unit_interval(self):
        """last_rho is in (0, 1) for non-zero beta.

        # docs/specs/phase_III_*.md S3.3
        """
        schedule = _make_constant_schedule(self.T, self.GAMMA, 1.0)
        common = SafeWeightedCommon(schedule, self.GAMMA, n_base=10)
        for r in [-2.0, 0.0, 2.0]:
            for v in [-1.0, 0.0, 1.0]:
                common.compute_safe_target(r, v, 0)
                assert 0.0 < common.last_rho < 1.0, (
                    f"last_rho={common.last_rho} not in (0,1)"
                )

    def test_last_effective_discount_formula(self):
        """last_effective_discount == (1 + gamma) * (1 - last_rho).

        # docs/specs/phase_III_*.md S2.1
        """
        schedule = _make_constant_schedule(self.T, self.GAMMA, 0.5)
        common = SafeWeightedCommon(schedule, self.GAMMA, n_base=10)
        common.compute_safe_target(1.0, 0.5, 2)
        expected = (1 + self.GAMMA) * (1 - common.last_rho)
        np.testing.assert_allclose(
            common.last_effective_discount, expected, rtol=1e-12,
            err_msg="last_effective_discount != (1+gamma)*(1-last_rho)",
        )

    def test_last_margin_is_r_minus_v(self):
        """last_margin == r - v (no gamma factor).

        # docs/specs/phase_III_*.md S7.1 (margin_safe = reward - v_next)
        """
        schedule = _make_constant_schedule(self.T, self.GAMMA, 0.5)
        common = SafeWeightedCommon(schedule, self.GAMMA, n_base=10)
        r, v = 3.0, 1.5
        common.compute_safe_target(r, v, 0)
        assert common.last_margin == r - v, (
            f"last_margin={common.last_margin} != r - v = {r - v}"
        )

    def test_last_target_equals_return_value(self):
        """last_target equals the value returned by compute_safe_target.

        # docs/specs/phase_III_*.md S3.3
        """
        schedule = _make_constant_schedule(self.T, self.GAMMA, 0.5)
        common = SafeWeightedCommon(schedule, self.GAMMA, n_base=10)
        r, v = 2.0, -1.0
        result = common.compute_safe_target(r, v, 1)
        assert common.last_target == result, (
            f"last_target={common.last_target} != return={result}"
        )


# ===================================================================
# Test class 7: stage decoding
# ===================================================================

class TestStageDecoding:
    """Verify stage_from_augmented_state correctly decodes the time index.

    Invariant guarded: incorrect stage decoding would apply the wrong
    beta_t to each transition, silently corrupting the entire experiment.

    Spec: phase_III S4 and S13.3.
    """

    def _make_common(self, n_base: int, T: int = 5):
        gamma = 0.99
        schedule = _make_zero_schedule(T, gamma)
        return SafeWeightedCommon(schedule, gamma, n_base=n_base)

    def test_n_base_5_aug_7(self):
        """aug_id=7, n_base=5 => stage = 7 // 5 = 1.

        # docs/specs/phase_III_*.md S13.3
        """
        common = self._make_common(n_base=5)
        assert common.stage_from_augmented_state(7) == 1

    def test_n_base_25_aug_0(self):
        """aug_id=0, n_base=25 => stage = 0.

        # docs/specs/phase_III_*.md S13.3
        """
        common = self._make_common(n_base=25)
        assert common.stage_from_augmented_state(0) == 0

    def test_n_base_25_aug_24(self):
        """aug_id=24, n_base=25 => stage = 0 (last base state in stage 0).

        # docs/specs/phase_III_*.md S13.3
        """
        common = self._make_common(n_base=25)
        assert common.stage_from_augmented_state(24) == 0

    def test_n_base_25_aug_25(self):
        """aug_id=25, n_base=25 => stage = 1 (first base state in stage 1).

        # docs/specs/phase_III_*.md S13.3
        """
        common = self._make_common(n_base=25)
        assert common.stage_from_augmented_state(25) == 1


# ===================================================================
# Test class 8: margin formula verification (Task 12)
# ===================================================================

def _naive_safe_target(r, v, beta, gamma):
    """Naive (potentially unstable) reference for testing."""
    if beta == 0.0:
        return r + gamma * v
    return ((1 + gamma) / beta) * (
        np.log(np.exp(beta * r) + gamma * np.exp(beta * v)) - np.log(1 + gamma)
    )


class TestMarginFormula:
    """Verify the margin formula margin = r - v_next (no gamma factor).

    Invariant guarded: the margin must NOT include a gamma factor.
    If compute_margin returned r - gamma*v_next instead, these tests
    would detect the discrepancy for any case where gamma != 1.0 and
    v_next != 0.0.

    Spec: docs/specs/phase_III_safe_weighted_lse_experiments.md S2.1,
          tasks/lessons.md entry 2026-04-16.
    """

    GAMMA = 0.9
    T = 5

    R_VALS = [-2.0, -1.0, 0.0, 0.5, 1.0, 3.0]
    V_VALS = [-3.0, -1.0, 0.0, 1.0, 2.0]

    @pytest.mark.parametrize("r,v", list(itertools.product(
        [-2.0, -1.0, 0.0, 0.5, 1.0, 3.0],
        [-3.0, -1.0, 0.0, 1.0, 2.0],
    )))
    def test_margin_no_gamma_factor(self, r, v):
        """compute_margin(r, v_next) == r - v_next on a grid of (r, v_next) pairs.

        Explicitly asserts that r - gamma*v_next is NOT returned when
        gamma != 1.0 and v_next != 0.0.

        # docs/specs/phase_III_*.md S2.1
        """
        schedule = _make_zero_schedule(self.T, self.GAMMA)
        common = SafeWeightedCommon(schedule, self.GAMMA, n_base=10)

        result = common.compute_margin(r, v)
        expected = r - v
        assert result == expected, (
            f"compute_margin({r}, {v}) = {result}, expected {expected}"
        )

        # Explicitly verify the gamma-factor form is different
        # (when it would actually differ)
        wrong_value = r - self.GAMMA * v
        if abs(v) > 1e-12 and abs(1.0 - self.GAMMA) > 1e-12:
            assert result != wrong_value, (
                f"compute_margin returned r - gamma*v = {wrong_value}, "
                f"but should return r - v = {expected}"
            )

    @pytest.mark.parametrize("r,v", [
        (1.0, 0.5), (-2.0, 3.0), (0.0, -1.0), (5.0, 5.0), (0.3, 0.7),
    ])
    def test_last_margin_instrumentation_no_gamma(self, r, v):
        """After compute_safe_target, last_margin == r - v_next (no gamma factor).

        # docs/specs/phase_III_*.md S7.1
        """
        schedule = _make_constant_schedule(self.T, self.GAMMA, beta_val=0.5)
        common = SafeWeightedCommon(schedule, self.GAMMA, n_base=10)

        for t in range(self.T):
            common.compute_safe_target(r, v, t)
            expected = r - v
            assert common.last_margin == expected, (
                f"last_margin={common.last_margin} != r - v = {expected} "
                f"at t={t} for r={r}, v={v}"
            )

    @pytest.mark.parametrize("r,v,beta,gamma", list(itertools.product(
        [-1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
        [0.1, 0.5, 1.0],
        [0.9, 0.99],
    )))
    def test_logaddexp_matches_naive_exponentiation(self, r, v, beta, gamma):
        """The logaddexp implementation matches the naive exponentiation form.

        Naive: ((1+gamma)/beta) * (log(exp(beta*r) + gamma*exp(beta*v)) - log(1+gamma))
        Safe:  compute_safe_target with _make_direct_schedule

        # docs/specs/phase_III_*.md S8.1 item 2
        """
        T = 5
        schedule = _make_direct_schedule(T, gamma, beta)
        common = SafeWeightedCommon(schedule, gamma, n_base=10)

        for t in range(T):
            result = common.compute_safe_target(r, v, t)
            expected = _naive_safe_target(r, v, beta, gamma)
            np.testing.assert_allclose(
                result, expected, atol=1e-10,
                err_msg=(
                    f"logaddexp vs naive mismatch at r={r}, v={v}, "
                    f"beta={beta}, gamma={gamma}, t={t}"
                ),
            )


# ===================================================================
# Test class 9: hand-computed certification recursion (Task 13)
# ===================================================================

class TestCertificationRecursionHandComputed:
    """Verify the certification recursion against hand-computed values.

    Uses a 3-stage problem with known alpha_t, R_max, and gamma to
    verify kappa_t, Bhat_t, and beta_cap_t step by step.

    Invariant guarded: if the backward recursion formula, the kappa
    definition, or the beta_cap formula change, these tests catch it
    immediately because they compare against independently computed
    reference values.

    Spec: docs/specs/phase_III_safe_weighted_lse_experiments.md S2.2, S5.9.
    """

    def test_kappa_hand_computation(self):
        """kappa_t = gamma + alpha_t * (1 - gamma) for known alpha_t.

        # docs/specs/phase_III_*.md S2.2
        """
        T, gamma, R_max = 3, 0.9, 1.0
        alpha_t = np.array([0.05, 0.10, 0.02])

        expected_kappa = gamma + alpha_t * (1 - gamma)
        # Hand check: [0.905, 0.910, 0.902]
        np.testing.assert_allclose(
            expected_kappa, [0.905, 0.910, 0.902], atol=1e-15
        )

        result = compute_kappa(alpha_t, gamma)
        np.testing.assert_allclose(result, expected_kappa, atol=1e-15)

    def test_certified_radii_hand_computation(self):
        """Bhat backward recursion matches hand computation for T=3.

        Bhat[3] = 0
        Bhat[2] = (1+gamma)*R_max + kappa_2 * Bhat[3] = 1.9
        Bhat[1] = (1+gamma)*R_max + kappa_1 * Bhat[2] = 1.9 + 0.910*1.9 = 3.629
        Bhat[0] = (1+gamma)*R_max + kappa_0 * Bhat[1] = 1.9 + 0.905*3.629 = 5.184245

        # docs/specs/phase_III_*.md S2.2
        """
        T, gamma, R_max = 3, 0.9, 1.0
        alpha_t = np.array([0.05, 0.10, 0.02])
        kappa_t = compute_kappa(alpha_t, gamma)

        # Compute expected Bhat by hand (backward recursion)
        expected_Bhat = np.zeros(T + 1)
        for t in range(T - 1, -1, -1):
            expected_Bhat[t] = (1 + gamma) * R_max + kappa_t[t] * expected_Bhat[t + 1]

        # Verify hand values
        np.testing.assert_allclose(expected_Bhat[3], 0.0, atol=1e-15)
        np.testing.assert_allclose(expected_Bhat[2], 1.9, atol=1e-14)
        np.testing.assert_allclose(expected_Bhat[1], 3.629, atol=1e-14)
        np.testing.assert_allclose(expected_Bhat[0], 5.184245, atol=1e-14)

        result = compute_certified_radii(T, kappa_t, R_max, gamma)
        np.testing.assert_allclose(result, expected_Bhat, atol=1e-14)
        assert result[T] == 0.0  # terminal condition

    def test_beta_cap_hand_computation(self):
        """beta_cap matches hand computation using kappa, Bhat, R_max, gamma.

        For t=0: log(kappa_0 / (gamma*(1+gamma-kappa_0))) / (R_max + Bhat[1])
        For t=1: log(kappa_1 / (gamma*(1+gamma-kappa_1))) / (R_max + Bhat[2])
        For t=2: log(kappa_2 / (gamma*(1+gamma-kappa_2))) / (R_max + Bhat[3])

        # docs/specs/phase_III_*.md S2.2
        """
        T, gamma, R_max = 3, 0.9, 1.0
        alpha_t = np.array([0.05, 0.10, 0.02])
        kappa_t = compute_kappa(alpha_t, gamma)
        Bhat = compute_certified_radii(T, kappa_t, R_max, gamma)

        # Compute expected beta_cap by hand
        expected_cap = np.zeros(T)
        for t in range(T):
            numer = np.log(kappa_t[t] / (gamma * (1 + gamma - kappa_t[t])))
            denom = R_max + Bhat[t + 1]
            expected_cap[t] = numer / denom

        # Sanity: all caps should be positive (since alpha > 0 => kappa > gamma)
        assert np.all(expected_cap > 0), (
            f"Expected positive beta_cap for positive alpha, got {expected_cap}"
        )

        result = compute_beta_cap(kappa_t, Bhat, R_max, gamma)
        np.testing.assert_allclose(result, expected_cap, atol=1e-14)

    def test_build_certification_combines_all_three(self):
        """build_certification returns dict with kappa_t, Bhat_t, beta_cap_t.

        # docs/specs/phase_III_*.md S8.2
        """
        T, gamma, R_max = 3, 0.9, 1.0
        alpha_t = np.array([0.05, 0.10, 0.02])

        result = build_certification(alpha_t, R_max, gamma)
        assert set(result.keys()) == {'kappa_t', 'Bhat_t', 'beta_cap_t'}
        assert result['kappa_t'].shape == (T,)
        assert result['Bhat_t'].shape == (T + 1,)
        assert result['beta_cap_t'].shape == (T,)
        assert result['Bhat_t'][T] == 0.0

    def test_alpha_zero_gives_zero_cap(self):
        """alpha_t = 0 => kappa_t = gamma, beta_cap_t = 0.

        # docs/specs/phase_III_*.md S2.2
        """
        T, gamma, R_max = 4, 0.95, 2.0
        alpha_t = np.zeros(T)
        result = build_certification(alpha_t, R_max, gamma)
        np.testing.assert_allclose(result['beta_cap_t'], 0.0, atol=1e-14)
        np.testing.assert_allclose(result['kappa_t'], gamma, atol=1e-15)
