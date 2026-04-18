"""
Certification invariant tests for the safe weighted-LSE operator.

Verifies the three certification guarantees from
docs/specs/phase_III_safe_weighted_lse_experiments.md section 8.2:

1. For every stage t and every grid point in the certified box,
   |partial_v g_t^safe(r, v)| <= kappa_t + tol.
2. alpha_t = 0 implies beta_cap_t = 0 and the deployed target
   collapses to classical r + gamma * v.
3. The safe operator maps the certified box into itself.

All tests import directly from safe_weighted_common so they exercise
the actual implementation, not an inline reference.
"""

import itertools
import pathlib
import sys

import numpy as np
import numpy.testing as npt
import pytest

# ---------------------------------------------------------------------------
# Path setup — search parent chain for the repo root containing mushroom-rl-dev
# (robust to both main worktree and git sparse worktrees).
# ---------------------------------------------------------------------------
def _find_repo_root() -> pathlib.Path:
    """Walk parent directories until we find one containing 'mushroom-rl-dev'."""
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "mushroom-rl-dev").is_dir():
            return parent
    # Fallback: assume standard layout (tests/algorithms/ → repo root)
    return here.parents[2]

_REPO_ROOT = _find_repo_root()
_MUSHROOM_DEV = _REPO_ROOT / "mushroom-rl-dev"
for _p in (_REPO_ROOT, _MUSHROOM_DEV):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mushroom_rl.algorithms.value.dp.safe_weighted_common import (  # noqa: E402
    BetaSchedule,
    SafeWeightedCommon,
    build_certification,
    compute_beta_cap,
    compute_certified_radii,
    compute_kappa,
)


# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------

def _make_constant_schedule(
    T: int,
    gamma: float,
    beta_val: float,
    alpha_val: float = 0.05,
    R_max: float = 1.0,
) -> BetaSchedule:
    """Full certification chain from constant alpha; returns a BetaSchedule."""
    alpha_t = np.full(T, alpha_val)
    kappa_t = compute_kappa(alpha_t, gamma)
    Bhat = compute_certified_radii(T, kappa_t, R_max, gamma)
    beta_cap_t = compute_beta_cap(kappa_t, Bhat, R_max, gamma)
    beta_raw_t = np.full(T, beta_val)
    beta_used_t = np.clip(beta_raw_t, -beta_cap_t, beta_cap_t)
    sign = 1 if beta_val >= 0 else -1
    return BetaSchedule({
        "gamma": gamma,
        "sign": sign,
        "task_family": "test",
        "alpha_t": alpha_t.tolist(),
        "kappa_t": kappa_t.tolist(),
        "Bhat_t": Bhat.tolist(),
        "beta_raw_t": beta_raw_t.tolist(),
        "beta_cap_t": beta_cap_t.tolist(),
        "beta_used_t": beta_used_t.tolist(),
    })


def _make_direct_schedule(T: int, gamma: float, beta_val: float) -> BetaSchedule:
    """Exact beta_used_t = beta_val with no certification clipping.

    Used when testing the operator at a specific beta without the
    confound of the cap formula reducing it.
    """
    cap = abs(beta_val) * 2.0 + 1.0
    sign = 1 if beta_val >= 0 else -1
    return BetaSchedule({
        "gamma": gamma,
        "sign": sign,
        "task_family": "test_direct",
        "beta_raw_t": [beta_val] * T,
        "beta_cap_t": [cap] * T,
        "beta_used_t": [beta_val] * T,
        "alpha_t": [0.0] * T,
        "kappa_t": [gamma] * T,
        "Bhat_t": [0.0] * (T + 1),
    })


# ===================================================================
# Test Class 1: Local derivative bound
# docs/specs/phase_III_safe_weighted_lse_experiments.md section 8.2 item 1
# ===================================================================

class TestLocalDerivativeBound:
    """Verify |partial_v g_t^safe(r,v)| <= kappa_t on the certified box.

    Invariant guarded: the safe clipping cap guarantees that the
    effective discount never exceeds kappa_t on the certified box.
    If this test fails, either the clip cap formula or the operator
    formula has a bug.
    """

    @pytest.mark.parametrize(
        "gamma, alpha_val, beta_val",
        [
            (0.9, 0.05, 2.0),
            (0.9, 0.05, -2.0),
            (0.99, 0.02, 5.0),
            (0.99, 0.10, 10.0),
            (0.5, 0.05, 1.0),
        ],
    )
    def test_derivative_within_kappa_on_certified_box(
        self, gamma, alpha_val, beta_val
    ):
        """For every (r, v) on a grid inside the certified box,
        |partial_v g_t^safe| <= kappa_t + tol.

        # docs/specs/phase_III_safe_weighted_lse_experiments.md section 8.2 item 1
        """
        T = 5
        R_max = 1.0
        schedule = _make_constant_schedule(
            T, gamma, beta_val, alpha_val=alpha_val, R_max=R_max
        )
        swc = SafeWeightedCommon(schedule, gamma, n_base=10)
        tol = 1e-5
        n_grid = 9

        for t in range(T):
            kappa_t = schedule.kappa_at(t)
            bhat_next = schedule._Bhat_t[t + 1]

            r_vals = np.linspace(-R_max, R_max, n_grid)
            v_vals = (
                np.linspace(-bhat_next, bhat_next, n_grid)
                if bhat_next > 1e-12
                else np.array([0.0])
            )

            for r, v in itertools.product(r_vals, v_vals):
                # Use the analytic effective discount = (1+gamma)*(1-rho_t).
                # This equals |∂_v g_t^safe| exactly (chain rule); FD would
                # introduce O(eps) noise that can exceed kappa_t near box
                # boundaries by a small numerical artefact.
                eff_d = swc.compute_effective_discount(r, v, t)
                assert eff_d <= kappa_t + tol, (
                    f"Derivative bound violated: t={t}, r={r:.4f}, v={v:.4f}, "
                    f"beta_used={schedule.beta_used_at(t):.6f}, "
                    f"eff_discount={eff_d:.8f}, kappa_t={kappa_t:.8f}"
                )

    @pytest.mark.parametrize(
        "gamma, beta_val",
        [(0.9, 0.5), (0.9, -0.5), (0.99, 1.0), (0.5, 2.0)],
    )
    def test_effective_discount_matches_finite_differences(
        self, gamma, beta_val
    ):
        """Analytic effective_discount matches numerical derivative of g_safe.

        effective_discount = (1+gamma)*(1-rho) = partial_v g_t^safe.

        # docs/specs/phase_III_safe_weighted_lse_experiments.md section 2.1
        """
        T = 5
        schedule = _make_direct_schedule(T, gamma, beta_val)
        swc = SafeWeightedCommon(schedule, gamma, n_base=10)
        eps = 1e-7
        t = 0  # all stages are identical for direct schedule

        r_vals = np.linspace(-2.0, 2.0, 5)
        v_vals = np.linspace(-2.0, 2.0, 5)

        for r, v in itertools.product(r_vals, v_vals):
            g_plus = swc.compute_safe_target(r, v + eps, t)
            g_minus = swc.compute_safe_target(r, v - eps, t)
            fd_deriv = (g_plus - g_minus) / (2.0 * eps)

            analytic = swc.compute_effective_discount(r, v, t)

            npt.assert_allclose(
                fd_deriv,
                analytic,
                rtol=1e-4,
                atol=1e-8,
                err_msg=(
                    f"FD vs analytic mismatch at r={r:.3f}, v={v:.3f}, "
                    f"beta={beta_val}"
                ),
            )

    def test_beta_zero_derivative_is_gamma(self):
        """When beta=0, |partial_v g| = gamma exactly.

        # docs/specs/phase_III_safe_weighted_lse_experiments.md section 2.1
        """
        T = 1
        for gamma in [0.5, 0.9, 0.99, 0.999]:
            schedule = BetaSchedule.zeros(T, gamma)
            swc = SafeWeightedCommon(schedule, gamma, n_base=10)
            for r in [-1.0, 0.0, 1.0]:
                for v in [-1.0, 0.0, 1.0]:
                    d = swc.compute_effective_discount(r, v, 0)
                    npt.assert_allclose(
                        d, gamma, rtol=1e-12,
                        err_msg=f"beta=0 derivative should be gamma={gamma}"
                    )

    def test_effective_discount_nonnegative(self):
        """effective_discount = (1+gamma)*(1-rho) is always non-negative.

        # docs/specs/phase_III_safe_weighted_lse_experiments.md section 2.1
        """
        T = 1
        gammas = [0.5, 0.9, 0.99]
        betas = [-5.0, -1.0, 0.0, 1.0, 5.0]
        vals = [-3.0, 0.0, 3.0]
        for gamma, beta, r, v in itertools.product(gammas, betas, vals, vals):
            schedule = _make_direct_schedule(T, gamma, beta)
            swc = SafeWeightedCommon(schedule, gamma, n_base=10)
            d = swc.compute_effective_discount(r, v, 0)
            assert d >= -1e-15, (
                f"Negative discount: gamma={gamma}, beta={beta}, "
                f"r={r}, v={v}, d={d}"
            )


# ===================================================================
# Test Class 2: alpha=0 collapse
# docs/specs/phase_III_safe_weighted_lse_experiments.md section 8.2 item 2
# ===================================================================

class TestAlphaZeroCollapse:
    """Verify alpha_t=0 implies beta_cap_t=0 and classical collapse.

    Invariant guarded: when no headroom is allocated (alpha_t=0),
    the certification must force beta_cap_t=0 and the operator
    must reduce exactly to the classical Bellman target r + gamma*v.
    If this test fails, the alpha-to-kappa or clip-cap formula is wrong.
    """

    @pytest.mark.parametrize("gamma", [0.5, 0.9, 0.95, 0.99, 0.999])
    def test_alpha_zero_gives_kappa_equals_gamma(self, gamma):
        """kappa_t = gamma + 0*(1-gamma) = gamma.

        # docs/specs/phase_III_safe_weighted_lse_experiments.md section 2.2
        """
        T = 4
        alpha_t = np.zeros(T)
        kappa_t = compute_kappa(alpha_t, gamma)
        npt.assert_allclose(kappa_t, gamma, atol=1e-15)

    @pytest.mark.parametrize("gamma", [0.5, 0.9, 0.95, 0.99])
    def test_alpha_zero_gives_beta_cap_zero(self, gamma):
        """When alpha_t=0 for all t, beta_cap_t = 0.

        Derivation: kappa_t = gamma, so
        log(gamma / [gamma*(1+gamma-gamma)]) = log(1) = 0,
        hence beta_cap_t = 0 / (R_max + Bhat_{t+1}) = 0.

        # docs/specs/phase_III_safe_weighted_lse_experiments.md section 8.2 item 2
        """
        T = 4
        R_max = 1.0
        alpha_t = np.zeros(T)
        cert = build_certification(alpha_t, R_max, gamma)
        npt.assert_allclose(cert["beta_cap_t"], 0.0, atol=1e-14)

    @pytest.mark.parametrize(
        "gamma, beta_raw",
        [(0.9, 5.0), (0.95, -3.0), (0.99, 100.0), (0.5, 0.1)],
    )
    def test_alpha_zero_clips_any_beta_to_zero(self, gamma, beta_raw):
        """With beta_cap=0, clip(beta_raw, -0, 0) = 0 regardless of beta_raw.

        # docs/specs/phase_III_safe_weighted_lse_experiments.md section 8.2 item 2
        """
        T = 4
        schedule = _make_constant_schedule(
            T, gamma, beta_raw, alpha_val=0.0, R_max=1.0
        )
        for t in range(T):
            assert abs(schedule.beta_used_at(t)) < 1e-14, (
                f"beta_used should be 0 when alpha=0, got {schedule.beta_used_at(t)}"
            )

    @pytest.mark.parametrize("gamma", [0.9, 0.95, 0.99])
    def test_alpha_zero_target_equals_classical(self, gamma):
        """g_safe(r, v, beta=0, gamma) = r + gamma*v exactly.

        # docs/specs/phase_III_safe_weighted_lse_experiments.md section 8.2 item 2
        """
        T = 4
        schedule = _make_constant_schedule(
            T, gamma, beta_val=10.0, alpha_val=0.0, R_max=1.0
        )
        swc = SafeWeightedCommon(schedule, gamma, n_base=10)

        test_points = [
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (-1.5, 0.8),
            (2.0, -3.0),
            (0.5, 0.5),
        ]
        for t in range(T):
            for r, v in test_points:
                target = swc.compute_safe_target(r, v, t)
                expected = r + gamma * v
                npt.assert_allclose(
                    target,
                    expected,
                    atol=1e-14,
                    err_msg=(
                        f"Classical collapse failed: t={t}, r={r}, v={v}, "
                        f"beta_used={schedule.beta_used_at(t)}"
                    ),
                )

    def test_alpha_zero_certified_radii_equal_classical_envelope(self):
        """When kappa_t = gamma, the certified radii Bhat_t reduce to the
        classical value envelope: Bhat_t = (1+gamma)*R_max * sum_{k=0}^{T-1-t} gamma^k.

        # docs/specs/phase_III_safe_weighted_lse_experiments.md section 2.2
        """
        T = 6
        gamma = 0.9
        R_max = 1.0
        alpha_t = np.zeros(T)
        kappa_t = compute_kappa(alpha_t, gamma)
        Bhat = compute_certified_radii(T, kappa_t, R_max, gamma)

        # Verify Bhat[T] = 0
        npt.assert_allclose(Bhat[T], 0.0, atol=1e-15)

        # Verify recursion: Bhat[t] = (1+gamma)*R_max + gamma*Bhat[t+1]
        for t in range(T):
            expected = (1.0 + gamma) * R_max + gamma * Bhat[t + 1]
            npt.assert_allclose(
                Bhat[t], expected, rtol=1e-14,
                err_msg=f"Bhat recursion failed at t={t}"
            )


# ===================================================================
# Test Class 3: Box invariance
# docs/specs/phase_III_safe_weighted_lse_experiments.md section 8.2 item 3
# ===================================================================

class TestBoxInvariance:
    """Verify the safe operator maps the certified box into itself.

    Invariant guarded: if r in [-R_max, R_max] and v in [-Bhat_{t+1}, Bhat_{t+1}],
    then g_t^safe(r, v) in [-Bhat_t, Bhat_t].
    If this test fails, the certified radius recursion or clip cap is wrong.
    """

    @pytest.mark.parametrize(
        "gamma, alpha_val, beta_val",
        [
            (0.9, 0.05, 2.0),
            (0.9, 0.05, -2.0),
            (0.99, 0.02, 5.0),
            (0.99, 0.10, 10.0),
            (0.5, 0.05, 1.0),
            (0.9, 0.10, 0.5),
        ],
    )
    def test_operator_maps_box_into_itself(self, gamma, alpha_val, beta_val):
        """g_t^safe(r, v) in [-Bhat_t, Bhat_t] for (r,v) on the certified box.

        # docs/specs/phase_III_safe_weighted_lse_experiments.md section 8.2 item 3
        """
        T = 5
        R_max = 1.0
        schedule = _make_constant_schedule(
            T, gamma, beta_val, alpha_val=alpha_val, R_max=R_max
        )
        swc = SafeWeightedCommon(schedule, gamma, n_base=10)
        tol = 1e-5
        n_grid = 9

        for t in range(T):
            bhat_t = schedule._Bhat_t[t]
            bhat_next = schedule._Bhat_t[t + 1]

            r_vals = np.linspace(-R_max, R_max, n_grid)
            v_vals = (
                np.linspace(-bhat_next, bhat_next, n_grid)
                if bhat_next > 1e-12
                else np.array([0.0])
            )

            for r, v in itertools.product(r_vals, v_vals):
                target = swc.compute_safe_target(r, v, t)
                assert abs(target) <= bhat_t + tol, (
                    f"Box invariance violated: t={t}, r={r:.4f}, v={v:.4f}, "
                    f"beta_used={schedule.beta_used_at(t):.6f}, "
                    f"g_safe={target:.6f}, Bhat_t={bhat_t:.6f}"
                )

    def test_box_invariance_at_corners(self):
        """Test the corners of the certified box (worst case for box invariance).

        # docs/specs/phase_III_safe_weighted_lse_experiments.md section 8.2 item 3
        """
        T = 5
        gamma = 0.9
        R_max = 1.0
        alpha_val = 0.05
        beta_val = 3.0
        schedule = _make_constant_schedule(
            T, gamma, beta_val, alpha_val=alpha_val, R_max=R_max
        )
        swc = SafeWeightedCommon(schedule, gamma, n_base=10)
        tol = 1e-5

        for t in range(T):
            bhat_t = schedule._Bhat_t[t]
            bhat_next = schedule._Bhat_t[t + 1]

            corners = list(itertools.product(
                [-R_max, R_max],
                [-bhat_next, bhat_next] if bhat_next > 1e-12 else [0.0],
            ))
            for r, v in corners:
                target = swc.compute_safe_target(r, v, t)
                assert abs(target) <= bhat_t + tol, (
                    f"Box invariance at corner: t={t}, r={r:.4f}, v={v:.4f}, "
                    f"g_safe={target:.6f}, Bhat_t={bhat_t:.6f}"
                )

    def test_classical_box_invariance(self):
        """When beta=0, g = r + gamma*v, and the box is the classical envelope.

        |r + gamma*v| <= R_max + gamma*Bhat_{t+1} < (1+gamma)*R_max + gamma*Bhat_{t+1} = Bhat_t.

        # docs/specs/phase_III_safe_weighted_lse_experiments.md section 8.2 item 3
        """
        T = 5
        gamma = 0.9
        R_max = 1.0
        schedule = _make_constant_schedule(
            T, gamma, beta_val=0.0, alpha_val=0.0, R_max=R_max
        )
        swc = SafeWeightedCommon(schedule, gamma, n_base=10)

        for t in range(T):
            bhat_t = schedule._Bhat_t[t]
            bhat_next = schedule._Bhat_t[t + 1]

            corners = list(itertools.product(
                [-R_max, R_max],
                [-bhat_next, bhat_next] if bhat_next > 1e-12 else [0.0],
            ))
            for r, v in corners:
                target = swc.compute_safe_target(r, v, t)
                # beta=0 → classical target
                classical = r + gamma * v
                npt.assert_allclose(target, classical, atol=1e-14)
                assert abs(target) <= bhat_t + 1e-14, (
                    f"Classical box invariance: t={t}, target={target:.6f}, "
                    f"Bhat_t={bhat_t:.6f}"
                )

    def test_box_invariance_large_beta_still_clipped(self):
        """Even with a very large raw beta, clipping ensures box invariance.

        # docs/specs/phase_III_safe_weighted_lse_experiments.md section 8.2 item 3
        """
        T = 5
        gamma = 0.9
        R_max = 1.0
        # Very large beta_raw -- should be heavily clipped
        schedule = _make_constant_schedule(
            T, gamma, beta_val=1000.0, alpha_val=0.05, R_max=R_max
        )
        swc = SafeWeightedCommon(schedule, gamma, n_base=10)
        tol = 1e-5

        for t in range(T):
            bhat_t = schedule._Bhat_t[t]
            bhat_next = schedule._Bhat_t[t + 1]

            # Verify clipping actually happened
            assert abs(schedule.beta_used_at(t)) < 1000.0, (
                f"Expected clipping at t={t}, beta_used={schedule.beta_used_at(t)}"
            )
            assert abs(schedule.beta_used_at(t)) <= schedule.beta_cap_at(t) + 1e-14

            r_vals = np.linspace(-R_max, R_max, 7)
            v_vals = (
                np.linspace(-bhat_next, bhat_next, 7)
                if bhat_next > 1e-12
                else np.array([0.0])
            )
            for r, v in itertools.product(r_vals, v_vals):
                target = swc.compute_safe_target(r, v, t)
                assert abs(target) <= bhat_t + tol, (
                    f"Box invariance with large beta: t={t}, "
                    f"g_safe={target:.6f}, Bhat_t={bhat_t:.6f}"
                )

    def test_bhat_monotone_decreasing_with_stage(self):
        """Bhat_t >= Bhat_{t+1}: certified radii are strictly decreasing toward
        the terminal stage (because (1+gamma)*R_max > 0).

        # docs/specs/phase_III_safe_weighted_lse_experiments.md section 2.2
        """
        T = 8
        gamma = 0.9
        R_max = 1.0
        alpha_t = np.full(T, 0.05)
        kappa_t = compute_kappa(alpha_t, gamma)
        Bhat = compute_certified_radii(T, kappa_t, R_max, gamma)

        for t in range(T):
            assert Bhat[t] > Bhat[t + 1], (
                f"Bhat not monotone: Bhat[{t}]={Bhat[t]:.6f} <= "
                f"Bhat[{t+1}]={Bhat[t+1]:.6f}"
            )
