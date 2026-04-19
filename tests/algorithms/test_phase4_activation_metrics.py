"""Tests for Phase IV-A geometry: activation_metrics module.

Covers spec sections S8.2 (aggregate diagnostics), S11.1 (event-conditioned),
S13 (activation gate checks).
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.weighted_lse_dp.geometry.activation_metrics import (
    compute_aggregate_diagnostics,
    activation_gate_check,
    compute_event_conditioned_diagnostics,
    compute_stage_aggregate,
)
from experiments.weighted_lse_dp.geometry.adaptive_headroom import (
    compute_u_safe_ref,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_transitions():
    """Generate sample transition data for diagnostics tests."""
    rng = np.random.default_rng(42)
    n = 100
    return {
        "natural_shift": rng.normal(0.0, 0.1, n),
        "delta_d": rng.normal(0.0, 0.01, n),
        "target_gap": rng.normal(0.0, 0.05, n),
        "reward_bound": 1.0,
        "beta_used": np.full(n, 0.5),
    }


@pytest.fixture
def active_transitions():
    """Transitions with clearly active (large) natural shifts."""
    n = 100
    return {
        "natural_shift": np.full(n, 0.1),   # all |u| = 0.1 >> 5e-3
        "delta_d": np.full(n, 0.01),
        "target_gap": np.full(n, 0.05),
        "reward_bound": 1.0,
        "beta_used": np.full(n, 1.0),
    }


@pytest.fixture
def inactive_transitions():
    """Transitions with near-zero natural shifts."""
    n = 100
    return {
        "natural_shift": np.full(n, 1e-6),
        "delta_d": np.full(n, 1e-7),
        "target_gap": np.full(n, 1e-7),
        "reward_bound": 1.0,
        "beta_used": np.full(n, 0.001),
    }


# ---------------------------------------------------------------------------
# S9.3: U_safe_ref = theta_safe * xi_ref
# ---------------------------------------------------------------------------

class TestUSafeRef:
    """Spec S9.3: U_safe_ref_t = Theta_safe_t * xi_ref_t."""

    def test_u_safe_ref_equals_theta_times_xi(self):
        """compute_u_safe_ref(theta, xi) = theta * xi element-wise.

        Invariant guarded: safe reference shift factorization is correct.
        """
        # docs/specs/phase_IV_A*.md S9.3
        theta_safe = np.array([0.5, 1.0, 1.5, 2.0])
        xi_ref = np.array([0.1, 0.2, -0.1, 0.05])
        result = compute_u_safe_ref(theta_safe, xi_ref)
        expected = theta_safe * xi_ref
        np.testing.assert_allclose(result, expected, rtol=1e-15)

    def test_u_safe_ref_zero_theta(self):
        """Zero theta_safe gives zero U_safe_ref."""
        theta_safe = np.zeros(3)
        xi_ref = np.array([0.1, 0.2, 0.3])
        result = compute_u_safe_ref(theta_safe, xi_ref)
        np.testing.assert_equal(result, np.zeros(3))


# ---------------------------------------------------------------------------
# S9.3: Event-conditioned diagnostics consistency
# ---------------------------------------------------------------------------

class TestEventConditionedDiagnostics:
    """Spec S9.3: Event-conditioned aggregation must be correct."""

    def test_event_conditioned_equals_masked_aggregate(self):
        """Event-conditioned diagnostics match aggregate on masked subset.

        Invariant guarded: event conditioning is just masking, not different logic.
        """
        # docs/specs/phase_IV_A*.md S11.1
        rng = np.random.default_rng(99)
        n = 200
        ns = rng.normal(0.0, 0.1, n)
        dd = rng.normal(0.0, 0.01, n)
        tg = rng.normal(0.0, 0.05, n)
        rb = 1.0
        mask = rng.random(n) > 0.5

        event_result = compute_event_conditioned_diagnostics(ns, dd, tg, rb, mask)

        # Manually mask and compute via aggregate (no informative_mask, no beta_used)
        # Event-conditioned uses dummy beta=ones, so replicate that
        ns_masked = ns[mask]
        dd_masked = dd[mask]
        tg_masked = tg[mask]
        beta_dummy = np.ones(np.sum(mask))

        agg_result = compute_aggregate_diagnostics(
            ns_masked, dd_masked, tg_masked, rb, beta_dummy
        )

        # They should match exactly (same code path after masking)
        for key in event_result:
            np.testing.assert_allclose(
                event_result[key], agg_result[key], rtol=1e-12, atol=1e-15,
                err_msg=f"Mismatch on key '{key}'"
            )


# ---------------------------------------------------------------------------
# S9.3: Counterfactual replay consistency
# ---------------------------------------------------------------------------

class TestCounterfactualReplay:
    """Spec S9.3: Counterfactual replay metrics match direct recomputation."""

    def test_counterfactual_replay_via_direct_recomputation(self):
        """Given (beta, r, v) triples, natural_shift from geometry matches
        what we pass to aggregate diagnostics.

        Invariant guarded: the pipeline from raw (beta, r, v) to aggregate
        diagnostics is self-consistent.
        """
        from experiments.weighted_lse_dp.geometry.natural_shift import (
            compute_natural_shift,
            small_signal_discount_gap,
            small_signal_target_gap,
        )

        rng = np.random.default_rng(77)
        n = 50
        betas = rng.uniform(0.01, 0.1, n)
        rewards = rng.uniform(-1.0, 1.0, n)
        values = rng.uniform(-1.0, 1.0, n)
        gamma_base = 0.99
        rb = 1.0

        # Direct computation
        ns_direct = compute_natural_shift(betas, rewards, values)
        dd_direct = small_signal_discount_gap(betas, rewards - values, gamma_base)
        tg_direct = small_signal_target_gap(betas, rewards - values, gamma_base)

        diag = compute_aggregate_diagnostics(
            ns_direct, dd_direct, tg_direct, rb, betas
        )

        # Verify mean_abs_u matches
        np.testing.assert_allclose(
            diag["mean_abs_u"],
            float(np.mean(np.abs(ns_direct))),
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            diag["mean_beta_used"],
            float(np.mean(betas)),
            rtol=1e-10,
        )


# ---------------------------------------------------------------------------
# S9.3: Informative mask filtering
# ---------------------------------------------------------------------------

class TestInformativeMask:
    """Spec S9.3: Activation thresholds evaluated only on informative stages."""

    def test_informative_mask_filters_stages(self):
        """When informative_mask is provided, informative_* keys reflect
        only the masked subset.

        Invariant guarded: informative mask actually filters transitions.
        """
        rng = np.random.default_rng(123)
        n = 100
        ns = rng.normal(0.0, 0.1, n)
        dd = rng.normal(0.0, 0.01, n)
        tg = rng.normal(0.0, 0.05, n)
        rb = 1.0
        beta = np.full(n, 0.5)

        # Make first half informative
        mask = np.zeros(n, dtype=bool)
        mask[:50] = True

        result = compute_aggregate_diagnostics(ns, dd, tg, rb, beta, informative_mask=mask)

        # Check informative_mean_abs_u is computed from first 50 only
        expected_informative_mean = float(np.mean(np.abs(ns[:50])))
        np.testing.assert_allclose(
            result["informative_mean_abs_u"],
            expected_informative_mean,
            rtol=1e-10,
        )

    def test_informative_mask_none_and_all_true_same(self):
        """informative_mask=None and informative_mask=all-True give same
        global diagnostics (but all-True also adds informative_ keys).

        Invariant guarded: no-mask default is equivalent to all-True mask
        for the base diagnostics.
        """
        rng = np.random.default_rng(456)
        n = 50
        ns = rng.normal(0.0, 0.1, n)
        dd = rng.normal(0.0, 0.01, n)
        tg = rng.normal(0.0, 0.05, n)
        rb = 1.0
        beta = np.full(n, 0.5)

        result_none = compute_aggregate_diagnostics(ns, dd, tg, rb, beta, informative_mask=None)
        result_true = compute_aggregate_diagnostics(
            ns, dd, tg, rb, beta, informative_mask=np.ones(n, dtype=bool)
        )

        # Base (non-informative_) keys should match
        base_keys = [k for k in result_none if not k.startswith("informative_")]
        for key in base_keys:
            np.testing.assert_allclose(
                result_none[key], result_true[key], rtol=1e-12,
                err_msg=f"Mismatch on base key '{key}'"
            )

        # informative_ keys in result_true should match base keys
        for key in base_keys:
            inf_key = f"informative_{key}"
            if inf_key in result_true:
                np.testing.assert_allclose(
                    result_true[inf_key], result_none[key], rtol=1e-12,
                    err_msg=f"informative_{key} should match base when mask=all-True"
                )


# ---------------------------------------------------------------------------
# S13: Activation gate check
# ---------------------------------------------------------------------------

class TestActivationGateCheck:
    """Spec S13: Activation gate threshold checks."""

    def test_gate_pass_when_all_exceed_thresholds(self, active_transitions):
        """global_gate_pass=True when all diagnostics exceed thresholds.

        Invariant guarded: gate opens when operator activation is meaningful.
        """
        # docs/specs/phase_IV_A*.md S13
        diag = compute_aggregate_diagnostics(**active_transitions)
        gate = activation_gate_check(diag)
        assert gate["global_gate_pass"] is True, (
            f"Expected gate pass with active transitions. Values: {gate['values']}"
        )

    def test_gate_fail_when_u_near_zero(self, inactive_transitions):
        """global_gate_pass=False when u is near zero.

        Invariant guarded: gate rejects when operator has negligible effect.
        """
        # docs/specs/phase_IV_A*.md S13
        diag = compute_aggregate_diagnostics(**inactive_transitions)
        gate = activation_gate_check(diag)
        assert gate["global_gate_pass"] is False, (
            f"Expected gate fail with inactive transitions. Values: {gate['values']}"
        )

    def test_gate_returns_required_keys(self, active_transitions):
        """Gate result has all required keys."""
        diag = compute_aggregate_diagnostics(**active_transitions)
        gate = activation_gate_check(diag)
        assert "global_gate_pass" in gate
        assert "preferred_gate_pass" in gate
        assert "individual_checks" in gate
        assert "values" in gate

    def test_gate_individual_checks_dict(self, active_transitions):
        """individual_checks contains all four check names."""
        diag = compute_aggregate_diagnostics(**active_transitions)
        gate = activation_gate_check(diag)
        expected_checks = {"mean_abs_u", "frac_active", "mean_abs_delta_d", "normalized_target_gap"}
        assert set(gate["individual_checks"].keys()) == expected_checks


# ---------------------------------------------------------------------------
# Edge case: empty arrays
# ---------------------------------------------------------------------------

class TestEmptyArrays:
    """Diagnostics on empty arrays should not crash."""

    def test_aggregate_diagnostics_empty(self):
        """compute_aggregate_diagnostics on empty arrays returns safe defaults.

        Invariant guarded: no crash on empty input (e.g., no transitions at a stage).
        """
        empty = np.array([], dtype=np.float64)
        result = compute_aggregate_diagnostics(empty, empty, empty, 1.0, empty)
        assert result["mean_abs_u"] == 0.0
        assert result["mean_beta_used"] == 0.0

    def test_event_conditioned_empty_mask(self):
        """Event-conditioned with all-False mask returns safe defaults."""
        n = 10
        ns = np.ones(n) * 0.1
        dd = np.ones(n) * 0.01
        tg = np.ones(n) * 0.05
        mask = np.zeros(n, dtype=bool)
        result = compute_event_conditioned_diagnostics(ns, dd, tg, 1.0, mask)
        assert result["mean_abs_u"] == 0.0


# ---------------------------------------------------------------------------
# Stage aggregate
# ---------------------------------------------------------------------------

class TestStageAggregate:
    """Per-stage aggregate diagnostics."""

    def test_stage_aggregate_returns_list_of_dicts(self):
        """compute_stage_aggregate returns a list of dicts with correct keys.

        Invariant guarded: per-stage breakdown has same schema as global aggregate.
        """
        T = 3
        rng = np.random.default_rng(11)
        ns_by_stage = [rng.normal(0, 0.1, 20) for _ in range(T)]
        dd_by_stage = [rng.normal(0, 0.01, 20) for _ in range(T)]
        tg_by_stage = [rng.normal(0, 0.05, 20) for _ in range(T)]

        result = compute_stage_aggregate(ns_by_stage, dd_by_stage, tg_by_stage, 1.0)

        assert isinstance(result, list)
        assert len(result) == T

        # All dicts should have the same keys
        ref_keys = set(result[0].keys())
        for d in result[1:]:
            assert set(d.keys()) == ref_keys

    def test_stage_aggregate_single_stage_matches_global(self):
        """Single-stage aggregate matches global aggregate (same data).

        Invariant guarded: single-stage is a degenerate case of the stage list.
        """
        rng = np.random.default_rng(22)
        ns = rng.normal(0, 0.1, 50)
        dd = rng.normal(0, 0.01, 50)
        tg = rng.normal(0, 0.05, 50)
        rb = 1.0

        stage_result = compute_stage_aggregate([ns], [dd], [tg], rb)
        global_result = compute_aggregate_diagnostics(
            ns, dd, tg, rb, np.ones_like(ns)
        )

        for key in stage_result[0]:
            np.testing.assert_allclose(
                stage_result[0][key], global_result[key], rtol=1e-12,
                err_msg=f"Mismatch on key '{key}'"
            )
