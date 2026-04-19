"""Geometry package for Phase IV natural-shift analysis and activation metrics.

Submodules
----------
natural_shift          Phase IV-A S2 natural-shift coordinate computations.
activation_metrics     Phase IV-A S8 activation diagnostic metrics.
trust_region           Phase IV-A S6 trust-region cap computations.
adaptive_headroom      Phase IV-A S6 adaptive headroom computations.
"""
from __future__ import annotations

from .natural_shift import (
    compute_natural_shift,
    compute_normalized_coordinates,
    compute_theta,
    compute_aligned_margin,
    small_signal_discount_gap,
    small_signal_target_gap,
)
from .activation_metrics import (
    compute_aggregate_diagnostics,
    activation_gate_check,
    compute_event_conditioned_diagnostics,
    compute_stage_aggregate,
)
from .trust_region import (
    kl_bernoulli,
    compute_eps_design,
    compute_stagewise_confidence,
    solve_u_tr_cap,
    compute_trust_region_cap,
)
from .adaptive_headroom import (
    compute_informativeness,
    compute_alpha_base,
    compute_kappa,
    compute_bhat_backward,
    compute_a_t,
    compute_theta_safe,
    compute_u_safe_ref,
    run_fixed_point,
)

__all__ = [
    "compute_natural_shift",
    "compute_normalized_coordinates",
    "compute_theta",
    "compute_aligned_margin",
    "small_signal_discount_gap",
    "small_signal_target_gap",
    "compute_aggregate_diagnostics",
    "activation_gate_check",
    "compute_event_conditioned_diagnostics",
    "compute_stage_aggregate",
    "kl_bernoulli",
    "compute_eps_design",
    "compute_stagewise_confidence",
    "solve_u_tr_cap",
    "compute_trust_region_cap",
    "compute_informativeness",
    "compute_alpha_base",
    "compute_kappa",
    "compute_bhat_backward",
    "compute_a_t",
    "compute_theta_safe",
    "compute_u_safe_ref",
    "run_fixed_point",
]
