"""Stress-test task factories for Phase II of the weighted-LSE DP paper suite.

Each factory reduces to its Phase I base task when the severity parameter is
zero.  All factories follow the same signature convention as the Phase I
factories in ``experiments.weighted_lse_dp.common.task_factories``.

Implemented here:
  - ``make_chain_sparse_long`` (spec S5.1.A)
  - ``make_chain_jackpot``     (spec S5.1.B)
  - ``make_chain_catastrophe`` (spec S5.1.C)
  - ``make_grid_sparse_goal``  (spec S5.2.A)
  - ``make_taxi_bonus_shock``  (spec S5.3.A)

Wrappers for nonstationary/hazard tasks live in the sibling modules.
"""

from __future__ import annotations

# Factories implemented by env-builder (tasks 3, 4, 5, 7, 10).

__all__: list[str] = [
    "make_chain_sparse_long",
    "make_chain_jackpot",
    "make_chain_catastrophe",
    "make_grid_sparse_goal",
    "make_taxi_bonus_shock",
]
