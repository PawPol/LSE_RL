"""Re-exports of Phase I base task factories for paired base/stress comparison.

All three factories are defined in
``experiments.weighted_lse_dp.common.task_factories``.  They are re-exported
here so Phase II code can import from one canonical location:

    from experiments.weighted_lse_dp.tasks.base_families import (
        make_chain_base,
        make_grid_base,
        make_taxi_base,
    )

Each factory signature: ``factory(cfg: dict) -> tuple[MDP, MDP, dict]``
where the first MDP is the base (non-time-augmented) and the second is the
time-augmented RL version.
"""

from __future__ import annotations

from experiments.weighted_lse_dp.common.task_factories import (
    make_chain_base,
    make_grid_base,
    make_taxi_base,
)

__all__ = [
    "make_chain_base",
    "make_grid_base",
    "make_taxi_base",
]
