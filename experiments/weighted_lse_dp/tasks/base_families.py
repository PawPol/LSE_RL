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

Phase V addition (WP1a, 2026-04-23)
-----------------------------------
The Phase V search protocol expects task factories to expose a
``contest_state`` attribute (see
``experiments.weighted_lse_dp.search.family_spec.ContestState``).  Phase
I--IV factories predate this protocol and default to ``None``; WP1c's
Phase V search driver treats ``None`` as "not a Phase V family" and
refuses to enroll the task into the search sweep.  Existing Phase I--IV
code paths are unaffected.
"""

from __future__ import annotations

from experiments.weighted_lse_dp.common.task_factories import (
    make_chain_base,
    make_grid_base,
    make_taxi_base,
)
from experiments.weighted_lse_dp.search.family_spec import ContestState

# Sentinel: Phase I--IV factories do NOT designate a Phase V contest state.
# WP2 Family A/B/C factories set this explicitly; WP1c's search driver
# requires it to be non-None before enrolling a task in the sweep.
contest_state: ContestState | None = None

__all__ = [
    "ContestState",
    "contest_state",
    "make_chain_base",
    "make_grid_base",
    "make_taxi_base",
]
