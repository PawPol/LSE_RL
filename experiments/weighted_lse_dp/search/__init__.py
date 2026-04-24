"""Phase V search infrastructure (WP1a).

Public entry points
-------------------
* :class:`ContestState`, :class:`FamilySpec` -- family protocol (see
  ``family_spec.py``).
* :func:`lambda_tie` -- exact-DP bisection solver for the classical tie
  (see ``tie_solver.py``).
* :func:`compute_d_ref`, :func:`save_occupancy` -- reference-occupancy
  helpers (see ``reference_occupancy.py``).
* :func:`evaluate_candidate` -- per-candidate metric computation
  (see ``candidate_metrics.py``).

Spec reference: ``docs/specs/phase_V_mechanism_experiments.md`` sections
5, 6, 13.
"""

from .family_spec import ContestState, FamilySpec
from .tie_solver import TieNotBracketed, lambda_tie
from .reference_occupancy import compute_d_ref, save_occupancy
from .candidate_metrics import evaluate_candidate

__all__ = [
    "ContestState",
    "FamilySpec",
    "TieNotBracketed",
    "lambda_tie",
    "compute_d_ref",
    "save_occupancy",
    "evaluate_candidate",
]
