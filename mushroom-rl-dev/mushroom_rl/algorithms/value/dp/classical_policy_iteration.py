"""
Finite-horizon classical (optimal) policy iteration.

Conventions
-----------
- Finite horizon ``T``: stages ``t in 0..T-1``.
- Terminal convention: ``V[T, :] = 0``.
- Deterministic policy: ``pi[t, s] in {0, ..., A-1}`` selects a single action.
- Each PI iteration performs:

  1. **Policy evaluation** of the current ``pi`` via exact backward induction
     (reusing :class:`ClassicalPolicyEvaluation`). Produces
     ``Q^pi[t, s, a]`` and ``V^pi[t, s]`` with ``V^pi[T, :] = 0``.
  2. **Policy improvement**:
     ``pi_new[t, s] = argmax_a Q^pi[t, s, a]``
     (``np.argmax`` tie-break: lowest action index wins).
  3. **Residual**: sup-norm over non-terminal stages
     ``||V_new[0..T-1] - V_old[0..T-1]||_inf``, where ``V_old`` is the value
     table from the previous PI iteration (or the zero-initialised table on
     the first iteration).

Stopping rule
-------------
The loop terminates at iteration ``k`` as soon as **any** of the following
holds:

- the improved policy equals the policy used for evaluation
  (``policy_stable``),
- the residual falls **strictly** below ``tol`` (when ``tol > 0``),
- ``k >= max_iter``.

Finite-horizon PI converges in at most ``|A|^(T * S)`` iterations in theory,
but the tight empirical bound on our small MDPs is typically 1-3 iterations.

Residual convention
-------------------
The ``k``-th entry of :attr:`residuals` is the sup-norm residual between the
value table produced by PI iteration ``k`` (the freshly-evaluated
``V^pi_new``) and the table from iteration ``k - 1``. For ``k = 1`` the
previous table is the zero-initialised ``V``, so the first entry equals
``sup_norm(V^pi_1[0..T-1], 0)`` — the magnitude of value propagated by the
first full policy evaluation. This matches the
:class:`ClassicalValueIteration` convention so per-iteration curves are
directly comparable across planners.
"""
from __future__ import annotations

import pathlib
import sys
import time
from typing import Dict, List, Optional

import numpy as np

from mushroom_rl.algorithms.value.dp.classical_policy_evaluation import (
    ClassicalPolicyEvaluation,
)
from mushroom_rl.algorithms.value.dp.finite_horizon_dp_utils import (
    allocate_value_tables,
    extract_mdp_arrays,
    greedy_policy,
    sup_norm_residual,
)

# Pull :class:`SweepTimer` in via an explicit ``sys.path`` insert so the
# ``experiments/`` tree (which has no package markers) is importable from
# inside the MushroomRL subpackage. Phase I spec §11.2 requires per-iteration
# wall-clock logging to live in the shared ``common.timing`` module;
# duplicating it here would fork the schema.
#
# mushroom-rl-dev/mushroom_rl/algorithms/value/dp/classical_policy_iteration.py
#   parents[0] -> dp/
#   parents[1] -> value/
#   parents[2] -> algorithms/
#   parents[3] -> mushroom_rl/
#   parents[4] -> mushroom-rl-dev/
#   parents[5] -> <repo-root>
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.weighted_lse_dp.common.timing import SweepTimer  # noqa: E402


__all__ = ["ClassicalPolicyIteration"]


class ClassicalPolicyIteration:
    """
    Finite-horizon exact policy iteration.

    Each iteration alternates exact policy evaluation (via
    :class:`ClassicalPolicyEvaluation`, a single backward pass) with greedy
    policy improvement from the resulting ``Q^pi``. The outer loop
    terminates as soon as the policy stabilises, the per-iteration sup-norm
    residual falls strictly below ``tol`` (when ``tol > 0``), or
    ``max_iter`` is reached.

    Convergence: finite-horizon PI converges in at most ``|A|^(T * S)``
    iterations in theory, but on small MDPs it typically reaches the optimal
    policy in 1-3 iterations.

    Attributes set by :meth:`run`:
        Q              : ``np.ndarray`` of shape ``(T, S, A)``, ``float64`` —
                         action-value table ``Q^pi`` for the *final* policy.
        V              : ``np.ndarray`` of shape ``(T + 1, S)``, ``float64`` —
                         value table ``V^pi`` for the *final* policy;
                         ``V[T, :] = 0`` by construction.
        pi             : ``np.ndarray`` of shape ``(T, S)``, ``int64`` —
                         final (greedy-stable) policy.
        residuals      : ``list[float]`` — one entry per PI iteration,
                         ``||V_new[0..T-1] - V_old[0..T-1]||_inf``.
        iter_times_s   : ``list[float]`` — per-iteration wall-clock seconds
                         (one entry per PI iteration, covering both the
                         evaluation and improvement steps).
        n_iters        : ``int`` — number of PI iterations executed.
        wall_clock_s   : ``float`` — total wall-clock seconds in :meth:`run`.
        policy_stable  : ``bool`` — ``True`` iff the loop exited because the
                         improved policy matched the evaluated policy
                         (i.e. an exact fixed point was found within the
                         ``max_iter`` budget).

    Args:
        mdp: a :class:`mushroom_rl.environments.finite_mdp.FiniteMDP` (or any
            object exposing ``p``, ``r``, and ``info.{gamma, horizon}``).
        init_pi: optional initial policy of shape ``(T, S)`` with integer
            entries in ``[0, A)``. If ``None``, the all-zeros policy
            ``pi[t, s] = 0`` is used. A defensive copy is taken.
        max_iter: positive integer maximum number of PI iterations.
            Defaults to ``100``.
        tol: non-negative early-stopping tolerance on the sup-norm residual
            between successive value tables. ``0.0`` (the default) disables
            residual-based early stopping, leaving the policy-stability and
            ``max_iter`` guards in charge. Strictly positive values trigger
            early termination as soon as an iteration's residual falls
            strictly below ``tol``.
    """

    def __init__(
        self,
        mdp,
        init_pi: Optional[np.ndarray] = None,
        max_iter: int = 100,
        tol: float = 0.0,
    ) -> None:
        if not isinstance(max_iter, (int, np.integer)) or int(max_iter) <= 0:
            raise ValueError(
                f"max_iter must be a positive integer; got {max_iter!r}."
            )
        if not isinstance(tol, (int, float, np.floating, np.integer)):
            raise ValueError(
                f"tol must be a real scalar; got type={type(tol).__name__}."
            )
        if float(tol) < 0.0:
            raise ValueError(f"tol must be non-negative; got {tol}.")

        # Extract + validate raw MDP tensors.
        p, r, horizon, gamma = extract_mdp_arrays(mdp)   # (S, A, S), (S, A, S)
        self._mdp = mdp                                   # held for re-use by PE
        self._p: np.ndarray = p                           # (S, A, S)
        self._r: np.ndarray = r                           # (S, A, S)
        self._gamma: float = float(gamma)
        self._T: int = int(horizon)
        self._S: int = int(p.shape[0])
        self._A: int = int(p.shape[1])

        self._max_iter: int = int(max_iter)
        self._tol: float = float(tol)

        # Initial policy: defensive copy, shape/range validated.
        if init_pi is None:
            init_pi_arr = np.zeros((self._T, self._S), dtype=np.int64)
        else:
            init_pi_arr = np.asarray(init_pi, dtype=np.int64)
            expected_shape = (self._T, self._S)
            if init_pi_arr.shape != expected_shape:
                raise ValueError(
                    f"init_pi must have shape (T, S) = {expected_shape}; "
                    f"got {init_pi_arr.shape}."
                )
            if np.any(init_pi_arr < 0) or np.any(init_pi_arr >= self._A):
                raise ValueError(
                    f"init_pi entries must lie in [0, {self._A}); "
                    f"got min={int(init_pi_arr.min())}, "
                    f"max={int(init_pi_arr.max())}."
                )
            init_pi_arr = init_pi_arr.copy()
        self._init_pi: np.ndarray = init_pi_arr            # (T, S)

        # Result tables, zero-allocated. Terminal V[T, :] = 0 by construction.
        Q0, V0, pi0 = allocate_value_tables(self._S, self._A, self._T)
        self.Q: np.ndarray = Q0                             # (T, S, A)
        self.V: np.ndarray = V0                             # (T + 1, S)
        self.pi: np.ndarray = pi0                           # (T, S)

        # Timing / logging scaffolding.
        self.residuals: List[float] = []
        self.iter_times_s: List[float] = []
        self.n_iters: int = 0
        self.wall_clock_s: float = 0.0
        self.policy_stable: bool = False
        self._has_run: bool = False
        #: After each PI iteration, a copy of ``V`` (shape ``(T+1, S)``).
        self.V_sweep_history: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> "ClassicalPolicyIteration":
        """
        Execute finite-horizon policy iteration.

        Runs up to ``max_iter`` iterations of (exact policy evaluation,
        greedy policy improvement). After each iteration, the sup-norm
        residual of the non-terminal value rows ``V[0..T-1]`` against the
        previous iteration is appended to :attr:`residuals`, and the
        wall-clock time is appended to :attr:`iter_times_s`. Safe to call
        multiple times; each call re-initialises the tables.

        Returns:
            ``self`` — for fluent chaining (``pi_alg = CPI(mdp).run()``).
        """
        # Reset tables so ``run()`` is idempotent.
        self.Q.fill(0.0)
        self.V.fill(0.0)
        self.pi = self._init_pi.copy()                     # (T, S)
        self.residuals = []
        self.iter_times_s = []
        self.n_iters = 0
        self.wall_clock_s = 0.0
        self.policy_stable = False
        self.V_sweep_history = []

        iter_timer = SweepTimer()
        t_start = time.perf_counter()

        # ``V_prev`` is used only for residual computation. The first
        # iteration compares against the zero-initialised table, matching
        # :class:`ClassicalValueIteration`'s convention.
        V_prev_non_terminal = np.zeros(
            (self._T, self._S), dtype=np.float64
        )                                                   # (T, S)

        for _ in range(self._max_iter):
            with iter_timer.sweep():
                # --- 1. Policy evaluation ---------------------------------
                # Exact backward induction: one pass, no inner loop. This
                # populates Q^pi (full (T, S, A) table) and V^pi.
                pe = ClassicalPolicyEvaluation(self._mdp, pi=self.pi).run()
                Q_pi: np.ndarray = pe.Q                      # (T, S, A)
                V_pi: np.ndarray = pe.V                      # (T + 1, S)

                # --- 2. Policy improvement --------------------------------
                # Greedy w.r.t. Q^pi, stage-by-stage. ``np.argmax``
                # tie-break: lowest action index wins — matches VI.
                pi_new = np.empty((self._T, self._S), dtype=np.int64)
                for t in range(self._T):
                    pi_new[t] = greedy_policy(Q_pi[t])       # (S,)

            # --- 3. Residual + bookkeeping --------------------------------
            residual = sup_norm_residual(
                V_pi[: self._T], V_prev_non_terminal
            )
            self.residuals.append(float(residual))
            self.n_iters += 1

            # Commit the freshly-evaluated tables as the current result
            # tables. If the improvement does not change the policy, we
            # keep these; otherwise they get replaced on the next pass.
            self.Q[...] = Q_pi
            self.V[...] = V_pi
            self.V_sweep_history.append(self.V.copy())

            # Stability check: has the improved policy equalled the policy
            # we just evaluated? If so, ``Q^pi`` above is already optimal
            # on the support of the evaluated policy, and the Bellman
            # optimality equations hold at (Q^pi, V^pi).
            policy_unchanged = bool(np.array_equal(pi_new, self.pi))
            if policy_unchanged:
                self.policy_stable = True
                break

            # --- 4. Advance: adopt the improved policy for the next PE ----
            self.pi = pi_new                                 # (T, S)
            V_prev_non_terminal = V_pi[: self._T].copy()     # (T, S)

            # Residual-based early stop: only kicks in when ``tol > 0``.
            # Policy stability is checked first because a stable policy is
            # the canonical PI termination condition.
            if self._tol > 0.0 and residual < self._tol:
                break

        self.wall_clock_s = float(time.perf_counter() - t_start)
        self.iter_times_s = list(iter_timer.sweep_times_s)
        self._has_run = True
        return self

    def results(self) -> Dict:
        """
        Return a JSON-friendly summary of the run.

        Returns:
            Dict with keys:
                - ``Q_shape``      : ``list[int]`` — shape of ``self.Q``.
                - ``V_shape``      : ``list[int]`` — shape of ``self.V``.
                - ``n_iters``      : ``int`` — number of PI iterations.
                - ``wall_clock_s`` : ``float`` — total seconds in ``run``.
                - ``residuals``    : ``list[float]`` — one per iteration.
                - ``iter_times_s`` : ``list[float]`` — per-iteration seconds.
                - ``policy_stable``: ``bool`` — loop exited on stability.
                - ``V0_sup_norm``  : ``float`` — ``sup_norm(V[0], 0)``.
                - ``V0_mean``      : ``float`` — mean of ``V[0, :]``.
                - ``V0_max``       : ``float`` — max of ``V[0, :]``.

        Raises:
            RuntimeError: if called before :meth:`run`.
        """
        if not self._has_run:
            raise RuntimeError(
                "ClassicalPolicyIteration.results() called before run(); "
                "call run() first."
            )
        V0 = self.V[0]                                       # (S,)
        return {
            "Q_shape": list(self.Q.shape),
            "V_shape": list(self.V.shape),
            "n_iters": int(self.n_iters),
            "wall_clock_s": float(self.wall_clock_s),
            "residuals": [float(x) for x in self.residuals],
            "iter_times_s": [float(x) for x in self.iter_times_s],
            "policy_stable": bool(self.policy_stable),
            "V0_sup_norm": float(np.max(np.abs(V0))),
            "V0_mean": float(V0.mean()),
            "V0_max": float(V0.max()),
        }
