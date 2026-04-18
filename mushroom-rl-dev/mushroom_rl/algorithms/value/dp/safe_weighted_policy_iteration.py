"""
Finite-horizon safe weighted-LSE policy iteration.

Mirrors :class:`ClassicalPolicyIteration` exactly, replacing the classical
Bellman operator with the safe weighted-LSE operator from
:class:`SafeWeightedPolicyEvaluation`.

Conventions
-----------
- Finite horizon ``T``: stages ``t in 0..T-1``.
- Terminal convention: ``V[T, :] = 0``.
- Deterministic policy: ``pi[t, s] in {0, ..., A-1}`` selects a single action.
- Each PI iteration performs:

  1. **Safe policy evaluation** of the current ``pi`` via exact backward
     induction (reusing :class:`SafeWeightedPolicyEvaluation`). Produces
     ``Q^pi[t, s, a]`` and ``V^pi[t, s]`` with ``V^pi[T, :] = 0``.
  2. **Policy improvement**:
     ``pi_new[t, s] = argmax_a Q^pi[t, s, a]``
     (``np.argmax`` tie-break: lowest action index wins).
  3. **Residual**: sup-norm over non-terminal stages
     ``||V_new[0..T-1] - V_old[0..T-1]||_inf``.

Classical recovery
------------------
When every ``beta_used_t == 0`` the safe operator reduces to the classical
Bellman operator exactly (no ``logaddexp`` call on the ``beta == 0`` path).
The output is bit-identical (within floating-point tolerance) to
:class:`ClassicalPolicyIteration`.

Stopping rule
-------------
The loop terminates at iteration ``k`` as soon as **any** of the following
holds:

- the improved policy equals the policy used for evaluation
  (``policy_stable``),
- the residual falls **strictly** below ``tol`` (when ``tol > 0``),
- ``k >= max_iter``.

Diagnostics
-----------
Per-iteration clipping diagnostics (clip fraction, effective discount
statistics) are accumulated from each PE run and exposed through
:attr:`clipping_summary`.
"""
from __future__ import annotations

import pathlib
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

from mushroom_rl.algorithms.value.dp.safe_weighted_policy_evaluation import (
    SafeWeightedPolicyEvaluation,
)
from mushroom_rl.algorithms.value.dp.safe_weighted_common import (
    BetaSchedule,
    SafeWeightedCommon,
)
from mushroom_rl.algorithms.value.dp.finite_horizon_dp_utils import (
    allocate_value_tables,
    extract_mdp_arrays,
    greedy_policy,
    sup_norm_residual,
)

# Pull :class:`SweepTimer` in via an explicit ``sys.path`` insert so the
# ``experiments/`` tree (which has no package markers) is importable from
# inside the MushroomRL subpackage. Phase I spec requires per-iteration
# wall-clock logging to live in the shared ``common.timing`` module;
# duplicating it here would fork the schema.
#
# mushroom-rl-dev/mushroom_rl/algorithms/value/dp/safe_weighted_policy_iteration.py
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


__all__ = ["SafeWeightedPolicyIteration"]


class SafeWeightedPolicyIteration:
    """
    Finite-horizon safe weighted-LSE policy iteration.

    Each iteration alternates safe policy evaluation (via
    :class:`SafeWeightedPolicyEvaluation`, a single backward pass using the
    safe weighted-LSE operator) with greedy policy improvement from the
    resulting ``Q^pi``. The outer loop terminates as soon as the policy
    stabilises, the per-iteration sup-norm residual falls strictly below
    ``tol`` (when ``tol > 0``), or ``max_iter`` is reached.

    Classical recovery: when ``schedule`` has all ``beta_used_t == 0``,
    the algorithm is bit-identical to :class:`ClassicalPolicyIteration`.

    Attributes set by :meth:`run`:
        Q              : ``np.ndarray`` of shape ``(T, S, A)``, ``float64`` --
                         action-value table ``Q^pi`` for the *final* policy.
        V              : ``np.ndarray`` of shape ``(T + 1, S)``, ``float64`` --
                         value table ``V^pi`` for the *final* policy;
                         ``V[T, :] = 0`` by construction.
        pi             : ``np.ndarray`` of shape ``(T, S)``, ``int64`` --
                         final (greedy-stable) policy.
        residuals      : ``list[float]`` -- one entry per PI iteration,
                         ``||V_new[0..T-1] - V_old[0..T-1]||_inf``.
        iter_times_s   : ``list[float]`` -- per-iteration wall-clock seconds
                         (one entry per PI iteration, covering both the
                         evaluation and improvement steps).
        n_iters        : ``int`` -- number of PI iterations executed.
        wall_clock_s   : ``float`` -- total wall-clock seconds in :meth:`run`.
        policy_stable  : ``bool`` -- ``True`` iff the loop exited because the
                         improved policy matched the evaluated policy.
        schedule_report: ``dict`` -- schedule metadata, set at start of ``run``.
        clipping_summary: ``dict`` -- aggregated clipping diagnostics across
                         all PI iterations, set at end of ``run``.

    Args:
        mdp: a :class:`mushroom_rl.environments.finite_mdp.FiniteMDP` (or any
            object exposing ``p``, ``r``, and ``info.{gamma, horizon}``).
        schedule: :class:`BetaSchedule` providing per-stage beta values and
            certification quantities.
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
        v_init: optional warm-start value table of shape ``(T+1, S)``.
            Terminal row ``V[T, :]`` is always forced to zero.
    """

    def __init__(
        self,
        mdp,
        schedule: BetaSchedule,
        init_pi: Optional[np.ndarray] = None,
        max_iter: int = 100,
        tol: float = 0.0,
        v_init: Optional[np.ndarray] = None,
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

        # Schedule validation: horizon must match.
        if not isinstance(schedule, BetaSchedule):
            raise TypeError(
                f"schedule must be a BetaSchedule instance; "
                f"got {type(schedule).__name__}."
            )
        if schedule.T != self._T:
            raise ValueError(
                f"Schedule horizon ({schedule.T}) does not match MDP "
                f"horizon ({self._T})."
            )
        self._schedule: BetaSchedule = schedule

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

        # Warm-start: copy caller-provided V table, then re-enforce terminal.
        if v_init is not None:
            v_init_arr = np.asarray(v_init, dtype=np.float64)
            if v_init_arr.shape != self.V.shape:
                raise ValueError(
                    f"v_init shape {v_init_arr.shape} != V shape {self.V.shape}; "
                    "v_init must be (H+1, S) matching horizon and state space."
                )
            self.V[:] = v_init_arr
            self.V[self._T, :] = 0.0  # terminal boundary is always zero

        # Timing / logging scaffolding.
        self.residuals: List[float] = []
        self.iter_times_s: List[float] = []
        self.n_iters: int = 0
        self.wall_clock_s: float = 0.0
        self.policy_stable: bool = False
        self._has_run: bool = False
        #: After each PI iteration, a copy of ``V`` (shape ``(T+1, S)``).
        self.V_sweep_history: List[np.ndarray] = []

        # Per-iteration clipping diagnostics (accumulated from PE runs).
        self._iter_clip_fraction: List[float] = []
        self._iter_eff_discount_mean: List[float] = []
        self._iter_frac_eff_lt_gamma: List[float] = []

        # Schedule / clipping summary reports (set by run()).
        self.schedule_report: Dict[str, Any] = {}
        self.clipping_summary: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_schedule_report(self) -> Dict[str, Any]:
        """Build a JSON-serialisable schedule metadata dict."""
        return {
            "gamma": self._schedule.gamma,
            "sign": self._schedule.sign,
            "task_family": self._schedule.task_family,
            "T": self._schedule.T,
            "beta_used_t": [
                self._schedule.beta_used_at(t) for t in range(self._T)
            ],
            "beta_raw_t": [
                self._schedule.beta_raw_at(t) for t in range(self._T)
            ],
            "beta_cap_t": [
                self._schedule.beta_cap_at(t) for t in range(self._T)
            ],
        }

    def _accumulate_pe_diagnostics(self, pe: SafeWeightedPolicyEvaluation) -> None:
        """Extract and accumulate per-iteration diagnostics from a PE run."""
        cs = pe.clipping_summary

        # clip_fraction: single float for this PE run.
        self._iter_clip_fraction.append(float(cs["clip_fraction"]))

        # Per-stage effective discount stats: mean across stages.
        stage_eff_mean = cs.get("stage_eff_discount_mean", [])
        if stage_eff_mean:
            self._iter_eff_discount_mean.append(
                float(np.mean(stage_eff_mean))
            )
        else:
            self._iter_eff_discount_mean.append(float(self._gamma))

        stage_frac_lt = cs.get("stage_frac_eff_lt_gamma", [])
        if stage_frac_lt:
            self._iter_frac_eff_lt_gamma.append(
                float(np.mean(stage_frac_lt))
            )
        else:
            self._iter_frac_eff_lt_gamma.append(0.0)

    def _build_clipping_summary(self) -> Dict[str, Any]:
        """Build the aggregated clipping summary across all PI iterations."""
        return {
            "n_iters": int(self.n_iters),
            "iter_clip_fraction": [
                float(x) for x in self._iter_clip_fraction
            ],
            "iter_eff_discount_mean": [
                float(x) for x in self._iter_eff_discount_mean
            ],
            "iter_frac_eff_lt_gamma": [
                float(x) for x in self._iter_frac_eff_lt_gamma
            ],
            "overall_clip_fraction": (
                float(np.mean(self._iter_clip_fraction))
                if self._iter_clip_fraction
                else 0.0
            ),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> "SafeWeightedPolicyIteration":
        """
        Execute finite-horizon safe weighted-LSE policy iteration.

        Runs up to ``max_iter`` iterations of (safe policy evaluation,
        greedy policy improvement). After each iteration, the sup-norm
        residual of the non-terminal value rows ``V[0..T-1]`` against the
        previous iteration is appended to :attr:`residuals`, and the
        wall-clock time is appended to :attr:`iter_times_s`. Safe to call
        multiple times; each call re-initialises the tables.

        Returns:
            ``self`` -- for fluent chaining
            (``pi_alg = SafeWeightedPolicyIteration(mdp, schedule).run()``).
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
        self._iter_clip_fraction = []
        self._iter_eff_discount_mean = []
        self._iter_frac_eff_lt_gamma = []

        # Schedule report: set at start of run().
        self.schedule_report = self._build_schedule_report()

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
                # --- 1. Safe policy evaluation --------------------------------
                # Exact backward induction using the safe weighted-LSE
                # operator. Produces Q^pi (full (T, S, A) table) and V^pi.
                pe = SafeWeightedPolicyEvaluation(
                    self._mdp, pi=self.pi, schedule=self._schedule
                ).run()
                Q_pi: np.ndarray = pe.Q                      # (T, S, A)
                V_pi: np.ndarray = pe.V                      # (T + 1, S)

                # --- 2. Policy improvement ------------------------------------
                # Greedy w.r.t. Q^pi, stage-by-stage. ``np.argmax``
                # tie-break: lowest action index wins -- matches VI.
                pi_new = np.empty((self._T, self._S), dtype=np.int64)
                for t in range(self._T):
                    pi_new[t] = greedy_policy(Q_pi[t])       # (S,)

            # --- 3. Accumulate PE diagnostics ---------------------------------
            self._accumulate_pe_diagnostics(pe)

            # --- 4. Residual + bookkeeping ------------------------------------
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

            # --- 5. Advance: adopt the improved policy for the next PE --------
            self.pi = pi_new                                 # (T, S)
            V_prev_non_terminal = V_pi[: self._T].copy()     # (T, S)

            # Residual-based early stop: only kicks in when ``tol > 0``.
            # Policy stability is checked first because a stable policy is
            # the canonical PI termination condition.
            if self._tol > 0.0 and residual < self._tol:
                break

        self.wall_clock_s = float(time.perf_counter() - t_start)
        self.iter_times_s = list(iter_timer.sweep_times_s)

        # Clipping summary: set at end of run().
        self.clipping_summary = self._build_clipping_summary()

        self._has_run = True
        return self

    def results(self) -> Dict[str, Any]:
        """
        Return a JSON-friendly summary of the run.

        Extends the classical PI results with ``schedule_report`` and
        ``clipping_summary``.

        Returns:
            Dict with keys:
                - ``Q_shape``          : ``list[int]`` -- shape of ``self.Q``.
                - ``V_shape``          : ``list[int]`` -- shape of ``self.V``.
                - ``n_iters``          : ``int`` -- number of PI iterations.
                - ``wall_clock_s``     : ``float`` -- total seconds in ``run``.
                - ``residuals``        : ``list[float]`` -- one per iteration.
                - ``iter_times_s``     : ``list[float]`` -- per-iteration seconds.
                - ``policy_stable``    : ``bool`` -- loop exited on stability.
                - ``V0_sup_norm``      : ``float`` -- ``sup_norm(V[0], 0)``.
                - ``V0_mean``          : ``float`` -- mean of ``V[0, :]``.
                - ``V0_max``           : ``float`` -- max of ``V[0, :]``.
                - ``schedule_report``  : ``dict`` -- schedule metadata.
                - ``clipping_summary`` : ``dict`` -- aggregated clipping stats.

        Raises:
            RuntimeError: if called before :meth:`run`.
        """
        if not self._has_run:
            raise RuntimeError(
                "SafeWeightedPolicyIteration.results() called before run(); "
                "call run() first."
            )
        V0 = self.V[0]                                       # (S,)
        base: Dict[str, Any] = {
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
        base.update({
            "schedule_report": self.schedule_report,
            "clipping_summary": self.clipping_summary,
        })
        return base
