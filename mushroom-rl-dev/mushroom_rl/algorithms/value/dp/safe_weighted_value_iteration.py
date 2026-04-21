"""
Finite-horizon safe weighted-LSE value iteration via backward induction.

Conventions
-----------
- Finite horizon ``T``: stages ``t in 0..T-1``.
- Terminal convention: ``V[T, :] = 0``.
- Safe weighted-LSE Bellman backup evaluated here (per stage ``t``):

  1. Compute ``E_v_next[s, a] = sum_{s'} P[s, a, s'] * V[t+1, s']`` via
     ``np.einsum``.
  2. Apply the safe weighted-LSE target via
     :meth:`SafeWeightedCommon.compute_safe_target_batch`:

     ``Q_safe[t, s, a] = g_t^safe(R_bar[s, a], E_v_next[s, a])``

  3. ``V[t, s] = max_a Q_safe[t, s, a]``,
     ``pi[t, s] = argmax_a Q_safe[t, s, a]`` (``np.argmax`` tie-break:
     lowest index).

When ``beta_used == 0`` at every stage, the safe target collapses to the
exact classical Bellman backup ``r + gamma * v`` and the algorithm is
bit-identical to :class:`ClassicalValueIteration` (within numerical
tolerance).

Single-pass vs multi-pass
-------------------------
The finite-horizon Bellman operator on the time-unrolled DAG is exact after
*one* full backward sweep: ``V[T] = 0`` is known, and each earlier stage is a
pure function of the stage immediately above. ``n_sweeps=1`` (the default)
therefore runs a single clean backward pass.

``n_sweeps=N > 1`` repeats the full backward pass ``N`` times, tracking the
sup-norm residual ``||V_new[0..T-1] - V_old[0..T-1]||_inf`` between successive
passes. By construction the residual is ``0`` for every pass after the first,
but exercising this path keeps the interface aligned with iterative planners
(policy iteration, modified policy iteration, async VI) and makes the
per-sweep timing plots comparable across planners.

Residual convention
-------------------
For sweep ``k`` we log the sup-norm residual against the value table produced
by sweep ``k - 1`` over stages ``0..T - 1`` (``V[T, :] = 0`` is terminal and
never changes, so excluding it avoids a trivial zero contribution). For
``k = 1`` the previous table is the zero-initialised ``V``, so the first
residual equals ``sup_norm(V[0..T-1], 0)`` -- the magnitude of value
propagated by the first exact pass.
"""
from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

from mushroom_rl.algorithms.value.dp.finite_horizon_dp_utils import (
    allocate_value_tables,
    bellman_v_from_q,
    expected_reward,
    extract_mdp_arrays,
    greedy_policy,
    sup_norm_residual,
)
from mushroom_rl.algorithms.value.dp.safe_weighted_common import (
    BetaSchedule,
    SafeWeightedCommon,
)

# The ``experiments/`` tree has no package markers, so pull
# :class:`SweepTimer` in via an explicit ``sys.path`` insert. Phase-I spec
# SS11.2 requires per-sweep wall-clock logging to live in the shared
# ``common.timing`` module; duplicating it here would fork the schema.
import pathlib
import sys

# mushroom-rl-dev/mushroom_rl/algorithms/value/dp/safe_weighted_value_iteration.py
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


__all__ = ["SafeWeightedValueIteration"]


class SafeWeightedValueIteration:
    """
    Finite-horizon safe weighted-LSE value iteration via backward induction.

    Replaces the classical Bellman backup with the safe weighted-LSE target
    from :class:`SafeWeightedCommon`. When ``beta_used == 0`` at every stage,
    this algorithm is bit-identical to :class:`ClassicalValueIteration`.

    In exact mode (``n_sweeps=1``): one backward pass over stages
    ``T - 1, ..., 0`` computes the safe target
    ``Q[t] = g_t^safe(R_bar, E[V[t+1] | s, a])`` and
    ``V[t] = max_a Q[t]``. This is exact for finite-horizon problems -- no
    iteration is required.

    In multi-pass mode (``n_sweeps=N > 1``): repeats the full backward pass
    ``N`` times, tracking the sup-norm residual
    ``||V_new[0..T-1] - V_old[0..T-1]||_inf`` per pass. Useful for diagnosing
    convergence and for matching the iterative-planner interface (so plots
    and logs are uniform). If ``tol > 0`` and a residual falls strictly
    below ``tol``, the loop breaks early and ``converged`` is reported
    ``True``. ``tol == 0`` (the default) disables early stopping -- the
    full ``n_sweeps`` budget is always consumed, which keeps the per-sweep
    timing curves length-stable for convergence plots.

    Attributes set by :meth:`run`:
        Q            : ``np.ndarray`` of shape ``(T, S, A)``, ``float64`` --
                       safe action-value table.
        V            : ``np.ndarray`` of shape ``(T + 1, S)``, ``float64`` --
                       safe value table; ``V[T, :] = 0`` by construction.
        pi           : ``np.ndarray`` of shape ``(T, S)``, ``int64`` --
                       greedy policy ``argmax_a Q[t, s, a]``.
        residuals    : ``list[float]`` -- one entry per completed sweep.
        sweep_times_s: ``list[float]`` -- per-sweep wall-clock seconds.
        n_sweeps     : ``int`` -- number of sweeps actually executed.
        wall_clock_s : ``float`` -- total wall-clock seconds inside ``run``.
        converged    : ``bool`` -- multi-pass only: last residual ``< tol``.
        schedule_report : ``dict`` -- summary of the beta schedule (set by
                          :meth:`run` at start).
        clipping_summary: ``dict`` -- per-stage clipping diagnostics (set by
                          :meth:`run` at end).

    Args:
        mdp: a :class:`mushroom_rl.environments.finite_mdp.FiniteMDP` (or any
            object exposing ``p``, ``r``, and ``info.{gamma, horizon}``).
        schedule: a :class:`BetaSchedule` providing per-stage beta values.
        n_sweeps: positive integer number of full backward passes to run.
            ``1`` (the default) is the exact single-pass mode.
        tol: non-negative early-stopping tolerance on the sup-norm residual.
            ``0.0`` (the default) disables early stopping so the full
            ``n_sweeps`` budget is always consumed. Strictly positive
            values trigger early termination as soon as a sweep's residual
            falls below ``tol``. Ignored when ``n_sweeps == 1``.
        v_init: optional initial value table of shape ``(T+1, S)``. If
            provided, ``V`` is warm-started from this array (terminal row
            is always reset to zero).
    """

    def __init__(
        self,
        mdp,
        schedule: BetaSchedule,
        n_sweeps: int = 1,
        tol: float = 0.0,
        v_init: "np.ndarray | None" = None,
    ) -> None:
        if not isinstance(n_sweeps, (int, np.integer)) or int(n_sweeps) <= 0:
            raise ValueError(
                f"n_sweeps must be a positive integer; got {n_sweeps!r}."
            )
        if not isinstance(tol, (int, float, np.floating, np.integer)):
            raise ValueError(
                f"tol must be a real scalar; got type={type(tol).__name__}."
            )
        if float(tol) < 0.0:
            raise ValueError(f"tol must be non-negative; got {tol}.")

        # Extract + validate raw MDP tensors.
        p, r, horizon, gamma = extract_mdp_arrays(mdp)  # shapes: (S,A,S), (S,A,S)
        self._p: np.ndarray = p                          # (S, A, S)
        self._r: np.ndarray = r                          # (S, A, S)
        self._gamma: float = float(gamma)
        self._T: int = int(horizon)
        self._S: int = int(p.shape[0])
        self._A: int = int(p.shape[1])

        # Precompute the (S, A) expected-reward matrix once.
        self._r_bar: np.ndarray = expected_reward(p, r)  # (S, A)

        self._n_sweeps_request: int = int(n_sweeps)
        self._tol: float = float(tol)

        # Safe weighted-LSE components.
        self._schedule = schedule
        if schedule.T != self._T:
            raise ValueError(
                f"{type(self).__name__}: schedule.T={schedule.T} does not "
                f"match MDP horizon={self._T}. The schedule must be "
                "calibrated for this exact horizon."
            )
        self._safe = SafeWeightedCommon(
            schedule, gamma=self._gamma, n_base=self._S
        )

        # Result tables, zero-allocated. Terminal V[T, :] = 0 by construction.
        Q0, V0, pi0 = allocate_value_tables(self._S, self._A, self._T)
        self.Q: np.ndarray = Q0                          # (T, S, A)
        self.V: np.ndarray = V0                          # (T + 1, S)
        self.pi: np.ndarray = pi0                        # (T, S)

        # Warm-start: copy caller-provided V table, then re-enforce terminal.
        # Store v_init for use in run() warm-start.
        self._v_init: np.ndarray | None = None
        if v_init is not None:
            v_init_arr = np.asarray(v_init, dtype=np.float64)
            if v_init_arr.shape != self.V.shape:
                raise ValueError(
                    f"v_init shape {v_init_arr.shape} != V shape {self.V.shape}; "
                    "v_init must be (H+1, S) matching horizon and state space."
                )
            self.V[:] = v_init_arr
            self.V[self._T, :] = 0.0  # terminal boundary is always zero
            self._v_init = self.V.copy()  # store the validated copy

        # Timing / logging scaffolding.
        self.residuals: List[float] = []
        self.sweep_times_s: List[float] = []
        self.n_sweeps: int = 0
        self.wall_clock_s: float = 0.0
        self.converged: bool = False
        self._has_run: bool = False
        #: After each backward sweep, a copy of ``V`` (shape ``(T+1, S)``).
        self.V_sweep_history: List[np.ndarray] = []

        # Safe diagnostics (populated per-stage during backward pass).
        self._stage_clip_active: List[bool] = []
        self._stage_beta_raw: List[float] = []
        self._stage_beta_cap: List[float] = []
        self._stage_beta_used: List[float] = []
        self._stage_eff_discount_mean: List[float] = []
        self._stage_eff_discount_std: List[float] = []
        self._stage_frac_eff_lt_gamma: List[float] = []

        # Safe result summaries (populated by run()).
        self.schedule_report: dict = {}
        self.clipping_summary: dict = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_stage_diagnostics(self) -> None:
        """Clear per-stage diagnostic accumulators."""
        self._stage_clip_active = []
        self._stage_beta_raw = []
        self._stage_beta_cap = []
        self._stage_beta_used = []
        self._stage_eff_discount_mean = []
        self._stage_eff_discount_std = []
        self._stage_frac_eff_lt_gamma = []

    def _collect_stage_diagnostics(self) -> None:
        """Read last_* fields from SafeWeightedCommon after a batch backup."""
        self._stage_clip_active.append(self._safe.last_clip_active)
        self._stage_beta_raw.append(self._safe.last_beta_raw)
        self._stage_beta_cap.append(self._safe.last_beta_cap)
        self._stage_beta_used.append(self._safe.last_beta_used)

        eff_d = self._safe.last_effective_discount
        if isinstance(eff_d, np.ndarray):
            self._stage_eff_discount_mean.append(float(eff_d.mean()))
            self._stage_eff_discount_std.append(float(eff_d.std()))
            self._stage_frac_eff_lt_gamma.append(
                float((eff_d < self._gamma).mean())
            )
        else:
            self._stage_eff_discount_mean.append(float(eff_d))
            self._stage_eff_discount_std.append(0.0)
            self._stage_frac_eff_lt_gamma.append(
                float(eff_d < self._gamma)
            )

    def _backward_pass(self) -> None:
        """Execute one full backward pass over stages ``T - 1, ..., 0``.

        Updates ``self.Q``, ``self.V[0..T-1]``, and ``self.pi`` in place.
        ``self.V[T, :]`` is the terminal zero row and is left untouched.

        Uses the safe weighted-LSE backup instead of the classical Bellman
        backup. Per-stage diagnostics are collected into the ``_stage_*``
        lists (which must be reset before calling this method).
        """
        for t in range(self._T - 1, -1, -1):
            # Safe weighted-LSE target: E_{s'}[g_t^safe(r_bar, V[t+1,s'])].
            # Uses compute_safe_target_ev_batch to correctly apply the
            # nonlinear target *before* averaging over next states.
            Q_t = self._safe.compute_safe_target_ev_batch(
                self._r_bar, self.V[t + 1], self._p, t
            )  # (S, A)

            self.Q[t] = Q_t
            # V[t, s] = max_a Q[t, s, a];  pi[t, s] = argmax_a Q[t, s, a].
            self.V[t] = bellman_v_from_q(Q_t)           # (S,)
            self.pi[t] = greedy_policy(Q_t)             # (S,)

            # Step 3: collect per-stage diagnostics.
            self._collect_stage_diagnostics()

    def _build_schedule_report(self) -> dict:
        """Build the schedule summary dict at run start."""
        return {
            "task_family": self._schedule.task_family,
            "gamma": self._schedule.gamma,
            "sign": self._schedule.sign,
            "T": self._schedule.T,
            "beta_used_range": [
                float(min(self._schedule._beta_used_t)),
                float(max(self._schedule._beta_used_t)),
            ],
        }

    def _build_clipping_summary(self) -> dict:
        """Build the clipping summary dict at run end."""
        n_clipped = int(sum(self._stage_clip_active))
        return {
            "n_stages_clipped": n_clipped,
            "clip_fraction": float(n_clipped) / self._T,
            "stage_beta_used": [float(x) for x in self._stage_beta_used],
            "stage_beta_cap": [float(x) for x in self._stage_beta_cap],
            "stage_clip_active": list(self._stage_clip_active),
            "stage_eff_discount_mean": [
                float(x) for x in self._stage_eff_discount_mean
            ],
            "stage_eff_discount_std": [
                float(x) for x in self._stage_eff_discount_std
            ],
            "stage_frac_eff_lt_gamma": [
                float(x) for x in self._stage_frac_eff_lt_gamma
            ],
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> "SafeWeightedValueIteration":
        """
        Execute safe weighted-LSE backward-induction value iteration.

        Runs up to ``n_sweeps`` full backward passes. After each pass, the
        sup-norm residual of the non-terminal value rows ``V[0..T-1]``
        against the previous pass is appended to :attr:`residuals`, and the
        wall-clock time for the pass is appended to :attr:`sweep_times_s`.
        Safe to call multiple times; each call re-initialises the tables.

        Returns:
            ``self`` -- for fluent chaining
            (``vi = SafeWeightedValueIteration(mdp, schedule).run()``).
        """
        # Reset tables so ``run()`` is idempotent.
        self.Q.fill(0.0)
        if self._v_init is not None:
            self.V[:] = self._v_init  # restore warm-start (terminal V[T,:]=0 already enforced in __init__)
        else:
            self.V.fill(0.0)
        self.pi.fill(0)
        self.residuals = []
        self.sweep_times_s = []
        self.n_sweeps = 0
        self.wall_clock_s = 0.0
        self.converged = False
        self.V_sweep_history = []

        # Build schedule report at run start.
        self.schedule_report = self._build_schedule_report()

        sweep_timer = SweepTimer()

        t_start = time.perf_counter()

        for _ in range(self._n_sweeps_request):
            # Reset per-stage diagnostics for this sweep.
            self._reset_stage_diagnostics()

            # Snapshot the non-terminal rows for residual computation.
            V_prev_non_terminal = self.V[: self._T].copy()      # (T, S)

            with sweep_timer.sweep():
                self._backward_pass()

            # The backward pass collects diagnostics in reverse stage order
            # (T-1, T-2, ..., 0). Reverse to get stage-indexed order (0..T-1).
            self._stage_clip_active.reverse()
            self._stage_beta_raw.reverse()
            self._stage_beta_cap.reverse()
            self._stage_beta_used.reverse()
            self._stage_eff_discount_mean.reverse()
            self._stage_eff_discount_std.reverse()
            self._stage_frac_eff_lt_gamma.reverse()

            residual = sup_norm_residual(
                self.V[: self._T], V_prev_non_terminal
            )
            self.residuals.append(float(residual))
            self.n_sweeps += 1
            self.V_sweep_history.append(self.V.copy())

            # Early stop (multi-pass only, and only if a strictly positive
            # tolerance was requested). A single-pass run has
            # ``_n_sweeps_request == 1`` and exits the loop naturally.
            if (
                self._n_sweeps_request > 1
                and self._tol > 0.0
                and residual < self._tol
            ):
                self.converged = True
                break

        # Single-pass is exact by construction -- flag it as converged.
        if self._n_sweeps_request == 1 and self.n_sweeps == 1:
            self.converged = True
        # If the full multi-pass budget was spent and we never tripped the
        # early-stop guard, fall back to "converged iff final residual is
        # numerically indistinguishable from zero". We use a fixed
        # post-hoc threshold (1e-12) rather than ``self._tol`` so that the
        # default ``tol=0`` (early-stop disabled) still reports the
        # obvious fact that a deterministic DAG converges after one sweep.
        elif not self.converged and self.residuals:
            self.converged = bool(self.residuals[-1] < 1e-12)

        self.wall_clock_s = float(time.perf_counter() - t_start)
        self.sweep_times_s = list(sweep_timer.sweep_times_s)

        # Build clipping summary at run end (uses last sweep's diagnostics).
        self.clipping_summary = self._build_clipping_summary()

        self._has_run = True
        return self

    def results(self) -> Dict:
        """
        Return a JSON-friendly summary of the run.

        Returns:
            Dict with keys:
                - ``Q_shape``         : ``list[int]`` -- shape of ``self.Q``.
                - ``V_shape``         : ``list[int]`` -- shape of ``self.V``.
                - ``n_sweeps``        : ``int`` -- number of sweeps executed.
                - ``wall_clock_s``    : ``float`` -- total seconds in ``run``.
                - ``residuals``       : ``list[float]`` -- one entry per sweep.
                - ``sweep_times_s``   : ``list[float]`` -- per-sweep seconds.
                - ``V0_sup_norm``     : ``float`` -- ``sup_norm(V[0], 0)``.
                - ``V0_mean``         : ``float`` -- mean of ``V[0, :]``.
                - ``V0_max``          : ``float`` -- max of ``V[0, :]``.
                - ``converged``       : ``bool`` -- see :attr:`converged`.
                - ``schedule_report`` : ``dict`` -- beta schedule summary.
                - ``clipping_summary``: ``dict`` -- per-stage clipping info.

        Raises:
            RuntimeError: if called before :meth:`run`.
        """
        if not self._has_run:
            raise RuntimeError(
                "SafeWeightedValueIteration.results() called before run(); "
                "call run() first."
            )
        V0 = self.V[0]                                           # (S,)
        base = {
            "Q_shape": list(self.Q.shape),
            "V_shape": list(self.V.shape),
            "n_sweeps": int(self.n_sweeps),
            "wall_clock_s": float(self.wall_clock_s),
            "residuals": [float(x) for x in self.residuals],
            "sweep_times_s": [float(x) for x in self.sweep_times_s],
            "V0_sup_norm": float(np.max(np.abs(V0))),
            "V0_mean": float(V0.mean()),
            "V0_max": float(V0.max()),
            "converged": bool(self.converged),
        }
        base.update({
            "schedule_report": self.schedule_report,
            "clipping_summary": self.clipping_summary,
        })
        return base
