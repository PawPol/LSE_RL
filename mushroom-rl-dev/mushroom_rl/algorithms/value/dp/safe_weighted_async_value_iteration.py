"""
Finite-horizon safe weighted-LSE Asynchronous Value Iteration.

This module mirrors :mod:`classical_async_value_iteration` but replaces the
standard Bellman backup with the safe weighted-LSE operator from
:class:`SafeWeightedCommon`.

Conventions (inherited from :mod:`finite_horizon_dp_utils`)
-----------------------------------------------------------
- Finite horizon ``T``: stages ``t in 0..T - 1``.
- Terminal convention: ``V[T, :] = 0``.
- Per-stage safe Bellman backup:

      Q^safe[t, s, a] = g_t^safe(R_bar[s, a], E[V[t+1, s'] | s, a])

  where ``g_t^safe`` is the safe weighted-LSE one-step target implemented
  via ``np.logaddexp`` for numerical stability.

      V*[t, s]  = max_a Q^safe[t, s, a]
      pi*[t, s] = argmax_a Q^safe[t, s, a]   (np.argmax tie-break: lowest index)

Classical recovery guarantee
----------------------------
When ``beta_used == 0`` at every stage (e.g. ``BetaSchedule.zeros(T, gamma)``),
the safe async VI produces bit-identical results to
:class:`ClassicalAsyncValueIteration` within numerical tolerance because
``compute_safe_target_batch`` collapses exactly to ``r_bar + gamma * E_v_next``.

Async semantics
---------------
Identical to the classical counterpart: stages are traversed backward
``T-1, ..., 0``; within each stage, states are visited in a configurable
order. Because ``Q[t, s, :]`` depends only on ``V[t+1, :]`` (which is frozen
during stage ``t``), the Q-slab is computed once per stage and the per-state
loop only writes ``V[t, s]`` and ``pi[t, s]``. All four update orders are
supported: ``"sequential"``, ``"reverse"``, ``"random"``, ``"priority"``.

Update orders
-------------
``"sequential"``
    States ``0, 1, ..., S - 1``. Deterministic. Bit-exact with
    :class:`SafeWeightedValueIteration` (and with ``ClassicalValueIteration``
    when ``beta_used == 0`` everywhere).
``"reverse"``
    States ``S - 1, S - 2, ..., 0``. Deterministic.
``"random"``
    States shuffled by ``np.random.default_rng(seed)`` at the start of each
    sweep. Reproducible given a fixed ``seed``.
``"priority"``
    States sorted by the Bellman-error proxy
    ``|max_a Q_safe[t, s, a] - V_pre[t, s]|`` descending, where ``Q_safe`` is
    the safe Q-slab computed from the *current* ``V[t+1, :]`` and ``V_pre`` is
    the value table at the start of the sweep. Deterministic for a fixed MDP.

Residual convention
-------------------
For sweep ``k`` we log the sup-norm residual against the value table produced
by sweep ``k - 1`` over the non-terminal stages ``0..T - 1``. ``V[T, :]`` is
terminal-zero and excluded.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from mushroom_rl.algorithms.value.dp.finite_horizon_dp_utils import (
    allocate_value_tables,
    expected_reward,
    extract_mdp_arrays,
    sup_norm_residual,
)
from mushroom_rl.algorithms.value.dp.safe_weighted_common import (
    BetaSchedule,
    SafeWeightedCommon,
)

# The ``experiments/`` tree has no package markers, so pull
# :class:`SweepTimer` in via an explicit ``sys.path`` insert.  Phase-I spec
# ss11.2 requires per-sweep wall-clock logging to live in the shared
# ``common.timing`` module; duplicating it here would fork the schema.
import pathlib
import sys

# mushroom-rl-dev/mushroom_rl/algorithms/value/dp/safe_weighted_async_value_iteration.py
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


__all__ = ["SafeWeightedAsyncValueIteration"]


_VALID_ORDERS = ("sequential", "reverse", "random", "priority")


class SafeWeightedAsyncValueIteration:
    """
    Finite-horizon safe weighted-LSE Asynchronous Value Iteration.

    Like :class:`ClassicalAsyncValueIteration` but uses the safe weighted-LSE
    Bellman operator from :class:`SafeWeightedCommon` instead of the classical
    ``r_bar + gamma * P @ V[t+1]`` backup. Within each stage ``t``, states are
    visited in a configurable order.

    Because ``V[t, s]``'s backup depends only on ``V[t + 1, :]`` (finite
    horizon), within-stage Gauss-Seidel updates converge to the same
    ``V*`` as sync safe VI regardless of order. The ``order`` knob exists to
    support diagnostic and baseline experiments.

    When ``beta_used == 0`` at every stage the planner is bit-identical to
    :class:`ClassicalAsyncValueIteration`.

    Update orders
    -------------
    ``"sequential"``
        States ``0, 1, ..., S - 1``. Bit-exact with safe sync VI.
    ``"reverse"``
        States ``S - 1, S - 2, ..., 0``.
    ``"random"``
        States shuffled with ``np.random.default_rng(seed)`` at each sweep.
    ``"priority"``
        States sorted by
        ``|max_a Q_safe[t, s, a] - V_pre[t, s]|`` descending, computed
        per sweep, per stage, from the safe Q-slab derived from the
        *current* ``V[t + 1, :]`` and the pre-sweep ``V`` snapshot.

    Attributes set by :meth:`run`:
        Q                : ``np.ndarray`` of shape ``(T, S, A)``, ``float64``.
        V                : ``np.ndarray`` of shape ``(T + 1, S)``, ``float64``;
                           ``V[T, :] = 0`` by construction.
        pi               : ``np.ndarray`` of shape ``(T, S)``, ``int64`` --
                           greedy ``argmax_a Q[t, s, a]``.
        residuals        : ``list[float]`` -- one entry per completed sweep.
        sweep_times_s    : ``list[float]`` -- per-sweep wall-clock seconds.
        n_sweeps         : ``int`` -- number of sweeps actually executed.
        wall_clock_s     : ``float`` -- total wall-clock seconds inside ``run``.
        converged        : ``bool`` -- multi-pass only: last residual ``< tol``.
        V_sweep_history  : ``list[np.ndarray]`` -- V snapshot after each sweep.
        schedule_report  : ``dict`` -- schedule metadata, set at ``run()`` start.
        clipping_summary : ``dict`` -- per-sweep clipping diagnostics.

    Args:
        mdp: a :class:`mushroom_rl.environments.finite_mdp.FiniteMDP`
            (or any object exposing ``p``, ``r``, and
            ``info.{gamma, horizon}``).
        schedule: a :class:`BetaSchedule` providing per-stage beta values.
        n_sweeps: positive integer number of full backward passes.
            ``1`` (the default) is exact on the finite-horizon DAG.
        tol: non-negative early-stopping tolerance on the sup-norm residual.
            ``0.0`` (the default) disables early stopping.
        order: within-stage update order; one of
            ``("sequential", "reverse", "random", "priority")``.
        seed: seed for the RNG used by ``order="random"``. Ignored
            otherwise. ``None`` uses fresh OS entropy per call.
        v_init: optional warm-start value table of shape ``(T+1, S)``.
    """

    def __init__(
        self,
        mdp,
        schedule: BetaSchedule,
        n_sweeps: int = 1,
        tol: float = 0.0,
        order: str = "sequential",
        seed: "int | None" = None,
        v_init: "np.ndarray | None" = None,
    ) -> None:
        if not isinstance(n_sweeps, (int, np.integer)) or int(n_sweeps) <= 0:
            raise ValueError(
                f"n_sweeps must be a positive integer; got {n_sweeps!r}."
            )
        if not isinstance(order, str) or order not in _VALID_ORDERS:
            raise ValueError(
                f"order must be one of {_VALID_ORDERS}; got {order!r}."
            )
        if not isinstance(tol, (int, float, np.floating, np.integer)):
            raise ValueError(
                f"tol must be a real scalar; got type={type(tol).__name__}."
            )
        if float(tol) < 0.0:
            raise ValueError(f"tol must be non-negative; got {tol}.")
        if seed is not None and not isinstance(seed, (int, np.integer)):
            raise ValueError(
                f"seed must be None or an integer; got "
                f"type={type(seed).__name__}."
            )

        # Extract + validate raw MDP tensors.
        p, r, horizon, gamma = extract_mdp_arrays(mdp)   # (S,A,S), (S,A,S)
        self._p: np.ndarray = p                           # (S, A, S)
        self._r: np.ndarray = r                           # (S, A, S)
        self._gamma: float = float(gamma)
        self._T: int = int(horizon)
        self._S: int = int(p.shape[0])
        self._A: int = int(p.shape[1])

        # Precompute the (S, A) expected-reward matrix once.
        self._r_bar: np.ndarray = expected_reward(p, r)   # (S, A)

        self._n_sweeps_request: int = int(n_sweeps)
        self._order: str = order
        self._tol: float = float(tol)
        self._seed: Optional[int] = (
            None if seed is None else int(seed)
        )

        # Safe weighted-LSE helper.
        self._schedule = schedule
        self._safe = SafeWeightedCommon(
            schedule, gamma=self._gamma, n_base=self._S
        )

        # Result tables, zero-allocated. Terminal V[T, :] = 0 by construction.
        Q0, V0, pi0 = allocate_value_tables(self._S, self._A, self._T)
        self.Q: np.ndarray = Q0                           # (T, S, A)
        self.V: np.ndarray = V0                           # (T + 1, S)
        self.pi: np.ndarray = pi0                         # (T, S)

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
        self.sweep_times_s: List[float] = []
        self.n_sweeps: int = 0
        self.wall_clock_s: float = 0.0
        self.converged: bool = False
        self._has_run: bool = False
        #: After each backward sweep, a copy of ``V`` (shape ``(T+1, S)``).
        self.V_sweep_history: List[np.ndarray] = []

        # Per-stage diagnostics (populated during run, most recent sweep).
        self._stage_clip_active: List[bool] = []
        self._stage_beta_used: List[float] = []
        self._stage_beta_cap: List[float] = []
        self._stage_eff_discount_mean: List[float] = []
        self._stage_eff_discount_std: List[float] = []
        self._stage_frac_eff_lt_gamma: List[float] = []

        # Per-sweep summaries.
        self._sweep_clip_fraction: List[float] = []
        self._sweep_eff_discount_mean: List[float] = []

        # Summary dicts (set during run).
        self.schedule_report: Dict[str, Any] = {}
        self.clipping_summary: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_stage_diagnostics(self, t: int) -> None:
        """Read instrumentation from ``self._safe`` and append to per-stage lists."""
        self._stage_clip_active.append(bool(self._safe.last_clip_active))
        self._stage_beta_used.append(float(self._safe.last_beta_used))
        self._stage_beta_cap.append(float(self._safe.last_beta_cap))

        eff_d = np.asarray(self._safe.last_effective_discount)
        self._stage_eff_discount_mean.append(float(eff_d.mean()))
        self._stage_eff_discount_std.append(float(eff_d.std()))
        self._stage_frac_eff_lt_gamma.append(
            float((eff_d < self._gamma).mean())
        )

    def _state_order_for_stage(
        self,
        t: int,
        Q_t_pre: np.ndarray,
        V_t_pre: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return the state-visit order for stage ``t`` under ``self._order``.

        Args:
            t: stage index in ``0..T - 1`` (unused by deterministic orders
                but kept for symmetry / future extension).
            Q_t_pre: pre-update Q-slab of shape ``(S, A)``; used by
                ``"priority"``.
            V_t_pre: pre-sweep V-row of shape ``(S,)``; used by
                ``"priority"``.
            rng: numpy RNG used by ``"random"``.

        Returns:
            Integer array of shape ``(S,)`` giving a permutation of
            ``0..S - 1``.
        """
        S = self._S
        if self._order == "sequential":
            return np.arange(S, dtype=np.int64)
        if self._order == "reverse":
            return np.arange(S - 1, -1, -1, dtype=np.int64)
        if self._order == "random":
            perm = np.arange(S, dtype=np.int64)
            rng.shuffle(perm)
            return perm
        if self._order == "priority":
            # Bellman-error proxy per state at stage t, based on safe Q-slab.
            bellman_error = np.abs(Q_t_pre.max(axis=1) - V_t_pre)  # (S,)
            return np.argsort(-bellman_error, kind="stable").astype(
                np.int64, copy=False
            )
        # Unreachable -- validated in ``__init__``.
        raise AssertionError(f"unhandled order: {self._order!r}")

    def _backward_pass(self, rng: np.random.Generator) -> None:
        """Execute one full backward pass over stages ``T - 1, ..., 0``.

        Within each stage the safe Q-slab is computed once via
        ``compute_safe_target_batch`` and then states are visited in the
        configured order. Because the Q-slab depends only on ``V[t + 1, :]``
        (frozen during this stage), within-stage ordering is observationally
        irrelevant -- the ``"sequential"`` path is bit-identical with safe
        sync VI.

        Updates ``self.Q``, ``self.V[0..T-1]``, and ``self.pi`` in place.
        ``self.V[T, :]`` is the terminal zero row and is left untouched.

        Also populates per-stage diagnostic lists for the current sweep.
        """
        # Reset per-stage diagnostics for this sweep.
        self._stage_clip_active = []
        self._stage_beta_used = []
        self._stage_beta_cap = []
        self._stage_eff_discount_mean = []
        self._stage_eff_discount_std = []
        self._stage_frac_eff_lt_gamma = []

        for t in range(self._T - 1, -1, -1):
            # Step 1: expected next-state value for all (s, a).
            # E_v_next[s, a] = sum_{s'} P[s, a, s'] * V[t+1, s']
            E_v_next = np.einsum(
                "ijk,k->ij", self._p, self.V[t + 1]
            )  # (S, A)

            # Step 2: safe Q slab for all (s, a).
            Q_t = self._safe.compute_safe_target_batch(
                self._r_bar, E_v_next, t
            )  # (S, A)
            self._record_stage_diagnostics(t)

            # Snapshot the stage-t V row *before* the within-stage visits
            # so "priority" uses a consistent pre-sweep reference.
            V_t_pre = self.V[t].copy()  # (S,)

            visit_order = self._state_order_for_stage(
                t=t,
                Q_t_pre=Q_t,
                V_t_pre=V_t_pre,
                rng=rng,
            )  # (S,)

            # Commit the Q slab once; then walk states in the configured
            # order and write V[t, s] / pi[t, s].
            self.Q[t] = Q_t
            for s in visit_order:
                s_int = int(s)
                self.V[t, s_int] = Q_t[s_int].max()
                self.pi[t, s_int] = int(Q_t[s_int].argmax())

        # The per-stage lists were appended in backward order (T-1, ..., 0).
        # Reverse so that index 0 corresponds to stage 0.
        self._stage_clip_active.reverse()
        self._stage_beta_used.reverse()
        self._stage_beta_cap.reverse()
        self._stage_eff_discount_mean.reverse()
        self._stage_eff_discount_std.reverse()
        self._stage_frac_eff_lt_gamma.reverse()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> "SafeWeightedAsyncValueIteration":
        """
        Execute safe weighted-LSE async value iteration.

        Runs up to ``n_sweeps`` full backward passes. After each pass,
        the sup-norm residual of the non-terminal value rows
        ``V[0..T - 1]`` against the previous pass is appended to
        :attr:`residuals`, and the wall-clock time for the pass is
        appended to :attr:`sweep_times_s`. Safe to call multiple times;
        each call re-initialises the tables (and re-seeds the RNG for
        ``order="random"``).

        Returns:
            ``self`` -- for fluent chaining.
        """
        # Reset tables so ``run()`` is idempotent.
        self.Q.fill(0.0)
        self.V.fill(0.0)
        self.pi.fill(0)
        self.residuals = []
        self.sweep_times_s = []
        self.n_sweeps = 0
        self.wall_clock_s = 0.0
        self.converged = False
        self.V_sweep_history = []
        self._sweep_clip_fraction = []
        self._sweep_eff_discount_mean = []

        # Build schedule report.
        self.schedule_report = {
            "task_family": self._schedule.task_family,
            "gamma": self._schedule.gamma,
            "sign": self._schedule.sign,
            "T": self._schedule.T,
            "beta_used_range": [
                float(min(self._schedule._beta_used_t)),
                float(max(self._schedule._beta_used_t)),
            ],
        }

        rng = np.random.default_rng(self._seed)
        sweep_timer = SweepTimer()

        t_start = time.perf_counter()

        for _ in range(self._n_sweeps_request):
            # Snapshot the non-terminal rows for residual computation.
            V_prev_non_terminal = self.V[: self._T].copy()  # (T, S)

            with sweep_timer.sweep():
                self._backward_pass(rng)

            # Record per-sweep summary from per-stage diagnostics.
            if self._stage_clip_active:
                clip_frac = float(
                    sum(self._stage_clip_active) / len(self._stage_clip_active)
                )
            else:
                clip_frac = 0.0
            self._sweep_clip_fraction.append(clip_frac)

            if self._stage_eff_discount_mean:
                eff_d_mean = float(np.mean(self._stage_eff_discount_mean))
            else:
                eff_d_mean = 0.0
            self._sweep_eff_discount_mean.append(eff_d_mean)

            residual = sup_norm_residual(
                self.V[: self._T], V_prev_non_terminal
            )
            self.residuals.append(float(residual))
            self.n_sweeps += 1
            self.V_sweep_history.append(self.V.copy())

            # Early stop (multi-pass only, and only if a strictly positive
            # tolerance was requested).
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
        # numerically indistinguishable from zero".
        elif not self.converged and self.residuals:
            self.converged = bool(self.residuals[-1] < 1e-12)

        self.wall_clock_s = float(time.perf_counter() - t_start)
        self.sweep_times_s = list(sweep_timer.sweep_times_s)

        # Build clipping summary.
        self.clipping_summary = {
            "n_sweeps": int(self.n_sweeps),
            "sweep_clip_fraction": [float(x) for x in self._sweep_clip_fraction],
            "sweep_eff_discount_mean": [float(x) for x in self._sweep_eff_discount_mean],
            "overall_clip_fraction": float(np.mean(self._sweep_clip_fraction))
            if self._sweep_clip_fraction
            else 0.0,
        }

        self._has_run = True
        return self

    def results(self) -> Dict[str, Any]:
        """
        Return a JSON-friendly summary of the run.

        Returns:
            Dict with keys:
                - ``Q_shape``            : ``list[int]`` -- shape of ``self.Q``.
                - ``V_shape``            : ``list[int]`` -- shape of ``self.V``.
                - ``n_sweeps``           : ``int`` -- number of sweeps executed.
                - ``order``              : ``str`` -- the configured update order.
                - ``wall_clock_s``       : ``float`` -- total seconds in ``run``.
                - ``residuals``          : ``list[float]`` -- one entry per sweep.
                - ``sweep_times_s``      : ``list[float]`` -- per-sweep seconds.
                - ``V0_sup_norm``        : ``float`` -- ``sup_norm(V[0], 0)``.
                - ``V0_mean``            : ``float`` -- mean of ``V[0, :]``.
                - ``V0_max``             : ``float`` -- max of ``V[0, :]``.
                - ``converged``          : ``bool`` -- see :attr:`converged`.
                - ``schedule_report``    : ``dict`` -- schedule metadata.
                - ``clipping_summary``   : ``dict`` -- per-sweep clipping info.

        Raises:
            RuntimeError: if called before :meth:`run`.
        """
        if not self._has_run:
            raise RuntimeError(
                "SafeWeightedAsyncValueIteration.results() called before "
                "run(); call run() first."
            )
        V0 = self.V[0]                                   # (S,)
        return {
            "Q_shape": list(self.Q.shape),
            "V_shape": list(self.V.shape),
            "n_sweeps": int(self.n_sweeps),
            "order": str(self._order),
            "wall_clock_s": float(self.wall_clock_s),
            "residuals": [float(x) for x in self.residuals],
            "sweep_times_s": [float(x) for x in self.sweep_times_s],
            "V0_sup_norm": float(np.max(np.abs(V0))),
            "V0_mean": float(V0.mean()),
            "V0_max": float(V0.max()),
            "converged": bool(self.converged),
            "schedule_report": self.schedule_report,
            "clipping_summary": self.clipping_summary,
        }
