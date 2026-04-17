"""
Finite-horizon classical Asynchronous Value Iteration.

Conventions (inherited from :mod:`finite_horizon_dp_utils`)
-----------------------------------------------------------
- Finite horizon ``T``: stages ``t in 0..T - 1``.
- Terminal convention: ``V[T, :] = 0``.
- Per-stage optimal Bellman backup:
  ``Q*[t, s, a] = R_bar[s, a]
                 + gamma * sum_{s'} P[s, a, s'] * V*[t + 1, s']``,
  ``V*[t, s]   = max_a Q*[t, s, a]``,
  ``pi*[t, s]  = argmax_a Q*[t, s, a]`` (``np.argmax`` tie-break: lowest index).

Async semantics
---------------
Asynchronous VI traverses stages in the same backward order as synchronous VI
(``T - 1, ..., 0``) but within each stage ``t`` it updates states one at a
time in a configurable order. In finite-horizon DP the per-state update
``V[t, s] <- max_a (R_bar[s, a] + gamma * P[s, a, :] @ V[t+1, :])`` depends
only on ``V[t+1, :]`` — which is fully committed before stage ``t`` begins —
so Gauss-Seidel within a stage is numerically equivalent to a vectorised
full-slab update. This planner is therefore an *interface* exercise: it
exposes the knobs an async planner needs (update order, per-sweep residuals,
priority heuristic) while guaranteeing convergence to the same ``V*`` as
synchronous VI.

Bit-exact recovery with ``ClassicalValueIteration``
---------------------------------------------------
With ``order="sequential"``, the per-stage update is implemented by computing
the full ``(S, A)`` Bellman-Q slab once — identical floating-point expression
to ``ClassicalValueIteration._backward_pass`` — and *then* iterating states in
the requested order purely to record per-state updates to ``V`` and ``pi``.
Because ``V[t, :] = Q[t, :, :].max(axis=1)`` is independent of within-stage
ordering, every supported order converges to the same tables as VI; the
"sequential" path is guaranteed bit-identical because it uses the identical
numpy reduction.

Update orders
-------------
``"sequential"``
    States ``0, 1, ..., S - 1``. Deterministic. Bit-exact with
    :class:`ClassicalValueIteration`.
``"reverse"``
    States ``S - 1, S - 2, ..., 0``. Deterministic.
``"random"``
    States shuffled by ``np.random.default_rng(seed)`` at the start of each
    sweep. Reproducible given a fixed ``seed``.
``"priority"``
    States sorted by the Bellman-error proxy
    ``|max_a Q_pre[t, s, a] - V_pre[t, s]|`` descending, where ``Q_pre`` is
    the Q-slab computed from the *current* ``V[t+1, :]`` and ``V_pre`` is
    the value table at the start of the sweep. Recomputed per sweep, per
    stage. Deterministic for a fixed MDP.

Residual convention
-------------------
For sweep ``k`` we log the sup-norm residual against the value table produced
by sweep ``k - 1`` over the non-terminal stages ``0..T - 1``. ``V[T, :]`` is
terminal-zero and excluded. For ``k = 1`` the previous table is the
zero-initialised ``V``, so the first residual equals
``sup_norm(V[0..T-1], 0)`` — the magnitude of value propagated by the first
backward pass.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np

from mushroom_rl.algorithms.value.dp.finite_horizon_dp_utils import (
    allocate_value_tables,
    bellman_q_backup,
    expected_reward,
    extract_mdp_arrays,
    sup_norm_residual,
)

# The ``experiments/`` tree has no package markers, so pull
# :class:`SweepTimer` in via an explicit ``sys.path`` insert. Phase-I spec
# §11.2 requires per-sweep wall-clock logging to live in the shared
# ``common.timing`` module; duplicating it here would fork the schema.
import pathlib
import sys

# mushroom-rl-dev/mushroom_rl/algorithms/value/dp/classical_async_value_iteration.py
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


__all__ = ["ClassicalAsyncValueIteration"]


_VALID_ORDERS = ("sequential", "reverse", "random", "priority")


class ClassicalAsyncValueIteration:
    """
    Finite-horizon Asynchronous Value Iteration.

    Like :class:`ClassicalValueIteration` but within each stage ``t``,
    states are visited in a configurable order instead of the implicit
    ``0..S - 1`` induced by the vectorised ``Q[t].max(axis=1)`` in sync VI.

    Because ``V[t, s]``'s backup depends only on ``V[t + 1, :]`` (finite
    horizon), within-stage Gauss-Seidel updates converge to the same
    ``V*`` as sync VI regardless of order. The ``order`` knob exists to
    support diagnostic and baseline experiments (e.g. priority sweeping,
    random-order robustness) while keeping convergence identical.

    Update orders
    -------------
    ``"sequential"``
        States ``0, 1, ..., S - 1``. Bit-exact with
        :class:`ClassicalValueIteration`.
    ``"reverse"``
        States ``S - 1, S - 2, ..., 0``.
    ``"random"``
        States shuffled with ``np.random.default_rng(seed)`` at each sweep.
    ``"priority"``
        States sorted by
        ``|max_a Q_pre[t, s, a] - V_pre[t, s]|`` descending, computed
        per sweep, per stage, from the Q-slab derived from the *current*
        ``V[t + 1, :]`` and the pre-sweep ``V`` snapshot.

    Attributes set by :meth:`run`:
        Q            : ``np.ndarray`` of shape ``(T, S, A)``, ``float64``.
        V            : ``np.ndarray`` of shape ``(T + 1, S)``, ``float64``;
                       ``V[T, :] = 0`` by construction.
        pi           : ``np.ndarray`` of shape ``(T, S)``, ``int64`` —
                       greedy ``argmax_a Q[t, s, a]``.
        residuals    : ``list[float]`` — one entry per completed sweep.
        sweep_times_s: ``list[float]`` — per-sweep wall-clock seconds.
        n_sweeps     : ``int`` — number of sweeps actually executed.
        wall_clock_s : ``float`` — total wall-clock seconds inside ``run``.
        converged    : ``bool`` — multi-pass only: last residual ``< tol``.

    Args:
        mdp: a :class:`mushroom_rl.environments.finite_mdp.FiniteMDP`
            (or any object exposing ``p``, ``r``, and
            ``info.{gamma, horizon}``).
        n_sweeps: positive integer number of full backward passes.
            ``1`` (the default) is exact on the finite-horizon DAG.
        order: within-stage update order; one of
            ``("sequential", "reverse", "random", "priority")``.
        tol: non-negative early-stopping tolerance on the sup-norm residual.
            ``0.0`` (the default) disables early stopping. Ignored when
            ``n_sweeps == 1``.
        seed: seed for the RNG used by ``order="random"``. Ignored
            otherwise. ``None`` uses fresh OS entropy per call.
    """

    def __init__(
        self,
        mdp,
        n_sweeps: int = 1,
        order: str = "sequential",
        tol: float = 0.0,
        seed: Optional[int] = None,
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

        # Result tables, zero-allocated. Terminal V[T, :] = 0 by construction.
        Q0, V0, pi0 = allocate_value_tables(self._S, self._A, self._T)
        self.Q: np.ndarray = Q0                           # (T, S, A)
        self.V: np.ndarray = V0                           # (T + 1, S)
        self.pi: np.ndarray = pi0                         # (T, S)

        # Timing / logging scaffolding.
        self.residuals: List[float] = []
        self.sweep_times_s: List[float] = []
        self.n_sweeps: int = 0
        self.wall_clock_s: float = 0.0
        self.converged: bool = False
        self._has_run: bool = False
        #: After each backward sweep, a copy of ``V`` (shape ``(T+1, S)``).
        self.V_sweep_history: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
            # Bellman-error proxy per state at stage t.
            bellman_error = np.abs(Q_t_pre.max(axis=1) - V_t_pre)  # (S,)
            # ``argsort`` is stable (kind="stable"); descending by negation
            # preserves index-order ties deterministically.
            return np.argsort(-bellman_error, kind="stable").astype(
                np.int64, copy=False
            )
        # Unreachable — validated in ``__init__``.
        raise AssertionError(f"unhandled order: {self._order!r}")

    def _backward_pass(self, rng: np.random.Generator) -> None:
        """Execute one full backward pass over stages ``T - 1, ..., 0``.

        Within each stage, states are updated one at a time in the order
        returned by :meth:`_state_order_for_stage`. Because ``Q[t, s, :]``
        depends only on ``V[t + 1, :]`` (which is frozen during this
        stage), the Q-slab is computed once for the stage using the same
        vectorised reduction as :class:`ClassicalValueIteration`, and the
        per-state loop only writes ``V[t, s]`` and ``pi[t, s]`` in the
        chosen order. This keeps ``order="sequential"`` bit-identical
        with sync VI.

        Updates ``self.Q``, ``self.V[0..T-1]``, and ``self.pi`` in place.
        ``self.V[T, :]`` is the terminal zero row and is left untouched.
        """
        for t in range(self._T - 1, -1, -1):
            # Pre-update Q-slab from the *current* V[t+1, :]. Identical
            # numpy expression to ClassicalValueIteration so sequential
            # order is bit-exact.
            Q_t = bellman_q_backup(
                t=t,
                V=self.V,
                r_bar=self._r_bar,
                p=self._p,
                gamma=self._gamma,
            )                                              # (S, A)
            # Snapshot the stage-t V row *before* the within-stage visits
            # so "priority" uses a consistent pre-sweep reference.
            V_t_pre = self.V[t].copy()                     # (S,)

            visit_order = self._state_order_for_stage(
                t=t,
                Q_t_pre=Q_t,
                V_t_pre=V_t_pre,
                rng=rng,
            )                                              # (S,)

            # Commit the Q slab once (sync); then walk states in the
            # configured order and write V[t, s] / pi[t, s]. Since the
            # Q slab does not change within the stage, commit order is
            # observationally irrelevant — but iterating in the
            # configured order keeps the planner honest to the "async
            # sweep order" contract and makes it trivial to extend to
            # genuinely value-dependent per-state updates in the future.
            self.Q[t] = Q_t
            for s in visit_order:
                s_int = int(s)
                self.V[t, s_int] = Q_t[s_int].max()
                self.pi[t, s_int] = int(Q_t[s_int].argmax())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> "ClassicalAsyncValueIteration":
        """
        Execute async value iteration.

        Runs up to ``n_sweeps`` full backward passes. After each pass,
        the sup-norm residual of the non-terminal value rows
        ``V[0..T - 1]`` against the previous pass is appended to
        :attr:`residuals`, and the wall-clock time for the pass is
        appended to :attr:`sweep_times_s`. Safe to call multiple times;
        each call re-initialises the tables (and re-seeds the RNG for
        ``order="random"``).

        Returns:
            ``self`` — for fluent chaining
            (``avi = ClassicalAsyncValueIteration(mdp).run()``).
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

        rng = np.random.default_rng(self._seed)
        sweep_timer = SweepTimer()

        t_start = time.perf_counter()

        for _ in range(self._n_sweeps_request):
            # Snapshot the non-terminal rows for residual computation.
            V_prev_non_terminal = self.V[: self._T].copy()  # (T, S)

            with sweep_timer.sweep():
                self._backward_pass(rng)

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

        # Single-pass is exact by construction — flag it as converged.
        if self._n_sweeps_request == 1 and self.n_sweeps == 1:
            self.converged = True
        # If the full multi-pass budget was spent and we never tripped the
        # early-stop guard, fall back to "converged iff final residual is
        # numerically indistinguishable from zero". Mirrors the VI
        # planner's post-hoc 1e-12 threshold so the default ``tol=0``
        # still reports the obvious fact that a deterministic DAG
        # converges after one sweep.
        elif not self.converged and self.residuals:
            self.converged = bool(self.residuals[-1] < 1e-12)

        self.wall_clock_s = float(time.perf_counter() - t_start)
        self.sweep_times_s = list(sweep_timer.sweep_times_s)
        self._has_run = True
        return self

    def results(self) -> Dict:
        """
        Return a JSON-friendly summary of the run.

        Returns:
            Dict with keys:
                - ``Q_shape``     : ``list[int]`` — shape of ``self.Q``.
                - ``V_shape``     : ``list[int]`` — shape of ``self.V``.
                - ``n_sweeps``    : ``int`` — number of sweeps executed.
                - ``order``       : ``str`` — the configured update order.
                - ``wall_clock_s``: ``float`` — total seconds in ``run``.
                - ``residuals``   : ``list[float]`` — one entry per sweep.
                - ``sweep_times_s``: ``list[float]`` — per-sweep seconds.
                - ``V0_sup_norm`` : ``float`` — ``sup_norm(V[0], 0)``.
                - ``V0_mean``     : ``float`` — mean of ``V[0, :]``.
                - ``V0_max``      : ``float`` — max of ``V[0, :]``.
                - ``converged``   : ``bool`` — see :attr:`converged`.

        Raises:
            RuntimeError: if called before :meth:`run`.
        """
        if not self._has_run:
            raise RuntimeError(
                "ClassicalAsyncValueIteration.results() called before "
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
        }
