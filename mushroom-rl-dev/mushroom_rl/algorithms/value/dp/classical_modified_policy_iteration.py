"""
Finite-horizon classical Modified Policy Iteration (MPI).

Conventions
-----------
- Finite horizon ``T``: stages ``t in 0..T-1``.
- Terminal convention: ``V[T, :] = 0``.
- Deterministic policy: ``pi[t, s] in {0, ..., A-1}`` selects a single action.
- Each outer MPI iteration alternates (a) a single combined
  *greedy-improvement + one-step-evaluation* backward pass with
  (b) ``m - 1`` additional pure fixed-policy partial-evaluation backward
  passes, and finally (c) records a sup-norm residual against the value
  table at the start of the iteration.

Algorithmic specification
-------------------------
At outer iteration ``k``, let ``V_k`` denote the value table at entry
(``V_0 = 0`` is the zero-initialised table). One outer iteration performs:

Step (a) — one greedy-improvement + one-step-evaluation backward sweep.

    For ``t`` in ``T - 1, ..., 0``:
        ``Q_full[t, s, a] = R_bar[s, a]
                            + gamma * sum_{s'} P[s, a, s'] * V[t + 1, s']``
        ``pi_new[t, s]    = argmax_a Q_full[t, s, a]``
        ``V[t, s]         = max_a Q_full[t, s, a]
                          = Q_full[t, s, pi_new[t, s]]``

    This is identical to one full backward sweep of value iteration.

Step (b) — ``m - 1`` additional fixed-policy backward sweeps under
``pi_new`` (skipped when ``m == 1``).

    For each of the remaining sweeps, for ``t`` in ``T - 1, ..., 0``:
        ``Q_pi[t, s, pi_new[t, s]] = R_bar[s, pi_new[t, s]]
                                     + gamma * sum_{s'} P[s, pi_new[t, s], s']
                                              * V[t + 1, s']``
        ``V[t, s]                  = Q_pi[t, s, pi_new[t, s]]``

Step (c) — bookkeeping.

    ``residual = ||V_new[0..T - 1] - V_k[0..T - 1]||_inf``
    append ``residual`` to :attr:`residuals`
    commit ``pi = pi_new`` and the full ``Q_full`` from step (a).

Extreme-case equivalences
-------------------------
- ``m == 1``: step (b) is empty, so the outer iteration reduces to one
  value-iteration backward sweep. With ``max_iter == 1`` the result
  matches :class:`ClassicalValueIteration` **bit-exactly** (up to
  floating-point tie-breaking in ``argmax``).
- ``m >= 2`` (in particular ``m -> infinity``): finite-horizon
  fixed-policy evaluation is exact after a single backward sweep.
  The second inner sweep therefore already produces the exact
  ``V^{pi_new}``, and any additional sweeps are idempotent. Each outer
  iteration is consequently equivalent to one policy-iteration step
  (exact evaluation + greedy improvement), so MPI with ``m >= 2`` and a
  large ``max_iter`` budget converges to the same optimum as
  :class:`ClassicalPolicyIteration`, which in turn coincides with
  :class:`ClassicalValueIteration`.

Stopping rule
-------------
The outer loop terminates at iteration ``k`` as soon as **any** of the
following holds:

- ``pi_new`` equals the policy used at the start of the iteration
  (``policy_stable``),
- the residual falls **strictly** below ``tol`` (when ``tol > 0``),
- ``k >= max_iter``.

Residual convention
-------------------
The ``k``-th entry of :attr:`residuals` is the sup-norm residual between
the value table produced by outer iteration ``k`` and the table from
iteration ``k - 1``. For ``k = 1`` the previous table is the
zero-initialised ``V``, so the first entry equals
``sup_norm(V_new[0..T - 1], 0)`` — the magnitude of value propagated by
the first outer iteration. This matches the
:class:`ClassicalValueIteration` / :class:`ClassicalPolicyIteration`
convention so per-iteration convergence plots are directly comparable
across planners.
"""
from __future__ import annotations

import pathlib
import sys
import time
from typing import Dict, List, Optional

import numpy as np

from mushroom_rl.algorithms.value.dp.finite_horizon_dp_utils import (
    allocate_value_tables,
    bellman_q_backup,
    bellman_q_policy_backup,
    expected_reward,
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
# mushroom-rl-dev/mushroom_rl/algorithms/value/dp/classical_modified_policy_iteration.py
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


__all__ = ["ClassicalModifiedPolicyIteration"]


class ClassicalModifiedPolicyIteration:
    """
    Finite-horizon classical Modified Policy Iteration (MPI).

    MPI interpolates between value iteration (``m = 1``) and policy
    iteration (``m -> infinity``, i.e. ``m >= 2`` in finite horizon). At
    each outer iteration:

    1. One combined *greedy-improvement + one-step-evaluation* backward
       sweep produces ``Q_full[t, s, a]`` from the current value table,
       sets ``pi_new[t, s] = argmax_a Q_full[t, s, a]``, and updates
       ``V[t, s] = max_a Q_full[t, s, a]``.
    2. ``m - 1`` additional fixed-policy backward sweeps under
       ``pi_new`` further evaluate the policy. In finite horizon a
       single sweep already yields exact policy evaluation, so any
       ``m >= 2`` behaves like one full PI step; ``m == 1`` skips this
       step entirely and collapses to a VI sweep.
    3. The sup-norm residual against the value table from the start of
       the outer iteration is recorded.

    Attributes set by :meth:`run`:
        Q             : ``np.ndarray`` of shape ``(T, S, A)``, ``float64``
                        — full action-value table ``Q_full`` from the
                        final iteration's step-(a) backward sweep
                        (optimal Bellman operator applied to the value
                        table at the start of that iteration). For
                        ``m == 1`` this coincides with
                        :attr:`ClassicalValueIteration.Q`. After
                        convergence, this equals ``Q^{pi_new}`` because
                        ``V^{pi_new} = V_k`` at a fixed point.
        V             : ``np.ndarray`` of shape ``(T + 1, S)``, ``float64``
                        — value table from the final iteration;
                        ``V[T, :] = 0`` by construction.
        pi            : ``np.ndarray`` of shape ``(T, S)``, ``int64`` —
                        final (greedy-improved) policy.
        residuals     : ``list[float]`` — one entry per outer iteration,
                        ``||V_new[0..T - 1] - V_old[0..T - 1]||_inf``.
        sweep_times_s : ``list[float]`` — per-outer-iteration wall-clock
                        seconds, covering all ``m`` inner sweeps plus
                        the improvement step.
        n_iters       : ``int`` — number of outer iterations executed.
        wall_clock_s  : ``float`` — total wall-clock seconds spent in
                        :meth:`run`.
        policy_stable : ``bool`` — ``True`` iff the loop exited because
                        ``pi_new`` matched the policy used at the start
                        of the iteration.

    Args:
        mdp: a :class:`mushroom_rl.environments.finite_mdp.FiniteMDP`
            (or any object exposing ``p``, ``r``, and
            ``info.{gamma, horizon}``).
        m: positive integer number of backward-induction sweeps per
            outer iteration. ``m == 1`` recovers value iteration, any
            ``m >= 2`` gives exact finite-horizon policy evaluation in
            the inner loop and thus behaves like policy iteration.
            Defaults to ``5``.
        max_iter: positive integer maximum number of outer MPI
            iterations. Defaults to ``100``.
        tol: non-negative early-stopping tolerance on the per-iteration
            sup-norm residual. ``0.0`` (the default) disables
            residual-based early stopping, leaving the policy-stability
            and ``max_iter`` guards in charge. Strictly positive values
            trigger early termination as soon as an iteration's
            residual falls strictly below ``tol``.
        init_pi: optional initial policy of shape ``(T, S)`` with
            integer entries in ``[0, A)``. When ``None``, the initial
            policy is irrelevant because step (a) re-derives
            ``pi_new`` from the zero-initialised value table before
            any fixed-policy sweep runs; the all-zeros policy is then
            stored only as a placeholder. Included for symmetry with
            :class:`ClassicalPolicyIteration` and for future
            warm-start experiments.
    """

    def __init__(
        self,
        mdp,
        m: int = 5,
        max_iter: int = 100,
        tol: float = 0.0,
        init_pi: Optional[np.ndarray] = None,
    ) -> None:
        if not isinstance(m, (int, np.integer)) or int(m) <= 0:
            raise ValueError(f"m must be a positive integer; got {m!r}.")
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
        self._p: np.ndarray = p                           # (S, A, S)
        self._r: np.ndarray = r                           # (S, A, S)
        self._gamma: float = float(gamma)
        self._T: int = int(horizon)
        self._S: int = int(p.shape[0])
        self._A: int = int(p.shape[1])

        # Precompute the (S, A) expected-reward matrix once.
        self._r_bar: np.ndarray = expected_reward(p, r)   # (S, A)

        self._m: int = int(m)
        self._max_iter: int = int(max_iter)
        self._tol: float = float(tol)

        # Optional initial policy (placeholder only; step (a) derives the
        # first real policy from the zero value table on iteration 1).
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
        self.Q: np.ndarray = Q0                            # (T, S, A)
        self.V: np.ndarray = V0                            # (T + 1, S)
        self.pi: np.ndarray = pi0                          # (T, S)

        # Timing / logging scaffolding.
        self.residuals: List[float] = []
        self.sweep_times_s: List[float] = []
        self.n_iters: int = 0
        self.wall_clock_s: float = 0.0
        self.policy_stable: bool = False
        self._has_run: bool = False
        #: After each outer MPI iteration, a copy of ``V`` (shape ``(T+1, S)``).
        self.V_sweep_history: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _greedy_eval_pass(self) -> np.ndarray:
        """Run step (a): one greedy-improvement + one-step-eval backward pass.

        For each stage ``t`` from ``T - 1`` down to ``0``, compute the
        full ``Q_full[t]`` from the current ``V[t + 1]``, extract the
        greedy policy ``pi_new[t] = argmax_a Q_full[t]``, and update
        ``V[t] = max_a Q_full[t]``. The full-Q table is committed to
        ``self.Q`` so that the externally visible
        :attr:`ClassicalModifiedPolicyIteration.Q` is always the full
        action-value tensor w.r.t. the value table at the start of the
        current outer iteration (matching VI's output exactly when
        ``m == 1``).

        Returns:
            ``pi_new`` of shape ``(T, S)`` as ``int64``.
        """
        pi_new = np.empty((self._T, self._S), dtype=np.int64)
        for t in range(self._T - 1, -1, -1):
            # Q_full[t] uses V[t + 1]; shape (S, A).
            Q_t = bellman_q_backup(
                t=t,
                V=self.V,
                r_bar=self._r_bar,
                p=self._p,
                gamma=self._gamma,
            )
            self.Q[t] = Q_t
            # Greedy improvement; argmax tie-break: lowest action index wins.
            pi_t = greedy_policy(Q_t)                       # (S,)
            pi_new[t] = pi_t
            # V[t, s] = Q_full[t, s, pi_new[t, s]] = max_a Q_full[t, s, a].
            self.V[t] = Q_t[np.arange(self._S), pi_t]       # (S,)
        return pi_new

    def _partial_eval_pass(self, pi_new: np.ndarray) -> None:
        """Run one fixed-policy partial-evaluation backward pass under ``pi_new``.

        Updates ``self.V[0..T - 1]`` in place. ``self.Q`` is NOT
        overwritten here: the externally visible Q is the full-Q table
        from :meth:`_greedy_eval_pass` (to preserve bit-exact VI
        recovery when ``m == 1``).

        Args:
            pi_new: deterministic policy of shape ``(T, S)`` to evaluate.
        """
        for t in range(self._T - 1, -1, -1):
            Q_pi_t = bellman_q_policy_backup(             # (S, A)
                t=t,
                V=self.V,
                r_bar=self._r_bar,
                p=self._p,
                gamma=self._gamma,
                pi_t=pi_new[t],                           # (S,)
            )
            # V[t, s] = Q^pi[t, s, pi_new[t, s]].
            self.V[t] = Q_pi_t[np.arange(self._S), pi_new[t]]  # (S,)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> "ClassicalModifiedPolicyIteration":
        """
        Execute finite-horizon Modified Policy Iteration.

        Runs up to ``max_iter`` outer iterations. Each outer iteration
        performs step (a) (one greedy-improvement + one-step-evaluation
        backward sweep) followed by step (b) (``m - 1`` fixed-policy
        partial-evaluation backward sweeps), then records the sup-norm
        residual between the value table produced by this iteration and
        the value table from the previous iteration (or the
        zero-initialised table on the first iteration). Safe to call
        multiple times; each call re-initialises the tables.

        Returns:
            ``self`` — for fluent chaining
            (``mpi = ClassicalModifiedPolicyIteration(mdp).run()``).
        """
        # Reset tables so run() is idempotent.
        self.Q.fill(0.0)
        self.V.fill(0.0)
        self.pi = self._init_pi.copy()                      # (T, S)
        self.residuals = []
        self.sweep_times_s = []
        self.n_iters = 0
        self.wall_clock_s = 0.0
        self.policy_stable = False
        self.V_sweep_history = []

        iter_timer = SweepTimer()
        t_start = time.perf_counter()

        # V at the start of the current outer iteration; used only for
        # the residual computation. The first iteration compares against
        # the zero-initialised table, matching VI/PI conventions.
        V_prev_non_terminal = np.zeros(
            (self._T, self._S), dtype=np.float64
        )                                                   # (T, S)

        for _ in range(self._max_iter):
            pi_old = self.pi                                # (T, S)

            with iter_timer.sweep():
                # Step (a): greedy-improvement + one-step-evaluation.
                pi_new = self._greedy_eval_pass()           # (T, S)

                # Step (b): ``m - 1`` additional fixed-policy sweeps.
                # Finite-horizon PE is exact after one sweep, so any
                # ``m >= 2`` already yields V^{pi_new}; further sweeps
                # are idempotent but exercised to match spec §4.3's
                # "bounded sweeps per policy" interface.
                for _ in range(self._m - 1):
                    self._partial_eval_pass(pi_new)

            # Step (c): residual + bookkeeping.
            residual = sup_norm_residual(
                self.V[: self._T], V_prev_non_terminal
            )
            self.residuals.append(float(residual))
            self.n_iters += 1
            self.V_sweep_history.append(self.V.copy())

            # Commit the improved policy as the current policy.
            policy_unchanged = bool(np.array_equal(pi_new, pi_old))
            self.pi = pi_new                                # (T, S)

            # Policy-stability check: if the improved policy matches the
            # policy we entered the iteration with, the Bellman
            # optimality equation holds at ``(Q_full, V)`` and we are at
            # a fixed point. (Skips iteration 1, where ``pi_old`` is the
            # placeholder ``_init_pi``.)
            if policy_unchanged and self.n_iters >= 2:
                self.policy_stable = True
                break

            # Residual-based early stop (opt-in via ``tol > 0``).
            if self._tol > 0.0 and residual < self._tol:
                break

            # Advance the residual anchor for the next iteration.
            V_prev_non_terminal = self.V[: self._T].copy()  # (T, S)

        self.wall_clock_s = float(time.perf_counter() - t_start)
        self.sweep_times_s = list(iter_timer.sweep_times_s)
        self._has_run = True
        return self

    def results(self) -> Dict:
        """
        Return a JSON-friendly summary of the run.

        Returns:
            Dict with keys:
                - ``Q_shape``       : ``list[int]`` — shape of ``self.Q``.
                - ``V_shape``       : ``list[int]`` — shape of ``self.V``.
                - ``m``             : ``int`` — inner-sweep budget.
                - ``n_iters``       : ``int`` — number of outer iterations.
                - ``wall_clock_s``  : ``float`` — total seconds in ``run``.
                - ``residuals``     : ``list[float]`` — one per iteration.
                - ``sweep_times_s`` : ``list[float]`` — per-iteration seconds.
                - ``policy_stable`` : ``bool`` — loop exited on stability.
                - ``V0_sup_norm``   : ``float`` — ``sup_norm(V[0], 0)``.
                - ``V0_mean``       : ``float`` — mean of ``V[0, :]``.
                - ``V0_max``        : ``float`` — max of ``V[0, :]``.

        Raises:
            RuntimeError: if called before :meth:`run`.
        """
        if not self._has_run:
            raise RuntimeError(
                "ClassicalModifiedPolicyIteration.results() called before "
                "run(); call run() first."
            )
        V0 = self.V[0]                                       # (S,)
        return {
            "Q_shape": list(self.Q.shape),
            "V_shape": list(self.V.shape),
            "m": int(self._m),
            "n_iters": int(self.n_iters),
            "wall_clock_s": float(self.wall_clock_s),
            "residuals": [float(x) for x in self.residuals],
            "sweep_times_s": [float(x) for x in self.sweep_times_s],
            "policy_stable": bool(self.policy_stable),
            "V0_sup_norm": float(np.max(np.abs(V0))),
            "V0_mean": float(V0.mean()),
            "V0_max": float(V0.max()),
        }
