"""
Finite-horizon safe weighted-LSE Modified Policy Iteration (MPI).

This planner mirrors :class:`ClassicalModifiedPolicyIteration` but replaces
every Bellman backup (in both the greedy-improvement sweep and the
fixed-policy evaluation sweeps) with the safe weighted-LSE operator
provided by :class:`SafeWeightedCommon`.

Conventions
-----------
- Finite horizon ``T``: stages ``t in 0..T-1``.
- Terminal convention: ``V[T, :] = 0``.
- Deterministic policy: ``pi[t, s] in {0, ..., A-1}`` selects a single action.
- Each outer MPI iteration alternates (a) a single combined
  *greedy-improvement + one-step safe evaluation* backward pass with
  (b) ``m - 1`` additional pure fixed-policy safe partial-evaluation
  backward passes, and finally (c) records a sup-norm residual against
  the value table at the start of the iteration.

Safe backup
-----------
The safe weighted-LSE one-step target at stage *t* is

    g_t^safe(r, v) = ((1+gamma) / beta_t) *
        [log(exp(beta_t * r) + gamma * exp(beta_t * v)) - log(1+gamma)]

implemented via ``np.logaddexp`` to avoid overflow/underflow.  When
``beta_used == 0`` the classical target ``r + gamma * v`` is returned
exactly (no logaddexp call), guaranteeing bit-identical classical
recovery.

Extreme-case equivalences (safe extension)
------------------------------------------
All equivalences from the classical MPI docstring carry over when a
constant (stage-independent) schedule is used:

- ``m == 1``, ``beta_used = 0``: collapses to one classical VI sweep.
- ``m >= 2``, ``beta_used = 0``: collapses to one classical PI step
  (exact finite-horizon evaluation + greedy improvement).
- ``m == 1``, ``beta_used != 0``: one safe-VI backward sweep.
- ``m >= 2``, ``beta_used != 0``: safe PI step per outer iteration.

For stage-varying schedules the extreme-case equivalence still holds
per-stage; the overall planner interpolates safe VI and safe PI in the
same sense as classical MPI does for classical VI/PI.

Stopping rule
-------------
Identical to :class:`ClassicalModifiedPolicyIteration`:

- ``pi_new`` equals the previous policy (``policy_stable``), or
- residual falls strictly below ``tol`` (when ``tol > 0``), or
- ``k >= max_iter``.
"""
from __future__ import annotations

import pathlib
import sys
import time
from typing import Dict, List, Optional

import numpy as np

from mushroom_rl.algorithms.value.dp.finite_horizon_dp_utils import (
    allocate_value_tables,
    expected_reward,
    extract_mdp_arrays,
    greedy_policy,
    sup_norm_residual,
)
from mushroom_rl.algorithms.value.dp.safe_weighted_common import (
    BetaSchedule,
    SafeWeightedCommon,
)

# Pull :class:`SweepTimer` in via an explicit ``sys.path`` insert so the
# ``experiments/`` tree (which has no package markers) is importable from
# inside the MushroomRL subpackage.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.weighted_lse_dp.common.timing import SweepTimer  # noqa: E402


__all__ = ["SafeWeightedModifiedPolicyIteration"]


class SafeWeightedModifiedPolicyIteration:
    """
    Finite-horizon safe weighted-LSE Modified Policy Iteration (MPI).

    MPI interpolates between safe value iteration (``m = 1``) and safe
    policy iteration (``m -> infinity``, i.e. ``m >= 2`` in finite
    horizon). At each outer iteration:

    1. One combined *greedy-improvement + one-step safe evaluation*
       backward sweep produces ``Q_full[t, s, a]`` via the safe
       weighted-LSE backup from the current value table, sets
       ``pi_new[t, s] = argmax_a Q_full[t, s, a]``, and updates
       ``V[t, s] = max_a Q_full[t, s, a]``.
    2. ``m - 1`` additional fixed-policy backward sweeps under
       ``pi_new`` further evaluate the policy using the safe backup.
       In finite horizon a single sweep already yields exact (safe)
       policy evaluation, so any ``m >= 2`` behaves like one full
       safe PI step; ``m == 1`` skips this step entirely and collapses
       to a safe VI sweep.
    3. The sup-norm residual against the value table from the start of
       the outer iteration is recorded.

    beta = 0 classical recovery
    ----------------------------
    When ``beta_used == 0`` at every stage, all safe backups reduce to
    the classical Bellman operator and this planner produces results
    bit-identical to :class:`ClassicalModifiedPolicyIteration` (within
    numerical tolerance).

    Attributes set by :meth:`run`:
        Q             : ``np.ndarray`` of shape ``(T, S, A)``
        V             : ``np.ndarray`` of shape ``(T + 1, S)``
        pi            : ``np.ndarray`` of shape ``(T, S)``
        residuals     : ``list[float]``
        sweep_times_s : ``list[float]``
        n_iters       : ``int``
        wall_clock_s  : ``float``
        policy_stable : ``bool``
        V_sweep_history : ``list[np.ndarray]``
        schedule_report : ``dict``
        clipping_summary : ``dict``

    Args:
        mdp: a :class:`~mushroom_rl.environments.finite_mdp.FiniteMDP`
            (or any object exposing ``p``, ``r``, and
            ``info.{gamma, horizon}``).
        schedule: a :class:`BetaSchedule` providing per-stage beta values.
        m: positive integer number of backward-induction sweeps per
            outer iteration. ``m == 1`` recovers safe VI, any
            ``m >= 2`` gives exact finite-horizon safe policy evaluation
            in the inner loop and thus behaves like safe PI.
            Defaults to ``2``.
        max_iter: positive integer maximum number of outer MPI
            iterations. Defaults to ``100``.
        tol: non-negative early-stopping tolerance on the per-iteration
            sup-norm residual. ``0.0`` (the default) disables
            residual-based early stopping.
        v_init: optional initial value table of shape ``(T+1, S)``.
            When ``None``, the value table is zero-initialised.
    """

    def __init__(
        self,
        mdp,
        schedule: BetaSchedule,
        m: int = 2,
        max_iter: int = 100,
        tol: float = 0.0,
        v_init: "np.ndarray | None" = None,
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

        # Safe weighted-LSE operator (composition).
        self._schedule = schedule
        if schedule.T != self._T:
            raise ValueError(
                f"{type(self).__name__}: schedule.T={schedule.T} does not "
                f"match MDP horizon={self._T}. The schedule must be "
                "calibrated for this exact horizon."
            )
        self._safe = SafeWeightedCommon(
            schedule=schedule,
            gamma=self._gamma,
            n_base=self._S,
        )

        # Result tables, zero-allocated. Terminal V[T, :] = 0.
        Q0, V0, pi0 = allocate_value_tables(self._S, self._A, self._T)
        self.Q: np.ndarray = Q0                            # (T, S, A)
        self.V: np.ndarray = V0                            # (T + 1, S)
        self.pi: np.ndarray = pi0                          # (T, S)

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
        self.n_iters: int = 0
        self.wall_clock_s: float = 0.0
        self.policy_stable: bool = False
        self._has_run: bool = False
        self.V_sweep_history: List[np.ndarray] = []

        # Per-stage diagnostics (reset each outer iteration).
        self._stage_clip_active: List[bool] = []
        self._stage_beta_used: List[float] = []
        self._stage_beta_cap: List[float] = []
        self._stage_eff_discount_mean: List[float] = []
        self._stage_eff_discount_std: List[float] = []
        self._stage_frac_eff_lt_gamma: List[float] = []

        # Per-outer-iteration summaries.
        self._iter_clip_fraction: List[float] = []
        self._iter_eff_discount_mean: List[float] = []

        # Result reports (populated after run()).
        self.schedule_report: dict = {}
        self.clipping_summary: dict = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_stage_diagnostics(self, t: int) -> None:
        """Record per-stage instrumentation from the last safe batch call."""
        self._stage_clip_active.append(bool(self._safe.last_clip_active))
        self._stage_beta_used.append(float(self._safe.last_beta_used))
        self._stage_beta_cap.append(float(self._safe.last_beta_cap))

        eff_d = np.asarray(self._safe.last_effective_discount)
        self._stage_eff_discount_mean.append(float(np.mean(eff_d)))
        self._stage_eff_discount_std.append(float(np.std(eff_d)))
        self._stage_frac_eff_lt_gamma.append(
            float(np.mean(eff_d < self._gamma))
        )

    def _record_iter_summary(self) -> None:
        """Summarise per-stage diagnostics into per-iteration aggregates."""
        if self._stage_clip_active:
            clip_frac = float(np.mean(self._stage_clip_active))
        else:
            clip_frac = 0.0
        self._iter_clip_fraction.append(clip_frac)

        if self._stage_eff_discount_mean:
            eff_mean = float(np.mean(self._stage_eff_discount_mean))
        else:
            eff_mean = float(self._gamma)
        self._iter_eff_discount_mean.append(eff_mean)

    def _greedy_eval_pass(self) -> np.ndarray:
        """Run step (a): one greedy-improvement + one-step safe eval backward pass.

        For each stage ``t`` from ``T - 1`` down to ``0``, compute the
        full ``Q_full[t]`` via the safe weighted-LSE backup from the
        current ``V[t + 1]``, extract the greedy policy
        ``pi_new[t] = argmax_a Q_full[t]``, and update
        ``V[t] = max_a Q_full[t]``.

        Returns:
            ``pi_new`` of shape ``(T, S)`` as ``int64``.
        """
        pi_new = np.empty((self._T, self._S), dtype=np.int64)
        for t in range(self._T - 1, -1, -1):
            # E[V_{t+1}(s') | s, a] via transition probabilities.
            E_v_next = np.einsum(
                "ijk,k->ij", self._p, self.V[t + 1]
            )  # (S, A)

            # Safe weighted-LSE backup: Q_full[t, s, a] = g_t^safe(r_bar, E_v_next).
            Q_t = self._safe.compute_safe_target_batch(
                self._r_bar, E_v_next, t
            )  # (S, A)

            # Record diagnostics from this stage's safe backup.
            self._record_stage_diagnostics(t)

            self.Q[t] = Q_t

            # Greedy improvement; argmax tie-break: lowest action index wins.
            pi_t = greedy_policy(Q_t)                       # (S,)
            pi_new[t] = pi_t

            # V[t, s] = max_a Q_full[t, s, a].
            self.V[t] = Q_t[np.arange(self._S), pi_t]       # (S,)
        return pi_new

    def _partial_eval_pass(self, pi_new: np.ndarray) -> None:
        """Run one fixed-policy safe partial-evaluation backward pass under ``pi_new``.

        Updates ``self.V[0..T - 1]`` in place. ``self.Q`` is NOT
        overwritten here: the externally visible Q is the full-Q table
        from :meth:`_greedy_eval_pass`.

        Args:
            pi_new: deterministic policy of shape ``(T, S)`` to evaluate.
        """
        states = np.arange(self._S)
        for t in range(self._T - 1, -1, -1):
            pi_new_t = pi_new[t]  # (S,)

            # Full (S, A) safe backup, then read off the policy column.
            E_v_next_full = np.einsum(
                "ijk,k->ij", self._p, self.V[t + 1]
            )  # (S, A)

            Q_all = self._safe.compute_safe_target_batch(
                self._r_bar, E_v_next_full, t
            )  # (S, A)

            # V[t, s] = Q^safe_pi[t, s, pi_new[t, s]].
            self.V[t] = Q_all[states, pi_new_t]  # (S,)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> "SafeWeightedModifiedPolicyIteration":
        """
        Execute finite-horizon safe weighted-LSE Modified Policy Iteration.

        Runs up to ``max_iter`` outer iterations. Each outer iteration
        performs step (a) (one greedy-improvement + one-step safe evaluation
        backward sweep) followed by step (b) (``m - 1`` fixed-policy
        safe partial-evaluation backward sweeps), then records the sup-norm
        residual. Safe to call multiple times; each call re-initialises
        the tables.

        Returns:
            ``self`` -- for fluent chaining.
        """
        # Reset tables so run() is idempotent.
        self.Q.fill(0.0)
        if self._v_init is not None:
            self.V[:] = self._v_init  # restore warm-start (terminal V[T,:]=0 already enforced in __init__)
        else:
            self.V.fill(0.0)
        self.pi = np.zeros((self._T, self._S), dtype=np.int64)
        self.residuals = []
        self.sweep_times_s = []
        self.n_iters = 0
        self.wall_clock_s = 0.0
        self.policy_stable = False
        self.V_sweep_history = []
        self._iter_clip_fraction = []
        self._iter_eff_discount_mean = []

        iter_timer = SweepTimer()
        t_start = time.perf_counter()

        # V at the start of the current outer iteration; used only for
        # the residual computation.
        V_prev_non_terminal = np.zeros(
            (self._T, self._S), dtype=np.float64
        )  # (T, S)

        for _ in range(self._max_iter):
            pi_old = self.pi  # (T, S)

            # Reset per-stage diagnostics for this outer iteration.
            self._stage_clip_active = []
            self._stage_beta_used = []
            self._stage_beta_cap = []
            self._stage_eff_discount_mean = []
            self._stage_eff_discount_std = []
            self._stage_frac_eff_lt_gamma = []

            with iter_timer.sweep():
                # Step (a): greedy-improvement + one-step safe evaluation.
                pi_new = self._greedy_eval_pass()  # (T, S)

                # Step (b): m - 1 additional fixed-policy safe sweeps.
                for _ in range(self._m - 1):
                    self._partial_eval_pass(pi_new)

            # Record per-iteration summary from stage diagnostics.
            self._record_iter_summary()

            # Step (c): residual + bookkeeping.
            residual = sup_norm_residual(
                self.V[: self._T], V_prev_non_terminal
            )
            self.residuals.append(float(residual))
            self.n_iters += 1
            self.V_sweep_history.append(self.V.copy())

            # Commit the improved policy as the current policy.
            policy_unchanged = bool(np.array_equal(pi_new, pi_old))
            self.pi = pi_new  # (T, S)

            # Policy-stability check (skip iteration 1 where pi_old is
            # the placeholder zero policy).
            if policy_unchanged and self.n_iters >= 2:
                self.policy_stable = True
                break

            # Residual-based early stop (opt-in via tol > 0).
            if self._tol > 0.0 and residual < self._tol:
                break

            # Advance the residual anchor for the next iteration.
            V_prev_non_terminal = self.V[: self._T].copy()  # (T, S)

        self.wall_clock_s = float(time.perf_counter() - t_start)
        self.sweep_times_s = list(iter_timer.sweep_times_s)
        self._has_run = True

        # Build result reports.
        self.schedule_report = {
            "gamma": float(self._schedule.gamma),
            "sign": int(self._schedule.sign),
            "task_family": str(self._schedule.task_family),
            "T": int(self._schedule.T),
            "beta_used_t": [
                float(self._schedule.beta_used_at(t))
                for t in range(self._schedule.T)
            ],
        }

        self.clipping_summary = {
            "n_iters": int(self.n_iters),
            "iter_clip_fraction": [
                float(x) for x in self._iter_clip_fraction
            ],
            "iter_eff_discount_mean": [
                float(x) for x in self._iter_eff_discount_mean
            ],
            "overall_clip_fraction": (
                float(np.mean(self._iter_clip_fraction))
                if self._iter_clip_fraction
                else 0.0
            ),
        }

        return self

    def results(self) -> Dict:
        """
        Return a JSON-friendly summary of the run.

        Returns:
            Dict with classical MPI keys plus ``schedule_report`` and
            ``clipping_summary``.

        Raises:
            RuntimeError: if called before :meth:`run`.
        """
        if not self._has_run:
            raise RuntimeError(
                "SafeWeightedModifiedPolicyIteration.results() called before "
                "run(); call run() first."
            )
        V0 = self.V[0]  # (S,)
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
            "schedule_report": dict(self.schedule_report),
            "clipping_summary": dict(self.clipping_summary),
        }
