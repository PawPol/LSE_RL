"""
Finite-horizon safe weighted-LSE policy evaluation by backward induction.

This module mirrors :mod:`classical_policy_evaluation` but replaces the
standard Bellman backup with the safe weighted-LSE operator from
:class:`SafeWeightedCommon`.

Conventions
-----------
- Finite horizon ``T``: stages ``t in 0..T-1``.
- Terminal convention: ``V[T, :] = 0``.
- Deterministic policy: ``pi[t, s] in {0, ..., A-1}`` selects a single action.
- The safe one-step target at stage ``t`` is

      g_t^safe(r, v) = ((1+gamma) / beta_t) *
          [log(exp(beta_t * r) + gamma * exp(beta_t * v)) - log(1+gamma)]

  implemented via ``np.logaddexp`` to avoid overflow/underflow.  When
  ``beta_used == 0`` the classical target ``r + gamma * v`` is returned
  exactly (no logaddexp call).

Classical recovery guarantee
----------------------------
When ``beta_used == 0`` at every stage (e.g. ``BetaSchedule.zeros(T, gamma)``),
the safe evaluator produces bit-identical results to
:class:`ClassicalPolicyEvaluation` within numerical tolerance.

Residual convention
-------------------
Exact backward induction is a *single* pass, so the classical "between-sweep"
Bellman residual is not well defined. We record one scalar proxy instead:
``sup_norm(V[0], 0)`` -- the magnitude of value propagated to stage ``0``.
This keeps the ``residuals`` field parallel to the iterative planners.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

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

__all__ = ["SafeWeightedPolicyEvaluation"]


class SafeWeightedPolicyEvaluation:
    """
    Finite-horizon safe weighted-LSE policy evaluation via backward induction.

    Given a fixed deterministic policy ``pi[t, s]`` (action index) and a
    :class:`BetaSchedule`, computes ``Q[t, s, a]`` and
    ``V[t, s] = Q[t, s, pi[t, s]]`` by a single backward pass over the
    ``T`` stages using the safe weighted-LSE Bellman operator.

    When the schedule has ``beta_used == 0`` at every stage, this evaluator
    produces results bit-identical (within numerical tolerance) to
    :class:`ClassicalPolicyEvaluation`.

    Attributes set by :meth:`run`:
        Q                : ``np.ndarray`` of shape ``(T, S, A)``, ``float64`` --
                           action-value table ``Q^{safe,pi}[t, s, a]``.
        V                : ``np.ndarray`` of shape ``(T + 1, S)``, ``float64`` --
                           value table ``V^{safe,pi}``; ``V[T, :] = 0``.
        pi               : ``np.ndarray`` of shape ``(T, S)``, ``int64`` -- the
                           evaluated policy (defensive copy of the input).
        residuals        : ``list[float]`` -- one entry, ``sup_norm(V[0], 0)``.
        n_sweeps         : ``int`` -- always ``1`` for exact backward induction.
        wall_clock_s     : ``float`` -- wall-clock seconds spent inside ``run``.
        V_sweep_history  : ``list[np.ndarray]`` -- one entry (the final V table).
        schedule_report  : ``dict`` -- schedule metadata, set at ``run()`` start.
        clipping_summary : ``dict`` -- per-stage clipping diagnostics, set at
                           ``run()`` end.

    Args:
        mdp: a :class:`mushroom_rl.environments.finite_mdp.FiniteMDP` (or any
            object exposing ``p``, ``r``, and ``info.{gamma, horizon}``).
        pi:  the deterministic policy to evaluate; shape ``(T, S)`` with
            integer entries in ``[0, A)``. A defensive copy is stored.
        schedule: a :class:`BetaSchedule` providing per-stage beta values
            and certification metadata.
    """

    def __init__(
        self,
        mdp,
        pi: np.ndarray,
        schedule: BetaSchedule,
    ) -> None:
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

        # Validate and defensively copy the input policy.
        pi_arr = np.asarray(pi, dtype=np.int64)
        expected_shape = (self._T, self._S)
        if pi_arr.shape != expected_shape:
            raise ValueError(
                f"pi must have shape (T, S) = {expected_shape}; "
                f"got {pi_arr.shape}."
            )
        if np.any(pi_arr < 0) or np.any(pi_arr >= self._A):
            raise ValueError(
                f"pi entries must lie in [0, {self._A}); "
                f"got min={int(pi_arr.min())}, max={int(pi_arr.max())}."
            )
        self.pi: np.ndarray = pi_arr.copy()              # (T, S)

        # Safe weighted-LSE helper.
        self._schedule = schedule
        self._safe = SafeWeightedCommon(
            schedule, gamma=self._gamma, n_base=self._S
        )

        # Result tables, allocated up-front (zeroed). Terminal V[T, :] = 0.
        Q0, V0, _ = allocate_value_tables(self._S, self._A, self._T)
        self.Q: np.ndarray = Q0                          # (T, S, A)
        self.V: np.ndarray = V0                          # (T + 1, S)

        # Timing / logging scaffolding.
        self.residuals: List[float] = []
        self.n_sweeps: int = 0
        self.wall_clock_s: float = 0.0
        self._has_run: bool = False
        #: One entry (shape ``(T+1, S)``): value table after exact evaluation.
        self.V_sweep_history: List[np.ndarray] = []

        # Per-stage diagnostics (populated during run).
        self._stage_clip_active: List[bool] = []
        self._stage_beta_raw: List[float] = []
        self._stage_beta_cap: List[float] = []
        self._stage_beta_used: List[float] = []
        self._stage_eff_discount_mean: List[float] = []
        self._stage_eff_discount_std: List[float] = []
        self._stage_frac_eff_lt_gamma: List[float] = []

        # Summary dicts (set during run).
        self.schedule_report: Dict[str, Any] = {}
        self.clipping_summary: Dict[str, Any] = {}

    def run(self) -> "SafeWeightedPolicyEvaluation":
        """
        Execute safe weighted-LSE backward induction for the stored policy.

        Fills ``self.Q``, ``self.V``, ``self.residuals``, ``self.n_sweeps``,
        ``self.wall_clock_s``, ``self.schedule_report``, and
        ``self.clipping_summary``. Safe to call multiple times; each call
        re-initialises the tables before running.

        Returns:
            ``self`` -- for fluent chaining.
        """
        # Reset tables (so run() is idempotent).
        self.Q.fill(0.0)
        self.V.fill(0.0)
        self.residuals = []
        self.V_sweep_history = []

        # Reset per-stage diagnostics.
        self._stage_clip_active = []
        self._stage_beta_raw = []
        self._stage_beta_cap = []
        self._stage_beta_used = []
        self._stage_eff_discount_mean = []
        self._stage_eff_discount_std = []
        self._stage_frac_eff_lt_gamma = []

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

        t_start = time.perf_counter()
        states = np.arange(self._S)

        # Backward induction: t = T-1, T-2, ..., 0.
        for t in range(self._T - 1, -1, -1):
            # Step 1: expected next-state value for all (s, a).
            # E_v_next[s, a] = sum_{s'} P[s, a, s'] * V[t+1, s']
            E_v_next = np.einsum(
                "ijk,k->ij", self._p, self.V[t + 1]
            )  # (S, A)

            # Step 2: safe Q table for all actions via the safe operator.
            Q_safe = self._safe.compute_safe_target_batch(
                self._r_bar, E_v_next, t
            )  # (S, A)

            # Step 3: store Q and extract policy-consistent V.
            self.Q[t] = Q_safe
            self.V[t] = Q_safe[states, self.pi[t]]  # (S,)

            # Collect per-stage diagnostics from the safe common object.
            self._stage_clip_active.append(bool(self._safe.last_clip_active))
            self._stage_beta_raw.append(float(self._safe.last_beta_raw))
            self._stage_beta_cap.append(float(self._safe.last_beta_cap))
            self._stage_beta_used.append(float(self._safe.last_beta_used))

            eff_d = np.asarray(self._safe.last_effective_discount)
            self._stage_eff_discount_mean.append(float(eff_d.mean()))
            self._stage_eff_discount_std.append(float(eff_d.std()))
            self._stage_frac_eff_lt_gamma.append(
                float((eff_d < self._gamma).mean())
            )

        # The lists were appended in backward order (T-1, T-2, ..., 0).
        # Reverse so that index 0 corresponds to stage 0.
        self._stage_clip_active.reverse()
        self._stage_beta_raw.reverse()
        self._stage_beta_cap.reverse()
        self._stage_beta_used.reverse()
        self._stage_eff_discount_mean.reverse()
        self._stage_eff_discount_std.reverse()
        self._stage_frac_eff_lt_gamma.reverse()

        self.wall_clock_s = float(time.perf_counter() - t_start)
        self.n_sweeps = 1

        # Residual proxy: magnitude of value propagated to stage 0.
        zero_like_V0 = np.zeros_like(self.V[0])  # (S,)
        self.residuals = [sup_norm_residual(self.V[0], zero_like_V0)]

        self.V_sweep_history = [self.V.copy()]

        # Build clipping summary.
        self.clipping_summary = {
            "n_stages_clipped": int(sum(self._stage_clip_active)),
            "clip_fraction": float(sum(self._stage_clip_active)) / self._T,
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

        self._has_run = True
        return self

    def results(self) -> Dict[str, Any]:
        """
        Return a JSON-friendly summary of the run.

        Returns:
            Dict with keys:
                - ``Q_shape``           : ``list[int]`` -- shape of ``self.Q``.
                - ``V_shape``           : ``list[int]`` -- shape of ``self.V``.
                - ``n_sweeps``          : ``int`` -- always ``1`` for exact PE.
                - ``wall_clock_s``      : ``float`` -- seconds spent in ``run``.
                - ``residuals``         : ``list[float]`` -- single-entry residual.
                - ``V0_sup_norm``       : ``float`` -- ``sup_norm(V[0], 0)``.
                - ``V0_mean``           : ``float`` -- mean of ``V[0, :]``.
                - ``V0_max``            : ``float`` -- max of ``V[0, :]``.
                - ``schedule_report``   : ``dict`` -- schedule metadata.
                - ``clipping_summary``  : ``dict`` -- per-stage clipping info.

        Raises:
            RuntimeError: if called before :meth:`run`.
        """
        if not self._has_run:
            raise RuntimeError(
                "SafeWeightedPolicyEvaluation.results() called before run(); "
                "call run() first."
            )
        V0 = self.V[0]  # (S,)
        return {
            "Q_shape": list(self.Q.shape),
            "V_shape": list(self.V.shape),
            "n_sweeps": int(self.n_sweeps),
            "wall_clock_s": float(self.wall_clock_s),
            "residuals": [float(x) for x in self.residuals],
            "V0_sup_norm": float(np.max(np.abs(V0))),
            "V0_mean": float(V0.mean()),
            "V0_max": float(V0.max()),
            "schedule_report": self.schedule_report,
            "clipping_summary": self.clipping_summary,
        }
