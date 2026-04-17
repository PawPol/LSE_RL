"""
Finite-horizon exact policy evaluation by backward induction.

Conventions
-----------
- Finite horizon ``T``: stages ``t in 0..T-1``.
- Terminal convention: ``V[T, :] = 0``.
- Deterministic policy: ``pi[t, s] in {0, ..., A-1}`` selects a single action.
- Bellman equation evaluated here (per stage ``t``):
  ``V^pi[t, s]    = R_bar[s, pi[t, s]]
                    + gamma * sum_{s'} P[s, pi[t, s], s'] * V^pi[t+1, s']``.
- After ``V^pi[t+1, :]`` is known, we additionally compute the *full*
  action-value table
  ``Q^pi[t, s, a]   = R_bar[s, a]
                      + gamma * sum_{s'} P[s, a, s'] * V^pi[t+1, s']``
  so downstream code can inspect ``Q^pi`` at the non-policy actions as well
  (useful for policy-improvement diagnostics). Note that
  ``Q^pi[t, s, pi[t, s]] == V^pi[t, s]`` by construction.

Residual convention
-------------------
Exact backward induction is a *single* pass, so the classical "between-sweep"
Bellman residual is not well defined. We record one scalar proxy instead:
``sup_norm(V[0], 0)`` — the magnitude of value propagated to stage ``0``.
This keeps the ``residuals`` field parallel to the iterative planners
(``value_iteration`` etc.) and makes logging uniform.
"""
from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

from mushroom_rl.algorithms.value.dp.finite_horizon_dp_utils import (
    allocate_value_tables,
    bellman_q_backup,
    bellman_q_policy_backup,
    expected_reward,
    extract_mdp_arrays,
    sup_norm_residual,
)


__all__ = ["ClassicalPolicyEvaluation"]


class ClassicalPolicyEvaluation:
    """
    Finite-horizon exact policy evaluation via backward induction.

    Given a fixed deterministic policy ``pi[t, s]`` (action index), computes
    ``Q[t, s, a]`` and ``V[t, s] = Q[t, s, pi[t, s]]`` by a single backward
    pass over the ``T`` stages (no iteration needed — the induction is exact).

    Attributes set by :meth:`run`:
        Q            : ``np.ndarray`` of shape ``(T, S, A)``, ``float64`` —
                       action-value table ``Q^pi[t, s, a]`` under ``pi``.
        V            : ``np.ndarray`` of shape ``(T + 1, S)``, ``float64`` —
                       value table ``V^pi``; ``V[T, :] = 0`` by construction.
        pi           : ``np.ndarray`` of shape ``(T, S)``, ``int64`` — the
                       evaluated policy (defensive copy of the input).
        residuals    : ``list[float]`` — one entry, ``sup_norm(V[0], 0)``.
        n_sweeps     : ``int`` — always ``1`` for exact backward induction.
        wall_clock_s : ``float`` — wall-clock seconds spent inside ``run``.

    Args:
        mdp: a :class:`mushroom_rl.environments.finite_mdp.FiniteMDP` (or any
            object exposing ``p``, ``r``, and ``info.{gamma, horizon}``).
        pi:  the deterministic policy to evaluate; shape ``(T, S)`` with
            integer entries in ``[0, A)``. A defensive copy is stored.
    """

    def __init__(
        self,
        mdp,
        pi: np.ndarray,
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

    def run(self) -> "ClassicalPolicyEvaluation":
        """
        Execute exact backward induction for the stored policy.

        Fills ``self.Q``, ``self.V``, ``self.residuals``, ``self.n_sweeps``,
        and ``self.wall_clock_s``. Safe to call multiple times; each call
        re-initialises the tables before running.

        Returns:
            ``self`` — for fluent chaining (e.g. ``pe = CPE(mdp, pi).run()``).
        """
        # Reset tables (so run() is idempotent).
        self.Q.fill(0.0)
        self.V.fill(0.0)
        self.residuals = []
        self.V_sweep_history = []

        t_start = time.perf_counter()

        # Backward induction: t = T-1, T-2, ..., 0.
        for t in range(self._T - 1, -1, -1):
            # 1) Policy-consistent Q entries at pi[t, s] — used to set V[t, :].
            #    Q_pi_t has zeros at non-policy actions by construction.
            Q_pi_t = bellman_q_policy_backup(                # (S, A)
                t=t,
                V=self.V,
                r_bar=self._r_bar,
                p=self._p,
                gamma=self._gamma,
                pi_t=self.pi[t],                             # (S,)
            )
            states = np.arange(self._S)
            # V^pi[t, s] = Q^pi[t, s, pi[t, s]].
            self.V[t] = Q_pi_t[states, self.pi[t]]           # (S,)

            # 2) Full Q^pi[t, s, a] for all a, using V^pi[t+1, :] as bootstrap.
            #    This is the standard Q^pi definition; the policy-action column
            #    matches Q_pi_t exactly (same r_bar + gamma * P @ V[t+1]).
            self.Q[t] = bellman_q_backup(                    # (S, A)
                t=t,
                V=self.V,
                r_bar=self._r_bar,
                p=self._p,
                gamma=self._gamma,
            )

        self.wall_clock_s = float(time.perf_counter() - t_start)
        self.n_sweeps = 1

        # Residual proxy: magnitude of value propagated to stage 0.
        zero_like_V0 = np.zeros_like(self.V[0])              # (S,)
        self.residuals = [sup_norm_residual(self.V[0], zero_like_V0)]

        self.V_sweep_history = [self.V.copy()]
        self._has_run = True
        return self

    def results(self) -> Dict:
        """
        Return a JSON-friendly summary of the run.

        Returns:
            Dict with keys:
                - ``Q_shape``    : ``list[int]`` — shape of ``self.Q``.
                - ``V_shape``    : ``list[int]`` — shape of ``self.V``.
                - ``n_sweeps``   : ``int`` — always ``1`` for exact PE.
                - ``wall_clock_s`` : ``float`` — seconds spent in ``run``.
                - ``residuals``  : ``list[float]`` — single-entry residual log.
                - ``V0_sup_norm``: ``float`` — ``sup_norm(V[0], 0)``.
                - ``V0_mean``    : ``float`` — mean of ``V[0, :]``.
                - ``V0_max``     : ``float`` — max of ``V[0, :]``.

        Raises:
            RuntimeError: if called before :meth:`run`.
        """
        if not self._has_run:
            raise RuntimeError(
                "ClassicalPolicyEvaluation.results() called before run(); "
                "call run() first."
            )
        V0 = self.V[0]                                       # (S,)
        return {
            "Q_shape": list(self.Q.shape),
            "V_shape": list(self.V.shape),
            "n_sweeps": int(self.n_sweeps),
            "wall_clock_s": float(self.wall_clock_s),
            "residuals": [float(x) for x in self.residuals],
            "V0_sup_norm": float(np.max(np.abs(V0))),
            "V0_mean": float(V0.mean()),
            "V0_max": float(V0.max()),
        }
