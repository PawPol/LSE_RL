"""Phase IV-C: Geometry-prioritized asynchronous DP planner.

Priority score (spec §5.1):
    geom_gain_t  = gamma * |rho_t - 1|   (deviation of safe discount from classical)
    priority(s,t) = |residual(s,t)| * (1 + lambda_geom * geom_gain_t
                                          + lambda_u  * |u_ref_stage_t|
                                          + lambda_kl * KL_Bern(rho_t, 0.5))

Uses the safe Bellman operator from SafeWeightedCommon for correctness.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[4]
_MUSHROOM_DEV = _REPO_ROOT / "mushroom-rl-dev"
for _p in (_REPO_ROOT, _MUSHROOM_DEV):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

__all__ = ["GeometryPriorityDP", "kl_bernoulli"]


def kl_bernoulli(
    p: np.ndarray,
    q: float = 0.5,
    eps: float = 1e-9,
) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    q_clipped = np.clip(q, eps, 1.0 - eps)
    return p * np.log(p / q_clipped) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q_clipped))


class GeometryPriorityDP:
    """Geometry-prioritized asynchronous DP for finite-horizon safe TAB MDPs."""

    def __init__(
        self,
        p: np.ndarray,              # [S, A, S']
        r: np.ndarray,              # [S, A]
        gamma: float,
        horizon: int,
        schedule_v3: dict[str, Any],
        lambda_geom: float = 1.0,
        lambda_u: float = 1.0,
        lambda_kl: float = 1.0,
        seed: int = 0,
    ) -> None:
        self._p = np.asarray(p, dtype=np.float64)    # (S, A, S)
        self._r = np.asarray(r, dtype=np.float64)    # (S, A)
        self._gamma = float(gamma)
        self._T = int(horizon)
        self._S = self._p.shape[0]
        self._A = self._p.shape[1]
        self._lambda_geom = lambda_geom
        self._lambda_u = lambda_u
        self._lambda_kl = lambda_kl
        self._rng = np.random.default_rng(seed)

        # Pre-compute expected rewards r_bar[s,a] = sum_s' P[s,a,s'] * R[s,a]
        self._r_bar = self._r  # deterministic tasks: r_bar = r

        # Extract schedule arrays
        self._beta_used_t = np.asarray(schedule_v3.get("beta_used_t", np.zeros(self._T)))
        self._alpha_t = np.asarray(schedule_v3.get("alpha_t", np.full(self._T, 0.05)))
        xi_ref = schedule_v3.get("xi_ref_t", None)
        if xi_ref is None:
            # Approximate from u_ref_t / beta_used_t
            xi_ref = np.ones(self._T)
        self._xi_ref_t = np.asarray(xi_ref)

        # Pre-compute per-stage geometry diagnostics (frozen, pilot-estimated)
        log_gamma = float(np.log(self._gamma))
        log_1pg = float(np.log(1.0 + self._gamma))
        self._rho_t = np.zeros(self._T)
        self._geom_gain_t = np.zeros(self._T)
        self._u_ref_t = np.zeros(self._T)
        self._kl_t = np.zeros(self._T)

        for t in range(self._T):
            beta = float(self._beta_used_t[t])
            xi = float(self._xi_ref_t[t]) if self._xi_ref_t.ndim > 0 else 0.1
            # rho_t = sigmoid(beta * xi + log(1/gamma)) — using xi as representative margin
            if abs(beta) < 1e-10:
                rho = 1.0 / (1.0 + self._gamma)
            else:
                x = beta * xi + np.log(1.0 / self._gamma)
                rho = 1.0 / (1.0 + np.exp(-x))
            self._rho_t[t] = rho
            # spec §5.1: geom_gain = |effective_discount_used - gamma_base|
            # effective_discount = rho * gamma, so = |rho*gamma - gamma| = gamma*|rho - 1|
            eff_discount = rho * self._gamma
            self._geom_gain_t[t] = abs(eff_discount - self._gamma)
            self._u_ref_t[t] = abs(beta * xi)
            # spec §5.1: KL_Bern(rho_used || p0) where p0 = 1/(1+gamma) is the beta=0 prior
            p0 = 1.0 / (1.0 + self._gamma)
            self._kl_t[t] = float(kl_bernoulli(np.array([rho]), q=p0)[0])

    def _safe_q_batch(self, V_next: np.ndarray, t: int) -> np.ndarray:
        """Compute Q[s,a] = sum_s' P[s,a,s'] * g_t^safe(r[s,a], V_next[s'])."""
        beta = float(self._beta_used_t[t])
        V_next = np.asarray(V_next, dtype=np.float64)
        r_bar = self._r_bar

        if abs(beta) < 1e-10:
            # Classical
            E_v = np.einsum("ijk,k->ij", self._p, V_next)  # (S, A)
            return r_bar + self._gamma * E_v

        log_gamma = float(np.log(self._gamma))
        log_1pg = float(np.log(1.0 + self._gamma))
        r_3d = r_bar[:, :, np.newaxis]                      # (S, A, 1)
        v_3d = V_next[np.newaxis, np.newaxis, :]            # (1, 1, S)
        log_sum = np.logaddexp(beta * r_3d, beta * v_3d + log_gamma)  # (S, A, S)
        g_3d = (1.0 + self._gamma) / beta * (log_sum - log_1pg)      # (S, A, S)
        Q = np.einsum("ijk,ijk->ij", self._p, g_3d)                   # (S, A)
        return Q

    def _residual(self, V: np.ndarray, t: int) -> np.ndarray:
        """Per-state Bellman residual |V[s] - max_a Q[s,a]| at stage t."""
        if t == self._T - 1:
            V_next = np.zeros(self._S)  # terminal boundary
        else:
            V_next = V[t + 1]
        Q = self._safe_q_batch(V_next, t)
        V_new = np.max(Q, axis=1)
        return np.abs(V[t] - V_new), Q

    def _priority_scores(self, residuals: np.ndarray) -> np.ndarray:
        """Compute priority[s, t] = |residual[t, s]| * scaling."""
        scores = np.zeros((self._T, self._S))
        for t in range(self._T):
            g_gain = self._geom_gain_t[t]
            u_ref = self._u_ref_t[t]
            kl = self._kl_t[t]
            scale = 1.0 + self._lambda_geom * g_gain + self._lambda_u * u_ref + self._lambda_kl * kl
            scores[t] = residuals[t] * scale
        return scores

    def plan(
        self,
        tol: float = 1e-6,
        max_sweeps: int = 500,
        top_k_fraction: float = 0.25,
        log_backups: bool = False,
    ) -> dict[str, Any]:
        """Run geometry-priority async VI until convergence.

        Parameters
        ----------
        log_backups : bool
            If True, accumulate per-backup records (spec §8.4) in
            ``result["backup_log"]``. Each record is a dict with keys
            sweep, rank, stage, state, residual, priority, geom_gain, kl.
            Disabled by default to avoid memory overhead in sweeps with
            many (s,t) pairs.
        """
        t0 = time.perf_counter()
        V = np.zeros((self._T + 1, self._S))
        residual_history = []
        n_backups = 0
        high_act_backups = 0
        convergence_sweep_1e2 = None
        backup_log: list[dict[str, Any]] = []

        for sweep in range(max_sweeps):
            max_residual = 0.0
            # Compute all residuals and priorities
            all_residuals = np.zeros((self._T, self._S))
            for t in range(self._T - 1, -1, -1):
                res, _ = self._residual(V, t)
                all_residuals[t] = res
                max_residual = max(max_residual, float(np.max(res)))

            # Priority-based update order
            priority = self._priority_scores(all_residuals)
            k = max(1, int(top_k_fraction * self._T * self._S))
            flat_idx = np.argsort(priority.ravel())[::-1][:k]
            ts, ss = np.unravel_index(flat_idx, (self._T, self._S))

            # Apply updates for top-k (s, t) pairs
            for rank, (t, s) in enumerate(zip(ts, ss)):
                t_idx = int(t)
                s_idx = int(s)
                if t_idx == self._T - 1:
                    V_next = np.zeros(self._S)
                else:
                    V_next = V[t_idx + 1]
                Q = self._safe_q_batch(V_next, t_idx)
                V[t_idx, s_idx] = float(np.max(Q[s_idx]))
                n_backups += 1
                if self._u_ref_t[t_idx] >= 5e-3:
                    high_act_backups += 1
                if log_backups:
                    backup_log.append({
                        "sweep": sweep,
                        "rank": rank,
                        "stage": t_idx,
                        "state": s_idx,
                        "residual": float(all_residuals[t_idx, s_idx]),
                        "priority": float(priority[t_idx, s_idx]),
                        "geom_gain": float(self._geom_gain_t[t_idx]),
                        "kl": float(self._kl_t[t_idx]),
                    })

            residual_history.append(float(max_residual))
            if max_residual < 1e-2 and convergence_sweep_1e2 is None:
                convergence_sweep_1e2 = sweep + 1

            if max_residual < tol:
                break

        # Compute final Q
        Q_final = np.zeros((self._T, self._S, self._A))
        for t in range(self._T - 1, -1, -1):
            V_next = V[t + 1] if t < self._T - 1 else np.zeros(self._S)
            Q_final[t] = self._safe_q_batch(V_next, t)

        elapsed = time.perf_counter() - t0
        frac_high = high_act_backups / max(n_backups, 1)

        result: dict[str, Any] = {
            "V": V[:self._T],
            "Q": Q_final,
            "n_sweeps": sweep + 1,
            "n_backups": n_backups,
            "residual_history": residual_history,
            "final_residual": residual_history[-1] if residual_history else 0.0,
            "convergence_sweep_1e-2": convergence_sweep_1e2,
            "geom_gain_per_stage": self._geom_gain_t.tolist(),
            "u_ref_per_stage": self._u_ref_t.tolist(),
            "frac_high_activation_backups": frac_high,
            "wall_clock_s": elapsed,
            "lambda_geom": self._lambda_geom,
            "lambda_u": self._lambda_u,
            "lambda_kl": self._lambda_kl,
        }
        if log_backups:
            result["backup_log"] = backup_log
        return result
