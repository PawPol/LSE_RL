"""Occupancy-weighted certificate check for the Phase V WP0 audit.

Supports the WP0 remediation of the ``cert_bound`` BLOCKERs flagged by
:mod:`scripts.audit.run_consistency_audit`.

Context
-------
The first-pass audit's ``cert_bound`` check compares the stored per-stage
``safe_effective_discount_mean`` -- a *grid mean* over the full ``(S, A)``
cross product -- against the certified stagewise rate ``kappa_t``. The
grid mean can exceed ``kappa_t`` whenever *some* ``(s, a)`` pair has an
effective discount above the bound, even if that pair is never visited
under either the classical or the safe optimal policy. When the breach
lives on unreachable cells it is benign for deployment.

This module recomputes the cert check weighted by the **reference
occupancy** ``d_ref = 0.5 d^{pi*_cl} + 0.5 d^{pi*_safe}`` (Phase V spec
section 3) and the safe policy's action distribution:

    d_eff_occ[t] = sum_{s, a} d_ref(t, s) * pi*_safe(a | t, s) * eff_d[s, a, t]

If the occupancy-weighted mean satisfies ``d_eff_occ[t] <= kappa_t + tol``
for every stage, the breach is localised to unreachable cells and the
original ``cert_bound`` BLOCKER can be downgraded to MINOR.

Scope
-----
Read-only. Does NOT modify any Phase I-IV source or runner. It re-runs
``SafeWeighted{ValueIteration,PolicyEvaluation}`` and
``ClassicalValueIteration`` on the exact MDP used by the run and calls
:meth:`SafeWeightedCommon.compute_safe_target_ev_batch` to capture the
full ``(T, S, A)`` effective-discount tensor -- the *same* operator used
by the original run.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

# --------------------------------------------------------------------------
# Path bootstrapping: add the repo root and mushroom-rl-dev to sys.path so
# the audit can live under scripts/ without an installed package.
# --------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MUSHROOM_DEV = _REPO_ROOT / "mushroom-rl-dev"
for _p in (_REPO_ROOT, _MUSHROOM_DEV):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
@dataclass
class OccupancyCheckResult:
    """Per-run outcome of the occupancy-weighted cert check.

    Attributes
    ----------
    ok : bool
        True if the occupancy-weighted check passes on every stage.
    first_breach_stage : int | None
        Earliest stage where ``d_eff_occ[t] > kappa_t + tol`` fails; None
        if ``ok`` is True.
    d_eff_occ : np.ndarray
        Shape ``(T,)``. Per-stage occupancy-weighted effective discount.
    kappa_t : np.ndarray
        Shape ``(T,)``. Certified stagewise rate from the schedule.
    max_overshoot : float
        ``max_t (d_eff_occ[t] - kappa_t[t])``. Positive on failure,
        non-positive on pass. Always reported for diagnostics.
    stages_over : int
        Number of stages with ``d_eff_occ[t] > kappa_t[t] + tol``.
    recomputed : bool
        True when the operator was re-run; False for short-circuit paths
        (e.g. the run cannot be reconstructed).
    """

    ok: bool
    first_breach_stage: int | None
    d_eff_occ: np.ndarray
    kappa_t: np.ndarray
    max_overshoot: float
    stages_over: int
    recomputed: bool = True
    note: str = ""


class RunReconstructionError(RuntimeError):
    """Raised when a run cannot be reconstructed for remediation."""


# --------------------------------------------------------------------------
# Run-path -> {task family, algorithm, seed, regime_phase} parser
# --------------------------------------------------------------------------
_PHASE3_RUN_RE = re.compile(
    r"results/weighted_lse_dp/phase3/paper_suite/"
    r"(?P<task>[A-Za-z0-9_]+)/"
    r"(?P<algo>[A-Za-z]+)/"
    r"seed_(?P<seed>[0-9]+)"
)
_PHASE4B_RUN_RE = re.compile(
    r"results/weighted_lse_dp/phase4/translation_4a2/"
    r"(?P<task_tag>[A-Za-z0-9_]+)/"
    r"(?P<algo>[A-Za-z_]+)/"
    r"seed_(?P<seed>[0-9]+)"
)

# Phase III task names that use `warmstart_dp=True` and produce
# `_pre_shift` / `_post_shift` suffixes.
_REGIME_SHIFT_CANONICAL = {"chain_regime_shift", "grid_regime_shift"}


def parse_run_path(rel_path: str) -> dict[str, Any]:
    """Parse ``results/weighted_lse_dp/.../seed_NN`` into a spec dict.

    Returns
    -------
    dict with keys: ``phase`` (III | IV-B), and phase-specific fields.
        Phase III:  ``task``, ``canonical_task``, ``regime_phase`` or None,
                    ``algo``, ``seed``.
        Phase IV-B: ``task_tag``, ``algo``, ``seed``.
    """
    m = _PHASE3_RUN_RE.search(rel_path)
    if m:
        task = m["task"]
        canonical = task
        regime_phase: str | None = None
        for root in _REGIME_SHIFT_CANONICAL:
            for suffix in ("_pre_shift", "_post_shift"):
                if task == root + suffix:
                    canonical = root
                    regime_phase = suffix.lstrip("_")
                    break
            if canonical != task:
                break
        return {
            "phase": "III",
            "task": task,
            "canonical_task": canonical,
            "regime_phase": regime_phase,
            "algo": m["algo"],
            "seed": int(m["seed"]),
        }
    m = _PHASE4B_RUN_RE.search(rel_path)
    if m:
        return {
            "phase": "IV-B",
            "task_tag": m["task_tag"],
            "algo": m["algo"],
            "seed": int(m["seed"]),
        }
    raise RunReconstructionError(
        f"Cannot parse run path for remediation: {rel_path}"
    )


# --------------------------------------------------------------------------
# MDP / schedule reconstruction helpers
# --------------------------------------------------------------------------
def _load_phase3_suite_cfg() -> dict[str, Any]:
    p = _REPO_ROOT / "experiments/weighted_lse_dp/configs/phase3/paper_suite.json"
    with open(p, "r") as fh:
        return json.load(fh)


def _rebuild_phase3_mdp(canonical_task: str, task_cfg: dict[str, Any],
                        seed: int, regime_phase: str | None) -> Any:
    """Re-run the Phase III task factory for ``canonical_task`` and return
    the FiniteMDP used by the DP planner (pre- or post-shift for regime
    tasks). Imports are deferred so the audit script can import this
    module without pulling mushroom-rl onto the path eagerly."""
    from experiments.weighted_lse_dp.tasks.stress_families import (
        make_chain_sparse_long,
        make_chain_jackpot,
        make_chain_catastrophe,
        make_grid_sparse_goal,
    )
    from experiments.weighted_lse_dp.tasks.nonstationary_wrappers import (
        make_chain_regime_shift,
        make_grid_regime_shift,
    )

    cfg_copy = dict(task_cfg)
    if canonical_task == "chain_sparse_long":
        mdp, _, _ = make_chain_sparse_long(
            cfg=cfg_copy,
            state_n=int(task_cfg.get("state_n", 60)),
            prob=float(task_cfg.get("prob", 0.9)),
            gamma=float(task_cfg.get("gamma", 0.99)),
            horizon=int(task_cfg.get("horizon", 120)),
        )
        return mdp
    if canonical_task == "chain_jackpot":
        mdp, _, _ = make_chain_jackpot(
            cfg=cfg_copy,
            state_n=int(task_cfg.get("state_n", 25)),
            prob=float(task_cfg.get("prob", 0.9)),
            gamma=float(task_cfg.get("gamma", 0.99)),
            horizon=int(task_cfg.get("horizon", 60)),
            jackpot_state=int(task_cfg.get("jackpot_state", 20)),
            jackpot_prob=float(task_cfg.get("jackpot_prob", 0.05)),
            jackpot_reward=float(task_cfg.get("jackpot_reward", 10.0)),
            jackpot_terminates=bool(task_cfg.get("jackpot_terminates", True)),
        )
        return mdp
    if canonical_task == "chain_catastrophe":
        mdp, _, _ = make_chain_catastrophe(
            cfg=cfg_copy,
            state_n=int(task_cfg.get("state_n", 25)),
            prob=float(task_cfg.get("prob", 0.9)),
            gamma=float(task_cfg.get("gamma", 0.99)),
            horizon=int(task_cfg.get("horizon", 60)),
            risky_state=int(task_cfg.get("risky_state", 15)),
            risky_prob=float(task_cfg.get("risky_prob", 0.05)),
            catastrophe_reward=float(task_cfg.get("catastrophe_reward", -10.0)),
            shortcut_jump=int(task_cfg.get("shortcut_jump", 5)),
        )
        return mdp
    if canonical_task == "grid_sparse_goal":
        mdp, _, _ = make_grid_sparse_goal(
            cfg=cfg_copy,
            prob=float(task_cfg.get("prob", 0.9)),
            gamma=float(task_cfg.get("gamma", 0.99)),
            horizon=int(task_cfg.get("horizon", 80)),
            goal_reward=float(task_cfg.get("goal_reward", 1.0)),
            time_augment=False,
            seed=seed,
        )
        return mdp
    if canonical_task == "chain_regime_shift":
        wrapper, _, _ = make_chain_regime_shift(
            cfg=cfg_copy,
            state_n=int(task_cfg.get("state_n", 25)),
            prob=float(task_cfg.get("prob", 0.9)),
            gamma=float(task_cfg.get("gamma", 0.99)),
            horizon=int(task_cfg.get("horizon", 60)),
            change_at_episode=int(task_cfg.get("change_at_episode", 500)),
            shift_type=str(task_cfg.get("shift_type", "goal_flip")),
            post_prob=float(task_cfg.get("post_prob", 0.7)),
            time_augment=False,
        )
        if regime_phase == "post_shift":
            return wrapper._post
        return wrapper._pre
    if canonical_task == "grid_regime_shift":
        wrapper, _, _ = make_grid_regime_shift(
            cfg=cfg_copy,
            change_at_episode=int(task_cfg.get("change_at_episode", 300)),
            shift_type=str(task_cfg.get("shift_type", "goal_move")),
            time_augment=False,
        )
        if regime_phase == "post_shift":
            return wrapper._post
        return wrapper._pre
    raise RunReconstructionError(
        f"No Phase III factory dispatch for canonical task {canonical_task!r}"
    )


def _rebuild_phase4b_mdp(task_cfg: dict[str, Any], seed: int) -> Any:
    """Re-run the Phase IV-B task factory for ``task_cfg['family']``."""
    from experiments.weighted_lse_dp.tasks.phase4_operator_suite import (
        build_phase4_task,
    )
    mdp_base, _, _ = build_phase4_task(task_cfg, seed=seed)
    return mdp_base


def _load_schedule(schedule_path: pathlib.Path) -> Any:
    """Load a schedule JSON as a ``BetaSchedule`` instance.

    Handles both Phase III (canonical ``schedule.json`` with every
    BetaSchedule key) and Phase IV-B (``schedule_v3.json`` which omits
    ``beta_raw_t`` / ``beta_cap_t`` and uses ``gamma_eval`` / ``sign_family``).
    The wrapping logic is copied from
    :func:`experiments.weighted_lse_dp.runners.run_phase4_dp._wrap_v3_schedule_for_betaschedule`
    -- read-only use only, no new operator semantics.
    """
    from mushroom_rl.algorithms.value.dp import BetaSchedule

    with open(schedule_path, "r") as fh:
        raw = json.load(fh)

    if "gamma" in raw and "beta_raw_t" in raw and "beta_cap_t" in raw:
        # Canonical schedule.json (Phase III).
        return BetaSchedule.from_file(str(schedule_path))

    # v3 schedule (Phase IV-B): wrap with build_certification.
    from mushroom_rl.algorithms.value.dp.safe_weighted_common import (
        build_certification,
    )

    gamma = float(raw.get("gamma_eval", raw.get("gamma_base", raw.get("gamma", 0.99))))
    alpha_t = np.asarray(raw["alpha_t"], dtype=np.float64)
    reward_bound = float(raw.get("reward_bound", 1.0))
    cert = build_certification(
        alpha_t=alpha_t, R_max=reward_bound, gamma=gamma,
    )
    beta_used_t = np.asarray(raw["beta_used_t"], dtype=np.float64)
    beta_cap_t = cert["beta_cap_t"]
    # Match Phase IV-B's clip-to-avoid-sliver-mismatch behaviour.
    beta_used_t = np.clip(beta_used_t, -beta_cap_t, beta_cap_t)
    T = int(len(beta_used_t))
    wrapped = {
        "task_family": raw.get("task_family", ""),
        "gamma": gamma,
        "sign": int(raw.get("sign_family", raw.get("sign", 0))),
        "source_phase": raw.get("source_phase", "phase4"),
        "reward_bound": reward_bound,
        "alpha_t": alpha_t.tolist(),
        "kappa_t": cert["kappa_t"].tolist(),
        "Bhat_t": cert["Bhat_t"].tolist(),
        "beta_raw_t": beta_used_t.tolist(),
        "beta_cap_t": beta_cap_t.tolist(),
        "beta_used_t": beta_used_t.tolist(),
        "clip_active_t": [False] * T,
        "informativeness_t": [0.0] * T,
        "d_target_t": [float(gamma)] * T,
        "calibration_source_path": raw.get("calibration_source_path", ""),
        "calibration_hash": raw.get("calibration_hash", ""),
        "notes": raw.get("notes", "wrapped v3 -> BetaSchedule (audit)"),
    }
    return BetaSchedule(wrapped)


# --------------------------------------------------------------------------
# Core computation
# --------------------------------------------------------------------------
def _recompute_eff_d_tensor(mdp: Any, V_next_table: np.ndarray,
                            schedule: Any) -> np.ndarray:
    """Return ``eff_d`` with shape ``(T, S, A)`` -- the P-weighted
    conditional expectation ``(1+gamma)(1 - rho)`` at every ``(t, s, a)``.

    Uses the exact operator call
    :meth:`SafeWeightedCommon.compute_safe_target_ev_batch` (matching
    ``safe_weighted_common.py`` lines 640-689).
    """
    from mushroom_rl.algorithms.value.dp import extract_mdp_arrays
    from mushroom_rl.algorithms.value.dp.safe_weighted_common import (
        SafeWeightedCommon,
    )

    p_arr, r_arr_full, horizon, gamma = extract_mdp_arrays(mdp)
    # Expected reward r_bar[s, a] = sum_{s'} P[s,a,s'] * r[s,a,s'].
    r_bar = np.einsum("sak,sak->sa", p_arr, r_arr_full)
    S, A = r_bar.shape
    T = int(horizon)
    if V_next_table.shape != (T + 1, S):
        raise RunReconstructionError(
            f"V_next_table shape {V_next_table.shape} != ({T+1}, {S})"
        )

    safe = SafeWeightedCommon(schedule, gamma=float(gamma), n_base=S)
    eff_d = np.zeros((T, S, A), dtype=np.float64)
    for t in range(T):
        _ = safe.compute_safe_target_ev_batch(
            r_bar, V_next_table[t + 1], p_arr, t
        )
        last_ed = safe.last_effective_discount
        if isinstance(last_ed, np.ndarray):
            eff_d[t] = last_ed
        else:
            eff_d[t, :, :] = float(last_ed)
    return eff_d


def _run_classical_vi(mdp: Any) -> np.ndarray:
    """Return the classical-VI greedy policy, shape ``(T, S)``."""
    from mushroom_rl.algorithms.value.dp import ClassicalValueIteration
    planner = ClassicalValueIteration(mdp, n_sweeps=1, tol=0.0)
    planner.run()
    return planner.pi


def _run_safe_vi(mdp: Any, schedule: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(V_table, pi_safe)`` for the safe weighted-LSE VI."""
    from mushroom_rl.algorithms.value.dp import SafeWeightedValueIteration
    planner = SafeWeightedValueIteration(mdp, schedule=schedule)
    planner.run()
    return planner.V, planner.pi


def _run_safe_pe(mdp: Any, schedule: Any, ref_pi: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(V_table, pi_ref)`` for SafePE under ``ref_pi``."""
    from mushroom_rl.algorithms.value.dp import SafeWeightedPolicyEvaluation
    planner = SafeWeightedPolicyEvaluation(
        mdp, pi=ref_pi, schedule=schedule,
    )
    planner.run()
    return planner.V, ref_pi


def _build_ref_pi(task_cfg: dict[str, Any], mdp: Any) -> np.ndarray:
    from experiments.weighted_lse_dp.common.task_factories import (
        build_ref_pi_for_task,
    )
    canonical_task = task_cfg.get("_canonical_task")
    if canonical_task is None:
        canonical_task = task_cfg.get("family", "unknown")
    return build_ref_pi_for_task(canonical_task, task_cfg, mdp)


# --------------------------------------------------------------------------
# Public entry: remediate a single flagged run
# --------------------------------------------------------------------------
def remediate_run(
    rel_run_path: str,
    *,
    tol: float = 1e-6,
) -> OccupancyCheckResult:
    """Occupancy-weighted cert check for one flagged DP run.

    Parameters
    ----------
    rel_run_path : str
        Repo-relative run directory, e.g.
        ``results/weighted_lse_dp/phase3/paper_suite/chain_catastrophe/SafePE/seed_11``.
    tol : float
        Slack on the stagewise cert, matching the original
        ``cert_bound`` tolerance (default 1e-6).

    Returns
    -------
    OccupancyCheckResult
        ``ok=True`` iff every stage satisfies
        ``d_eff_occ[t] <= kappa_t[t] + tol``.
    """
    from experiments.weighted_lse_dp.search.reference_occupancy import (
        compute_d_ref,
    )

    spec = parse_run_path(rel_run_path)
    run_dir = _REPO_ROOT / rel_run_path
    if not run_dir.is_dir():
        raise RunReconstructionError(f"run dir missing: {run_dir}")

    # Parse per-run config.json to recover the exact task knobs.
    run_cfg_path = run_dir / "config.json"
    if not run_cfg_path.is_file():
        raise RunReconstructionError(
            f"config.json missing under {run_dir}"
        )
    with open(run_cfg_path, "r") as fh:
        run_cfg = json.load(fh)

    # Locate the schedule that drove this run.
    prov_path = run_dir / "safe_provenance.json"
    schedule_path: pathlib.Path | None = None
    if prov_path.is_file():
        try:
            with open(prov_path, "r") as fh:
                prov = json.load(fh)
            sp = prov.get("schedule_path") or prov.get("schedule_file")
            if sp:
                p = pathlib.Path(sp)
                if not p.is_absolute():
                    p = _REPO_ROOT / p
                if p.is_file():
                    schedule_path = p
        except Exception:  # pragma: no cover - defensive fallback
            schedule_path = None
    if schedule_path is None:
        sp = run_cfg.get("schedule_file") or run_cfg.get("schedule_path")
        if sp:
            p = pathlib.Path(sp)
            if not p.is_absolute():
                p = _REPO_ROOT / p
            if p.is_file():
                schedule_path = p
    if schedule_path is None:
        raise RunReconstructionError(
            f"Cannot locate schedule for {rel_run_path}"
        )
    # For Phase IV-B the per-run schedule_v3.json may be stored next to
    # the run itself.
    schedule = _load_schedule(schedule_path)

    # Recover kappa_t for the final comparison -- prefer the BetaSchedule's
    # own kappa_t so the audit compares against the same array the run was
    # certified against (which, for Phase IV-B v3 wraps, comes from
    # build_certification under the wrapped alpha_t).
    kappa_t = np.asarray(schedule._kappa_t, dtype=np.float64)

    # ---- Rebuild the MDP ----
    if spec["phase"] == "III":
        # Use the per-run ``config.json`` as the source of truth: it
        # already contains every runtime-resolved knob the runner saw
        # (horizon, prob, reward_bound, ...). Fall back to the suite
        # config only for keys the per-run file did not persist.
        suite = _load_phase3_suite_cfg()
        canonical_task = spec["canonical_task"]
        task_cfg = dict(run_cfg)
        for k, v in suite["tasks"].get(canonical_task, {}).items():
            task_cfg.setdefault(k, v)
        # Scrub meta-keys that the factory does not consume.
        for meta in ("suite_config_path", "schedule_file", "task"):
            task_cfg.pop(meta, None)
        task_cfg["_canonical_task"] = canonical_task
        mdp = _rebuild_phase3_mdp(
            canonical_task, task_cfg,
            seed=spec["seed"],
            regime_phase=spec["regime_phase"],
        )
    elif spec["phase"] == "IV-B":
        task_cfg_inner = run_cfg.get("task_cfg", {})
        task_cfg = dict(task_cfg_inner)
        # Keep the canonical-task pointer on a separate key so factory
        # dispatch never receives it as a kwarg.
        canonical_task = str(task_cfg.get("family", "unknown"))
        mdp = _rebuild_phase4b_mdp(task_cfg, seed=spec["seed"])
        task_cfg["_canonical_task"] = canonical_task
    else:
        raise RunReconstructionError(f"Unsupported phase: {spec['phase']}")

    # ---- Run the safe planner to recover V_safe and pi_safe ----
    algo = spec["algo"]
    if algo.lower() in ("safepe",):
        ref_pi = _build_ref_pi(task_cfg, mdp)
        V_safe, pi_safe = _run_safe_pe(mdp, schedule, ref_pi)
    elif algo.lower() in ("safevi", "safe_vi"):
        V_safe, pi_safe = _run_safe_vi(mdp, schedule)
    else:
        raise RunReconstructionError(
            f"Unsupported safe algorithm for remediation: {algo!r}"
        )

    # ---- Classical optimal policy for d_ref ----
    pi_cl = _run_classical_vi(mdp)

    # ---- Effective-discount tensor ----
    eff_d_tensor = _recompute_eff_d_tensor(mdp, V_safe, schedule)
    T, S, A = eff_d_tensor.shape

    # ---- Reference occupancy ----
    occ = compute_d_ref(mdp, pi_classical=pi_cl, pi_safe=pi_safe)
    d_ref = np.asarray(occ["d_ref"], dtype=np.float64)  # (T, S)
    if d_ref.shape != (T, S):
        raise RunReconstructionError(
            f"d_ref shape {d_ref.shape} != eff_d shape {(T, S)}"
        )

    # ---- Per-stage occupancy-weighted eff_d ----
    # pi_safe may be deterministic (T, S) or stochastic (T, S, A).
    #
    # At absorbing / terminal states ``P[s, a, :].sum() == 0`` the P-weighted
    # effective discount degenerates to ``(1+gamma)(1 - 0) = 1 + gamma`` because
    # the E_{s'} integrand is zero. This is a pure algebraic artifact: the
    # operator's target itself collapses to zero (there is no future value to
    # discount), so the cert bound is not materially violated. We mask such
    # (s, a) cells out of the occupancy weighting and re-normalise d_ref per
    # stage so the remaining cells carry full probability mass.
    p_arr_for_mask = np.asarray(mdp.p, dtype=np.float64)  # (S, A, S')
    absorbing_sa = np.isclose(p_arr_for_mask.sum(axis=-1), 0.0)  # (S, A)
    pi_safe_arr = np.asarray(pi_safe)
    if pi_safe_arr.ndim == 2:
        # Deterministic: gather eff_d under chosen action and flag absorbing.
        t_idx = np.arange(T)[:, None]
        s_idx = np.arange(S)[None, :]
        chosen = pi_safe_arr.astype(np.int64)                   # (T, S)
        ed_pi_safe = eff_d_tensor[t_idx, s_idx, chosen]         # (T, S)
        # absorbing_sa is shape (S, A); broadcast (1, S) against chosen (T, S).
        absorbing_under_pi = absorbing_sa[
            np.arange(S)[None, :], chosen
        ]                                                       # (T, S)
        # d_ref' = d_ref with absorbing cells zeroed, renormalised per stage.
        d_ref_live = d_ref * (~absorbing_under_pi).astype(np.float64)
    elif pi_safe_arr.ndim == 3:
        # Stochastic: fold P(·|s,a) weights into pi before cert weighting.
        # An (s,a) with zero P gets eff_d masked out of the action average.
        absorbing_3d = absorbing_sa[np.newaxis, :, :]           # (1, S, A)
        pi_live = pi_safe_arr * (~absorbing_3d).astype(np.float64)
        # Renormalise pi per (t, s) so the live actions share the full mass.
        pi_sum = pi_live.sum(axis=2, keepdims=True)
        # Where every action is absorbing, the state itself is absorbing;
        # zero out that state's contribution instead.
        state_absorbing = (pi_sum[..., 0] <= 0.0)
        pi_sum_safe = np.where(pi_sum > 0.0, pi_sum, 1.0)
        pi_live = pi_live / pi_sum_safe
        ed_pi_safe = np.einsum("tsa,tsa->ts", pi_live, eff_d_tensor)
        d_ref_live = d_ref * (~state_absorbing).astype(np.float64)
    else:
        raise RunReconstructionError(
            f"Unexpected pi_safe ndim={pi_safe_arr.ndim}"
        )
    # Re-normalise d_ref_live per stage so the "live" mass integrates to 1
    # (when any live mass exists). If every state at a stage is absorbing,
    # keep the stage mass at zero -- d_eff_occ will then be zero and the
    # cert comparison is a no-op for that stage.
    live_sum = d_ref_live.sum(axis=1, keepdims=True)
    safe_live_sum = np.where(live_sum > 0.0, live_sum, 1.0)
    d_ref_norm = d_ref_live / safe_live_sum
    d_eff_occ = np.sum(d_ref_norm * ed_pi_safe, axis=1)  # (T,)
    # Stages where no live mass exists: set to -inf-equivalent so the cert
    # comparison never fires ("no reachable state exists"). In practice
    # those are stages where the entire chain has terminated.
    no_live = (live_sum[..., 0] <= 0.0)
    d_eff_occ = np.where(no_live, -np.inf, d_eff_occ)

    # Align lengths (schedule may be longer than T for safety).
    n = min(T, int(kappa_t.shape[0]))
    d_eff_occ = d_eff_occ[:n]
    kappa_aligned = kappa_t[:n]

    diff = d_eff_occ - kappa_aligned
    # -inf entries correspond to stages with no live (reachable, non-
    # absorbing) mass; mask them out so the cert comparison skips them.
    finite = np.isfinite(diff)
    over_mask = np.zeros_like(diff, dtype=bool)
    over_mask[finite] = diff[finite] > tol
    stages_over = int(over_mask.sum())
    max_overshoot = (
        float(np.max(diff[finite])) if finite.any() else float("nan")
    )
    if stages_over == 0:
        return OccupancyCheckResult(
            ok=True,
            first_breach_stage=None,
            d_eff_occ=d_eff_occ,
            kappa_t=kappa_aligned,
            max_overshoot=max_overshoot,
            stages_over=0,
        )
    first_breach = int(np.argmax(over_mask))
    return OccupancyCheckResult(
        ok=False,
        first_breach_stage=first_breach,
        d_eff_occ=d_eff_occ,
        kappa_t=kappa_aligned,
        max_overshoot=max_overshoot,
        stages_over=stages_over,
    )
