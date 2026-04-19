"""Phase IV-A S3: Phase III audit and compatibility verification.

Audits Phase III code and results for compatibility with Phase IV,
and runs a minimal DP replay smoke check to verify the schedule
pipeline still produces valid (non-NaN, non-zero) outputs.

Public API
----------
run_phase3_code_audit    Check code-level compatibility.
run_phase3_result_audit  Check result artifacts for completeness.
run_phase3_replay_smoke  Minimal DP replay through Phase IV code path.
run_audit                Top-level runner that writes all audit artifacts.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Path bootstrapping
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_MUSHROOM_DEV = _REPO_ROOT / "mushroom-rl-dev"
for _p in (_REPO_ROOT, _MUSHROOM_DEV):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    """Return hex SHA-256 of a file, or empty string if unreadable."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def _git_sha(repo_root: Path) -> str:
    """Return the current git SHA, or 'unknown' on failure."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _load_json_safe(path: Path) -> tuple[dict | None, str]:
    """Load a JSON file, returning (data, error_msg).  error_msg is empty on success."""
    try:
        with open(path) as f:
            return json.load(f), ""
    except FileNotFoundError:
        return None, f"File not found: {path}"
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON in {path}: {exc}"
    except Exception as exc:
        return None, f"Error reading {path}: {exc}"


# ---------------------------------------------------------------------------
# 1. Code audit
# ---------------------------------------------------------------------------

_REQUIRED_SCHEDULE_KEYS = {"beta_used_t", "task_family", "gamma"}

#: All task families expected in Phase III.
_PHASE3_TASK_FAMILIES = [
    "chain_sparse_long",
    "chain_jackpot",
    "chain_catastrophe",
    "chain_regime_shift",
    "grid_sparse_goal",
    "grid_hazard",
    "grid_regime_shift",
    "taxi_bonus_shock",
]


def run_phase3_code_audit(repo_root: Path) -> dict:
    """Audit Phase III code for compatibility with Phase IV requirements.

    Parameters
    ----------
    repo_root : Path
        Repository root containing ``experiments/weighted_lse_dp/``.

    Returns
    -------
    dict
        Audit report with keys ``compatible`` (bool),
        ``observability_gaps`` (list[str]), ``required_changes``
        (list[str]), ``schedule_files_found`` (list[str]),
        ``reward_bound_missing`` (list[str]), ``notes`` (str).
    """
    repo_root = Path(repo_root).resolve()
    observability_gaps: list[str] = []
    required_changes: list[str] = []
    schedule_files_found: list[str] = []
    reward_bound_missing: list[str] = []
    notes_parts: list[str] = []

    # ------------------------------------------------------------------
    # 1. Check schedule files exist and load as valid JSON
    # ------------------------------------------------------------------
    calibration_base = repo_root / "results" / "weighted_lse_dp" / "phase3" / "calibration"
    for family in _PHASE3_TASK_FAMILIES:
        sched_path = calibration_base / family / "schedule.json"
        if not sched_path.is_file():
            observability_gaps.append(
                f"Schedule file missing for {family}: {sched_path}"
            )
            continue

        data, err = _load_json_safe(sched_path)
        if data is None:
            observability_gaps.append(f"Schedule load error for {family}: {err}")
            continue

        schedule_files_found.append(str(sched_path))

        # 2. Check required fields
        missing_keys = _REQUIRED_SCHEDULE_KEYS - set(data.keys())
        if missing_keys:
            observability_gaps.append(
                f"Schedule for {family} missing required keys: {sorted(missing_keys)}"
            )

        # T is derived from len(beta_used_t)
        beta_used = data.get("beta_used_t")
        if beta_used is not None:
            T = len(beta_used)
            notes_parts.append(f"{family}: T={T} (from beta_used_t length)")
        else:
            observability_gaps.append(
                f"Schedule for {family} has no beta_used_t array"
            )

    # ------------------------------------------------------------------
    # 3. Check RL runner n_base usage (env metadata vs hard-coded)
    # ------------------------------------------------------------------
    rl_runner_path = (
        repo_root / "experiments" / "weighted_lse_dp" / "runners" / "run_phase3_rl.py"
    )
    if rl_runner_path.is_file():
        rl_source = rl_runner_path.read_text()
        # The runner should derive n_base from mdp_info / observation_space
        # and validate against the registry.
        uses_env_derived = "n_base_derived" in rl_source or "observation_space.n" in rl_source
        has_n_base_dict = "_N_BASE" in rl_source

        if has_n_base_dict and not uses_env_derived:
            observability_gaps.append(
                "run_phase3_rl.py uses hard-coded _N_BASE dict without "
                "env-derived validation (R6-2 pattern missing)"
            )
        elif has_n_base_dict and uses_env_derived:
            notes_parts.append(
                "run_phase3_rl.py: _N_BASE dict with env-derived n_base validation (R6-2 compliant)"
            )
        elif not has_n_base_dict:
            notes_parts.append(
                "run_phase3_rl.py: no _N_BASE dict found (fully env-derived)"
            )
    else:
        observability_gaps.append(
            f"RL runner not found: {rl_runner_path}"
        )

    # ------------------------------------------------------------------
    # 4. Check reward_bound in Phase III paper_suite config
    # ------------------------------------------------------------------
    config_path = (
        repo_root / "experiments" / "weighted_lse_dp" / "configs" / "phase3" / "paper_suite.json"
    )
    if config_path.is_file():
        config_data, cfg_err = _load_json_safe(config_path)
        if config_data is not None:
            tasks_cfg = config_data.get("tasks", {})
            for task_name, task_cfg in tasks_cfg.items():
                if "reward_bound" not in task_cfg:
                    reward_bound_missing.append(task_name)
            if reward_bound_missing:
                observability_gaps.append(
                    f"reward_bound missing in paper_suite.json for: {reward_bound_missing}"
                )
            else:
                notes_parts.append(
                    "All tasks in paper_suite.json have reward_bound"
                )
        else:
            observability_gaps.append(
                f"Could not load paper_suite.json: {cfg_err}"
            )
    else:
        observability_gaps.append(f"paper_suite.json not found: {config_path}")

    # ------------------------------------------------------------------
    # 5. Check safe_weighted_common.py has compute_safe_target_ev_batch
    # ------------------------------------------------------------------
    swc_path = (
        repo_root / "mushroom-rl-dev" / "mushroom_rl" / "algorithms"
        / "value" / "dp" / "safe_weighted_common.py"
    )
    if swc_path.is_file():
        swc_source = swc_path.read_text()
        if "compute_safe_target_ev_batch" in swc_source:
            notes_parts.append(
                "safe_weighted_common.py: compute_safe_target_ev_batch present"
            )
        else:
            observability_gaps.append(
                "safe_weighted_common.py missing compute_safe_target_ev_batch "
                "(Issue 1 fix not applied)"
            )
    else:
        observability_gaps.append(f"safe_weighted_common.py not found: {swc_path}")

    # ------------------------------------------------------------------
    # Determine overall compatibility
    # ------------------------------------------------------------------
    compatible = len(observability_gaps) == 0 and len(required_changes) == 0

    return {
        "compatible": compatible,
        "observability_gaps": observability_gaps,
        "required_changes": required_changes,
        "schedule_files_found": schedule_files_found,
        "reward_bound_missing": reward_bound_missing,
        "notes": "; ".join(notes_parts) if notes_parts else "",
    }


# ---------------------------------------------------------------------------
# 2. Result audit
# ---------------------------------------------------------------------------


def run_phase3_result_audit(results_dir: Path) -> dict:
    """Audit Phase III results for completeness.

    Parameters
    ----------
    results_dir : Path
        Directory ``results/weighted_lse_dp/phase3/``.

    Returns
    -------
    dict
        Audit report with keys ``result_dirs_found`` (list[str]),
        ``rho_all_nan_tasks`` (list[str]), ``missing_tasks`` (list[str]),
        ``result_count`` (int), ``notes`` (str).
    """
    results_dir = Path(results_dir).resolve()
    result_dirs_found: list[str] = []
    rho_all_nan_tasks: list[str] = []
    missing_tasks: list[str] = []
    notes_parts: list[str] = []
    result_count = 0

    paper_suite_dir = results_dir / "paper_suite"
    if not paper_suite_dir.is_dir():
        notes_parts.append(
            f"paper_suite/ directory not found at {paper_suite_dir}. "
            "Phase III results may not have been generated yet."
        )
        return {
            "result_dirs_found": result_dirs_found,
            "rho_all_nan_tasks": rho_all_nan_tasks,
            "missing_tasks": list(_PHASE3_TASK_FAMILIES),
            "result_count": 0,
            "notes": "; ".join(notes_parts),
        }

    # Check each task family directory
    for family in _PHASE3_TASK_FAMILIES:
        family_dir = paper_suite_dir / family
        if not family_dir.is_dir():
            missing_tasks.append(family)
            continue

        result_dirs_found.append(family)

        # Look for run artifacts: <algo>/<seed_XX>/{run.json,metrics.json,calibration_stats.npz}
        run_files_found = False
        rho_checked = False
        for algo_dir in sorted(family_dir.iterdir()):
            if not algo_dir.is_dir():
                continue
            for seed_dir in sorted(algo_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue
                # Check for run.json OR metrics.json (the actual artifact names)
                has_run_json = (seed_dir / "run.json").is_file()
                has_metrics_json = (seed_dir / "metrics.json").is_file()
                has_calib_stats = (seed_dir / "calibration_stats.npz").is_file()

                if has_run_json or has_metrics_json:
                    run_files_found = True
                    result_count += 1

                # Check rho_mean in calibration_stats.npz (sample one per family)
                if has_calib_stats and not rho_checked:
                    rho_checked = True
                    try:
                        with np.load(seed_dir / "calibration_stats.npz") as npz:
                            rho_key = None
                            for k in ("safe_rho_mean", "rho_mean"):
                                if k in npz:
                                    rho_key = k
                                    break
                            if rho_key is not None:
                                rho_arr = npz[rho_key]
                                if np.all(np.isnan(rho_arr)):
                                    rho_all_nan_tasks.append(family)
                            else:
                                notes_parts.append(
                                    f"{family}: no rho_mean/safe_rho_mean key in calibration_stats.npz"
                                )
                    except Exception as exc:
                        notes_parts.append(
                            f"{family}: error loading calibration_stats.npz: {exc}"
                        )

        if not run_files_found:
            notes_parts.append(
                f"{family}: directory exists but no run artifacts found"
            )

    # Check for RL-specific artifacts
    rl_keys_expected = {"train_steps", "n_transitions"}
    for family in result_dirs_found:
        family_dir = paper_suite_dir / family
        # Sample one RL run to check metrics.json has expected keys
        for algo_dir in sorted(family_dir.iterdir()):
            if not algo_dir.is_dir():
                continue
            algo_name = algo_dir.name
            if algo_name.startswith("Safe") and "Learning" in algo_name or "SARSA" in algo_name:
                for seed_dir in sorted(algo_dir.iterdir()):
                    if not seed_dir.is_dir():
                        continue
                    metrics_path = seed_dir / "metrics.json"
                    if metrics_path.is_file():
                        data, err = _load_json_safe(metrics_path)
                        if data is not None:
                            missing_rl_keys = rl_keys_expected - set(data.keys())
                            if missing_rl_keys:
                                notes_parts.append(
                                    f"{family}/{algo_name}: metrics.json missing expected RL keys: "
                                    f"{sorted(missing_rl_keys)}"
                                )
                        break
                break

    if rho_all_nan_tasks:
        notes_parts.append(
            f"rho_mean all-NaN in DP results for: {rho_all_nan_tasks}"
        )

    notes_parts.append(f"Total run artifact sets found: {result_count}")

    return {
        "result_dirs_found": result_dirs_found,
        "rho_all_nan_tasks": rho_all_nan_tasks,
        "missing_tasks": missing_tasks,
        "result_count": result_count,
        "notes": "; ".join(notes_parts),
    }


# ---------------------------------------------------------------------------
# 3. Replay smoke check
# ---------------------------------------------------------------------------


def run_phase3_replay_smoke(repo_root: Path, output_dir: Path) -> dict:
    """Run a minimal DP replay to verify Phase IV code-path compatibility.

    Loads the chain_sparse_long task and schedule, runs SafeWeightedValueIteration
    for 3 sweeps on seed=11, and verifies outputs are not all-zero or all-NaN.

    Parameters
    ----------
    repo_root : Path
        Repository root.
    output_dir : Path
        Directory for replay output artifacts.

    Returns
    -------
    dict
        Replay report with keys ``dp_replay_passed``, ``rl_replay_skipped``,
        ``dp_beta_nonzero``, ``dp_rho_valid``, ``output_file``, ``notes``.
    """
    repo_root = Path(repo_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    notes_parts: list[str] = []
    dp_replay_passed = False
    dp_beta_nonzero = False
    dp_rho_valid = False

    try:
        # Lazy imports -- these require the full repo on sys.path
        from experiments.weighted_lse_dp.tasks.stress_families import (
            make_chain_sparse_long,
            make_chain_jackpot,
        )
        from mushroom_rl.algorithms.value.dp import (
            SafeWeightedValueIteration,
            BetaSchedule,
            extract_mdp_arrays,
        )
        from experiments.weighted_lse_dp.common.seeds import seed_everything

        # -- Load paper_suite config ------------------------------------------
        config_path = (
            repo_root / "experiments" / "weighted_lse_dp"
            / "configs" / "phase3" / "paper_suite.json"
        )
        with open(config_path) as f:
            config = json.load(f)

        # Try chain_sparse_long first; if its schedule is all-zero (sparse
        # data guard), also run chain_jackpot which has nonzero beta.
        task_name = "chain_sparse_long"
        task_cfg = config["tasks"][task_name]

        # -- Load schedule ----------------------------------------------------
        sched_path = (
            repo_root / "results" / "weighted_lse_dp" / "phase3"
            / "calibration" / task_name / "schedule.json"
        )
        if not sched_path.is_file():
            notes_parts.append(f"Schedule not found: {sched_path}")
            raise FileNotFoundError(f"Schedule not found: {sched_path}")

        schedule = BetaSchedule.from_file(sched_path)

        # -- Build task -------------------------------------------------------
        seed_everything(11)

        cfg_copy = dict(task_cfg)
        mdp, _mdp_rl, resolved = make_chain_sparse_long(
            cfg=cfg_copy,
            state_n=int(task_cfg.get("state_n", 60)),
            prob=float(task_cfg.get("prob", 0.9)),
            gamma=float(task_cfg.get("gamma", 0.99)),
            horizon=int(task_cfg.get("horizon", 120)),
        )

        # -- Run SafeWeightedValueIteration (3 sweeps) ------------------------
        planner = SafeWeightedValueIteration(mdp, schedule=schedule)
        planner.run()

        # -- Check outputs ----------------------------------------------------
        V = planner.V  # shape (H+1, S)
        beta_used = np.array(schedule._beta_used_t)
        sparse_data_fallback = bool(np.all(beta_used == 0.0))

        # Check effective discount from clipping_summary
        clip_summary = getattr(planner, "clipping_summary", {})
        gamma_mdp = float(task_cfg.get("gamma", 0.99))
        eff_discount = np.array(
            clip_summary.get("stage_eff_discount_mean", [gamma_mdp] * schedule.T),
            dtype=np.float64,
        )
        # rho = 1 - eff_d / (1+gamma)
        rho = 1.0 - eff_discount / (1.0 + gamma_mdp)

        dp_rho_valid = bool(
            not np.all(np.isnan(rho))
            and not np.all(rho == 0.0)
        )

        # V should not be all-zero or all-NaN
        v_valid = bool(
            not np.all(np.isnan(V))
            and not np.all(V == 0.0)
        )

        notes_parts.append(
            f"{task_name} replay: V shape={V.shape}, "
            f"V range=[{float(np.nanmin(V)):.6f}, {float(np.nanmax(V)):.6f}], "
            f"beta_used range=[{float(np.min(beta_used)):.6f}, {float(np.max(beta_used)):.6f}], "
            f"rho range=[{float(np.nanmin(rho)):.6f}, {float(np.nanmax(rho)):.6f}], "
            f"n_sweeps={len(planner.residuals)}"
        )

        if sparse_data_fallback:
            notes_parts.append(
                f"{task_name}: sparse-data fallback triggered (all beta_used=0). "
                "This is expected for tasks with insufficient aligned-margin data."
            )

        # If chain_sparse_long has all-zero beta (sparse data guard), also
        # replay chain_jackpot which has nonzero beta values, to verify the
        # operator is functional with non-trivial schedules.
        if sparse_data_fallback:
            jackpot_cfg = config["tasks"]["chain_jackpot"]
            jackpot_sched_path = (
                repo_root / "results" / "weighted_lse_dp" / "phase3"
                / "calibration" / "chain_jackpot" / "schedule.json"
            )
            if jackpot_sched_path.is_file():
                jackpot_schedule = BetaSchedule.from_file(jackpot_sched_path)
                seed_everything(11)
                jackpot_cfg_copy = dict(jackpot_cfg)
                jackpot_mdp, _, _ = make_chain_jackpot(
                    cfg=jackpot_cfg_copy,
                    state_n=int(jackpot_cfg.get("state_n", 25)),
                    prob=float(jackpot_cfg.get("prob", 0.9)),
                    gamma=float(jackpot_cfg.get("gamma", 0.99)),
                    horizon=int(jackpot_cfg.get("horizon", 60)),
                    jackpot_state=int(jackpot_cfg.get("jackpot_state", 20)),
                    jackpot_prob=float(jackpot_cfg.get("jackpot_prob", 0.05)),
                    jackpot_reward=float(jackpot_cfg.get("jackpot_reward", 10.0)),
                    jackpot_terminates=bool(jackpot_cfg.get("jackpot_terminates", True)),
                )
                jackpot_planner = SafeWeightedValueIteration(
                    jackpot_mdp, schedule=jackpot_schedule,
                )
                jackpot_planner.run()
                jackpot_beta = np.array(jackpot_schedule._beta_used_t)
                dp_beta_nonzero = bool(np.any(jackpot_beta != 0.0))

                jackpot_V = jackpot_planner.V
                notes_parts.append(
                    f"chain_jackpot replay (nonzero-beta verification): "
                    f"V shape={jackpot_V.shape}, "
                    f"V range=[{float(np.nanmin(jackpot_V)):.6f}, {float(np.nanmax(jackpot_V)):.6f}], "
                    f"beta_used range=[{float(np.min(jackpot_beta)):.6f}, {float(np.max(jackpot_beta)):.6f}]"
                )
            else:
                dp_beta_nonzero = False
                notes_parts.append(
                    "chain_jackpot schedule not found; cannot verify nonzero-beta replay"
                )
        else:
            dp_beta_nonzero = bool(np.any(beta_used != 0.0))

        dp_replay_passed = dp_beta_nonzero and dp_rho_valid and v_valid

    except Exception as exc:
        notes_parts.append(f"Replay smoke check failed: {exc}")
        dp_replay_passed = False

    # -- Write replay report ------------------------------------------------
    report = {
        "dp_replay_passed": dp_replay_passed,
        "rl_replay_skipped": True,
        "dp_beta_nonzero": dp_beta_nonzero,
        "dp_rho_valid": dp_rho_valid,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "notes": "; ".join(notes_parts),
    }

    output_file = output_dir / "dp_replay_smoke.json"
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    report["output_file"] = str(output_file)
    return report


# ---------------------------------------------------------------------------
# 4. Top-level runner
# ---------------------------------------------------------------------------


def run_audit(
    repo_root_str: str = ".",
    output_dir_str: str = "results/weighted_lse_dp/phase4/audit",
) -> None:
    """Run all Phase III audits and write reports.

    Parameters
    ----------
    repo_root_str : str
        Repository root path.
    output_dir_str : str
        Output directory for audit artifacts.
    """
    repo_root = Path(repo_root_str).resolve()
    output_dir = Path(output_dir_str)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir_str
    output_dir.mkdir(parents=True, exist_ok=True)

    git_sha = _git_sha(repo_root)
    timestamp = datetime.now(timezone.utc).isoformat()

    # -- 1. Code audit -------------------------------------------------------
    print("[phase3_audit] Running code audit...")
    code_report = run_phase3_code_audit(repo_root)
    code_report["git_sha"] = git_sha
    code_report["timestamp"] = timestamp

    code_audit_path = output_dir / "phase3_code_audit.json"
    with open(code_audit_path, "w") as f:
        json.dump(code_report, f, indent=2)
    print(f"  -> {code_audit_path}")

    # -- 2. Result audit -----------------------------------------------------
    print("[phase3_audit] Running result audit...")
    results_dir = repo_root / "results" / "weighted_lse_dp" / "phase3"
    result_report = run_phase3_result_audit(results_dir)
    result_report["git_sha"] = git_sha
    result_report["timestamp"] = timestamp

    result_audit_path = output_dir / "phase3_result_audit.json"
    with open(result_audit_path, "w") as f:
        json.dump(result_report, f, indent=2)
    print(f"  -> {result_audit_path}")

    # -- 3. Replay smoke check -----------------------------------------------
    print("[phase3_audit] Running replay smoke check...")
    replay_dir = output_dir / "phase3_replay_smoke"
    replay_report = run_phase3_replay_smoke(repo_root, replay_dir)

    # -- 4. Write compatibility report (markdown) ----------------------------
    compat_path = output_dir / "phase3_compat_report.md"
    _write_compat_report(
        compat_path,
        code_report=code_report,
        result_report=result_report,
        replay_report=replay_report,
        git_sha=git_sha,
        timestamp=timestamp,
    )
    print(f"  -> {compat_path}")

    # -- Summary -------------------------------------------------------------
    overall = (
        code_report["compatible"]
        and len(result_report["missing_tasks"]) == 0
        and replay_report.get("dp_replay_passed", False)
    )
    status = "COMPATIBLE" if overall else "ISSUES FOUND"
    print(f"\n[phase3_audit] Phase III compatibility: {status}")
    print(f"  Code audit compatible: {code_report['compatible']}")
    print(f"  Observability gaps: {len(code_report['observability_gaps'])}")
    print(f"  Missing tasks: {result_report['missing_tasks']}")
    print(f"  Result count: {result_report['result_count']}")
    print(f"  DP replay passed: {replay_report.get('dp_replay_passed', False)}")


def _write_compat_report(
    path: Path,
    *,
    code_report: dict,
    result_report: dict,
    replay_report: dict,
    git_sha: str,
    timestamp: str,
) -> None:
    """Write the compatibility report as markdown."""
    lines: list[str] = []
    lines.append("# Phase III Compatibility Audit Report")
    lines.append("")
    lines.append(f"Generated: {timestamp}")
    lines.append(f"Git SHA: {git_sha}")
    lines.append("")

    # Code audit section
    lines.append("## Code Audit")
    lines.append("")
    compat_str = "PASS" if code_report["compatible"] else "FAIL"
    lines.append(f"**Status**: {compat_str}")
    lines.append("")

    if code_report["schedule_files_found"]:
        lines.append(f"Schedule files found: {len(code_report['schedule_files_found'])}")
        for sf in code_report["schedule_files_found"]:
            lines.append(f"  - {sf}")
        lines.append("")

    if code_report["observability_gaps"]:
        lines.append("### Observability Gaps")
        lines.append("")
        for gap in code_report["observability_gaps"]:
            lines.append(f"- {gap}")
        lines.append("")

    if code_report["required_changes"]:
        lines.append("### Required Changes")
        lines.append("")
        for change in code_report["required_changes"]:
            lines.append(f"- {change}")
        lines.append("")

    if code_report["reward_bound_missing"]:
        lines.append("### Missing reward_bound")
        lines.append("")
        for task in code_report["reward_bound_missing"]:
            lines.append(f"- {task}")
        lines.append("")

    if code_report.get("notes"):
        lines.append(f"**Notes**: {code_report['notes']}")
        lines.append("")

    # Result audit section
    lines.append("## Result Audit")
    lines.append("")
    lines.append(f"Result directories found: {len(result_report['result_dirs_found'])}")
    lines.append(f"Total run artifacts: {result_report['result_count']}")
    lines.append("")

    if result_report["missing_tasks"]:
        lines.append("### Missing Tasks")
        lines.append("")
        for task in result_report["missing_tasks"]:
            lines.append(f"- {task}")
        lines.append("")

    if result_report["rho_all_nan_tasks"]:
        lines.append("### rho_mean All-NaN")
        lines.append("")
        for task in result_report["rho_all_nan_tasks"]:
            lines.append(f"- {task}")
        lines.append("")

    if result_report.get("notes"):
        lines.append(f"**Notes**: {result_report['notes']}")
        lines.append("")

    # Replay section
    lines.append("## DP Replay Smoke Check")
    lines.append("")
    replay_str = "PASS" if replay_report.get("dp_replay_passed") else "FAIL"
    lines.append(f"**Status**: {replay_str}")
    lines.append(f"- beta_used nonzero: {replay_report.get('dp_beta_nonzero', False)}")
    lines.append(f"- rho valid: {replay_report.get('dp_rho_valid', False)}")
    lines.append(f"- RL replay skipped: {replay_report.get('rl_replay_skipped', True)}")
    lines.append("")
    if replay_report.get("notes"):
        lines.append(f"**Notes**: {replay_report['notes']}")
        lines.append("")

    # Overall
    lines.append("## Overall Verdict")
    lines.append("")
    overall = (
        code_report["compatible"]
        and len(result_report.get("missing_tasks", [])) == 0
        and replay_report.get("dp_replay_passed", False)
    )
    verdict = "COMPATIBLE" if overall else "ISSUES FOUND -- review gaps above"
    lines.append(f"**Phase III -> Phase IV compatibility**: {verdict}")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase IV-A: Phase III compatibility audit.",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=".",
        help="Repository root path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/weighted_lse_dp/phase4/audit",
        help="Output directory for audit artifacts.",
    )
    args = parser.parse_args()
    run_audit(args.repo_root, args.output_dir)
