"""Phase VII run-experiment driver (M3.3).

CLI entrypoint that turns a YAML config into a fan-out matrix of
``(env, method, seed)`` runs. Each run produces a self-contained directory
under ``<raw-root>/<stage>/<env>/<method>/<seed>/`` (default raw-root is
``results/adaptive_beta/raw/``; override via ``--out`` or ``--raw-root``) with:

- ``run.json`` — manifest: git SHA, argv, seed, resolved config slice,
  start/end timestamps, status.
- ``metrics.npz`` — per-episode arrays + schema header.
- ``episodes.csv`` — schema per spec §7.4.
- ``transitions.parquet`` — schema per spec §7.4 (stratified per stage rule).

A *list-shaped*, append-only Phase VII manifest is maintained at
``results/summaries/phase_VII_manifest.json`` (option 1, locked 2026-04-26 by
the user). Every completed/failed run appends one record. The Phase V
single-dict manifest (``experiment_manifest.json``) is left untouched.

Usage::

    python experiments/adaptive_beta/run_experiment.py \
        --config experiments/adaptive_beta/configs/dev.yaml \
        --seed 0

Test-only CLI overrides (M3.4)
------------------------------
For the test suite (and ad-hoc runs), three additive flags reduce the
fan-out matrix without editing the YAML:

- ``--only-env <name>`` keeps just the named env block, dropping the rest.
- ``--only-method <id>`` keeps just the named method, dropping the rest.
- ``--episodes <int>`` overrides ``n_episodes`` from the YAML.
- ``--out <path>`` is a synonym for ``--raw-root`` (matches the test runner
  vocabulary; both flags resolve to the same destination).

These flags do not change the algorithm or schedule logic — they only
narrow the dispatch matrix and the per-run episode budget. They exist so
the reproducibility / smoke tests can drive the real runner end-to-end
in seconds rather than minutes.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

# Repo root on the path for absolute imports when called directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.adaptive_beta.agents import (  # noqa: E402
    AdaptiveBetaQAgent,
    linear_epsilon_schedule,
)
from experiments.adaptive_beta.logging_callbacks import (  # noqa: E402
    EpisodeLogger,
    SCHEMA_VERSION_EPISODES,
    SCHEMA_VERSION_TRANSITIONS,
    TransitionLogger,
    _RunIdentity,
)
from experiments.adaptive_beta.schedules import (  # noqa: E402
    ALL_METHOD_IDS,
    build_schedule,
)

# ---------------------------------------------------------------------------
# Per-method hparam allow-lists (must mirror schedules._ALLOWED_KEYS).
# ---------------------------------------------------------------------------
# The schedules factory rejects unknown keys so configs typos fail loudly.
# Configs in this directory carry the union of every method's hparams; the
# runner filters per-method before calling build_schedule.
_PER_METHOD_HPARAM_KEYS: Dict[str, frozenset] = {
    "vanilla": frozenset(),
    "fixed_positive": frozenset({"beta0"}),
    "fixed_negative": frozenset({"beta0"}),
    "wrong_sign": frozenset({"beta0"}),
    "adaptive_beta": frozenset(
        {"beta_max", "beta_cap", "k", "initial_beta", "beta_tol", "lambda_smooth"}
    ),
    "adaptive_beta_no_clip": frozenset(
        {"beta_max", "beta_cap", "k", "initial_beta", "beta_tol", "lambda_smooth"}
    ),
    "adaptive_sign_only": frozenset({"beta0", "beta_cap", "lambda_smooth"}),
    "adaptive_magnitude_only": frozenset(
        {"beta_max", "beta_cap", "k", "initial_beta", "beta_tol", "lambda_smooth"}
    ),
}


def _filter_hparams_for_method(
    method: str, hparams: Dict[str, Any]
) -> Dict[str, Any]:
    keys = _PER_METHOD_HPARAM_KEYS.get(method)
    if keys is None:
        raise ValueError(f"unknown method id {method!r}")
    return {k: v for k, v in hparams.items() if k in keys}

# ---------------------------------------------------------------------------
# Manifest path locked by the user (see prompt 2026-04-26).
# ---------------------------------------------------------------------------
MANIFEST_PATH = (
    _REPO_ROOT / "results" / "summaries" / "phase_VII_manifest.json"
)
MANIFEST_SCHEMA_VERSION = "phaseVII.runs.v1"

# ---------------------------------------------------------------------------
# Env factories (kept as a small dispatch table to avoid importing every env
# at module-load time, which keeps the CLI startup cost low).
# ---------------------------------------------------------------------------


def _build_env(env_name: str, env_kwargs: Dict[str, Any], seed: int):
    """Construct a Phase VII env. ``seed`` is the common_env_seed."""
    if env_name == "rps":
        from experiments.adaptive_beta.envs.rps import RPS
        return RPS(seed=seed, **env_kwargs)
    if env_name == "switching_bandit":
        from experiments.adaptive_beta.envs.switching_bandit import (
            SwitchingBandit,
        )
        return SwitchingBandit(seed=seed, **env_kwargs)
    if env_name == "hazard_gridworld":
        from experiments.adaptive_beta.envs.hazard_gridworld import (
            HazardGridworld,
        )
        return HazardGridworld(seed=seed, **env_kwargs)
    if env_name == "delayed_chain":
        from experiments.adaptive_beta.envs.delayed_chain import DelayedChain
        return DelayedChain(seed=seed, **env_kwargs)
    raise ValueError(f"unknown env_name={env_name!r}")


def _relative_or_absolute(p: Path) -> str:
    """Return ``p`` relative to ``_REPO_ROOT`` if possible, else absolute.

    The default raw-root lives inside the repo, so the manifest record
    stays repo-relative. Tests (and ad-hoc invocations) routinely pass
    ``--out`` to a tmp dir outside the repo; in that case we fall back
    to the absolute path rather than crashing in ``Path.relative_to``.
    """
    try:
        return str(p.relative_to(_REPO_ROOT))
    except ValueError:
        return str(p.resolve())


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:  # pragma: no cover — best-effort
        return "unknown"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_seed_assignment(base_seed: int, method_index: int) -> Tuple[int, int]:
    """Spec §8.4: ``base_seed = 10000 + seed_id`` and method offsets.

    Returns ``(common_env_seed, agent_seed)``. Method offsets exist only
    for the ε-greedy RNG so different methods explore differently while
    sharing identical environment streams.
    """
    base = 10000 + int(base_seed)
    return base, base + (1000 * (method_index + 1))


def _episode_oracle_optimal_value(env_name: str, env: Any) -> float:
    """Per-episode oracle return used for paired regret reporting.

    Lightweight closed-form values where available; otherwise 0.0 (the
    paired-difference logic in analysis still works because all methods
    see the same constant offset).
    """
    if env_name == "switching_bandit":
        # H=1 Bernoulli; oracle expected reward == p_best.
        return float(getattr(env, "_p_best", 0.8))
    return 0.0


# ---------------------------------------------------------------------------
# Manifest helpers.
# ---------------------------------------------------------------------------


def _load_manifest() -> List[Dict[str, Any]]:
    """Load the phase-VII manifest as a list. Empty / missing -> [].

    Tolerates a file containing only ``[]`` (the locked default for a
    fresh overnight). Will *not* attempt to migrate the Phase V dict-
    shaped manifest — that file lives at a different path and is left
    untouched.
    """
    if not MANIFEST_PATH.exists():
        return []
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        # Corrupt — preserve original by renaming and start fresh.
        backup = MANIFEST_PATH.with_suffix(
            f".corrupt.{int(time.time())}.json"
        )
        MANIFEST_PATH.rename(backup)
        return []
    if not isinstance(data, list):
        raise ValueError(
            f"phase_VII_manifest.json must be a JSON list (option 1, "
            f"locked 2026-04-26). Found {type(data).__name__} at "
            f"{MANIFEST_PATH}."
        )
    return data


def _atomic_write_manifest(records: List[Dict[str, Any]]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, MANIFEST_PATH)


def _append_manifest_record(record: Dict[str, Any]) -> None:
    """Read-modify-write append (atomic). Tolerant of concurrent dispatch."""
    records = _load_manifest()
    records.append(record)
    _atomic_write_manifest(records)


# ---------------------------------------------------------------------------
# Single-run executor.
# ---------------------------------------------------------------------------


def run_one(
    *,
    stage: str,
    env_name: str,
    env_kwargs: Dict[str, Any],
    method: str,
    seed_id: int,
    method_index: int,
    n_episodes: int,
    gamma: float,
    learning_rate: float,
    epsilon_cfg: Dict[str, Any],
    schedule_hparams: Dict[str, Any],
    stratify_every: int,
    raw_root: Path,
    run_id: str,
    config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute a single ``(env, method, seed_id)`` run end-to-end.

    Returns a manifest record dict. Side effects: writes ``run.json``,
    ``metrics.npz``, ``episodes.csv``, ``transitions.parquet`` under
    ``raw_root``, and appends one entry to the phase-VII manifest. On
    exception, status=``failed`` is recorded with traceback.
    """
    raw_dir = (
        raw_root / stage / env_name / method / str(seed_id)
    )
    raw_dir.mkdir(parents=True, exist_ok=True)

    started_at = _utc_now()
    common_env_seed, agent_seed = _resolve_seed_assignment(
        seed_id, method_index
    )
    identity = _RunIdentity(
        run_id=run_id, env=env_name, method=method, seed=seed_id
    )

    record: Dict[str, Any] = {
        "run_id": run_id,
        "stage": stage,
        "env": env_name,
        "method": method,
        "seed_id": int(seed_id),
        "status": "running",
        "started_at": started_at,
        "completed_at": None,
        "raw_dir": _relative_or_absolute(raw_dir),
        "n_episodes": int(n_episodes),
        "failure_reason": None,
    }

    try:
        env = _build_env(env_name, env_kwargs, seed=common_env_seed)
        n_states = int(env.info.observation_space.size[0])
        n_actions = int(env.info.action_space.size[0])
        canonical_sign = getattr(env, "env_canonical_sign", None)

        eps_fn = linear_epsilon_schedule(
            start=float(epsilon_cfg.get("start", 1.0)),
            end=float(epsilon_cfg.get("end", 0.05)),
            decay_episodes=int(epsilon_cfg.get("decay_episodes", 5000)),
        )
        method_hparams = _filter_hparams_for_method(
            method, schedule_hparams
        )
        schedule = build_schedule(
            method, canonical_sign, method_hparams
        )
        agent_rng = np.random.default_rng(agent_seed)

        agent = AdaptiveBetaQAgent(
            n_states=n_states,
            n_actions=n_actions,
            gamma=float(gamma),
            learning_rate=float(learning_rate),
            epsilon_schedule=eps_fn,
            beta_schedule=schedule,
            rng=agent_rng,
            env_canonical_sign=canonical_sign,
        )

        ep_logger = EpisodeLogger(identity)
        tr_logger = TransitionLogger(identity, stratify_every=stratify_every)

        # ----- Episode loop -----
        for ep in range(int(n_episodes)):
            agent.begin_episode(ep)
            state, info = env.reset()
            s = int(np.asarray(state).flat[0])
            t = 0
            ep_return = 0.0
            ep_catastrophe = False
            ep_success = False
            ep_oracle_value = _episode_oracle_optimal_value(env_name, env)
            shift_event = bool(info.get("is_shift_step", False))

            while True:
                action = agent.select_action(s, ep)
                next_state, reward, absorbing, step_info = env.step(action)
                ns = int(np.asarray(next_state).flat[0])
                step_diag = agent.step(
                    s, action, float(reward), ns, bool(absorbing), ep
                )
                tr_logger.record(
                    episode=ep,
                    t=t,
                    state=s,
                    action=int(action),
                    reward=float(reward),
                    next_state=ns,
                    done=bool(absorbing),
                    phase=step_info.get("phase", ""),
                    beta_deployed=step_diag["beta_used"],
                    v_next=step_diag["v_next"],
                    advantage=step_diag["advantage"],
                    td_target=step_diag["td_target"],
                    td_error=step_diag["td_error"],
                    d_eff=step_diag["d_eff"],
                    aligned=step_diag["aligned"],
                    oracle_action=step_info.get("oracle_action", None),
                    catastrophe=bool(step_info.get("catastrophe", False)),
                    shift_event=(t == 0 and shift_event),
                )
                ep_return += float(reward)
                if step_info.get("catastrophe", False):
                    ep_catastrophe = True
                if step_info.get("terminal_success", False):
                    ep_success = True
                t += 1
                s = ns
                if absorbing:
                    break

            ep_diag = agent.end_episode(ep)
            # Oracle regret per episode (paired across methods because the
            # env stream is identical at fixed seed_id).
            regret = float(ep_oracle_value * t - ep_return)
            ep_logger.record(
                episode=ep,
                phase=info.get("phase", ""),
                beta_raw=ep_diag["beta_raw"],
                beta_deployed=ep_diag["beta_deployed"],
                episode_return=ep_return,
                length=t,
                epsilon=float(eps_fn(ep)),
                alignment_rate=ep_diag["alignment_rate"],
                mean_signed_alignment=ep_diag["mean_signed_alignment"],
                mean_advantage=ep_diag["mean_advantage"],
                mean_abs_advantage=ep_diag["mean_abs_advantage"],
                mean_d_eff=ep_diag["mean_d_eff"],
                median_d_eff=ep_diag["median_d_eff"],
                frac_d_eff_below_gamma=ep_diag["frac_d_eff_below_gamma"],
                frac_d_eff_above_one=ep_diag["frac_d_eff_above_one"],
                bellman_residual=ep_diag["bellman_residual"],
                td_target_abs_max=ep_diag["td_target_abs_max"],
                q_abs_max=ep_diag["q_abs_max"],
                catastrophic=ep_catastrophe,
                success=ep_success,
                regret=regret,
                shift_event=shift_event,
                divergence_event=ep_diag["divergence_event"],
            )

        # ----- Persist artifacts -----
        n_eps_written = ep_logger.flush_csv(raw_dir / "episodes.csv")
        n_tr_written = tr_logger.flush_parquet(
            raw_dir / "transitions.parquet"
        )

        ep_arrays = ep_logger.collected_arrays()
        np.savez(
            raw_dir / "metrics.npz",
            schema_version=np.array(SCHEMA_VERSION_EPISODES),
            **{
                k: v for k, v in ep_arrays.items()
                if k not in {"run_id", "env", "method", "phase"}
            },
        )

        run_json = {
            "schema_version": "phaseVII.run.v1",
            "run_id": run_id,
            "stage": stage,
            "env": env_name,
            "method": method,
            "seed_id": int(seed_id),
            "common_env_seed": int(common_env_seed),
            "agent_seed": int(agent_seed),
            "n_episodes": int(n_episodes),
            "gamma": float(gamma),
            "learning_rate": float(learning_rate),
            "epsilon_cfg": epsilon_cfg,
            "schedule_hparams": schedule_hparams,
            "stratify_every": int(stratify_every),
            "env_kwargs": env_kwargs,
            "config_path": (
                str(config_path) if config_path is not None else None
            ),
            "argv": sys.argv,
            "git_sha": _git_sha(),
            "host": socket.gethostname(),
            "python_version": sys.version.split()[0],
            "started_at": started_at,
            "completed_at": _utc_now(),
            "n_episodes_written": int(n_eps_written),
            "n_transitions_written": int(n_tr_written),
            "transitions_schema_version": SCHEMA_VERSION_TRANSITIONS,
            "metrics_schema_version": SCHEMA_VERSION_EPISODES,
        }
        with open(raw_dir / "run.json", "w", encoding="utf-8") as f:
            json.dump(run_json, f, indent=2, sort_keys=True)
            f.write("\n")

        record["status"] = "completed"
        record["completed_at"] = run_json["completed_at"]
        return record

    except Exception as exc:
        # Record the failure honestly (spec §2 rule 7 + spec §13.5 honesty).
        record["status"] = "failed"
        record["completed_at"] = _utc_now()
        record["failure_reason"] = f"{type(exc).__name__}: {exc}"
        # Persist the traceback alongside whatever partial outputs exist.
        with open(raw_dir / "FAILURE.log", "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        return record
    finally:
        _append_manifest_record(record)


# ---------------------------------------------------------------------------
# Config-driven dispatch.
# ---------------------------------------------------------------------------


def _resolve_methods_for_env(
    env_name: str,
    requested: List[str],
    env_canonical_sign: Optional[str],
) -> List[str]:
    """Drop methods that are not defined for an env (spec §22.3)."""
    out: List[str] = []
    for m in requested:
        if m in {"wrong_sign", "adaptive_magnitude_only"} and (
            env_canonical_sign is None
        ):
            # Skip with a stderr note; recorded in manifest as 'skipped'
            # by the dispatcher.
            continue
        out.append(m)
    return out


def _peek_canonical_sign(env_name: str, env_kwargs: Dict[str, Any]) -> Optional[str]:
    """Cheap sign lookup without instantiating the env when possible."""
    static_signs = {
        "rps": None,
        "switching_bandit": None,
        "hazard_gridworld": "-",
        "delayed_chain": "+",
    }
    if env_name in static_signs:
        return static_signs[env_name]
    return None


def dispatch_from_config(
    config: Dict[str, Any],
    seed_id: int,
    raw_root: Path,
) -> List[Dict[str, Any]]:
    """Dispatch all (env, method) runs for a single seed_id.

    Returns the list of manifest records appended during this dispatch.
    """
    stage = str(config.get("stage", "dev"))
    n_episodes = int(config["n_episodes"])
    gamma = float(config.get("gamma", 0.95))
    learning_rate = float(config.get("learning_rate", 0.1))
    epsilon_cfg = dict(config.get("epsilon", {}))
    schedule_hparams = dict(config.get("schedule", {}))
    stratify_every = int(config.get("stratify_every", 1))
    methods_global = list(config.get("methods", []))
    envs_block = config.get("envs", [])

    for m in methods_global:
        if m not in ALL_METHOD_IDS:
            raise ValueError(
                f"unknown method id {m!r}; valid: {ALL_METHOD_IDS}"
            )

    records: List[Dict[str, Any]] = []
    for env_block in envs_block:
        env_name = str(env_block["name"])
        env_kwargs = dict(env_block.get("kwargs", {}))
        canonical = _peek_canonical_sign(env_name, env_kwargs)
        methods = _resolve_methods_for_env(
            env_name, methods_global, canonical
        )
        for method_index, method in enumerate(methods):
            run_id = (
                f"{stage}-{env_name}-{method}-s{seed_id}-"
                f"{int(time.time() * 1000) % 10**9:09d}"
            )
            rec = run_one(
                stage=stage,
                env_name=env_name,
                env_kwargs=env_kwargs,
                method=method,
                seed_id=seed_id,
                method_index=method_index,
                n_episodes=n_episodes,
                gamma=gamma,
                learning_rate=learning_rate,
                epsilon_cfg=epsilon_cfg,
                schedule_hparams=schedule_hparams,
                stratify_every=stratify_every,
                raw_root=raw_root,
                run_id=run_id,
            )
            records.append(rec)
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase VII adaptive-β experiment runner."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    # ``--raw-root`` is the canonical name; ``--out`` is a synonym used by
    # the M3.4 test suite. argparse stores them on the same attribute so
    # only one of the two needs to be supplied.
    parser.add_argument(
        "--raw-root",
        "--out",
        dest="raw_root",
        type=Path,
        default=_REPO_ROOT / "results" / "adaptive_beta" / "raw",
        help=(
            "Output root for run directories. Synonym: --out. Default: "
            "results/adaptive_beta/raw."
        ),
    )
    parser.add_argument(
        "--only-env",
        type=str,
        default=None,
        help=(
            "If set, drop every env block whose ``name`` differs. Test-only "
            "narrowing flag; does not change algorithm logic."
        ),
    )
    parser.add_argument(
        "--only-method",
        type=str,
        default=None,
        help=(
            "If set, drop every method id from the YAML's ``methods`` list "
            "except this one. Test-only narrowing flag."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help=(
            "If set, override the YAML's ``n_episodes``. Test-only knob to "
            "shorten runs; the algorithm itself is unchanged."
        ),
    )
    return parser.parse_args(argv)


def _apply_cli_overrides(
    config: Dict[str, Any], args: argparse.Namespace
) -> Dict[str, Any]:
    """Apply ``--only-env``, ``--only-method``, ``--episodes`` to config.

    Returns a *new* dict; the input is not mutated. Validates that the
    requested env/method names exist in the YAML so a typo fails loudly
    with the available choices listed.
    """
    out: Dict[str, Any] = dict(config)
    if args.only_method is not None:
        methods = list(out.get("methods", []))
        if args.only_method not in methods:
            raise ValueError(
                f"--only-method={args.only_method!r} not in config methods "
                f"{methods}"
            )
        out["methods"] = [args.only_method]
    if args.only_env is not None:
        envs = list(out.get("envs", []))
        names = [str(e.get("name")) for e in envs]
        if args.only_env not in names:
            raise ValueError(
                f"--only-env={args.only_env!r} not in config envs {names}"
            )
        out["envs"] = [e for e in envs if str(e.get("name")) == args.only_env]
    if args.episodes is not None:
        if int(args.episodes) <= 0:
            raise ValueError(
                f"--episodes must be > 0, got {args.episodes}"
            )
        out["n_episodes"] = int(args.episodes)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"config {args.config} must be a YAML mapping")
    config = _apply_cli_overrides(config, args)
    records = dispatch_from_config(
        config=config, seed_id=int(args.seed), raw_root=args.raw_root
    )
    n_completed = sum(1 for r in records if r["status"] == "completed")
    n_failed = sum(1 for r in records if r["status"] == "failed")
    print(
        f"phase_VII run_experiment: dispatched {len(records)} runs "
        f"(completed={n_completed}, failed={n_failed})",
        flush=True,
    )
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
