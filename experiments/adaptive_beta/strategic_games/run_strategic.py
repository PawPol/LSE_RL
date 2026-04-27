"""Phase VII-B strategic-learning experiment runner.

CLI entrypoint that turns a Stage B2 YAML config into a fan-out matrix
of ``(game, adversary, method, seed)`` runs. Each cell produces a
self-contained directory under
``<output_dir>/<game>_<adversary>_<method>/seed_<seed_id>/`` with:

- ``run.json`` — manifest: git SHA, argv, seed, resolved config slice,
  start/end timestamps, status.
- ``metrics.npz`` — per-episode arrays + schema header.
- ``episodes.csv`` — Phase VII §7.4 episode schema.
- ``transitions.parquet`` — Phase VII §7.4 transition schema (stratified).

Manifest contract (Phase VII-B spec §13–§14)
---------------------------------------------
Per-stage append-only manifest at ``<output_dir>/manifest.json``. Every
``(game, adversary, method, seed)`` cell present in the matrix appears
in the manifest with status one of ``completed | failed | skipped``.
``wrong_sign`` and ``adaptive_magnitude_only`` against a game with
``env_canonical_sign = None`` are recorded as ``status="skipped"`` with
``reason="§22.3 — no canonical sign for game X"`` — never silently
omitted.

This is a SEPARATE manifest from Phase VII-A's
``results/summaries/phase_VII_manifest.json`` global file. The Phase
VII-A runner is left untouched.

Reuse contract
--------------
- The agent (``AdaptiveBetaQAgent``) and ε / β schedule factories are
  imported from ``experiments.adaptive_beta.{agents,schedules}`` —
  identical code path to Phase VII-A; only the schedule object differs
  per method.
- The episode and transition loggers are imported from
  ``experiments.adaptive_beta.logging_callbacks`` — schema parity with
  Phase VII-A.
- Game and adversary instances are created via the strategic-games
  registries (``GAME_REGISTRY`` / ``ADVERSARY_REGISTRY``).

Seed protocol (Phase VII-B prompt; matches parent §8.4 with method
hashing)
--------------------------------------------------------------------
``base_seed = 10000 + seed_id``
``common_env_seed = base_seed``     # paired across methods
``agent_seed = base_seed + (stable_hash(method) % 1000)``

``stable_hash`` is a SHA-256 digest of the method name (NOT Python's
``hash`` builtin, which is randomised per-process). This guarantees the
same method always lands on the same agent-seed offset across runs and
hosts.

Usage
-----
::

    python -m experiments.adaptive_beta.strategic_games.run_strategic \\
        --config experiments/adaptive_beta/strategic_games/configs/stage_B2_dev.yaml

    # Dry-run a single cell:
    python -m experiments.adaptive_beta.strategic_games.run_strategic \\
        --config experiments/adaptive_beta/strategic_games/configs/stage_B2_dev.yaml \\
        --limit-cells 1
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml

# Repo root on the path for absolute imports when called directly.
_REPO_ROOT = Path(__file__).resolve().parents[3]
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
from experiments.adaptive_beta.run_experiment import (  # noqa: E402
    _PER_METHOD_HPARAM_KEYS,
    _filter_hparams_for_method,
)
from experiments.adaptive_beta.schedules import (  # noqa: E402
    ALL_METHOD_IDS,
    build_schedule,
)

# Importing the strategic_games package triggers registry population.
import experiments.adaptive_beta.strategic_games as _sg  # noqa: E402,F401
from experiments.adaptive_beta.strategic_games.logging import (  # noqa: E402
    SCHEMA_VERSION_EPISODES_STRATEGIC,
    StrategicLogger,
    episode_to_row,
)
from experiments.adaptive_beta.strategic_games.registry import (  # noqa: E402
    ADVERSARY_REGISTRY,
    GAME_REGISTRY,
    make_adversary,
    make_game,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MANIFEST_SCHEMA_VERSION = "phaseVII_B.runs.v1"
RUN_JSON_SCHEMA_VERSION = "phaseVII_B.run.v1"
# Per-cell manifest record schema (verifier MAJOR-2: every record carries
# its own ``schema_version`` so consumers can branch without inspecting
# the envelope).
CELL_RECORD_SCHEMA_VERSION = "phaseVII_B.cell.v1"

# Methods that are only meaningful when the game has a defined canonical
# sign (parent VII-A spec §22.3 / VII-B spec §8 rules).
_SIGN_DEPENDENT_METHODS: frozenset = frozenset(
    {"wrong_sign", "adaptive_magnitude_only"}
)

# Sentinel placeholder values that disqualify a config from dispatch
# (used by stage_B2_main.yaml until the promotion gate fills it).
_PLACEHOLDER_TOKENS: frozenset = frozenset(
    {"${promoted_games}", "${promoted_adversaries}"}
)


# ---------------------------------------------------------------------------
# Seed protocol
# ---------------------------------------------------------------------------
def stable_hash(method: str) -> int:
    """Stable, process-independent hash of a method name.

    Python's built-in ``hash`` is randomised per-process via
    ``PYTHONHASHSEED``, so it is unsafe for reproducible seeding. We
    therefore digest the UTF-8 bytes with SHA-256 and take the first 8
    bytes as an unsigned 64-bit integer. The result is identical across
    Python versions, hosts, and invocations.
    """
    digest = hashlib.sha256(method.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def resolve_seed_assignment(seed_id: int, method: str) -> Tuple[int, int, int]:
    """Phase VII-B seed protocol.

    Returns ``(base_seed, common_env_seed, agent_seed)``. The agent
    seed is offset from the base seed by ``stable_hash(method) % 1000``
    so different methods explore differently while sharing the same
    environment stream.
    """
    base = 10000 + int(seed_id)
    return base, base, base + (stable_hash(method) % 1000)


# ---------------------------------------------------------------------------
# Manifest helpers (per-output-dir append-only)
# ---------------------------------------------------------------------------
def _manifest_path(output_dir: Path) -> Path:
    return output_dir / "manifest.json"


def _load_manifest(path: Path) -> Dict[str, Any]:
    """Load the per-stage manifest dict. Empty / missing -> empty shell.

    Manifest schema (top-level)::

        {
          "schema_version": "phaseVII_B.runs.v1",
          "stage": <str>,
          "records": [<record>, ...]
        }

    Records are append-only; we never rewrite an existing record once
    written.
    """
    if not path.exists():
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "stage": None,
            "records": [],
        }
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        # Preserve corrupt manifest by renaming and start fresh.
        backup = path.with_suffix(f".corrupt.{int(time.time())}.json")
        path.rename(backup)
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "stage": None,
            "records": [],
        }
    if not isinstance(data, dict):
        raise ValueError(
            f"manifest at {path} must be a JSON object, got {type(data).__name__}"
        )
    data.setdefault("schema_version", MANIFEST_SCHEMA_VERSION)
    data.setdefault("stage", None)
    data.setdefault("records", [])
    if not isinstance(data["records"], list):
        raise ValueError(
            f"manifest 'records' field at {path} must be a list, "
            f"got {type(data['records']).__name__}"
        )
    return data


def _atomic_write_manifest(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def _append_manifest_record(
    record: Dict[str, Any],
    output_dir: Path,
    stage: str,
) -> None:
    """Append a single cell record to the per-stage manifest atomically.

    Read-modify-write append. NOT concurrent-safe; serialise dispatcher
    processes upstream if you parallelise.
    """
    path = _manifest_path(output_dir)
    data = _load_manifest(path)
    if data["stage"] is None:
        data["stage"] = stage
    elif data["stage"] != stage:
        # Different stages must not share a manifest file. Refuse rather
        # than corrupting the bookkeeping.
        raise ValueError(
            f"manifest at {path} already pinned to stage={data['stage']!r}; "
            f"refusing to append a record for stage={stage!r}. Use a "
            f"different output_dir."
        )
    data["records"].append(record)
    _atomic_write_manifest(data, path)


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------
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


def _relative_or_absolute(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(p.resolve())


# ---------------------------------------------------------------------------
# Game-module helpers (payoff matrices, factory kwarg filtering)
# ---------------------------------------------------------------------------
_GAME_MODULE_PREFIX = "experiments.adaptive_beta.strategic_games.games."


def _game_module(game_name: str) -> ModuleType:
    """Return the imported game module for ``game_name``.

    Spec §6 fixes the module path; if a future game is registered from a
    different namespace the lookup falls back to ``ImportError``.
    """
    return importlib.import_module(_GAME_MODULE_PREFIX + game_name)


def _game_payoff_matrices(game_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Fetch ``(payoff_agent, payoff_opponent)`` from the game module.

    All Phase VII-B games expose the matrices at module-level by spec
    convention (see games/matching_pennies.py header). This is the
    runner's single source of truth for adversaries that need access
    to the opponent payoff (best-response, hypothesis-testing, etc.).
    """
    mod = _game_module(game_name)
    pa = getattr(mod, "payoff_agent", None)
    po = getattr(mod, "payoff_opponent", None)
    if pa is None:
        raise AttributeError(
            f"game module {mod.__name__} does not expose 'payoff_agent' "
            f"at module level (Phase VII-B convention)"
        )
    if po is None:
        # Zero-sum default per the matrix-game env contract.
        po = -np.asarray(pa, dtype=np.float64)
    return np.asarray(pa, dtype=np.float64), np.asarray(po, dtype=np.float64)


# Documentation-only YAML keys that some game configs include for human
# clarity but the game factory itself does not consume (it derives the
# value from another field, typically ``horizon``). These are silently
# dropped by ``_filter_factory_kwargs`` regardless of factory signature.
# Adding to this set is a deliberate act of policy — extend it only
# when a new YAML key is genuinely informational and never consumed.
_DOC_ONLY_GAME_KWARGS: frozenset = frozenset(
    {
        # Auto-derived from ``horizon == 1`` in every game's ``build()``.
        "mechanism_degenerate",
    }
)


def _filter_factory_kwargs(
    factory: Any, kwargs: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[str]]:
    """Drop kwargs the factory does not accept.

    Returns ``(accepted_kwargs, dropped_keys)``. Two-stage filter:

    1. Always drop documentation-only YAML keys (see
       ``_DOC_ONLY_GAME_KWARGS``) — these are informational labels that
       the game factory either auto-derives or ignores. The strict
       ``**kwargs`` validation inside game ``build()`` would otherwise
       reject them.
    2. If the factory does NOT have a ``**kwargs`` catch-all, also drop
       any kwarg whose name is not in the factory's named-parameter set.
       Factories with ``**kwargs`` get everything else forwarded;
       they're responsible for their own strict validation.
    """
    out_kwargs = dict(kwargs)
    dropped: List[str] = []
    # Stage 1: documentation-only keys.
    for k in list(out_kwargs.keys()):
        if k in _DOC_ONLY_GAME_KWARGS:
            del out_kwargs[k]
            dropped.append(k)

    # Stage 2: signature-driven filter (only when factory rejects extras).
    try:
        sig = inspect.signature(factory)
    except (TypeError, ValueError):
        return out_kwargs, sorted(dropped)
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )
    if has_var_keyword:
        return out_kwargs, sorted(dropped)
    accepted_names = {
        name
        for name, p in sig.parameters.items()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    for k in list(out_kwargs.keys()):
        if k not in accepted_names:
            del out_kwargs[k]
            dropped.append(k)
    return out_kwargs, sorted(set(dropped))


def _build_adversary(
    adversary_name: str,
    *,
    n_opponent_actions: int,
    payoff_opponent: np.ndarray,
    seed: int,
    user_kwargs: Dict[str, Any],
) -> Any:
    """Construct an adversary, injecting ``payoff_opponent`` if its
    constructor accepts it (best-response, regret-matching,
    hypothesis-testing variants).

    The runner does not require the adversary class to be known
    statically; it inspects the factory signature to decide whether
    ``payoff_opponent`` is a valid kwarg.
    """
    factory = ADVERSARY_REGISTRY[adversary_name]
    base_kwargs: Dict[str, Any] = {
        "n_actions": int(n_opponent_actions),
        "seed": int(seed),
    }
    try:
        sig = inspect.signature(factory)
        accepted = set(sig.parameters)
        if "payoff_opponent" in accepted:
            base_kwargs["payoff_opponent"] = np.asarray(
                payoff_opponent, dtype=np.float64
            )
    except (TypeError, ValueError):
        # Best-effort: pass payoff_opponent unconditionally; the
        # factory's strict kwarg check will catch any mismatch.
        base_kwargs["payoff_opponent"] = np.asarray(
            payoff_opponent, dtype=np.float64
        )

    # Adversary kwargs from YAML take precedence (e.g. memory_m,
    # tolerance_tau). Don't allow YAML to override seed / n_actions —
    # those are runner-controlled.
    merged: Dict[str, Any] = dict(base_kwargs)
    for k, v in user_kwargs.items():
        if k in {"n_actions", "seed", "payoff_opponent"}:
            # YAML must not override runner-injected fields.
            continue
        merged[k] = v
    return make_adversary(adversary_name, **merged)


# ---------------------------------------------------------------------------
# Cell skip / sign-rule guard
# ---------------------------------------------------------------------------
# Static canonical-sign table for the 5 Phase VII-B games. Source: each
# game module's ``metadata['canonical_sign']`` literal (see games/*.py).
# Only ``asymmetric_coordination`` carries a non-None sign in this phase
# because its stag-hunt structure has a clear optimistic equilibrium.
_GAME_CANONICAL_SIGNS: Dict[str, Optional[str]] = {
    "matching_pennies": None,
    "shapley": None,
    "rules_of_road": None,
    "strategic_rps": None,
    "asymmetric_coordination": "+",
}


def _peek_canonical_sign(game_name: str) -> Optional[str]:
    """Static lookup of a game's canonical sign without env instantiation.

    Returns ``"+" / "-"`` or ``None``. Unknown games return ``None`` and
    cause sign-dependent methods to be skipped conservatively.
    """
    return _GAME_CANONICAL_SIGNS.get(game_name, None)


# ---------------------------------------------------------------------------
# Strategic-field helpers (Phase VII-B spec §13)
# ---------------------------------------------------------------------------
def _empirical_distribution(actions: List[int], n_actions: int) -> np.ndarray:
    """Empirical action distribution over an episode, shape ``(n_actions,)``.

    Returns a uniform NaN-free zero vector for empty input so the caller
    can short-circuit entropy / TV computations to NaN.
    """
    if not actions or n_actions <= 0:
        return np.zeros(int(n_actions), dtype=np.float64)
    arr = np.asarray(actions, dtype=np.int64)
    counts = np.bincount(arr, minlength=int(n_actions)).astype(np.float64)
    total = counts.sum()
    if total <= 0.0:
        return np.zeros(int(n_actions), dtype=np.float64)
    return counts / total


def _shannon_entropy_nats(probs: np.ndarray) -> float:
    """Shannon entropy in nats. Robust to zeros."""
    p = np.asarray(probs, dtype=np.float64)
    if p.size == 0:
        return float("nan")
    nz = p > 0.0
    if not np.any(nz):
        return 0.0
    return float(-np.sum(p[nz] * np.log(p[nz])))


def _tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Total-variation distance between two same-shape distributions."""
    return float(0.5 * np.sum(np.abs(np.asarray(p) - np.asarray(q))))


# ---------------------------------------------------------------------------
# Single-cell executor
# ---------------------------------------------------------------------------
def run_one_cell(
    *,
    stage: str,
    game_name: str,
    game_kwargs: Dict[str, Any],
    adversary_name: str,
    adversary_kwargs: Dict[str, Any],
    method: str,
    seed_id: int,
    n_episodes: int,
    gamma: float,
    learning_rate: float,
    epsilon_cfg: Dict[str, Any],
    schedule_hparams: Dict[str, Any],
    stratify_every: int,
    output_dir: Path,
    config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute one ``(game, adversary, method, seed)`` cell end-to-end.

    Returns the manifest record dict. Side effects: writes ``run.json``,
    ``metrics.npz``, ``episodes.csv``, ``transitions.parquet`` under the
    cell directory.
    """
    cell_id = (
        f"{game_name}__{adversary_name}__{method}__s{seed_id}"
    )
    cell_subdir = f"{game_name}_{adversary_name}_{method}"
    raw_dir = output_dir / cell_subdir / f"seed_{seed_id}"
    raw_dir.mkdir(parents=True, exist_ok=True)

    started_at = _utc_now()
    started_t = time.time()
    base_seed, common_env_seed, agent_seed = resolve_seed_assignment(
        seed_id, method
    )

    run_id = (
        f"{stage}-{game_name}-{adversary_name}-{method}-s{seed_id}-"
        f"{int(time.time() * 1000) % 10**9:09d}"
    )

    record: Dict[str, Any] = {
        "schema_version": CELL_RECORD_SCHEMA_VERSION,
        "cell_id": cell_id,
        "run_id": run_id,
        "stage": stage,
        "game": game_name,
        "adversary": adversary_name,
        "method": method,
        "seed_id": int(seed_id),
        "status": "running",
        "started_at": started_at,
        "completed_at": None,
        "wall_clock_s": None,
        "raw_dir": _relative_or_absolute(raw_dir),
        "n_episodes": int(n_episodes),
        "error_msg": None,
        "runtime_keys": None,
        "reason": None,
    }

    # Identity for the loggers. We pack game+adversary into the env tag
    # so downstream aggregations can split on it.
    env_tag = f"{game_name}|{adversary_name}"
    identity = _RunIdentity(
        run_id=run_id, env=env_tag, method=method, seed=int(seed_id)
    )

    try:
        # Validate registry membership up front (loud key errors, not
        # silent fallbacks — spec §4 + parent §22.3).
        if adversary_name not in ADVERSARY_REGISTRY:
            known = ", ".join(sorted(ADVERSARY_REGISTRY.keys()))
            raise KeyError(
                f"Unknown adversary {adversary_name!r}; registered: [{known}]"
            )
        if game_name not in GAME_REGISTRY:
            known = ", ".join(sorted(GAME_REGISTRY.keys())) or "(none yet)"
            raise KeyError(
                f"Unknown game {game_name!r}; registered: [{known}]"
            )

        # Pull payoff matrices from the game module so we can build
        # adversaries that need ``payoff_opponent`` (best-response,
        # finite-memory regret-matching, hypothesis-testing variants).
        payoff_agent, payoff_opponent = _game_payoff_matrices(game_name)
        n_opp_actions = int(payoff_opponent.shape[1])

        adv_seed = base_seed + (stable_hash(adversary_name) % 1000)
        adversary = _build_adversary(
            adversary_name,
            n_opponent_actions=n_opp_actions,
            payoff_opponent=payoff_opponent,
            seed=adv_seed,
            user_kwargs=adversary_kwargs,
        )

        # Filter game kwargs to those the factory accepts. Documentation-
        # only YAML keys (e.g. ``mechanism_degenerate`` on matching_pennies,
        # which the game derives from horizon) are dropped here so they
        # don't trip the strict ``**kwargs`` checks inside game factories.
        game_factory = GAME_REGISTRY[game_name]
        accepted_game_kwargs, dropped_game_kwargs = _filter_factory_kwargs(
            game_factory, game_kwargs
        )
        env = make_game(
            game_name,
            adversary=adversary,
            seed=common_env_seed,
            **accepted_game_kwargs,
        )
        n_states = int(env.info.observation_space.size[0])
        n_actions = int(env.info.action_space.size[0])
        canonical_sign = getattr(env, "env_canonical_sign", None)

        # ε-greedy schedule + β schedule (identical to Phase VII-A path).
        eps_fn = linear_epsilon_schedule(
            start=float(epsilon_cfg["start"]),
            end=float(epsilon_cfg["end"]),
            decay_episodes=int(epsilon_cfg["decay_episodes"]),
        )
        method_hparams = _filter_hparams_for_method(method, schedule_hparams)
        schedule = build_schedule(method, canonical_sign, method_hparams)
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
        # Phase VII-B §13 strategic-schema logger runs in parallel to the
        # VII-A loggers above (does NOT replace them). Its output goes to
        # ``episodes_strategic.csv`` so VII-A consumers keep working
        # against ``episodes.csv`` unchanged.
        strategic_logger = StrategicLogger(
            identity,
            game=game_name,
            adversary=adversary_name,
            parent_episode_logger=ep_logger,
            parent_transition_logger=tr_logger,
            stratify_every=stratify_every,
        )

        # Pull adversary knobs once (constant across the run); §13
        # requires per-episode logging for analysis convenience but the
        # values do not change. Missing fields are NaN per spec.
        try:
            adv_info_seed = adversary.info()
        except Exception:
            adv_info_seed = {}
        adv_memory_m = adv_info_seed.get("memory_m", None)
        adv_inertia_lambda = adv_info_seed.get("inertia_lambda", None)
        adv_temperature = adv_info_seed.get("temperature", None)
        adv_tau = adv_info_seed.get("tolerance_tau", None)
        # ``tau`` falls back to the adversary's "tau" key if it doesn't
        # expose tolerance_tau (different families use different names;
        # the spec column is always ``tau``).
        if adv_tau is None:
            adv_tau = adv_info_seed.get("tau", None)

        # Cumulative bookkeeping for the strategic schema.
        auc_running = 0.0
        prev_opp_dist: Optional[np.ndarray] = None  # shape (n_opp_actions,)
        # Buffers for the strategic-only npz arrays.
        strat_auc: List[float] = []
        strat_mean_d_eff: List[float] = []
        strat_opp_entropy: List[float] = []
        strat_policy_tv: List[float] = []
        strat_support_shift: List[bool] = []
        strat_model_rejected: List[bool] = []
        strat_search_phase: List[bool] = []
        strat_nan_count: List[int] = []
        strat_diverged: List[bool] = []

        # ----- Episode loop (mirrors run_experiment.run_one) -----
        for ep in range(int(n_episodes)):
            agent.begin_episode(ep)
            state, info = env.reset()
            s = int(np.asarray(state).flat[0])
            t = 0
            ep_return = 0.0
            ep_catastrophe = False
            ep_success = False
            shift_event = bool(info.get("is_shift_step", False))
            # Per-episode strategic buffers.
            ep_opp_actions: List[int] = []
            ep_d_eff: List[float] = []
            ep_model_rejected = False  # latches True if any step rejected
            ep_search_phase = False    # latches True if any step in search
            last_adv_info: Dict[str, Any] = {}

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
                # Strategic-only per-step bookkeeping.
                opp_action = step_info.get("opponent_action", None)
                if opp_action is not None:
                    ep_opp_actions.append(int(opp_action))
                ep_d_eff.append(float(step_diag["d_eff"]))
                adv_info_step = step_info.get("adversary_info", {}) or {}
                if bool(adv_info_step.get("model_rejected", False)):
                    ep_model_rejected = True
                if bool(adv_info_step.get("search_phase", False)):
                    ep_search_phase = True
                last_adv_info = adv_info_step
                t += 1
                s = ns
                if absorbing:
                    break

            ep_diag = agent.end_episode(ep)
            # Strategic-games regret is undefined without an oracle; we
            # propagate NaN so downstream aggregations stay NaN-aware.
            regret = float("nan")
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

            # ----- Strategic schema record (spec §13) -----
            # Cumulative AUC (sum of episode returns inclusive of this ep).
            auc_running += float(ep_return)
            # Mean effective discount aggregated from per-transition log.
            if ep_d_eff:
                mean_d_eff_ep = float(np.mean(ep_d_eff))
            else:
                mean_d_eff_ep = float("nan")
            # Opponent empirical distribution + Shannon entropy (nats).
            opp_dist = _empirical_distribution(ep_opp_actions, n_opp_actions)
            if ep_opp_actions:
                opp_entropy = _shannon_entropy_nats(opp_dist)
            else:
                opp_entropy = float("nan")
            # TV vs previous episode (NaN on episode 0).
            if prev_opp_dist is None:
                policy_tv = float("nan")
                support_shift = False
            else:
                policy_tv = _tv_distance(opp_dist, prev_opp_dist)
                support_shift = bool(policy_tv > 0.1)
            prev_opp_dist = opp_dist
            # nan_count / diverged from agent diagnostics (mirrored).
            nan_count = int(ep_diag.get("nan_count", 0))
            diverged = bool(ep_diag.get("divergence_event", False))

            strat_row = episode_to_row(
                run_id=run_id,
                seed=int(seed_id),
                game=game_name,
                adversary=adversary_name,
                method=method,
                episode=ep,
                episode_return=ep_return,
                auc_so_far=auc_running,
                beta=float(ep_diag["beta_deployed"]),
                alignment_rate=float(ep_diag["alignment_rate"]),
                mean_effective_discount=mean_d_eff_ep,
                bellman_residual=float(ep_diag["bellman_residual"]),
                catastrophic=bool(ep_catastrophe),
                diverged=diverged,
                nan_count=nan_count,
                opponent_policy_entropy=opp_entropy,
                policy_total_variation=policy_tv,
                support_shift=support_shift,
                model_rejected=bool(ep_model_rejected),
                search_phase=bool(ep_search_phase),
                phase=str(last_adv_info.get("phase") or info.get("phase") or ""),
                memory_m=adv_memory_m,
                inertia_lambda=adv_inertia_lambda,
                temperature=adv_temperature,
                tau=adv_tau,
            )
            strategic_logger.record_episode_strategic(strat_row)

            strat_auc.append(auc_running)
            strat_mean_d_eff.append(mean_d_eff_ep)
            strat_opp_entropy.append(opp_entropy)
            strat_policy_tv.append(policy_tv)
            strat_support_shift.append(support_shift)
            strat_model_rejected.append(bool(ep_model_rejected))
            strat_search_phase.append(bool(ep_search_phase))
            strat_nan_count.append(nan_count)
            strat_diverged.append(diverged)

        # ----- Persist artifacts -----
        # VII-A schema (backward compat): episodes.csv + transitions.parquet.
        n_eps_written = ep_logger.flush_csv(raw_dir / "episodes.csv")
        n_tr_written = tr_logger.flush_parquet(
            raw_dir / "transitions.parquet"
        )
        # VII-B §13 strategic schema: episodes_strategic.csv (parallel,
        # additive). The transitions parquet already covers the VII-A
        # schema; the strategic-transition file is not part of this
        # blocker fix.
        n_eps_strategic_written = strategic_logger.flush_episodes_csv(
            raw_dir / "episodes_strategic.csv"
        )

        ep_arrays = ep_logger.collected_arrays()
        # ``schema_version`` retains the VII-A v1 marker for backward
        # compatibility; ``schema_version_strategic`` is the new VII-B
        # marker. Downstream consumers branch on whichever they
        # recognise.
        np.savez(
            raw_dir / "metrics.npz",
            schema_version=np.array(SCHEMA_VERSION_EPISODES),
            schema_version_strategic=np.array(
                SCHEMA_VERSION_EPISODES_STRATEGIC
            ),
            auc_so_far=np.asarray(strat_auc, dtype=np.float64),
            mean_effective_discount=np.asarray(
                strat_mean_d_eff, dtype=np.float64
            ),
            opponent_policy_entropy=np.asarray(
                strat_opp_entropy, dtype=np.float64
            ),
            policy_total_variation=np.asarray(
                strat_policy_tv, dtype=np.float64
            ),
            support_shift=np.asarray(strat_support_shift, dtype=bool),
            model_rejected=np.asarray(strat_model_rejected, dtype=bool),
            search_phase=np.asarray(strat_search_phase, dtype=bool),
            nan_count=np.asarray(strat_nan_count, dtype=np.int64),
            diverged=np.asarray(strat_diverged, dtype=bool),
            **{
                k: v for k, v in ep_arrays.items()
                if k not in {"run_id", "env", "method", "phase"}
            },
        )

        run_json = {
            "schema_version": RUN_JSON_SCHEMA_VERSION,
            "run_id": run_id,
            "cell_id": cell_id,
            "stage": stage,
            "game": game_name,
            "game_kwargs": game_kwargs,
            "game_kwargs_dropped": dropped_game_kwargs,
            "adversary": adversary_name,
            "adversary_kwargs": adversary_kwargs,
            "method": method,
            "seed_id": int(seed_id),
            "common_env_seed": int(common_env_seed),
            "agent_seed": int(agent_seed),
            "adversary_seed": int(adv_seed),
            "n_episodes": int(n_episodes),
            "gamma": float(gamma),
            "learning_rate": float(learning_rate),
            "epsilon_cfg": dict(epsilon_cfg),
            "schedule_hparams": dict(schedule_hparams),
            "stratify_every": int(stratify_every),
            "config_path": (
                str(config_path) if config_path is not None else None
            ),
            "argv": list(sys.argv),
            "git_sha": _git_sha(),
            "host": socket.gethostname(),
            "python_version": sys.version.split()[0],
            "started_at": started_at,
            "completed_at": _utc_now(),
            "n_episodes_written": int(n_eps_written),
            "n_episodes_strategic_written": int(n_eps_strategic_written),
            "n_transitions_written": int(n_tr_written),
            "transitions_schema_version": SCHEMA_VERSION_TRANSITIONS,
            "metrics_schema_version": SCHEMA_VERSION_EPISODES,
            "metrics_schema_version_strategic": (
                SCHEMA_VERSION_EPISODES_STRATEGIC
            ),
            "env_canonical_sign": canonical_sign,
        }
        with open(raw_dir / "run.json", "w", encoding="utf-8") as f:
            json.dump(run_json, f, indent=2, sort_keys=True)
            f.write("\n")

        record["status"] = "completed"
        record["completed_at"] = run_json["completed_at"]
        record["wall_clock_s"] = float(time.time() - started_t)
        record["runtime_keys"] = sorted(
            ["common_env_seed", "agent_seed", "adversary_seed", "git_sha"]
        )
        return record

    except Exception as exc:
        record["status"] = "failed"
        record["completed_at"] = _utc_now()
        record["wall_clock_s"] = float(time.time() - started_t)
        record["error_msg"] = f"{type(exc).__name__}: {exc}"
        # Persist the traceback alongside whatever partial outputs exist.
        with open(raw_dir / "FAILURE.log", "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        return record
    finally:
        _append_manifest_record(record, output_dir, stage)


# ---------------------------------------------------------------------------
# Skipped-cell record (keeps manifest accounting honest per spec §2.3)
# ---------------------------------------------------------------------------
def _record_skipped_cell(
    *,
    stage: str,
    game_name: str,
    adversary_name: str,
    method: str,
    seed_id: int,
    output_dir: Path,
    reason: str,
) -> Dict[str, Any]:
    """Append a ``status='skipped'`` manifest record for a cell that
    cannot be dispatched (e.g. ``wrong_sign`` against a game with no
    canonical sign).

    NO artifacts are written under the cell directory. The manifest
    entry is the *only* trace of the skipped cell.
    """
    cell_id = f"{game_name}__{adversary_name}__{method}__s{seed_id}"
    base_seed, common_env_seed, agent_seed = resolve_seed_assignment(
        seed_id, method
    )
    record: Dict[str, Any] = {
        "schema_version": CELL_RECORD_SCHEMA_VERSION,
        "cell_id": cell_id,
        "run_id": None,
        "stage": stage,
        "game": game_name,
        "adversary": adversary_name,
        "method": method,
        "seed_id": int(seed_id),
        "status": "skipped",
        "started_at": _utc_now(),
        "completed_at": _utc_now(),
        "wall_clock_s": 0.0,
        "raw_dir": None,
        "n_episodes": 0,
        "error_msg": None,
        "runtime_keys": None,
        "reason": reason,
    }
    _append_manifest_record(record, output_dir, stage)
    return record


# ---------------------------------------------------------------------------
# Config-driven dispatch
# ---------------------------------------------------------------------------
def _normalise_blocks(value: Any, label: str) -> Dict[str, Dict[str, Any]]:
    """Normalise a YAML ``games``/``adversaries`` block.

    Accepts either:
    - a mapping ``{name: kwargs_dict}`` (preferred Stage B2 form),
    - a list of strings ``[name, ...]`` (treated as ``{name: {}}``),
    - a list of mappings ``[{name: kwargs_dict}, ...]``.

    Returns a dict keyed by name. Raises on duplicate keys or a
    placeholder sentinel.
    """
    if isinstance(value, str):
        if value.strip() in _PLACEHOLDER_TOKENS:
            raise ValueError(
                f"{label} block is still a placeholder ({value!r}); "
                f"refuse to dispatch. Fill in the promotion-gate verdict "
                f"first."
            )
        raise ValueError(
            f"{label} block must be a mapping or list, got string {value!r}"
        )
    if value is None:
        return {}
    if isinstance(value, dict):
        if not value:
            raise ValueError(f"{label} block is empty; nothing to dispatch.")
        out: Dict[str, Dict[str, Any]] = {}
        for k, v in value.items():
            out[str(k)] = dict(v) if v else {}
        return out
    if isinstance(value, list):
        if not value:
            raise ValueError(f"{label} block is empty; nothing to dispatch.")
        out = {}
        for item in value:
            if isinstance(item, str):
                if item in out:
                    raise ValueError(f"duplicate {label} entry: {item!r}")
                out[item] = {}
            elif isinstance(item, dict):
                if len(item) != 1:
                    raise ValueError(
                        f"{label} list entry must have exactly one key, "
                        f"got {item!r}"
                    )
                (k, v), = item.items()
                if k in out:
                    raise ValueError(f"duplicate {label} entry: {k!r}")
                out[str(k)] = dict(v) if v else {}
            else:
                raise ValueError(
                    f"{label} list entry must be str or single-key dict, "
                    f"got {type(item).__name__}"
                )
        return out
    raise ValueError(
        f"{label} block must be a mapping or list, got {type(value).__name__}"
    )


def _epsilon_cfg_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Read either the flat ``epsilon_*`` keys (Stage B2 schema per
    Phase VII-B prompt) or the nested ``epsilon: {start, end,
    decay_episodes}`` block (Phase VII-A back-compat).
    """
    if "epsilon" in config and isinstance(config["epsilon"], dict):
        eps = config["epsilon"]
        return {
            "start": float(eps.get("start", 1.0)),
            "end": float(eps.get("end", 0.05)),
            "decay_episodes": int(eps.get("decay_episodes", 5000)),
        }
    return {
        "start": float(config.get("epsilon_start", 1.0)),
        "end": float(config.get("epsilon_end", 0.05)),
        "decay_episodes": int(config.get("epsilon_decay_episodes", 5000)),
    }


def _schedule_hparams_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Pull β-schedule hyperparameters from the flat Stage B2 schema."""
    keys = (
        "beta_max",
        "beta_cap",
        "k",
        "initial_beta",
        "beta_tol",
        "lambda_smooth",
        "beta0",
    )
    out: Dict[str, Any] = {}
    for k in keys:
        if k in config:
            out[k] = config[k]
    # Phase VII-A back-compat: support nested ``schedule:`` block too.
    if isinstance(config.get("schedule"), dict):
        for k, v in config["schedule"].items():
            out.setdefault(k, v)
    return out


def iter_cells(config: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any], str, Dict[str, Any], str, int]]:
    """Generate ``(game, game_kwargs, adv, adv_kwargs, method, seed_id)``
    tuples for the full matrix declared in ``config``.

    Order is: games (config order) × adversaries (config order) ×
    methods (config order) × seeds (config order). Stable ordering
    matters for reproducible manifests.
    """
    games = _normalise_blocks(config.get("games"), "games")
    advs = _normalise_blocks(config.get("adversaries"), "adversaries")
    methods = list(config.get("methods", []))
    seeds = list(config.get("seeds", []))
    for g_name, g_kwargs in games.items():
        for a_name, a_kwargs in advs.items():
            for m in methods:
                for sid in seeds:
                    yield g_name, dict(g_kwargs), a_name, dict(a_kwargs), m, int(sid)


def dispatch_from_config(
    config: Dict[str, Any],
    *,
    output_dir: Path,
    config_path: Optional[Path] = None,
    limit_cells: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Dispatch every (game, adversary, method, seed) cell in ``config``.

    Returns the list of manifest records appended during this dispatch.
    Skipped cells (canonical-sign rule) are recorded explicitly in the
    manifest.
    """
    stage = str(config.get("stage", "B2_dev"))
    n_episodes = int(config["episodes"])
    gamma = float(config.get("gamma", 0.95))
    learning_rate = float(config.get("learning_rate", 0.1))
    stratify_every = int(config.get("stratify_every", 1))
    methods = list(config.get("methods", []))
    epsilon_cfg = _epsilon_cfg_from_config(config)
    schedule_hparams = _schedule_hparams_from_config(config)

    # Validate methods up front (fail loud on typos).
    for m in methods:
        if m not in ALL_METHOD_IDS:
            raise ValueError(
                f"unknown method id {m!r}; valid: {ALL_METHOD_IDS}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []

    cells = list(iter_cells(config))
    if limit_cells is not None:
        if limit_cells < 1:
            raise ValueError(
                f"--limit-cells must be >= 1, got {limit_cells}"
            )
        cells = cells[: int(limit_cells)]

    for g_name, g_kwargs, a_name, a_kwargs, method, seed_id in cells:
        # Sign-rule skip (defensive — Stage B2 method list does not
        # currently include sign-dependent methods, but a future config
        # tweak might).
        if method in _SIGN_DEPENDENT_METHODS:
            sign = _peek_canonical_sign(g_name)
            if sign is None:
                rec = _record_skipped_cell(
                    stage=stage,
                    game_name=g_name,
                    adversary_name=a_name,
                    method=method,
                    seed_id=seed_id,
                    output_dir=output_dir,
                    reason=(
                        f"§22.3 — no canonical sign for game {g_name!r}; "
                        f"method {method!r} is undefined here."
                    ),
                )
                records.append(rec)
                continue

        rec = run_one_cell(
            stage=stage,
            game_name=g_name,
            game_kwargs=g_kwargs,
            adversary_name=a_name,
            adversary_kwargs=a_kwargs,
            method=method,
            seed_id=seed_id,
            n_episodes=n_episodes,
            gamma=gamma,
            learning_rate=learning_rate,
            epsilon_cfg=epsilon_cfg,
            schedule_hparams=schedule_hparams,
            stratify_every=stratify_every,
            output_dir=output_dir,
            config_path=config_path,
        )
        records.append(rec)

    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase VII-B strategic-learning experiment runner."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output root for cell directories and manifest. Defaults to "
            "the YAML's ``output_dir`` field."
        ),
    )
    parser.add_argument(
        "--limit-cells",
        type=int,
        default=None,
        help=(
            "Dry-run / smoke knob: cap the number of dispatched cells. "
            "Used by the verifier flow to validate end-to-end wiring "
            "without burning the full matrix."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help=(
            "Override the YAML ``episodes`` field. Test-only knob to "
            "shorten cells without editing the config."
        ),
    )
    return parser.parse_args(argv)


def _resolve_output_dir(
    cli_arg: Optional[Path], config: Dict[str, Any]
) -> Path:
    """Pick the output directory: CLI flag wins, then YAML."""
    if cli_arg is not None:
        return cli_arg.resolve()
    yaml_dir = config.get("output_dir")
    if not yaml_dir:
        raise ValueError(
            "no output_dir specified (set --output-dir or 'output_dir' "
            "in the YAML)"
        )
    p = Path(str(yaml_dir))
    if not p.is_absolute():
        p = (_REPO_ROOT / p).resolve()
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"config {args.config} must be a YAML mapping")

    if args.episodes is not None:
        if int(args.episodes) <= 0:
            raise ValueError(
                f"--episodes must be > 0, got {args.episodes}"
            )
        config["episodes"] = int(args.episodes)

    output_dir = _resolve_output_dir(args.output_dir, config)
    records = dispatch_from_config(
        config=config,
        output_dir=output_dir,
        config_path=args.config,
        limit_cells=args.limit_cells,
    )
    n_completed = sum(1 for r in records if r["status"] == "completed")
    n_failed = sum(1 for r in records if r["status"] == "failed")
    n_skipped = sum(1 for r in records if r["status"] == "skipped")
    print(
        f"phase_VII_B run_strategic: dispatched {len(records)} cells "
        f"(completed={n_completed}, failed={n_failed}, "
        f"skipped={n_skipped}) -> {output_dir}",
        flush=True,
    )
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
