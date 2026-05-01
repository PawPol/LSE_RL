"""Phase VIII Stage 1 — fixed-β operator sweep runner (M6 main pass).

Spec authority
--------------
- ``docs/specs/phase_VIII_tab_six_games.md`` §10.1 (Stage A dev pass) and
  §10.2 (Stage 1 main pass).
- §8.1 logging schema (run.json + metrics.npz contract).
- §8.2 Phase8RunRoster (every requested run is registered).
- §13.6 result-root regression (route through
  ``results/adaptive_beta/tab_six_games`` explicitly; never inherit
  ``results/weighted_lse_dp``).
- ``tasks/lessons.md`` #11 default out-root rule: callers pass the bare
  result root; ``make_run_dir`` builds the
  ``phase/suite/task/algorithm/seed`` hierarchy.
- M2 v5b headline routing: delayed_chain advance-only subcases use the
  β-specific Bellman residual AUC (``auc_neg_log_residual`` over
  ``bellman_residual_beta`` against the canonical TAB target via
  ``src.lse_rl.operator.tab_operator.g``); DC-Branching20 and the matrix
  games use cumulative-return AUC.
- M2 v5b T11 guard routing: deterministic cells (DC advance-only) flag
  ``t11_guard: gap_based``; stochastic cells (matrix games,
  DC-Branching20) flag ``t11_guard: cohens_d``.

Boundaries
----------
- Reuses, never duplicates: :class:`AdaptiveBetaQAgent`,
  :func:`linear_epsilon_schedule`, :func:`build_schedule` from
  :mod:`experiments.adaptive_beta.agents` /
  :mod:`experiments.adaptive_beta.schedules`; the strategic-games
  registries (``GAME_REGISTRY`` / ``ADVERSARY_REGISTRY``); the Phase
  VIII delta metrics from
  :mod:`experiments.adaptive_beta.tab_six_games.metrics`; the run-dir
  helper ``make_run_dir`` and the schema'd npz writer
  ``save_npz_with_schema`` from
  :mod:`experiments.weighted_lse_dp.common.io`; and
  :class:`Phase8RunRoster` from
  :mod:`experiments.adaptive_beta.tab_six_games.manifests`.
- Does NOT touch ``mushroom-rl-dev/`` (CLAUDE.md §4).
- Does NOT recompute or alter operator math (single source of truth in
  ``src/lse_rl/operator/tab_operator.py``).

Method ID convention (Phase VIII)
---------------------------------
The Phase VIII method names ``vanilla`` and ``fixed_beta_<signed_value>``
are unique to this stage. They are mapped to the
:func:`build_schedule` factory IDs as follows:

* ``vanilla`` → :data:`METHOD_VANILLA` (``ZeroBetaSchedule``).
* ``fixed_beta_+x`` → :data:`METHOD_FIXED_POSITIVE` with ``beta0=x``.
* ``fixed_beta_-x`` → :data:`METHOD_FIXED_NEGATIVE` with ``beta0=x``.

The mapping is performed exclusively by :func:`_resolve_method_to_schedule`
so the rest of the runner is method-agnostic.

CLI
---
::

    python -m experiments.adaptive_beta.tab_six_games.runners.\
run_phase_VIII_stage1_beta_sweep \
        --config experiments/adaptive_beta/tab_six_games/configs/dev.yaml \
        --seed-list 0,1,2

The runner is intentionally serial. Parallel dispatch is out of scope
for M6 wave 1; the smoke test exercises the full single-process path
end-to-end.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import socket
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

# Repo root on the path for absolute imports when called directly.
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.adaptive_beta.agents import (  # noqa: E402
    AdaptiveBetaQAgent,
    linear_epsilon_schedule,
)
from experiments.adaptive_beta.schedules import (  # noqa: E402
    METHOD_FIXED_NEGATIVE,
    METHOD_FIXED_POSITIVE,
    METHOD_VANILLA,
    build_schedule,
)
from experiments.adaptive_beta.tab_six_games.manifests import (  # noqa: E402
    Phase8RunRoster,
)
from experiments.weighted_lse_dp.common.io import (  # noqa: E402
    make_npz_schema,
    make_run_dir,
    save_npz_with_schema,
)

# Importing the strategic_games package triggers the registry population.
import experiments.adaptive_beta.strategic_games as _sg  # noqa: E402,F401
from experiments.adaptive_beta.strategic_games.registry import (  # noqa: E402
    ADVERSARY_REGISTRY,
    GAME_REGISTRY,
    make_adversary,
    make_game,
)



# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------
RUN_JSON_SCHEMA_VERSION: str = "phaseVIII.run.v1"
METRICS_NPZ_SCHEMA_VERSION: str = "phaseVIII.metrics.v1"

#: Canonical Phase VIII result root (lessons.md #11 + spec §13.6). The
#: runner refuses to inherit ``results/weighted_lse_dp/``.
PHASE8_RESULT_ROOT: Path = Path("results/adaptive_beta/tab_six_games")

#: Per-episode array names always written to ``metrics.npz``. The smoke
#: test verifies the union ``REQUIRED_METRICS`` ⊆ keys-on-disk.
REQUIRED_METRICS: Tuple[str, ...] = (
    "return",
    "bellman_residual",
    "beta_used",
    "beta_raw",
    "alignment_rate",
    "effective_discount_mean",
    "q_abs_max",
)


# ---------------------------------------------------------------------------
# Subcase + cell descriptors (parsed from YAML)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SubcaseSpec:
    """One ``(game, subcase)`` cell description, parsed from YAML.

    ``subcase_id`` is the human-readable subcase identifier (e.g.
    ``MP-Stationary``, ``DC-Long50``). ``game`` is the registered game
    name (e.g. ``matching_pennies``). ``game_kwargs`` are forwarded to
    the game's ``build()`` factory; ``adversary_name`` and
    ``adversary_kwargs`` parameterize the opponent.

    ``headline_metric`` and ``t11_guard`` are routed through to the
    aggregator via ``run.json``; the runner does not interpret them
    beyond logging them, but downstream verifiers / aggregators key off
    them.
    """

    subcase_id: str
    game: str
    game_kwargs: Dict[str, Any]
    adversary_name: str
    adversary_kwargs: Dict[str, Any]
    headline_metric: str
    t11_guard: str

    def cell_label(self) -> str:
        """``"<game>/<subcase>"`` — used as the ``task`` for run paths."""
        return f"{self.game}/{self.subcase_id}"


# ---------------------------------------------------------------------------
# Method ID resolver
# ---------------------------------------------------------------------------
def _resolve_method_to_schedule(method_id: str) -> Tuple[str, Dict[str, Any]]:
    """Map a Phase VIII method name to ``(schedule_method_id, hparams)``.

    Phase VIII names are stage-specific; they never appear in
    :data:`schedules.ALL_METHOD_IDS`. The resolver is the single
    translation point so the rest of the runner can be method-agnostic.

    Supported names
    ---------------
    * ``vanilla`` → ``METHOD_VANILLA``, no hparams.
    * ``fixed_beta_+<x>`` → ``METHOD_FIXED_POSITIVE`` with ``beta0=x``.
    * ``fixed_beta_-<x>`` → ``METHOD_FIXED_NEGATIVE`` with ``beta0=x``.

    The numeric magnitude ``x`` is parsed as ``float(x)`` and must be
    strictly positive (the sign is encoded in the prefix). Raises
    :class:`ValueError` on any other shape so configs typos fail loudly.
    """
    name = str(method_id).strip()
    if name == "vanilla":
        return METHOD_VANILLA, {}

    if not name.startswith("fixed_beta_"):
        raise ValueError(
            f"unknown Phase VIII method_id={method_id!r}; expected "
            f"'vanilla' or 'fixed_beta_<signed_value>'"
        )

    payload = name[len("fixed_beta_"):]
    if not payload:
        raise ValueError(
            f"malformed method_id={method_id!r}; empty signed-value suffix"
        )
    sign_char = payload[0]
    if sign_char not in ("+", "-"):
        raise ValueError(
            f"malformed method_id={method_id!r}; signed-value suffix must "
            f"start with '+' or '-' (got {sign_char!r})"
        )
    try:
        magnitude = float(payload[1:])
    except ValueError as exc:
        raise ValueError(
            f"malformed method_id={method_id!r}; could not parse magnitude "
            f"from {payload[1:]!r}"
        ) from exc
    if magnitude <= 0.0:
        raise ValueError(
            f"malformed method_id={method_id!r}; magnitude must be > 0 "
            f"(got {magnitude}); use 'vanilla' for β = 0"
        )

    schedule_method = (
        METHOD_FIXED_POSITIVE if sign_char == "+" else METHOD_FIXED_NEGATIVE
    )
    return schedule_method, {"beta0": float(magnitude)}


# ---------------------------------------------------------------------------
# Adversary construction (Phase VIII variant)
# ---------------------------------------------------------------------------
_GAME_MODULE_PREFIX = "experiments.adaptive_beta.strategic_games.games."

# Registry keys that have NO dedicated module file but delegate to a
# different module's payoff matrices. Per HALT 7 fix (2026-05-01):
# `rules_of_road_sparse` is registered in registry.py but only exists as
# a wrapper around `rules_of_road.build(..., sparse_terminal=True)`. The
# payoff structure is identical to dense RR (sparsity zeroes per-step
# rewards but the (Stag, Hare) payoff matrix is the same), so opponents
# requiring payoff_opponent must read it from the dense module.
_PAYOFF_ALIAS: Dict[str, str] = {
    "rules_of_road_sparse": "rules_of_road",
}


def _import_game_module(game_name: str) -> ModuleType:
    """Import the game module by registered name (with alias support)."""
    name = _PAYOFF_ALIAS.get(game_name, game_name)
    return importlib.import_module(_GAME_MODULE_PREFIX + name)


def _resolve_payoff_opponent(
    game_name: str,
    n_actions: int,
) -> Optional[np.ndarray]:
    """Best-effort lookup of the opponent payoff matrix.

    Matrix games expose ``payoff_agent`` / ``payoff_opponent`` at module
    level (Phase VII-B convention). Non-matrix games (e.g.
    ``delayed_chain``) do not; in that case we return ``None`` and the
    caller skips ``payoff_opponent`` injection.

    Registry aliases like ``rules_of_road_sparse`` resolve to their
    underlying module via :data:`_PAYOFF_ALIAS` so opponents requiring
    ``payoff_opponent`` (e.g. ``finite_memory_fictitious_play``,
    ``hypothesis_testing``) can be built against the alias's payoff
    structure.
    """
    try:
        mod = _import_game_module(game_name)
    except ImportError:
        return None
    pa = getattr(mod, "payoff_agent", None)
    po = getattr(mod, "payoff_opponent", None)
    if po is not None:
        return np.asarray(po, dtype=np.float64)
    if pa is not None:
        # Zero-sum default per the matrix-game env contract.
        return -np.asarray(pa, dtype=np.float64)
    return None


def _build_adversary(
    *,
    adversary_name: str,
    n_actions: int,
    payoff_opponent: Optional[np.ndarray],
    seed: int,
    user_kwargs: Mapping[str, Any],
) -> Any:
    """Construct an adversary, auto-supplying signature-driven kwargs.

    Behavior mirrors
    :func:`experiments.adaptive_beta.strategic_games.run_strategic._build_adversary`
    but tolerates non-matrix games (``payoff_opponent is None``).

    Resolution rules
    ----------------
    1. Always inject ``seed`` and (if the factory accepts it) ``n_actions``.
    2. Inject ``payoff_opponent`` iff the factory accepts it AND a payoff
       matrix is available.
    3. ``user_kwargs`` overlay everything except ``seed`` /
       ``n_actions`` / ``payoff_opponent`` (runner-controlled fields).
    """
    factory = ADVERSARY_REGISTRY[adversary_name]

    try:
        sig = inspect.signature(factory)
        params = sig.parameters
    except (TypeError, ValueError):
        params = {}

    accepts_n_actions = "n_actions" in params
    accepts_payoff = "payoff_opponent" in params

    base: Dict[str, Any] = {"seed": int(seed)}
    if accepts_payoff and payoff_opponent is not None:
        base["payoff_opponent"] = np.asarray(payoff_opponent, dtype=np.float64)

    # Infer n_actions:
    #   * For non-matrix games (no ``payoff_opponent``), prefer the
    #     length of the user-supplied ``probs`` if present; otherwise
    #     fall through to the runner-supplied ``n_actions`` argument.
    #   * For matrix games we use ``n_actions`` (computed from
    #     ``payoff_opponent.shape[1]`` upstream).
    inferred_n_actions = int(n_actions)
    if (
        payoff_opponent is None
        and isinstance(user_kwargs.get("probs", None), (list, tuple))
    ):
        inferred_n_actions = int(len(user_kwargs["probs"]))

    if accepts_n_actions:
        base["n_actions"] = inferred_n_actions

    merged: Dict[str, Any] = dict(base)
    for k, v in (user_kwargs or {}).items():
        if k in {"seed", "payoff_opponent"}:
            # Runner-controlled fields; YAML must not override.
            continue
        if k == "n_actions" and accepts_n_actions:
            # Allow the YAML to be authoritative for non-matrix games
            # where the runner cannot infer the action count from a
            # payoff matrix.
            merged[k] = int(v)
            continue
        merged[k] = v

    return make_adversary(adversary_name, **merged)


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------
def _git_sha() -> str:
    """Return the current git SHA, or ``"unknown"`` on any failure."""
    try:
        out = subprocess.check_output(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:  # pragma: no cover — best-effort
        return "unknown"


def _utc_now_iso() -> str:
    """Current UTC time as ISO-8601 with a trailing ``Z``."""
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    if now.endswith("+00:00"):
        now = now[: -len("+00:00")] + "Z"
    return now


def _config_hash(config: Mapping[str, Any]) -> str:
    """Stable hash of the resolved config dict (process-independent)."""
    blob = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _make_run_id(
    *,
    game: str,
    subcase: str,
    method: str,
    seed: int,
) -> str:
    """Generate a unique run identifier.

    The identifier is deterministic-ish: it concatenates the cell
    coordinates with a uuid4 hex tail so concurrent dispatches can never
    collide. The roster invariant is duplicate-cell rejection (per
    cell_id), not duplicate-run-id rejection at construction time, so
    the uuid suffix is just a tiebreaker.
    """
    short_uuid = uuid.uuid4().hex[:8]
    safe_subcase = subcase.replace("/", "_")
    return (
        f"phaseVIII-{game}-{safe_subcase}-{method}-s{int(seed):04d}-"
        f"{short_uuid}"
    )


# ---------------------------------------------------------------------------
# Subcase / config parsing
# ---------------------------------------------------------------------------
def _parse_subcase(raw: Mapping[str, Any]) -> SubcaseSpec:
    """Construct a :class:`SubcaseSpec` from a YAML dict.

    Required keys: ``id``, ``game``, ``adversary``. Optional keys:
    ``game_kwargs`` (dict, default {}), ``adversary_kwargs`` (dict,
    default {}), ``headline_metric`` (default ``"auc_return"``),
    ``t11_guard`` (default ``"cohens_d"``).
    """
    if "id" not in raw:
        raise KeyError(f"subcase entry missing 'id': {raw!r}")
    if "game" not in raw:
        raise KeyError(f"subcase entry {raw['id']!r} missing 'game'")
    if "adversary" not in raw:
        raise KeyError(
            f"subcase entry {raw['id']!r} missing 'adversary'"
        )
    return SubcaseSpec(
        subcase_id=str(raw["id"]),
        game=str(raw["game"]),
        game_kwargs=dict(raw.get("game_kwargs", {}) or {}),
        adversary_name=str(raw["adversary"]),
        adversary_kwargs=dict(raw.get("adversary_kwargs", {}) or {}),
        headline_metric=str(raw.get("headline_metric", "auc_return")),
        t11_guard=str(raw.get("t11_guard", "cohens_d")),
    )


def _load_config(path: Path) -> Dict[str, Any]:
    """Load a YAML config and apply Phase VIII defaults."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"config at {path} must deserialise to a mapping, got "
            f"{type(raw).__name__}"
        )
    return raw


# ---------------------------------------------------------------------------
# Single-cell runner
# ---------------------------------------------------------------------------
def run_one_cell(
    *,
    config: Mapping[str, Any],
    subcase: SubcaseSpec,
    method: str,
    seed: int,
    output_root: Path,
    stage: str,
    roster: Phase8RunRoster,
    config_path: Optional[Path],
    git_commit: str,
) -> Dict[str, Any]:
    """Execute one ``(game, subcase, method, seed)`` cell end-to-end.

    Side effects: writes ``run.json`` and ``metrics.npz`` under
    ``<output_root>/raw/<phase>/<suite>/<game>/<subcase>/<method>/
    seed_<seed>/`` and registers the run in ``roster``.

    Returns the constructed ``run.json`` dict.
    """
    n_episodes = int(config["episodes"])
    gamma = float(config.get("gamma", 0.95))
    learning_rate = float(config.get("learning_rate", 0.1))
    eps_cfg = config.get("epsilon", {}) or {}
    epsilon_start = float(eps_cfg.get("start", 1.0))
    epsilon_end = float(eps_cfg.get("end", 0.05))
    epsilon_decay_episodes = int(
        eps_cfg.get("decay_episodes", max(1, n_episodes // 2))
    )
    q_init = float(config.get("q_init", 0.0))

    schedule_method, schedule_hparams = _resolve_method_to_schedule(method)

    # Resolve the absolute run dir. Path layout (lessons.md #11):
    #
    #   <output_root>/raw/<phase>/<suite>/<game>/<subcase>/<method>/seed_<seed>/
    #
    # The "raw" segment differentiates raw artifacts from processed
    # aggregations under the same Phase VIII root.
    run_dir = make_run_dir(
        base=output_root / "raw",
        phase="VIII",
        suite=stage,
        task=subcase.cell_label(),
        algorithm=method,
        seed=int(seed),
        exist_ok=True,
    )

    run_id = _make_run_id(
        game=subcase.game,
        subcase=subcase.subcase_id,
        method=method,
        seed=int(seed),
    )

    cfg_hash = _config_hash(config)

    roster.append(
        run_id=run_id,
        config_hash=cfg_hash,
        seed=int(seed),
        game=subcase.game,
        subcase=subcase.subcase_id,
        method=method,
        git_commit=git_commit,
    )
    roster.update_status(
        run_id,
        status="running",
        start_time=_utc_now_iso(),
        result_path=str(run_dir),
    )

    start_utc = _utc_now_iso()
    start_perf = time.time()

    try:
        # Validate registry membership up front (loud key errors).
        if subcase.game not in GAME_REGISTRY:
            known = ", ".join(sorted(GAME_REGISTRY.keys()))
            raise KeyError(
                f"unknown game {subcase.game!r}; registered: [{known}]"
            )
        if subcase.adversary_name not in ADVERSARY_REGISTRY:
            known = ", ".join(sorted(ADVERSARY_REGISTRY.keys()))
            raise KeyError(
                f"unknown adversary {subcase.adversary_name!r}; "
                f"registered: [{known}]"
            )

        # Build adversary, then env. Seeds are deterministic on (seed,
        # method, adversary) — agent and adversary share the cell-level
        # seed so paired-seed comparisons across methods at fixed seed
        # remain comparable.
        #
        # Adversary cardinality lookup: for matrix games ``payoff_opponent``
        # at module level encodes the opponent action count; for non-matrix
        # games (delayed_chain) we fall back to ``user_kwargs["n_actions"]``
        # if present, else 1 (PassiveOpponent default).
        payoff_opp = _resolve_payoff_opponent(
            subcase.game, n_actions=2  # placeholder, refined below
        )
        if payoff_opp is not None:
            n_opp_actions = int(np.asarray(payoff_opp).shape[1])
        else:
            n_opp_actions = int(
                subcase.adversary_kwargs.get("n_actions", 1)
            )

        adversary = _build_adversary(
            adversary_name=subcase.adversary_name,
            n_actions=n_opp_actions,
            payoff_opponent=payoff_opp,
            seed=int(seed),
            user_kwargs=subcase.adversary_kwargs,
        )

        env = make_game(
            subcase.game,
            adversary=adversary,
            seed=int(seed),
            **dict(subcase.game_kwargs),
        )

        n_states = int(env.info.observation_space.size[0])
        n_actions = int(env.info.action_space.size[0])
        env_canonical_sign = getattr(env, "env_canonical_sign", None)

        eps_fn = linear_epsilon_schedule(
            start=epsilon_start,
            end=epsilon_end,
            decay_episodes=epsilon_decay_episodes,
        )
        schedule = build_schedule(
            schedule_method, env_canonical_sign, schedule_hparams
        )

        agent_rng = np.random.default_rng(int(seed))
        agent = AdaptiveBetaQAgent(
            n_states=n_states,
            n_actions=n_actions,
            gamma=gamma,
            learning_rate=learning_rate,
            epsilon_schedule=eps_fn,
            beta_schedule=schedule,
            rng=agent_rng,
            env_canonical_sign=env_canonical_sign,
            q_init=q_init,
        )

        # Per-episode buffers.
        ep_returns = np.zeros(n_episodes, dtype=np.float64)
        ep_bellman_residual = np.zeros(n_episodes, dtype=np.float64)
        ep_beta_used = np.zeros(n_episodes, dtype=np.float64)
        ep_beta_raw = np.zeros(n_episodes, dtype=np.float64)
        ep_alignment_rate = np.zeros(n_episodes, dtype=np.float64)
        ep_eff_discount_mean = np.zeros(n_episodes, dtype=np.float64)
        ep_q_abs_max = np.zeros(n_episodes, dtype=np.float64)
        ep_length = np.zeros(n_episodes, dtype=np.int64)
        ep_nan_count = np.zeros(n_episodes, dtype=np.int64)
        ep_divergence_event = np.zeros(n_episodes, dtype=bool)
        # Subcase-specific arrays (populated when the env reports them).
        ep_goal_reaches = np.zeros(n_episodes, dtype=np.int64)
        ep_trap_entries = np.zeros(n_episodes, dtype=np.int64)

        for e in range(n_episodes):
            agent.begin_episode(e)
            state, info = env.reset()
            s = int(np.asarray(state).flat[0])
            t = 0
            ret = 0.0
            n_goal = 0
            n_trap = 0
            while True:
                action = agent.select_action(s, e)
                next_state, reward, absorbing, step_info = env.step(action)
                ns = int(np.asarray(next_state).flat[0])
                agent.step(
                    state=s,
                    action=int(action),
                    reward=float(reward),
                    next_state=ns,
                    absorbing=bool(absorbing),
                    episode_index=e,
                )
                ret += float(reward)
                if step_info.get("terminal_success", False):
                    n_goal += 1
                if step_info.get("trap_entered", False):
                    n_trap += 1
                t += 1
                s = ns
                if absorbing:
                    break

            ep_diag = agent.end_episode(e)
            ep_returns[e] = ret
            ep_bellman_residual[e] = float(ep_diag["bellman_residual"])
            ep_beta_used[e] = float(ep_diag["beta_used"])
            ep_beta_raw[e] = float(ep_diag["beta_raw"])
            ep_alignment_rate[e] = float(ep_diag["alignment_rate"])
            ep_eff_discount_mean[e] = float(ep_diag["mean_d_eff"])
            ep_q_abs_max[e] = float(ep_diag["q_abs_max"])
            ep_length[e] = int(ep_diag["length"])
            ep_nan_count[e] = int(ep_diag["nan_count"])
            ep_divergence_event[e] = bool(ep_diag["divergence_event"])
            ep_goal_reaches[e] = int(n_goal)
            ep_trap_entries[e] = int(n_trap)

        end_utc = _utc_now_iso()
        wallclock = float(time.time() - start_perf)

        # Write metrics.npz with schema header.
        arrays: Dict[str, np.ndarray] = {
            "return": ep_returns,
            "bellman_residual": ep_bellman_residual,
            "beta_used": ep_beta_used,
            "beta_raw": ep_beta_raw,
            "alignment_rate": ep_alignment_rate,
            "effective_discount_mean": ep_eff_discount_mean,
            "q_abs_max": ep_q_abs_max,
            "length": ep_length,
            "nan_count": ep_nan_count,
            "divergence_event": ep_divergence_event.astype(np.uint8),
            "goal_reaches": ep_goal_reaches,
            "trap_entries": ep_trap_entries,
        }
        schema_header = make_npz_schema(
            phase="VIII",
            task=subcase.cell_label(),
            algorithm=method,
            seed=int(seed),
            storage_mode="rl_online",
            arrays=sorted(arrays.keys()),
            schema_version=METRICS_NPZ_SCHEMA_VERSION,
        )
        save_npz_with_schema(run_dir / "metrics.npz", schema_header, arrays)

        # Build run.json.
        run_json: Dict[str, Any] = {
            "schema_version": RUN_JSON_SCHEMA_VERSION,
            "run_id": run_id,
            "phase": "VIII",
            "stage": stage,
            "method": method,
            "schedule_method": schedule_method,
            "schedule_hparams": dict(schedule_hparams),
            "game": subcase.game,
            "subcase": subcase.subcase_id,
            "game_kwargs": dict(subcase.game_kwargs),
            "adversary": subcase.adversary_name,
            "adversary_kwargs": dict(subcase.adversary_kwargs),
            "seed": int(seed),
            "episodes": int(n_episodes),
            "gamma": float(gamma),
            "learning_rate": float(learning_rate),
            "epsilon_schedule_params": {
                "start": float(epsilon_start),
                "end": float(epsilon_end),
                "decay_episodes": int(epsilon_decay_episodes),
            },
            "q_init": float(q_init),
            "headline_metric": subcase.headline_metric,
            "t11_guard": subcase.t11_guard,
            "config": {
                "path": str(config_path) if config_path is not None else None,
                "hash": cfg_hash,
            },
            "env": {
                "n_states": int(n_states),
                "n_actions": int(n_actions),
                "canonical_sign": env_canonical_sign,
            },
            "git_sha": git_commit,
            "host": socket.gethostname(),
            "python_version": sys.version.split()[0],
            "argv": list(sys.argv),
            "start_utc": start_utc,
            "end_utc": end_utc,
            "wallclock_sec": float(wallclock),
            "metrics_schema_version": METRICS_NPZ_SCHEMA_VERSION,
            "metrics_arrays": sorted(arrays.keys()),
            "diverged": bool(ep_divergence_event.any()),
            "nan_count_total": int(ep_nan_count.sum()),
            "result_dir": str(run_dir),
        }

        with open(run_dir / "run.json", "w", encoding="utf-8") as f:
            json.dump(run_json, f, indent=2, sort_keys=True)
            f.write("\n")

        roster.update_status(
            run_id,
            status="completed",
            end_time=end_utc,
        )
        return run_json

    except Exception as exc:  # noqa: BLE001 — capture & quarantine
        end_utc = _utc_now_iso()
        wallclock = float(time.time() - start_perf)
        # Persist traceback alongside whatever partial outputs exist.
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "FAILURE.log", "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        # Mark roster row failed.
        try:
            roster.update_status(
                run_id,
                status="failed",
                end_time=end_utc,
                failure_reason=f"{type(exc).__name__}: {exc}",
            )
        except Exception:  # pragma: no cover — defensive
            pass
        # Re-raise so the caller can decide whether to stop.
        raise


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------
def dispatch(
    *,
    config: Mapping[str, Any],
    seed_override: Optional[Sequence[int]],
    output_root: Path,
    config_path: Optional[Path] = None,
    fail_fast: bool = False,
) -> Phase8RunRoster:
    """Iterate over the (subcase × method × seed) matrix.

    Returns the populated :class:`Phase8RunRoster`. The roster snapshot
    is also written atomically to
    ``<output_root>/raw/<phase>/<suite>/manifest.jsonl``.
    """
    stage = str(config.get("stage", "stage1_beta_sweep"))
    seeds: Sequence[int] = (
        list(seed_override) if seed_override is not None else
        [int(s) for s in config.get("seeds", [0])]
    )
    methods: Sequence[str] = list(config.get("methods", []))
    if not methods:
        raise ValueError("config must declare a non-empty 'methods' list")

    raw_subcases: Sequence[Mapping[str, Any]] = list(
        config.get("subcases", []) or []
    )
    if not raw_subcases:
        raise ValueError("config must declare a non-empty 'subcases' list")
    subcases: List[SubcaseSpec] = [_parse_subcase(s) for s in raw_subcases]

    output_root = Path(output_root)
    roster_dir = output_root / "raw" / "VIII" / stage
    roster_dir.mkdir(parents=True, exist_ok=True)
    roster_path = roster_dir / "manifest.jsonl"

    roster = Phase8RunRoster(base_path=output_root)
    git_commit = _git_sha()

    failures: List[Tuple[SubcaseSpec, str, int, str]] = []
    for sc in subcases:
        for method in methods:
            for seed in seeds:
                try:
                    run_one_cell(
                        config=config,
                        subcase=sc,
                        method=str(method),
                        seed=int(seed),
                        output_root=output_root,
                        stage=stage,
                        roster=roster,
                        config_path=config_path,
                        git_commit=git_commit,
                    )
                except Exception as exc:  # noqa: BLE001
                    msg = f"{type(exc).__name__}: {exc}"
                    failures.append((sc, str(method), int(seed), msg))
                    if fail_fast:
                        # Snapshot the roster before propagating so
                        # post-mortems still have the partial state.
                        roster.write_atomic(roster_path)
                        raise
                # Snapshot incrementally so a crash mid-dispatch leaves
                # a recoverable manifest on disk.
                roster.write_atomic(roster_path)

    if failures:
        # Persist a sibling failure summary for easy triage; never
        # silently swallow failures.
        with open(roster_dir / "failures.json", "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "subcase_id": sc.subcase_id,
                        "game": sc.game,
                        "method": method,
                        "seed": seed,
                        "error": err,
                    }
                    for (sc, method, seed, err) in failures
                ],
                f,
                indent=2,
                sort_keys=True,
            )
            f.write("\n")

    return roster


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_phase_VIII_stage1_beta_sweep",
        description=(
            "Phase VIII Stage 1 fixed-β operator sweep dispatcher "
            "(spec §10.1 / §10.2)"
        ),
    )
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the Stage A dev or Stage 1 main YAML config.",
    )
    p.add_argument(
        "--seed-list",
        type=str,
        default=None,
        help=(
            "Comma-separated override of the YAML 'seeds' list "
            "(e.g. '0,1,2'). Falls back to the config when omitted."
        ),
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=PHASE8_RESULT_ROOT,
        help=(
            "Root directory for raw artifacts. Defaults to "
            "results/adaptive_beta/tab_six_games (lessons.md #11)."
        ),
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Re-raise the first per-cell exception instead of continuing.",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    config = _load_config(args.config)
    seed_override: Optional[List[int]] = None
    if args.seed_list is not None:
        seed_override = [int(s) for s in args.seed_list.split(",") if s.strip()]
    dispatch(
        config=config,
        seed_override=seed_override,
        output_root=Path(args.output_root),
        config_path=Path(args.config),
        fail_fast=bool(args.fail_fast),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI entry
    sys.exit(main())
