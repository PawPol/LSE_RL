"""Phase VIII Stage 2 — non-stationary Q-learning baselines runner (M7.1).

Spec authority
--------------
- ``docs/specs/phase_VIII_tab_six_games.md`` §6.3 (baseline contracts +
  per-episode metric subset) and §10.3 (Stage-2 fixed-TAB vs vanilla and
  external baselines).
- §8.1 logging schema (run.json + metrics.npz contract; this stage
  honours the same envelope but writes a SUBSET of per-episode arrays
  per §6.3 because baselines have no β / no operator-mechanism metrics).
- §8.2 Phase8RunRoster (every requested run is registered).
- §13.6 result-root regression (route through
  ``results/adaptive_beta/tab_six_games`` explicitly; never inherit
  ``results/weighted_lse_dp``).

Boundaries
----------
- Reuses, never duplicates: the baseline classes from
  :mod:`experiments.adaptive_beta.baselines`
  (:class:`RestartQLearningAgent`, :class:`SlidingWindowQLearningAgent`,
  :class:`TunedEpsilonGreedyQLearningAgent`);
  :func:`linear_epsilon_schedule` from
  :mod:`experiments.adaptive_beta.agents`; the strategic-games registries
  (``GAME_REGISTRY`` / ``ADVERSARY_REGISTRY``); the run-dir helper
  ``make_run_dir`` and the schema'd npz writer ``save_npz_with_schema``
  from :mod:`experiments.weighted_lse_dp.common.io`; and
  :class:`Phase8RunRoster` from
  :mod:`experiments.adaptive_beta.tab_six_games.manifests`.
- Does NOT touch ``mushroom-rl-dev/`` (CLAUDE.md §4).
- Does NOT touch ``src/lse_rl/operator/``: baselines have no β.
- Mirrors ``run_phase_VIII_stage1_beta_sweep`` for IO / manifest /
  run-dir layout 1:1; the only material divergence is method dispatch
  (build a baseline agent instead of an :class:`AdaptiveBetaQAgent` +
  :class:`BetaSchedule`) and the metrics.npz column subset.

Method ID convention (Phase VIII Stage 2 baselines)
---------------------------------------------------
The Phase VIII Stage 2 method names map 1:1 to the baseline classes:

* ``restart_Q_learning`` → :class:`RestartQLearningAgent`.
* ``sliding_window_Q_learning`` → :class:`SlidingWindowQLearningAgent`.
* ``tuned_epsilon_greedy_Q_learning`` →
  :class:`TunedEpsilonGreedyQLearningAgent`.
* ``regret_matching_agent`` → :class:`RegretMatchingAgent` (M7.2,
  spec §6.3 patch §3 — strategic-learning agent baseline).
* ``smoothed_fictitious_play_agent`` →
  :class:`SmoothedFictitiousPlayAgent` (M7.2, spec §6.3 patch §3).

The mapping is performed exclusively by :func:`_build_baseline_agent` so
the rest of the runner is method-agnostic. The strategic-learning
baselines additionally consume two env-derived inputs (``payoff_agent``
matrix, ``env.history`` provider) which are resolved per-cell in
:func:`run_one_cell` and forwarded through ``_build_baseline_agent``.

Metrics.npz schema (Stage 2 baselines)
--------------------------------------
Per spec §6.3 the baselines emit a SUBSET of the Phase VIII per-episode
metric vocabulary: ``return``, ``length``, ``epsilon``,
``bellman_residual`` (per-episode mean ``|td_error|``), ``q_abs_max``,
``nan_count``, ``divergence_event`` (constant 0 unless the agent itself
flags it). Operator-mechanism arrays (``alignment_rate``, ``beta_used``,
``beta_raw``, ``effective_discount_mean``, etc.) are NOT emitted —
baselines have no β.

Aggregator handling: the Phase VIII long-CSV aggregator
(``analysis/aggregate.py``) flags **foreign** keys as ``schema_drift``
but treats **missing** expected keys as ``np.nan`` rows transparently
(see ``_build_rows_for_run`` — the ``if col in metrics:`` check). We
therefore write ONLY the in-schema columns the baselines actually
produce and let the aggregator default the rest. No baseline-specific
diag fields (``rolling_mean_return``, ``restart_event``,
``buffer_size``, ``n_states_reset``) are written, since they would be
foreign keys and trigger the schema-drift warning. Aggregator changes
are out of scope for this milestone (orchestrator patches separately).

CLI
---
::

    python -m experiments.adaptive_beta.tab_six_games.runners.\
run_phase_VIII_stage2_baselines \
        --config experiments/adaptive_beta/tab_six_games/configs/\
stage2_baselines_headline.yaml \
        --seed-list 0,1,2

The runner is intentionally serial. Parallel dispatch is out of scope
for M7.1; the smoke test exercises the full single-process path
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
    linear_epsilon_schedule,
)
from experiments.adaptive_beta.baselines import (  # noqa: E402
    RestartQLearningAgent,
    SlidingWindowQLearningAgent,
    TunedEpsilonGreedyQLearningAgent,
)
from experiments.adaptive_beta.strategic_games.agents import (  # noqa: E402
    RegretMatchingAgent,
    SmoothedFictitiousPlayAgent,
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

#: Per-episode array names always written to ``metrics.npz`` for Stage 2
#: baselines. Strict subset of the Phase VIII columns expected by the
#: long-CSV aggregator (spec §6.3): the baselines have no β, so the
#: operator-mechanism metrics are deliberately omitted. The aggregator
#: defaults missing columns to ``np.nan`` (see
#: ``analysis/aggregate.py::_build_rows_for_run``).
REQUIRED_METRICS: Tuple[str, ...] = (
    "return",
    "length",
    "epsilon",
    "bellman_residual",
    "q_abs_max",
    "nan_count",
    "divergence_event",
)

#: Recognised method ids for Stage 2. Exposed for tests and the smoke
#: harness so the dispatch surface is auditable.
SUPPORTED_BASELINE_METHODS: Tuple[str, ...] = (
    "restart_Q_learning",
    "sliding_window_Q_learning",
    "tuned_epsilon_greedy_Q_learning",
    # M7.2 (spec §6.3 patch-2026-05-01 §3): strategic-learning agent
    # baselines wrapping the existing regret-matching / smoothed FP
    # opponent classes in the agent interface.
    "regret_matching_agent",
    "smoothed_fictitious_play_agent",
)


# ---------------------------------------------------------------------------
# Subcase + cell descriptors (parsed from YAML)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SubcaseSpec:
    """One ``(game, subcase)`` cell description, parsed from YAML.

    Mirrors the Stage 1 :class:`SubcaseSpec` 1:1 so configs are
    interchangeable between stages.
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
def _build_baseline_agent(
    *,
    method_id: str,
    n_states: int,
    n_actions: int,
    gamma: float,
    learning_rate: float,
    epsilon_schedule,
    rng: np.random.Generator,
    q_init: float,
    method_kwargs: Mapping[str, Any],
    payoff_agent: Optional[np.ndarray] = None,
    env_history_provider: Optional[Any] = None,
    seed: Optional[int] = None,
) -> Any:
    """Construct the baseline agent object for ``method_id``.

    The single translation point between the Phase VIII Stage 2 method
    name and the baseline class. Forwards the shared
    (n_states, n_actions, γ, α, ε-schedule, rng, q_init) tuple plus any
    method-specific kwargs from the YAML ``method_kwargs:`` field
    (default: empty dict, so the baseline class's documented defaults
    apply).

    Supported names
    ---------------
    * ``restart_Q_learning`` → :class:`RestartQLearningAgent`. Honours
      kwargs: ``restart_window``, ``restart_drop``,
      ``divergence_threshold``.
    * ``sliding_window_Q_learning`` →
      :class:`SlidingWindowQLearningAgent`. Honours kwargs:
      ``window_size``.
    * ``tuned_epsilon_greedy_Q_learning`` →
      :class:`TunedEpsilonGreedyQLearningAgent`. The tuned ε-schedule
      is applied by the class default unless ``epsilon_schedule`` is
      supplied (we always pass the runner's ε-schedule built from the
      YAML so ε is paired across methods at fixed seed).
    * ``regret_matching_agent`` (M7.2) → :class:`RegretMatchingAgent`.
      Honours kwargs: ``mode`` (``"full_info"`` / ``"realized_payoff"``),
      ``value_lr``. Requires the env-derived ``payoff_agent`` matrix
      (resolved per-cell in :func:`run_one_cell`); ``None`` triggers
      the documented uniform-random fallback (e.g. ``delayed_chain``).
    * ``smoothed_fictitious_play_agent`` (M7.2) →
      :class:`SmoothedFictitiousPlayAgent`. Honours kwargs:
      ``temperature``, ``memory_m``. Same env-derived inputs as the
      regret-matching wrapper.

    The ``payoff_agent`` / ``env_history_provider`` / ``seed`` kwargs
    are consumed only by the M7.2 strategic-learning agent baselines.
    The three M7.1 Q-learning baselines ignore them and remain
    bit-identical to their pre-M7.2 dispatch path.

    Unknown ``method_id`` raises :class:`ValueError` so config typos
    fail loudly.
    """
    name = str(method_id).strip()
    extra: Dict[str, Any] = dict(method_kwargs or {})

    if name == "restart_Q_learning":
        return RestartQLearningAgent(
            n_states=n_states,
            n_actions=n_actions,
            gamma=gamma,
            learning_rate=learning_rate,
            epsilon_schedule=epsilon_schedule,
            rng=rng,
            q_init=q_init,
            **{
                k: v
                for k, v in extra.items()
                if k
                in {
                    "restart_window",
                    "restart_drop",
                    "divergence_threshold",
                }
            },
        )
    if name == "sliding_window_Q_learning":
        return SlidingWindowQLearningAgent(
            n_states=n_states,
            n_actions=n_actions,
            gamma=gamma,
            learning_rate=learning_rate,
            epsilon_schedule=epsilon_schedule,
            rng=rng,
            q_init=q_init,
            **{k: v for k, v in extra.items() if k == "window_size"},
        )
    if name == "tuned_epsilon_greedy_Q_learning":
        # M7.1 fix (2026-05-02): Do NOT override the tuned ε-schedule
        # with the runner's vanilla schedule — that reduces this baseline
        # to vanilla and the comparison becomes a tautology. The class
        # default (start=1.0, end=0.01, decay_episodes=2000) is the
        # whole point of the "tuned" baseline. If the user wants a
        # custom schedule, they pass it through method_kwargs.
        explicit_eps = extra.get("epsilon_schedule", None)
        return TunedEpsilonGreedyQLearningAgent(
            n_states=n_states,
            n_actions=n_actions,
            gamma=gamma,
            learning_rate=learning_rate,
            epsilon_schedule=explicit_eps,  # None → class tuned default
            rng=rng,
            q_init=q_init,
        )

    # ------------------------------------------------------------------
    # M7.2 (spec §6.3 patch-2026-05-01 §3): strategic-learning agent
    # baselines. The wrapper classes consume a payoff_agent matrix and
    # an env.history accessor (see RegretMatchingAgent /
    # SmoothedFictitiousPlayAgent docstring). When the cell has no
    # matrix-game payoff (e.g. delayed_chain), payoff_agent is None
    # and the wrapper falls back to a uniform-random policy by design
    # (spec §10.3: "expected to fail; diagnostic feature").
    # ------------------------------------------------------------------
    if name == "regret_matching_agent":
        return RegretMatchingAgent(
            n_states=n_states,
            n_actions=n_actions,
            gamma=gamma,
            learning_rate=learning_rate,
            epsilon_schedule=epsilon_schedule,
            rng=rng,
            q_init=q_init,
            payoff_agent=payoff_agent,
            env_history_provider=env_history_provider,
            seed=seed,
            **{k: v for k, v in extra.items() if k in {"mode", "value_lr"}},
        )
    if name == "smoothed_fictitious_play_agent":
        return SmoothedFictitiousPlayAgent(
            n_states=n_states,
            n_actions=n_actions,
            gamma=gamma,
            learning_rate=learning_rate,
            epsilon_schedule=epsilon_schedule,
            rng=rng,
            q_init=q_init,
            payoff_agent=payoff_agent,
            env_history_provider=env_history_provider,
            seed=seed,
            **{
                k: v
                for k, v in extra.items()
                if k in {"temperature", "memory_m"}
            },
        )

    raise ValueError(
        f"unknown Phase VIII Stage-2 baseline method_id={method_id!r}; "
        f"expected one of {sorted(SUPPORTED_BASELINE_METHODS)}"
    )


# ---------------------------------------------------------------------------
# Adversary construction (Phase VIII variant — same shape as Stage 1)
# ---------------------------------------------------------------------------
_GAME_MODULE_PREFIX = "experiments.adaptive_beta.strategic_games.games."

# See ``run_phase_VIII_stage1_beta_sweep`` for the rationale: payoff
# aliases let opponents that read ``payoff_opponent`` work against
# games that do not expose their own module-level payoff matrices.
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

    Mirrors :func:`run_phase_VIII_stage1_beta_sweep._resolve_payoff_opponent`
    verbatim — the adversary construction logic is identical between
    Stage 1 and Stage 2.
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
        return -np.asarray(pa, dtype=np.float64)
    return None


def _resolve_payoff_agent(
    game_name: str,
) -> Optional[np.ndarray]:
    """Best-effort lookup of the AGENT-side payoff matrix.

    Used by the M7.2 strategic-learning agent baselines (spec §6.3
    patch §3): the wrapper agents need the agent-side payoff (not
    opponent) because the wrapper plays as the agent and best-responds
    over its own action space against the env adversary.

    Returns ``None`` for games that do not expose ``payoff_agent``
    (notably ``delayed_chain``, where the wrapper's documented
    fallback is uniform-random action selection — see
    ``RegretMatchingAgent`` / ``SmoothedFictitiousPlayAgent`` module
    docstring "DC-Long50 (and other non-matrix games) handling").
    """
    try:
        mod = _import_game_module(game_name)
    except ImportError:
        return None
    pa = getattr(mod, "payoff_agent", None)
    if pa is None:
        return None
    return np.asarray(pa, dtype=np.float64)


def _build_adversary(
    *,
    adversary_name: str,
    n_actions: int,
    payoff_opponent: Optional[np.ndarray],
    seed: int,
    user_kwargs: Mapping[str, Any],
) -> Any:
    """Construct an adversary, auto-supplying signature-driven kwargs.

    Mirrors :func:`run_phase_VIII_stage1_beta_sweep._build_adversary`
    verbatim. See that function for the full resolution rules.
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
            continue
        if k == "n_actions" and accepts_n_actions:
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
    """Generate a unique run identifier (mirrors Stage 1)."""
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
    """Construct a :class:`SubcaseSpec` from a YAML dict."""
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
def _format_gamma_segment(gamma: float) -> str:
    """Render a γ value as a path segment ``gamma_<value>``.

    Pinned format (two decimals) matches the Stage 1 convention so
    Stage 1 (TAB) and Stage 2 (baselines) results land in identical
    γ-segment trees and downstream pairing logic stays trivial.
    """
    return f"gamma_{float(gamma):.2f}"


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
    gamma_override: Optional[float] = None,
    gamma_in_path: bool = False,
) -> Dict[str, Any]:
    """Execute one ``(game, subcase, method, seed)`` cell end-to-end.

    Side effects: writes ``run.json`` and ``metrics.npz`` under the
    canonical Phase VIII path layout (lessons.md #11) and registers the
    run in ``roster``. The metrics.npz column subset is the
    baseline-specific :data:`REQUIRED_METRICS` (spec §6.3).

    Returns the constructed ``run.json`` dict.
    """
    n_episodes = int(config["episodes"])
    gamma = float(
        gamma_override if gamma_override is not None else config.get("gamma", 0.95)
    )
    learning_rate = float(config.get("learning_rate", 0.1))
    eps_cfg = config.get("epsilon", {}) or {}
    epsilon_start = float(eps_cfg.get("start", 1.0))
    epsilon_end = float(eps_cfg.get("end", 0.05))
    epsilon_decay_episodes = int(
        eps_cfg.get("decay_episodes", max(1, n_episodes // 2))
    )
    q_init = float(config.get("q_init", 0.0))
    method_kwargs: Dict[str, Any] = dict(config.get("method_kwargs", {}) or {})
    # Per-method overrides take precedence over the global block, so
    # different baselines can declare different ``window_size``/etc. in
    # the same config without colliding.
    per_method_block: Mapping[str, Any] = (
        config.get("method_kwargs_per_method", {}) or {}
    )
    if isinstance(per_method_block, Mapping) and method in per_method_block:
        per_method_extra = per_method_block.get(method, {}) or {}
        if isinstance(per_method_extra, Mapping):
            method_kwargs.update(per_method_extra)

    # Resolve the absolute run dir. Path layout (lessons.md #11):
    #
    #   Tier I  (single γ, gamma_in_path=False):
    #     <output_root>/raw/<phase>/<suite>/<game>/<subcase>/<method>/seed_<seed>/
    #   Tier II/III (γ-grid, gamma_in_path=True):
    #     <output_root>/raw/<phase>/<suite>/<game>/<subcase>/gamma_<g>/<method>/seed_<seed>/
    if gamma_in_path:
        algorithm_segment = f"{_format_gamma_segment(gamma)}/{method}"
    else:
        algorithm_segment = method
    run_dir = make_run_dir(
        base=output_root / "raw",
        phase="VIII",
        suite=stage,
        task=subcase.cell_label(),
        algorithm=algorithm_segment,
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
        gamma=float(gamma),
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

        payoff_opp = _resolve_payoff_opponent(
            subcase.game, n_actions=2  # placeholder; refined below
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

        agent_rng = np.random.default_rng(int(seed))

        # M7.2: env-derived inputs for the strategic-learning agent
        # baselines (regret_matching_agent /
        # smoothed_fictitious_play_agent). The Q-learning baselines
        # ignore these — they keep the same dispatch path as M7.1.
        payoff_agent_matrix = _resolve_payoff_agent(subcase.game)
        # ``env.history`` is a property; wrap as a provider so the
        # wrapper agent does not hold a hard env reference (also
        # protects against test fixtures that swap envs mid-run).
        env_history_provider = (lambda env=env: env.history)

        agent = _build_baseline_agent(
            method_id=method,
            n_states=n_states,
            n_actions=n_actions,
            gamma=gamma,
            learning_rate=learning_rate,
            epsilon_schedule=eps_fn,
            rng=agent_rng,
            q_init=q_init,
            method_kwargs=method_kwargs,
            payoff_agent=payoff_agent_matrix,
            env_history_provider=env_history_provider,
            seed=int(seed),
        )

        # Per-episode buffers (subset of Phase VIII columns; spec §6.3).
        ep_returns = np.zeros(n_episodes, dtype=np.float64)
        ep_length = np.zeros(n_episodes, dtype=np.int64)
        ep_epsilon = np.zeros(n_episodes, dtype=np.float64)
        ep_bellman_residual = np.zeros(n_episodes, dtype=np.float64)
        ep_q_abs_max = np.zeros(n_episodes, dtype=np.float64)
        ep_nan_count = np.zeros(n_episodes, dtype=np.int64)
        ep_divergence_event = np.zeros(n_episodes, dtype=bool)
        # Subcase-specific arrays (populated when the env reports them).
        # Carried verbatim from Stage 1 so the Phase VIII expected
        # column set is honoured for delayed_chain success / trap
        # diagnostics. These remain zero on matrix games.
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
            sum_abs_td_error = 0.0
            n_steps_for_residual = 0
            while True:
                action = agent.select_action(s, e)
                next_state, reward, absorbing, step_info = env.step(action)
                ns = int(np.asarray(next_state).flat[0])
                step_diag = agent.step(
                    state=s,
                    action=int(action),
                    reward=float(reward),
                    next_state=ns,
                    absorbing=bool(absorbing),
                    episode_index=e,
                )
                # Per-episode mean |td_error| := bellman_residual.
                # All three baseline classes return ``td_error`` in
                # their per-step diag dict; defensively guard with a
                # default of 0.0 in case a future baseline omits it.
                td_err = float(step_diag.get("td_error", 0.0))
                sum_abs_td_error += abs(td_err)
                n_steps_for_residual += 1

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
            ep_length[e] = int(ep_diag.get("length", t))
            # ε at episode-end mirrors the value the agent acted under
            # for action selection across the episode (the schedule is
            # episode-indexed; per-step value is constant within an
            # episode). M7.2: strategic-learning agent baselines do
            # NOT use ε-greedy (their stochasticity is intrinsic to
            # the regret-matching / logit-FP policy); spec §6.3 patch
            # §3 mandates ε == 0 in their per-episode metrics.
            if getattr(agent, "is_strategic_learning_agent", False):
                ep_epsilon[e] = 0.0
            else:
                ep_epsilon[e] = float(eps_fn(e))
            ep_bellman_residual[e] = (
                sum_abs_td_error / n_steps_for_residual
                if n_steps_for_residual > 0
                else 0.0
            )
            ep_q_abs_max[e] = float(ep_diag.get("q_abs_max", 0.0))
            ep_nan_count[e] = int(ep_diag.get("nan_count", 0))
            ep_divergence_event[e] = bool(ep_diag.get("divergence_event", False))
            ep_goal_reaches[e] = int(n_goal)
            ep_trap_entries[e] = int(n_trap)

        end_utc = _utc_now_iso()
        wallclock = float(time.time() - start_perf)

        # Write metrics.npz with schema header. Strict subset per spec
        # §6.3 — operator-mechanism arrays (alignment_rate, beta_used,
        # etc.) are deliberately omitted; the aggregator defaults
        # missing columns to NaN (see analysis/aggregate.py).
        arrays: Dict[str, np.ndarray] = {
            "return": ep_returns,
            "length": ep_length,
            "epsilon": ep_epsilon,
            "bellman_residual": ep_bellman_residual,
            "q_abs_max": ep_q_abs_max,
            "nan_count": ep_nan_count,
            "divergence_event": ep_divergence_event.astype(np.uint8),
            "goal_reaches": ep_goal_reaches,
            "trap_entries": ep_trap_entries,
            # γ as a 0-dim scalar so the aggregator can recover it
            # without re-reading run.json (matches Stage 1).
            "gamma": np.float64(gamma),
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

        # Build run.json. Marker ``baseline_method=true`` makes this
        # stage trivially separable downstream even though the
        # aggregator does not require it (the subset emission alone
        # suffices for the long-CSV pass).
        run_json: Dict[str, Any] = {
            "schema_version": RUN_JSON_SCHEMA_VERSION,
            "run_id": run_id,
            "phase": "VIII",
            "stage": stage,
            "method": method,
            "baseline_method": True,
            "method_kwargs": dict(method_kwargs),
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
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "FAILURE.log", "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        try:
            roster.update_status(
                run_id,
                status="failed",
                end_time=end_utc,
                failure_reason=f"{type(exc).__name__}: {exc}",
            )
        except Exception:  # pragma: no cover — defensive
            pass
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

    Mirrors :func:`run_phase_VIII_stage1_beta_sweep.dispatch` 1:1 but
    routes method dispatch through :func:`_build_baseline_agent`. The
    populated :class:`Phase8RunRoster` is also written atomically to
    ``<output_root>/raw/<phase>/<suite>/manifest.jsonl``.
    """
    stage = str(config.get("stage", "stage2_baselines"))
    seeds: Sequence[int] = (
        list(seed_override) if seed_override is not None else
        [int(s) for s in config.get("seeds", [0])]
    )
    methods: Sequence[str] = list(config.get("methods", []))
    if not methods:
        raise ValueError("config must declare a non-empty 'methods' list")

    # Reject unknown methods up front so a bad config does not silently
    # waste compute.
    bad = [m for m in methods if m not in SUPPORTED_BASELINE_METHODS]
    if bad:
        raise ValueError(
            f"unsupported Stage-2 baseline method(s) in config: {bad!r}; "
            f"supported: {sorted(SUPPORTED_BASELINE_METHODS)}"
        )

    raw_subcases: Sequence[Mapping[str, Any]] = list(
        config.get("subcases", []) or []
    )
    if not raw_subcases:
        raise ValueError("config must declare a non-empty 'subcases' list")
    subcases: List[SubcaseSpec] = [_parse_subcase(s) for s in raw_subcases]

    # γ resolution (matches Stage 1 / v10 spec §6.4 / §10.2.γ).
    raw_gamma_grid = config.get("gamma_grid")
    if raw_gamma_grid is not None:
        if not isinstance(raw_gamma_grid, (list, tuple)):
            raise TypeError(
                f"config 'gamma_grid' must be a list/tuple, got "
                f"{type(raw_gamma_grid).__name__}"
            )
        if len(raw_gamma_grid) == 0:
            raise ValueError("config 'gamma_grid' must be non-empty")
        gamma_values: List[float] = [float(g) for g in raw_gamma_grid]
        gamma_in_path = True
    else:
        gamma_values = [float(config.get("gamma", 0.95))]
        gamma_in_path = False

    output_root = Path(output_root)
    roster_dir = output_root / "raw" / "VIII" / stage
    roster_dir.mkdir(parents=True, exist_ok=True)
    roster_path = roster_dir / "manifest.jsonl"

    roster = Phase8RunRoster(base_path=output_root)
    git_commit = _git_sha()

    failures: List[Tuple[SubcaseSpec, str, int, str, float]] = []
    for gamma_val in gamma_values:
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
                            gamma_override=float(gamma_val),
                            gamma_in_path=gamma_in_path,
                        )
                    except Exception as exc:  # noqa: BLE001
                        msg = f"{type(exc).__name__}: {exc}"
                        failures.append(
                            (sc, str(method), int(seed), msg, float(gamma_val))
                        )
                        if fail_fast:
                            roster.write_atomic(roster_path)
                            raise
                    roster.write_atomic(roster_path)

    if failures:
        with open(roster_dir / "failures.json", "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "subcase_id": sc.subcase_id,
                        "game": sc.game,
                        "method": method,
                        "seed": seed,
                        "gamma": gamma_failed,
                        "error": err,
                    }
                    for (sc, method, seed, err, gamma_failed) in failures
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
        prog="run_phase_VIII_stage2_baselines",
        description=(
            "Phase VIII Stage 2 non-stationary Q-learning baselines "
            "dispatcher (spec §6.3 / §10.3)"
        ),
    )
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the Stage 2 baselines YAML config.",
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
