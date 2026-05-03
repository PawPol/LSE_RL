"""Phase VIII Stage 5 — adaptive-method runner on standalone cells (M10).

Spec authority
--------------
- ``docs/specs/phase_VIII_tab_six_games.md`` §10.6 (Stage 5 contraction-
  adaptive β; M10).
- §6.1 (method roster: ``adaptive_beta``, ``contraction_UCB_beta``,
  ``return_UCB_beta``, ``hand_adaptive_beta``, ``oracle_beta``).
- §6.5 (UCB schedule contracts; standardised reward, c, warm-start).
- §6.6 (Oracle / hand-adaptive contracts).
- §7.2 (UCB-specific delta metrics: ``ucb_arm_count``,
  ``ucb_arm_value``).
- §8.1 logging schema (run.json + metrics.npz contract).
- §8.2 ``Phase8RunRoster`` (every dispatched run is registered).
- §13.6 result-root regression.

Boundaries
----------
- DOES NOT modify M6 (Stage 1) or M7 (Stage 2) runners. Mirrors the
  Stage 1 IO contract verbatim except for method dispatch.
- DOES NOT touch ``mushroom-rl-dev/`` or ``src/lse_rl/operator/``.
- DOES NOT modify ``experiments/adaptive_beta/agents.py`` or any
  schedule class definitions in
  ``experiments/adaptive_beta/schedules.py``. The runner consumes the
  existing factory ``build_schedule`` and the public ``BetaSchedule``
  protocol surface.
- DOES NOT modify the M9 runner's existing method branches; the
  parallel ``return_UCB_beta`` branch added there is an addition only.

Method dispatch (six branches)
------------------------------
- ``vanilla`` → :class:`ZeroBetaSchedule`.
- ``fixed_beta_<signed_value>`` → :class:`FixedBetaSchedule(±1, β0=value)`
  (parsed identically to Stage 1; see :func:`_parse_fixed_beta_method`).
- ``contraction_UCB_beta`` → :class:`ContractionUCBBetaSchedule`.
- ``return_UCB_beta`` → :class:`ReturnUCBBetaSchedule`.
- ``adaptive_beta`` → :class:`AdaptiveBetaSchedule` (Phase VII-A).
- ``hand_adaptive_beta`` → :class:`HandAdaptiveBetaSchedule`.

``oracle_beta`` is REJECTED on standalone cells: the oracle requires a
regime label from the env (spec §6.6), which non-composite envs do
not expose. Configuring ``oracle_beta`` raises ``ValueError`` at
config-parse time. Use the M9 stage4 composite runner instead.

Per-method hyperparameter override
----------------------------------
The YAML key ``method_kwargs_per_method`` supports per-method
overrides (mirrors M7.2 / M9 conventions). Recognised keys per method:

- ``contraction_UCB_beta``: ``ucb_c``, ``epsilon_floor``,
  ``residual_smoothing_window``, ``arm_grid``.
- ``return_UCB_beta``: ``ucb_c``, ``epsilon_floor``, ``arm_grid``.
- ``adaptive_beta``: ``beta_max``, ``beta_cap``, ``k``,
  ``initial_beta``, ``beta_tol``, ``lambda_smooth``.
- ``hand_adaptive_beta``: ``beta0``, ``A_scale``, ``lambda_smooth``.
- ``vanilla``, ``fixed_beta_*``: no overrides.

UCB warm-start length
---------------------
The UCB warm-start period is hard-coded to ``len(arm_grid)`` inside
:class:`_BaseUCBBetaSchedule._select_next_arm`. To adjust the warm-
start length, override ``arm_grid`` (default 21 arms = 21 forced
pulls). The user-facing ``warm_start_pulls`` knob does not exist on
the schedule; documented here so override expectations are clear.

Per-episode metric emission
---------------------------
Standard Phase VIII columns (mirrors Stage 1 plus ``epsilon``):
``return``, ``length``, ``epsilon``, ``bellman_residual``,
``alignment_rate``, ``effective_discount_mean``, ``q_abs_max``,
``nan_count``, ``divergence_event``, ``goal_reaches``, ``trap_entries``,
``gamma``.

UCB-specific columns (per spec §7.2):
- ``ucb_arm_index``: which arm UCB selected this episode (int);
  NaN-coded as -1 for non-UCB methods.
- ``ucb_most_pulled_arm_index``: index of the arm with the highest
  cumulative pull count at episode end (running argmax over
  ``schedule.pull_counts()``); -1 for non-UCB methods.
- ``ucb_most_pulled_arm_value``: per-arm Welford running mean for the
  most-pulled arm at episode end (NaN for non-UCB methods).

CLI
---
::

    python -m experiments.adaptive_beta.tab_six_games.runners.\\
run_phase_VIII_stage5_adaptive \\
        --config experiments/adaptive_beta/tab_six_games/configs/\\
m10_phase_A_ucb_canonical.yaml \\
        --seed-list 0,1,2 --methods contraction_UCB_beta
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
    METHOD_ADAPTIVE_BETA,
    METHOD_CONTRACTION_UCB_BETA,
    METHOD_FIXED_NEGATIVE,
    METHOD_FIXED_POSITIVE,
    METHOD_HAND_ADAPTIVE_BETA,
    METHOD_RETURN_UCB_BETA,
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

#: Canonical Phase VIII result root.
PHASE8_RESULT_ROOT: Path = Path("results/adaptive_beta/tab_six_games")

#: Per-episode array names always written to ``metrics.npz``.
REQUIRED_METRICS: Tuple[str, ...] = (
    "return",
    "length",
    "epsilon",
    "bellman_residual",
    "alignment_rate",
    "effective_discount_mean",
    "q_abs_max",
    "beta_used",
    "beta_raw",
)

#: Stage 5 method ID set understood by this runner.
#: ``oracle_beta`` is intentionally absent — see module docstring.
M10_STANDALONE_METHOD_IDS: frozenset = frozenset({
    "vanilla",
    "contraction_UCB_beta",
    "return_UCB_beta",
    "adaptive_beta",
    "hand_adaptive_beta",
})

#: Sentinel value for "method does not produce a valid arm index".
#: Stored as int8 in metrics.npz; ``ucb_*`` columns are -1 / NaN for
#: non-UCB methods so the aggregator can drop them with a single
#: filter.
_UCB_ARM_NA_INT: int = -1


# ---------------------------------------------------------------------------
# Subcase descriptor (parsed from YAML; identical schema to Stage 1)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SubcaseSpec:
    """One ``(game, subcase)`` cell description, parsed from YAML."""

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
# Method-id parsing
# ---------------------------------------------------------------------------
def _is_fixed_beta_method(method_id: str) -> bool:
    return str(method_id).strip().startswith("fixed_beta_")


def _parse_fixed_beta_method(method_id: str) -> Tuple[str, Dict[str, Any]]:
    """Mirror Stage 1's ``fixed_beta_<signed_value>`` parser.

    Re-implemented locally rather than imported so this runner stays
    self-contained (Stage 1's resolver lives in
    ``run_phase_VIII_stage1_beta_sweep`` with a slightly different
    return contract).
    """
    name = str(method_id).strip()
    if not name.startswith("fixed_beta_"):
        raise ValueError(
            f"_parse_fixed_beta_method received non-fixed_beta id: {method_id!r}"
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


def _resolve_method(
    method_id: str,
    method_kwargs: Mapping[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """Map an M10 method ID to ``(schedule_method_id, hparams)``.

    The five supported standalone branches are covered. ``oracle_beta``
    is rejected here because the standalone envs do not expose a
    ``regime`` label (spec §6.6).
    """
    name = str(method_id).strip()

    # --- vanilla ---------------------------------------------------------
    if name == "vanilla":
        return METHOD_VANILLA, {}

    # --- fixed_beta_<signed_value> ---------------------------------------
    if _is_fixed_beta_method(name):
        return _parse_fixed_beta_method(name)

    # --- oracle_beta — explicit rejection on standalone ------------------
    if name == "oracle_beta":
        raise ValueError(
            "oracle_beta is NOT supported on standalone cells: it requires "
            "a regime label that only sign-switching composites expose "
            "(spec §6.6). Use the Stage 4 composite runner "
            "(run_phase_VIII_stage4_composite) for oracle_beta dispatch."
        )

    # --- adaptive_beta (Phase VII-A) -------------------------------------
    if name == "adaptive_beta":
        hp: Dict[str, Any] = {}
        for k in (
            "beta_max", "beta_cap", "k", "initial_beta", "beta_tol",
            "lambda_smooth",
        ):
            if k in method_kwargs:
                hp[k] = method_kwargs[k]
        return METHOD_ADAPTIVE_BETA, hp

    # --- hand_adaptive_beta ----------------------------------------------
    if name == "hand_adaptive_beta":
        hp = {}
        for k in ("beta0", "A_scale", "lambda_smooth"):
            if k in method_kwargs:
                hp[k] = method_kwargs[k]
        return METHOD_HAND_ADAPTIVE_BETA, hp

    # --- contraction_UCB_beta --------------------------------------------
    if name == "contraction_UCB_beta":
        hp = {}
        for k in (
            "ucb_c", "epsilon_floor", "residual_smoothing_window", "arm_grid",
        ):
            if k in method_kwargs:
                hp[k] = method_kwargs[k]
        return METHOD_CONTRACTION_UCB_BETA, hp

    # --- return_UCB_beta -------------------------------------------------
    if name == "return_UCB_beta":
        hp = {}
        for k in ("ucb_c", "epsilon_floor", "arm_grid"):
            if k in method_kwargs:
                hp[k] = method_kwargs[k]
        return METHOD_RETURN_UCB_BETA, hp

    raise ValueError(
        f"unknown M10 standalone method_id={method_id!r}; valid ids: "
        f"{sorted(M10_STANDALONE_METHOD_IDS)} or 'fixed_beta_+x' / "
        f"'fixed_beta_-x'."
    )


# ---------------------------------------------------------------------------
# Adversary construction (verbatim from Stage 1)
# ---------------------------------------------------------------------------
_GAME_MODULE_PREFIX = "experiments.adaptive_beta.strategic_games.games."
_PAYOFF_ALIAS: Dict[str, str] = {"rules_of_road_sparse": "rules_of_road"}


def _import_game_module(game_name: str) -> ModuleType:
    name = _PAYOFF_ALIAS.get(game_name, game_name)
    return importlib.import_module(_GAME_MODULE_PREFIX + name)


def _resolve_payoff_opponent(
    game_name: str,
    n_actions: int,
) -> Optional[np.ndarray]:
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


def _build_adversary(
    *,
    adversary_name: str,
    n_actions: int,
    payoff_opponent: Optional[np.ndarray],
    seed: int,
    user_kwargs: Mapping[str, Any],
) -> Any:
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
# M10 agent subclass — threads ``bellman_residual`` and ``episode_return``
# through ``schedule.update_after_episode``. Mirrors :class:`_M9Agent`.
# ---------------------------------------------------------------------------
class _M10Agent(AdaptiveBetaQAgent):
    """Local subclass of :class:`AdaptiveBetaQAgent` for M10.

    The base class's ``end_episode`` calls
    ``schedule.update_after_episode(episode_index, rewards, v_nexts,
    divergence_event=...)`` with NO ``bellman_residual`` /
    ``episode_return`` channels. ``ContractionUCBBetaSchedule`` and
    ``ReturnUCBBetaSchedule`` need those signals (spec §6.5). This
    subclass overrides ``end_episode`` to forward them while keeping
    the underlying TD-update path untouched (the β=0 collapse identity
    in ``_step_update`` is preserved verbatim).

    ``episode_info`` is always ``None`` on standalone cells — the
    runner never builds a regime dict here. (Oracle is rejected at
    config-parse time.)
    """

    def end_episode(  # type: ignore[override]
        self, episode_index: int
    ) -> Dict[str, Any]:
        if int(episode_index) != self._current_episode:
            raise AssertionError(
                f"end_episode received {episode_index} but current episode "
                f"is {self._current_episode}"
            )

        rewards = np.asarray(self._ep_rewards, dtype=np.float64)
        v_nexts = np.asarray(self._ep_v_nexts, dtype=np.float64)
        aligned = np.asarray(self._ep_aligned, dtype=bool)
        d_eff = np.asarray(self._ep_d_eff, dtype=np.float64)
        signed = np.asarray(self._ep_signed_align, dtype=np.float64)
        advantages = np.asarray(self._ep_advantages, dtype=np.float64)
        td_errors = np.asarray(self._ep_td_errors, dtype=np.float64)
        td_targets = np.asarray(self._ep_td_targets, dtype=np.float64)

        q_abs_max = float(np.max(np.abs(self._Q))) if self._Q.size else 0.0
        nan_count = int(np.isnan(self._Q).sum())
        if nan_count > 0 or q_abs_max > self._divergence_threshold:
            self._ep_divergence_event = True

        # Diagnostics snapshot BEFORE the schedule advances (preserves
        # FINAL-BLOCKER-1 semantics carried over from M9).
        sched_diag = self._beta_schedule.diagnostics()

        episode_return = float(rewards.sum()) if rewards.size else 0.0
        bellman_residual = (
            float(np.abs(td_errors).mean()) if td_errors.size else 0.0
        )

        # Push to the schedule with the M9/M10 kwarg surface. Schedules
        # that don't consume the extras silently ignore them.
        self._beta_schedule.update_after_episode(
            self._current_episode,
            rewards,
            v_nexts,
            divergence_event=self._ep_divergence_event,
            episode_info=None,  # standalone has no regime
            bellman_residual=bellman_residual,
            episode_return=episode_return,
        )

        if rewards.size > 0:
            alignment_rate = float(aligned.mean())
            mean_signed_alignment = float(signed.mean())
            frac_positive_signed = float((signed >= 0.0).mean())
            mean_abs_advantage = float(np.abs(advantages).mean())
            mean_d_eff = float(d_eff.mean())
            median_d_eff = float(np.median(d_eff))
            frac_d_eff_below_gamma = float((d_eff < self._gamma).mean())
            frac_d_eff_above_one = float((d_eff > 1.0).mean())
            mean_gamma_minus_d_eff = float((self._gamma - d_eff).mean())
            td_target_abs_max = float(np.max(np.abs(td_targets)))
        else:
            alignment_rate = 0.0
            mean_signed_alignment = 0.0
            frac_positive_signed = 0.0
            mean_abs_advantage = 0.0
            mean_d_eff = 0.0
            median_d_eff = 0.0
            frac_d_eff_below_gamma = 0.0
            frac_d_eff_above_one = 0.0
            mean_gamma_minus_d_eff = 0.0
            td_target_abs_max = 0.0

        return {
            "episode_index": int(self._current_episode),
            "beta_used": float(self._current_beta),
            "beta_raw": float(sched_diag["beta_raw"]),
            "beta_deployed": float(sched_diag["beta_used"]),
            "alignment_rate": alignment_rate,
            "mean_signed_alignment": mean_signed_alignment,
            "frac_positive_signed_alignment": frac_positive_signed,
            "mean_advantage": float(advantages.mean()) if advantages.size else 0.0,
            "mean_abs_advantage": mean_abs_advantage,
            "mean_d_eff": mean_d_eff,
            "median_d_eff": median_d_eff,
            "frac_d_eff_below_gamma": frac_d_eff_below_gamma,
            "frac_d_eff_above_one": frac_d_eff_above_one,
            "mean_gamma_minus_d_eff": mean_gamma_minus_d_eff,
            "bellman_residual": bellman_residual,
            "td_target_abs_max": td_target_abs_max,
            "q_abs_max": q_abs_max,
            "nan_count": nan_count,
            "divergence_event": bool(self._ep_divergence_event),
            "length": int(rewards.size),
            "episode_return": episode_return,
        }


# ---------------------------------------------------------------------------
# UCB schedule introspection helpers
# ---------------------------------------------------------------------------
def _is_ucb_method(method_id: str) -> bool:
    return str(method_id).strip() in ("contraction_UCB_beta", "return_UCB_beta")


def _ucb_arm_snapshot(
    schedule: Any,
) -> Tuple[int, int, float]:
    """Return ``(current_arm_idx, most_pulled_idx, most_pulled_arm_mean)``.

    The schedule must satisfy the ``_BaseUCBBetaSchedule`` introspection
    contract (``_current_arm_idx`` attr; ``pull_counts()`` and
    ``arm_means()`` methods). Caller guarantees ``_is_ucb_method``.
    """
    current_arm_idx = int(getattr(schedule, "_current_arm_idx"))
    pulls = list(schedule.pull_counts())
    means = list(schedule.arm_means())
    if not pulls:
        return current_arm_idx, _UCB_ARM_NA_INT, float("nan")
    most_pulled_idx = int(np.argmax(pulls))
    arm_mean = float(means[most_pulled_idx])
    return current_arm_idx, most_pulled_idx, arm_mean


# ---------------------------------------------------------------------------
# Provenance helpers (verbatim from Stage 1)
# ---------------------------------------------------------------------------
def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _utc_now_iso() -> str:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    if now.endswith("+00:00"):
        now = now[: -len("+00:00")] + "Z"
    return now


def _config_hash(config: Mapping[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _make_run_id(
    *, game: str, subcase: str, method: str, seed: int,
) -> str:
    short_uuid = uuid.uuid4().hex[:8]
    safe_subcase = subcase.replace("/", "_")
    return (
        f"phaseVIII-stage5-{game}-{safe_subcase}-{method}-s{int(seed):04d}-"
        f"{short_uuid}"
    )


# ---------------------------------------------------------------------------
# Subcase / config parsing
# ---------------------------------------------------------------------------
def _parse_subcase(raw: Mapping[str, Any]) -> SubcaseSpec:
    if "id" not in raw:
        raise KeyError(f"subcase entry missing 'id': {raw!r}")
    if "game" not in raw:
        raise KeyError(f"subcase entry {raw['id']!r} missing 'game'")
    if "adversary" not in raw:
        raise KeyError(f"subcase entry {raw['id']!r} missing 'adversary'")
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
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"config at {path} must deserialise to a mapping, got "
            f"{type(raw).__name__}"
        )
    return raw


# ---------------------------------------------------------------------------
# Path-segment helper (verbatim from Stage 1)
# ---------------------------------------------------------------------------
def _format_gamma_segment(gamma: float) -> str:
    return f"gamma_{float(gamma):.2f}"


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
    gamma_override: Optional[float] = None,
    gamma_in_path: bool = False,
) -> Dict[str, Any]:
    """Execute one ``(game, subcase, method, seed)`` cell end-to-end.

    Mirrors :func:`run_phase_VIII_stage1_beta_sweep.run_one_cell`'s
    side-effect contract: writes ``run.json`` and ``metrics.npz``
    under the canonical Phase VIII path and registers the run in
    ``roster``. Method dispatch is the only divergence (see
    :func:`_resolve_method`).
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

    # Per-method kwargs (mirrors M7.2 / M9 conventions). Per-method
    # block has precedence over a global ``method_kwargs`` block.
    method_kwargs: Dict[str, Any] = dict(config.get("method_kwargs", {}) or {})
    per_method_block: Mapping[str, Any] = (
        config.get("method_kwargs_per_method", {}) or {}
    )
    if isinstance(per_method_block, Mapping) and method in per_method_block:
        per_method_extra = per_method_block.get(method, {}) or {}
        if isinstance(per_method_extra, Mapping):
            method_kwargs.update(per_method_extra)

    schedule_method, schedule_hparams = _resolve_method(method, method_kwargs)

    # Run-dir layout (lessons.md #11). Identical to Stage 1.
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

        payoff_opp = _resolve_payoff_opponent(subcase.game, n_actions=2)
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
        agent = _M10Agent(
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

        is_ucb = _is_ucb_method(method)

        # Per-episode buffers — Stage 1 column set, plus ``epsilon`` and
        # the UCB-introspection columns.
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
        ep_epsilon = np.zeros(n_episodes, dtype=np.float64)
        ep_goal_reaches = np.zeros(n_episodes, dtype=np.int64)
        ep_trap_entries = np.zeros(n_episodes, dtype=np.int64)
        # UCB-specific. NaN-coded per spec §7.2 for non-UCB methods.
        ep_ucb_arm_index = np.full(
            n_episodes, _UCB_ARM_NA_INT, dtype=np.int64
        )
        ep_ucb_most_pulled_idx = np.full(
            n_episodes, _UCB_ARM_NA_INT, dtype=np.int64
        )
        ep_ucb_most_pulled_value = np.full(n_episodes, np.nan, dtype=np.float64)

        for e in range(n_episodes):
            agent.begin_episode(e)
            # Snapshot the UCB arm index BEFORE the episode runs — this
            # is the arm whose β is deployed during the current episode
            # (per the schedule's strict-current-only contract).
            if is_ucb:
                arm_idx_now = int(getattr(schedule, "_current_arm_idx"))
                ep_ucb_arm_index[e] = arm_idx_now

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
            ep_epsilon[e] = float(eps_fn(e))
            ep_goal_reaches[e] = int(n_goal)
            ep_trap_entries[e] = int(n_trap)

            # UCB-specific bookkeeping AFTER ``end_episode`` advanced
            # the schedule (so pull_counts / arm_means reflect the
            # just-finished episode's sample).
            if is_ucb:
                _, most_pulled_idx, most_pulled_mean = _ucb_arm_snapshot(
                    schedule
                )
                ep_ucb_most_pulled_idx[e] = most_pulled_idx
                ep_ucb_most_pulled_value[e] = most_pulled_mean

        end_utc = _utc_now_iso()
        wallclock = float(time.time() - start_perf)

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
            "epsilon": ep_epsilon,
            "goal_reaches": ep_goal_reaches,
            "trap_entries": ep_trap_entries,
            # UCB-specific (NaN/-1 for non-UCB methods).
            "ucb_arm_index": ep_ucb_arm_index,
            "ucb_most_pulled_arm_index": ep_ucb_most_pulled_idx,
            "ucb_most_pulled_arm_value": ep_ucb_most_pulled_value,
            # γ as a 0-dim scalar (mirrors Stage 1).
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

        # Final UCB diagnostics snapshot for run.json (cheap; cumulative
        # pull counts and arm-mean vector after the last episode).
        ucb_final_diag: Optional[Dict[str, Any]] = None
        if is_ucb:
            ucb_final_diag = {
                "arm_grid": list(schedule.arm_grid),
                "pull_counts": list(schedule.pull_counts()),
                "arm_means": list(schedule.arm_means()),
                "ucb_c": float(getattr(schedule, "_ucb_c", float("nan"))),
                "n_arms": int(len(schedule.arm_grid)),
                "warm_start_len": int(len(schedule.arm_grid)),
            }

        run_json: Dict[str, Any] = {
            "schema_version": RUN_JSON_SCHEMA_VERSION,
            "run_id": run_id,
            "phase": "VIII",
            "stage": stage,
            "method": method,
            "schedule_method": schedule_method,
            "schedule_hparams": dict(schedule_hparams),
            "method_kwargs": dict(method_kwargs),
            "is_ucb_method": bool(is_ucb),
            "ucb_final_diag": ucb_final_diag,
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

    except Exception as exc:  # noqa: BLE001
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
        except Exception:
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
    method_filter: Optional[Sequence[str]] = None,
    subcase_filter: Optional[Sequence[str]] = None,
    episodes_override: Optional[int] = None,
) -> Phase8RunRoster:
    """Iterate over the (subcase × method × γ × seed) matrix.

    Returns the populated :class:`Phase8RunRoster`. Snapshot written to
    ``<output_root>/raw/<phase>/<suite>/manifest.jsonl``.
    """
    stage = str(config.get("stage", "stage5_adaptive"))
    seeds: Sequence[int] = (
        list(seed_override) if seed_override is not None else
        [int(s) for s in config.get("seeds", [0])]
    )
    methods: List[str] = list(config.get("methods", []))
    if method_filter is not None:
        methods = [m for m in methods if m in set(method_filter)]
        if not methods:
            raise ValueError(
                f"method_filter={list(method_filter)!r} eliminated all "
                f"configured methods"
            )
    if not methods:
        raise ValueError("config must declare a non-empty 'methods' list")

    # Validate methods up front so config typos / oracle_beta on
    # standalone fail before any cell runs.
    for m in methods:
        m_str = str(m).strip()
        if (
            m_str not in M10_STANDALONE_METHOD_IDS
            and not _is_fixed_beta_method(m_str)
        ):
            # oracle_beta is excluded from M10_STANDALONE_METHOD_IDS
            # by design; we re-raise the explicit guidance from
            # _resolve_method here for early failure.
            if m_str == "oracle_beta":
                raise ValueError(
                    "oracle_beta is NOT supported on standalone cells: it "
                    "requires a regime label that only sign-switching "
                    "composites expose (spec §6.6). Use Stage 4 "
                    "(run_phase_VIII_stage4_composite) instead."
                )
            raise ValueError(
                f"unknown M10 standalone method_id={m!r}; valid ids: "
                f"{sorted(M10_STANDALONE_METHOD_IDS)} or 'fixed_beta_+x' "
                f"/ 'fixed_beta_-x'."
            )

    raw_subcases: Sequence[Mapping[str, Any]] = list(
        config.get("subcases", []) or []
    )
    if not raw_subcases:
        raise ValueError("config must declare a non-empty 'subcases' list")
    subcases: List[SubcaseSpec] = [_parse_subcase(s) for s in raw_subcases]
    if subcase_filter is not None:
        sf = set(str(s) for s in subcase_filter)
        subcases = [sc for sc in subcases if sc.subcase_id in sf]
        if not subcases:
            raise ValueError(
                f"subcase_filter={list(subcase_filter)!r} eliminated all "
                f"configured subcases"
            )

    # γ resolution (mirrors Stage 1).
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

    if episodes_override is not None:
        config = dict(config)
        config["episodes"] = int(episodes_override)

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
        prog="run_phase_VIII_stage5_adaptive",
        description=(
            "Phase VIII Stage 5 adaptive-method dispatcher (M10; spec §10.6)"
        ),
    )
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the Stage 5 YAML config.",
    )
    p.add_argument(
        "--seed-list",
        type=str,
        default=None,
        help="Comma-separated override of the YAML 'seeds' list.",
    )
    p.add_argument(
        "--methods",
        type=str,
        default=None,
        help=(
            "Comma-separated method-id filter (subset of the YAML "
            "'methods' list)."
        ),
    )
    p.add_argument(
        "--subcases",
        type=str,
        default=None,
        help="Comma-separated subcase-id filter (subset of YAML 'subcases').",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override the YAML 'episodes' value (smoke runs).",
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
    method_filter: Optional[List[str]] = None
    if args.methods is not None:
        method_filter = [s.strip() for s in args.methods.split(",") if s.strip()]
    subcase_filter: Optional[List[str]] = None
    if args.subcases is not None:
        subcase_filter = [s.strip() for s in args.subcases.split(",") if s.strip()]
    dispatch(
        config=config,
        seed_override=seed_override,
        output_root=Path(args.output_root),
        config_path=Path(args.config),
        fail_fast=bool(args.fail_fast),
        method_filter=method_filter,
        subcase_filter=subcase_filter,
        episodes_override=args.episodes,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI entry
    sys.exit(main())
