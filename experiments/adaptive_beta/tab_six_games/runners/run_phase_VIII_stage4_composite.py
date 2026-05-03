"""Phase VIII Stage 4 — sign-switching composite runner (M9).

Spec authority
--------------
- ``docs/specs/phase_VIII_tab_six_games.md`` §10.5 (Stage 4 sign-switching
  composite + oracle-validation gate).
- §6.6 (Oracle / hand-adaptive contracts; oracle reads
  ``info["regime"]`` exactly once per episode-end update).
- §6.5 (UCB schedule contracts).
- §8.1 logging schema (run.json + metrics.npz).
- §8.2 ``Phase8RunRoster`` (every dispatched run is registered).
- §13.6 result-root regression.
- ``tasks/M9_plan.md`` (canonical scope).

Boundaries
----------
- DOES NOT modify M7.1 / M7.2 runners — Stage 1 dispatches fixed-β
  arms only, Stage 2 dispatches Q-learning baselines only. M9 needs
  composite-env construction + oracle-with-env-reference + UCB
  scheduling, which doesn't fit Stage 1's mental model.
- DOES NOT touch ``mushroom-rl-dev/`` (CLAUDE.md §4) or
  ``src/lse_rl/operator/``.
- DOES NOT modify ``experiments/adaptive_beta/agents.py``: the
  ``AdaptiveBetaQAgent`` class lacks an ``episode_info`` channel for
  the schedule, so this runner subclasses it locally as
  :class:`_M9Agent` to thread regime / bellman-residual / return
  through ``schedule.update_after_episode``. The subclass is
  internal to this module.

Method dispatch
---------------
- ``vanilla`` → ZeroBetaSchedule (β = 0).
- ``fixed_positive_TAB`` → FixedBetaSchedule(+1, β0=cfg).
- ``fixed_negative_TAB`` → FixedBetaSchedule(−1, β0=cfg).
- ``oracle_beta`` → OracleBetaSchedule with regime-keyed lookup.
  Receives ``episode_info={"regime": env.regime}`` at episode end.
- ``hand_adaptive_beta`` → HandAdaptiveBetaSchedule (rule-based,
  consumes the agent's own running A_e signal; NO env access).
- ``contraction_UCB_beta`` → ContractionUCBBetaSchedule (21-arm UCB
  over Bellman-residual log-ratio reward).

Regime-exposure discipline
--------------------------
- The composite env exposes ``env.regime`` publicly so the runner can
  feed it to OracleBetaSchedule. The runner reads ``env.regime``
  ONLY inside the oracle method branch.
- For non-oracle methods, ``episode_info=None`` is passed through to
  ``schedule.update_after_episode``. The schedule's own behaviour
  (vanilla / fixed / hand / UCB) does NOT inspect episode_info, per
  the spec §10.5 / §6.6 contract pinned by
  ``test_oracle_beta_schedule.test_oracle_is_only_schedule_reading_regime``.

CLI
---
::

    python -m experiments.adaptive_beta.tab_six_games.runners.\\
run_phase_VIII_stage4_composite \\
        --config experiments/adaptive_beta/tab_six_games/configs/\\
stage4_composite_AC_RR_gamma060.yaml \\
        --seed-list 0,1,2 --dwell 250 --methods vanilla,oracle_beta
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
    METHOD_CONTRACTION_UCB_BETA,
    METHOD_FIXED_NEGATIVE,
    METHOD_FIXED_POSITIVE,
    METHOD_HAND_ADAPTIVE_BETA,
    METHOD_ORACLE_BETA,
    METHOD_VANILLA,
    build_schedule,
)
from experiments.adaptive_beta.tab_six_games.composites import (  # noqa: E402
    SignSwitchingComposite,
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
    "regime_per_episode",
    "episodes_since_switch",
)

#: Phase VIII method ID set understood by this runner. Any method
#: outside this set raises ``ValueError`` at config-parse time.
M9_METHOD_IDS: frozenset = frozenset({
    "vanilla",
    "fixed_positive_TAB",
    "fixed_negative_TAB",
    "oracle_beta",
    "hand_adaptive_beta",
    "contraction_UCB_beta",
})


# ---------------------------------------------------------------------------
# Composite + per-component descriptors (parsed from YAML)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _ComponentSpec:
    """One side of the (G_+, G_-) composite parsed from YAML."""

    game: str
    game_kwargs: Dict[str, Any]
    adversary_name: str
    adversary_kwargs: Dict[str, Any]
    gamma: float


@dataclass(frozen=True)
class CompositeSpec:
    """Full M9 composite specification parsed from YAML."""

    composite_type: str
    g_plus: _ComponentSpec
    g_minus: _ComponentSpec
    dwell_grid: Tuple[int, ...]


def _parse_component(raw: Mapping[str, Any], side: str) -> _ComponentSpec:
    if "game" not in raw:
        raise KeyError(f"composite.{side} missing 'game'")
    if "adversary" not in raw:
        raise KeyError(f"composite.{side} missing 'adversary'")
    return _ComponentSpec(
        game=str(raw["game"]),
        game_kwargs=dict(raw.get("game_kwargs", {}) or {}),
        adversary_name=str(raw["adversary"]),
        adversary_kwargs=dict(raw.get("adversary_kwargs", {}) or {}),
        gamma=float(raw.get("gamma", raw.get("γ", 0.95))),
    )


def _parse_composite(raw: Mapping[str, Any]) -> CompositeSpec:
    if "composite" not in raw:
        raise KeyError("config missing 'composite' section")
    comp = raw["composite"]
    if "g_plus" not in comp or "g_minus" not in comp:
        raise KeyError("composite section requires 'g_plus' and 'g_minus'")
    if "dwell_grid" not in comp:
        raise KeyError("composite section requires 'dwell_grid' (non-empty list)")
    dwell_grid = tuple(int(d) for d in comp["dwell_grid"])
    if not dwell_grid:
        raise ValueError("composite.dwell_grid must be non-empty")
    if any(d < 1 for d in dwell_grid):
        raise ValueError(
            f"composite.dwell_grid entries must be >= 1, got {dwell_grid}"
        )
    return CompositeSpec(
        composite_type=str(comp.get("type", "sign_switching")),
        g_plus=_parse_component(comp["g_plus"], "g_plus"),
        g_minus=_parse_component(comp["g_minus"], "g_minus"),
        dwell_grid=dwell_grid,
    )


# ---------------------------------------------------------------------------
# Adversary construction (mirror of Stage 1's helper).
# ---------------------------------------------------------------------------
_GAME_MODULE_PREFIX = "experiments.adaptive_beta.strategic_games.games."
_PAYOFF_ALIAS: Dict[str, str] = {"rules_of_road_sparse": "rules_of_road"}


def _import_game_module(game_name: str) -> ModuleType:
    name = _PAYOFF_ALIAS.get(game_name, game_name)
    return importlib.import_module(_GAME_MODULE_PREFIX + name)


def _resolve_payoff_opponent(
    game_name: str, n_actions: int
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


def _build_component_env(
    component: _ComponentSpec,
    *,
    seed: int,
):
    if component.game not in GAME_REGISTRY:
        known = ", ".join(sorted(GAME_REGISTRY.keys()))
        raise KeyError(
            f"unknown game {component.game!r}; registered: [{known}]"
        )
    if component.adversary_name not in ADVERSARY_REGISTRY:
        known = ", ".join(sorted(ADVERSARY_REGISTRY.keys()))
        raise KeyError(
            f"unknown adversary {component.adversary_name!r}; "
            f"registered: [{known}]"
        )

    payoff_opp = _resolve_payoff_opponent(component.game, n_actions=2)
    if payoff_opp is not None:
        n_opp_actions = int(np.asarray(payoff_opp).shape[1])
    else:
        n_opp_actions = int(component.adversary_kwargs.get("n_actions", 2))

    adversary = _build_adversary(
        adversary_name=component.adversary_name,
        n_actions=n_opp_actions,
        payoff_opponent=payoff_opp,
        seed=int(seed),
        user_kwargs=component.adversary_kwargs,
    )
    # Pass γ into the env via game_kwargs (the matrix-game build()
    # signature accepts ``gamma`` in **kwargs).
    game_kwargs = dict(component.game_kwargs)
    game_kwargs["gamma"] = float(component.gamma)
    env = make_game(
        component.game,
        adversary=adversary,
        seed=int(seed),
        **game_kwargs,
    )
    return env


# ---------------------------------------------------------------------------
# M9 agent subclass: extends end_episode to thread episode_info /
# bellman_residual / episode_return into the schedule.
# ---------------------------------------------------------------------------
class _M9Agent(AdaptiveBetaQAgent):
    """Local subclass of :class:`AdaptiveBetaQAgent` for M9.

    The base class's ``end_episode`` calls
    ``schedule.update_after_episode(episode_index, rewards, v_nexts,
    divergence_event=...)`` with NO ``episode_info`` channel. Three
    of the five M9 schedules need extra signals:

    - :class:`OracleBetaSchedule` — needs ``episode_info["regime"]``.
    - :class:`HandAdaptiveBetaSchedule` — uses the base advantage
      stream (already wired); accepts the smoothed-A path.
    - :class:`ContractionUCBBetaSchedule` — needs ``bellman_residual``.

    This subclass overrides ``end_episode`` to accept the extra
    kwargs and forward them. The underlying TD-update path in
    :meth:`AdaptiveBetaQAgent._step_update` is untouched, so the
    agent's β=0 collapse identity (FINAL-BLOCKER preserve) holds.
    """

    def end_episode(  # type: ignore[override]
        self,
        episode_index: int,
        *,
        episode_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Mirror the parent's pre-update bookkeeping so we can pass
        # the extra kwargs into ``update_after_episode``. We do NOT
        # delegate to ``super().end_episode`` because the parent's
        # implementation calls update_after_episode with a fixed
        # signature; we re-implement its body to inject the extras
        # at the single update call site.
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

        # Diagnostics snapshot BEFORE the schedule advances (parent
        # FINAL-BLOCKER-1 fix retained).
        sched_diag = self._beta_schedule.diagnostics()

        # Compute the per-episode return + bellman_residual to feed
        # the UCB / contraction schedules. Defined as the agent's
        # cumulative reward and the mean |TD error| respectively
        # (consistent with the base class's diagnostics block).
        episode_return = float(rewards.sum()) if rewards.size else 0.0
        bellman_residual = (
            float(np.abs(td_errors).mean()) if td_errors.size else 0.0
        )

        # Push to the schedule with the full kwarg surface. Schedules
        # that don't consume the extras silently ignore them (per
        # spec §6.5/§6.6).
        self._beta_schedule.update_after_episode(
            self._current_episode,
            rewards,
            v_nexts,
            divergence_event=self._ep_divergence_event,
            episode_info=episode_info,
            bellman_residual=bellman_residual,
            episode_return=episode_return,
        )

        # Episode-level aggregates — mirror the parent verbatim.
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
# Method → schedule resolution (M9-specific)
# ---------------------------------------------------------------------------
def _resolve_method(
    method_id: str,
    method_kwargs: Mapping[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """Map an M9 method ID to ``(schedule_method_id, hparams)``.

    The M9 vocabulary is explicit (no ``fixed_beta_+0.5`` style; the
    sign is in the method name). Hyperparameters come from the per-
    method ``method_kwargs_per_method`` block of the YAML.
    """
    name = str(method_id).strip()
    if name not in M9_METHOD_IDS:
        raise ValueError(
            f"unknown M9 method_id={method_id!r}; valid: {sorted(M9_METHOD_IDS)}"
        )
    if name == "vanilla":
        return METHOD_VANILLA, {}
    if name == "fixed_positive_TAB":
        beta0 = float(method_kwargs.get("beta0", 0.10))
        return METHOD_FIXED_POSITIVE, {"beta0": beta0}
    if name == "fixed_negative_TAB":
        beta0 = float(method_kwargs.get("beta0", 0.50))
        return METHOD_FIXED_NEGATIVE, {"beta0": beta0}
    if name == "oracle_beta":
        beta_g_plus = float(method_kwargs.get("beta_g_plus", 0.10))
        beta_g_minus = float(method_kwargs.get("beta_g_minus", -0.50))
        regime_to_beta = {"plus": beta_g_plus, "minus": beta_g_minus}
        return METHOD_ORACLE_BETA, {"regime_to_beta": regime_to_beta}
    if name == "hand_adaptive_beta":
        # Pre-registered defaults (spec §6.6); no per-task tuning.
        return METHOD_HAND_ADAPTIVE_BETA, {}
    if name == "contraction_UCB_beta":
        # Defaults from spec §6.5: warm-start over 21 arms, c=1.0.
        return METHOD_CONTRACTION_UCB_BETA, {}
    raise ValueError(f"unhandled method_id={method_id!r}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Provenance helpers
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
    *, method: str, dwell: int, seed: int
) -> str:
    short_uuid = uuid.uuid4().hex[:8]
    return (
        f"phaseVIII-stage4-dwell{int(dwell)}-{method}-s{int(seed):04d}-"
        f"{short_uuid}"
    )


# ---------------------------------------------------------------------------
# Composite-cell label (for run-dir layout)
# ---------------------------------------------------------------------------
def _composite_cell_label(spec: CompositeSpec) -> str:
    """Cell label used as the ``task`` for run paths.

    Form: ``composite_sign_switching/<g_plus_game>__<g_minus_game>``.
    """
    return (
        "composite_sign_switching/"
        f"{spec.g_plus.game}__{spec.g_minus.game}"
    )


# ---------------------------------------------------------------------------
# Single-cell runner
# ---------------------------------------------------------------------------
def run_one_cell(
    *,
    config: Mapping[str, Any],
    composite: CompositeSpec,
    method: str,
    dwell: int,
    seed: int,
    output_root: Path,
    stage: str,
    roster: Phase8RunRoster,
    config_path: Optional[Path],
    git_commit: str,
) -> Dict[str, Any]:
    """Execute one ``(composite, dwell, method, seed)`` cell end-to-end.

    Side effects: writes ``run.json`` and ``metrics.npz`` under
    ``<output_root>/raw/<phase>/<suite>/<cell>/dwell_<d>/<method>/
    seed_<seed>/`` and registers the run in ``roster``.
    """
    n_episodes = int(config["episodes"])
    learning_rate = float(config.get("learning_rate", 0.1))
    eps_cfg = config.get("epsilon", {}) or {}
    epsilon_start = float(eps_cfg.get("start", 1.0))
    epsilon_end = float(eps_cfg.get("end", 0.05))
    epsilon_decay_episodes = int(
        eps_cfg.get("decay_episodes", max(1, n_episodes // 2))
    )
    q_init = float(config.get("q_init", 0.0))

    method_kwargs_all = config.get("method_kwargs_per_method", {}) or {}
    method_kwargs = dict(method_kwargs_all.get(method, {}) or {})

    schedule_method, schedule_hparams = _resolve_method(method, method_kwargs)

    # Resolve the absolute run dir.
    cell_label = _composite_cell_label(composite)
    algorithm_segment = f"dwell_{int(dwell)}/{method}"
    run_dir = make_run_dir(
        base=output_root / "raw",
        phase="VIII",
        suite=stage,
        task=cell_label,
        algorithm=algorithm_segment,
        seed=int(seed),
        exist_ok=True,
    )

    run_id = _make_run_id(method=method, dwell=int(dwell), seed=int(seed))
    cfg_hash = _config_hash(config)

    roster.append(
        run_id=run_id,
        config_hash=cfg_hash,
        seed=int(seed),
        game=cell_label,
        subcase=f"dwell_{int(dwell)}",
        method=method,
        git_commit=git_commit,
        gamma=float(composite.g_plus.gamma),
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
        # Build component envs at distinct seed offsets so the two
        # sub-envs' adversary RNGs don't shadow each other when both
        # are seeded with the cell-level seed. Offsets are deterministic.
        seed_plus = int(seed)
        seed_minus = int(seed) + 100_000
        env_plus = _build_component_env(composite.g_plus, seed=seed_plus)
        env_minus = _build_component_env(composite.g_minus, seed=seed_minus)

        env = SignSwitchingComposite(
            env_g_plus=env_plus,
            env_g_minus=env_minus,
            dwell=int(dwell),
            seed=int(seed),
        )

        n_states = int(env.info.observation_space.size[0])
        n_actions = int(env.info.action_space.size[0])
        gamma = float(env.info.gamma)
        # Composite has no fixed canonical sign — wrong_sign and
        # adaptive_magnitude_only are not legal here (handled by the
        # method dispatch above; both are excluded from M9_METHOD_IDS).
        env_canonical_sign: Optional[str] = None

        eps_fn = linear_epsilon_schedule(
            start=epsilon_start,
            end=epsilon_end,
            decay_episodes=epsilon_decay_episodes,
        )
        schedule = build_schedule(
            schedule_method, env_canonical_sign, schedule_hparams
        )

        agent_rng = np.random.default_rng(int(seed))
        agent = _M9Agent(
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
        # Composite-specific arrays.
        ep_regime = np.zeros(n_episodes, dtype=np.int8)
        ep_eps_since_switch = np.zeros(n_episodes, dtype=np.int64)
        ep_switch_count = np.zeros(n_episodes, dtype=np.int64)
        ep_switch_event = np.zeros(n_episodes, dtype=np.uint8)

        prev_switch_count = 0

        for e in range(n_episodes):
            agent.begin_episode(e)
            state, info = env.reset()
            s = int(np.asarray(state).flat[0])
            # Capture regime AT EPISODE START (this is the regime
            # under which this episode is played; ``env.regime``
            # remains constant throughout the episode by composite
            # contract — flips only on episode boundaries).
            regime_label_this_episode = env.regime
            ep_regime[e] = int(env.regime_int)

            t = 0
            ret = 0.0
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
                t += 1
                s = ns
                if absorbing:
                    break

            # Build episode_info for the schedule. Only the oracle
            # method consumes ``regime``; for non-oracle methods
            # we pass None so the schedule does not see the regime
            # (spec §10.5 / §6.6 enforced AT THE RUNNER BOUNDARY).
            if method == "oracle_beta":
                episode_info: Optional[Dict[str, Any]] = {
                    "regime": regime_label_this_episode,
                }
            else:
                episode_info = None

            ep_diag = agent.end_episode(e, episode_info=episode_info)
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
            # Composite-specific bookkeeping. Both ``episodes_since_switch``
            # and ``switch_count`` are read AFTER ``env.step(absorbing)``
            # incremented them, so they reflect the just-completed
            # episode's post-state.
            ep_eps_since_switch[e] = int(env.episodes_since_switch)
            ep_switch_count[e] = int(env.switch_count)
            switch_event = int(env.switch_count) > prev_switch_count
            ep_switch_event[e] = 1 if switch_event else 0
            prev_switch_count = int(env.switch_count)

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
            # Composite-specific.
            "regime_per_episode": ep_regime,
            "episodes_since_switch": ep_eps_since_switch,
            "switch_count": ep_switch_count,
            "switch_event": ep_switch_event,
            "gamma": np.float64(gamma),
            "dwell": np.int64(int(dwell)),
        }
        schema_header = make_npz_schema(
            phase="VIII",
            task=cell_label,
            algorithm=method,
            seed=int(seed),
            storage_mode="rl_online",
            arrays=sorted(arrays.keys()),
            schema_version=METRICS_NPZ_SCHEMA_VERSION,
        )
        save_npz_with_schema(run_dir / "metrics.npz", schema_header, arrays)

        run_json: Dict[str, Any] = {
            "schema_version": RUN_JSON_SCHEMA_VERSION,
            "run_id": run_id,
            "phase": "VIII",
            "stage": stage,
            "method": method,
            "schedule_method": schedule_method,
            "schedule_hparams": dict(schedule_hparams),
            "method_kwargs": dict(method_kwargs),
            "composite": {
                "type": composite.composite_type,
                "g_plus": {
                    "game": composite.g_plus.game,
                    "game_kwargs": dict(composite.g_plus.game_kwargs),
                    "adversary": composite.g_plus.adversary_name,
                    "adversary_kwargs": dict(composite.g_plus.adversary_kwargs),
                    "gamma": float(composite.g_plus.gamma),
                },
                "g_minus": {
                    "game": composite.g_minus.game,
                    "game_kwargs": dict(composite.g_minus.game_kwargs),
                    "adversary": composite.g_minus.adversary_name,
                    "adversary_kwargs": dict(composite.g_minus.adversary_kwargs),
                    "gamma": float(composite.g_minus.gamma),
                },
                "dwell": int(dwell),
            },
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
            "config": {
                "path": str(config_path) if config_path is not None else None,
                "hash": cfg_hash,
            },
            "env": {
                "n_states": int(n_states),
                "n_actions": int(n_actions),
                "canonical_sign": env_canonical_sign,
            },
            "regime_summary": {
                "switch_count_total": int(env.switch_count),
                "initial_regime": "plus",
                "regime_history_len": len(env.regime_history),
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
    dwell_filter: Optional[Sequence[int]] = None,
    episodes_override: Optional[int] = None,
) -> Phase8RunRoster:
    """Iterate over (method × dwell × seed) for the M9 composite.

    Returns the populated :class:`Phase8RunRoster`. Snapshot written
    to ``<output_root>/raw/<phase>/<suite>/manifest.jsonl``.
    """
    stage = str(config.get("stage", "stage4_composite"))
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
    for m in methods:
        if m not in M9_METHOD_IDS:
            raise ValueError(
                f"unknown method {m!r}; valid M9 methods: {sorted(M9_METHOD_IDS)}"
            )

    composite = _parse_composite(config)
    dwell_grid: Sequence[int] = list(composite.dwell_grid)
    if dwell_filter is not None:
        dwell_filter_set = {int(d) for d in dwell_filter}
        dwell_grid = [d for d in dwell_grid if d in dwell_filter_set]
        if not dwell_grid:
            raise ValueError(
                f"dwell_filter={list(dwell_filter)!r} eliminated all "
                f"configured dwell values"
            )

    # Allow CLI to override the episode count without rewriting the
    # whole config file (smoke runs).
    if episodes_override is not None:
        config = dict(config)
        config["episodes"] = int(episodes_override)

    output_root = Path(output_root)
    roster_dir = output_root / "raw" / "VIII" / stage
    roster_dir.mkdir(parents=True, exist_ok=True)
    roster_path = roster_dir / "manifest.jsonl"

    roster = Phase8RunRoster(base_path=output_root)
    git_commit = _git_sha()

    failures: List[Tuple[str, int, int, str]] = []
    for dwell in dwell_grid:
        for method in methods:
            for seed in seeds:
                try:
                    run_one_cell(
                        config=config,
                        composite=composite,
                        method=str(method),
                        dwell=int(dwell),
                        seed=int(seed),
                        output_root=output_root,
                        stage=stage,
                        roster=roster,
                        config_path=config_path,
                        git_commit=git_commit,
                    )
                except Exception as exc:  # noqa: BLE001
                    msg = f"{type(exc).__name__}: {exc}"
                    failures.append((str(method), int(dwell), int(seed), msg))
                    if fail_fast:
                        roster.write_atomic(roster_path)
                        raise
                roster.write_atomic(roster_path)

    if failures:
        with open(roster_dir / "failures.json", "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "method": method,
                        "dwell": dwell,
                        "seed": seed,
                        "error": err,
                    }
                    for (method, dwell, seed, err) in failures
                ],
                f,
                indent=2,
                sort_keys=True,
            )
            f.write("\n")

    return roster


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
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
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_phase_VIII_stage4_composite",
        description=(
            "Phase VIII Stage 4 sign-switching composite dispatcher (M9; "
            "spec §10.5)"
        ),
    )
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the Stage 4 YAML config.",
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
        "--dwell",
        type=str,
        default=None,
        help="Comma-separated dwell-value filter (subset of dwell_grid).",
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
            "results/adaptive_beta/tab_six_games."
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
    dwell_filter: Optional[List[int]] = None
    if args.dwell is not None:
        dwell_filter = [int(s) for s in args.dwell.split(",") if s.strip()]
    dispatch(
        config=config,
        seed_override=seed_override,
        output_root=Path(args.output_root),
        config_path=Path(args.config),
        fail_fast=bool(args.fail_fast),
        method_filter=method_filter,
        dwell_filter=dwell_filter,
        episodes_override=args.episodes,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI entry
    sys.exit(main())
