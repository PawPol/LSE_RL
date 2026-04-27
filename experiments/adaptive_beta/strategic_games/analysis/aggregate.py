"""Phase VII-B Stage B2 aggregator.

Spec authority:
- ``tasks/phase_VII_B_strategic_learning_coding_agent_spec.md`` §§9.1, 10,
  11.1, 15.
- Parallels: :mod:`experiments.adaptive_beta.analyze` (Phase VII aggregator).

Public API
----------
- :func:`load_run_summary` — discover and load every run under a
  ``raw/`` directory; return one DataFrame row per
  ``(game, adversary, method, seed)`` triple with both performance and
  strategic-mechanism columns.
- :func:`paired_diffs` — paired-bootstrap (10 000 resamples) means + 95%
  CIs of method - baseline for each ``(game, adversary, method)`` cell.
- :func:`event_aligned_panel` — build the §10 event-aligned panels (return,
  beta, effective discount, alignment rate, opponent entropy) by reading
  ``transitions.parquet`` (or the strategic-transitions file) and slicing
  ``±half_window`` around event indices identified by an event flag.
- :func:`promotion_gate` — apply the Stage B2-Dev promotion gate (spec §11.1)
  per ``(game, adversary)`` cell.

Determinism
-----------
The bootstrap uses a fixed seed (``BOOTSTRAP_SEED``) so promotion verdicts
are reproducible across machines. Spec §15: paired seeds, CIs.

Notes
-----
- Plotting is NOT done here (spec §4 hands that to ``plot_*`` modules in
  the same ``analysis/`` directory). This module returns DataFrames /
  arrays only.
- ``load_run_summary`` tolerates missing strategic columns: a run that
  predates the strategic schema fills them with ``np.nan``. This makes the
  aggregator usable on the existing Phase VII raw/ tree for regression.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from experiments.adaptive_beta.strategic_games.metrics import event_aligned_window

# ---------------------------------------------------------------------------
# Constants (paralleling Phase VII analyze.py for reproducibility).
# ---------------------------------------------------------------------------
BOOTSTRAP_RESAMPLES: int = 10_000
BOOTSTRAP_SEED: int = 0xB2DEF
DEFAULT_GROUP_KEYS: Tuple[str, ...] = ("game", "adversary", "method")
DEFAULT_PROMOTION_KEYS: Tuple[str, ...] = ("game", "adversary")

# Episode columns we expect to exist on the strategic side. Missing ones
# are tolerated and filled with NaN.
_STRATEGIC_EPISODE_OPTIONAL: Tuple[str, ...] = (
    "opponent_policy_entropy",
    "policy_total_variation",
    "support_shift",
    "model_rejected",
    "search_phase",
    "memory_m",
    "inertia_lambda",
    "temperature",
    "tau",
)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_run_summary(raw_dir: Path) -> pd.DataFrame:
    """Walk ``raw_dir`` and return a per-run summary DataFrame.

    Parameters
    ----------
    raw_dir
        Root directory containing per-run subfolders. Each leaf is expected
        to have a ``metrics.npz`` and a ``run.json`` (matching the existing
        Phase VII contract).

    Returns
    -------
    df : pd.DataFrame
        One row per run with columns:

        ``run_id, game, adversary, method, seed, n_episodes, auc_return,
        final_return, mean_alignment_rate, mean_d_eff, catastrophic_count,
        diverged_count, support_shift_count, model_rejected_count,
        search_phase_count, opponent_policy_entropy_mean,
        policy_total_variation_mean, memory_m, inertia_lambda,
        temperature, tau, raw_dir``.

    Notes
    -----
    Missing strategic columns are filled with ``np.nan``. Runs whose
    ``run.json`` cannot be parsed are skipped (counted in stderr-style log
    only).
    """
    raw_root = Path(raw_dir)
    if not raw_root.exists():
        raise FileNotFoundError(f"raw_dir not found: {raw_root}")

    rows: List[Dict[str, Any]] = []
    for metrics_path in sorted(raw_root.rglob("metrics.npz")):
        run_dir = metrics_path.parent
        run_json = run_dir / "run.json"
        if not run_json.exists():
            continue
        try:
            with open(run_json, "r", encoding="utf-8") as f:
                run_meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        try:
            npz = np.load(metrics_path, allow_pickle=False)
        except (OSError, ValueError):
            continue

        row = _summarize_run(
            run_dir=run_dir,
            run_meta=run_meta,
            metrics={k: npz[k] for k in npz.files},
        )
        rows.append(row)

    if not rows:
        # Return an empty frame with the canonical column set so downstream
        # callers don't have to special-case empty runs.
        return _empty_summary_frame()
    return pd.DataFrame(rows)


def _summarize_run(
    *,
    run_dir: Path,
    run_meta: Dict[str, Any],
    metrics: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Compute a single per-run summary row."""
    # Identity tags. Phase VII-B run.json must include game/adversary/method/seed.
    game = str(run_meta.get("game", run_meta.get("env", "unknown")))
    adversary = str(run_meta.get("adversary", "unknown"))
    method = str(run_meta.get("method", "unknown"))
    seed = int(run_meta.get("seed", -1))
    run_id = str(run_meta.get("run_id", run_dir.name))

    returns = metrics.get("return")
    if returns is None:
        returns = np.zeros(0, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64).reshape(-1)
    n_episodes = int(returns.size)
    auc = float(returns.sum()) if n_episodes > 0 else float("nan")
    final = (
        float(returns[-min(500, n_episodes):].mean())
        if n_episodes > 0
        else float("nan")
    )

    align_arr = _safe_array(metrics, "alignment_rate")
    d_eff_arr = _safe_array(metrics, "mean_d_eff")
    cata_arr = _safe_array(metrics, "catastrophic")
    div_arr = _safe_array(metrics, "divergence_event")
    if div_arr.size == 0:
        # Older Phase VII metrics may name it diverged.
        div_arr = _safe_array(metrics, "diverged")

    # Strategic-side optional columns.
    opp_ent = _safe_array(metrics, "opponent_policy_entropy")
    pol_tv = _safe_array(metrics, "policy_total_variation")
    support_shift = _safe_array(metrics, "support_shift")
    model_rejected = _safe_array(metrics, "model_rejected")
    search_phase = _safe_array(metrics, "search_phase")

    # Adversary parameters live in run.json (single per-run scalars).
    memory_m = run_meta.get("memory_m")
    inertia_lambda = run_meta.get("inertia_lambda")
    temperature = run_meta.get("temperature")
    tau = run_meta.get("tau")

    return {
        "run_id": run_id,
        "game": game,
        "adversary": adversary,
        "method": method,
        "seed": seed,
        "n_episodes": n_episodes,
        "auc_return": auc,
        "final_return": final,
        "mean_alignment_rate": _nanmean(align_arr),
        "mean_d_eff": _nanmean(d_eff_arr),
        "catastrophic_count": _nansum_bool(cata_arr),
        "diverged_count": _nansum_bool(div_arr),
        "support_shift_count": _nansum_bool(support_shift),
        "model_rejected_count": _nansum_bool(model_rejected),
        "search_phase_count": _nansum_bool(search_phase),
        "opponent_policy_entropy_mean": _nanmean(opp_ent),
        "policy_total_variation_mean": _nanmean(pol_tv),
        "memory_m": memory_m,
        "inertia_lambda": inertia_lambda,
        "temperature": temperature,
        "tau": tau,
        "raw_dir": str(run_dir),
    }


def _safe_array(metrics: Dict[str, np.ndarray], key: str) -> np.ndarray:
    a = metrics.get(key)
    if a is None:
        return np.zeros(0, dtype=np.float64)
    arr = np.asarray(a).reshape(-1)
    # Bool arrays must remain bool for sum semantics; everything else float.
    if arr.dtype == bool:
        return arr
    return arr.astype(np.float64, copy=False)


def _nanmean(a: np.ndarray) -> float:
    if a.size == 0:
        return float("nan")
    if a.dtype == bool:
        return float(a.mean())
    finite = np.isfinite(a)
    if not np.any(finite):
        return float("nan")
    return float(a[finite].mean())


def _nansum_bool(a: np.ndarray) -> int:
    if a.size == 0:
        return 0
    if a.dtype == bool:
        return int(a.sum())
    finite = np.isfinite(a)
    if not np.any(finite):
        return 0
    return int((a[finite] > 0).sum())


def _empty_summary_frame() -> pd.DataFrame:
    cols = [
        "run_id", "game", "adversary", "method", "seed",
        "n_episodes", "auc_return", "final_return",
        "mean_alignment_rate", "mean_d_eff",
        "catastrophic_count", "diverged_count",
        "support_shift_count", "model_rejected_count",
        "search_phase_count",
        "opponent_policy_entropy_mean", "policy_total_variation_mean",
        "memory_m", "inertia_lambda", "temperature", "tau",
        "raw_dir",
    ]
    return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})


# ---------------------------------------------------------------------------
# Paired bootstrap
# ---------------------------------------------------------------------------

def _paired_bootstrap_ci(
    diffs: np.ndarray,
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Tuple[float, float, float]:
    """Return (mean, ci_lo, ci_hi) of the paired-diff vector via bootstrap.

    Empty / all-NaN inputs return ``(nan, nan, nan)``. With <2 finite
    samples returns ``(mean(finite_or_nan), nan, nan)``.
    """
    arr = np.asarray(diffs, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean_d = float(finite.mean())
    if finite.size < 2:
        return (mean_d, float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, finite.size, size=(n_resamples, finite.size))
    boot_means = finite[idx].mean(axis=1)  # shape (n_resamples,)
    lo = float(np.percentile(boot_means, 2.5))
    hi = float(np.percentile(boot_means, 97.5))
    return (mean_d, lo, hi)


def paired_diffs(
    df: pd.DataFrame,
    baseline: str = "vanilla",
    metric: str = "auc_return",
    *,
    group_keys: Iterable[str] = DEFAULT_GROUP_KEYS,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Paired-bootstrap difference (method - baseline) per cell.

    Parameters
    ----------
    df
        Run-summary DataFrame from :func:`load_run_summary`. Must contain
        ``seed``, ``method``, the ``group_keys`` columns, and ``metric``.
    baseline
        Baseline method label; defaults to ``"vanilla"``.
    metric
        Per-run metric column name. Default ``"auc_return"`` is the spec §15
        primary endpoint.
    group_keys
        Column tuple defining the cells over which paired-diffs are
        computed (defaults to ``(game, adversary, method)``; baseline is
        held constant within each ``(game, adversary)``).
    n_resamples, seed
        Bootstrap configuration (deterministic).

    Returns
    -------
    out : pd.DataFrame
        Columns: ``game, adversary, method, n_seeds, mean_diff, ci_lo,
        ci_hi, baseline, metric, ci_excludes_zero``.
    """
    keys = tuple(group_keys)
    if "method" not in keys:
        raise ValueError("group_keys must include 'method'")
    cell_keys = tuple(k for k in keys if k != "method")

    if df.empty:
        return pd.DataFrame(
            columns=[
                *cell_keys, "method", "n_seeds", "mean_diff", "ci_lo", "ci_hi",
                "baseline", "metric", "ci_excludes_zero",
            ]
        )

    rows: List[Dict[str, Any]] = []
    # Group by the non-method portion of the cell key.
    for cell_vals, cell_df in df.groupby(list(cell_keys), dropna=False):
        if not isinstance(cell_vals, tuple):
            cell_vals = (cell_vals,)
        baseline_df = cell_df[cell_df["method"] == baseline].set_index("seed")
        if baseline_df.empty:
            continue
        for method, m_df in cell_df.groupby("method"):
            if method == baseline:
                continue
            m_df_idx = m_df.set_index("seed")
            common_seeds = sorted(
                set(baseline_df.index).intersection(m_df_idx.index)
            )
            if not common_seeds:
                continue
            base_vals = baseline_df.loc[common_seeds, metric].to_numpy(
                dtype=np.float64
            )
            method_vals = m_df_idx.loc[common_seeds, metric].to_numpy(
                dtype=np.float64
            )
            diffs = method_vals - base_vals  # shape (n_seeds,)
            mean_d, lo, hi = _paired_bootstrap_ci(
                diffs, n_resamples=n_resamples, seed=seed
            )
            ci_excludes = (
                bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))
            )
            row: Dict[str, Any] = {k: v for k, v in zip(cell_keys, cell_vals)}
            row.update(
                {
                    "method": method,
                    "n_seeds": int(len(common_seeds)),
                    "mean_diff": mean_d,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "baseline": baseline,
                    "metric": metric,
                    "ci_excludes_zero": ci_excludes,
                }
            )
            rows.append(row)
    if not rows:
        return pd.DataFrame(
            columns=[
                *cell_keys, "method", "n_seeds", "mean_diff", "ci_lo", "ci_hi",
                "baseline", "metric", "ci_excludes_zero",
            ]
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Event-aligned panel
# ---------------------------------------------------------------------------

# Mapping from spec §10 event kind to (column flag, column-source).
_EVENT_FLAG_COLUMNS = {
    "model_rejected": "model_rejected",
    "search_phase_start": "search_phase",  # rising-edge detection.
    "support_shift": "support_shift",
    "tremble": "tremble",  # only present in rules-of-the-road runs.
    "hidden_type_resample": "hidden_type_resample",  # soda game.
}

# Default per-event metrics to slice (spec §10 calls for 5 panels).
_EVENT_PANEL_METRICS: Tuple[str, ...] = (
    "return",
    "beta",
    "effective_discount",
    "alignment_rate",
    "opponent_policy_entropy",
)


def event_aligned_panel(
    transitions_dir: Path,
    event_kind: str,
    half_window: int = 50,
    *,
    metrics: Iterable[str] = _EVENT_PANEL_METRICS,
    episode_filename: str = "episodes_strategic.csv",
    fallback_episode_filename: str = "episodes.csv",
) -> Dict[str, np.ndarray]:
    """Build event-aligned panels for the spec §10 plots.

    Parameters
    ----------
    transitions_dir
        Directory containing per-run leaf folders. Each leaf must contain
        an ``episodes_strategic.csv`` (preferred) or ``episodes.csv``
        (fallback) with the required columns. Optional Parquet transition
        files are NOT required: per spec §10 the panels are episode-level.
    event_kind
        One of the keys in :data:`_EVENT_FLAG_COLUMNS`.
    half_window
        Half-window radius (default 50, OQ-3 binding).
    metrics
        Metric column names to extract. Each becomes one entry in the
        return dict, with shape ``(n_events_total, 2*half_window+1)``.
    episode_filename, fallback_episode_filename
        Filenames searched in each leaf.

    Returns
    -------
    panels : Dict[str, np.ndarray]
        Mapping ``metric_name -> stacked panel of shape
        (sum_of_events_per_run, 2*half_window+1)``. Boundary samples are
        ``np.nan``-padded by :func:`event_aligned_window`. An additional
        ``"_meta"`` key carries a DataFrame-shaped dict with per-event
        identity tags (run_id, game, adversary, method, seed, episode).
    """
    if event_kind not in _EVENT_FLAG_COLUMNS:
        raise ValueError(
            f"unknown event_kind={event_kind!r}; "
            f"valid: {sorted(_EVENT_FLAG_COLUMNS)}"
        )
    flag_col = _EVENT_FLAG_COLUMNS[event_kind]
    metric_list = list(metrics)
    width = 2 * int(half_window) + 1
    root = Path(transitions_dir)
    if not root.exists():
        raise FileNotFoundError(f"transitions_dir not found: {root}")

    # Stacks of per-metric panels and per-event metadata.
    stacks: Dict[str, List[np.ndarray]] = {m: [] for m in metric_list}
    meta_runs: List[str] = []
    meta_games: List[str] = []
    meta_advs: List[str] = []
    meta_methods: List[str] = []
    meta_seeds: List[int] = []
    meta_episodes: List[int] = []

    leaves = _discover_episode_files(
        root,
        primary=episode_filename,
        fallback=fallback_episode_filename,
    )
    for ep_path in leaves:
        df = _read_episode_csv(ep_path)
        if df is None or df.empty:
            continue
        events = _detect_events(df, flag_col=flag_col, event_kind=event_kind)
        if events.size == 0:
            continue

        # Identity tags from the first row (constant per run).
        run_id = str(df["run_id"].iloc[0]) if "run_id" in df.columns else ep_path.parent.name
        game = str(df["game"].iloc[0]) if "game" in df.columns else "unknown"
        adv = str(df["adversary"].iloc[0]) if "adversary" in df.columns else "unknown"
        method = str(df["method"].iloc[0]) if "method" in df.columns else "unknown"
        seed = int(df["seed"].iloc[0]) if "seed" in df.columns else -1

        for m in metric_list:
            if m in df.columns:
                vals = df[m].to_numpy(dtype=np.float64)
            else:
                # Spec §10 expects best-effort: missing column → NaN row.
                vals = np.full(len(df), np.nan, dtype=np.float64)
            panel = event_aligned_window(vals, events, half_window=half_window)
            stacks[m].append(panel)

        for e in events:
            meta_runs.append(run_id)
            meta_games.append(game)
            meta_advs.append(adv)
            meta_methods.append(method)
            meta_seeds.append(seed)
            meta_episodes.append(int(e))

    # Concatenate.
    panels: Dict[str, np.ndarray] = {}
    for m in metric_list:
        if stacks[m]:
            panels[m] = np.vstack(stacks[m])
        else:
            panels[m] = np.zeros((0, width), dtype=np.float64)
    panels["_meta"] = {
        "run_id": np.asarray(meta_runs, dtype=object),
        "game": np.asarray(meta_games, dtype=object),
        "adversary": np.asarray(meta_advs, dtype=object),
        "method": np.asarray(meta_methods, dtype=object),
        "seed": np.asarray(meta_seeds, dtype=np.int64),
        "episode": np.asarray(meta_episodes, dtype=np.int64),
    }
    return panels


def _discover_episode_files(
    root: Path,
    *,
    primary: str,
    fallback: str,
) -> List[Path]:
    """Return one episode file per leaf directory, preferring ``primary``."""
    out: List[Path] = []
    seen_dirs: set = set()
    for path in sorted(root.rglob(primary)):
        out.append(path)
        seen_dirs.add(path.parent.resolve())
    for path in sorted(root.rglob(fallback)):
        if path.parent.resolve() not in seen_dirs:
            out.append(path)
            seen_dirs.add(path.parent.resolve())
    return out


def _read_episode_csv(path: Path) -> Optional[pd.DataFrame]:
    """Read an episodes CSV, tolerant of a leading ``# schema_version`` line."""
    try:
        # Skip lines that start with '#'.
        return pd.read_csv(path, comment="#")
    except (OSError, pd.errors.ParserError):
        return None


def _detect_events(
    df: pd.DataFrame,
    *,
    flag_col: str,
    event_kind: str,
) -> np.ndarray:
    """Return episode-index array of events.

    For ``"search_phase_start"``: rising-edge detection on the
    ``search_phase`` boolean. For all other kinds: positive entries on
    the named flag column.
    """
    if flag_col not in df.columns:
        return np.zeros(0, dtype=np.int64)
    flag = df[flag_col]
    # Coerce booleans/strings/ints to a clean bool array.
    if flag.dtype == bool:
        arr = flag.to_numpy(dtype=bool)
    else:
        arr = (
            flag.replace({"True": True, "False": False, "1": True, "0": False})
            .fillna(False)
            .astype(bool)
            .to_numpy()
        )
    if event_kind == "search_phase_start":
        # Rising-edge: True at i AND not True at i-1.
        if arr.size == 0:
            return np.zeros(0, dtype=np.int64)
        prev = np.concatenate([[False], arr[:-1]])
        events = arr & ~prev
    else:
        events = arr
    idx = np.flatnonzero(events).astype(np.int64)
    return idx


# ---------------------------------------------------------------------------
# Promotion gate
# ---------------------------------------------------------------------------

def promotion_gate(
    df: pd.DataFrame,
    criterion: str = "auc_diff_ci_excludes_zero",
    *,
    promotion_keys: Iterable[str] = DEFAULT_PROMOTION_KEYS,
    method: str = "adaptive_beta_clipped",
    baseline: str = "vanilla",
    metric: str = "auc_return",
    diverged_method: str = "adaptive_beta_clipped",
) -> Dict[Tuple[Any, ...], bool]:
    """Stage B2-Dev promotion gate per spec §11.1.

    Promotion criteria (per spec):
    1. adaptive-β improves AUC over vanilla on paired seeds in at least
       one ``(game, adversary)`` setting (criterion ``auc_diff_ci_excludes_zero``
       requires the paired-bootstrap 95% CI to exclude zero on the upside;
       ``auc_diff_positive_mean`` only requires the mean diff to exceed zero).
    2. No clipped adaptive-β divergence in that cell.

    Parameters
    ----------
    df
        Run-summary DataFrame.
    criterion
        ``"auc_diff_ci_excludes_zero"`` (strict) or
        ``"auc_diff_positive_mean"`` (lenient mean-only).
    promotion_keys
        Column tuple defining the promotion cells (default
        ``(game, adversary)``).
    method
        The candidate method whose AUC is compared to ``baseline``.
    baseline
        The baseline method.
    metric
        Comparison metric (default ``"auc_return"``).
    diverged_method
        Method whose ``diverged_count`` must be zero for promotion. By
        default this is the same candidate method (per spec §11.1
        criterion 2: clipped adaptive-β must not diverge).

    Returns
    -------
    verdict : Dict[Tuple, bool]
        Mapping ``cell_tuple -> promoted``. The cell tuple is a tuple of
        values in ``promotion_keys`` order.
    """
    valid_criteria = {"auc_diff_ci_excludes_zero", "auc_diff_positive_mean"}
    if criterion not in valid_criteria:
        raise ValueError(
            f"unknown criterion={criterion!r}; valid: {sorted(valid_criteria)}"
        )
    keys = tuple(promotion_keys)
    if df.empty:
        return {}

    diffs = paired_diffs(
        df,
        baseline=baseline,
        metric=metric,
        group_keys=(*keys, "method"),
    )
    diffs = diffs[diffs["method"] == method]

    verdict: Dict[Tuple[Any, ...], bool] = {}
    for cell_vals, cell_df in df.groupby(list(keys), dropna=False):
        if not isinstance(cell_vals, tuple):
            cell_vals = (cell_vals,)
        # Check criterion 1: paired-diff signal.
        match = diffs
        for k, v in zip(keys, cell_vals):
            match = match[match[k] == v]
        if match.empty:
            verdict[cell_vals] = False
            continue
        row = match.iloc[0]
        if criterion == "auc_diff_ci_excludes_zero":
            cond_signal = bool(row["ci_excludes_zero"]) and float(row["mean_diff"]) > 0
        else:  # auc_diff_positive_mean
            cond_signal = float(row["mean_diff"]) > 0

        # Check criterion 2: candidate method did not diverge in this cell.
        method_runs = cell_df[cell_df["method"] == diverged_method]
        if method_runs.empty:
            cond_no_div = True  # nothing to check
        else:
            cond_no_div = (
                method_runs["diverged_count"].fillna(0).astype(int).sum() == 0
            )

        verdict[cell_vals] = bool(cond_signal and cond_no_div)
    return verdict
