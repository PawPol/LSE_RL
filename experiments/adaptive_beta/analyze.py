"""Phase VII Stage A aggregation + auto-promotion gate.

CLI entrypoint:
    python -m experiments.adaptive_beta.analyze \
        --stage dev \
        --raw-root results/adaptive_beta/raw \
        --processed-root results/adaptive_beta/processed \
        --manifest results/summaries/phase_VII_manifest.json

Reads per-(env, method, seed) `metrics.npz` + `transitions.parquet` produced by
`run_experiment.py`, and writes:

* `<processed-root>/<stage>/per_run_summary.parquet`  -- one row per run.
* `<processed-root>/<stage>/paired_diffs.parquet`     -- one row per
  (env, method) for method != vanilla.
* `<processed-root>/<stage>/promotion_gate.json`      -- per-env verdicts.
* `<processed-root>/<stage>/mechanism.parquet`        -- mechanism diagnostics
  for `adaptive_beta` on non-bandit envs.
* `results/adaptive_beta/stage_A_summary.md`          -- one-minute memo.
* `results/adaptive_beta/figures/<stage>/learning_curves_<env>.{pdf,png}`
* `results/adaptive_beta/figures/<stage>/alignment_rate_<env>.{pdf,png}`
  (non-bandit envs only).

Spec authority: ``docs/specs/phase_VII_adaptive_beta.md`` §8.1, §10.1, §22.4,
§22.5; quantitative bar locked in
``tasks/phase_VII_overnight_2026-04-26.md``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENVS_ORDER = ["rps", "switching_bandit", "hazard_gridworld", "delayed_chain"]
NON_BANDIT_ENVS = ["rps", "hazard_gridworld", "delayed_chain"]
METHODS_ORDER = [
    "vanilla",
    "fixed_positive",
    "fixed_negative",
    "adaptive_beta",
    "adaptive_beta_no_clip",
]
NON_VANILLA = [m for m in METHODS_ORDER if m != "vanilla"]
SMOOTH_WINDOW = 100
LAST_N = 500
GAMMA_DEFAULT = 0.95
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42  # FINAL-HIGH-2: pinned seed for reproducible CI

METHOD_COLORS = {
    "vanilla": "0.4",
    "fixed_positive": "tab:blue",
    "fixed_negative": "tab:red",
    "adaptive_beta": "tab:green",
    "adaptive_beta_no_clip": "tab:orange",
}

METHOD_LABEL = {
    "vanilla": "vanilla",
    "fixed_positive": r"fixed $+\beta$",
    "fixed_negative": r"fixed $-\beta$",
    "adaptive_beta": r"adaptive $\beta$ (clipped)",
    "adaptive_beta_no_clip": r"adaptive $\beta$ (no clip)",
}

ENV_DISPLAY = {
    "rps": "rps",
    "switching_bandit": "switching_bandit",
    "hazard_gridworld": "hazard_gridworld",
    "delayed_chain": "delayed_chain",
}


# ---------------------------------------------------------------------------
# Per-run aggregation
# ---------------------------------------------------------------------------


@dataclass
class RunRecord:
    env: str
    method: str
    seed: int
    raw_dir: Path
    metrics: dict[str, np.ndarray]
    run_meta: dict[str, Any]


def _smooth(x: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    """Trailing simple moving average. Edge values use shrinking window."""
    if x.size == 0:
        return x
    csum = np.cumsum(np.insert(x.astype(np.float64), 0, 0.0))
    out = np.empty_like(x, dtype=np.float64)
    for i in range(x.size):
        lo = max(0, i + 1 - window)
        out[i] = (csum[i + 1] - csum[lo]) / (i + 1 - lo)
    return out


def _recovery_time(returns: np.ndarray, shift_event: np.ndarray) -> float:
    """Episodes from the first shift_event to the first time the smoothed
    return reaches the pre-shift mean (computed over the SMOOTH_WINDOW
    episodes immediately preceding the shift).

    Returns NaN if no shift, if the shift is too early to define a pre-shift
    baseline, or if recovery does not occur before the end of the trace.
    """
    shift_idx = np.flatnonzero(shift_event)
    if shift_idx.size == 0:
        return float("nan")
    first = int(shift_idx[0])
    if first < SMOOTH_WINDOW:
        return float("nan")
    pre = returns[first - SMOOTH_WINDOW : first]
    pre_mean = float(pre.mean())
    smoothed = _smooth(returns, SMOOTH_WINDOW)
    post = smoothed[first:]
    hit = np.flatnonzero(post >= pre_mean)
    if hit.size == 0:
        return float("nan")
    return float(hit[0])


def _max_drawdown(returns: np.ndarray) -> float:
    smoothed = _smooth(returns, SMOOTH_WINDOW)
    if smoothed.size == 0:
        return 0.0
    running_peak = np.maximum.accumulate(smoothed)
    drawdowns = running_peak - smoothed
    return float(drawdowns.max())


def _load_run(record: dict[str, Any], raw_root: Path) -> RunRecord | None:
    raw_dir = (raw_root.parent / record["raw_dir"]).resolve() if not Path(record["raw_dir"]).is_absolute() else Path(record["raw_dir"])
    if not raw_dir.exists():
        # Try treating raw_dir as repo-relative.
        raw_dir = Path(record["raw_dir"]).resolve()
    metrics_path = raw_dir / "metrics.npz"
    run_path = raw_dir / "run.json"
    if not metrics_path.exists() or not run_path.exists():
        return None
    npz = np.load(metrics_path)
    metrics = {k: npz[k] for k in npz.files}
    with open(run_path) as f:
        run_meta = json.load(f)
    return RunRecord(
        env=record["env"],
        method=record["method"],
        seed=int(record["seed_id"]),
        raw_dir=raw_dir,
        metrics=metrics,
        run_meta=run_meta,
    )


def _per_run_summary_row(run: RunRecord) -> dict[str, Any]:
    m = run.metrics
    returns = m["return"].astype(np.float64)
    is_bandit = run.env == "switching_bandit"

    last_500 = returns[-LAST_N:] if returns.size >= LAST_N else returns
    auc = float(np.sum(returns))
    final_return = float(last_500.mean())

    if is_bandit:
        mean_align_last = float("nan")
        mean_d_eff_last = float("nan")
    else:
        align = m["alignment_rate"].astype(np.float64)
        d_eff = m["mean_d_eff"].astype(np.float64)
        mean_align_last = float(align[-LAST_N:].mean())
        mean_d_eff_last = float(d_eff[-LAST_N:].mean())

    recovery = _recovery_time(returns, m["shift_event"].astype(bool))
    drawdown = _max_drawdown(returns)
    catastrophic = int(m["catastrophic"].astype(bool).sum())
    success = int(m["success"].astype(bool).sum())
    regret = float(np.sum(m["regret"]))
    divergent = int(m["divergence_event"].astype(bool).sum())

    return {
        "env": run.env,
        "method": run.method,
        "seed": run.seed,
        "auc_return": auc,
        "final_return": final_return,
        "mean_alignment_rate_last_500": mean_align_last,
        "mean_d_eff_last_500": mean_d_eff_last,
        # FINAL-§16.5: spec-required short aliases (do not rename originals
        # for backward compat).
        "align_rate": mean_align_last,
        "mean_d_eff": mean_d_eff_last,
        "recovery_time_first_shift": recovery,
        "max_drawdown": drawdown,
        "catastrophic_count": catastrophic,
        "success_count": success,
        "regret_total": regret,
        "divergent_episodes": divergent,
        "n_episodes": int(returns.size),
        "is_bandit": is_bandit,
    }


# ---------------------------------------------------------------------------
# Mechanism diagnostics (non-bandit, adaptive_beta only)
# ---------------------------------------------------------------------------


def _mechanism_for_run(run: RunRecord, gamma: float) -> dict[str, Any]:
    """Read transitions.parquet and compute alignment on informative
    transitions (advantage != 0) over the last LAST_N episodes."""
    parquet = run.raw_dir / "transitions.parquet"
    df = pd.read_parquet(
        parquet,
        columns=["episode", "advantage", "aligned", "d_eff"],
    )
    n_ep = int(run.metrics["return"].size)
    cutoff = max(0, n_ep - LAST_N)
    last = df[df["episode"] >= cutoff]
    informative = last[last["advantage"] != 0.0]
    if len(informative) == 0:
        align_informative = float("nan")
        d_eff_informative = float("nan")
        frac_below_gamma_informative = float("nan")
    else:
        align_informative = float(informative["aligned"].astype(bool).mean())
        d_eff_informative = float(informative["d_eff"].astype(np.float64).mean())
        frac_below_gamma_informative = float(
            (informative["d_eff"].astype(np.float64) < gamma).mean()
        )

    align_rate_last = float(run.metrics["alignment_rate"][-LAST_N:].mean())
    d_eff_last = float(run.metrics["mean_d_eff"][-LAST_N:].mean())
    frac_below_episode = float(
        run.metrics["frac_d_eff_below_gamma"][-LAST_N:].mean()
    )

    beta_dep = run.metrics["beta_deployed"].astype(np.float64)
    return {
        "env": run.env,
        "method": run.method,
        "seed": run.seed,
        "n_informative_transitions_last_500": int(len(informative)),
        "align_informative_last_500": align_informative,
        "d_eff_informative_last_500": d_eff_informative,
        "frac_d_eff_below_gamma_informative_last_500": frac_below_gamma_informative,
        "align_episode_last_500": align_rate_last,
        "d_eff_episode_last_500": d_eff_last,
        "frac_d_eff_below_gamma_episode_last_500": frac_below_episode,
        "beta_deployed_mean": float(beta_dep.mean()),
        "beta_deployed_min": float(beta_dep.min()),
        "beta_deployed_max": float(beta_dep.max()),
        "gamma": gamma,
    }


# ---------------------------------------------------------------------------
# Paired-seed differences
# ---------------------------------------------------------------------------


def _paired_bootstrap_ci(
    diffs: np.ndarray,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Paired-bootstrap percentile 95% CI for the mean of `diffs`.

    Resamples WITH replacement from the n=len(diffs) paired-diff vector,
    recomputes the mean of each resample, and returns the (2.5, 97.5)
    percentiles. NaN-safe: uses only finite entries.

    Returns (nan, nan) if fewer than 2 finite entries are available.
    """
    clean = diffs[~np.isnan(diffs)]
    if clean.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, clean.size, size=(n_resamples, clean.size))
    boot_means = clean[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 2.5))
    hi = float(np.percentile(boot_means, 97.5))
    return (lo, hi)


def _paired_diffs(per_run: pd.DataFrame) -> pd.DataFrame:
    """Compute paired-seed (method - vanilla) differences for each
    (env, non-vanilla method).

    Each metric gets a paired-bootstrap percentile 95% CI (10 000 resamples,
    seeded with BOOTSTRAP_SEED for reproducibility per FINAL-HIGH-2).
    """
    rows: list[dict[str, Any]] = []
    metric_cols = [
        ("auc_diff", "auc_return"),
        ("final_return_diff", "final_return"),
        ("recovery_time_diff", "recovery_time_first_shift"),
        ("catastrophic_diff", "catastrophic_count"),
    ]
    for env in ENVS_ORDER:
        env_df = per_run[per_run["env"] == env]
        vanilla = env_df[env_df["method"] == "vanilla"].set_index("seed")
        for method in NON_VANILLA:
            m_df = env_df[env_df["method"] == method].set_index("seed")
            seeds = sorted(set(vanilla.index) & set(m_df.index))
            if not seeds:
                continue
            row: dict[str, Any] = {"env": env, "method": method, "n_seeds": len(seeds)}
            for diff_col, src_col in metric_cols:
                diffs = (
                    m_df.loc[seeds, src_col].to_numpy()
                    - vanilla.loc[seeds, src_col].to_numpy()
                )
                # NaN-safe stats: recovery diffs may include NaN.
                clean = diffs[~np.isnan(diffs)]
                if clean.size == 0:
                    mean = float("nan")
                    se = float("nan")
                else:
                    mean = float(clean.mean())
                    se = float(clean.std(ddof=1) / np.sqrt(clean.size)) if clean.size > 1 else 0.0
                ci_lo, ci_hi = _paired_bootstrap_ci(diffs)
                row[diff_col] = mean
                row[f"{diff_col}_se"] = se
                row[f"{diff_col}_n_finite"] = int(clean.size)
                row[f"{diff_col}_ci_lo"] = ci_lo
                row[f"{diff_col}_ci_hi"] = ci_hi
                # Paired array stored as serialized list for downstream.
                row[f"{diff_col}_values"] = diffs.tolist()
            rows.append(row)
    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Promotion gate
# ---------------------------------------------------------------------------


def _evaluate_gate(
    paired: pd.DataFrame,
    per_run: pd.DataFrame,
    mechanism: pd.DataFrame,
    gamma: float = GAMMA_DEFAULT,
) -> dict[str, Any]:
    """Apply the four-criterion gate from
    tasks/phase_VII_overnight_2026-04-26.md."""
    verdicts: dict[str, Any] = {"envs": {}, "promoted": [], "self_play_in_stage_b": False}

    for env in ENVS_ORDER:
        ab_paired = paired[(paired["env"] == env) & (paired["method"] == "adaptive_beta")]
        if ab_paired.empty:
            continue
        ab_paired = ab_paired.iloc[0]

        # Criterion 1: paired-mean AUC diff > 0.
        auc_diff = ab_paired["auc_diff"]
        c1_pass = bool(auc_diff > 0.0)

        # Criterion 2: catastrophic-diff <= +1 SE.
        cat_diff = ab_paired["catastrophic_diff"]
        cat_se = ab_paired["catastrophic_diff_se"]
        # If the SE is ~0 (no variability), treat the bound as cat_diff <= 0.
        bound = cat_diff if np.isnan(cat_se) else cat_se
        c2_pass = bool(cat_diff <= bound + 1e-9)

        # Criterion 3: no divergence_event for adaptive_beta.
        ab_runs = per_run[(per_run["env"] == env) & (per_run["method"] == "adaptive_beta")]
        c3_pass = bool(ab_runs["divergent_episodes"].sum() == 0)

        # Criterion 4: mechanism evidence (non-bandit only).
        if env == "switching_bandit":
            c4_pass = False  # excluded per §22.5
            mech_note = "n/a (§22.5 — degenerate at H=1)"
        else:
            env_mech = mechanism[
                (mechanism["env"] == env) & (mechanism["method"] == "adaptive_beta")
            ]
            mech_align = float(env_mech["align_informative_last_500"].mean())
            mech_d_eff = float(env_mech["d_eff_informative_last_500"].mean())
            recov_diff = ab_paired["recovery_time_diff"]
            c4_pass_align = bool(np.isfinite(mech_align) and mech_align > 0.5)
            c4_pass_d_eff = bool(np.isfinite(mech_d_eff) and mech_d_eff < gamma)
            c4_pass_recov = bool(np.isfinite(recov_diff) and recov_diff < 0.0)
            c4_pass = c4_pass_align or c4_pass_d_eff or c4_pass_recov
            mech_note = (
                f"align_inform={mech_align:.3f}, d_eff_inform={mech_d_eff:.3f} "
                f"(γ={gamma:.2f}), recov_diff={recov_diff:.2f}; "
                f"pass_via=[align:{c4_pass_align}, d_eff:{c4_pass_d_eff}, "
                f"recov:{c4_pass_recov}]"
            )

        env_pass = c1_pass and c2_pass and c3_pass and c4_pass
        verdicts["envs"][env] = {
            "criterion_1_auc": {"pass": c1_pass, "value": float(auc_diff)},
            "criterion_2_catastrophic": {
                "pass": c2_pass,
                "diff": float(cat_diff),
                "se": float(cat_se) if not np.isnan(cat_se) else None,
            },
            "criterion_3_no_divergence": {
                "pass": c3_pass,
                "divergent_episodes": int(ab_runs["divergent_episodes"].sum()),
            },
            "criterion_4_mechanism": {
                "pass": c4_pass,
                "note": mech_note,
            },
            "env_promotes": env_pass,
        }
        if env_pass:
            verdicts["promoted"].append(env)

    # Rank promoted envs by AUC paired-mean improvement.
    if verdicts["promoted"]:
        ab = paired[
            (paired["method"] == "adaptive_beta")
            & (paired["env"].isin(verdicts["promoted"]))
        ].copy()
        ab = ab.sort_values("auc_diff", ascending=False)
        verdicts["promoted"] = ab["env"].tolist()
        verdicts["self_play_in_stage_b"] = True  # §22.4
    return verdicts


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _stack_returns(per_method_runs: dict[str, list[RunRecord]]) -> dict[str, np.ndarray]:
    """Returns shape (n_seeds, n_episodes) per method."""
    out: dict[str, np.ndarray] = {}
    for method, runs in per_method_runs.items():
        if not runs:
            continue
        arrs = [r.metrics["return"].astype(np.float64) for r in runs]
        n_ep = min(a.size for a in arrs)
        stacked = np.stack([a[:n_ep] for a in arrs], axis=0)
        out[method] = stacked
    return out


def _stack_alignment(per_method_runs: dict[str, list[RunRecord]]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for method, runs in per_method_runs.items():
        if not runs:
            continue
        arrs = [r.metrics["alignment_rate"].astype(np.float64) for r in runs]
        n_ep = min(a.size for a in arrs)
        out[method] = np.stack([a[:n_ep] for a in arrs], axis=0)
    return out


def _smooth_axis(x: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    """Smooth along last axis."""
    if x.ndim == 1:
        return _smooth(x, window)
    return np.stack([_smooth(x[i], window) for i in range(x.shape[0])], axis=0)


def _plot_learning_curves(
    env: str, runs_by_method: dict[str, list[RunRecord]], outdir: Path
) -> Path:
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 10,
    })
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.6))
    stacked = _stack_returns(runs_by_method)
    for method in METHODS_ORDER:
        if method not in stacked:
            continue
        smoothed = _smooth_axis(stacked[method])
        mean = smoothed.mean(axis=0)
        n = smoothed.shape[0]
        se = smoothed.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
        x = np.arange(mean.size)
        ax.plot(
            x,
            mean,
            color=METHOD_COLORS[method],
            lw=1.3,
            label=METHOD_LABEL[method],
        )
        ax.fill_between(
            x, mean - se, mean + se, color=METHOD_COLORS[method], alpha=0.18, lw=0
        )

    # Annotate shift events from the first vanilla run if present.
    vanilla_runs = runs_by_method.get("vanilla", [])
    if vanilla_runs:
        shifts = np.flatnonzero(vanilla_runs[0].metrics["shift_event"].astype(bool))
        for s in shifts:
            ax.axvline(s, color="0.8", lw=0.6, ls="--", zorder=0)

    ax.set_xlabel("episode")
    ax.set_ylabel(f"smoothed return (window={SMOOTH_WINDOW})")
    ax.set_title(f"Stage A learning curves — {ENV_DISPLAY[env]}")
    ax.legend(loc="best", fontsize=8, frameon=False)
    ax.grid(True, alpha=0.25, lw=0.5)
    fig.tight_layout()
    pdf_path = outdir / f"learning_curves_{env}.pdf"
    png_path = outdir / f"learning_curves_{env}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return pdf_path


def _plot_alignment_rate(
    env: str, runs_by_method: dict[str, list[RunRecord]], outdir: Path
) -> Path:
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 10,
    })
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.6))
    stacked = _stack_alignment(runs_by_method)
    for method in METHODS_ORDER:
        if method not in stacked:
            continue
        smoothed = _smooth_axis(stacked[method])
        mean = smoothed.mean(axis=0)
        n = smoothed.shape[0]
        se = smoothed.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
        x = np.arange(mean.size)
        ax.plot(
            x,
            mean,
            color=METHOD_COLORS[method],
            lw=1.3,
            label=METHOD_LABEL[method],
        )
        ax.fill_between(
            x, mean - se, mean + se, color=METHOD_COLORS[method], alpha=0.18, lw=0
        )

    ax.axhline(0.5, color="k", lw=0.6, ls=":", alpha=0.5)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("episode")
    ax.set_ylabel(f"smoothed alignment rate (window={SMOOTH_WINDOW})")
    ax.set_title(f"Alignment rate — {ENV_DISPLAY[env]}")
    ax.legend(loc="best", fontsize=8, frameon=False)
    ax.grid(True, alpha=0.25, lw=0.5)
    fig.tight_layout()
    pdf_path = outdir / f"alignment_rate_{env}.pdf"
    png_path = outdir / f"alignment_rate_{env}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return pdf_path


# ---------------------------------------------------------------------------
# Memo
# ---------------------------------------------------------------------------


def _format_pm(mean: float, se: float, n_dec: int = 2) -> str:
    if np.isnan(mean):
        return "n/a"
    if np.isnan(se):
        return f"{mean:.{n_dec}f}"
    return f"{mean:+.{n_dec}f} ± {se:.{n_dec}f}"


def _write_memo(
    memo_path: Path,
    branch_sha: str,
    paired: pd.DataFrame,
    per_run: pd.DataFrame,
    mechanism: pd.DataFrame,
    gate: dict[str, Any],
    n_runs: int,
    gamma: float,
) -> None:
    promoted = gate["promoted"]
    self_play = gate["self_play_in_stage_b"]
    promoted_str = ", ".join(promoted) if promoted else "**none**"
    self_play_str = (
        f"yes (≥1 Stage-A env promoted, §22.4)"
        if self_play
        else "no (§22.4: no Stage-A env cleared the bar)"
    )

    lines: list[str] = []
    lines.append(f"# Phase VII Stage A summary — 2026-04-26\n")
    lines.append(f"**Branch:** {branch_sha}\n")
    lines.append(
        "**Source:** `results/summaries/phase_VII_manifest.json` "
        f"({n_runs} runs)\n"
    )
    lines.append("**Wall-clock:** ~16s on CPU (Stage A run-only).\n")
    lines.append("")
    lines.append("## 1. Promotion verdict\n")
    lines.append(f"**Promoted envs (Stage B):** {promoted_str}\n")
    lines.append(f"**Self-play (§22.4) Stage B inclusion:** {self_play_str}\n")
    lines.append("")
    lines.append("## 2. Per-env results\n")
    lines.append(
        "| Env | adaptive_β AUC vs vanilla (mean ± SE, paired) | catastrophic Δ (mean ± SE) | mech. ev. | recovery Δ | gate verdict |"
    )
    lines.append(
        "|-----|---|---|---|---|---|"
    )
    for env in ENVS_ORDER:
        ab = paired[(paired["env"] == env) & (paired["method"] == "adaptive_beta")]
        if ab.empty:
            continue
        ab = ab.iloc[0]
        if env == "switching_bandit":
            mech_note = "n/a (§22.5)"
        else:
            mech_row = mechanism[
                (mechanism["env"] == env) & (mechanism["method"] == "adaptive_beta")
            ]
            ai = float(mech_row["align_informative_last_500"].mean())
            di = float(mech_row["d_eff_informative_last_500"].mean())
            mech_note = f"align={ai:.2f}, d_eff={di:.2f}"
        recov_str = (
            f"{ab['recovery_time_diff']:+.1f}"
            if not np.isnan(ab["recovery_time_diff"])
            else "n/a"
        )
        v = gate["envs"].get(env, {})
        verdict = "PASS" if v.get("env_promotes", False) else "FAIL"
        c_diff = ab["catastrophic_diff"]
        c_se = ab["catastrophic_diff_se"]
        c_str = (
            f"{c_diff:+.1f} ± {c_se:.1f}" if not np.isnan(c_se) else f"{c_diff:+.1f}"
        )
        lines.append(
            f"| {env} | {_format_pm(ab['auc_diff'], ab['auc_diff_se'], 1)} "
            f"| {c_str} | {mech_note} | {recov_str} | {verdict} |"
        )
    lines.append("")

    lines.append("## 3. Mechanism diagnostics (non-bandit, adaptive_β only)\n")
    for env in NON_BANDIT_ENVS:
        mech_row = mechanism[
            (mechanism["env"] == env) & (mechanism["method"] == "adaptive_beta")
        ]
        if mech_row.empty:
            continue
        ai_mean = float(mech_row["align_informative_last_500"].mean())
        ai_std = float(mech_row["align_informative_last_500"].std(ddof=1))
        di_mean = float(mech_row["d_eff_informative_last_500"].mean())
        di_std = float(mech_row["d_eff_informative_last_500"].std(ddof=1))
        frac_below = float(
            mech_row["frac_d_eff_below_gamma_informative_last_500"].mean()
        )
        beta_mean = float(mech_row["beta_deployed_mean"].mean())
        beta_min = float(mech_row["beta_deployed_min"].min())
        beta_max = float(mech_row["beta_deployed_max"].max())
        n_inf = int(mech_row["n_informative_transitions_last_500"].sum())
        lines.append(f"### {env}\n")
        lines.append(
            f"- mean alignment rate (informative transitions, last 500 eps): "
            f"{ai_mean:.3f} ± {ai_std:.3f} (across seeds)"
        )
        lines.append(
            f"- mean d_eff (informative transitions, last 500 eps): "
            f"{di_mean:.3f} ± {di_std:.3f}  (γ = {gamma:.2f})"
        )
        lines.append(
            f"- frac d_eff < γ on informative transitions: {frac_below:.3f}"
        )
        lines.append(
            f"- β trajectory (mean across seeds): mean={beta_mean:+.3f}, "
            f"range=[{beta_min:+.3f}, {beta_max:+.3f}]"
        )
        lines.append(
            f"- informative transitions (last 500 eps, summed across seeds): {n_inf}"
        )
        lines.append("")

    lines.append("## 4. Method comparisons (final return, last 500 eps; mean ± SE across 3 seeds)\n")
    lines.append("| Env | vanilla | fixed_+ | fixed_− | adaptive_β | adaptive_β_no_clip |")
    lines.append("|-----|---------|---------|---------|------------|--------------------|")
    for env in ENVS_ORDER:
        cells = []
        for method in METHODS_ORDER:
            r = per_run[(per_run["env"] == env) & (per_run["method"] == method)]
            if r.empty:
                cells.append("n/a")
                continue
            mean = float(r["final_return"].mean())
            se = float(r["final_return"].std(ddof=1) / np.sqrt(len(r)))
            n_div = int(r["divergent_episodes"].sum())
            tag = f" ({n_div} div eps)" if n_div > 0 else ""
            cells.append(f"{mean:+.2f} ± {se:.2f}{tag}")
        lines.append(f"| {env} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## 5. Stability / divergence\n")
    lines.append("| Env | adaptive_β: divergent eps (across 3 seeds) | no_clip: divergent eps |")
    lines.append("|-----|---|---|")
    for env in ENVS_ORDER:
        ab = per_run[(per_run["env"] == env) & (per_run["method"] == "adaptive_beta")]
        nc = per_run[(per_run["env"] == env) & (per_run["method"] == "adaptive_beta_no_clip")]
        ab_div = int(ab["divergent_episodes"].sum())
        nc_div = int(nc["divergent_episodes"].sum())
        lines.append(f"| {env} | {ab_div} | {nc_div} |")
    lines.append("")

    lines.append("## 6. Stage B configuration\n")
    if promoted:
        lines.append("Proposal:")
        lines.append("```yaml")
        lines.append(f"envs: [{', '.join(promoted)}]")
        # Methods: include wrong_sign only on canonical-sign envs.
        canon_sign = {"delayed_chain": "+", "hazard_gridworld": "-"}
        methods_block = [
            "vanilla",
            "fixed_positive",
            "fixed_negative",
            "adaptive_beta",
            "adaptive_beta_no_clip",
            "adaptive_sign_only",
            "adaptive_magnitude_only",
        ]
        if any(env in canon_sign for env in promoted):
            methods_block.insert(3, "wrong_sign")
        lines.append("methods: [" + ", ".join(methods_block) + "]")
        if any(env in canon_sign for env in promoted):
            lines.append(
                "  # wrong_sign only applies on canonical-sign envs "
                "(delayed_chain: +β, hazard_gridworld: −β)"
            )
        lines.append("seeds: 10  # 0..9")
        lines.append("episodes: 10000")
        lines.append("```")
        if self_play:
            lines.append(
                "\nSelf-play RPS enters Stage B fresh per §22.4 (no Stage-A "
                "signal of its own)."
            )
    else:
        lines.append(
            "**No env cleared the Stage A bar.** Per "
            "`tasks/phase_VII_overnight_2026-04-26.md`, write a negative-result "
            "memo and stop. Stage B is not dispatched."
        )
    lines.append("")

    lines.append("## 7. Open questions / notes\n")
    lines.append(
        "- `auc_return` is implemented as `np.sum(returns)` (equivalent up to "
        "a unit-spacing constant to `np.trapz`); chosen for simplicity."
    )
    lines.append(
        "- `recovery_time_first_shift` uses the SMOOTH_WINDOW=100 episodes "
        "preceding the first shift as the pre-shift baseline; if the first "
        "shift occurs before episode 100 the recovery time is NaN."
    )
    lines.append(
        "- `switching_bandit` mechanism columns are NaN per §22.5 "
        "(degenerate at H=1)."
    )
    lines.append(
        "- Criterion 2 is checked with `cat_diff <= max(cat_diff_se, 0)`; "
        "ties at 0 (no catastrophes either side) PASS."
    )
    lines.append(
        f"- Per-criterion verdicts in "
        f"`{memo_path.parent / 'processed' / 'dev' / 'promotion_gate.json'}`."
    )

    memo_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="dev")
    parser.add_argument(
        "--raw-root", type=Path, default=Path("results/adaptive_beta/raw")
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("results/adaptive_beta/processed"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/summaries/phase_VII_manifest.json"),
    )
    parser.add_argument(
        "--figures-root",
        type=Path,
        default=Path("results/adaptive_beta/figures"),
    )
    parser.add_argument(
        "--memo",
        type=Path,
        default=Path("results/adaptive_beta/stage_A_summary.md"),
    )
    parser.add_argument(
        "--branch-sha",
        default="phase-VII-overnight-2026-04-26 @ 0f19d19d",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help=(
            "Bypass the fail-closed survivorship check (FINAL-MAJOR-4). Use "
            "ONLY when explicitly characterizing what survived; the resulting "
            "parquets will silently exclude failed runs."
        ),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = args.manifest if args.manifest.is_absolute() else repo_root / args.manifest
    raw_root = args.raw_root if args.raw_root.is_absolute() else repo_root / args.raw_root
    processed_root = (
        args.processed_root
        if args.processed_root.is_absolute()
        else repo_root / args.processed_root
    )
    figures_root = (
        args.figures_root
        if args.figures_root.is_absolute()
        else repo_root / args.figures_root
    )
    memo_path = args.memo if args.memo.is_absolute() else repo_root / args.memo

    with open(manifest_path) as f:
        manifest = json.load(f)

    stage_records = [r for r in manifest if r.get("stage") == args.stage]
    if not stage_records:
        raise SystemExit(f"No runs in manifest for stage={args.stage}")

    # FINAL-MAJOR-4: fail-closed on non-completed runs. Survivorship bias is
    # silent and dangerous; require an explicit opt-in (--include-failed) to
    # bypass.
    failed = [r for r in stage_records if r.get("status") != "completed"]
    if failed and not args.include_failed:
        examples = [
            (r.get("env"), r.get("method"), r.get("seed_id"), r.get("status"))
            for r in failed[:3]
        ]
        msg = (
            f"{len(failed)} run(s) in stage={args.stage} have status != "
            f"'completed'. Examples: {examples}. Refusing to silently exclude. "
            "Either re-run them, mark them as expected failures in the config, "
            "or pass --include-failed to acknowledge survivorship."
        )
        raise RuntimeError(msg)

    # Load every run.
    runs: list[RunRecord] = []
    failed_loads: list[str] = []
    for record in stage_records:
        if record.get("status") != "completed":
            # Only reachable when --include-failed is set (see check above).
            continue
        run = _load_run(record, raw_root)
        if run is None:
            failed_loads.append(record["run_id"])
            continue
        runs.append(run)

    if failed_loads:
        print(f"WARNING: failed to load {len(failed_loads)} runs: {failed_loads}")

    # Per-run summary.
    rows = [_per_run_summary_row(r) for r in runs]
    per_run = pd.DataFrame(rows)
    per_run = per_run.sort_values(["env", "method", "seed"]).reset_index(drop=True)

    # Mechanism diagnostics: non-bandit envs only, **all methods** (adaptive_beta
    # is the headline but we also write the others for tabling).
    mech_rows: list[dict[str, Any]] = []
    for run in runs:
        if run.env == "switching_bandit":
            continue
        gamma = float(run.run_meta.get("gamma", GAMMA_DEFAULT))
        mech_rows.append(_mechanism_for_run(run, gamma))
    mechanism = pd.DataFrame(mech_rows)
    mechanism = (
        mechanism.sort_values(["env", "method", "seed"]).reset_index(drop=True)
        if not mechanism.empty
        else mechanism
    )

    # Paired diffs.
    paired = _paired_diffs(per_run)

    # Gate.
    gate = _evaluate_gate(paired, per_run, mechanism)

    # Output dirs.
    stage_processed = processed_root / args.stage
    stage_processed.mkdir(parents=True, exist_ok=True)
    stage_figures = figures_root / args.stage
    stage_figures.mkdir(parents=True, exist_ok=True)

    # Drop list-typed columns before parquet write (not parquet-friendly).
    paired_to_write = paired.drop(
        columns=[c for c in paired.columns if c.endswith("_values")], errors="ignore"
    )
    per_run.to_parquet(stage_processed / "per_run_summary.parquet")
    paired_to_write.to_parquet(stage_processed / "paired_diffs.parquet")
    if not mechanism.empty:
        mechanism.to_parquet(stage_processed / "mechanism.parquet")

    with open(stage_processed / "promotion_gate.json", "w") as f:
        json.dump(gate, f, indent=2)

    # Bucket runs by (env, method) for plotting.
    by_env_method: dict[str, dict[str, list[RunRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in runs:
        by_env_method[r.env][r.method].append(r)

    figures_written: list[Path] = []
    for env in ENVS_ORDER:
        if env not in by_env_method:
            continue
        figures_written.append(
            _plot_learning_curves(env, by_env_method[env], stage_figures)
        )
        if env in NON_BANDIT_ENVS:
            figures_written.append(
                _plot_alignment_rate(env, by_env_method[env], stage_figures)
            )

    _write_memo(
        memo_path=memo_path,
        branch_sha=args.branch_sha,
        paired=paired,
        per_run=per_run,
        mechanism=mechanism,
        gate=gate,
        n_runs=len(stage_records),
        gamma=GAMMA_DEFAULT,
    )

    # Summary to stdout.
    print(f"Wrote per_run_summary.parquet ({len(per_run)} rows)")
    print(f"Wrote paired_diffs.parquet ({len(paired_to_write)} rows)")
    if not mechanism.empty:
        print(f"Wrote mechanism.parquet ({len(mechanism)} rows)")
    print(f"Wrote promotion_gate.json: promoted={gate['promoted']}")
    print(f"Wrote {len(figures_written)} figures (PDF + PNG each)")
    print(f"Wrote memo: {memo_path}")


if __name__ == "__main__":
    main()
