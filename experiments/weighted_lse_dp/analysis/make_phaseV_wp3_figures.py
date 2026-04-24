#!/usr/bin/env python
"""Phase V WP3 figures -- propagation, stagewise heatmap, greedy recovery.

Invoked in-process by ``run_phase_V_planning.run_planning`` when the lead
Family A task's sweep tensors and ``d_ref`` are already in memory. Writes
three PDFs to ``figures/main/`` on the lead task (``argmax |value_gap_norm|``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Palette matches Phase IV-B conventions (analysis/make_phase4B_figures.py).
_CLR: dict[str, str] = {"classical": "#2166ac", "safe": "#d6604d"}
_MARKERS: dict[str, str] = {"classical": "o", "safe": "s"}


def emit_wp3_figures(
    *,
    lead: dict[str, Any],
    metrics_df: pd.DataFrame,
    scalars_df: pd.DataFrame,
    fig_dir: Path,
) -> None:
    """Emit the three WP3 main-paper figures on a single lead task.

    ``lead`` is a dict with keys ``task_id, horizon, k_values, d_ref, sweep``
    (sweep holds ``Q_cl, Q_safe, Q_cl_star, Q_safe_star``) -- the dict
    returned by the in-memory code path inside ``run_phase_V_planning``.
    """
    task_id = str(lead["task_id"])
    T = int(lead["horizon"])
    k_values = list(lead["k_values"])
    d_ref = np.asarray(lead["d_ref"], dtype=np.float64)

    lead_metrics = metrics_df[metrics_df["task_id"] == task_id]
    lead_scalars = scalars_df[scalars_df["task_id"] == task_id]

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    _plot_propagation(lead_scalars, task_id, fig_dir)
    _plot_stagewise_heatmap(lead_metrics, task_id, T, k_values, fig_dir)
    _plot_greedy_recovery(lead, k_values, d_ref, task_id, fig_dir)


def _plot_propagation(
    lead_scalars: pd.DataFrame,
    task_id: str,
    fig_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for op in ("classical", "safe"):
        sub = lead_scalars[lead_scalars["operator"] == op].sort_values("k")
        ax.plot(
            sub["k"], sub["propagation_distance_t"],
            marker=_MARKERS[op], color=_CLR[op], label=op,
        )
    ax.set_xlabel("backup budget k")
    ax.set_ylabel("propagation distance t (smaller = deeper)")
    ax.set_title(f"Propagation distance vs backup budget ({task_id})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_propagation_curve.pdf", dpi=300)
    plt.close(fig)


def _plot_stagewise_heatmap(
    lead_metrics: pd.DataFrame,
    task_id: str,
    T: int,
    k_values: list[int],
    fig_dir: Path,
) -> None:
    K = len(k_values)
    hm_cl = np.zeros((K, T), dtype=np.float64)
    hm_safe = np.zeros((K, T), dtype=np.float64)
    k_to_idx = {k: i for i, k in enumerate(k_values)}
    for _, row in lead_metrics.iterrows():
        k_idx = k_to_idx.get(int(row["k"]))
        if k_idx is None:
            continue
        t_idx = int(row["t"])
        if str(row["operator"]) == "classical":
            hm_cl[k_idx, t_idx] = float(row["v_err_inf"])
        else:
            hm_safe[k_idx, t_idx] = float(row["v_err_inf"])
    vmax = float(max(hm_cl.max(), hm_safe.max()))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
    for ax, data, title in ((axes[0], hm_cl, "classical"), (axes[1], hm_safe, "safe")):
        im = ax.imshow(
            data, aspect="auto", origin="lower", cmap="viridis",
            vmin=0.0, vmax=vmax if vmax > 0 else 1.0,
            extent=(-0.5, T - 0.5, 0.5, K + 0.5),
        )
        ax.set_xlabel("stage t")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="||V_k[t] - V*[t]||_inf")
    axes[0].set_ylabel("backup budget k")
    fig.suptitle(f"Stagewise sup-norm error ({task_id})")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_stagewise_error_heatmap.pdf", dpi=300)
    plt.close(fig)


def _plot_greedy_recovery(
    lead: dict[str, Any],
    k_values: list[int],
    d_ref: np.ndarray,
    task_id: str,
    fig_dir: Path,
) -> None:
    sweep = lead["sweep"]
    Q_cl = np.asarray(sweep["Q_cl"])
    Q_safe = np.asarray(sweep["Q_safe"])
    Q_cl_star = np.asarray(sweep["Q_cl_star"])
    Q_safe_star = np.asarray(sweep["Q_safe_star"])
    pi_cl_star = np.argmax(Q_cl_star, axis=-1)
    pi_safe_star = np.argmax(Q_safe_star, axis=-1)
    K = len(k_values)
    cl_recovery = np.zeros(K, dtype=np.float64)
    safe_recovery = np.zeros(K, dtype=np.float64)
    d_ref_sum = float(d_ref.sum())
    norm = d_ref_sum if d_ref_sum > 0 else 1.0
    for i in range(K):
        pi_cl_k = np.argmax(Q_cl[i], axis=-1)
        pi_safe_k = np.argmax(Q_safe[i], axis=-1)
        cl_recovery[i] = float(
            (d_ref * (pi_cl_k == pi_cl_star).astype(np.float64)).sum() / norm
        )
        safe_recovery[i] = float(
            (d_ref * (pi_safe_k == pi_safe_star).astype(np.float64)).sum() / norm
        )
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(k_values, cl_recovery, marker=_MARKERS["classical"],
            color=_CLR["classical"], label="classical")
    ax.plot(k_values, safe_recovery, marker=_MARKERS["safe"],
            color=_CLR["safe"], label="safe")
    ax.set_xlabel("backup budget k")
    ax.set_ylabel("greedy correctness under d_ref")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(f"Greedy recovery vs backup budget ({task_id})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_greedy_recovery.pdf", dpi=300)
    plt.close(fig)


# Note: this module is driven by ``run_phase_V_planning.run_planning`` which
# already holds the sweep tensors in memory. It is not meant to be invoked
# standalone: the greedy-recovery panel requires the (Q_cl, Q_safe,
# Q_cl_star, Q_safe_star) arrays from the sweep, which are not persisted.
# A standalone replay path would duplicate the runner's DP + calibration,
# against the "reuse, do not reimplement" constraint.
