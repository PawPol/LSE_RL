"""Build the Phase V + VI main-paper summary table.

Reads from:
- results/search/shortlist.csv (Family A + C frozen shortlist)
- results/search/shortlist_VI_A_C.csv (stronger C stress shortlist)
- results/search/shortlist_VI_E_A_stoch.csv (stochastic A shortlist)
- results/rl_VI_F/dual_eval.parquet
- results/planning_VI_A/safe_vs_raw_stability.parquet

Writes:
- results/summaries/main_table.csv
- results/summaries/main_table.md (human-readable)
- results/summaries/main_table.tex (LaTeX, for paper inclusion)
"""
from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_family_A_frozen() -> pd.DataFrame:
    df = pd.read_csv(_REPO_ROOT / "results/search/shortlist.csv")
    return df[df["family"] == "A"].copy()


def _load_family_C_stronger() -> pd.DataFrame:
    path = _REPO_ROOT / "results/search/shortlist_VI_A_C.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_family_A_stoch() -> pd.DataFrame:
    path = _REPO_ROOT / "results/search/shortlist_VI_E_A_stoch.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_dual_eval() -> pd.DataFrame:
    path = _REPO_ROOT / "results/rl_VI_F/dual_eval.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _load_stability_VI_A() -> pd.DataFrame:
    path = _REPO_ROOT / "results/planning_VI_A/safe_vs_raw_stability.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _fmt_sci(x: float, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if abs(x) < 1e-3 and x != 0:
        return f"{x:.{digits}e}"
    return f"{x:.{digits + 2}f}"


def _build_main_table() -> pd.DataFrame:
    """Aggregate per-family headline metrics for the summary table."""
    rows: list[dict[str, Any]] = []

    # Row 1: Family A deterministic (frozen shortlist).
    fA = _load_family_A_frozen()
    if len(fA):
        rows.append({
            "family": "A (det.)",
            "interpretation": "Jackpot vs smooth stream, deterministic",
            "n_tasks": int(len(fA)),
            "promotion_mode": ", ".join(sorted(fA["promotion_mode"].unique())),
            "max_abs_value_gap_norm": float(fA["value_gap_norm"].abs().max()),
            "max_policy_disagreement": float(fA["policy_disagreement"].max()),
            "start_state_flip_rate": float(fA["start_state_flip"].mean()),
            "raw_deriv_max": float(fA["raw_local_deriv_max"].max()),
            "clip_fraction_max": float(fA["clip_fraction"].max()),
            "gap_cl_eval_mean": float("nan"),
            "gap_safe_eval_mean": float("nan"),
        })

    # Row 2: Family A stochastic (VI-E) + dual-eval gaps (VI-F).
    fA_stoch = _load_family_A_stoch()
    dual = _load_dual_eval()
    if len(fA_stoch):
        row = {
            "family": "A (stoch., VI-E)",
            "interpretation": "Jackpot vs smooth stream, p_transit<1",
            "n_tasks": int(len(fA_stoch)),
            "promotion_mode": ", ".join(sorted(fA_stoch["promotion_mode"].unique())),
            "max_abs_value_gap_norm": float(fA_stoch["value_gap_norm"].abs().max()),
            "max_policy_disagreement": float(fA_stoch["policy_disagreement"].max()),
            "start_state_flip_rate": float(fA_stoch["start_state_flip"].mean()),
            "raw_deriv_max": float(fA_stoch["raw_local_deriv_max"].max()),
            "clip_fraction_max": float(fA_stoch["clip_fraction"].max()),
            "gap_cl_eval_mean": (
                float(dual["gap_cl_eval"].mean()) if len(dual) else float("nan")
            ),
            "gap_safe_eval_mean": (
                float(dual["gap_safe_eval"].mean()) if len(dual) else float("nan")
            ),
        }
        rows.append(row)

    # Row 3: Family C stronger (VI-A).
    fC = _load_family_C_stronger()
    if len(fC):
        rows.append({
            "family": "C (safety, VI-A)",
            "interpretation": "Raw-stress: safe preserves cert, raw breaches it",
            "n_tasks": int(len(fC)),
            "promotion_mode": ", ".join(sorted(fC["promotion_mode"].unique())),
            "max_abs_value_gap_norm": float("nan"),
            "max_policy_disagreement": float("nan"),
            "start_state_flip_rate": float("nan"),
            "raw_deriv_max": float(fC["raw_local_deriv_max"].max()),
            "clip_fraction_max": float(fC["clip_fraction"].max()),
            "gap_cl_eval_mean": float("nan"),
            "gap_safe_eval_mean": float("nan"),
        })

    return pd.DataFrame(rows)


def render_table() -> None:
    out_dir = _REPO_ROOT / "results/summaries"
    out_dir.mkdir(parents=True, exist_ok=True)

    table = _build_main_table()
    csv_path = out_dir / "main_table.csv"
    table.to_csv(csv_path, index=False)

    md_lines = [
        "# Phase V + VI Main-Paper Summary Table",
        "",
        "One row per family grouping with headline metrics. `promotion_mode`:",
        "`binding_clip` (clipping active in [5%, 80%]) or "
        "`safe_active_no_distortion` (raw already within cert; no clip).",
        "",
        "| family | interpretation | n_tasks | promotion_mode | max \\|vgn\\| "
        "| max pol_disagree | flip rate | max d_raw | max clip frac "
        "| mean gap_cl_eval | mean gap_safe_eval |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, row in table.iterrows():
        md_lines.append(
            f"| {row['family']} | {row['interpretation']} | {row['n_tasks']} "
            f"| {row['promotion_mode']} "
            f"| {_fmt_sci(row['max_abs_value_gap_norm'])} "
            f"| {_fmt_sci(row['max_policy_disagreement'])} "
            f"| {_fmt_sci(row['start_state_flip_rate'])} "
            f"| {_fmt_sci(row['raw_deriv_max'])} "
            f"| {_fmt_sci(row['clip_fraction_max'])} "
            f"| {_fmt_sci(row['gap_cl_eval_mean'])} "
            f"| {_fmt_sci(row['gap_safe_eval_mean'])} |"
        )
    (out_dir / "main_table.md").write_text("\n".join(md_lines))

    tex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Phase V + Phase VI empirical summary. "
        r"Family A (deterministic): Pareto-better-on-safe-metric via "
        r"\texttt{safe\_active\_no\_distortion} clipping regime. "
        r"Family A (stochastic, VI-E/F): dual-objective result "
        r"$V_\mathrm{cl}^{\pi^*_\mathrm{cl}} - V_\mathrm{cl}^{\pi^*_\mathrm{safe}} \approx 0$ "
        r"but $V_\mathrm{safe}^{\pi^*_\mathrm{safe}} - V_\mathrm{safe}^{\pi^*_\mathrm{cl}} > 0$. "
        r"Family C (VI-A): clipping preserves contraction cert where raw breaches it.}",
        r"\label{tab:phaseVI_summary}",
        r"\begin{tabular}{l l r l r r r r r r r}",
        r"\toprule",
        r"Family & Interpretation & $n$ & Promo mode & $\max |vgn|$ & $\max \mathrm{PD}$ "
        r"& Flip rate & $\max d_\mathrm{raw}$ & $\max \mathrm{clip}$ "
        r"& $\bar g_\mathrm{cl}$ & $\bar g_\mathrm{safe}$ \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        fam = row['family'].replace('_', r'\_')
        interp = row['interpretation'].replace('_', r'\_')
        promo = row['promotion_mode'].replace('_', r'\_')
        tex_lines.append(
            f"{fam} & {interp} & {int(row['n_tasks'])} & "
            f"\\texttt{{{promo}}} & "
            f"{_fmt_sci(row['max_abs_value_gap_norm'])} & "
            f"{_fmt_sci(row['max_policy_disagreement'])} & "
            f"{_fmt_sci(row['start_state_flip_rate'])} & "
            f"{_fmt_sci(row['raw_deriv_max'])} & "
            f"{_fmt_sci(row['clip_fraction_max'])} & "
            f"{_fmt_sci(row['gap_cl_eval_mean'])} & "
            f"{_fmt_sci(row['gap_safe_eval_mean'])} \\\\"
        )
    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    (out_dir / "main_table.tex").write_text("\n".join(tex_lines))
    print(f"wrote {csv_path}")
    print(f"wrote {out_dir / 'main_table.md'}")
    print(f"wrote {out_dir / 'main_table.tex'}")


if __name__ == "__main__":
    render_table()
