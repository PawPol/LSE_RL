"""Phase VIII v10: aggregate raw metrics into per-(stage,cell,gamma,method,seed) frame.

Reads manifest.jsonl from each of v10_tier1_canonical, v10_tier2_gamma_beta_headline,
v10_tier3_gamma_cell_coverage, then opens each metrics.npz once and computes the
v5b headline-routed AUC plus auxiliary metrics (alignment, q_abs_max final,
divergence_event_sum, return_AUC, bellman_residual_AUC).

Headline routing per spec section 13.10 / M6_summary:
- DC-Long50 / DC-Medium20 / DC-Short10: bellman_residual_beta_AUC = trapezoid(-log(bellman_residual + 1e-8), episode)
- All other cells: cumulative_return_AUC = trapezoid(return, episode)

Outputs:
  results/adaptive_beta/tab_six_games/figures/v10/tables/v10_per_cell_per_method_per_gamma.csv (long-format, NOT committed)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/liq/Documents/Claude/Projects/LSE_RL")
RAW_BASE = ROOT / "results/adaptive_beta/tab_six_games/raw/VIII"
OUT_DIR = ROOT / "results/adaptive_beta/tab_six_games/figures/v10/tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STAGES = {
    "tier1": "v10_tier1_canonical",
    "tier2": "v10_tier2_gamma_beta_headline",
    "tier3": "v10_tier3_gamma_cell_coverage",
}

DC_CELLS = {"DC-Long50", "DC-Medium20", "DC-Short10"}

BETA_RE = re.compile(r"^fixed_beta_([+-][0-9.]+)$")


def beta_from_method(method: str) -> float:
    if method == "vanilla":
        return 0.0
    m = BETA_RE.match(method)
    if not m:
        raise ValueError(f"Cannot parse beta from method '{method}'")
    return float(m.group(1))


def headline_field(subcase: str) -> str:
    return "bellman_residual" if subcase in DC_CELLS else "return"


def auc_for_field(arr: np.ndarray, field: str) -> float:
    """Compute per-spec AUC over the episode index.

    return: trapezoid(return, episode_idx)
    bellman_residual: trapezoid(-log(bellman_residual + 1e-8), episode_idx)
    """
    epi = np.arange(arr.shape[0], dtype=np.float64)
    if field == "return":
        return float(np.trapezoid(arr, epi))
    if field == "bellman_residual":
        return float(np.trapezoid(-np.log(arr + 1e-8), epi))
    raise ValueError(field)


def aggregate_stage(stage: str) -> pd.DataFrame:
    base = RAW_BASE / STAGES[stage]
    manifest_path = base / "manifest.jsonl"
    rows: list[dict] = []
    with open(manifest_path) as f:
        for ln_no, ln in enumerate(f):
            ln = ln.strip()
            if not ln:
                continue
            entry = json.loads(ln)
            if "_schema" in entry:
                continue
            if entry.get("status") != "completed":
                continue
            result_path = ROOT / entry["result_path"]
            metrics_path = result_path / "metrics.npz"
            run_json_path = result_path / "run.json"
            if not metrics_path.exists():
                continue
            try:
                with np.load(metrics_path) as m:
                    ret = m["return"]
                    bres = m["bellman_residual"]
                    align = m["alignment_rate"]
                    qmax = m["q_abs_max"]
                    div = m["divergence_event"]
            except Exception as e:
                print(f"[warn] failed to read {metrics_path}: {e}")
                continue
            try:
                with open(run_json_path) as fr:
                    rj = json.load(fr)
            except Exception:
                rj = {}

            game = entry["game"]
            subcase = entry["subcase"]
            method = entry["method"]
            seed = int(entry["seed"])
            gamma = float(entry.get("gamma") or rj.get("gamma") or 0.95)

            beta = beta_from_method(method)
            field = headline_field(subcase)
            if field == "return":
                headline_auc = auc_for_field(ret, "return")
            else:
                headline_auc = auc_for_field(bres, "bellman_residual")

            rows.append(
                dict(
                    stage=stage,
                    game=game,
                    subcase=subcase,
                    method=method,
                    beta=beta,
                    gamma=gamma,
                    seed=seed,
                    headline_metric=rj.get("headline_metric", "cumulative_return_auc"),
                    AUC=headline_auc,
                    return_AUC=auc_for_field(ret, "return"),
                    bellman_residual_AUC=auc_for_field(bres, "bellman_residual"),
                    align_final=float(align[-1]),
                    align_last200=float(align[-200:].mean()),
                    q_abs_max_final=float(qmax[-1]),
                    q_abs_max_max=float(qmax.max()),
                    divergence_event_sum=int(div.sum()),
                    return_final=float(ret[-1]),
                    return_last200=float(ret[-200:].mean()),
                    diverged=bool(rj.get("diverged", False)),
                )
            )
            if (ln_no % 500) == 0:
                print(f"  {stage}: {ln_no} manifest lines processed")
    df = pd.DataFrame(rows)
    print(f"[done] {stage}: {len(df)} rows")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=OUT_DIR / "v10_per_cell_per_method_per_gamma.csv",
    )
    args = ap.parse_args()
    parts = []
    for stage in ("tier1", "tier2", "tier3"):
        parts.append(aggregate_stage(stage))
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["stage", "game", "subcase", "gamma", "beta", "seed"]).reset_index(drop=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
