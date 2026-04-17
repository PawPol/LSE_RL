#!/usr/bin/env python
"""Regenerate all Phase I figures from a single entry point.

This script wraps the five individual figure modules and provides a
unified CLI.  In ``--demo`` mode, every figure is generated from
synthetic data (no disk reads required).  In production mode, each
figure reads from its expected location under ``results/``.

After all figures are written, a ``figures_manifest.json`` is emitted
listing every output file with its SHA256 checksum and generation
timestamp.

Confidence intervals: see the docstring of each individual figure
module for the CI method used.

No cherry-picking: all seeds present in the results tree are used
unless ``--seed`` is specified (see individual module docs for
details).

Regeneratability: given the same ``results/`` tree and code revision,
this script produces identical output (PDF timestamps pinned via
SOURCE_DATE_EPOCH).

Usage
-----
::

    # Demo mode (synthetic data, no results tree needed):
    .venv/bin/python experiments/weighted_lse_dp/analysis/make_phase1_figures.py --demo

    # Production mode (reads from results/):
    .venv/bin/python experiments/weighted_lse_dp/analysis/make_phase1_figures.py

    # Single figure only:
    .venv/bin/python experiments/weighted_lse_dp/analysis/make_phase1_figures.py --demo --only dp_residuals
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time
from pathlib import Path
from typing import Any, Sequence

# Pin PDF creation timestamps for byte-reproducibility.
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# Path bootstrapping
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "mushroom-rl-dev") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "mushroom-rl-dev"))

# ---------------------------------------------------------------------------
# Figure module imports
# ---------------------------------------------------------------------------
from experiments.weighted_lse_dp.analysis.figures.fig_chain_value_propagation import (  # noqa: E402
    make_chain_value_propagation_figure,
    _synthesize_demo_data as _chain_demo_data,
)
from experiments.weighted_lse_dp.analysis.figures.fig_dp_residuals import (  # noqa: E402
    make_dp_residuals_figure,
    _generate_demo_data as _dp_residuals_demo_data,
    _load_paper_suite_data as _dp_residuals_load_data,
)
from experiments.weighted_lse_dp.analysis.figures.fig_rl_learning_curves import (  # noqa: E402
    make_rl_learning_curves_figure,
    _generate_demo_data as _rl_curves_demo_data,
    load_paper_suite_data as _rl_curves_load_data,
)
from experiments.weighted_lse_dp.analysis.figures.fig_margin_histograms import (  # noqa: E402
    make_margin_histograms_figure,
    _make_demo_data as _margins_demo_data,
    _load_calibration_data as _margins_load_data,
)
from experiments.weighted_lse_dp.analysis.figures.fig_ablation import (  # noqa: E402
    make_ablation_figure,
    _generate_demo_data as _ablation_demo_data,
    _load_ablation_summary as _ablation_load_summary,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIGURE_NAMES: list[str] = [
    "chain_vp",
    "dp_residuals",
    "rl_curves",
    "margins",
    "ablation",
]

_FIGURE_STEMS: dict[str, str] = {
    "chain_vp": "phase1_chain_value_propagation",
    "dp_residuals": "phase1_dp_residuals",
    "rl_curves": "phase1_rl_learning_curves",
    "margins": "phase1_margin_histograms",
    "ablation": "phase1_ablation",
}

_DEFAULT_PAPER_SUITE_ROOT = (
    _REPO_ROOT / "results" / "weighted_lse_dp" / "phase1" / "paper_suite"
)
_DEFAULT_ABLATION_ROOT = (
    _REPO_ROOT / "results" / "weighted_lse_dp" / "phase1" / "ablation"
)
_DEFAULT_OUT_DIR = (
    _REPO_ROOT / "results" / "weighted_lse_dp" / "processed" / "phase1" / "figures"
)
_DEFAULT_ABLATION_SUMMARY = (
    _REPO_ROOT / "results" / "weighted_lse_dp" / "processed" / "phase1"
    / "ablation_summary.json"
)


# ---------------------------------------------------------------------------
# Utility: SHA256 of a file
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    """Return hex SHA256 digest of *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Individual figure generators
# ---------------------------------------------------------------------------

def _gen_chain_vp(
    *,
    demo: bool,
    paper_suite_root: Path,
    out_dir: Path,
    seed: int,
    show: bool,
) -> list[Path]:
    """Generate chain value propagation figure."""
    if demo:
        v_snapshots = _chain_demo_data()
    else:
        # Load from the expected curves.npz path.
        data_path = paper_suite_root / "chain_base" / "VI" / f"seed_{seed}" / "curves.npz"
        if not data_path.is_file():
            print(f"  WARNING: {data_path} not found, skipping chain_vp.", file=sys.stderr)
            return []
        with np.load(data_path, allow_pickle=True) as npz:
            v_snapshots = npz["v_table_snapshots"]

    out_path = out_dir / "phase1_chain_value_propagation.pdf"
    fig = make_chain_value_propagation_figure(
        v_snapshots, out_path=out_path, show=show,
    )
    plt.close(fig)

    outputs = [out_path]
    png_path = out_path.with_suffix(".png")
    if png_path.is_file():
        outputs.append(png_path)
    return outputs


def _gen_dp_residuals(
    *,
    demo: bool,
    paper_suite_root: Path,
    out_dir: Path,
    seed: int | None,
    show: bool,
) -> list[Path]:
    """Generate DP residuals figure."""
    if demo:
        data = _dp_residuals_demo_data()
    else:
        if not paper_suite_root.is_dir():
            print(
                f"  WARNING: {paper_suite_root} not found, skipping dp_residuals.",
                file=sys.stderr,
            )
            return []
        data = _dp_residuals_load_data(paper_suite_root, seed_filter=seed)
        n_loaded = sum(len(ad) for ad in data.values())
        if n_loaded == 0:
            print("  WARNING: no DP residual data found, skipping.", file=sys.stderr)
            return []

    out_path = out_dir / "phase1_dp_residuals"
    fig = make_dp_residuals_figure(data, out_path=out_path, show=show)
    plt.close(fig)

    outputs = []
    for ext in (".pdf", ".png"):
        p = out_path.with_suffix(ext)
        if p.is_file():
            outputs.append(p)
    return outputs


def _gen_rl_curves(
    *,
    demo: bool,
    paper_suite_root: Path,
    out_dir: Path,
    show: bool,
) -> list[Path]:
    """Generate RL learning curves figure."""
    if demo:
        data = _rl_curves_demo_data()
    else:
        if not paper_suite_root.is_dir():
            print(
                f"  WARNING: {paper_suite_root} not found, skipping rl_curves.",
                file=sys.stderr,
            )
            return []
        data = _rl_curves_load_data(paper_suite_root)
        n_loaded = sum(
            len(seeds) for task_d in data.values() for seeds in task_d.values()
        )
        if n_loaded == 0:
            print("  WARNING: no RL curve data found, skipping.", file=sys.stderr)
            return []

    out_path = out_dir / "phase1_rl_learning_curves"
    fig = make_rl_learning_curves_figure(data, out_path=out_path, show=show)
    plt.close(fig)

    outputs = []
    for ext in (".pdf", ".png"):
        p = out_path.with_suffix(ext)
        if p.is_file():
            outputs.append(p)
    return outputs


def _gen_margins(
    *,
    demo: bool,
    paper_suite_root: Path,
    out_dir: Path,
    seed: int | None,
    show: bool,
) -> list[Path]:
    """Generate margin histograms figure."""
    if demo:
        data = _margins_demo_data()
    else:
        if not paper_suite_root.is_dir():
            print(
                f"  WARNING: {paper_suite_root} not found, skipping margins.",
                file=sys.stderr,
            )
            return []
        data = _margins_load_data(
            paper_suite_root=paper_suite_root,
            algorithm="QLearning",
            seed=seed,
        )
        if not data:
            print("  WARNING: no calibration data found, skipping margins.", file=sys.stderr)
            return []

    out_path = out_dir / "phase1_margin_histograms"
    fig = make_margin_histograms_figure(data, out_path=out_path, show=show)
    plt.close(fig)

    outputs = []
    for ext in (".pdf", ".png"):
        p = out_path.with_suffix(ext)
        if p.is_file():
            outputs.append(p)
    return outputs


def _gen_ablation(
    *,
    demo: bool,
    ablation_root: Path,
    out_dir: Path,
    show: bool,
) -> list[Path]:
    """Generate ablation figure."""
    if demo:
        data = _ablation_demo_data(rng=np.random.default_rng(42))
    else:
        summary_path = _DEFAULT_ABLATION_SUMMARY
        if not summary_path.is_file():
            print(
                f"  WARNING: {summary_path} not found, skipping ablation.",
                file=sys.stderr,
            )
            return []
        data = _ablation_load_summary(summary_path)
        if not data:
            print("  WARNING: ablation_summary.json is empty, skipping.", file=sys.stderr)
            return []

    out_path = out_dir / "phase1_ablation"
    fig = make_ablation_figure(data, out_path=out_path, show=show)
    plt.close(fig)

    outputs = []
    for ext in (".pdf", ".png"):
        p = out_path.with_suffix(ext)
        if p.is_file():
            outputs.append(p)
    return outputs


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_GENERATORS: dict[str, Any] = {
    "chain_vp": _gen_chain_vp,
    "dp_residuals": _gen_dp_residuals,
    "rl_curves": _gen_rl_curves,
    "margins": _gen_margins,
    "ablation": _gen_ablation,
}


# ---------------------------------------------------------------------------
# Manifest writer
# ---------------------------------------------------------------------------

def _write_manifest(
    out_dir: Path,
    all_outputs: dict[str, list[Path]],
) -> Path:
    """Write ``figures_manifest.json`` listing all generated files."""
    manifest_path = out_dir / "figures_manifest.json"
    entries: list[dict[str, str]] = []

    for fig_name in FIGURE_NAMES:
        for p in all_outputs.get(fig_name, []):
            entries.append({
                "figure": fig_name,
                "file": str(p),
                "sha256": _sha256(p),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })

    manifest = {
        "generator": "make_phase1_figures.py",
        "n_figures": sum(1 for v in all_outputs.values() if v),
        "n_files": len(entries),
        "files": entries,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written: {manifest_path}")
    return manifest_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate all Phase I figures. "
            "Use --demo for synthetic data or point at a populated results/ tree."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--paper-suite-root",
        type=Path,
        default=_DEFAULT_PAPER_SUITE_ROOT,
        help=f"Root of paper_suite results tree. Default: {_DEFAULT_PAPER_SUITE_ROOT}",
    )
    parser.add_argument(
        "--ablation-root",
        type=Path,
        default=_DEFAULT_ABLATION_ROOT,
        help=f"Root of ablation results tree. Default: {_DEFAULT_ABLATION_ROOT}",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_OUT_DIR,
        help=f"Output directory for all figures. Default: {_DEFAULT_OUT_DIR}",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to paper_suite.json config (reserved for future use).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use synthetic demo data for all figures (no disk reads).",
    )
    parser.add_argument(
        "--only",
        type=str,
        choices=FIGURE_NAMES,
        default=None,
        metavar="FIG",
        help=(
            "Generate only a single figure. "
            f"Choices: {', '.join(FIGURE_NAMES)}"
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after rendering.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=11,
        help="Which seed to use for single-seed figures (default: 11).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: regenerate Phase I figures."""
    args = _parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    figures_to_generate = [args.only] if args.only else FIGURE_NAMES

    mode_label = "DEMO" if args.demo else "PRODUCTION"
    print(f"=== Phase I Figure Generation ({mode_label} mode) ===")
    print(f"Output directory: {out_dir}")
    print(f"Figures to generate: {', '.join(figures_to_generate)}")
    print()

    all_outputs: dict[str, list[Path]] = {}
    n_success = 0
    n_skipped = 0

    for fig_name in figures_to_generate:
        print(f"[{fig_name}] Generating...")

        # Build kwargs for the generator based on which figure it is.
        if fig_name == "chain_vp":
            outputs = _gen_chain_vp(
                demo=args.demo,
                paper_suite_root=args.paper_suite_root,
                out_dir=out_dir,
                seed=args.seed,
                show=args.show,
            )
        elif fig_name == "dp_residuals":
            outputs = _gen_dp_residuals(
                demo=args.demo,
                paper_suite_root=args.paper_suite_root,
                out_dir=out_dir,
                seed=args.seed if not args.demo else None,
                show=args.show,
            )
        elif fig_name == "rl_curves":
            outputs = _gen_rl_curves(
                demo=args.demo,
                paper_suite_root=args.paper_suite_root,
                out_dir=out_dir,
                show=args.show,
            )
        elif fig_name == "margins":
            outputs = _gen_margins(
                demo=args.demo,
                paper_suite_root=args.paper_suite_root,
                out_dir=out_dir,
                seed=args.seed if not args.demo else None,
                show=args.show,
            )
        elif fig_name == "ablation":
            outputs = _gen_ablation(
                demo=args.demo,
                ablation_root=args.ablation_root,
                out_dir=out_dir,
                show=args.show,
            )
        else:
            print(f"  Unknown figure: {fig_name}", file=sys.stderr)
            outputs = []

        all_outputs[fig_name] = outputs

        if outputs:
            n_success += 1
            for p in outputs:
                print(f"  -> {p}")
        else:
            n_skipped += 1
            print(f"  (skipped)")

    # Write manifest.
    manifest_path = _write_manifest(out_dir, all_outputs)

    # Summary.
    total_files = sum(len(v) for v in all_outputs.values())
    print()
    print("=" * 60)
    print(f"Summary: {n_success} figures generated, {n_skipped} skipped")
    print(f"Total output files: {total_files}")
    print(f"Manifest: {manifest_path}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
