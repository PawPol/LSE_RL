"""Smoke tests for Phase VIII tab-six-games figure scripts."""

from __future__ import annotations

import ast
import importlib
import io
import tokenize
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from experiments.adaptive_beta.tab_six_games.analysis.aggregate import LONG_CSV_COLUMNS


FIGURE_MODULES = [
    (
        "beta_sweep_plots",
        "experiments.adaptive_beta.tab_six_games.analysis.beta_sweep_plots",
    ),
    (
        "learning_curves",
        "experiments.adaptive_beta.tab_six_games.analysis.learning_curves",
    ),
    (
        "contraction_plots",
        "experiments.adaptive_beta.tab_six_games.analysis.contraction_plots",
    ),
    (
        "sign_switching_plots",
        "experiments.adaptive_beta.tab_six_games.analysis.sign_switching_plots",
    ),
    (
        "safety_catastrophe",
        "experiments.adaptive_beta.tab_six_games.analysis.safety_catastrophe",
    ),
]


@pytest.fixture
def minimal_long_csv(tmp_path: Path) -> Path:
    """Write an 8-row, all-49-column long CSV that exercises every plot."""
    csv_path = tmp_path / "minimal_long.csv"
    methods = [
        "fixed_beta_-1",
        "fixed_beta_0",
        "fixed_beta_+1",
        "vanilla",
        "contraction_ucb",
        "return_ucb",
        "oracle",
        "hand_adaptive",
    ]

    rows: list[dict[str, object]] = []
    for idx, method in enumerate(methods):
        row: dict[str, object] = {}
        for col in LONG_CSV_COLUMNS:
            if col == "run_id":
                row[col] = f"run_{idx}"
            elif col == "config_hash":
                row[col] = f"cfg_{idx}"
            elif col == "phase":
                row[col] = "VIII"
            elif col == "stage":
                row[col] = "stage_smoke"
            elif col == "game":
                row[col] = "matching_pennies"
            elif col == "subcase":
                row[col] = "canonical"
            elif col == "method":
                row[col] = method
            elif col == "seed":
                row[col] = idx
            elif col == "episode":
                row[col] = idx
            elif col == "return":
                row[col] = float(idx + 1)
            elif col == "regime":
                row[col] = "post" if idx >= 4 else "pre"
            elif col == "ucb_arm_index":
                row[col] = float(idx % 3)
            elif col in {"switch_event", "shift_event"}:
                row[col] = 1.0
            elif col == "oracle_beta":
                row[col] = -1.0 if idx < 4 else 1.0
            elif col == "beta_sign_correct":
                row[col] = 1.0 if idx % 2 == 0 else 0.0
            elif col == "catastrophic":
                row[col] = 1.0 if idx in {1, 6} else 0.0
            elif col == "worst_window_return_percentile":
                row[col] = float(10 - idx)
            else:
                row[col] = float(idx) / 10.0
        rows.append(row)

    pd.DataFrame(rows, columns=list(LONG_CSV_COLUMNS)).to_csv(
        csv_path, index=False
    )
    return csv_path


def _import_module(module_path: str) -> ModuleType:
    return importlib.import_module(module_path)


def _run_make_figure(
    module_path: str,
    processed_long_csv: Path,
    out_dir: Path,
) -> dict[str, Path]:
    module = _import_module(module_path)
    result = module.make_figure(processed_long_csv, out_dir)
    assert isinstance(result, dict)
    return result


def _docstring_line_numbers(source: str) -> set[int]:
    tree = ast.parse(source)
    allowed: set[int] = set()

    def visit(node: ast.AST) -> None:
        body = getattr(node, "body", None)
        if isinstance(body, list) and body:
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                end_lineno = getattr(first, "end_lineno", first.lineno)
                allowed.update(range(first.lineno, end_lineno + 1))
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return allowed


def _comment_line_numbers(source: str) -> set[int]:
    allowed: set[int] = set()
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    for token in tokens:
        if token.type == tokenize.COMMENT:
            allowed.add(token.start[0])
    return allowed


@pytest.mark.parametrize("module_name,module_path", FIGURE_MODULES)
def test_make_figure_runs_on_minimal_fixture(
    module_name: str,
    module_path: str,
    minimal_long_csv: Path,
    tmp_path: Path,
) -> None:
    result = _run_make_figure(
        module_path, minimal_long_csv, tmp_path / module_name
    )

    assert result


@pytest.mark.parametrize("module_name,module_path", FIGURE_MODULES)
def test_outputs_are_pdfs(
    module_name: str,
    module_path: str,
    minimal_long_csv: Path,
    tmp_path: Path,
) -> None:
    result = _run_make_figure(
        module_path, minimal_long_csv, tmp_path / module_name
    )

    assert all(isinstance(path, Path) for path in result.values())
    assert all(path.suffix == ".pdf" for path in result.values())


@pytest.mark.parametrize("module_name,module_path", FIGURE_MODULES)
def test_pdfs_non_empty(
    module_name: str,
    module_path: str,
    minimal_long_csv: Path,
    tmp_path: Path,
) -> None:
    result = _run_make_figure(
        module_path, minimal_long_csv, tmp_path / module_name
    )

    for path in result.values():
        assert path.is_file(), path
        assert path.stat().st_size > 100, path


@pytest.mark.parametrize("module_name,module_path", FIGURE_MODULES)
def test_no_demo_synth_path(module_name: str, module_path: str) -> None:
    module = _import_module(module_path)
    source_path = Path(module.__file__)
    source = source_path.read_text(encoding="utf-8")
    allowed_lines = _docstring_line_numbers(source) | _comment_line_numbers(source)

    for lineno, line in enumerate(source.splitlines(), start=1):
        lowered = line.lower()
        if "demo" in lowered or "synth" in lowered:
            assert lineno in allowed_lines, (
                f"{module_name} contains demo/synth outside docstring/comment "
                f"at {source_path}:{lineno}"
            )
