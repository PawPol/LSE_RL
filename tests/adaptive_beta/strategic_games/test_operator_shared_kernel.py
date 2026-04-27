"""Operator-duplication audit (Phase VII-B spec §2.1, §12.2).

Spec authority: ``tasks/phase_VII_B_strategic_learning_coding_agent_spec.md``
§2.1 ("Do not duplicate the TAB / safe weighted log-sum-exp operator. Reuse
the shared operator from src/lse_rl/operator/tab_operator.py") and §12.2
("TAB operator outputs are imported from shared kernel, not reimplemented").

Invariants guarded
------------------
- ``np.logaddexp`` calls in the strategic_games tree appear ONLY inside
  ``smoothed_fictitious_play.py`` (where they compute the opponent-policy
  softmax — NOT a Bellman target).
- No file under ``strategic_games/`` reimplements ``g_β,γ``, ``ρ`` or
  ``d_eff``: the only operator-math source for any cell that needs it is
  imported from ``src.lse_rl.operator.tab_operator`` (or one of the
  Phase VII-A schedule modules that depends on it).
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable, List, Set

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_STRATEGIC_GAMES = _REPO_ROOT / "experiments" / "adaptive_beta" / "strategic_games"


def _iter_python_files(root: Path) -> Iterable[Path]:
    """Yield every .py file under ``root`` (excluding ``__pycache__``)."""
    for p in sorted(root.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        yield p


# ---------------------------------------------------------------------------
# logaddexp localisation
# ---------------------------------------------------------------------------
def test_logaddexp_only_in_smoothed_fictitious_play() -> None:
    """`spec §2.1` — ``np.logaddexp`` (or ``logaddexp.reduce``) appears ONLY
    in ``adversaries/smoothed_fictitious_play.py``. A regression that
    re-implements the TAB Bellman operator using ``logaddexp`` elsewhere
    would silently bypass the shared kernel.
    """
    offenders: List[Path] = []
    allowed_basename = "smoothed_fictitious_play.py"
    for path in _iter_python_files(_STRATEGIC_GAMES):
        if path.name == "__init__.py":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if "logaddexp" in text and path.name != allowed_basename:
            offenders.append(path.relative_to(_REPO_ROOT))
    assert not offenders, (
        f"``logaddexp`` calls found outside smoothed_fictitious_play.py: "
        f"{offenders}. Spec §2.1 forbids re-implementing the TAB / safe "
        f"weighted log-sum-exp operator."
    )


# ---------------------------------------------------------------------------
# Operator-math shared-kernel audit
# ---------------------------------------------------------------------------
_FORBIDDEN_NAMES: Set[str] = {
    # Names that would indicate a local re-implementation of the TAB
    # operator's primitives. Any *function definition* with one of these
    # names under strategic_games/ is a violation.
    "g_beta",
    "g_beta_gamma",
    "tab_operator_primitive",
    "weighted_lse",
    "safe_weighted_lse",
    "compute_d_eff",
    "rho_beta",
    "rho_beta_gamma",
}


def _function_names_defined(path: Path) -> Set[str]:
    """Return the set of top-level + class-level function names defined in
    ``path``.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return set()
    out: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.add(node.name)
    return out


def test_strategic_games_does_not_redefine_operator_primitives() -> None:
    """`spec §2.1` — no file under ``strategic_games/`` defines a function
    whose name matches one of the operator-math primitives (g_beta, rho,
    d_eff, weighted_lse, ...). A reimplementation under any of these
    canonical names would silently bypass the shared kernel.
    """
    offenders: List[str] = []
    for path in _iter_python_files(_STRATEGIC_GAMES):
        names = _function_names_defined(path)
        bad = names & _FORBIDDEN_NAMES
        if bad:
            offenders.append(
                f"{path.relative_to(_REPO_ROOT)}: {sorted(bad)}"
            )
    assert not offenders, (
        f"strategic_games/ contains operator-primitive redefinitions: "
        f"{offenders}. Spec §2.1: import from src.lse_rl.operator.tab_operator."
    )


def test_strategic_games_does_not_compute_bellman_targets_with_logaddexp() -> None:
    """`spec §2.1` — even within smoothed_fictitious_play.py, ``logaddexp``
    is used for the OPPONENT-policy softmax, not a Bellman target. We
    assert the only call site is inside ``_logsumexp_stable`` / the
    ``_logit_policy`` codepath — i.e. the function names that touch
    ``logaddexp`` are exclusively softmax-related.
    """
    target = (
        _STRATEGIC_GAMES / "adversaries" / "smoothed_fictitious_play.py"
    )
    text = target.read_text(encoding="utf-8")
    # Any function whose body contains ``logaddexp`` should be one of:
    #   _logsumexp_stable, _logit_policy
    tree = ast.parse(text)
    fn_names_with_logaddexp: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body_src = ast.unparse(node)
            if "logaddexp" in body_src:
                fn_names_with_logaddexp.append(node.name)
    # Allowed function names that may legitimately call logaddexp.
    allowed = {"_logsumexp_stable", "_logit_policy"}
    bad = [n for n in fn_names_with_logaddexp if n not in allowed]
    assert not bad, (
        f"smoothed_fictitious_play.py uses ``logaddexp`` inside "
        f"unexpected functions {bad}; allowed callers are {sorted(allowed)}."
    )


# ---------------------------------------------------------------------------
# Tripwire
# ---------------------------------------------------------------------------
def test_invariant_subpackage_init_documents_no_operator_duplication() -> None:
    """`spec §2.1` — the strategic_games ``__init__.py`` documents the no-
    duplication contract referencing ``tab_operator``. A regression that
    quietly removed this contract from the docstring would weaken the
    no-duplication audit.
    """
    init = _STRATEGIC_GAMES / "__init__.py"
    text = init.read_text(encoding="utf-8")
    assert "tab_operator" in text, (
        "__init__.py docstring no longer references tab_operator; the "
        "no-duplication contract has been silently lost."
    )
    assert "MUST NOT" in text or "do not" in text.lower(), (
        "__init__.py docstring no longer states the no-duplication contract."
    )
