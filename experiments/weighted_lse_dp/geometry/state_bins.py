"""Phase IV-C §4: State binning for state-dependent frozen schedulers."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def construct_bins(
    states: NDArray[np.int64],
    mode: str = "exact_state",
    n_bins: int | None = None,
) -> NDArray[np.int64]:
    """Construct state bins for state-dependent scheduling.

    Parameters
    ----------
    states : NDArray[np.int64], shape (|S|,)
        State indices to be binned (assumed contiguous integers 0..S-1).
    mode : str
        Binning strategy:
        - ``"exact_state"``: each unique state is its own bin (identity map).
        - ``"uniform"``: uniform partition of [0, S-1] into ``n_bins`` bins.
        - ``"margin_quantile"``: placeholder; falls back to ``"uniform"``.
    n_bins : int, optional
        Number of bins for ``"uniform"`` mode. Defaults to ``len(unique(states))``.

    Returns
    -------
    NDArray[np.int64], shape (|S|,)
        Bin assignment for each state, values in ``[0, n_bins_actual)``.
    """
    states = np.asarray(states, dtype=np.int64)
    unique = np.unique(states)
    S = len(unique)

    if mode == "exact_state":
        # 1-to-1: state i → bin i (identity for contiguous states)
        state_to_bin = {int(s): idx for idx, s in enumerate(unique)}
        return np.array([state_to_bin[int(s)] for s in states], dtype=np.int64)

    if mode in ("uniform", "margin_quantile"):
        k = n_bins if n_bins is not None else S
        k = max(1, int(k))
        # Assign states to bins uniformly
        indices = np.searchsorted(unique, states)
        bin_ids = (indices * k) // S
        return bin_ids.astype(np.int64)

    raise ValueError(f"Unknown binning mode: {mode!r}. "
                     f"Supported: 'exact_state', 'uniform', 'margin_quantile'.")


def count_state_visits(
    bin_ids: NDArray[np.int64],
    n_bins: int,
    transitions: NDArray[np.int64] | None = None,
) -> NDArray[np.int64]:
    """Count visits per bin from a trajectory.

    Parameters
    ----------
    bin_ids : NDArray[np.int64], shape (|S|,)
        Bin assignment per base state index.
    n_bins : int
        Total number of bins.
    transitions : NDArray[np.int64], shape (N,), optional
        Sequence of visited base state indices. If None, returns zero counts.

    Returns
    -------
    NDArray[np.int64], shape (n_bins,)
        Visit count per bin.
    """
    counts = np.zeros(n_bins, dtype=np.int64)
    if transitions is None:
        return counts
    for s in np.asarray(transitions, dtype=np.int64):
        b = int(bin_ids[int(s)])
        if 0 <= b < n_bins:
            counts[b] += 1
    return counts
